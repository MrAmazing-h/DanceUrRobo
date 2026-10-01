"""Audit a GVHMR reconstruction and export explicitly named, Z-up joints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hmr4d.utils.smplx_utils import make_smplx
from smplx.joint_names import JOINT_NAMES


def stats(x):
    x = np.asarray(x)
    return dict(
        mean=float(x.mean()),
        median=float(np.median(x)),
        p95=float(np.percentile(x, 95)),
        maximum=float(x.max()),
    )


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--keypoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    pred = torch.load(a.prediction, map_location="cpu")
    g = pred["smpl_params_global"]
    c = pred["smpl_params_incam"]
    n = len(g["body_pose"])
    assert n >= 4 and all(torch.isfinite(v).all() for d in (g, c) for v in d.values())
    model = make_smplx("supermotion").to(a.device)
    coco = make_smplx("supermotion_coco17").to(a.device)
    native = []
    coco_cam = []
    feet_min = []
    for i in range(0, n, 32):
        out = model(**{k: v[i : i + 32].to(a.device) for k, v in g.items()})
        native.append(out.joints[:, :55].cpu().numpy())
        feet_min.append(out.vertices[:, :, 1].min(dim=1).values.cpu().numpy())
        coco_cam.append(coco(**{k: v[i : i + 32].to(a.device) for k, v in c.items()}).cpu().numpy())
    native = np.concatenate(native)
    cam = np.concatenate(coco_cam)
    rest = model(betas=g["betas"].median(dim=0, keepdim=True).values.to(a.device))
    height = float((rest.vertices[0, :, 1].max() - rest.vertices[0, :, 1].min()).cpu())
    # GVHMR world is Y-up. Rx(+90deg) makes it Z-up while preserving handedness.
    r_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    z = native @ r_up.T
    left = (z[:30, 16] - z[:30, 17]).mean(axis=0)
    yaw = np.pi / 2 - np.arctan2(left[1], left[0])
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    z = z @ r_yaw.T
    offset = np.array([z[0, 0, 0], z[0, 0, 1], float(np.min(np.concatenate(feet_min)))])
    z -= offset
    names = [
        "Pelvis",
        "L_Hip",
        "R_Hip",
        "Spine1",
        "L_Knee",
        "R_Knee",
        "Spine2",
        "L_Ankle",
        "R_Ankle",
        "Spine3",
        "L_Foot",
        "R_Foot",
        "Neck",
        "L_Collar",
        "R_Collar",
        "Head",
        "L_Shoulder",
        "R_Shoulder",
        "L_Elbow",
        "R_Elbow",
        "L_Wrist",
        "R_Wrist",
    ]
    aliases = {k: k for k in names}
    aliases.update(
        {
            "Spine1": "Torso",
            "Spine2": "Spine",
            "Spine3": "Chest",
            "L_Foot": "L_Toe",
            "R_Foot": "R_Toe",
            "L_Collar": "L_Thorax",
            "R_Collar": "R_Thorax",
        }
    )
    names = [aliases[k] for k in names] + ["Jaw", "L_Eye", "R_Eye"]
    for k in JOINT_NAMES[25:55]:
        side, part = k.split("_", 1)
        names.append(("L_" if side == "left" else "R_") + part.capitalize())
    assert len(names) == 55
    np.savez_compressed(
        a.output / "smplx_native_export.npz",
        global_joint_positions=z.astype(np.float32),
        native_world_joint_positions=native,
        height=height,
        fps=30.0,
        joint_names=np.array(names),
        rotation_world_to_zup=r_yaw @ r_up,
        translation_offset=offset,
        body_pose=g["body_pose"].numpy(),
        global_orient=g["global_orient"].numpy(),
        transl=g["transl"].numpy(),
        betas=g["betas"].numpy(),
        hand_pose_source=np.array("GVHMR default mean hand; finger articulation NOT recovered"),
    )
    k = pred["K_fullimg"].numpy()
    if k.ndim == 2:
        k = np.repeat(k[None], n, axis=0)
    pix = np.einsum("fij,fkj->fki", k, cam)
    pix = pix[:, :, :2] / pix[:, :, 2:3]
    observed = torch.load(a.keypoints, map_location="cpu").numpy()
    error = np.linalg.norm(pix - observed[:, :, :2], axis=-1)
    torso = np.linalg.norm(
        observed[:, [5, 6], :2].mean(axis=1) - observed[:, [11, 12], :2].mean(axis=1), axis=-1
    )
    delta = np.linalg.norm(np.diff(z[:, [16, 17, 18, 19, 20, 21]], axis=0), axis=-1)
    acc = np.linalg.norm(np.diff(z[:, [16, 17, 18, 19, 20, 21]], n=2, axis=0) * 30**2, axis=-1)
    bone = np.linalg.norm(native[:, 20] - native[:, 18], axis=-1)
    report = {
        "frames": n,
        "fps": 30,
        "duration_seconds": n / 30,
        "finite_values": True,
        "estimated_body_height_m": height,
        "reference_definition": "2D comparison uses ViTPose observations on the same video; no measured 3D ground truth.",
        "coco17_reprojection_error_pixels": stats(error),
        "upper_limb_reprojection_error_pixels": stats(error[:, 5:11]),
        "wrist_reprojection_error_pixels": stats(error[:, 9:11]),
        "upper_limb_reprojection_error_torso_normalized": stats(error[:, 5:11] / torso[:, None]),
        "upper_limb_frame_displacement_m": stats(delta),
        "upper_limb_acceleration_m_per_s2": stats(acc),
        "left_forearm_length_range_m": [float(bone.min()), float(bone.max())],
        "hands": "SMPL-X includes hands, but GVHMR did not infer finger articulation. Do not evaluate these as recovered fingers.",
        "coordinate_transform": {
            "source": "GVHMR Y-up",
            "target": "Z-up, initial forward approximately +X, left +Y",
            "R": (r_yaw @ r_up).tolist(),
            "offset_after_rotation": offset.tolist(),
        },
        "decision": "Candidate requires visual overlay review before robot retargeting.",
        "not_validated": [
            "absolute metric scale",
            "occluded 3D depth",
            "physical balance",
            "finger pose accuracy",
        ],
    }
    np.savez_compressed(
        a.output / "reprojection_diagnostics.npz",
        projected=pix,
        observed=observed,
        error_pixels=error,
    )
    (a.output / "evaluation_smplx.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

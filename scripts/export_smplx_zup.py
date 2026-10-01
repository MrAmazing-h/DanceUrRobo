"""Export consistent SMPL-X rotations, translation and joints in Z-up coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hmr4d.utils.smplx_utils import make_smplx
from scipy.spatial.transform import Rotation


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source = np.load(a.source)
    model = make_smplx("supermotion")
    rotation = source["rotation_world_to_zup"]
    offset = source["translation_offset"]
    betas = torch.from_numpy(source["betas"]).float()
    j0 = model.get_skeleton(betas)[:, 0].numpy()
    global_orient = (
        Rotation.from_matrix(rotation @ Rotation.from_rotvec(source["global_orient"]).as_matrix())
        .as_rotvec()
        .astype(np.float32)
    )
    transl = ((source["transl"] + j0) @ rotation.T - j0 - offset).astype(np.float32)
    pose = {
        "global_orient": torch.from_numpy(global_orient),
        "body_pose": torch.from_numpy(source["body_pose"]).float(),
        "betas": betas,
        "transl": torch.from_numpy(transl),
    }
    joints = []
    for i in range(0, len(betas), 24):
        joints.append(model(**{k: v[i : i + 24] for k, v in pose.items()}).joints[:, :55].numpy())
    joints = np.concatenate(joints)
    error = float(np.max(np.abs(joints - source["global_joint_positions"])))
    assert error < 1e-5, error
    np.savez_compressed(
        a.output / "human_smplx_zup.npz",
        **{k: v.numpy() for k, v in pose.items()},
        global_joint_positions=joints,
        joint_names=source["joint_names"],
        fps=30.0,
        gender="neutral",
        model_type="smplx",
        up_axis="Z",
        units="meters",
        use_pca=True,
        num_pca_comps=12,
        flat_hand_mean=False,
        finger_pose_inferred=False,
        hand_pose_source="SMPL-X default mean; body estimator does not recover fingers",
    )
    (a.output / "human_export_validation.json").write_text(
        json.dumps(
            {
                "frames": len(joints),
                "fps": 30,
                "max_reconstructed_joint_coordinate_error_m": error,
                "transform": "Root-pivot-aware conversion of global orientation and translation; all 55 joints verified against independently transformed source joints.",
                "isaac_status": "Z-up motion parameters supplied; this is not a packaged Isaac humanoid asset or a dynamics controller.",
            },
            indent=2,
        )
    )
    print("Z-up SMPL-X FK validation error", error)


if __name__ == "__main__":
    main()

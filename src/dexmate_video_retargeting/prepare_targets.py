"""Build robot-proportioned landmarks and body rotation targets from Z-up SMPL-X."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .model_contract import validate_model
from .posture_ik import chest_frames, unit
from .quality_metrics import frame

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def body_rotations(source):
    points = source["global_joint_positions"]
    chest = chest_frames(points, source["joint_names"].tolist())
    n = len(points)
    rotations = (
        Rotation.from_rotvec(
            np.concatenate(
                [source["global_orient"][:, None], source["body_pose"].reshape(n, 21, 3)], axis=1
            ).reshape(-1, 3)
        )
        .as_matrix()
        .reshape(n, 22, 3, 3)
    )
    for j in range(1, 22):
        rotations[:, j] = rotations[:, PARENTS[j]] @ rotations[:, j]
    head = rotations[:, 15] @ rotations[0, 15].T @ chest[0]
    return chest, head


def adapt_landmarks(points, names, model, seed):
    """Use the original scale, shoulder alignment and segment-length adaptation."""
    points = np.asarray(points, dtype=float).copy()
    if not np.isfinite(points).all():
        raise ValueError("Body landmarks must be finite; repair/review body reconstruction first")
    data = mujoco.MjData(model)
    data.qpos[:] = seed
    mujoco.mj_forward(model, data)

    def pos(link):
        return data.xpos[model.body(link).id].copy()

    robot_lengths, human_lengths = [], []
    for side in ("L", "R"):
        rp = [pos(side + "_arm_l" + k) for k in ("1", "4", "7")]
        hp = points[:, [names.index(side + "_" + k) for k in ("Shoulder", "Elbow", "Wrist")]]
        robot_lengths.append(sum(np.linalg.norm(rp[j + 1] - rp[j]) for j in range(2)))
        human_lengths.append(
            np.median(
                np.linalg.norm(hp[:, 1] - hp[:, 0], axis=-1)
                + np.linalg.norm(hp[:, 2] - hp[:, 1], axis=-1)
            )
        )
    if min(human_lengths) < 1e-6:
        raise ValueError("Degenerate human arm lengths")
    scale = float(np.mean(robot_lengths) / np.mean(human_lengths))
    points *= scale
    shoulders = [names.index(side + "_Shoulder") for side in ("L", "R")]
    shift = (pos("L_arm_l1") + pos("R_arm_l1")) / 2 - points[0, shoulders].mean(0)
    points += shift
    original = points.copy()
    center = points[:, shoulders].mean(1)
    left = unit(points[:, shoulders[0]] - points[:, shoulders[1]])
    width = np.linalg.norm(pos("L_arm_l1") - pos("R_arm_l1"))
    mapping = {}
    for side, sign in (("L", 1), ("R", -1)):
        si, ei, wi = [names.index(side + "_" + k) for k in ("Shoulder", "Elbow", "Wrist")]
        rp = [pos(side + "_" + k) for k in ("arm_l1", "arm_l4", "hand_base")]
        points[:, si] = center + sign * 0.5 * width * left
        points[:, ei] = points[:, si] + np.linalg.norm(rp[1] - rp[0]) * unit(
            original[:, ei] - original[:, si]
        )
        points[:, wi] = points[:, ei] + np.linalg.norm(rp[2] - rp[1]) * unit(
            original[:, wi] - original[:, ei]
        )
        ids = [names.index(side + "_" + k) for k in ("Index1", "Middle1", "Pinky1")]
        basis = frame(original[:, wi], *[original[:, i] for i in ids])
        palm = [pos(side + "_" + k) for k in ("ff_l1", "mf_l1", "lf_l1")]
        rest = frame(rp[2], *palm)
        for i, point in zip(ids, palm):
            points[:, i] = points[:, wi] + np.einsum("fij,j->fi", basis, rest.T @ (point - rp[2]))
        for key, link in zip(
            ("Shoulder", "Elbow", "Wrist", "Index1", "Middle1", "Pinky1"),
            ("arm_l1", "arm_l4", "hand_base", "ff_l1", "mf_l1", "lf_l1"),
        ):
            mapping[side + "_" + key] = side + "_" + link
    keys = list(mapping)
    return (
        points[:, [names.index(k) for k in keys]],
        mapping,
        {"scale": scale, "target_shift_m": shift.tolist()},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--seed",
        type=Path,
        help="Optional existing robot trajectory; otherwise use a neutral initial pose",
    )
    a = p.parse_args()
    source = np.load(a.human, allow_pickle=False)
    if float(source["fps"]) != 30.0:
        p.error("Inputs must be sampled at 30 Hz")
    model = mujoco.MjModel.from_xml_path(str(a.model_dir / "robot.xml"))
    validate_model(model)
    n = len(source["global_joint_positions"])
    q = model.qpos0.copy()
    for name, value in zip(("torso_j1", "torso_j2", "torso_j3"), (0.65, 1.3, 0.65)):
        q[model.joint(name).qposadr[0]] = value
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            adr = model.jnt_qposadr[j]
            q[adr] = np.clip(q[adr], *model.jnt_range[j])
    points, mapping, metadata = adapt_landmarks(
        source["global_joint_positions"], source["joint_names"].tolist(), model, q
    )
    chest, head = body_rotations(source)
    seed = np.tile(q, (n, 1)) if a.seed is None else np.load(a.seed, allow_pickle=False)["qpos"]
    if seed.shape != (n, model.nq):
        p.error("Seed must match the source frame count and robot layout exactly")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model").symlink_to(a.model_dir.resolve(), target_is_directory=True)
    np.savez_compressed(a.output / "seed.npz", qpos=seed, fps=30.0)
    np.savez_compressed(
        a.output / "target_inputs.npz",
        human_joint_motions=points,
        joint_names=np.array(list(mapping)),
        mapping_keys=np.array(list(mapping)),
        mapping_links=np.array(list(mapping.values())),
        fps=30.0,
    )
    np.savez_compressed(a.output / "body_targets.npz", chest_rotation=chest, head_rotation=head)
    np.savez_compressed(a.output / "hands.npz", valid=np.zeros((n, 2), bool))
    metadata.update(
        initialization="Provided robot seed"
        if a.seed
        else "Neutral pose; a new-video initialization, not the historical Laplacian seed",
        hands="No observed finger articulation is provided by this adapter. Run the optional hand estimator/fitter separately.",
    )
    (a.output / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

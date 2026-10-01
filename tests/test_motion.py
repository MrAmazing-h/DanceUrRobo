import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexmate_video_retargeting.posture_ik import chest_frames
from dexmate_video_retargeting.prepare_targets import body_rotations
from dexmate_video_retargeting.upsample_review import resample


def test_yaw_wrap_and_waypoint_preservation():
    q = np.zeros((4, 9))
    q[:, 3] = 1
    q[:, 7] = [0, 0.2, 0.3, 0.4]
    yaw = np.deg2rad([175, 179, -179, -175])
    q[:, 3] = np.cos(yaw / 2)
    q[:, 6] = np.sin(yaw / 2)
    dense, t = resample(q)
    np.testing.assert_allclose(dense[::4, 7:], q[:, 7:], atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(dense[:, 3:7], axis=1), 1, atol=1e-14)
    rotation = Rotation.from_quat(dense[:, [4, 5, 6, 3]])
    assert np.max((rotation[:-1].inv() * rotation[1:]).magnitude()) < np.deg2rad(3)
    np.testing.assert_allclose(dense[-3:], np.tile(dense[-4], (3, 1)))
    assert t[-1] == 3 / 30


def test_nonplanar_root_is_rejected():
    q = np.zeros((4, 8))
    q[:, 3] = np.cos(0.1)
    q[:, 4] = np.sin(0.1)
    with pytest.raises(ValueError, match="planar"):
        resample(q)


def test_chest_frame_is_right_handed():
    names = ["L_Shoulder", "R_Shoulder", "Neck", "Pelvis"]
    points = np.array([[[0, 1, 1], [0, -1, 1], [0, 0, 2], [0, 0, 0]]], float)
    frame = chest_frames(points, names)
    np.testing.assert_allclose(frame[0], np.eye(3))
    np.testing.assert_allclose(np.linalg.det(frame), 1)


def test_head_calibration_uses_initial_head_frame():
    names = ["L_Shoulder", "R_Shoulder", "Neck", "Pelvis"]
    points = np.tile(np.array([[[0, 1, 1], [0, -1, 1], [0, 0, 2], [0, 0, 0]]]), (4, 1, 1))
    body = np.zeros((4, 63))
    body[:, 42:45] = [0, 0, 0.5]
    chest, head = body_rotations(
        {
            "global_joint_positions": points,
            "joint_names": np.array(names),
            "body_pose": body,
            "global_orient": np.zeros((4, 3)),
        }
    )
    np.testing.assert_allclose(head[0], chest[0], atol=1e-12)


def test_released_inputs_share_reference_and_frame_count():
    root = Path(__file__).resolve().parents[1]
    for name in ["seed", "target_inputs", "body_targets", "hands"]:
        z = np.load(root / "examples/inputs" / f"{name}.npz", allow_pickle=False)
        key = {
            "seed": "qpos",
            "target_inputs": "human_joint_motions",
            "body_targets": "chest_rotation",
            "hands": "valid",
        }[name]
        assert len(z[key]) == 590
    z = np.load(root / "examples/reference.npz", allow_pickle=False)
    assert z["qpos"].shape == (2360, 55)
    assert float(z["fps"]) == 120
    assert np.isfinite(z["qpos"]).all()
    report = json.loads((root / "results/after.json").read_text())
    assert report["frames"] == 590

"""The two demo tracks must refer to the same source before they are aligned."""

from pathlib import Path

import numpy as np

from dexmate_video_retargeting.prepare_targets import body_rotations


def test_demo_source_identity():
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "examples/g1/human_smplx_zup.npz", allow_pickle=False) as source:
        chest, head = body_rotations(source)
        assert float(source["fps"]) == 30 and len(chest) == 590
    with np.load(root / "examples/inputs/body_targets.npz", allow_pickle=False) as target:
        np.testing.assert_array_equal(chest, target["chest_rotation"])
        np.testing.assert_array_equal(head, target["head_rotation"])


def test_showcase_native_speed_timeline():
    source_time = np.arange(588) / 30
    g1_video_frame = np.floor((source_time + 2) * 25 + 0.5).astype(int)
    assert g1_video_frame[0] == 50
    assert g1_video_frame[-1] < 600
    assert source_time[-1] < 19.62
    assert np.max(np.abs(g1_video_frame / 25 - 2 - source_time)) <= 0.02 + 1e-10

import json
import sys

import mujoco
import numpy as np
import pytest

from dexmate_video_retargeting.audit import main, validate_states

XML = '<mujoco><worldbody><geom type="plane" size="2 2 .1"/><body name="ball"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'


def test_ground_collision_at_stored_pose_fails(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    (tmp_path / "model/robot.xml").write_text(XML)
    q = np.array([[0, 0, 0, 1, 0, 0, 0]], float)
    np.savez(tmp_path / "reference.npz", qpos=q, fps=30.0)
    out = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys, "argv", ["audit", "--run", str(tmp_path), "--output", str(out), "--substeps", "4"]
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    report = json.loads(out.read_text())
    assert report["total_samples_including_saved_poses"] == 1
    assert report["ground_max_penetration_m"] > 0.09
    assert report["violating_contacts"] >= 1


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_pose_cannot_pass(bad):
    model = mujoco.MjModel.from_xml_string(XML)
    q = np.array([[0, 0, 0.2, 1, 0, 0, 0]], float)
    q[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        validate_states(model, q, 30.0)


def test_unnormalized_quaternion_rejected():
    model = mujoco.MjModel.from_xml_string(XML)
    q = np.array([[0, 0, 0.2, 0, 0, 0, 0]], float)
    with pytest.raises(ValueError, match="quaternion"):
        validate_states(model, q, 30.0)

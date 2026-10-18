"""No network or model weights needed for the public CI contract tests."""

import json
import sys
import xml.etree.ElementTree as ET
from importlib.resources import files
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from danceurrobo import controller as c
from danceurrobo.fetch_assets import fetch


def test_joint_order_and_gains():
    assert len(c.NAMES) == len(set(c.NAMES)) == 29
    assert set(c.ORDER) == set(range(29))
    assert c.NAMES[:3] == ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint"]
    k = c.NAMES.index("left_knee_joint")
    a = c.NAMES.index("left_ankle_pitch_joint")
    assert np.isclose(c.DEFAULT[k], 0.669)
    assert np.isclose(c.SCALE[k] * c.KP[k], 0.25 * 139)
    assert np.isclose(c.SCALE[a] * c.KP[a], 0.5 * 25)


def test_orientation_and_input_dimensions():
    a = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])
    b = Rotation.from_euler("xyz", [0.4, -0.2, 0.7])
    expected = (a.as_matrix().T @ b.as_matrix())[:, :2].ravel()
    np.testing.assert_allclose(c.orientation6(a, b), expected, atol=1e-14)
    assert sum(n for _, n in c.FIELDS) == 1762
    assert 64 + 10 * (3 + 29 + 29 + 29 + 3) == 994


def test_mode_is_scalar_not_one_hot(monkeypatch):
    class Session:
        def __init__(self, name, *args, **kwargs):
            self.encoder = "encoder" in name

        def get_inputs(self):
            return [SimpleNamespace(shape=[1, 1762 if self.encoder else 994])]

        def run(self, _, inputs):
            assert inputs["obs_dict"].shape == tuple(self.get_inputs()[0].shape)
            return [np.zeros((1, 64 if self.encoder else 29), dtype=np.float32)]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session),
    )
    p = c.Policy()
    p.step(c.DEFAULT, np.zeros(29), Rotation.identity(), np.zeros(3))
    np.testing.assert_array_equal(p.last_encoder_input[0, :4], [0, 0, 0, 0])
    p.step(
        c.DEFAULT,
        np.zeros(29),
        Rotation.identity(),
        np.zeros(3),
        np.zeros((10, 24, 3)),
        Rotation.identity(10),
    )
    np.testing.assert_array_equal(p.last_encoder_input[0, :4], [2, 0, 0, 0])
    assert all(len(h) == 10 for h in p.hist)


def test_model_assets_are_public_pinned_and_relative():
    data = files("danceurrobo").joinpath("data")
    manifest = json.loads(data.joinpath("assets.json").read_text())
    xml = ET.fromstring(data.joinpath("g1_inspire_sonic.xml").read_text())
    paths = {row["path"] for row in manifest["files"]}
    for mesh in xml.findall(".//asset/mesh"):
        assert "/" not in mesh.get("file")
        assert "models/meshes/" + mesh.get("file") in paths
    for row in manifest["files"]:
        assert len(row["sha256"]) == 64
        assert row["url"].startswith("https://")
        assert "/main/" not in row["url"] and "/master/" not in row["url"]
    assert xml.find("option").get("gravity") == "0 0 -9.81"
    assert len(xml.findall(".//freejoint")) == 1


def test_fetch_refuses_modified_existing_file(tmp_path):
    import pytest

    (tmp_path / "asset").write_text("modified")
    with pytest.raises(ValueError, match="refusing overwrite"):
        fetch({"path": "asset", "sha256": "0" * 64, "url": "https://invalid.invalid"}, tmp_path)


def test_sampled_limit_audit():
    import mujoco
    import pytest

    from danceurrobo.audit import audit

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body>'
        '<joint type="hinge" range="-1 1"/><geom type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    result = audit(model, np.array([[0.0], [1.2]]))
    assert np.isclose(result["max_joint_limit_violation_rad"], 0.2)
    assert result["negative_self_contact_pairs_minimum_m"] == {}
    with pytest.raises(ValueError):
        audit(model, np.array([[np.nan]]))

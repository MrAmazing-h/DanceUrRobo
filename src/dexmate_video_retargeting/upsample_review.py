"""Shape-preserving 30-to-120 Hz export with planar yaw interpolation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation


def resample(q, fps=30.0, output_fps=120.0):
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] < 7 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("Expected a finite trajectory with a free-root quaternion")
    if fps <= 0 or output_fps <= 0 or not float(output_fps / fps).is_integer():
        raise ValueError("Output rate must be a positive integer multiple of input rate")
    euler = Rotation.from_quat(q[:, [4, 5, 6, 3]]).as_euler("xyz")
    if np.max(np.abs(euler[:, :2])) > 1e-6:
        raise ValueError("This Vega profile requires planar root orientation")
    factor = int(output_fps / fps)
    old = np.arange(len(q)) / fps
    time = np.minimum(np.arange(len(q) * factor) / output_fps, old[-1])
    result = PchipInterpolator(old, q, axis=0)(time)
    yaw = PchipInterpolator(old, np.unwrap(euler[:, 2]))(time)
    result[:, 3] = np.cos(yaw / 2)
    result[:, 4:6] = 0
    result[:, 6] = np.sin(yaw / 2)
    return result, time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    z = np.load(a.input / "reference.npz", allow_pickle=False)
    q, time = resample(z["qpos"], float(z["fps"]), 120.0)
    factor = int(120 / float(z["fps"]))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model").symlink_to((a.input / "model").resolve(), target_is_directory=True)
    observed = np.zeros((len(q), 2), bool)
    if "hand_observed" in z:
        observed[::factor] = z["hand_observed"]
    np.savez_compressed(
        a.output / "reference.npz",
        qpos=q,
        fps=120.0,
        source_frame_index=time * float(z["fps"]),
        hand_observed=observed,
    )
    t = np.load(a.input / "target_inputs.npz", allow_pickle=False)
    h = t["human_joint_motions"]
    old = np.arange(len(h)) / float(z["fps"])
    flat = h.reshape(len(h), -1)
    targets = np.stack(
        [np.interp(time, old, flat[:, i]) for i in range(flat.shape[1])], axis=-1
    ).reshape(len(time), *h.shape[1:])
    payload = {k: t[k] for k in t.files if k not in ("human_joint_motions", "fps")}
    np.savez_compressed(
        a.output / "target_inputs.npz", **payload, human_joint_motions=targets, fps=120.0
    )
    report = {
        "frames": len(q),
        "fps": 120.0,
        "collision_audit_required": True,
        "note": "PCHIP hinges and unwrapped planar yaw; final pose held for the last source-frame interval. Quaternion signs may canonicalize.",
    }
    (a.output / "upsampling.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

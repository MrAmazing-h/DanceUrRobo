"""Validate a portable rerun against the published recorded state trajectory."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    a = p.parse_args()
    out = a.workdir / "outputs/public_native"
    baseline = np.load("examples/g1/native_motion.npz", allow_pickle=False)
    actual = np.load(out / "trajectory.npz", allow_pickle=False)
    delta = float(np.max(np.abs(actual["qpos"] - baseline["qpos"])))
    np.testing.assert_allclose(actual["qpos"], baseline["qpos"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(actual["time"], baseline["time_s"], rtol=0, atol=1e-8)
    videos = {}
    for name in ("dynamic_rollout.mp4", "source_vs_robot.mp4"):
        with imageio.get_reader(out / name) as reader:
            fps = reader.get_meta_data()["fps"]
            n = sum(1 for _ in reader)
        assert n == 600 and fps == 25
        videos[name] = {"frames": n, "fps": fps}
    report = {
        "status": "PASS",
        "scope": "public-asset reproduction, not hardware or collision acceptance",
        "mujoco": mujoco.__version__,
        "max_abs_qpos_difference": delta,
        "qpos_tolerance": 1e-5,
        "samples": len(actual["time"]),
        "videos": videos,
    }
    Path("results/g1/portable_reproduction.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()

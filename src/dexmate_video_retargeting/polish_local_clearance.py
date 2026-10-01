"""Small, smooth local joint correction after dense interpolation audit."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model").symlink_to((a.run / "model").resolve(), target_is_directory=True)
    shutil.copy2(a.run / "target_inputs.npz", a.output / "target_inputs.npz")
    shutil.copy2(__file__, a.output / "local_clearance_driver.py")
    data = np.load(a.run / "reference.npz")
    q = data["qpos"].copy()
    m = mujoco.MjModel.from_xml_path(str(a.output / "model/robot.xml"))
    d = mujoco.MjData(m)
    report = json.loads((a.run / "evaluation/interpolation_audit.json").read_text())
    centers = sorted(set(x["interval_start_frame"] for x in report["details"]))
    mimics = []
    for node in ET.parse(a.output / "model/robot.urdf").findall("joint"):
        rel = node.find("mimic")
        if rel is not None:
            mimics.append(
                (
                    node.get("name"),
                    rel.get("joint"),
                    float(rel.get("multiplier", "1")),
                    float(rel.get("offset", "0")),
                )
            )

    def seam(g1, g2):
        ns = [m.body(int(m.geom_bodyid[g])).name.lower() for g in [g1, g2]]
        return ns[0][:2] == ns[1][:2] and (
            all(
                any(t in name for t in ["_th_", "_ff_", "_mf_", "_rf_", "_lf_", "hand_base"])
                for name in ns
            )
            or all(any(name.endswith("_arm_l" + str(j)) for j in [5, 6, 7, 8]) for name in ns)
        )

    def good(path):
        for k in range(len(path) - 1):
            for f in np.linspace(0, 1, 65):
                row = path[k] + f * (path[k + 1] - path[k])
                row[3:7] /= np.linalg.norm(row[3:7])
                d.qpos[:] = row
                mujoco.mj_forward(m, d)
                if any(c.dist < -(0.002 if seam(c.geom1, c.geom2) else 0.0001) for c in d.contact):
                    return False
        return True

    logs = []
    for center in centers:
        left = max(0, center - 12)
        right = min(len(q) - 1, center + 13)
        original = q[left : right + 1].copy()
        accepted = False
        contacts = [v for v in report["details"] if v["interval_start_frame"] == center]
        involved = set(
            n.upper()[0] for v in contacts for n in v["pair"] if n.lower().startswith(("l_", "r_"))
        )
        candidates = [
            side + "_th_" + joint for side in sorted(involved) for joint in ["j1", "j0"]
        ] + [side + "_arm_j" + str(j) for side in sorted(involved) for j in [4, 3, 5, 2, 6, 1, 7]]
        for amplitude in [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]:
            for name in candidates:
                jid = m.joint(name).id
                adr = int(m.jnt_qposadr[jid])
                followers = [
                    (int(m.joint(slave).qposadr[0]), m.joint(slave).range.copy(), mul, offset)
                    for slave, master, mul, offset in mimics
                    if master == name
                ]
                for sign in [-1, 1]:
                    trial = original.copy()
                    envelope = np.sin(np.linspace(0, np.pi, len(trial))) ** 2
                    trial[:, adr] += sign * amplitude * envelope
                    if (
                        trial[:, adr].min() < m.jnt_range[jid, 0]
                        or trial[:, adr].max() > m.jnt_range[jid, 1]
                    ):
                        continue
                    limits_ok = True
                    for slave, limits, mul, offset in followers:
                        trial[:, slave] = trial[:, adr] * mul + offset
                        limits_ok &= (
                            trial[:, slave].min() >= limits[0]
                            and trial[:, slave].max() <= limits[1]
                        )
                    if not limits_ok or not good(trial):
                        continue
                    q[left : right + 1] = trial
                    logs.append(
                        {
                            "center_dense_frame": center,
                            "window": [left, right],
                            "joint": name,
                            "peak_delta_rad": sign * amplitude,
                            "wrist_body_trajectory_unchanged": "_th_" in name,
                        }
                    )
                    accepted = True
                    break
                if accepted:
                    break
            if accepted:
                break
        if not accepted:
            raise RuntimeError(f"No small joint correction found at {center}")
    arrays = {k: data[k] for k in data.files}
    arrays["qpos"] = q
    np.savez_compressed(a.output / "reference.npz", **arrays)
    (a.output / "local_clearance_correction.json").write_text(
        json.dumps(
            {
                "source": str(a.run),
                "corrections": logs,
                "scope": "Local post-interpolation correction; all changes remain included in final metrics.",
            },
            indent=2,
        )
    )
    print(logs, flush=True)


if __name__ == "__main__":
    main()

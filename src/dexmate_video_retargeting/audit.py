"""Audit saved poses and interpolated transitions against the same collision proxies."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def validate_states(model, q, fps):
    if q.ndim != 2 or q.shape[1] != model.nq or not len(q) or not np.isfinite(q).all():
        raise ValueError("Expected nonempty finite qpos with the model's exact dimension")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be positive and finite")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise ValueError("This auditor expects a free root at qpos[0:7]")
    quaternion_error = float(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1).max())
    if quaternion_error > 1e-6:
        raise ValueError("Root quaternion must be normalized before auditing")
    limits = []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            v = q[:, model.jnt_qposadr[j]]
            if np.any(v < model.jnt_range[j, 0] - 1e-6) or np.any(v > model.jnt_range[j, 1] + 1e-6):
                limits.append(model.joint(j).name)
    mimic_error = 0.0
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            first, second = model.eq_obj1id[i], model.eq_obj2id[i]
            x = np.zeros(len(q)) if second < 0 else q[:, model.jnt_qposadr[second]]
            predicted = np.polynomial.polynomial.polyval(x, model.eq_data[i, :5])
            mimic_error = max(
                mimic_error, float(np.abs(q[:, model.jnt_qposadr[first]] - predicted).max())
            )
    return {
        "quaternion_norm_max_error": quaternion_error,
        "joint_position_limit_violations": limits,
        "mimic_max_residual_rad": mimic_error,
        "state_checks_passed": not limits and mimic_error < 1e-6,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--substeps", type=int, default=4)
    a = p.parse_args()
    assert a.substeps >= 2, "Use at least 2 substeps"
    data = np.load(a.run / "reference.npz")
    q = data["qpos"]
    fps = float(data["fps"])
    m = mujoco.MjModel.from_xml_path(str(a.run / "model/robot.xml"))
    state = validate_states(m, q, fps)
    d = mujoco.MjData(m)
    max_body = max_seam = max_ground = 0.0
    bad = []
    samples = 0
    for i in range(len(q)):
        for k in range(a.substeps if i < len(q) - 1 else 1):
            alpha = k / a.substeps
            row = q[i].copy() if k == 0 else (1 - alpha) * q[i] + alpha * q[i + 1]
            # Adjacent root rotations are planar and small. Normalized quaternion interpolation.
            if k and np.dot(q[i, 3:7], q[i + 1, 3:7]) < 0:
                row[3:7] = (1 - alpha) * q[i, 3:7] - alpha * q[i + 1, 3:7]
            row[3:7] /= np.linalg.norm(row[3:7])
            d.qpos[:] = row
            mujoco.mj_forward(m, d)
            samples += 1
            for c in d.contact:
                pen = max(0.0, -float(c.dist))
                b1, b2 = [int(m.geom_bodyid[g]) for g in [c.geom1, c.geom2]]
                if not b1 or not b2:
                    max_ground = max(max_ground, pen)
                    if pen > 0.0001:
                        bad.append(
                            {
                                "interval_start_frame": i,
                                "alpha": alpha,
                                "time_seconds": (i + alpha) / fps,
                                "pair": ["ground", m.body(max(b1, b2)).name],
                                "depth_m": pen,
                                "tolerance_m": 0.0001,
                            }
                        )
                    continue
                ns = [m.body(b).name.lower() for b in [b1, b2]]
                hand = ns[0][:2] == ns[1][:2] and all(
                    any(t in n for t in ["_th_", "_ff_", "_mf_", "_rf_", "_lf_", "hand_base"])
                    for n in ns
                )
                wrist = ns[0][:2] == ns[1][:2] and all(
                    any(n.endswith("_arm_l" + str(j)) for j in [5, 6, 7, 8]) for n in ns
                )
                tol = 0.002 if hand or wrist else 0.0001
                if hand or wrist:
                    max_seam = max(max_seam, pen)
                else:
                    max_body = max(max_body, pen)
                if pen > tol:
                    bad.append(
                        {
                            "interval_start_frame": i,
                            "alpha": alpha,
                            "time_seconds": (i + alpha) / fps,
                            "pair": ns,
                            "depth_m": pen,
                            "tolerance_m": tol,
                        }
                    )
    report = {
        **state,
        "total_samples_including_saved_poses": samples,
        "substeps_per_interval": a.substeps,
        "nonseam_max_penetration_m": max_body,
        "seam_max_penetration_m": max_seam,
        "ground_max_penetration_m": max_ground,
        "violating_contacts": len(bad),
        "violating_intervals": len(set(x["interval_start_frame"] for x in bad)),
        "details": bad,
        "scope": "Discrete interpolation sampling of configured collision proxies; not continuous collision detection.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    if bad or not state["state_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

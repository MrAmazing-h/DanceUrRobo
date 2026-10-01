"""Fit F5D6 finger articulation to observed hand shape, keeping body motion fixed."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


def flex(a, b, c):
    x = b - a
    y = c - b
    return np.arccos(
        np.clip(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-10), -1, 1)
    )


def features(points):
    # Ordered wrist, thumb CMC/MCP/IP/tip, then MCP/PIP/tip for four fingers.
    fs = []
    for start in [5, 8, 11, 14]:
        fs += [
            flex(points[0], points[start], points[start + 1]),
            flex(points[start], points[start + 1], points[start + 2]),
        ]
    fs += [
        flex(points[0], points[1], points[2]),
        flex(points[1], points[2], points[3]),
        flex(points[2], points[3], points[4]),
    ]
    palm = max(np.linalg.norm(points[5] - points[14]), 0.01)
    fs += [
        np.linalg.norm(points[4] - points[7]) / palm,
        np.linalg.norm(points[4] - points[10]) / palm,
    ]
    return np.asarray(fs)


def main():
    p = argparse.ArgumentParser()
    for name in ["body-run", "hands", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "model").symlink_to((a.body_run / "model").resolve(), target_is_directory=True)
    shutil.copy2(a.body_run / "target_inputs.npz", a.output / "target_inputs.npz")
    base = np.load(a.body_run / "reference.npz")
    q = base["qpos"].copy()
    n = len(q)
    fps = float(base["fps"])
    assert fps == 30.0, "Finger fitting expects 30 Hz"
    hand = np.load(a.hands)
    valid = hand["valid"][:n].copy()
    world = hand["world"][:n]
    m = mujoco.MjModel.from_xml_path(str(a.output / "model/robot.xml"))
    d = mujoco.MjData(m)
    urdf = ET.parse(a.output / "model/robot.urdf")
    mimics = []
    vlimits = {}
    for node in urdf.findall("joint"):
        rel = node.find("mimic")
        limit = node.find("limit")
        if limit is not None and limit.get("velocity"):
            vlimits[node.get("name")] = float(limit.get("velocity"))
        if rel is not None:
            mimics.append(
                (
                    node.get("name"),
                    rel.get("joint"),
                    float(rel.get("multiplier", "1")),
                    float(rel.get("offset", "0")),
                )
            )
    finger_indices = []
    fit_error = np.full((n, 2), np.nan)
    q_proposed = q.copy()
    for side_index, side in enumerate(["L", "R"]):
        masters = [side + "_" + s for s in ["th_j0", "th_j1", "ff_j1", "mf_j1", "rf_j1", "lf_j1"]]
        ids = np.array([int(m.joint(k).qposadr[0]) for k in masters])
        finger_indices.extend(ids.tolist())
        lo = np.array([m.joint(k).range[0] for k in masters])
        hi = np.array([m.joint(k).range[1] for k in masters])
        vmax = np.array([vlimits.get(k, 1.0) for k in masters])
        followers = []
        for slave, master, mul, offset in mimics:
            if master not in masters:
                continue
            j = masters.index(master)
            bounds = sorted((m.joint(slave).range - offset) / mul)
            lo[j] = max(lo[j], bounds[0])
            hi[j] = min(hi[j], bounds[1])
            vmax[j] = min(vmax[j], vlimits.get(slave, np.inf) / abs(mul))
            sid = int(m.joint(slave).qposadr[0])
            followers.append((sid, j, mul, offset))
            finger_indices.append(sid)
        body_names = [side + "_hand_base"] + [
            side + "_" + s
            for s in [
                "th_l0",
                "th_l1",
                "th_l2",
                "th_tip",
                "ff_l1",
                "ff_l2",
                "ff_tip",
                "mf_l1",
                "mf_l2",
                "mf_tip",
                "rf_l1",
                "rf_l2",
                "rf_tip",
                "lf_l1",
                "lf_l2",
                "lf_tip",
            ]
        ]
        bids = [m.body(k).id for k in body_names]
        mp_ids = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]

        def put(row, x):
            d.qpos[:] = row
            d.qpos[ids] = x
            for sid, j, mul, offset in followers:
                d.qpos[sid] = x[j] * mul + offset
            mujoco.mj_kinematics(m, d)

        goals = np.full((n, 13), np.nan)
        for i in np.flatnonzero(valid[:, side_index]):
            goals[i] = features(world[i, side_index, mp_ids])
        observed = np.flatnonzero(valid[:, side_index])
        put(q[0], np.clip(q[0, ids], lo, hi))
        initial = features(d.xpos[bids])
        if not len(observed):
            goals[:] = initial
        else:
            # Short gaps interpolate; longer gaps hold the previous observed hand shape.
            previous = initial.copy()
            for i in range(n):
                if valid[i, side_index]:
                    previous = goals[i].copy()
                else:
                    goals[i] = previous
            for left, right in zip(observed[:-1], observed[1:]):
                if right - left <= 10:
                    for i in range(left + 1, right):
                        goals[i] = goals[left] + (goals[right] - goals[left]) * (i - left) / (
                            right - left
                        )
        goals = gaussian_filter1d(goals, 1.3, axis=0, mode="nearest")
        previous = np.clip(q[0, ids], lo, hi)
        for i in range(n):

            def objective(x):
                put(q[i], x)
                return np.r_[features(d.xpos[bids]) - goals[i], 0.3 * (x - previous)]

            step = np.minimum(vmax / fps, 0.06)
            lower = np.maximum(lo, previous - step)
            upper = np.minimum(hi, previous + step)
            fitted = least_squares(
                objective,
                np.clip(previous, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper),
                max_nfev=25,
                ftol=1e-5,
                xtol=1e-5,
                gtol=1e-5,
            ).x
            put(q[i], fitted)
            q_proposed[i, ids] = fitted
            for sid, j, mul, offset in followers:
                q_proposed[i, sid] = fitted[j] * mul + offset
            fit_error[i, side_index] = float(
                np.sqrt(np.mean((features(d.xpos[bids]) - goals[i]) ** 2))
            )
            previous = fitted
            if i % 100 == 0:
                print("finger fit", side, i, flush=True)
    # Reduce hand articulation smoothly around collision-prone frames. Body remains fixed.
    finger_indices = sorted(set(finger_indices))
    raw_fit = q_proposed.copy()
    np.savez_compressed(a.output / "raw_hand_fit.npz", qpos=raw_fit, hand_observed=valid)

    def feasible(row):
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        for c in d.contact:
            ns = [m.body(int(m.geom_bodyid[g])).name.lower() for g in [c.geom1, c.geom2]]
            same_hand = ns[0][:2] == ns[1][:2] and all(
                any(t in name for t in ["_th_", "_ff_", "_mf_", "_rf_", "_lf_", "hand_base"])
                for name in ns
            )
            same_wrist = ns[0][:2] == ns[1][:2] and all(
                any(name.endswith("_arm_l" + str(j)) for j in [5, 6, 7, 8]) for name in ns
            )
            tol = 0.002 if same_hand or same_wrist else 0.0001
            if c.dist < -tol:
                return False
        return True

    def pose(i, alpha):
        row = q[i].copy()
        row[finger_indices] += alpha * (raw_fit[i, finger_indices] - q[i, finger_indices])
        return row

    caps = np.ones(n)
    rejected = []
    for i in range(n):
        if feasible(raw_fit[i]):
            continue
        rejected.append(i)
        assert feasible(q[i]), ("Body-only baseline is not collision-valid", i)
        # Search conservative contraction of the observed hand shape.
        caps[i] = 0.0
        for alpha in np.arange(0.95, 0.0, -0.05):
            if feasible(pose(i, alpha)):
                caps[i] = alpha
                break
    iterations = 0
    while True:
        # Lipschitz lower envelope spreads reductions before/after the affected frame.
        alpha = caps.copy()
        for i in range(1, n):
            alpha[i] = min(alpha[i], alpha[i - 1] + 0.03)
        for i in range(n - 2, -1, -1):
            alpha[i] = min(alpha[i], alpha[i + 1] + 0.03)
        q_proposed = np.asarray([pose(i, alpha[i]) for i in range(n)])
        invalid = [i for i in range(n) if not feasible(q_proposed[i])]
        for i in range(n - 1):
            for fraction in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
                row = q_proposed[i] + fraction * (q_proposed[i + 1] - q_proposed[i])
                row[3:7] /= np.linalg.norm(row[3:7])
                if not feasible(row):
                    invalid.extend([i, i + 1])
                    break
        invalid = sorted(set(invalid))
        if not invalid:
            break
        for i in invalid:
            caps[i] = max(0.0, alpha[i] - 0.05)
        iterations += 1
        if iterations > 25:
            raise RuntimeError("Hand collision contraction did not converge")
    np.savez_compressed(
        a.output / "collision_contraction.npz",
        raw_caps=caps,
        applied_alpha=alpha,
        raw_fit_qpos=raw_fit,
    )
    np.savez_compressed(
        a.output / "reference.npz",
        qpos=q_proposed,
        fps=fps,
        hand_observed=valid,
        hand_collision_fallback_frames=np.asarray(rejected),
        model_path="model/robot.xml",
    )
    np.savez_compressed(
        a.output / "hand_fit_diagnostics.npz",
        feature_rmse=fit_error,
        observed=valid,
        base_qpos=q,
        hand_qpos=q_proposed,
    )
    report = {
        "method": "Bounded least-squares on finger bend angles and normalized thumb-tip distances; body and wrist trajectory unchanged.",
        "observation_source": "MediaPipe relative 3D hand estimates with ViTPose wrist identity matching",
        "valid_frames_per_hand": valid.sum(axis=0).tolist(),
        "frames": n,
        "collision_contraction_frames": rejected,
        "contraction_iterations": iterations,
        "minimum_articulation_scale": float(alpha.min()),
        "frames_with_reduced_articulation": int((alpha < 0.999).sum()),
        "occlusion_policy": "Interpolate gaps up to 0.3 s; hold previous observed shape for longer gaps; smooth 1.3-frame Gaussian. All inferred frames retain observed=false.",
        "limitation": "Estimated shape transfer to a coupled 6-DOF hand, not exact human finger-angle recovery. Collision-prone hand articulation is contracted toward the audited open-hand pose using a smooth 0.03/frame scale envelope and independently rechecked. Eight samples per frame transition are also collision-gated. Velocity and denser interpolation are audited afterwards.",
    }
    (a.output / "hand_fit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

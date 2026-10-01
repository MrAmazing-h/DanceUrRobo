"""Task-based differential IK for a planar Vega; isolated experiment, not upstream GMR."""

import argparse
import json
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .model_contract import validate_model


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def chest_frames(h, names):
    left = unit(h[:, names.index("L_Shoulder")] - h[:, names.index("R_Shoulder")])
    up = unit(h[:, names.index("Neck")] - h[:, names.index("Pelvis")])
    forward = unit(np.cross(left, up))
    up = unit(np.cross(forward, left))
    return np.stack([forward, left, up], axis=-1)


def seam(m, g1, g2):
    ns = [m.body(int(m.geom_bodyid[g])).name.lower() for g in [g1, g2]]
    return ns[0][:2] == ns[1][:2] and (
        all(any(t in n for t in ["_th_", "_ff_", "_mf_", "_rf_", "_lf_", "hand_base"]) for n in ns)
        or all(any(n.endswith("_arm_l" + str(j)) for j in [5, 6, 7, 8]) for n in ns)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--torso-weight", type=float, default=1.0)
    p.add_argument("--arm-weight", type=float, default=1.5)
    p.add_argument("--palm-weight", type=float, default=0.6)
    p.add_argument("--smooth", type=float, default=0.03)
    p.add_argument("--accel", type=float, default=0.04)
    p.add_argument("--clearance", type=float, default=0.004)
    p.add_argument("--max-frames", type=int, default=2147483647)
    p.add_argument("--iterations", type=int, default=18)
    p.add_argument("--seed", type=Path)
    p.add_argument("--frame-cap", type=float, default=10.0)
    p.add_argument("--filter", type=float, default=0.0)
    p.add_argument("--finger-filter", type=float, default=0.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model").symlink_to((a.inputs / "model").resolve(), target_is_directory=True)
    shutil.copy2(__file__, a.output / "driver.py")
    tar = np.load(a.inputs / "target_inputs.npz")
    names = tar["joint_names"].tolist()
    keys = tar["mapping_keys"].tolist()
    links = tar["mapping_links"].tolist()
    targets = tar["human_joint_motions"][: a.max_frames].copy()
    N = len(targets)
    evaluation_targets = targets.copy()
    if a.filter:
        targets = gaussian_filter1d(targets, a.filter, axis=0, mode="nearest")
    chest = np.load(a.inputs / "body_targets.npz")["chest_rotation"][:N]
    # Anatomical chest axes are neutral x-forward,y-left,z-up; no seed-pose angular bias.
    m = mujoco.MjModel.from_xml_path(str(a.output / "model/robot.xml"))
    validate_model(m)
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0
    active = np.array([0, 1, 5, 12, 13, 14, 15, 16, 17] + list(range(18, 25)) + list(range(36, 43)))
    qa = np.array([0, 1, -1, 13, 14, 15, 16, 17, 18] + list(range(19, 26)) + list(range(37, 44)))
    D = len(active)
    joint_ids = [int(m.dof_jntid[v]) for v in active]
    lower = np.array([-5, -5, -100] + [float(m.jnt_range[j, 0]) for j in joint_ids[3:]])
    upper = np.array([5, 5, 100] + [float(m.jnt_range[j, 1]) for j in joint_ids[3:]])
    bid = {x: int(m.body(x).id) for x in links + ["torso_l3", "head_l3"]}
    seed = np.load(a.seed or a.inputs / "seed.npz")["qpos"]
    if len(seed) != N:
        seed = seed[np.rint(np.linspace(0, len(seed) - 1, N)).astype(int)]
    finger = np.load(a.inputs / "seed.npz")["qpos"][:N]
    if a.finger_filter:
        finger = gaussian_filter1d(finger, a.finger_filter, axis=0, mode="nearest")
    q = seed[0].copy()
    q[13:16] = [0.65, 1.3, 0.65]
    q[16:19] = 0
    q[3:7] = [1, 0, 0, 0]
    output = []
    logs = []
    start = time.monotonic()
    # Broad phase reports nearby contacts for explicit collision inequalities.
    m.geom_margin[:] = np.maximum(m.geom_margin, 0.012)
    rest = q[13:16].copy()
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))

    def fk(row):
        d.qpos[:] = row
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    def posjac(body):
        mujoco.mj_jacBody(m, d, jp, jr, bid[body])
        return d.xpos[bid[body]].copy(), jp[:, active].copy(), jr[:, active].copy()

    def coords(row):
        x = np.empty(D)
        x[:2] = row[:2]
        x[2] = Rotation.from_quat(row[[4, 5, 6, 3]]).as_euler("xyz")[2]
        x[3:] = row[qa[3:]]
        return x

    def step(row, delta):
        v = np.zeros(m.nv)
        v[active] = delta
        out = row.copy()
        mujoco.mj_integratePos(m, out, v, 1.0)
        return out

    for f in range(N):
        prev = q.copy()
        prev2 = output[-2] if len(output) > 1 else prev
        q[26:37] = finger[f, 26:37]
        q[44:55] = finger[f, 44:55]
        maxdepth = 0.0
        fail = 0
        for it in range(a.iterations if f else 50):
            fk(q)
            A = []
            b = []
            cache = {k: posjac(k) for k in links}
            x = coords(q)
            xp = coords(prev)
            xpp = coords(prev2)

            def term(J, e, w):
                A.append(J * w)
                b.append(np.asarray(e) * w)

            # Absolute targets retain the old fixed evaluation definition.
            for key, link in zip(keys, links):
                if any(t in key for t in ["Index", "Middle", "Pinky"]):
                    continue
                weight = 5.0 if "Wrist" in key else (1.5 if "Elbow" in key else 2.0)
                pos, J, _ = cache[link]
                term(J, targets[f, names.index(key)] - pos, weight)
            for side in ["L", "R"]:
                for prox, dist, pl, dl in [
                    ("Shoulder", "Elbow", "arm_l1", "arm_l4"),
                    ("Elbow", "Wrist", "arm_l4", "hand_base"),
                ]:
                    pp, Jp, _ = cache[side + "_" + pl]
                    pd, Jd, _ = cache[side + "_" + dl]
                    v = pd - pp
                    length = np.linalg.norm(v)
                    u = v / length
                    goal = unit(
                        targets[f, names.index(side + "_" + dist)]
                        - targets[f, names.index(side + "_" + prox)]
                    )
                    term((np.eye(3) - np.outer(u, u)) @ (Jd - Jp) / length, goal - u, a.arm_weight)
                # Relative MCP geometry encodes the palm frame without world-position duplication.
                pw, Jw, _ = cache[side + "_hand_base"]
                tw = targets[f, names.index(side + "_Wrist")]
                for k, link_suffix in [
                    ("Index1", "ff_l1"),
                    ("Middle1", "mf_l1"),
                    ("Pinky1", "lf_l1"),
                ]:
                    pp, Jp, _ = cache[side + "_" + link_suffix]
                    goal = targets[f, names.index(side + "_" + k)] - tw
                    term((Jp - Jw) / 0.08, (goal - (pp - pw)) / 0.08, a.palm_weight)
            _, _, Jr = posjac("torso_l3")
            R = d.xmat[bid["torso_l3"]].reshape(3, 3)
            term(Jr, Rotation.from_matrix(chest[f] @ R.T).as_rotvec(), a.torso_weight)
            # Keep lift geometry near neutral; pitch and shoulder-height tasks may depart smoothly.
            selector = np.zeros((3, D))
            selector[:, 3:6] = np.eye(3)
            term(selector, rest - q[13:16], 0.08)
            term(np.eye(D), xp - x, a.smooth if f else 0.002)
            if f > 1:
                term(np.eye(D), 2 * xp - xpp - x, a.accel)
            # Head follows chest-relative estimated head direction, using a neutral-axis calibration.
            selector = np.zeros((3, D))
            selector[:, 6:9] = np.eye(3)
            term(selector, -q[16:19], 0.1)
            A = np.vstack(A)
            b = np.concatenate(b)
            H = A.T @ A + np.eye(D) * 0.004
            g = -A.T @ b
            trust = np.full(D, 0.13)
            trust[:2] = 0.025
            trust[2] = 0.08
            lo = np.maximum(lower - x, -trust)
            hi = np.minimum(upper - x, trust)
            if f and a.frame_cap < 1:
                cap = np.full(D, a.frame_cap)
                cap[:2] = 0.02
                cap[2] = 0.08
                cap[3:6] = 0.045
                lo = np.maximum(lo, xp - cap - x)
                hi = np.minimum(hi, xp + cap - x)
                # Round-off may place the previous solution microradians outside a bound.
                hi = np.maximum(hi, lo)
            C = [np.eye(D)]
            cl = [lo]
            cu = [hi]
            maxdepth = 0.0
            for c in d.contact:
                g1, g2 = int(c.geom1), int(c.geom2)
                if seam(m, g1, g2):
                    continue
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                normal = c.frame[:3].copy()
                point = c.pos.copy()
                j1 = np.zeros((3, m.nv))
                j2 = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, j1, None, point, b1)
                mujoco.mj_jac(m, d, j2, None, point, b2)
                jac = (normal @ (j2 - j1))[active]
                if np.linalg.norm(jac) < 1e-8:
                    continue
                maxdepth = max(maxdepth, -float(c.dist))
                C.append(jac[None])
                cl.append(np.array([min(a.clearance - float(c.dist), 0.025)]))
                cu.append(np.array([np.inf]))
            solver = osqp.OSQP()
            solver.setup(
                P=sp.csc_matrix(np.triu(H)),
                q=g,
                A=sp.csc_matrix(np.vstack(C)),
                l=np.concatenate(cl),
                u=np.concatenate(cu),
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=2500,
                polishing=False,
            )
            result = solver.solve(raise_error=False)
            if result.info.status_val not in (1, 2):
                # Retry infeasible local collision linearizations with a soft penalty.
                fail += 1
                for cc, ll in zip(C[1:], cl[1:]):
                    H += 300 * cc.T @ cc
                    g -= 300 * cc.ravel() * ll[0]
                solver = osqp.OSQP()
                solver.setup(
                    P=sp.csc_matrix(np.triu(H)),
                    q=g,
                    A=sp.eye(D, format="csc"),
                    l=lo,
                    u=hi,
                    verbose=False,
                    eps_abs=1e-5,
                    eps_rel=1e-5,
                    max_iter=2500,
                )
                result = solver.solve(raise_error=False)
                if result.x is None:
                    break
            delta = result.x
            q = step(q, delta)
            # Preserve the selected experiment: bounds are enforced inside the QP.
            if np.linalg.norm(delta) < 1e-4:
                break
        output.append(q.copy())
        logs.append(
            {
                "frame": f,
                "iterations": it + 1,
                "soft_collision_fallbacks": fail,
                "last_contact_depth_m": maxdepth,
            }
        )
        if f % 30 == 0:
            print(
                "FRAME",
                f,
                "wall",
                round(time.monotonic() - start, 1),
                "depth",
                maxdepth,
                flush=True,
            )
    out = np.asarray(output)
    np.savez_compressed(
        a.output / "reference.npz",
        qpos=out,
        fps=30.0,
        source_frame_index=np.arange(N),
        hand_observed=np.load(a.inputs / "hands.npz")["valid"][:N],
    )
    payload = {k: tar[k] for k in tar.files}
    payload["human_joint_motions"] = evaluation_targets
    np.savez_compressed(a.output / "target_inputs.npz", **payload)
    np.savez_compressed(a.output / "posture_targets.npz", chest_rotation=chest)
    (a.output / "solver_log.json").write_text(json.dumps(logs, indent=2))
    (a.output / "configuration.json").write_text(
        json.dumps(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                "method": "Custom task-based differential IK, orientation/segment objectives; inspired by GMR, not an upstream GMR reproduction",
                "fixed_reference": "Exactly stage11_f5d6 targets",
                "gravity": [0, 0, 0],
                "runtime_seconds": time.monotonic() - start,
            },
            indent=2,
        )
    )
    print("COMPLETE", a.output, flush=True)


if __name__ == "__main__":
    main()

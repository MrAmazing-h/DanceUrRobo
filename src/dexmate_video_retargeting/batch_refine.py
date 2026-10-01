"""Offline sparse SQP: fit anatomical tasks jointly over time, including collision collocation."""

import argparse
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .model_contract import validate_model
from .posture_ik import seam, unit


def main():
    p = argparse.ArgumentParser()
    for name in ["inputs", "seed", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--accel", type=float, default=1.0)
    p.add_argument("--velocity", type=float, default=0.08)
    p.add_argument("--clearance", type=float, default=0.004)
    p.add_argument("--torso-weight", type=float, default=1.8)
    p.add_argument("--palm-weight", type=float, default=0.6)
    p.add_argument("--velocity-factor", type=float, default=4.0)
    p.add_argument("--head-weight", type=float, default=0.3)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--line-search", action="store_true")
    p.add_argument("--trust", type=float, default=0.12)
    p.add_argument("--audit", type=Path)
    p.add_argument("--dense-substeps", type=int, default=32)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "model").symlink_to((a.seed / "model").resolve(), target_is_directory=True)
    shutil.copy2(__file__, a.output / "driver.py")
    z = np.load(a.seed / "reference.npz")
    q = z["qpos"].copy()
    N = len(q)
    fps = float(z["fps"])
    assert N >= 4 and fps == 30, "Expected at least four frames at 30 Hz"
    t = np.load(a.inputs / "target_inputs.npz")
    names = t["joint_names"].tolist()
    keys = t["mapping_keys"].tolist()
    links = t["mapping_links"].tolist()
    targets = t["human_joint_motions"]
    body = np.load(a.inputs / "body_targets.npz")
    chest = body["chest_rotation"]
    head = body["head_rotation"]
    assert len(chest) == len(head) == N, "Body targets and seed must align frame for frame"
    m = mujoco.MjModel.from_xml_path(str(a.output / "model/robot.xml"))
    validate_model(m)
    d = mujoco.MjData(m)
    base_v = [0, 1, 5]
    body_q = [13, 14, 15, 16, 17, 18] + list(range(19, 26)) + list(range(37, 44))
    finger_q = [26, 27, 29, 31, 33, 35, 44, 45, 47, 49, 51, 53]
    qa = np.array([0, 1, -1] + body_q + finger_q)
    active = np.array(base_v + [x - 1 for x in body_q + finger_q])
    D = len(active)
    M = N * D
    E = np.zeros((m.nv, D))
    E[active, np.arange(D)] = 1
    lower = np.full(D, -100.0)
    upper = -lower
    lower[:2] = -5
    upper[:2] = 5
    speed = np.full(D, np.inf)
    speed[:3] = [0.5, 0.5, 1.5]
    for j in range(3, D):
        lower[j], upper[j] = m.jnt_range[m.dof_jntid[active[j]]]
    mimic = []
    urdf = ET.parse(a.seed / "model/robot.urdf")
    for node in urdf.findall("joint"):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, node.get("name")) < 0:
            continue
        adr = int(m.joint(node.get("name")).qposadr[0])
        lim = node.find("limit")
        if adr in qa and lim is not None and lim.get("velocity"):
            speed[list(qa).index(adr)] = float(lim.get("velocity"))
        rel = node.find("mimic")
        if rel is not None:
            master = int(m.joint(rel.get("joint")).qposadr[0])
            mult = float(rel.get("multiplier", "1"))
            offset = float(rel.get("offset", "0"))
            col = list(qa).index(master)
            E[adr - 1, col] = mult
            mimic.append((adr, master, mult, offset))
            jl = m.jnt_range[m.joint(node.get("name")).id]
            lo, hi = sorted((jl - offset) / mult)
            lower[col] = max(lower[col], lo)
            upper[col] = min(upper[col], hi)
            if lim is not None and lim.get("velocity"):
                speed[col] = min(speed[col], float(lim.get("velocity")) / abs(mult))

    def coords(rows):
        x = np.zeros((len(rows), D))
        x[:, :2] = rows[:, :2]
        x[:, 2] = np.unwrap(Rotation.from_quat(rows[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2])
        x[:, 3:] = rows[:, qa[3:]]
        return x

    def to_q(x):
        out = q.copy()
        out[:, :2] = x[:, :2]
        out[:, 3] = np.cos(x[:, 2] / 2)
        out[:, 4:6] = 0
        out[:, 6] = np.sin(x[:, 2] / 2)
        out[:, qa[3:]] = x[:, 3:]
        for slave, master, mult, off in mimic:
            out[:, slave] = mult * out[:, master] + off
        return out

    x = coords(q)
    x[:, 3:] = np.clip(x[:, 3:], lower[3:], upper[3:])
    nom = x.copy()
    nom[:, 23:] = gaussian_filter1d(nom[:, 23:], 1.0, axis=0, mode="nearest")
    bid = {k: int(m.body(k).id) for k in links + ["torso_l3", "head_l3"]}
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    m.geom_margin[:] = np.maximum(m.geom_margin, 0.01)

    def fk(row):
        d.qpos[:] = row
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    def posjac(name):
        mujoco.mj_jacBody(m, d, jp, jr, bid[name])
        return d.xpos[bid[name]].copy(), jp @ E, jr @ E

    diff1 = sp.diags([-np.ones(N - 1), np.ones(N - 1)], [0, 1], shape=(N - 1, N))
    diff2 = sp.diags(
        [np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N)
    )
    weights = np.ones(D)
    weights[:2] = 5
    weights[2] = 1.5
    V = sp.kron(diff1, sp.diags(weights), format="csc")
    Acc = sp.kron(diff2, sp.diags(weights), format="csc")
    DV = sp.kron(diff1, sp.eye(D), format="csc")
    temporal = a.velocity**2 * (V.T @ V) + a.accel**2 * (Acc.T @ Acc)
    history = []
    start = time.monotonic()
    critical = set()
    if a.audit:
        for contact in json.loads(a.audit.read_text()).get("details", []):
            i = int(contact["interval_start_frame"])
            critical.update(j for j in [i - 1, i, i + 1] if 0 <= j < N - 1)
    sample_counts = [a.dense_substeps if f in critical else a.samples for f in range(N)]

    def collision_merit(xx):
        rows = to_q(xx)
        depths = []
        for f in range(N):
            for k in range(sample_counts[f] if f < N - 1 else 1):
                alpha = k / sample_counts[f]
                row = rows[f].copy() if not k else (1 - alpha) * rows[f] + alpha * rows[f + 1]
                row[3:7] /= np.linalg.norm(row[3:7])
                fk(row)
                for c in d.contact:
                    if c.dist < -0.00005 and not seam(m, int(c.geom1), int(c.geom2)):
                        depths.append(max(0.0, -float(c.dist) - 0.00005))
        return float(np.sum(np.square(depths)) + np.max(np.square(depths), initial=0)), float(
            np.max(depths, initial=0)
        )

    for iteration in range(a.iterations):
        qnow = to_q(x)
        blocks = []
        rhs = []
        cr = []
        cc = []
        cv = []
        cl = []
        collision_count = 0
        maxpen = 0.0
        crows = 0

        def collision_constraints(row, frame, alpha=0.0):
            nonlocal crows, collision_count, maxpen
            fk(row)
            for c in d.contact:
                g1, g2 = int(c.geom1), int(c.geom2)
                is_seam = seam(m, g1, g2)
                if is_seam:
                    continue
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                normal = c.frame[:3].copy()
                point = c.pos.copy()
                j1 = np.zeros((3, m.nv))
                j2 = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, j1, None, point, b1)
                mujoco.mj_jac(m, d, j2, None, point, b2)
                jac = normal @ (j2 - j1) @ E
                if np.linalg.norm(jac) < 1e-8:
                    continue
                nz = np.flatnonzero(np.abs(jac) > 1e-10)
                cr.extend([crows] * len(nz))
                cc.extend((frame * D + nz).tolist())
                cv.extend(((1 - alpha) * jac[nz]).tolist())
                if alpha:
                    cr.extend([crows] * len(nz))
                    cc.extend(((frame + 1) * D + nz).tolist())
                    cv.extend((alpha * jac[nz]).tolist())
                cl.append(a.clearance - float(c.dist))
                crows += 1
                maxpen = max(maxpen, -float(c.dist))
                collision_count += int(c.dist < -0.0001)

        for f in range(N):
            fk(qnow[f])
            cache = {k: posjac(k) for k in links}
            A = []
            b = []

            def term(J, e, w):
                A.append(J * w)
                b.append(np.asarray(e) * w)

            for key, link in zip(keys, links):
                if any(k in key for k in ["Index", "Middle", "Pinky"]):
                    continue
                weight = 5.0 if "Wrist" in key else (1.5 if "Elbow" in key else 2.0)
                pp, J, _ = cache[link]
                term(J, targets[f, names.index(key)] - pp, weight)
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
                    term((np.eye(3) - np.outer(u, u)) @ (Jd - Jp) / length, goal - u, 1.5)
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
            if a.head_weight:
                _, _, Jr = posjac("head_l3")
                R = d.xmat[bid["head_l3"]].reshape(3, 3)
                term(Jr, Rotation.from_matrix(head[f] @ R.T).as_rotvec(), a.head_weight)
            sel = np.zeros((3, D))
            sel[:, 3:6] = np.eye(3)
            term(sel, np.array([0.65, 1.3, 0.65]) - x[f, 3:6], 0.08)
            sel = np.zeros((12, D))
            sel[:, 23:] = np.eye(12)
            term(sel, nom[f, 23:] - x[f, 23:], 0.25)
            term(np.eye(D), nom[f] - x[f], 0.015)
            A = np.vstack(A)
            b = np.concatenate(b)
            blocks.append(A.T @ A)
            rhs.append(-A.T @ b)
            collision_constraints(qnow[f], f)
            if f < N - 1:
                for k in range(1, sample_counts[f]):
                    alpha = k / sample_counts[f]
                    row = (1 - alpha) * qnow[f] + alpha * qnow[f + 1]
                    row[3:7] /= np.linalg.norm(row[3:7])
                    collision_constraints(row, f, alpha)
        H = sp.block_diag(blocks, format="csc") + temporal + sp.eye(M, format="csc") * 0.015
        g = np.concatenate(rhs) + temporal @ x.ravel()
        C = sp.coo_matrix((cv, (cr, cc)), shape=(crows, M)).tocsc()
        trust = np.full(D, a.trust)
        trust[:2] = min(0.025, a.trust)
        trust[2] = min(0.08, a.trust)
        lo = np.maximum(lower[None] - x, -trust).ravel()
        hi = np.minimum(upper[None] - x, trust).ravel()
        # Collision slacks keep the sequential approximation feasible; they are reported, never counted as a pass.
        P = sp.block_diag([H, sp.eye(crows, format="csc") * 20000], format="csc")
        gg = np.r_[g, np.full(crows, 5.0)]
        bound = sp.hstack([sp.eye(M), sp.csc_matrix((M, crows))])
        collision = sp.hstack([C, sp.eye(crows)])
        positive = sp.hstack([sp.csc_matrix((crows, M)), sp.eye(crows)])
        mats = [bound, collision, positive]
        ls = [lo, np.array(cl), np.zeros(crows)]
        us = [hi, np.full(crows, np.inf), np.full(crows, np.inf)]
        if a.velocity_factor > 0:
            velocity_bound = sp.hstack([DV, sp.csc_matrix((DV.shape[0], crows))])
            current = DV @ x.ravel()
            vlim = np.tile(speed * a.velocity_factor / fps, N - 1)
            # Allow infeasible initial speeds to converge through bounded trust-region iterations.
            vlim = np.maximum(vlim, np.abs(current) - 0.12)
            mats.append(velocity_bound)
            ls.append(-vlim - current)
            us.append(vlim - current)
        opt = osqp.OSQP()
        opt.setup(
            P=sp.triu(P, format="csc"),
            q=gg,
            A=sp.vstack(mats, format="csc"),
            l=np.concatenate(ls),
            u=np.concatenate(us),
            verbose=False,
            eps_abs=2e-5,
            eps_rel=2e-5,
            max_iter=12000,
            polishing=False,
        )
        result = opt.solve(raise_error=False)
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError(f"QP failed at iteration {iteration}: {result.info.status}")
        delta = result.x[:M].reshape(N, D)
        alpha = 1.0
        if a.line_search:
            before, _ = collision_merit(x)
            candidates = []
            for fraction in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                merit, depth = collision_merit(x + fraction * delta)
                candidates.append((merit, fraction, depth))
                if merit <= before + 1e-10:
                    break
            merit, alpha, depth = min(candidates)
            if merit > before + 1e-8:
                alpha = 0.0
        x += alpha * delta
        x[:, 3:] = np.clip(x[:, 3:], lower[3:], upper[3:])
        entry = {
            "iteration": iteration,
            "accepted_alpha": alpha,
            "max_step_rad_or_m": float(np.max(np.abs(alpha * delta))),
            "before_max_depth_m": maxpen,
            "before_violating_contacts": collision_count,
            "max_linear_collision_slack_m": float(np.max(result.x[M:], initial=0)),
            "wall_seconds": time.monotonic() - start,
            "qp_iterations": int(result.info.iter),
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        np.savez_compressed(a.output / "progress.npz", qpos=to_q(x), fps=fps)
    qfinal = to_q(x)
    np.savez_compressed(
        a.output / "reference.npz",
        qpos=qfinal,
        fps=fps,
        source_frame_index=np.arange(N),
        hand_observed=np.load(a.inputs / "hands.npz")["valid"],
    )
    shutil.copy2(a.inputs / "target_inputs.npz", a.output / "target_inputs.npz")
    np.savez_compressed(a.output / "posture_targets.npz", chest_rotation=chest, head_rotation=head)
    (a.output / "optimization_history.json").write_text(json.dumps(history, indent=2))
    (a.output / "configuration.json").write_text(
        json.dumps(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                "method": "Offline trajectory SQP with segment and orientation fitting, velocity/acceleration regularization, and sampled collision constraints with reported slacks",
                "head_calibration": "Initial observed global head orientation aligned to initial anatomical chest frame; SMPL-X local rotations composed through parents",
                "runtime_seconds": time.monotonic() - start,
            },
            indent=2,
        )
    )
    print("COMPLETE", a.output, flush=True)


if __name__ == "__main__":
    main()

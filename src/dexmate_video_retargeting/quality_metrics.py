"""Common source-frame perceptual diagnostics, using unchanged estimated references."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .model_contract import validate_model
from .posture_ik import unit


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), p95=float(np.percentile(x, 95)), max=float(x.max()))


def frame(w, i, mid, p):
    x = unit(i - p)
    y = mid - w
    y = unit(y - (y * x).sum(-1, keepdims=True) * x)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    z = np.load(a.run / "reference.npz")
    fps = float(z["fps"])
    source_n = len(np.load(a.inputs / "body_targets.npz")["chest_rotation"])
    inds = np.rint(np.arange(source_n) * fps / 30).astype(int)
    inds = inds[inds < len(z["qpos"])]
    q = z["qpos"][inds]
    body = np.load(a.inputs / "body_targets.npz")
    C = body["chest_rotation"][: len(q)]
    t = np.load(a.inputs / "target_inputs.npz")
    names = t["joint_names"].tolist()
    keys = t["mapping_keys"].tolist()
    links = t["mapping_links"].tolist()
    goals = t["human_joint_motions"][: len(q), [names.index(k) for k in keys]]
    m = mujoco.MjModel.from_xml_path(str(a.run / "model/robot.xml"))
    validate_model(m)
    d = mujoco.MjData(m)
    ch = []
    actual = []
    heads = []
    for row in q:
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        ch.append(d.xmat[m.body("torso_l3").id].reshape(3, 3).copy())
        heads.append(d.xmat[m.body("head_l3").id].reshape(3, 3).copy())
        actual.append([d.xpos[m.body(link).id].copy() for link in links])
    actual = np.asarray(actual)
    ch = np.asarray(ch)
    ce = Rotation.from_matrix(C @ ch.transpose(0, 2, 1)).magnitude() * 180 / np.pi
    tilt = np.degrees(np.arccos(np.clip(ch[:, 2, 2], -1, 1)))
    htilt = np.degrees(np.arccos(np.clip(C[:, 2, 2], -1, 1)))
    wrist = (
        np.linalg.norm(
            actual[:, [keys.index(s + "_Wrist") for s in ["L", "R"]]]
            - goals[:, [keys.index(s + "_Wrist") for s in ["L", "R"]]],
            axis=-1,
        )
        * 100
    )
    angles = {}
    palms = {}
    basis = {}
    for side in ["L", "R"]:
        for label, prox, dist in [("upper", "Shoulder", "Elbow"), ("forearm", "Elbow", "Wrist")]:
            i, j = [keys.index(side + "_" + k) for k in [prox, dist]]
            u = unit(actual[:, j] - actual[:, i])
            v = unit(goals[:, j] - goals[:, i])
            angles[side + "_" + label] = np.degrees(np.arccos(np.clip((u * v).sum(-1), -1, 1)))
        ix = [keys.index(side + "_" + k) for k in ["Wrist", "Index1", "Middle1", "Pinky1"]]
        rb = frame(*[actual[:, i] for i in ix])
        hb = frame(*[goals[:, i] for i in ix])
        basis[side] = rb
        palms[side] = Rotation.from_matrix(hb @ rb.transpose(0, 2, 1)).magnitude() * 180 / np.pi
    pair = actual[:, keys.index("L_Wrist")] - actual[:, keys.index("R_Wrist")]
    tpair = goals[:, keys.index("L_Wrist")] - goals[:, keys.index("R_Wrist")]
    report = {
        "reference": "Unchanged stage11_f5d6 estimated and proportion-adapted targets; not measured human ground truth. All comparisons use the same source frames.",
        "frames": len(q),
        "wrist_cm": stats(wrist),
        "chest_orientation_deg": stats(ce),
        "chest_tilt_from_vertical_deg": stats(tilt),
        "human_chest_tilt_deg": stats(htilt),
        "arm_direction_deg": {k: stats(v) for k, v in angles.items()},
        "palm_orientation_deg": {k: stats(v) for k, v in palms.items()},
        "inter_wrist_vector_error_cm": stats(100 * np.linalg.norm(pair - tpair, axis=-1)),
        "note": "Torso roll is structurally unavailable in the three parallel-axis Vega torso joints. Palm orientation uses the same wrist/MCP frame convention on source and target. No contact success is inferred from wrist distance.",
    }
    report["common_30hz_hinge_acceleration_rad_s2"] = stats(
        np.abs(np.diff(q[:, 7:], n=2, axis=0) * 900)
    )
    report["common_30hz_hinge_jerk_rad_s3"] = stats(np.abs(np.diff(q[:, 7:], n=3, axis=0) * 27000))
    report["chest_pitch_error_deg"] = stats(
        np.abs(
            Rotation.from_matrix(ch).as_euler("ZYX", degrees=True)[:, 1]
            - Rotation.from_matrix(C).as_euler("ZYX", degrees=True)[:, 1]
        )
    )
    head_goal = body["head_rotation"][: len(q)]
    report["head_orientation_error_deg"] = stats(
        Rotation.from_matrix(head_goal @ np.asarray(heads).transpose(0, 2, 1)).magnitude()
        * 180
        / np.pi
    )
    out = a.run / "evaluation"
    out.mkdir(exist_ok=True)
    (out / "quality_metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out / "quality_per_frame.npz",
        wrist_cm=wrist,
        chest_error_deg=ce,
        chest_tilt_deg=tilt,
        human_chest_tilt_deg=htilt,
        **{k + "_direction_deg": v for k, v in angles.items()},
        **{k + "_palm_deg": v for k, v in palms.items()},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

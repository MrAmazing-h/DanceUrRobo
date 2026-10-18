"""Sampled geometry/limit diagnostics; not continuous or hardware acceptance."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def audit(model, qpos):
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("Expected finite [frames, model.nq] states")
    data = mujoco.MjData(model)
    joints = np.flatnonzero(
        (model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE) & model.jnt_limited.astype(bool)
    )
    addresses = model.jnt_qposadr[joints]
    pairs = {}
    max_limit = 0.0
    for row in qpos:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        excess = np.maximum(
            model.jnt_range[joints, 0] - row[addresses],
            row[addresses] - model.jnt_range[joints, 1],
        )
        max_limit = max(max_limit, float(np.max(excess, initial=0)))
        for contact in data.contact:
            a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if a and b and contact.dist < 0:
                key = " / ".join(sorted([model.body(int(a)).name, model.body(int(b)).name]))
                pairs[key] = min(pairs.get(key, 0), float(contact.dist))
    return {
        "frames": len(qpos),
        "max_joint_limit_violation_rad": max_limit,
        "negative_self_contact_pairs_minimum_m": pairs,
        "scope": "sampled MuJoCo contacts; not exact triangles, continuous clearance or safety",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    m = mujoco.MjModel.from_xml_path(str(a.model))
    with np.load(a.trajectory, allow_pickle=False) as z:
        result = audit(m, z["qpos"])
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

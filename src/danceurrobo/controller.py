"""Offline GEAR-SONIC adapter. No hardware interface; old experiments untouched."""

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .render import render_saved

ROOT = Path("outputs/g1").resolve()
SOURCE = None
LEG = [
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
WAIST = [f"waist_{a}_joint" for a in ("yaw", "roll", "pitch")]
ARM = [
    f"{s}_{j}_joint"
    for s in ("left", "right")
    for j in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
BODY = LEG + WAIST + ARM
ORDER = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ]
)
NAMES = [BODY[i] for i in ORDER]
DEFAULT = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0] * 2
    + [0, 0, 0]
    + [0.2, 0.2, 0, 0.6, 0, 0, 0]
    + [0.2, -0.2, 0, 0.6, 0, 0, 0]
)[ORDER]
TYPES = [2, 2, 1, 2, 0, 0] * 2 + [1, 0, 0] + [0, 0, 0, 0, 0, 3, 3] * 2
ARMATURE = np.array([0.003609725, 0.010177520, 0.025101925, 0.00425])[TYPES][ORDER]
EFFORT = np.array([25, 88, 139, 5])[TYPES][ORDER]
KP = ARMATURE * (20 * np.pi) ** 2
KD = 4 * ARMATURE * 20 * np.pi
SCALE = 0.25 * EFFORT / KP
DOUBLE = np.array([("ankle" in n or n in ("waist_roll_joint", "waist_pitch_joint")) for n in NAMES])
KP[DOUBLE] *= 2
KD[DOUBLE] *= 2
FIELDS = [
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]


def orientation6(robot, refs):
    """Official row-major first two columns of R(robot)^T R(reference)."""
    return (robot.inv() * refs).as_matrix()[..., :, :2].reshape(-1)


def prepare():
    import torch

    if SOURCE is None:
        raise ValueError("--source is required for --prepare")
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    p = ROOT / "policies/human_joints_info.pkl"
    assert (
        hashlib.sha256(p.read_bytes()).hexdigest()
        == "4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975"
    )
    info = torch.load(p, weights_only=False, map_location="cpu")
    J = info["J"].numpy()
    parents = info["parents_list"]
    src = np.load(SOURCE, allow_pickle=False)
    assert str(src["up_axis"]) == "Z" and str(src["units"]) == "meters"
    ts = np.arange(len(src["body_pose"])) / float(src["fps"])
    t = np.arange(int(ts[-1] * 50) + 1) / 50
    poses = np.concatenate(
        [
            src["global_orient"][:, None, :],
            src["body_pose"].reshape(-1, 21, 3),
            np.zeros((len(ts), 33, 3)),
        ],
        axis=1,
    )
    rot = np.stack(
        [Slerp(ts, R.from_rotvec(poses[:, j]))(t).as_matrix() for j in range(55)], axis=1
    )
    pos = np.empty((len(t), 55, 3))
    glob = np.empty_like(rot)
    pos[:, 0] = J[0]
    glob[:, 0] = rot[:, 0]
    for j in range(1, 55):
        par = parents[j]
        glob[:, j] = glob[:, par] @ rot[:, j]
        pos[:, j] = pos[:, par] + np.einsum("tij,j->ti", glob[:, par], J[j] - J[par])
    anchors = R.from_matrix(glob[:, 0]) * R.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    # Match the published Pico FK adapter, including its nonzero canonical root offset.
    selected = np.r_[np.arange(22), 39, 54]
    local = np.einsum("tij,tkj->tki", anchors.inv().as_matrix(), pos[:, selected])
    centered = np.einsum(
        "tij,tkj->tki", anchors.inv().as_matrix(), pos[:, selected] - pos[:, 0, None]
    )
    yaw = anchors[0].as_euler("xyz")[2]
    aligned = R.from_euler("z", -yaw) * anchors
    dest = ROOT / "reference"
    dest.mkdir(exist_ok=True)
    np.savez_compressed(
        dest / "smpl50.npz",
        time=t,
        local=local,
        centered=centered,
        anchor_xyzw=aligned.as_quat(),
        source_positions=src["global_joint_positions"],
        source_time=ts,
        source_pose=src["body_pose"],
        source_names=src["joint_names"],
        selected_indices=selected,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    )
    model = ROOT / "models/g1_inspire_sonic.xml"
    if not model.is_file():
        raise FileNotFoundError("Run danceurrobo-fetch --output WORKDIR first")
    report = dict(
        source=str(SOURCE),
        frames=len(ts),
        source_fps=float(src["fps"]),
        policy_frames=len(t),
        policy_hz=50,
        canonical_root=J[0].tolist(),
        smpl_point_indices=selected.tolist(),
        smpl_point_names=src["joint_names"][selected].tolist(),
        canonical_body_shape=True,
        finger_poses_zero_in_fk=True,
        robot_hand_targets_neutral=True,
        source_translation_not_a_root_servo=True,
        heading_alignment_only_at_initialization=True,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        body_names=NAMES,
        kp=KP.tolist(),
        kd=KD.tolist(),
        action_scale=SCALE.tolist(),
        known_model_gap="Inspire hand inertia and contact geometry differ from pretrained embodiment. Physical effort caps retained; passive damping/friction retained.",
    )
    (ROOT / "reports/adapter.json").write_text(json.dumps(report, indent=2))
    assert len(FIELDS) == 14 and sum(n for _, n in FIELDS) == 1762
    assert len(set(NAMES)) == 29 and np.allclose(
        orientation6(R.identity(), R.identity()), [1, 0, 0, 1, 0, 0]
    )
    print(
        "PREPARE PASS",
        json.dumps(
            {k: report[k] for k in ("frames", "source_fps", "policy_frames", "smpl_point_names")}
        ),
        flush=True,
    )


class Policy:
    def __init__(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.enc = ort.InferenceSession(
            str(ROOT / "policies/model_encoder.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.dec = ort.InferenceSession(
            str(ROOT / "policies/model_decoder.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        assert self.enc.get_inputs()[0].shape == [1, 1762] and self.dec.get_inputs()[0].shape == [
            1,
            994,
        ]
        self.hist = [collections.deque(maxlen=10) for _ in range(5)]
        self.last = np.zeros(29)

    def step(self, q, dq, root, omega, local=None, anchors=None):
        vals = {name: np.zeros(n) for name, n in FIELDS}
        if local is None:
            # Official GatherEncoderMode: scalar mode id then three zeros (NOT one-hot).
            vals["encoder_mode_4"][0] = 0
            vals["motion_joint_positions_10frame_step5"] = np.tile(DEFAULT, 10)
            vals["motion_anchor_orientation_10frame_step5"] = orientation6(
                root, R.from_quat(np.tile([0, 0, 0, 1], (10, 1)))
            )
        else:
            vals["encoder_mode_4"][0] = 2
            vals["smpl_joints_10frame_step1"] = local.ravel()
            vals["smpl_anchor_orientation_10frame_step1"] = orientation6(root, anchors)
        encin = np.concatenate([vals[name] for name, _ in FIELDS]).astype(np.float32)[None]
        self.last_encoder_input = encin.copy()
        token = self.enc.run(None, {"obs_dict": encin})[0].ravel()
        current = [omega, q - DEFAULT, dq, self.last, root.inv().apply([0, 0, -1])]
        for hist, x in zip(self.hist, current):
            if not hist:
                hist.extend([x.copy() for _ in range(9)])
            hist.append(x.copy())
        obs = np.concatenate([token] + [np.asarray(h).ravel() for h in self.hist]).astype(
            np.float32
        )[None]
        assert obs.shape == (1, 994) and np.isfinite(obs).all()
        self.last = self.dec.run(None, {"obs_dict": obs})[0].ravel().astype(float)
        assert np.isfinite(self.last).all()
        return DEFAULT + SCALE * self.last


def run(case, duration, rate=1.0, centered=False, render=True):
    out = ROOT / "outputs" / case
    out.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models/g1_inspire_sonic.xml"))
    d = mujoco.MjData(m)
    assert np.allclose(m.opt.gravity, [0, 0, -9.81]) and not np.any(m.body_gravcomp)
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and all(
        t == mujoco.mjtEq.mjEQ_JOINT for t in m.eq_type
    )
    qadr = np.array([m.joint(n).qposadr[0] for n in NAMES])
    vadr = np.array([m.joint(n).dofadr[0] for n in NAMES])
    targets = np.zeros(m.nq)
    targets[qadr] = DEFAULT
    d.qpos[qadr] = DEFAULT
    driven = m.actuator_trnid[:, 0]
    aq = m.jnt_qposadr[driven]
    av = m.jnt_dofadr[driven]
    kp = np.full(m.nu, 5.0)
    kd = np.full(m.nu, 0.15)
    for i, j in enumerate(driven):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j))
        if n in NAMES:
            kp[i] = KP[NAMES.index(n)]
            kd[i] = KD[NAMES.index(n)]
    feet = [m.body(s + "_ankle_roll_link").id for s in ("left", "right")]
    mujoco.mj_forward(m, d)
    distances = [
        c.dist
        for c in d.contact
        if (int(m.geom_bodyid[c.geom1]) == 0 and int(m.geom_bodyid[c.geom2]) in feet)
        or (int(m.geom_bodyid[c.geom2]) == 0 and int(m.geom_bodyid[c.geom1]) in feet)
    ]
    correction = max(0.0, 0.001 - min(distances)) if distances else 0.0
    d.qpos[2] += correction
    mujoco.mj_forward(m, d)
    ref = np.load(ROOT / "reference/smpl50.npz")
    local = ref["centered" if centered else "local"]
    anchors = ref["anchor_xyzw"]
    p = Policy()
    rows = []
    failure = None
    tic = time.monotonic()
    isstand = case.startswith("stand")
    timestep = float(m.opt.timestep)
    decim = round(0.02 / timestep)
    assert abs(decim * timestep - 0.02) < 1e-10
    for step in range(round(duration / timestep)):
        now = step * timestep
        source_time = max(0.0, now - 2) * rate
        if step % decim == 0:
            root = R.from_quat(d.qpos[[4, 5, 6, 3]])
            if now < 2 or isstand:
                target = p.step(d.qpos[qadr], d.qvel[vadr], root, d.qvel[3:6])
            else:
                ids = np.minimum(
                    (source_time * 50 + np.arange(10) * rate).astype(int), len(local) - 1
                )
                target = p.step(
                    d.qpos[qadr],
                    d.qvel[vadr],
                    root,
                    d.qvel[3:6],
                    local[ids],
                    R.from_quat(anchors[ids]),
                )
            targets[qadr] = target
            tilt = np.arccos(np.clip(root.as_matrix()[2, 2], -1, 1))
            bad = False
            for c in d.contact:
                a, b = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if min(a, b) == 0 and max(a, b) not in feet:
                    bad = True
            rows.append(
                dict(
                    time=now,
                    qpos=d.qpos.copy(),
                    qvel=d.qvel.copy(),
                    target=target.copy(),
                    action=p.last.copy(),
                    tilt=tilt,
                    bad_ground=bad,
                    source_time=min(source_time, float(ref["time"][-1])),
                    body_pos=d.xpos.copy(),
                )
            )
            if step % 250 == 0:
                print(
                    case,
                    "t",
                    round(now, 2),
                    "height",
                    round(float(d.qpos[2]), 3),
                    "tilt",
                    round(float(np.rad2deg(tilt)), 2),
                    flush=True,
                )
            if (
                d.qpos[2] < 0.4
                or tilt > np.deg2rad(50)
                or not np.isfinite(d.qpos).all()
                or any(w.number for w in d.warning)
            ):
                failure = "fall_or_numerical_warning"
                break
        d.ctrl[:] = kp * (targets[aq] - d.qpos[aq]) - kd * d.qvel[av]
        assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
        mujoco.mj_step(m, d)
    arr = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    hinge_ids = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    full_names = [m.joint(int(j)).name for j in hinge_ids]
    np.savez_compressed(
        out / "trajectory.npz",
        **arr,
        joint_names=full_names,
        qpos_addresses=m.jnt_qposadr[hinge_ids],
        qvel_addresses=m.jnt_dofadr[hinge_ids],
        target_joint_names=NAMES,
        body_names=body_names,
        fps=50,
        root_quaternion="wxyz",
        model_sha256=hashlib.sha256(
            (ROOT / "models/g1_inspire_sonic.xml").read_bytes()
        ).hexdigest(),
        units="m rad s",
        coordinates="world Z-up, base angular velocity local",
        rollout_type="dynamic gravity rollout",
    )
    checks = dict(
        completed=failure is None,
        nonfoot_ground=not arr["bad_ground"].any(),
        tilt_below_35deg=bool(arr["tilt"].max() < np.deg2rad(35)),
    )
    result = dict(
        case=case,
        status="PASS" if all(checks.values()) else "FAIL",
        scope="balance gate only; dance fidelity not certified",
        checks=checks,
        failure=failure,
        simulated_seconds=float(d.time),
        wall_seconds=time.monotonic() - tic,
        max_tilt_deg=float(np.rad2deg(arr["tilt"].max())),
        rate=rate,
        centered=centered,
        body_joint_range_rad=np.ptp(arr["qpos"][:, qadr], axis=0).tolist(),
        initial_root_height_correction=correction,
        root_override=False,
        gravity=m.opt.gravity.tolist(),
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if render:
        render_saved(m, arr, out, case + " SONIC Inspire / full body attempt")
    return result


def main():
    global ROOT, SOURCE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("outputs/g1"))
    ap.add_argument("--source", type=Path)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--case", default="smoke")
    ap.add_argument("--duration", type=float, default=8)
    ap.add_argument("--rate", type=float, default=1)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    if not a.case or Path(a.case).name != a.case or a.case in (".", ".."):
        ap.error("--case must be a simple new folder name")
    if a.duration <= 0 or a.rate <= 0:
        ap.error("duration and rate must be positive")
    ROOT = a.workdir.resolve()
    SOURCE = a.source.resolve() if a.source else None
    if a.prepare:
        prepare()
    else:
        run(a.case, a.duration, a.rate, False, not a.no_render)


if __name__ == "__main__":
    main()

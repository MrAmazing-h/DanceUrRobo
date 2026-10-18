"""Decode every recorded frame; show source skeleton next to physical robot rollout."""

import argparse
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

ROOT = Path("outputs/g1").resolve()
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SEGMENTS = [
    ("left_thigh", "left_hip_yaw_link", "left_knee_link", 1, 4),
    ("right_thigh", "right_hip_yaw_link", "right_knee_link", 2, 5),
    ("left_shin", "left_knee_link", "left_ankle_pitch_link", 4, 7),
    ("right_shin", "right_knee_link", "right_ankle_pitch_link", 5, 8),
    ("left_upperarm", "left_shoulder_roll_link", "left_elbow_link", 16, 18),
    ("right_upperarm", "right_shoulder_roll_link", "right_elbow_link", 17, 19),
    ("left_forearm", "left_elbow_link", "left_wrist_yaw_link", 18, 20),
    ("right_forearm", "right_elbow_link", "right_wrist_yaw_link", 19, 21),
]


def review(case):
    out = ROOT / "outputs" / case
    for name in ("source_vs_robot.mp4", "preview.gif", "review.json", "contact_sheet.jpg"):
        if (out / name).exists():
            raise FileExistsError(out / name)
    r = np.load(ROOT / "reference/smpl50.npz")
    a = np.load(out / "trajectory.npz")
    metric = json.loads((out / "metrics.json").read_text())
    assert np.isfinite(a["qpos"]).all()
    np.testing.assert_allclose(np.diff(a["time"]), 0.02, atol=1e-8)
    bnames = list(a["body_names"])
    active = (a["time"] >= 2) & (a["source_time"] < r["time"][-1])
    ids = np.minimum((a["source_time"] * 50).round().astype(int), len(r["time"]) - 1)
    anchor = R.from_quat(r["anchor_xyzw"][ids])
    angles = {}
    # Root translation cancels; compare bone DIRECTIONS, not unlike limb lengths.
    for name, b0, b1, j0, j1 in SEGMENTS:
        ref = anchor.apply(r["centered"][ids, j1] - r["centered"][ids, j0])
        actual = a["body_pos"][:, bnames.index(b1)] - a["body_pos"][:, bnames.index(b0)]
        dot = np.sum(ref * actual, axis=1) / (
            np.linalg.norm(ref, axis=1) * np.linalg.norm(actual, axis=1)
        )
        error = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))[active]
        angles[name] = dict(mean_deg=float(error.mean()), p95_deg=float(np.percentile(error, 95)))
    source = r["source_positions"]
    stime = r["source_time"]
    frames = 0
    shots = []
    reader = imageio.get_reader(str(out / "dynamic_rollout.mp4"))
    fps = reader.get_meta_data()["fps"]
    assert abs(fps - 25) < 1e-6
    az = np.deg2rad(130)
    el = np.deg2rad(15)
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])

    def project(x):
        return np.stack([240 + 220 * (x @ right), 240 - 220 * (x @ up)], axis=-1)

    with imageio.get_writer(
        str(out / "source_vs_robot.mp4"), fps=fps, codec="libx264", quality=7
    ) as writer:
        for i, frame in enumerate(reader):
            k = min(i * 2, len(a["time"]) - 1)
            s = float(a["source_time"][k])
            si = int(np.argmin(np.abs(stime - s)))
            pic = Image.new("RGB", (1120, 480), (20, 25, 32))
            pic.paste(Image.fromarray(frame), (480, 0))
            draw = ImageDraw.Draw(pic)
            points = source[si, :22].copy()
            points[:, :2] -= points[0, :2]
            points[:, 2] -= 0.9
            xy = project(points)
            for j, p in enumerate(PARENTS):
                if p >= 0:
                    draw.line([tuple(xy[p]), tuple(xy[j])], fill=(90, 210, 240), width=5)
                x, y = xy[j]
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(245, 200, 80))
            draw.text(
                (10, 10),
                "Uploaded HUMAN SMPL-X | kinematic source\nSkeleton only; NOT a simulated human",
                fill="white",
            )
            draw.text(
                (10, 445),
                f"source t={s:.2f}s | rate={metric['rate']:.2f}x | robot t={a['time'][k]:.2f}s",
                fill="white",
            )
            if a["time"][k] < 2:
                draw.text((10, 65), "Robot standing initialization (source held)", fill="orange")
            writer.append_data(np.asarray(pic))
            frames += 1
            if i % max(1, len(a["time"]) // 12) == 0:
                shots.append(pic.resize((560, 240)))
    reader.close()
    assert frames == (len(a["time"]) + 1) // 2
    canvas = Image.new("RGB", (1120, 240 * ((len(shots) + 1) // 2)))
    for i, p in enumerate(shots):
        canvas.paste(p, ((i % 2) * 560, (i // 2) * 240))
    canvas.save(out / "contact_sheet.jpg")
    result = dict(
        case=case,
        balance_gate=metric["status"],
        motion_fidelity="NOT automatically accepted",
        direction_errors=angles,
        note="Human and robot bone segment conventions are approximate. Direction error is diagnostic, not exact joint correspondence or deployment certification.",
        frames=frames,
        video_fps=fps,
        video_seconds=frames / fps,
        npz_samples=len(a["time"]),
        npz_seconds=len(a["time"]) / 50,
        full_source_reached=bool(a["source_time"].max() >= r["time"][-1] - 0.021),
        source_sha256=str(r["source_sha256"]),
    )
    (out / "review.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-n",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(out / "source_vs_robot.mp4"),
            "-ss",
            "2",
            "-t",
            str(min(float(r["time"][-1]) / metric["rate"], max(0, len(a["time"]) / 50 - 2))),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            "[0:v]fps=10,scale=840:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(out / "preview.gif"),
        ],
        check=True,
    )


def main():
    global ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case")
    p.add_argument("--workdir", type=Path, default=Path("outputs/g1"))
    a = p.parse_args()
    if Path(a.case).name != a.case or a.case in (".", ".."):
        p.error("case must be a folder name")
    ROOT = a.workdir.resolve()
    review(a.case)


if __name__ == "__main__":
    main()

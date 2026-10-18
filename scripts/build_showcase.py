"""Compose source-time-aligned demos; never alter or interpolate robot states."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dexmate_video_retargeting.prepare_targets import PARENTS, body_rotations


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g1", type=Path, required=True)
    p.add_argument("--dexmate", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("examples/g1/human_smplx_zup.npz"))
    p.add_argument("--body-targets", type=Path, default=Path("examples/inputs/body_targets.npz"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = np.load(a.source, allow_pickle=False)
    body = np.load(a.body_targets, allow_pickle=False)
    chest, head = body_rotations(source)
    np.testing.assert_array_equal(chest, body["chest_rotation"])
    np.testing.assert_array_equal(head, body["head_rotation"])
    assert float(source["fps"]) == 30 and len(chest) == 590
    human = source["global_joint_positions"][:, :22].copy()
    human[:, :, :2] -= human[:, :1, :2]
    az, el = np.deg2rad([130, 15])
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
    xy = np.stack((human @ right, -(human @ up)), axis=-1)
    lo, hi = xy.min(axis=(0, 1)), xy.max(axis=(0, 1))
    scale = min(260 / (hi[0] - lo[0]), 440 / (hi[1] - lo[1]))
    xy = (xy - (lo + hi) / 2) * scale + [170, 416]

    g1, dex = iio.get_reader(a.g1), iio.get_reader(a.dexmate)
    assert g1.get_meta_data()["fps"] == 25
    assert dex.get_meta_data()["fps"] == 30
    assert g1.get_meta_data()["size"] == (640, 480)
    assert dex.get_meta_data()["size"] == (720, 576)
    fps, frames = 30, 588  # 19.6 s common window; no speed change or end hold.
    movie = a.output / "danceurrobo_showcase.mp4"
    errors = []
    with iio.get_writer(
        movie,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-threads", "2", "-movflags", "+faststart"],
    ) as writer:
        for i in range(frames):
            t = i / fps
            gi = int(np.floor((t + 2) * 25 + 0.5))
            errors.append(abs(gi / 25 - 2 - t))
            # Remove only baked captions / empty border, not robot body parts.
            g = Image.fromarray(g1.get_data(gi)).crop((0, 48, 640, 480))
            d = Image.fromarray(dex.get_data(i)).crop((0, 32, 720, 540))
            g.thumbnail((540, 480), Image.Resampling.LANCZOS)
            d.thumbnail((680, 480), Image.Resampling.LANCZOS)
            im = Image.new("RGB", (1600, 760), "#0b1221")
            draw = ImageDraw.Draw(im)
            draw.text((28, 16), "DANCEURROBO", font=font(17), fill="#63e5d0")
            draw.text((26, 44), "One dance. Two robots.", font=font(40), fill="#f4f7ff")
            draw.text(
                (980, 61),
                "Human reference  /  native-speed comparison",
                font=font(21),
                fill="#a5b3cd",
            )
            panels = [
                (20, 300, "01 / HUMAN", "SMPL-X reference", "#63e5d0"),
                (340, 540, "02 / G1 + INSPIRE", "Dynamic control | gravity ON", "#b3a2ff"),
                (900, 680, "03 / DEXMATE 1P", "Kinematic replay | gravity OFF", "#efbe73"),
            ]
            for x, w, title, subtitle, color in panels:
                draw.rounded_rectangle((x, 112, x + w, 676), radius=15, fill="#141e32")
                draw.line((x + 16, 113, x + w - 16, 113), fill=color, width=3)
                draw.text((x + 16, 125), title, font=font(22), fill=color)
                draw.text((x + 16, 155), subtitle, font=font(17), fill="#c3cde0")
            im.paste(g, (340 + (540 - g.width) // 2, 186 + (480 - g.height) // 2))
            im.paste(d, (900 + (680 - d.width) // 2, 186 + (480 - d.height) // 2))
            draw = ImageDraw.Draw(im)
            for j, parent in enumerate(PARENTS):
                if parent >= 0:
                    draw.line([tuple(xy[i, parent]), tuple(xy[i, j])], fill="#63e5d0", width=6)
                x, y = xy[i, j]
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#e5faff")
            draw.text((35, 642), "Root-centered view", font=font(16), fill="#91a5c3")
            draw.rounded_rectangle((24, 699, 1576, 703), radius=2, fill="#24324a")
            draw.rounded_rectangle(
                (24, 699, 24 + 1552 * (i + 1) / frames, 703), radius=2, fill="#63e5d0"
            )
            draw.text(
                (24, 719),
                f"REFERENCE {t:05.2f} s / 19.60 s   |   1x",
                font=font(18),
                fill="#d9e3f7",
            )
            draw.text(
                (628, 719),
                "Different control modes, not a performance benchmark. Simulation only.",
                font=font(17),
                fill="#9aaac5",
            )
            writer.append_data(np.asarray(im))
            if i in (0, 180, 420, frames - 1):
                im.save(a.output / f"frame_{i:04d}.jpg", quality=95)
    g1.close()
    dex.close()
    with iio.get_reader(movie) as reader:
        count = sum(1 for _ in reader)
        assert count == frames and reader.get_meta_data()["fps"] == fps
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-n",
            "-loglevel",
            "error",
            "-threads",
            "2",
            "-i",
            str(movie),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            "[0:v]fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(a.output / "showcase.gif"),
        ],
        check=True,
    )
    with Image.open(a.output / "showcase.gif") as gif:
        assert gif.n_frames == 196
    report = {
        "source_body_rotation_identity": "590 chest/head frames exactly equal to DexMate frozen inputs",
        "source_sha256": digest(a.source),
        "inputs": {a.g1.name: digest(a.g1), a.dexmate.name: digest(a.dexmate)},
        "g1_video_offset_seconds": 2,
        "dexmate_video_offset_seconds": 0,
        "source_window_seconds": [0, 19.6],
        "frames": count,
        "fps": fps,
        "max_nearest_frame_time_error_seconds": max(errors),
        "speed_factor": 1,
        "motion_states_changed": False,
        "g1_crop_xyxy": [0, 48, 640, 480],
        "dexmate_crop_xyxy": [0, 32, 720, 540],
        "gif_frames": 196,
        "gif_fps": 10,
        "output_sha256": digest(movie),
        "scope": "Presentation only. Gravity-enabled G1 vs kinematic DexMate; not equivalent physical validation.",
    }
    (a.output / "showcase_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Trim the visually inspected outro without changing source motion timing."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--end-frame-exclusive", type=int, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dst = args.output / "dance_30fps_verified.mp4"
    reader = imageio.get_reader(args.input)
    metadata = reader.get_meta_data()
    reader.close()
    assert abs(metadata["fps"] - 30) < 0.001, metadata
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-n",
        "-i",
        str(args.input),
        "-map",
        "0:v:0",
        "-vf",
        f"trim=end_frame={args.end_frame_exclusive},setpts=PTS-STARTPTS",
        "-r",
        "30",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    reader = imageio.get_reader(dst)
    frame_count = sum(1 for _ in reader)
    reader.close()
    assert frame_count == args.end_frame_exclusive, frame_count
    report = {
        "source_name": args.input.name,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_name": dst.name,
        "fps": 30,
        "frames": frame_count,
        "duration_seconds": frame_count / 30,
        "source_frame_range_inclusive": [0, frame_count - 1],
        "trim_basis": "User-specified end-frame-exclusive; inspect the input before trimming.",
        "image_geometry_changed": False,
        "time_warp_applied": False,
        "validation": "Full output decoded; all expected frames present.",
        "limitations": [
            "Monocular video has no measured 3D ground truth.",
            "Occluded hand and body poses must be marked as inferred.",
            "A video quality pass does not validate SMPL-X or robot motion.",
        ],
    }
    (args.output / "video_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

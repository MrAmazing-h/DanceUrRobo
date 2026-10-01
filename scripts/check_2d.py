"""Run GVHMR's actual tracker/ViTPose front end and audit observations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
from hmr4d.utils.net_utils import moving_average_smooth
from hmr4d.utils.preproc import Tracker, VitPoseExtractor
from PIL import Image, ImageDraw


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "GPU allocation is required."
    tracker = Tracker()
    history = tracker.track(str(args.video))
    frame_ids, track_boxes, order = tracker.sort_track_length(history, str(args.video))
    if not order:
        raise RuntimeError("No person track detected; do not continue to 3D.")
    tid = order[0]
    ids = np.asarray(frame_ids[tid])
    raw = np.asarray(track_boxes[tid])
    frames = len(history)
    observed = np.zeros(frames, bool)
    observed[ids] = True
    boxes = torch.from_numpy(
        np.stack([np.interp(np.arange(frames), ids, raw[:, j]) for j in range(4)], axis=-1)
    ).float()
    boxes = moving_average_smooth(boxes, window_size=5, dim=0)
    boxes = moving_average_smooth(boxes, window_size=5, dim=0)
    xys = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
    torch.save(
        {"bbx_xyxy": boxes, "bbx_xys": xys, "observed_mask": torch.from_numpy(observed)},
        args.output / "bbx.pt",
    )
    (args.output / "raw_tracks.json").write_text(
        json.dumps(
            [
                [{"id": int(d["id"]), "bbx_xyxy": d["bbx_xyxy"].tolist()} for d in row]
                for row in history
            ]
        )
    )
    del tracker
    torch.cuda.empty_cache()
    pose = VitPoseExtractor()
    kp = pose.extract(str(args.video), xys)
    torch.save(kp, args.output / "vitpose.pt")
    a = kp.numpy()
    assert a.shape == (frames, 17, 3) and np.isfinite(a).all(), a.shape
    np.savez_compressed(
        args.output / "keypoints_2d.npz", keypoints=a, bbox_observed=observed, fps=np.float32(30)
    )
    names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    conf = a[:, :, 2]
    report = {
        "stage": "2D observations only; not a SMPL-X or DexMate validation",
        "frames": frames,
        "selected_track_id": int(tid),
        "bbox_observed_frames": int(observed.sum()),
        "bbox_interpolated_frames": np.flatnonzero(~observed).tolist(),
        "person_track_count": len(order),
        "keypoint_score_threshold": 0.5,
        "score_interpretation": "ViTPose heatmap score; not calibrated visibility or correctness probability.",
        "fraction_below_threshold": {
            name: float((conf[:, j] < 0.5).mean()) for j, name in enumerate(names)
        },
        "low_score_wrist_frames": np.flatnonzero((conf[:, 9:11] < 0.5).any(axis=1)).tolist(),
        "finite_values": True,
        "next_gate": "Inspect overlay for swaps/occlusions; 3D still requires SMPL-X model assets.",
    }
    (args.output / "validation_2d.json").write_text(json.dumps(report, indent=2))
    edges = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    cap = cv2.VideoCapture(str(args.video))
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(
        str(args.output / "overlay_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
    )
    assert writer.isOpened()
    samples = np.linspace(0, frames - 1, 12, dtype=int).tolist()
    sheet = Image.new("RGB", (4 * 240, 3 * 408), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(frames):
        ok, frame = cap.read()
        assert ok, i
        for j, k in edges:
            if min(conf[i, j], conf[i, k]) >= 0.5:
                cv2.line(
                    frame,
                    tuple(a[i, j, :2].astype(int)),
                    tuple(a[i, k, :2].astype(int)),
                    (0, 220, 0),
                    3,
                )
        for j in range(17):
            if conf[i, j] >= 0.5:
                cv2.circle(frame, tuple(a[i, j, :2].astype(int)), 5, (0, 200, 255), -1)
        cv2.putText(
            frame,
            f"2D estimate | frame {i} | {i / 30:.2f}s",
            (12, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
        if i in samples:
            k = samples.index(i)
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((240, 382))
            sheet.paste(im, ((k % 4) * 240, (k // 4) * 408))
            draw.text(
                ((k % 4) * 240 + 3, (k // 4) * 408 + 385),
                f"{i / 30:.2f}s | frame {i}",
                fill="black",
            )
    writer.release()
    cap.release()
    sheet.save(args.output / "overlay_contact_sheet.jpg")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

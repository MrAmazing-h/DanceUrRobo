"""Estimate hands and assign their identity using the already audited body wrists."""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--body-2d", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    body = np.load(a.body_2d)["keypoints"]
    n = len(body)
    world = np.full((n, 2, 21, 3), np.nan, np.float32)
    image = world.copy()
    scores = np.zeros((n, 2), np.float32)
    cap = cv2.VideoCapture(str(a.video))
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(
        str(a.output / "hands_overlay_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
    )
    assert writer.isOpened()
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as tracker:
        for i in range(n):
            ok, frame = cap.read()
            assert ok
            result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                detected = np.array(
                    [
                        [[v.x * w, v.y * h, v.z * w] for v in hand.landmark]
                        for hand in result.multi_hand_landmarks
                    ]
                )
                dist = np.linalg.norm(detected[:, None, 0, :2] - body[None, i, 9:11, :2], axis=-1)
                rows, cols = linear_sum_assignment(dist)
                torso = np.linalg.norm(body[i, [5, 6], :2].mean(0) - body[i, [11, 12], :2].mean(0))
                for row, side in zip(rows, cols):
                    score = result.multi_handedness[row].classification[0].score
                    # Handedness probability is recorded, not used to swap labels in a mirrored clip.
                    if dist[row, side] > max(25.0, torso * 0.35):
                        continue
                    image[i, side] = detected[row]
                    world[i, side] = [
                        [v.x, v.y, v.z] for v in result.multi_hand_world_landmarks[row].landmark
                    ]
                    scores[i, side] = score
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.multi_hand_landmarks[row], mp.solutions.hands.HAND_CONNECTIONS
                    )
                    xy = tuple(detected[row, 0, :2].astype(int))
                    cv2.putText(
                        frame, ["L", "R"][side], xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                    )
            writer.write(frame)
            if i % 60 == 0:
                print("hands", i, "/", n, flush=True)
    cap.release()
    writer.release()
    valid = np.isfinite(world).all(axis=(2, 3))
    np.savez_compressed(
        a.output / "hands.npz",
        world=world,
        image=image,
        valid=valid,
        handedness_score=scores,
        fps=30.0,
    )
    report = {
        "frames": n,
        "detected_left_frames": int(valid[:, 0].sum()),
        "detected_right_frames": int(valid[:, 1].sum()),
        "both_hands_frames": int(valid.all(axis=1).sum()),
        "identity": "Assigned by proximity to ViTPose left/right wrists; not by mirrored handedness labels.",
        "limitation": "Estimated relative 3D hand shape, not measured 3D ground truth; missing frames remain NaN and flagged.",
    }
    (a.output / "validation_hands.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

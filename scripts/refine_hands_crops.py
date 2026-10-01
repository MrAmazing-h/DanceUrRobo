"""Recover additional visible hands from body-wrist-guided image crops."""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def main():
    p = argparse.ArgumentParser()
    for name in ["video", "body-2d", "initial", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    body = np.load(a.body_2d)["keypoints"]
    old = np.load(a.initial)
    world = old["world"].copy()
    image = old["image"].copy()
    scores = old["handedness_score"].copy()
    original = old["valid"].copy()
    n = len(body)
    added = np.zeros((n, 2), bool)
    cap = cv2.VideoCapture(str(a.video))
    w, h = int(cap.get(3)), int(cap.get(4))
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5
    ) as tracker:
        for i in range(n):
            ok, frame = cap.read()
            assert ok
            for side in range(2):
                if original[i, side]:
                    continue
                wrist = body[i, 9 + side, :2]
                elbow = body[i, 7 + side, :2]
                center = wrist + 0.15 * (wrist - elbow)
                half = max(75, int(np.linalg.norm(wrist - elbow) * 0.85))
                x0, y0 = np.maximum((center - half).astype(int), 0)
                x1, y1 = np.minimum((center + half).astype(int), [w, h])
                crop = frame[y0:y1, x0:x1]
                result = tracker.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if not result.multi_hand_landmarks:
                    continue
                candidates = np.array(
                    [
                        [
                            [v.x * (x1 - x0) + x0, v.y * (y1 - y0) + y0, v.z * (x1 - x0)]
                            for v in hand.landmark
                        ]
                        for hand in result.multi_hand_landmarks
                    ]
                )
                errors = np.linalg.norm(candidates[:, 0, :2] - wrist, axis=-1)
                j = int(errors.argmin())
                other = np.linalg.norm(candidates[j, 0, :2] - body[i, 10 - side, :2])
                if errors[j] > 35 or other < errors[j] + 5:
                    continue
                world[i, side] = [
                    [v.x, v.y, v.z] for v in result.multi_hand_world_landmarks[j].landmark
                ]
                image[i, side] = candidates[j]
                scores[i, side] = result.multi_handedness[j].classification[0].score
                added[i, side] = True
            if i % 100 == 0:
                print("crop", i, flush=True)
    cap.release()
    valid = np.isfinite(world).all(axis=(2, 3))
    np.savez_compressed(
        a.output / "hands.npz",
        world=world,
        image=image,
        valid=valid,
        handedness_score=scores,
        crop_added=added,
        fps=30.0,
    )
    report = {
        "frames": n,
        "detected_left_frames": int(valid[:, 0].sum()),
        "detected_right_frames": int(valid[:, 1].sum()),
        "both_hands_frames": int(valid.all(axis=1).sum()),
        "crop_added_per_hand": added.sum(axis=0).tolist(),
        "missing_policy": "Still missing observations remain NaN; never relabel an inferred pose as observed.",
    }
    (a.output / "validation_hands.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

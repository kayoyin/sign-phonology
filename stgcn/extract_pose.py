"""Extract MediaPipe Holistic keypoints from a directory of sign videos.

Saves one ``.npy`` per video, of shape (T, 543, 3): 33 body landmarks,
21 right-hand, 21 left-hand, 468 face. The pose dataset (``stgcn/dataset.py``)
later subsamples this layout down to a 27-node graph.
"""

import argparse
import csv
import os
from timeit import default_timer as timer

import cv2
import mediapipe as mp
import numpy as np


N_LANDMARKS = 543  # 33 body + 21 right-hand + 21 left-hand + 468 face


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_dir", required=True, help="Directory of source videos.")
    p.add_argument("--output_dir", required=True, help="Where to write .npy keypoints.")
    p.add_argument("--metadata_csv", required=True,
                   help="Split CSV; column 1 is the video filename.")
    p.add_argument("--min_detection_confidence", type=float, default=0.5)
    return p.parse_args()


def extract_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, image = cap.read()
        if not success:
            break
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        feature = np.zeros((N_LANDMARKS, 3))

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                feature[i] = (lm.x, lm.y, lm.z)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                feature[33 + i] = (lm.x, lm.y, lm.z)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                feature[54 + i] = (lm.x, lm.y, lm.z)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                feature[75 + i] = (lm.x, lm.y, lm.z)
        frames.append(feature)
    cap.release()
    return np.array(frames)


def main():
    args = parse_args()
    mp_holistic = mp.solutions.holistic

    n_done = 0
    start = timer()
    with open(args.metadata_csv, "r") as fh:
        reader = csv.reader(fh)
        for row in reader:
            video_rel = row[1]
            stem = os.path.splitext(video_rel)[0]
            out_path = os.path.join(args.output_dir, f"{stem}.npy")
            if os.path.exists(out_path):
                continue
            video_path = os.path.join(args.video_dir, video_rel)
            if not os.path.exists(video_path):
                print(f"missing video: {video_path}")
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=args.min_detection_confidence,
            ) as holistic:
                features = extract_video(video_path, holistic)

            if features.size == 0 or np.all(features == 0):
                print(f"all-zero pose for {video_rel}")
                continue
            np.save(out_path, features)

            n_done += 1
            if n_done % 10 == 0:
                elapsed = timer() - start
                print(f"{n_done} videos, last 10 took {elapsed:.1f}s")
                start = timer()

    print(f"Done: {n_done} pose files written to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Articulatory handshape distance from MediaPipe hand landmarks.

For every ordered pair of handshapes in the Handshapes-in-Context Stimuli (HCS)
data, computes the mean absolute angular difference at 15 hand joints (per
MediaPipe's standard layout). The output, one CSV per phonological context
(location x movement x orientation), is the geometric reference metric used
in the paper's Table 3 and Figure 4 (HD).

The HCS data is a directory of pose ``.npy`` files in subfolders named
``"<HS> Handshape"``, with files ``"<HS>_<Location>_<Movement>_<Orientation>.npy"``.
By default we read the *left* hand (indices 54-74); pass ``--hand right`` to
read the right hand (indices 33-53).
"""

import argparse
import os

import numpy as np
import pandas as pd


# MediaPipe joint definitions: (previous, vertex, next) hand landmark indices.
JOINT_DEFINITIONS = {
    # Thumb
    "Thumb_CMC": (0, 1, 2),
    "Thumb_MCP": (1, 2, 3),
    "Thumb_IP":  (2, 3, 4),
    # Index
    "Index_MCP": (0, 5, 6),
    "Index_PIP": (5, 6, 7),
    "Index_DIP": (6, 7, 8),
    # Middle
    "Middle_MCP": (0, 9, 10),
    "Middle_PIP": (9, 10, 11),
    "Middle_DIP": (10, 11, 12),
    # Ring
    "Ring_MCP":   (0, 13, 14),
    "Ring_PIP":   (13, 14, 15),
    "Ring_DIP":   (14, 15, 16),
    # Pinky
    "Pinky_MCP":  (0, 17, 18),
    "Pinky_PIP":  (17, 18, 19),
    "Pinky_DIP":  (18, 19, 20),
}

LOCATIONS = ["Elbow", "Mouth", "Neutral"]
MOVEMENTS = ["Circle", "Lateral", "Twist"]
ORIENTATIONS = ["Down", "Forward", "Self"]

# Handshape inventory mirrors HCS (36 ASL handshapes).
HANDSHAPES = [
    "1", "i-1", "3", "4", "5", "8", "10", "A", "Closed 5", "Adducted Bent 5",
    "B", "Bent L", "C", "Bent 3", "Bent 5", "Bent V", "D", "E", "F", "Flat O",
    "G", "I", "K", "L", "L-i", "O", "R", "Open 8", "Radial U", "S",
    "Closed X", "U", "V", "W", "X", "Y",
]


def angle_3d(a, b, c):
    """Interior angle at vertex b formed by points a-b-c (degrees)."""
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def compare_hand_poses(hand_a, hand_b):
    """Mean absolute joint-angle difference between two single-frame hand poses."""
    diffs = []
    for prev, curr, nxt in JOINT_DEFINITIONS.values():
        diffs.append(abs(angle_3d(hand_a[prev], hand_a[curr], hand_a[nxt])
                         - angle_3d(hand_b[prev], hand_b[curr], hand_b[nxt])))
    return float(np.mean(diffs))


def load_hand_pose(pose_dir, handshape, location, movement, orientation, hand_slice):
    fname = f"{handshape.replace(' ', '_')}_{location}_{movement}_{orientation}.npy"
    path = os.path.join(pose_dir, f"{handshape} Handshape", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    return data[len(data) // 2, hand_slice]  # 21 hand landmarks at the median frame


def make_distance_matrix(pose_dir, location, movement, orientation, hand_slice):
    n = len(HANDSHAPES)
    matrix = np.full((n, n), np.nan)
    for i, name_a in enumerate(HANDSHAPES):
        try:
            hand_a = load_hand_pose(pose_dir, name_a, location, movement, orientation, hand_slice)
        except FileNotFoundError as e:
            print(f"missing: {e}")
            continue
        for j, name_b in enumerate(HANDSHAPES):
            try:
                hand_b = load_hand_pose(pose_dir, name_b, location, movement, orientation, hand_slice)
            except FileNotFoundError as e:
                print(f"missing: {e}")
                continue
            try:
                matrix[i, j] = compare_hand_poses(hand_a, hand_b)
            except Exception as exc:
                print(f"error on {name_a} vs {name_b}: {exc}")
    return pd.DataFrame(matrix, index=HANDSHAPES, columns=HANDSHAPES)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose_dir", required=True,
                   help="Directory of HCS pose .npy files, organized in '<HS> Handshape/' subfolders.")
    p.add_argument("--output_dir", required=True,
                   help="Directory where one CSV per (location, movement, orientation) is written.")
    p.add_argument("--hand", choices=["left", "right"], default="left",
                   help="Which hand's landmarks to read (default: left, indices 54-74).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    hand_slice = slice(54, 75) if args.hand == "left" else slice(33, 54)

    for loc in LOCATIONS:
        for mov in MOVEMENTS:
            for ori in ORIENTATIONS:
                cm = make_distance_matrix(args.pose_dir, loc, mov, ori, hand_slice)
                out = os.path.join(args.output_dir, f"{loc}_{mov}_{ori}.csv")
                cm.to_csv(out)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()

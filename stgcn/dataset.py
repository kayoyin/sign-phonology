"""ASL Citizen / Sem-Lex pose-graph dataset for ST-GCN.

Loads MediaPipe Holistic keypoints (.npy) and emits a (2, T, 27) skeletal
graph normalized by inter-shoulder distance. The 27 retained nodes cover the
upper body (5 nodes) and both hands (11 nodes each). See ``KEYPOINT_INDICES``
for the exact subset.
"""

import csv
import os

import numpy as np
import torch
import torch.utils.data as data_utl


# Indices into the 75-point upper-body + hands MediaPipe layout used by
# ``extract_pose.py``. 0 = nose; 11/12 = shoulders; 33-53 = right hand;
# 54-74 = left hand.
KEYPOINT_INDICES = [
    0, 2, 5, 11, 12, 13, 14,
    33, 37, 38, 41, 42, 45, 46, 49, 50, 53,
    54, 58, 59, 62, 63, 66, 67, 70, 71, 74,
]


def downsample(frames, max_frames):
    """Linear downsample so that ``len(frames) <= max_frames``."""
    length = frames.shape[0]
    increment = min(max_frames / length, 1.0)
    out = []
    cur, target = 0.0, 0
    for f in frames:
        cur += increment
        if cur > target:
            target += 1
            out.append(f)
    return np.array(out[:max_frames])


class ASLCitizen(data_utl.Dataset):
    """Reads MediaPipe pose ``.npy`` files for ST-GCN classification.

    Args:
        datadir: directory containing pose ``.npy`` files (one per clip).
        video_file: split CSV with columns ``Participant ID, Video file, Gloss, ...``.
            ``.mp4`` / ``.webm`` extensions are mapped to ``.npy`` to find the pose file.
        gloss_dict: optional pre-built ``gloss -> class index`` mapping.
        transforms: optional pose-space transform (e.g. ``ShearTransform``).
        gloss_filter: glosses to drop from the split.
        max_frames: temporal length to pad/downsample to.
    """

    def __init__(self, datadir, video_file=None, gloss_dict=None, transforms=None,
                 gloss_filter=None, max_frames=128):
        self.max_frames = max_frames
        self.transforms = transforms
        self.pose_paths = []
        self.video_info = []
        self.labels = []
        gloss_filter = set(gloss_filter or [])

        if gloss_dict is None:
            seen = []
            with open(video_file, "r") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for row in reader:
                    g = row[2].strip()
                    if g not in seen:
                        seen.append(g)
            seen.sort()
            self.gloss_dict = {g: i for i, g in enumerate(seen)}
        else:
            self.gloss_dict = gloss_dict

        with open(video_file, "r") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                g = row[2].strip()
                if g in gloss_filter or g not in self.gloss_dict:
                    continue
                pose_fname = row[1].replace(".mp4", ".npy").replace(".webm", ".npy")
                pose_path = os.path.join(datadir, pose_fname)
                if not os.path.exists(pose_path):
                    print(f"Pose file not found: {pose_path}")
                    continue
                self.pose_paths.append(pose_path)
                self.video_info.append(row)
                self.labels.append(self.gloss_dict[g])

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, index):
        pose = np.load(self.pose_paths[index])[:, :, :2]
        length = pose.shape[0]
        if length > self.max_frames:
            pose = downsample(pose, self.max_frames)
        elif length < self.max_frames:
            pose = np.pad(pose, ((0, self.max_frames - length), (0, 0), (0, 0)))

        # Normalize to inter-shoulder distance, centered on the midpoint.
        shoulder_l = pose[:, 11, :]
        shoulder_r = pose[:, 12, :]
        center = ((shoulder_l + shoulder_r) / 2).mean(axis=0)
        mean_dist = np.mean(np.linalg.norm(shoulder_l - shoulder_r, axis=-1))
        if mean_dist != 0:
            pose = (pose - center) / mean_dist

        # Reorder so right and left hands trail the body, then take the keypoint subset.
        body, rh, lh = pose[:, :33, :], pose[:, 33:54, :], pose[:, 54:75, :]
        graph = np.concatenate([body, lh, rh], axis=1)[:, KEYPOINT_INDICES, :]
        # ST-GCN expects (channels, time, nodes).
        data = np.transpose(graph, (2, 0, 1))

        out = torch.from_numpy(data).double()
        if self.transforms:
            out = self.transforms(out)

        info = self.video_info[index]
        name = {"user": info[0], "filename": info[1], "gloss": info[2]}

        label = np.zeros(len(self.gloss_dict))
        label[self.labels[index]] = 1
        return out, name, torch.tensor(label, dtype=torch.float)

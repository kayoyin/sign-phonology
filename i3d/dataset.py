"""ASL Citizen / Sem-Lex video dataset for I3D.

Loads RGB frames from .mp4/.webm clips, downsamples to 64 frames,
and emits one-hot gloss labels alongside metadata.
"""

import csv
import math
import os
import subprocess

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def _ffprobe_frame_count(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", video_path,
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return int(out) if out else None


def load_rgb_frames_from_video(video_path, max_frames=64):
    """Read up to ``max_frames`` RGB frames centered in the clip and rescale to ~256px."""
    vidcap = cv2.VideoCapture(video_path)
    # OpenCV's frame count is unreliable for .webm; fall back to ffprobe.
    if video_path.endswith(".webm"):
        total_frames = _ffprobe_frame_count(video_path)
        if total_frames is None:
            raise ValueError(f"Could not retrieve frame count from {video_path}")
    else:
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    if total_frames >= 160:
        frameskip = 3
    elif total_frames >= 96:
        frameskip = 2
    else:
        frameskip = 1

    if frameskip == 3:
        start = np.clip(int((total_frames - 192) // 2), 0, 160)
    elif frameskip == 2:
        start = np.clip(int((total_frames - 128) // 2), 0, 96)
    else:
        start = np.clip(int((total_frames - 64) // 2), 0, 64)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    for offset in range(0, min(max_frames * frameskip, int(total_frames - start))):
        success, img = vidcap.read()
        if not success:
            break
        if offset % frameskip != 0:
            continue
        h, w, _ = img.shape
        if h < 226 or w < 226:
            scale = 1 + (226.0 - min(h, w)) / min(h, w)
            img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        if h > 256 or w > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
        img = (img / 255.0) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic):
    """numpy (T, H, W, C) -> torch (C, T, H, W)."""
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class ASLCitizen(data_utl.Dataset):
    """Reads ASL Citizen / Sem-Lex video clips for I3D classification.

    Args:
        datadir: directory containing the video files referenced in ``video_file``.
        transforms: torchvision-style spatial transform applied to (T, H, W, C) frames.
        video_file: CSV with columns ``Participant ID, Video file, Gloss, ...``.
        gloss_dict: optional pre-built ``gloss -> class index`` mapping (reuse the
            train split's mapping when constructing val/test datasets).
        gloss_filter: glosses to drop from the split. Used to exclude minimal-pair
            test signs from training (see paper Sec. "Strict exclusion of test signs").
    """

    def __init__(self, datadir, transforms, video_file, gloss_dict=None, gloss_filter=None):
        self.transforms = transforms
        self.video_paths = []
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
                self.video_paths.append(os.path.join(datadir, row[1]))
                self.video_info.append(row)
                self.labels.append(self.gloss_dict[g])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        total_frames = 64
        imgs = load_rgb_frames_from_video(self.video_paths[index], total_frames)
        imgs = self._pad(imgs, total_frames)
        imgs = self.transforms(imgs)

        label = np.zeros(len(self.gloss_dict))
        label[self.labels[index]] = 1
        label = np.tile(label, (total_frames, 1)).T  # (n_classes, T)

        info = self.video_info[index]
        name = {"user": info[0], "filename": info[1], "gloss": info[2]}
        return video_to_tensor(imgs), name, torch.tensor(label, dtype=torch.float)

    @staticmethod
    def _pad(imgs, total_frames):
        if imgs.shape[0] >= total_frames:
            return imgs
        pad_idx = 0 if np.random.random_sample() > 0.5 else -1
        pad = np.tile(np.expand_dims(imgs[pad_idx], 0), (total_frames - imgs.shape[0], 1, 1, 1))
        return np.concatenate([imgs, pad], axis=0)

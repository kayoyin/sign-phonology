# ST-GCN (pose-based)

The Spatio-Temporal Graph Convolutional Network operates on a 27-node
skeletal graph derived from MediaPipe Holistic keypoints. Two variants are
evaluated in the paper:

| Variant       | Source                                                    |
| ------------- | --------------------------------------------------------- |
| `STGCN-Rand`  | randomly initialized weights                              |
| `STGCN-ASL`   | re-trained on ASL Citizen (test-sign exclusion applied)   |

The paper does **not** evaluate an ST-GCN trained on Kinetics — standard
action-recognition pose graphs lack the fine-grained hand landmarks needed
to capture sign phonology.

## Files

- **`architecture/`** — ST-GCN backbone, adapted from
  [AI4Bharat/OpenHands](https://github.com/AI4Bharat/OpenHands).
  - `st_gcn.py` — encoder.
  - `fc.py`     — classification head.
  - `graph_utils.py` / `network.py` — supporting utilities.
- **`graph_args.py`** — shared graph topology (27 nodes covering body
  shoulders/elbows + both hands). Imported by training, testing, and
  feature extraction so they always agree.
- **`pose_transforms.py`** — `Compose`, `ShearTransform`, `RotatationTransform`
  on pose tensors.
- **`dataset.py`** — `ASLCitizen` pose dataset. Loads `.npy` files, normalizes
  by inter-shoulder distance, takes the 27-node subset, and returns
  `(channels=2, time=128, nodes=27)` tensors.
- **`extract_pose.py`** — MediaPipe Holistic keypoint extraction. Walks a
  split CSV and writes one `.npy` (T, 543, 3) per video.
- **`train.py`** — train loop. Supports `--gloss_filter`.
- **`test.py`** — top-k accuracy / MRR / DCG eval.
- **`extract_features.py`** — runs the encoder in eval mode and dumps
  per-clip representations into a JSON. Same format as `i3d/extract_features.py`.

## Quick start

```bash
# 1. Pose extraction
python stgcn/extract_pose.py \
    --video_dir /path/to/ASL_Citizen/videos \
    --output_dir /path/to/ASL_Citizen/pose_files \
    --metadata_csv /path/to/splits/all.csv

# 2. Train STGCN-ASL
python stgcn/train.py \
    --pose_dir /path/to/ASL_Citizen/pose_files \
    --train_csv /path/to/splits/train.csv \
    --val_csv /path/to/splits/val.csv \
    --gloss_filter /path/to/splits/minimal_pair_glosses.txt

# 3. Extract penultimate features
python stgcn/extract_features.py \
    --pose_dir /path/to/ASL_Citizen/pose_files \
    --metadata_csv /path/to/splits/test.csv \
    --output_json features/stgcn_asl_test.json \
    --train_pose_dir /path/to/ASL_Citizen/pose_files \
    --train_csv /path/to/splits/train.csv \
    --weights checkpoints/stgcn_asl.pt
```

The `--train_pose_dir` / `--train_csv` arguments to `extract_features.py`
are only used to recreate the gloss vocabulary so the FC head's
output dimension matches the saved checkpoint.

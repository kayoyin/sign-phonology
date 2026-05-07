# I3D (pixel-based)

The Inflated 3D ConvNet operates on RGB video frames. Three variants are
evaluated in the paper:

| Variant     | Source                                     |
| ----------- | ------------------------------------------ |
| `I3D-Rand`  | randomly initialized weights               |
| `I3D-Kine`  | pre-trained on Kinetics-400                |
| `I3D-ASL`   | re-trained on ASL Citizen (test-sign exclusion applied) |

## Files

- **`pytorch_i3d.py`** — I3D architecture (from
  [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)).
- **`videotransforms.py`** — RandomCrop / CenterCrop / RandomHorizontalFlip
  for (T, H, W, C) numpy stacks.
- **`dataset.py`** — `ASLCitizen` dataset. Reads the split CSV, decodes
  videos with OpenCV (with `ffprobe` fallback for `.webm`), centers the clip
  at 64 frames, returns `(C, T, H, W)` tensors.
- **`train.py`** — train loop. Supports a `--gloss_filter` argument to
  exclude minimal-pair glosses from training (the paper's "Strict exclusion
  of test signs"). Optional W&B logging via `--wandb_project`.
- **`test.py`** — top-k accuracy / MRR / DCG over the test split. Not used
  for the paper's primary results, but useful for sanity-checking
  re-trained checkpoints.
- **`extract_features.py`** — runs the model in eval mode and dumps the
  penultimate-layer (post-`extract_features`) representation per clip into
  a JSON file. This is the format consumed by
  `analysis/minimal_pair_ttest.py` and `analysis/synthetic_cm.py`.

## Quick start

See the top-level [`README.md`](../README.md) for a fully wired pipeline.
The minimal CLI:

```bash
# Train (re-train ASL-trained variant, excluding test glosses)
python i3d/train.py \
    --video_dir /path/to/ASL_Citizen/videos \
    --train_csv /path/to/splits/train.csv \
    --val_csv /path/to/splits/val.csv \
    --gloss_filter /path/to/splits/minimal_pair_glosses.txt \
    --init_weights /path/to/kinetics_pretrained.pt

# Extract penultimate features
python i3d/extract_features.py \
    --video_dir /path/to/ASL_Citizen/videos \
    --metadata_csv /path/to/splits/test.csv \
    --output_json features/i3d_asl_test.json \
    --weights checkpoints/i3d_asl.pt --num_classes 2731
```

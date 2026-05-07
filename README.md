# Phonological Perception of Sign Language Models

Code for the CogSci 2026 paper *Phonological Perception of Sign Language
Models* (Yin, Carter, Kocab & Lu). We probe whether I3D and ST-GCN models
implicitly represent sign phonology by comparing penultimate-layer features
across minimal pairs from ASL Citizen and Sem-Lex, and against human
handshape perception on the Handshapes-in-Context Stimuli (HCS) data.

## Layout

```
i3d/        I3D pixel-based model (dataset, train, test, extract_features)
stgcn/      ST-GCN pose-based model (incl. extract_pose, graph_args)
analysis/   minimal-pair t-test, handshape distance, HCS confusion matrices
scripts/    shell wrappers around the Python entry points
```

## Setup

```bash
pip install -r requirements.txt
pip install mediapipe   # only needed for stgcn/extract_pose.py
```

## Datasets

| Dataset      | Use                          | Source |
| ------------ | ---------------------------- | ------ |
| ASL Citizen  | Training, Tables 1 & 3       | https://www.microsoft.com/en-us/research/project/asl-citizen/ |
| Sem-Lex      | Out-of-domain test (Table 2) | https://github.com/leekezar/sem-lex |
| HCS          | Handshape stimuli (Table 3, Figure 4) | (released with the accompanying stimuli paper) |

> **TODO:** the curated minimal-pair stimuli (`minimal_pairs_asl_citizen.csv`,
> `minimal_pairs_semlex.csv`), per-pair t-test outputs, handshape-distance
> CSVs, and the `results.ipynb` paper-table notebook will be hosted at a
> separate URL — link to come.

`data/` and `results/` are gitignored: scripts read curated stimuli from
`data/` and write paper artifacts to `results/`.

## Pipeline

All shell wrappers read environment variables and forward to a Python entry
point with the same name. The Python scripts also work standalone — see each
module's README for the full CLI.

**1. Pose extraction (ST-GCN only).**

```bash
VIDEO_DIR=… POSE_DIR=… METADATA_CSV=…/all.csv  scripts/extract_pose.sh
```

**2. Train (optional; paper uses checkpoints with minimal-pair test signs excluded).**

```bash
VIDEO_DIR=… TRAIN_CSV=… VAL_CSV=… GLOSS_FILTER=…/minimal_pair_glosses.txt \
  INIT_WEIGHTS=…/kinetics.pt  scripts/train_i3d.sh

POSE_DIR=… TRAIN_CSV=… VAL_CSV=… GLOSS_FILTER=…/minimal_pair_glosses.txt \
  scripts/train_stgcn.sh
```

**3. Extract penultimate features** (one JSON per `model × variant × split`):

```bash
MODEL=i3d   MODEL_TAG=asl  SPLIT_NAME=test  FEATURE_DIR=features \
  VIDEO_DIR=… METADATA_CSV=…/test.csv \
  I3D_WEIGHTS=…/i3d_asl.pt  I3D_CLASSES=2731  scripts/extract_features.sh

MODEL=stgcn MODEL_TAG=asl  SPLIT_NAME=test  FEATURE_DIR=features \
  POSE_DIR=… METADATA_CSV=…/test.csv \
  TRAIN_POSE_DIR=… TRAIN_CSV=…/train.csv \
  STGCN_WEIGHTS=…/stgcn_asl.pt  scripts/extract_features.sh
```

Use `MODEL_TAG=rand` (no `*_WEIGHTS`) for the random baseline; `kine` with
`I3D_CLASSES=400` for the Kinetics I3D.

**4. Minimal-pair t-test (Tables 1–2).**

```bash
FEATURES=features/i3d_asl_test.json \
MINIMAL_PAIRS=data/minimal_pairs_asl_citizen.csv \
OUTPUT_DIR=outputs/i3d_asl_test  scripts/run_minimal_pair_ttest.sh

# Sem-Lex: pass SEMLEX_METADATA to filter to videos present in the split.
FEATURES=features/i3d_asl_semlex.json \
MINIMAL_PAIRS=data/minimal_pairs_semlex.csv \
SEMLEX_METADATA=…/semlex/splits/all.csv \
OUTPUT_DIR=outputs/i3d_asl_semlex  scripts/run_minimal_pair_ttest.sh
```

**5. Handshape distance and HCS confusion matrices (Table 3, Figure 4).**

```bash
python analysis/handshape_distance.py \
  --pose_dir …/HCS/pose_files --output_dir results/handshape_distance --hand left

python analysis/synthetic_cm.py \
  --features features/stgcn_asl_hcs.json \
  --output_dir results/hcs_cm --model_tag stgcn_asl
```

## Citation

```bibtex
@inproceedings{yin2026phonological,
  title={Phonological Perception of Sign Language Models},
  author={Yin, Kayo and Carter, Jessica and Kocab, Annemarie and Lu, Alex X.},
  booktitle={Proceedings of the Annual Conference of the Cognitive Science Society},
  year={2026},
}
```

## Attribution

I3D backbone: [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d).
ST-GCN backbone: [AI4Bharat/OpenHands](https://github.com/AI4Bharat/OpenHands).
Pose transforms: [AmitMY/pose-format](https://github.com/AmitMY/pose-format).
Keypoints from MediaPipe Holistic.

## License

MIT — see [`LICENSE`](LICENSE).

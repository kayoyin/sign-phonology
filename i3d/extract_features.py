"""Extract I3D penultimate-layer features for every clip in a CSV split.

Outputs one JSON file with a list of ``{user, filename, gloss, features}``
records, time-averaged from the layer that precedes the classification head.
This is the input format consumed by ``analysis/minimal_pair_ttest.py`` and
``analysis/synthetic_cm.py``.

Variants used in the paper:
  * ``i3d_rand``    — randomly initialized weights (no checkpoint).
  * ``i3d_kine``    — Kinetics action-recognition checkpoint (pre-trained, 400 logits).
  * ``i3d_asl``     — re-trained on ASL Citizen with the minimal-pair test
                      glosses excluded (2731-way logit head).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# Allow ``python i3d/extract_features.py`` from the repo root.
sys.path.insert(0, os.path.dirname(__file__))
import videotransforms
from dataset import ASLCitizen
from pytorch_i3d import InceptionI3d


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_dir", required=True, help="Directory of input videos.")
    p.add_argument("--metadata_csv", required=True,
                   help="Split CSV listing the videos to encode.")
    p.add_argument("--output_json", required=True, help="Output JSON file.")
    p.add_argument("--weights", default=None,
                   help="Optional .pt checkpoint. If omitted, a randomly "
                        "initialized I3D is used (the I3D-Rand baseline).")
    p.add_argument("--num_classes", type=int, default=2731,
                   help="Logit dimension of the loaded checkpoint. Use 400 "
                        "for the Kinetics checkpoint, 2731 for ASL Citizen.")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    test_t = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = ASLCitizen(args.video_dir, test_t, args.metadata_csv)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    i3d = InceptionI3d(400, in_channels=3)
    if args.weights is not None:
        if args.num_classes != 400:
            i3d.replace_logits(args.num_classes)
        i3d.load_state_dict(torch.load(args.weights, map_location="cpu"))
    i3d.remove_last()
    i3d.cuda().eval()

    with open(args.output_json, "w") as fh, torch.no_grad():
        fh.write("[\n")
        n_written = 0
        for i, data in enumerate(tqdm(loader)):
            inputs, name, _ = data
            if inputs.sum() == 0:
                continue
            inputs = inputs.cuda()
            features = i3d.extract_features(inputs).cpu().numpy()
            features = np.average(np.squeeze(features), axis=1)
            name["features"] = features.tolist()
            sep = "" if n_written == 0 else ",\n"
            fh.write(sep + json.dumps(name))
            n_written += 1
        fh.write("\n]")

    print(f"Wrote {n_written} feature vectors to {args.output_json}")


if __name__ == "__main__":
    main()

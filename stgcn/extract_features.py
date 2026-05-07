"""Extract ST-GCN encoder features for every clip in a CSV split.

Mirrors ``i3d/extract_features.py``: emits one JSON of
``{user, filename, gloss, features}`` records, where ``features`` is the
output of the ST-GCN encoder (the input to the FC classifier head). This
is the input format consumed by the analysis scripts.

We need a reference training CSV in order to rebuild the same gloss -> index
mapping the model was trained with, so the FC head shape matches.
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import pose_transforms
from architecture.fc import FC
from architecture.network import Network
from architecture.st_gcn import STGCN
from dataset import ASLCitizen
from graph_args import GRAPH_ARGS, N_FEATURES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose_dir", required=True, help="Directory of pose .npy files to encode.")
    p.add_argument("--metadata_csv", required=True, help="Split CSV listing the poses to encode.")
    p.add_argument("--output_json", required=True)
    p.add_argument("--train_pose_dir", required=True,
                   help="Pose dir used during training (only the train CSV is consulted "
                        "to recover the gloss vocabulary).")
    p.add_argument("--train_csv", required=True,
                   help="CSV used at train time, to recover the gloss -> class mapping.")
    p.add_argument("--weights", default=None,
                   help="Optional .pt checkpoint. If omitted, a randomly initialized "
                        "ST-GCN is used (the STGCN-Rand baseline).")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    transform = pose_transforms.Compose([
        pose_transforms.ShearTransform(0.1),
        pose_transforms.RotatationTransform(0.1),
    ])
    train_ds = ASLCitizen(args.train_pose_dir, args.train_csv, transforms=transform)
    eval_ds = ASLCitizen(args.pose_dir, args.metadata_csv, gloss_dict=train_ds.gloss_dict)
    n_classes = len(train_ds.gloss_dict)

    encoder = STGCN(in_channels=2, graph_args=GRAPH_ARGS, edge_importance_weighting=True)
    decoder = FC(n_features=N_FEATURES, num_class=n_classes, dropout_ratio=0.05)
    model = Network(encoder=encoder, decoder=decoder)
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.cuda().train(False)

    loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    seen_features = None
    n_written = 0
    with open(args.output_json, "w") as fh, torch.no_grad():
        fh.write("[\n")
        for inputs, name, _ in tqdm(loader):
            inputs = inputs.float().cuda()
            features = model.encoder(inputs).cpu().numpy()
            # Catch a sentinel failure mode where the encoder returned a constant
            # (e.g. NaN inputs collapsing to zero).
            if seen_features is not None and (features == seen_features).all():
                print(f"Duplicate features for {name['filename']}, aborting.")
                break
            seen_features = features

            name["features"] = features.tolist()
            sep = "" if n_written == 0 else ",\n"
            fh.write(sep + json.dumps(name))
            n_written += 1
        fh.write("\n]")
    print(f"Wrote {n_written} feature vectors to {args.output_json}")


if __name__ == "__main__":
    main()

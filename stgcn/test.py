"""Evaluate a trained ST-GCN classifier on ASL Citizen."""

import argparse
import math
import os
from operator import add

import numpy as np
import torch
from tqdm import tqdm

import pose_transforms
from architecture.fc import FC
from architecture.network import Network
from architecture.st_gcn import STGCN
from dataset import ASLCitizen
from graph_args import GRAPH_ARGS, N_FEATURES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose_dir", required=True)
    p.add_argument("--train_csv", required=True,
                   help="Used only to recover the gloss -> index mapping.")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out_dir", default="outputs/stgcn_eval")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def rank_metrics(sorted_args, label):
    (res,) = np.where(sorted_args == label)
    rank = res[0]
    dcg = 1.0 / math.log2(rank + 2)
    mrr = 1.0 / (rank + 1)
    if rank < 1:
        return rank, [dcg, 1, 1, 1, 1, mrr]
    if rank < 5:
        return rank, [dcg, 0, 1, 1, 1, mrr]
    if rank < 10:
        return rank, [dcg, 0, 0, 1, 1, mrr]
    if rank < 20:
        return rank, [dcg, 0, 0, 0, 1, mrr]
    return rank, [dcg, 0, 0, 0, 0, mrr]


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    os.makedirs(args.out_dir, exist_ok=True)

    transform = pose_transforms.Compose([
        pose_transforms.ShearTransform(0.1),
        pose_transforms.RotatationTransform(0.1),
    ])
    train_ds = ASLCitizen(args.pose_dir, args.train_csv, transforms=transform)
    test_ds = ASLCitizen(args.pose_dir, args.test_csv, gloss_dict=train_ds.gloss_dict)
    n_classes = len(train_ds.gloss_dict)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    encoder = STGCN(in_channels=2, graph_args=GRAPH_ARGS, edge_importance_weighting=True)
    decoder = FC(n_features=N_FEATURES, num_class=n_classes, dropout_ratio=0.05)
    model = Network(encoder=encoder, decoder=decoder)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.cuda().train(False)

    totals = [0, 0, 0, 0, 0, 0]
    n_total = 0
    conf = np.zeros((n_classes, n_classes))

    for inputs, _, labels in tqdm(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        logits = model(inputs)
        ranked = torch.argsort(torch.softmax(logits, dim=1), dim=1, descending=True).cpu().numpy()
        truth = torch.argmax(labels, dim=1).cpu().numpy()
        for i in range(len(ranked)):
            _, counts = rank_metrics(ranked[i], truth[i])
            totals = list(map(add, counts, totals))
            n_total += 1
            conf[truth[i], ranked[i, 0]] += 1

    summary = {
        "n_eval": n_total,
        "DCG": totals[0] / n_total,
        "Top-1": totals[1] / n_total,
        "Top-5": totals[2] / n_total,
        "Top-10": totals[3] / n_total,
        "Top-20": totals[4] / n_total,
        "MRR": totals[5] / n_total,
    }
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")
    np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), conf)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

"""Evaluate a trained I3D classifier on the ASL Citizen test split.

Reports top-k accuracy, MRR, and DCG. Not used for the paper's main results
(which depend on penultimate features rather than predictions), but useful as
a sanity check that re-trained checkpoints still classify reasonably.
"""

import argparse
import math
import os
from operator import add

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import videotransforms
from dataset import ASLCitizen
from pytorch_i3d import InceptionI3d


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--train_csv", required=True,
                   help="Used only to recover the gloss -> index mapping.")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--weights", required=True, help=".pt checkpoint to evaluate.")
    p.add_argument("--out_dir", default="outputs/i3d_eval")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_classes", type=int, default=2731,
                   help="Logit dimension of the saved checkpoint.")
    return p.parse_args()


def rank_metrics(sorted_args, label):
    """Returns (rank, [DCG, top1, top5, top10, top20, MRR])."""
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
    os.makedirs(args.out_dir, exist_ok=True)

    train_t = transforms.Compose([videotransforms.RandomCrop(224),
                                  videotransforms.RandomHorizontalFlip()])
    test_t = transforms.Compose([videotransforms.CenterCrop(224)])
    train_ds = ASLCitizen(args.video_dir, train_t, args.train_csv)
    test_ds = ASLCitizen(args.video_dir, test_t, args.test_csv,
                         gloss_dict=train_ds.gloss_dict)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    idx2gloss = {i: g for g, i in train_ds.gloss_dict.items()}
    n_gloss = len(idx2gloss)

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(args.num_classes)
    i3d.load_state_dict(torch.load(args.weights, map_location="cpu"))
    i3d.cuda().train(False)

    totals = [0, 0, 0, 0, 0, 0]
    n_total = 0
    conf = np.zeros((n_gloss, n_gloss))

    for inputs, _, labels in tqdm(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        per_frame_logits = i3d(inputs, pretrained=False)
        per_frame_logits = F.upsample(per_frame_logits, inputs.size(2), mode="linear")
        preds = torch.softmax(torch.max(per_frame_logits, dim=2)[0], dim=1)
        ranked = torch.argsort(preds, dim=1, descending=True).cpu().numpy()
        truth = torch.argmax(torch.max(labels, dim=2)[0], dim=1).cpu().numpy()

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

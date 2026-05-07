"""Train an ST-GCN classifier on ASL Citizen pose graphs.

Re-trained model corresponds to ``STGCN-ASL`` in the paper. Pass
``--gloss_filter`` to exclude minimal-pair test signs from training.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import pose_transforms
from architecture.fc import FC
from architecture.network import Network
from architecture.st_gcn import STGCN
from dataset import ASLCitizen
from graph_args import GRAPH_ARGS, N_FEATURES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose_dir", required=True, help="Directory of MediaPipe .npy pose files.")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--gloss_filter", default=None,
                   help="Optional .txt file listing glosses to exclude (one per line).")
    p.add_argument("--save_dir", default="checkpoints/stgcn_asl")
    p.add_argument("--log_dir", default="logs/stgcn_asl")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_gloss_filter(path):
    if not path:
        return []
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64)


def main():
    args = parse_args()
    seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    transform = pose_transforms.Compose([
        pose_transforms.ShearTransform(0.1),
        pose_transforms.RotatationTransform(0.1),
    ])
    gloss_filter = load_gloss_filter(args.gloss_filter)
    train_ds = ASLCitizen(args.pose_dir, args.train_csv, transforms=transform,
                          gloss_filter=gloss_filter)
    val_ds = ASLCitizen(args.pose_dir, args.val_csv,
                        gloss_dict=train_ds.gloss_dict, gloss_filter=gloss_filter)
    n_classes = len(train_ds.gloss_dict)
    print(f"Training {n_classes} classes "
          f"({len(train_ds)} train / {len(val_ds)} val poses).")

    def seed_worker(_):
        np.random.seed(args.seed)
        random.seed(args.seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
    )

    encoder = STGCN(in_channels=2, graph_args=GRAPH_ARGS, edge_importance_weighting=True)
    decoder = FC(n_features=N_FEATURES, num_class=n_classes, dropout_ratio=0.05)
    model = Network(encoder=encoder, decoder=decoder).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, last_epoch=-1)
    ce_loss = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        log_path = os.path.join(args.log_dir, f"log{epoch}.txt")
        with open(log_path, "w") as log:
            log.write(f"Epoch {epoch}\n")
            for phase, loader in [("train", train_loader), ("val", val_loader)]:
                model.train(phase == "train")
                tot_loss = 0.0
                correct = total = 0
                for inputs, _, labels in tqdm(loader, desc=f"epoch {epoch} {phase}"):
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    if phase == "train":
                        optimizer.zero_grad()
                    logits = model(inputs)
                    loss = ce_loss(logits, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    tot_loss += loss.item()
                    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    truth = torch.argmax(labels, dim=1)
                    correct += (pred == truth).sum().item()
                    total += labels.shape[0]

                acc = correct / max(total, 1)
                avg_loss = tot_loss / max(len(loader), 1)
                log.write(f"{phase} loss={avg_loss:.4f} acc={acc:.4f}\n")
                print(f"epoch {epoch} {phase} loss={avg_loss:.4f} acc={acc:.4f}")

                if phase == "val":
                    scheduler.step()
                    if acc > best_val or epoch % 2 == 0:
                        ckpt = os.path.join(args.save_dir, f"epoch{epoch:03d}_acc{acc:.3f}.pt")
                        torch.save(model.state_dict(), ckpt)
                        log.write(f"saved {ckpt}\n")
                    if acc > best_val:
                        best_val = acc


if __name__ == "__main__":
    main()

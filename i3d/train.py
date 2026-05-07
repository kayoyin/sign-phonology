"""Train an I3D classifier on ASL Citizen.

Re-trained model corresponds to ``I3D-ASL`` in the paper. To reproduce the
"Strict exclusion of test signs" setup, pass ``--gloss_filter`` pointing at the
list of glosses that appear in the minimal-pair evaluation.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

import videotransforms
from dataset import ASLCitizen
from pytorch_i3d import InceptionI3d


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_dir", required=True, help="Directory of ASL Citizen videos.")
    p.add_argument("--train_csv", required=True, help="Training split CSV.")
    p.add_argument("--val_csv", required=True, help="Validation split CSV.")
    p.add_argument("--gloss_filter", default=None,
                   help="Optional .txt file listing glosses to exclude from training "
                        "(one per line). Used for the minimal-pair test-sign exclusion.")
    p.add_argument("--save_dir", default="checkpoints/i3d_asl",
                   help="Where to write .pt checkpoints.")
    p.add_argument("--log_dir", default="logs/i3d_asl",
                   help="Where to write per-epoch text logs.")
    p.add_argument("--init_weights", default=None,
                   help="Optional .pt file to warm-start from (e.g. Kinetics-pretrained).")
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--weight_decay", type=float, default=1e-8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default=None,
                   help="If set, log metrics to this W&B project. Requires WANDB_API_KEY in env.")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--run_name", default="i3d_asl")
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


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    wandb = None
    if args.wandb_project:
        import wandb as _wandb
        _wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name)
        wandb = _wandb

    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    gloss_filter = load_gloss_filter(args.gloss_filter)
    train_ds = ASLCitizen(args.video_dir, train_transforms, args.train_csv,
                          gloss_filter=gloss_filter)
    val_ds = ASLCitizen(args.video_dir, test_transforms, args.val_csv,
                        gloss_dict=train_ds.gloss_dict, gloss_filter=gloss_filter)
    n_classes = len(train_ds.gloss_dict)
    print(f"Training {n_classes} classes "
          f"({len(train_ds)} train / {len(val_ds)} val clips).")

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
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
    )

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(n_classes)
    if args.init_weights:
        i3d.load_state_dict(torch.load(args.init_weights, map_location="cpu"))
    i3d.to(device)

    optimizer = optim.Adam(i3d.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.3)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        log_path = os.path.join(args.log_dir, f"log{epoch}.txt")
        with open(log_path, "w") as log:
            log.write(f"Epoch {epoch}\n")

            for phase, loader in [("train", train_loader), ("val", val_loader)]:
                i3d.train(phase == "train")
                tot_loss = tot_loc = tot_cls = 0.0
                correct = total = 0
                for inputs, _, labels in tqdm(loader, desc=f"epoch {epoch} {phase}"):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if phase == "train":
                        optimizer.zero_grad()
                    per_frame_logits = i3d(inputs, pretrained=False)
                    per_frame_logits = F.upsample(per_frame_logits, inputs.size(2), mode="linear")
                    ground_truth = torch.max(labels, dim=2)[0]
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    cls_loss = F.binary_cross_entropy_with_logits(
                        torch.max(per_frame_logits, dim=2)[0], ground_truth,
                    )
                    loss = 0.5 * loc_loss + 0.5 * cls_loss
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    tot_loss += loss.item()
                    tot_loc += loc_loss.item()
                    tot_cls += cls_loss.item()
                    pred = torch.argmax(torch.max(per_frame_logits, dim=2)[0], dim=1)
                    correct += (pred == torch.argmax(ground_truth, dim=1)).sum().item()
                    total += ground_truth.shape[0]

                acc = correct / max(total, 1)
                avg_loss = tot_loss / max(len(loader), 1)
                log.write(f"{phase} loss={avg_loss:.4f} acc={acc:.4f}\n")
                if wandb is not None:
                    wandb.log({f"{phase}_loss": avg_loss, f"{phase}_acc": acc, "epoch": epoch})

                if phase == "val":
                    scheduler.step(avg_loss)
                    if acc > best_val or epoch % 2 == 0:
                        ckpt = os.path.join(args.save_dir, f"epoch{epoch:03d}_acc{acc:.3f}.pt")
                        torch.save(i3d.state_dict(), ckpt)
                        log.write(f"saved {ckpt}\n")
                    if acc > best_val:
                        best_val = acc


if __name__ == "__main__":
    main()

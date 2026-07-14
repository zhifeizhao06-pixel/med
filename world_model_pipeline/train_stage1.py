import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .common import device_from_arg, load_manifest, save_checkpoint, seed_everything
    from .data import make_stage1_dataset
    from .stage1_model import Stage1SegResNetVAE
except ImportError:
    from common import device_from_arg, load_manifest, save_checkpoint, seed_everything
    from data import make_stage1_dataset
    from stage1_model import Stage1SegResNetVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="results/world_model/stage1_best.pt")
    parser.add_argument("--roi_size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vae_weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference

    seed_everything(args.seed)
    device = device_from_arg(args.device)
    train_ds = make_stage1_dataset(load_manifest(args.manifest, "train"), args.roi_size, True)
    val_ds = make_stage1_dataset(load_manifest(args.manifest, "val"), args.roi_size, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = Stage1SegResNetVAE(args.roi_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    seg_loss = DiceCELoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    best = -1.0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, vae_loss = model(image)
            loss = seg_loss(logits, label) + args.vae_weight * vae_loss
            loss.backward()
            optimizer.step()
            running += float(loss.detach())

        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                image, label = batch["image"].to(device), batch["label"].to(device)
                # Validation volumes are full-size; sliding-window inference avoids OOM.
                logits = sliding_window_inference(
                    image,
                    roi_size=tuple(args.roi_size),
                    sw_batch_size=1,
                    predictor=lambda patch: model(patch)[0],
                    overlap=0.5,
                )
                dice_metric((torch.sigmoid(logits) > 0.5).float(), label)
        dice = float(dice_metric.aggregate())
        print(f"epoch={epoch + 1} train_loss={running / max(len(train_loader), 1):.5f} val_dice={dice:.5f}")
        if dice > best:
            best = dice
            save_checkpoint(args.output, model=model.state_dict(), roi_size=tuple(args.roi_size), epoch=epoch + 1, val_dice=dice)


if __name__ == "__main__":
    main()

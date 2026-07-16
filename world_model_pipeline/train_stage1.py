import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .common import (
        device_from_arg,
        last_checkpoint_path,
        load_manifest,
        restore_optimizer,
        save_checkpoint,
        seed_everything,
    )
    from .data import make_stage1_dataset
    from .stage1_model import Stage1SegResNetVAE
except ImportError:
    from common import (
        device_from_arg,
        last_checkpoint_path,
        load_manifest,
        restore_optimizer,
        save_checkpoint,
        seed_everything,
    )
    from data import make_stage1_dataset
    from stage1_model import Stage1SegResNetVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="results/world_model/stage1_best.pt")
    parser.add_argument("--last_output", default=None, help="rolling checkpoint path (default: *_last.pt)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from, normally *_last.pt")
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
    from monai.data import list_data_collate

    seed_everything(args.seed)
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.resume, map_location="cpu") if args.resume else None
    roi_size = tuple(checkpoint.get("roi_size", args.roi_size)) if checkpoint else tuple(args.roi_size)
    if checkpoint and roi_size != tuple(args.roi_size):
        print(f"resume: using checkpoint roi_size={roi_size} instead of command-line roi_size={tuple(args.roi_size)}")
    train_ds = make_stage1_dataset(load_manifest(args.manifest, "train"), roi_size, True)
    val_ds = make_stage1_dataset(load_manifest(args.manifest, "val"), roi_size, False)
    # RandCropByPosNegLabeld returns a list of sampled patches per case.
    # MONAI's collate flattens those lists into a regular dictionary batch.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = Stage1SegResNetVAE(roi_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    seg_loss = DiceCELoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    best = -1.0
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best = float(checkpoint.get("best_val_dice", checkpoint.get("val_dice", -1.0)))
        optimizer_restored = restore_optimizer(optimizer, checkpoint, device)
        mode = "model and optimizer" if optimizer_restored else "model only (fresh optimizer)"
        print(f"resumed {mode} from {args.resume}: completed_epoch={start_epoch} best_val_dice={best:.5f}")

    last_output = last_checkpoint_path(args.output, args.last_output)
    if Path(last_output).resolve() == Path(args.output).resolve():
        raise ValueError("--last_output must be different from --output")
    if start_epoch >= args.epochs:
        print(f"nothing to do: checkpoint epoch {start_epoch} already reached --epochs {args.epochs}")
        return

    for epoch in range(start_epoch, args.epochs):
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
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=lambda patch: model(patch)[0],
                    overlap=0.5,
                )
                dice_metric((torch.sigmoid(logits) > 0.5).float(), label)
        dice = float(dice_metric.aggregate())
        print(f"epoch={epoch + 1} train_loss={running / max(len(train_loader), 1):.5f} val_dice={dice:.5f}")
        if dice > best:
            best = dice
            save_checkpoint(
                args.output,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                roi_size=roi_size,
                epoch=epoch + 1,
                val_dice=dice,
                best_val_dice=best,
            )
        save_checkpoint(
            last_output,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            roi_size=roi_size,
            epoch=epoch + 1,
            val_dice=dice,
            best_val_dice=best,
        )


if __name__ == "__main__":
    main()

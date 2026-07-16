import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .common import device_from_arg, last_checkpoint_path, restore_optimizer, save_checkpoint, seed_everything
    from .data import CachedSliceDataset
    from .stage2_model import build_stage2
except ImportError:
    from common import device_from_arg, last_checkpoint_path, restore_optimizer, save_checkpoint, seed_everything
    from data import CachedSliceDataset
    from stage2_model import build_stage2


def dice_score(pred, target):
    pred = (pred > 0.5).float()
    return (2 * (pred * target).sum() + 1e-5) / (pred.sum() + target.sum() + 1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", default="results/world_model/stage2_best.pt")
    parser.add_argument("--last_output", default=None, help="rolling checkpoint path (default: *_last.pt)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from, normally *_last.pt")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.resume, map_location="cpu") if args.resume else None
    checkpoint_model_args = checkpoint.get("model_args") if checkpoint else None
    image_size = int(checkpoint_model_args.get("image_size", args.image_size)) if checkpoint_model_args else args.image_size
    if checkpoint_model_args and image_size != args.image_size:
        print(f"resume: using checkpoint image_size={image_size} instead of command-line image_size={args.image_size}")
    train_ds = CachedSliceDataset(args.cache_dir, "train", image_size, training=True)
    val_ds = CachedSliceDataset(args.cache_dir, "val", image_size, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if checkpoint_model_args:
        model, diffusion, model_args = build_stage2(**checkpoint_model_args)
    else:
        model, diffusion, model_args = build_stage2(
            image_size=args.image_size,
            num_channels=args.num_channels,
            num_res_blocks=args.num_res_blocks,
            diffusion_steps=args.diffusion_steps,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = float("inf")
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
        optimizer_restored = restore_optimizer(optimizer, checkpoint, device)
        mode = "model and optimizer" if optimizer_restored else "model only (fresh optimizer)"
        print(f"resumed {mode} from {args.resume}: completed_epoch={start_epoch} best_val_loss={best:.6f}")

    last_output = last_checkpoint_path(args.output, args.last_output)
    if Path(last_output).resolve() == Path(args.output).resolve():
        raise ValueError("--last_output must be different from --output")
    if start_epoch >= args.epochs:
        print(f"nothing to do: checkpoint epoch {start_epoch} already reached --epochs {args.epochs}")
        return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for condition, mask, _ in train_loader:
            condition, mask = condition.to(device), mask.to(device)
            full = torch.cat((condition, mask), dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (full.shape[0],), device=device)
            optimizer.zero_grad(set_to_none=True)
            terms, _ = diffusion.training_losses_segmentation(model, None, full, t)
            loss = (terms["loss"] + 10.0 * terms["loss_cal"]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())

        # A stable, cheap validation objective; full reverse diffusion is done by sample_stage2.py.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for condition, mask, _ in val_loader:
                condition, mask = condition.to(device), mask.to(device)
                full = torch.cat((condition, mask), dim=1)
                t = torch.randint(0, diffusion.num_timesteps, (full.shape[0],), device=device)
                terms, _ = diffusion.training_losses_segmentation(model, None, full, t)
                val_loss += float((terms["loss"] + 10.0 * terms["loss_cal"]).mean())
        train_loss = running / max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(
                args.output,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                model_args=model_args,
                epoch=epoch + 1,
                val_loss=val_loss,
                best_val_loss=best,
            )
        save_checkpoint(
            last_output,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            model_args=model_args,
            epoch=epoch + 1,
            val_loss=val_loss,
            best_val_loss=best,
        )


if __name__ == "__main__":
    main()

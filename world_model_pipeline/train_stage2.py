import argparse

import torch
from torch.utils.data import DataLoader

try:
    from .common import device_from_arg, save_checkpoint, seed_everything
    from .data import CachedSliceDataset
    from .stage2_model import build_stage2
except ImportError:
    from common import device_from_arg, save_checkpoint, seed_everything
    from data import CachedSliceDataset
    from stage2_model import build_stage2


def dice_score(pred, target):
    pred = (pred > 0.5).float()
    return (2 * (pred * target).sum() + 1e-5) / (pred.sum() + target.sum() + 1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", default="results/world_model/stage2_best.pt")
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
    train_ds = CachedSliceDataset(args.cache_dir, "train", args.image_size, training=True)
    val_ds = CachedSliceDataset(args.cache_dir, "val", args.image_size, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, diffusion, model_args = build_stage2(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        diffusion_steps=args.diffusion_steps,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best = float("inf")

    for epoch in range(args.epochs):
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
            save_checkpoint(args.output, model=model.state_dict(), model_args=model_args, epoch=epoch + 1, val_loss=val_loss)


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .common import device_from_arg, seed_everything
    from .data import CachedSliceDataset
    from .stage2_model import build_stage2
except ImportError:
    from common import device_from_arg, seed_everything
    from data import CachedSliceDataset
    from stage2_model import build_stage2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/world_model/predictions")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num_ensemble", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = device_from_arg(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["model_args"]
    model, diffusion, _ = build_stage2(**model_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    dataset = CachedSliceDataset(args.cache_dir, args.split, model_args["image_size"], training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    output = Path(args.output_dir) / args.split
    output.mkdir(parents=True, exist_ok=True)
    steps = args.sampling_steps or model_args["diffusion_steps"]
    if not 1 <= steps <= diffusion.num_timesteps:
        raise ValueError(f"sampling_steps must be in [1, {diffusion.num_timesteps}], got {steps}")

    with torch.no_grad():
        for condition, mask, name in loader:
            condition = condition.to(device)
            predictions = []
            for _ in range(args.num_ensemble):
                noisy_mask = torch.randn_like(condition[:, :1])
                known = torch.cat((condition, noisy_mask), dim=1)
                sample, _, _, _, _ = diffusion.p_sample_loop_known(
                    model, (1, 1, model_args["image_size"], model_args["image_size"]),
                    known, step=steps, clip_denoised=True, model_kwargs={},
                )
                predictions.append(((sample[:, -1:] + 1.0) * 0.5).clamp(0, 1))
            probability = torch.stack(predictions).mean(0)[0, 0].cpu().numpy()
            np.save(output / f"{name[0]}.npy", probability.astype(np.float32))
            print(f"saved {name[0]}")


if __name__ == "__main__":
    main()

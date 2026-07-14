import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .common import device_from_arg, load_manifest
    from .data import make_stage1_dataset
    from .stage1_model import Stage1SegResNetVAE
except ImportError:
    from common import device_from_arg, load_manifest
    from data import make_stage1_dataset
    from stage1_model import Stage1SegResNetVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/world_model/conditions")
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from monai.inferers import sliding_window_inference

    device = device_from_arg(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    roi_size = tuple(checkpoint.get("roi_size", (96, 96, 96)))
    model = Stage1SegResNetVAE(roi_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for split in args.splits:
        records = load_manifest(args.manifest, split)
        dataset = make_stage1_dataset(records, roi_size, False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        split_dir = Path(args.output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for record, batch in zip(records, loader):
                output_path = split_dir / f"{record['id']}.npz"
                if output_path.exists():
                    print(f"skipped existing {split}/{record['id']}")
                    continue
                image = batch["image"].to(device)
                label = batch["label"].to(device)

                # Some BraTS NIfTI files contain an extra singleton dimension.
                # Export always uses batch_size=1, so canonicalize explicitly to
                # [B, C, H, W, D] before MONAI sliding-window inference.
                image = image.squeeze()
                if image.ndim == 4:
                    image = image.unsqueeze(0)
                label = label.squeeze()
                while label.ndim < 5:
                    label = label.unsqueeze(0)
                if image.ndim != 5 or image.shape[1] != 4:
                    raise RuntimeError(
                        f"{record['id']}: expected image [1,4,H,W,D], got {tuple(image.shape)}"
                    )
                if label.ndim != 5 or label.shape[1] != 1:
                    raise RuntimeError(
                        f"{record['id']}: expected label [1,1,H,W,D], got {tuple(label.shape)}"
                    )

                def predictor(patch):
                    return model(patch)[0]

                logits = sliding_window_inference(image, roi_size, 1, predictor, overlap=args.overlap)
                coarse = torch.sigmoid(logits)
                # Bernoulli predictive entropy proxy, normalized to [0, 1].
                eps = 1e-6
                uncertainty = -(coarse * torch.log(coarse + eps) + (1 - coarse) * torch.log(1 - coarse + eps)) / np.log(2.0)
                np.savez_compressed(
                    output_path,
                    image=image[0].cpu().numpy().astype(np.float16),
                    mask=(label[0].cpu().numpy() > 0.5).astype(np.uint8),
                    coarse=coarse[0].cpu().numpy().astype(np.float16),
                    uncertainty=uncertainty[0].cpu().numpy().astype(np.float16),
                )
                print(f"exported {split}/{record['id']}")


if __name__ == "__main__":
    main()

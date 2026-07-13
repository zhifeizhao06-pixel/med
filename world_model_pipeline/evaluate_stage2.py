import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    case_scores = []
    for cache_path in sorted((Path(args.cache_dir) / args.split).glob("*.npz")):
        with np.load(cache_path) as data:
            mask = torch.from_numpy(data["mask"]).float()
        intersection = pred_sum = target_sum = 0.0
        found = 0
        for z in range(mask.shape[-1]):
            pred_path = Path(args.prediction_dir) / args.split / f"{cache_path.stem}_z{z:03d}.npy"
            if not pred_path.exists():
                continue
            pred = torch.from_numpy(np.load(pred_path)).float()[None, None]
            target = mask[..., z][None]
            target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
            pred = pred > args.threshold
            target = target > 0.5
            intersection += float((pred & target).sum())
            pred_sum += float(pred.sum())
            target_sum += float(target.sum())
            found += 1
        if not found:
            continue
        dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        iou = (intersection + 1e-6) / (pred_sum + target_sum - intersection + 1e-6)
        case_scores.append((dice, iou))
        print(f"{cache_path.stem}: dice={dice:.5f} iou={iou:.5f}")
    if not case_scores:
        raise RuntimeError("No matching predictions found")
    values = np.asarray(case_scores)
    print(f"mean_dice={values[:, 0].mean():.5f} mean_iou={values[:, 1].mean():.5f} cases={len(values)}")


if __name__ == "__main__":
    main()


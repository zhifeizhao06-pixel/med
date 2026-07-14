from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def make_stage1_dataset(records, roi_size, training: bool):
    try:
        from monai.data import CacheDataset
        from monai.transforms import (
            Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, Lambdad,
            LoadImaged, NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld,
            RandFlipd, RandScaleIntensityd, RandShiftIntensityd, Spacingd, SpatialPadd,
        )
    except ImportError as exc:
        raise ImportError("Stage 1 requires MONAI. Install world_model_pipeline/requirements.txt") from exc

    transforms = [
        LoadImaged(keys=("image", "label")),
        EnsureChannelFirstd(keys=("image", "label")),
        Orientationd(keys=("image", "label"), axcodes="RAS"),
        Spacingd(keys=("image", "label"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Lambdad(keys="label", func=lambda x: (x > 0).astype(np.float32) if isinstance(x, np.ndarray) else (x > 0).float()),
        CropForegroundd(keys=("image", "label"), source_key="image"),
    ]
    if training:
        transforms += [
            # CropForegroundd may make one dimension smaller than the requested
            # VAE input ROI. Pad image and label together before random crops.
            SpatialPadd(keys=("image", "label"), spatial_size=tuple(roi_size), mode="constant"),
            RandCropByPosNegLabeld(
                keys=("image", "label"), label_key="label", spatial_size=tuple(roi_size),
                pos=1, neg=1, num_samples=2, image_key="image", image_threshold=0,
            ),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]
    transforms.append(EnsureTyped(keys=("image", "label")))
    return CacheDataset(records, transform=Compose(transforms), cache_rate=0.1, num_workers=2)


class CachedSliceDataset(Dataset):
    """Axial slices exported by Stage 1 for Stage 2 training/inference."""

    def __init__(self, cache_dir: str, split: str, image_size: int, training: bool = False):
        self.image_size = image_size
        self.training = training
        self.cases = sorted((Path(cache_dir) / split).glob("*.npz"))
        if not self.cases:
            raise RuntimeError(f"No .npz conditions found in {Path(cache_dir) / split}")
        self.index = []
        for case_idx, path in enumerate(self.cases):
            with np.load(path) as data:
                image = data["image"]
                mask = data["mask"]
                brain = np.any(np.abs(image) > 1e-6, axis=(0, 1, 2))
                tumor = np.any(mask > 0, axis=(0, 1, 2))
                valid = np.where(brain | tumor)[0]
            self.index.extend((case_idx, int(z)) for z in valid)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        case_idx, z = self.index[idx]
        path = self.cases[case_idx]
        with np.load(path) as data:
            image = torch.from_numpy(data["image"][..., z]).float()
            mask = torch.from_numpy(data["mask"][..., z]).float()
            coarse = torch.from_numpy(data["coarse"][..., z]).float()
            uncertainty = torch.from_numpy(data["uncertainty"][..., z]).float()

        tensors = [image, mask, coarse, uncertainty]
        tensors = [F.interpolate(t.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0) for t in tensors]
        image, mask, coarse, uncertainty = tensors
        mask = (mask > 0.5).float()
        if self.training and torch.rand(()) < 0.3:
            # Condition corruption prevents Stage 2 from simply copying Stage 1.
            k = 3 if torch.rand(()) < 0.5 else 5
            if torch.rand(()) < 0.5:
                coarse = F.max_pool2d(coarse.unsqueeze(0), k, stride=1, padding=k // 2).squeeze(0)
            else:
                coarse = -F.max_pool2d(-coarse.unsqueeze(0), k, stride=1, padding=k // 2).squeeze(0)
        condition = torch.cat((image, coarse.clamp(0, 1), uncertainty.clamp(0, 1)), dim=0)
        return condition, mask, f"{path.stem}_z{z:03d}"

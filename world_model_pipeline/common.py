import json
import random
from pathlib import Path

import numpy as np
import torch


MODALITIES = ("t1", "t1ce", "t2", "flair")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_manifest(path: str, split: str):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if split not in manifest:
        raise KeyError(f"split {split!r} not found in {path}")
    return manifest[split]


def save_checkpoint(path: str, **payload) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


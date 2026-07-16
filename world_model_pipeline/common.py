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
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def last_checkpoint_path(best_path: str, override: str | None = None) -> str:
    """Return the rolling checkpoint path associated with a best checkpoint."""
    if override:
        return override
    best = Path(best_path)
    stem = best.stem[:-5] if best.stem.endswith("_best") else best.stem
    return str(best.with_name(f"{stem}_last{best.suffix}"))


def restore_optimizer(optimizer, checkpoint, device: torch.device) -> bool:
    """Restore optimizer state when present; old model-only checkpoints stay valid."""
    state_dict = checkpoint.get("optimizer")
    if state_dict is None:
        return False
    optimizer.load_state_dict(state_dict)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    return True


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)

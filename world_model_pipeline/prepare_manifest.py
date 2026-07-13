"""Create patient-level BraTS train/val/test splits."""

import argparse
import json
import random
from pathlib import Path

try:
    from .common import MODALITIES
except ImportError:
    from common import MODALITIES


def find_case(case_dir: Path):
    files = {}
    nii_files = list(case_dir.glob("*.nii.gz")) + list(case_dir.glob("*.nii"))
    for modality in MODALITIES:
        matches = [p for p in nii_files if p.name.lower().replace(".nii.gz", "").replace(".nii", "").endswith("_" + modality)]
        if len(matches) != 1:
            return None
        files[modality] = str(matches[0].resolve())
    labels = [p for p in nii_files if p.name.lower().replace(".nii.gz", "").replace(".nii", "").endswith("_seg")]
    if len(labels) != 1:
        return None
    return {
        "id": case_dir.name,
        "image": [files[m] for m in MODALITIES],
        "label": str(labels[0].resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="world_model_pipeline/brats_manifest.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = [find_case(p) for p in sorted(Path(args.data_dir).iterdir()) if p.is_dir()]
    cases = [c for c in cases if c is not None]
    if not cases:
        raise RuntimeError("No complete BraTS cases found. Expected *_t1, *_t1ce, *_t2, *_flair and *_seg NIfTI files.")

    random.Random(args.seed).shuffle(cases)
    n_test = round(len(cases) * args.test_ratio)
    n_val = round(len(cases) * args.val_ratio)
    manifest = {
        "test": cases[:n_test],
        "val": cases[n_test:n_test + n_val],
        "train": cases[n_test + n_val:],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({k: len(v) for k, v in manifest.items()})
    print(f"saved: {output}")


if __name__ == "__main__":
    main()


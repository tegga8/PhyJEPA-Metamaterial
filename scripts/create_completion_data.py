"""Create deterministic Phase 3 mask manifests from complete geometries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.completion_dataset import CompletionDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-type", choices=("central_block", "random_holes"), required=True)
    parser.add_argument("--missing-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "subset_root": str(args.subset_root), "mask_type": args.mask_type, "missing_ratio": args.missing_ratio,
        "base_seed": args.seed, "height": 16, "width": 16,
    }
    for split in ("train", "val", "test"):
        dataset = CompletionDataset(args.subset_root, split, args.mask_type, args.missing_ratio, args.seed)
        path = args.output_dir / f"mask_manifest_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "index", "mask_seed", "masked_pixels", "mask_ratio"))
            writer.writeheader()
            for index in range(len(dataset)):
                sample = dataset[index]
                mask = sample["mask"]
                writer.writerow({
                    "sample_id": sample["sample_id"], "index": index, "mask_seed": dataset.mask_seed(index),
                    "masked_pixels": int(mask.sum().item()), "mask_ratio": float(mask.mean().item()),
                })
    summary["mask_pixels"] = int(256 * args.missing_ratio + 0.5)
    (args.output_dir / "mask_manifest_config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build the fixed 5,000-sample Phase 1 subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import build_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = build_subset(args.raw_root, args.output_root, args.size, args.seed)
    print(f"Created {args.output_root} with split sizes: {counts}")


if __name__ == "__main__":
    main()

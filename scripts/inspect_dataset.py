"""Inspect the supplied SUTD PRT shards without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import FAMILIES, discover_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("."))
    parser.add_argument("--check-duplicates", action="store_true")
    args = parser.parse_args()

    shards = discover_shards(args.raw_root)
    by_family: dict[str, int] = defaultdict(int)
    geometry_min, geometry_max = np.inf, -np.inf
    response_min, response_max = np.inf, -np.inf
    invalid_geometry = invalid_response = 0
    signatures: Counter[bytes] = Counter()

    for shard in shards:
        image = np.load(shard.image_path, mmap_mode="r")
        curve = np.load(shard.curve_path, mmap_mode="r")
        by_family[shard.family] += shard.length
        geometry_min = min(geometry_min, image.min())
        geometry_max = max(geometry_max, image.max())
        invalid_geometry += int(np.count_nonzero((image != 0) & (image != 1)))
        for start in range(0, shard.length, 128):
            block = np.asarray(curve[start : start + 128])
            invalid_response += int(block.size - np.count_nonzero(np.isfinite(block)))
            response_min = min(response_min, float(np.abs(block).min()))
            response_max = max(response_max, float(np.abs(block).max()))
        if args.check_duplicates:
            for pattern in image:
                signatures[hashlib.blake2b(pattern.tobytes(), digest_size=16).digest()] += 1

    print("SUTD PRT raw-data inspection")
    print("families:", dict(sorted(by_family.items())))
    print("number of samples:", sum(by_family.values()))
    print("number of paired shards:", len(shards))
    print("geometry shape per sample: [16, 16] (stored flattened as [256])")
    print("geometry dtype(s):", sorted({str(np.load(item.image_path, mmap_mode="r").dtype) for item in shards}))
    print(f"geometry range: [{geometry_min}, {geometry_max}]")
    print("invalid geometry values:", invalid_geometry)
    print("raw response shape per sample: [2, 1001] complex (T=y-reflection, R=x-reflection)")
    print("raw response dtype(s):", sorted({str(np.load(item.curve_path, mmap_mode="r").dtype) for item in shards}))
    print(f"response magnitude range: [{response_min:.8g}, {response_max:.8g}]")
    print("non-finite response values:", invalid_response)
    print("frequency axis: 2.00–12.00 GHz, 1001 points, 0.01 GHz spacing")
    if args.check_duplicates:
        duplicates = sum(count - 1 for count in signatures.values() if count > 1)
        print("duplicate geometries:", duplicates)
        print("unique geometries:", len(signatures))
    else:
        print("duplicate geometries: not scanned; rerun with --check-duplicates")


if __name__ == "__main__":
    main()

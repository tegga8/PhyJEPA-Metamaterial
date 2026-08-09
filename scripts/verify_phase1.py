"""Run all Phase 1 acceptance checks against a processed SUTD-PRCM subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset, build_dataloaders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = args.subset_root
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected_size = int(metadata["subset_size"])
    expected_splits = {name: int(value) for name, value in metadata["split_counts"].items()}

    geometries = np.load(root / "geometries.npy", mmap_mode="r")
    responses = np.load(root / "responses.npy", mmap_mode="r")
    frequency_ghz = np.load(root / "frequency_ghz.npy")
    source_ids = (root / "source_ids.txt").read_text(encoding="utf-8").splitlines()
    splits = {
        name: (root / "splits" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        for name in ("train", "val", "test")
    }

    assert geometries.shape == (expected_size, 1, 16, 16), geometries.shape
    assert responses.shape == (expected_size, 4, 1001), responses.shape
    assert len(source_ids) == expected_size
    assert np.isin(geometries, (0, 1)).all()
    assert np.isfinite(responses).all()
    assert np.allclose(frequency_ghz, np.linspace(2.0, 12.0, 1001, dtype=np.float32))
    assert {name: len(ids) for name, ids in splits.items()} == expected_splits
    assert all(len(ids) == len(set(ids)) for ids in splits.values())
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    assert set().union(*map(set, splits.values())) == set(source_ids)

    train = SUTDPRCMDataset(root, "train")
    first_geometry, first_response = train[0]
    second_geometry, second_response = train[0]
    assert first_geometry.shape == (1, 16, 16)
    assert first_response.shape == (4, 1001)
    assert first_geometry.equal(second_geometry) and first_response.equal(second_response)
    for name, loader in build_dataloaders(root, batch_size=args.batch_size).items():
        geometry_batch, response_batch = next(iter(loader))
        assert geometry_batch.shape[1:] == (1, 16, 16), (name, geometry_batch.shape)
        assert response_batch.shape[1:] == (4, 1001), (name, response_batch.shape)

    print("Phase 1 verification passed")
    print(f"- {expected_size:,} samples; {expected_splits['train']:,} / {expected_splits['val']:,} / {expected_splits['test']:,} train/val/test")
    print("- binary [1, 16, 16] geometries; finite [4, 1001] responses")
    print("- 2.00–12.00 GHz frequency vector; disjoint, deterministic splits")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import gc

import numpy as np
from src.dataset import SUTDPRCMDataset, build_dataloaders


def _write_subset(root):
    root.mkdir()
    np.save(root / "geometries.npy", np.zeros((10, 1, 16, 16), dtype=np.uint8))
    np.save(root / "responses.npy", np.ones((10, 4, 1001), dtype=np.float32))
    np.save(root / "frequency_ghz.npy", np.linspace(2.0, 12.0, 1001, dtype=np.float32))
    (root / "source_ids.txt").write_text("\n".join(f"RDN/Data_001/{i:06d}" for i in range(10)) + "\n")
    splits = root / "splits"
    splits.mkdir()
    for name, indexes in {"train": range(8), "val": range(8, 9), "test": range(9, 10)}.items():
        (splits / f"{name}.txt").write_text("\n".join(f"RDN/Data_001/{i:06d}" for i in indexes) + "\n")
    np.savez_compressed(root / "train_response_stats.npz", mean=np.zeros((4, 1), dtype=np.float32), std=np.ones((4, 1), dtype=np.float32))
    (root / "metadata.json").write_text(json.dumps({"subset_size": 10}))


def test_dataset_and_dataloader_shapes(tmp_path):
    root = tmp_path / "subset"
    _write_subset(root)
    dataset = SUTDPRCMDataset(root, "train")
    geometry, response = dataset[0]
    assert len(dataset) == 8
    assert geometry.shape == (1, 16, 16)
    assert response.shape == (4, 1001)
    assert dataset.frequency_ghz[[0, -1]].tolist() == [2.0, 12.0]
    assert set(geometry.unique().tolist()) == {0.0}
    loaders = build_dataloaders(root, batch_size=4)
    batch = next(iter(loaders["train"]))
    assert batch[0].shape == (4, 1, 16, 16)
    assert batch[1].shape == (4, 4, 1001)
    del batch, loaders, dataset, geometry, response
    gc.collect()

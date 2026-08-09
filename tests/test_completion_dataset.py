from __future__ import annotations

import json

import numpy as np
import torch

from src.completion_dataset import CompletionDataset, create_central_mask, create_random_mask
from src.completion_model import compose_binary_completion
from src.dataset import SUTDPRCMDataset


def _write_subset(root):
    root.mkdir()
    geometries = np.zeros((6, 1, 16, 16), dtype=np.uint8)
    geometries[0, 0, 2:6, 2:6] = 1
    geometries[1, 0, 8:12, 8:12] = 1
    np.save(root / "geometries.npy", geometries)
    np.save(root / "responses.npy", np.ones((6, 4, 1001), dtype=np.float32))
    np.save(root / "frequency_ghz.npy", np.linspace(2.0, 12.0, 1001, dtype=np.float32))
    ids = [f"RDN/Data_001/{i:06d}" for i in range(6)]
    (root / "source_ids.txt").write_text("\n".join(ids) + "\n")
    splits = root / "splits"
    splits.mkdir()
    for name, indexes in {"train": range(4), "val": range(4, 5), "test": range(5, 6)}.items():
        (splits / f"{name}.txt").write_text("\n".join(ids[i] for i in indexes) + "\n")
    np.savez_compressed(root / "train_response_stats.npz", mean=np.zeros((4, 1), dtype=np.float32), std=np.ones((4, 1), dtype=np.float32))
    (root / "metadata.json").write_text(json.dumps({"subset_size": 6}))


def test_central_mask_ratio_and_shape():
    mask25 = create_central_mask(16, 16, 0.25)
    mask50 = create_central_mask(16, 16, 0.50)
    assert mask25.shape == (1, 16, 16)
    assert int(mask25.sum()) == 64
    assert int(mask50.sum()) == 128
    assert set(np.unique(mask25)) <= {0, 1}


def test_random_mask_ratio_and_reproducibility():
    first = create_random_mask(seed=42)
    second = create_random_mask(seed=42)
    other = create_random_mask(seed=43)
    assert int(first.sum()) == 64
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_completion_dataset_shapes_partial_and_target_unchanged(tmp_path):
    root = tmp_path / "subset"
    _write_subset(root)
    source = SUTDPRCMDataset(root, "train", normalize_response=False)
    dataset = CompletionDataset(root, "train", "random_holes", 0.25, 42)
    sample = dataset[0]
    assert sample["input"].shape == (2, 16, 16)
    assert sample["target"].shape == (1, 16, 16)
    assert sample["mask"].shape == (1, 16, 16)
    assert set(sample["target"].unique().tolist()) <= {0.0, 1.0}
    assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(sample["input"][0:1] * (1 - sample["mask"]), sample["target"] * (1 - sample["mask"]))
    assert torch.equal(sample["input"][0:1] * sample["mask"], torch.zeros_like(sample["target"]))
    assert torch.equal(sample["target"], source[0][0])
    assert torch.equal(dataset[0]["mask"], dataset[0]["mask"])


def test_known_pixels_are_preserved_by_final_compositing():
    inputs = torch.zeros(1, 2, 16, 16)
    inputs[:, 0, 0, 1] = 1.0
    inputs[:, 1, 1, 0] = 1.0
    inputs[:, 1, 1, 1] = 1.0
    mask = inputs[:, 1:2]
    probabilities = torch.zeros(1, 1, 16, 16)
    probabilities[:, :, 1, 0] = 1.0
    completed = compose_binary_completion(probabilities, inputs, mask)
    assert completed[:, :, 0, 0].item() == 0.0
    assert completed[:, :, 0, 1].item() == 1.0
    assert completed[:, :, 1, 1].item() == 0.0

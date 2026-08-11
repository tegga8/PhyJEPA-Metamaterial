"""Paired geometry/EM samples for Phase 5A physics-conditioned completion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.completion_dataset import CompletionDataset, MaskType


class PhysicsCompletionDataset(CompletionDataset):
    """Return a partial geometry and its originally paired normalized EM curve.

    ``response`` is the Phase-2-normalized ``[4, 1001]`` target response.
    ``response_raw`` remains available only for evaluation/visualization.  Both
    use the exact same processed-array position as ``target`` and ``sample_id``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mask_type: MaskType = "central_block",
        missing_ratio: float = 0.25,
        base_seed: int = 42,
    ) -> None:
        super().__init__(root, split, mask_type, missing_ratio, base_seed)
        stats = np.load(self.root / "train_response_stats.npz")
        self.response_mean = stats["mean"].astype(np.float32)
        self.response_std = stats["std"].astype(np.float32)
        if self.response_mean.shape != (4, 1) or self.response_std.shape != (4, 1):
            raise ValueError("Expected Phase 2 response statistics with shape [4, 1]")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = super().__getitem__(index)
        position = int(self.source_dataset.indices[index])
        response_raw = np.asarray(self.source_dataset.responses[position], dtype=np.float32).copy()
        response = (response_raw - self.response_mean) / self.response_std
        sample["response"] = torch.from_numpy(response.astype(np.float32, copy=False))
        sample["response_raw"] = torch.from_numpy(response_raw)
        return sample


def build_physics_completion_dataloaders(
    root: str | Path,
    mask_type: MaskType,
    missing_ratio: float,
    base_seed: int = 42,
    batch_size: int = 64,
) -> dict[str, DataLoader]:
    """Build Phase 4.2-compatible loaders while retaining response pairing."""
    datasets = {
        split: PhysicsCompletionDataset(root, split, mask_type, missing_ratio, base_seed)
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(base_seed)
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }

"""Synthetic partial-structure data for Phase 3 completion experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import SUTDPRCMDataset


MaskType = Literal["central_block", "random_holes"]
SPLIT_SEED_OFFSETS = {"train": 0, "val": 100_003, "test": 200_003, "all": 300_007}


def _validate_mask_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive")


def _centered_rectangle(height: int, width: int, missing_ratio: float) -> tuple[int, int]:
    target_pixels = int(round(height * width * missing_ratio))
    candidates = [(abs(rows * cols - target_pixels), abs(rows - cols), rows, cols) for rows in range(1, height + 1) for cols in range(1, width + 1)]
    _, _, rows, cols = min(candidates)
    return rows, cols


def create_central_mask(height: int = 16, width: int = 16, missing_ratio: float = 0.25, seed: int | None = None) -> np.ndarray:
    """Create a deterministic centered rectangular mask with a near-exact ratio."""
    del seed
    _validate_mask_shape(height, width)
    if not 0 < missing_ratio <= 1:
        raise ValueError("missing_ratio must be in (0, 1]")
    rows, cols = _centered_rectangle(height, width, missing_ratio)
    row_start = (height - rows) // 2
    col_start = (width - cols) // 2
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[row_start:row_start + rows, col_start:col_start + cols] = 1
    return mask[None]


def create_random_mask(height: int = 16, width: int = 16, missing_ratio: float = 0.25, seed: int | None = None) -> np.ndarray:
    """Create an exact-count reproducible random-hole mask."""
    _validate_mask_shape(height, width)
    if not 0 < missing_ratio <= 1:
        raise ValueError("missing_ratio must be in (0, 1]")
    if seed is None:
        seed = 0
    count = int(round(height * width * missing_ratio))
    generator = np.random.default_rng(seed)
    selected = generator.choice(height * width, size=count, replace=False)
    mask = np.zeros(height * width, dtype=np.uint8)
    mask[selected] = 1
    return mask.reshape(1, height, width)


def create_mask(mask_type: MaskType, height: int = 16, width: int = 16, missing_ratio: float = 0.25, seed: int | None = None) -> np.ndarray:
    if mask_type == "central_block":
        return create_central_mask(height, width, missing_ratio, seed)
    if mask_type == "random_holes":
        return create_random_mask(height, width, missing_ratio, seed)
    raise ValueError(f"Unknown mask type: {mask_type}")


class CompletionDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Turn complete processed geometries into deterministic partial examples."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mask_type: MaskType = "central_block",
        missing_ratio: float = 0.25,
        base_seed: int = 42,
    ) -> None:
        if split not in SPLIT_SEED_OFFSETS:
            raise ValueError(f"Unknown split: {split}")
        if mask_type not in {"central_block", "random_holes"}:
            raise ValueError(f"Unknown mask type: {mask_type}")
        self.root = Path(root)
        self.split = split
        self.mask_type = mask_type
        self.missing_ratio = float(missing_ratio)
        self.base_seed = int(base_seed)
        self.split_seed = self.base_seed + SPLIT_SEED_OFFSETS[split]
        self.source_dataset = SUTDPRCMDataset(self.root, split, normalize_response=False)

    def __len__(self) -> int:
        return len(self.source_dataset)

    def mask_seed(self, index: int) -> int:
        return self.split_seed + int(index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        position = int(self.source_dataset.indices[index])
        target = torch.from_numpy(np.asarray(self.source_dataset.geometries[position], dtype=np.float32).copy())
        seed = self.mask_seed(index) if self.mask_type == "random_holes" else self.split_seed
        mask = torch.from_numpy(create_mask(self.mask_type, 16, 16, self.missing_ratio, seed).astype(np.float32))
        partial = target * (1.0 - mask)
        inputs = torch.cat((partial, mask), dim=0)
        return {"input": inputs, "target": target, "mask": mask, "sample_id": self.source_dataset.source_id(index)}


def build_completion_dataloaders(
    root: str | Path,
    mask_type: MaskType,
    missing_ratio: float,
    base_seed: int = 42,
    batch_size: int = 64,
) -> dict[str, DataLoader]:
    datasets = {split: CompletionDataset(root, split, mask_type, missing_ratio, base_seed) for split in ("train", "val", "test")}
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

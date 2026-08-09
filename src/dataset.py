"""PyTorch access to the reproducible SUTD PRT processed subset.

The original release stores each electromagnetic response as complex values with
shape ``[T, R, frequency]``.  In this purely reflective dataset, ``T`` is the
y-polarized reflection and ``R`` is the x-polarized reflection. This module
exposes them as four real-valued channels in that fixed order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


Split = Literal["train", "val", "test", "all"]


class SUTDPRCMDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load a processed Phase 1 subset without touching the raw files.

    Args:
        root: Processed subset directory, e.g. ``data/processed/sutd_prcm_5k``.
        split: One of ``train``, ``val``, ``test`` or ``all``.
        normalize_response: Apply mean/std calculated from train samples only.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        normalize_response: bool = True,
    ) -> None:
        self.root = Path(root)
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unknown split: {split}")
        self.split = split

        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Processed subset metadata not found at {metadata_path}. "
                "Run scripts/build_subset.py first."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.geometries = np.load(self.root / "geometries.npy", mmap_mode="r")
        self.responses = np.load(self.root / "responses.npy", mmap_mode="r")
        self.frequency_ghz = np.load(self.root / "frequency_ghz.npy")
        self.source_ids = (self.root / "source_ids.txt").read_text(encoding="utf-8").splitlines()

        if split == "all":
            self.indices = np.arange(len(self.source_ids), dtype=np.int64)
        else:
            ids = (self.root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            positions = {source_id: i for i, source_id in enumerate(self.source_ids)}
            try:
                self.indices = np.asarray([positions[source_id] for source_id in ids], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"Split references unknown source ID: {exc.args[0]}") from exc

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if normalize_response:
            stats = np.load(self.root / "train_response_stats.npz")
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

        self._validate_layout()

    def _validate_layout(self) -> None:
        if self.geometries.ndim != 4 or self.geometries.shape[1:] != (1, 16, 16):
            raise ValueError(f"Expected geometries [N, 1, 16, 16], got {self.geometries.shape}")
        if self.responses.ndim != 3 or self.responses.shape[1:] != (4, 1001):
            raise ValueError(f"Expected responses [N, 4, 1001], got {self.responses.shape}")
        if self.frequency_ghz.shape != (1001,):
            raise ValueError(f"Expected 1001 frequency points, got {self.frequency_ghz.shape}")
        if not (len(self.geometries) == len(self.responses) == len(self.source_ids)):
            raise ValueError("Processed arrays and source ID manifest have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = int(self.indices[index])
        geometry = torch.from_numpy(np.asarray(self.geometries[position], dtype=np.float32).copy())
        response = np.asarray(self.responses[position], dtype=np.float32).copy()
        if self.mean is not None and self.std is not None:
            response = (response - self.mean) / self.std
        return geometry, torch.from_numpy(response)

    def source_id(self, index: int) -> str:
        """Return the immutable raw-data identifier for a dataset index."""
        return self.source_ids[int(self.indices[index])]


def build_dataloaders(
    root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create deterministic split DataLoaders for the processed subset."""
    datasets = {
        "train": SUTDPRCMDataset(root, "train", normalize_response=True),
        "val": SUTDPRCMDataset(root, "val", normalize_response=True),
        "test": SUTDPRCMDataset(root, "test", normalize_response=True),
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for name, dataset in datasets.items()
    }

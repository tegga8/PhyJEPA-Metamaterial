"""Create Phase 1 geometry and EM-response inspection figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    dataset = SUTDPRCMDataset(args.subset_root, split="all", normalize_response=False)
    chosen = np.random.default_rng(args.seed).choice(len(dataset), size=min(100, len(dataset)), replace=False)

    figure, axes = plt.subplots(10, 10, figsize=(12, 12))
    for axis, index in zip(axes.flat, chosen):
        geometry, _ = dataset[int(index)]
        axis.imshow(geometry[0], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axis.set_axis_off()
    figure.suptitle("100 random 16×16 SUTD PRT geometries")
    figure.tight_layout()
    figure.savefig(args.reports_dir / "geometry_samples.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(4, 4, figsize=(12, 9), sharex=True, sharey=True)
    for axis, index in zip(axes.flat, chosen[:16]):
        _, response = dataset[int(index)]
        x_reflection = response[2].numpy() + 1j * response[3].numpy()
        axis.plot(dataset.frequency_ghz, np.abs(x_reflection), linewidth=0.8)
        axis.set_title(dataset.source_id(int(index)), fontsize=7)
        axis.set_xlabel("frequency (GHz)")
        axis.set_ylabel("|x-polarized reflection|")
    figure.suptitle("X-polarized reflection magnitude for selected geometries")
    figure.tight_layout()
    figure.savefig(args.reports_dir / "response_samples.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

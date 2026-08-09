"""Confirm that the forward model can overfit a tiny deterministic subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SUTDPRCMDataset
from src.models import ForwardSurrogateCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("data/processed/sutd_prcm_5k"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    dataset = SUTDPRCMDataset(args.subset_root, "train")
    geometry = torch.stack([dataset[index][0] for index in range(args.samples)])
    target = torch.stack([dataset[index][1] for index in range(args.samples)])
    model = ForwardSurrogateCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    initial_loss = float(loss_function(model(geometry), target).item())
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(geometry), target)
        loss.backward()
        optimizer.step()
    final_loss = float(loss_function(model(geometry), target).item())
    print(f"overfit loss: {initial_loss:.6f} -> {final_loss:.6f}")
    if final_loss >= initial_loss * 0.30:
        raise RuntimeError("Tiny-subset overfit check did not reduce loss by 70%")


if __name__ == "__main__":
    main()

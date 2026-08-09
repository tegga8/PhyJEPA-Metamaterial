"""Small neural baselines used after Phase 1 data validation."""

from __future__ import annotations

import torch
from torch import nn


class ForwardSurrogateCNN(nn.Module):
    """Predict the four real-valued reflection channels from a 16×16 pattern."""

    def __init__(self, output_channels: int = 4, frequency_points: int = 1001) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.frequency_points = frequency_points
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),  # 16×16 -> 8×8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),  # 8×8 -> 4×4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 512),
            nn.GELU(),
            nn.Linear(512, output_channels * frequency_points),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        batch_size = geometry.shape[0]
        response = self.head(self.encoder(geometry))
        return response.view(batch_size, self.output_channels, self.frequency_points)

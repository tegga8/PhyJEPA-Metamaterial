"""Small CNN encoder-decoder for Phase 3 partial-structure completion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CompletionCNN(nn.Module):
    """Predict completion logits from partial geometry and an explicit mask."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (2, 16, 16):
            raise ValueError(f"Expected [B, 2, 16, 16], got {tuple(inputs.shape)}")
        features = self.bottleneck(self.encoder(inputs))
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        features = self.decoder[0:2](features)
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        return self.decoder[2:](features)

    def predict_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(inputs))


def compose_completion(probabilities: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Composite predictions with observed pixels preserved exactly."""
    if probabilities.shape != mask.shape or inputs.shape[1:] != (2, 16, 16):
        raise ValueError("Expected probabilities/mask [B, 1, 16, 16] and inputs [B, 2, 16, 16]")
    partial = inputs[:, :1]
    return partial * (1.0 - mask) + probabilities * mask


def compose_binary_completion(probabilities: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return compose_completion((probabilities >= threshold).to(probabilities.dtype), inputs, mask)

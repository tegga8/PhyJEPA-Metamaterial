"""Response-only deterministic inverse-design baselines."""

from __future__ import annotations

import torch
from torch import nn


class SpectrumMLPBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 1001, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 3 or response.shape[1:] != (4, 1001):
            raise ValueError(f"Expected normalized response [B,4,1001], got {tuple(response.shape)}")
        return self.network(response)


class SpectrumToGeometryLatent(nn.Module):
    """Deterministic response-only predictor of the frozen geometry latent."""

    def __init__(self, hidden_dim: int = 1024, latent_channels: int = 64) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.backbone = SpectrumMLPBackbone(hidden_dim)
        self.head = nn.Linear(hidden_dim, latent_channels * 8 * 8)

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        latent = self.head(self.backbone(response))
        return latent.view(response.shape[0], self.latent_channels, 8, 8)


class SpectrumToGeometryDirectMLP(nn.Module):
    """Direct response-only geometry-logit baseline."""

    def __init__(self, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.backbone = SpectrumMLPBackbone(hidden_dim)
        self.head = nn.Linear(hidden_dim, 16 * 16)

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(response)).view(response.shape[0], 1, 16, 16)


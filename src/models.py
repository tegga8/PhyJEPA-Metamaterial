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


class _ResidualConvBlock(nn.Module):
    """A small pre-activation residual block for the 16x16 geometry encoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(features + self.block(features))


class ResponseAwareSurrogateCNN(nn.Module):
    """A modest response-aware CNN that preserves more geometry detail.

    Compared with the historical baseline, this encoder keeps a 4x4 feature
    map rather than collapsing to 2x2 and uses residual blocks.  A shallow
    1-D refinement head couples neighbouring frequency predictions, which gives
    the loss a way to improve local spectral features without introducing a
    large sequence model.
    """

    def __init__(self, output_channels: int = 4, frequency_points: int = 1001) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.frequency_points = frequency_points
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.GELU(),
            _ResidualConvBlock(24),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.GELU(),
            _ResidualConvBlock(48),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.GELU(),
        )
        self.base_head = nn.Linear(256, output_channels * frequency_points)
        self.spectral_refinement = nn.Sequential(
            nn.Conv1d(output_channels, 32, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(32, output_channels, kernel_size=9, padding=4),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        batch_size = geometry.shape[0]
        base = self.base_head(self.latent(self.encoder(geometry)))
        base = base.view(batch_size, self.output_channels, self.frequency_points)
        return base + self.spectral_refinement(base)


def build_forward_model(name: str, output_channels: int = 4, frequency_points: int = 1001) -> nn.Module:
    """Build a named forward model while keeping the baseline default stable."""
    models = {
        "ForwardSurrogateCNN": ForwardSurrogateCNN,
        "ResponseAwareSurrogateCNN": ResponseAwareSurrogateCNN,
    }
    try:
        model_class = models[name]
    except KeyError as exc:
        raise ValueError(f"Unknown forward model {name!r}; choose from {sorted(models)}") from exc
    return model_class(output_channels=output_channels, frequency_points=frequency_points)

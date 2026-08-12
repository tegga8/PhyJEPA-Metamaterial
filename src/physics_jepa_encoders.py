"""Encoders and predictor for the cross-modal Physics-JEPA experiment.

The geometry encoder reuses the Phase 4.2 ``SpatialGeometryEncoder`` trunk and
adds a compact vector head, keeping the representation small enough to study
compactness (32/64 dimensions) while staying close to the existing model
family.  The spectrum encoder is a small tokenized 1-D CNN followed by a
lightweight self-attention mixer and an attention pool, mapping the Phase-2
normalized ``[B, 4, 1001]`` response to a ``[B, latent_dim]`` vector.
"""

from __future__ import annotations

import torch
from torch import nn

from src.spatial_jepa_completion_model import SpatialGeometryEncoder


class GeometryLatentEncoder(nn.Module):
    """Encode a binary 16x16 geometry into a compact ``[B, latent_dim]`` vector."""

    def __init__(self, latent_dim: int = 64, spatial_channels: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_channels = spatial_channels
        self.spatial = SpatialGeometryEncoder(1, spatial_channels)
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(spatial_channels, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.ndim != 4 or geometry.shape[1:] != (1, 16, 16):
            raise ValueError(f"Expected geometry [B, 1, 16, 16], got {tuple(geometry.shape)}")
        latent = self.project(self.spatial(geometry))
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise RuntimeError(f"Geometry encoder produced unexpected shape {tuple(latent.shape)}")
        return latent


class PhysicsSpectrumEncoder(nn.Module):
    """Tokenized 1-D spectrum encoder mapping ``[B, 4, 1001]`` to ``[B, latent_dim]``."""

    def __init__(self, latent_dim: int = 64, num_tokens: int = 12, token_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        if token_dim % 8 != 0:
            raise ValueError(f"token_dim must be divisible by 8 for GroupNorm, got {token_dim}")
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim must be divisible by num_heads, got {token_dim} and {num_heads}")
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=9, padding=4),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, token_dim, kernel_size=9, padding=4),
            nn.GroupNorm(8, token_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(num_tokens),
        )
        self.attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, latent_dim),
        )

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 3 or response.shape[1:] != (4, 1001):
            raise ValueError(f"Expected normalized response [B, 4, 1001], got {tuple(response.shape)}")
        tokens = self.features(response).transpose(1, 2)  # [B, L, token_dim]
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        scores = torch.softmax((self.query @ tokens.transpose(1, 2)).squeeze(1), dim=-1)  # [B, L]
        pooled = (tokens * scores.unsqueeze(-1)).sum(dim=1)  # [B, token_dim]
        latent = self.head(pooled)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise RuntimeError(f"Spectrum encoder produced unexpected shape {tuple(latent.shape)}")
        return latent


class PhysicsPredictor(nn.Module):
    """Small MLP predictor, used both for geometry->physics and spectrum self-prediction."""

    def __init__(self, latent_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or 2 * latent_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent [B, {self.latent_dim}], got {tuple(latent.shape)}")
        return self.network(latent)

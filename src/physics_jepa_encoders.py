"""Encoders and predictor for the cross-modal Physics-JEPA experiment.

The geometry encoder reuses the Phase 4.2 ``SpatialGeometryEncoder`` trunk and
adds a compact vector head, keeping the representation small enough to study
compactness (32/64 dimensions) while staying close to the existing model
family.  The spectrum encoder is a small tokenized 1-D CNN followed by a
lightweight self-attention mixer and an attention pool, mapping the Phase-2
normalized ``[B, 4, 1001]`` response to a ``[B, latent_dim]`` vector.

``FrequencySpectrumEncoder`` (Physics-JEPA v3, label
``physics_jepa_v3_frequency_relational``) is a frequency-aware replacement:
local frequency resolution is preserved with multiscale dilated 1-D
convolutions that run at the full 1001-point resolution *before* any
tokenization, an explicit sinusoidal frequency-position embedding over the
2-12 GHz grid is added at the feature level, and a small self-attention
aggregates the ordered frequency tokens.  No large Transformer; no unlabeled
early 1001 -> small compression.
"""

from __future__ import annotations

import numpy as np
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


class FrequencySpectrumEncoder(nn.Module):
    """Frequency-aware spectrum encoder for Physics-JEPA v3.

    Maps the Phase-2 normalized ``[B, 4, 1001]`` response to ``[B, latent_dim]``
    while preserving frequency-local structure:

    ``4 x 1001 -> stem Conv1d -> + sinusoidal frequency-position embedding ->
    multiscale dilated Conv1d (full resolution) -> fused features -> strided
    Conv1d tokenizer (ordered local windows) -> small self-attention ->
    attention pool -> projection -> z_S [B, latent_dim]``.

    Positional encoding (exact): frequency grid ``f = linspace(2, 12, 1001)``,
    normalized coordinate ``tilde_f = (f - 2) / 10`` in ``[0, 1]``, and a
    sinusoidal code ``sin/cos(2*pi*h*tilde_f)`` for harmonics ``h = 0..H-1``,
    projected with one Linear layer onto the feature channels.  A small learned
    per-token position embedding is added after the strided tokenizer.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        channels: int = 48,
        token_dim: int = 64,
        num_heads: int = 4,
        num_harmonics: int = 8,
        kernel_size: int = 9,
        dilations: tuple[int, ...] = (1, 4, 16),
        token_stride: int = 16,
    ) -> None:
        super().__init__()
        if token_dim % 8 != 0:
            raise ValueError(f"token_dim must be divisible by 8 for GroupNorm, got {token_dim}")
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim must be divisible by num_heads, got {token_dim} and {num_heads}")
        if token_stride <= 0:
            raise ValueError(f"token_stride must be positive, got {token_stride}")
        self.latent_dim = latent_dim
        self.channels = channels
        self.token_dim = token_dim
        self.num_harmonics = num_harmonics
        self.dilations = list(dilations)
        self.token_stride = token_stride

        freq_ghz = np.linspace(2.0, 12.0, 1001)
        tilde_f = (freq_ghz - 2.0) / 10.0
        harmonics = np.arange(num_harmonics, dtype=np.float32)
        phase = 2.0 * np.pi * harmonics[None, :] * tilde_f[:, None]
        frequency_code = np.concatenate([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)
        self.register_buffer("frequency_code", torch.from_numpy(frequency_code))
        self.frequency_projection = nn.Linear(2 * num_harmonics, channels)

        self.stem = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.multiscale = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                ),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            )
            for dilation in dilations
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels * (1 + len(dilations)), channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.tokenizer = nn.Sequential(
            nn.Conv1d(channels, token_dim, kernel_size=token_stride + 1, stride=token_stride, padding=token_stride // 2),
            nn.GELU(),
        )
        num_tokens = int(np.floor((1001 + 2 * (token_stride // 2) - (token_stride + 1)) / token_stride)) + 1
        self.num_tokens = num_tokens
        self.token_position = nn.Parameter(torch.zeros(num_tokens, token_dim))
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
        features = self.stem(response)
        position = self.frequency_projection(self.frequency_code).transpose(0, 1).unsqueeze(0)
        features = features + position
        branches = [features, *[branch(features) for branch in self.multiscale]]
        features = self.mix(torch.cat(branches, dim=1))
        tokens = self.tokenizer(features).transpose(1, 2)
        tokens = tokens + self.token_position
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        scores = torch.softmax((self.query @ tokens.transpose(1, 2)).squeeze(1), dim=-1)
        pooled = (tokens * scores.unsqueeze(-1)).sum(dim=1)
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

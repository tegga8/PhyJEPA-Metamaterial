"""Complete-geometry autoencoder for the Phase 8 representation gate."""

from __future__ import annotations

import torch
from torch import nn

from src.spatial_jepa_completion_model import SpatialGeometryDecoder, SpatialGeometryEncoder


class GeometryAutoencoder(nn.Module):
    """Encode a complete binary geometry and decode it without EM or masks."""

    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = SpatialGeometryEncoder(1, latent_channels)
        self.decoder = SpatialGeometryDecoder(latent_channels)

    def encode(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.encoder(geometry)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(geometry)
        logits = self.decode(latent)
        return {"latent": latent, "logits": logits}


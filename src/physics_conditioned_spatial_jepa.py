"""Phase 5A EM-conditioned extension of the Phase 4.2 spatial JEPA."""

from __future__ import annotations

import torch
from torch import nn

from src.spatial_jepa_completion_model import SpatialJEPACompletionModel


class EMResponseEncoder(nn.Module):
    """Small 1-D CNN mapping normalized complex spectra [B,4,1001] to [B,128]."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 3 or response.shape[1:] != (4, 1001):
            raise ValueError(f"Expected normalized response [B, 4, 1001], got {tuple(response.shape)}")
        return self.projection(self.features(response).squeeze(-1))


class EMResponseDecoder(nn.Module):
    """Linear readout used only to validate the information in a 128-D EM embedding."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.readout = nn.Linear(embedding_dim, 4 * 1001)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding [B, {self.embedding_dim}], got {tuple(embedding.shape)}")
        return self.readout(embedding).view(embedding.shape[0], 4, 1001)


class FiLMConditioner(nn.Module):
    """Conservative FiLM modulation for a 64-channel Phase 4.2 context map."""

    def __init__(self, physics_dim: int = 128, latent_channels: int = 64) -> None:
        super().__init__()
        self.physics_dim = physics_dim
        self.latent_channels = latent_channels
        self.network = nn.Sequential(
            nn.Linear(physics_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * latent_channels),
        )
        # At initialization gamma=1 and beta=0, preserving the Phase 4.2 predictor path.
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, latent: torch.Tensor, physics_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latent.ndim != 4 or latent.shape[1:] != (self.latent_channels, 8, 8):
            raise ValueError(f"Expected latent [B, {self.latent_channels}, 8, 8], got {tuple(latent.shape)}")
        if physics_embedding.ndim != 2 or physics_embedding.shape != (latent.shape[0], self.physics_dim):
            raise ValueError(f"Expected physics embedding [B, {self.physics_dim}], got {tuple(physics_embedding.shape)}")
        delta_gamma, beta = self.network(physics_embedding).chunk(2, dim=1)
        gamma = 1.0 + delta_gamma
        conditioned = gamma[:, :, None, None] * latent + beta[:, :, None, None]
        return conditioned, gamma, beta


class PhysicsConditionedSpatialJEPA(SpatialJEPACompletionModel):
    """Phase 4.2 spatial JEPA with an EM encoder and FiLM before its predictor.

    Context encoder, target encoder, predictor network, decoder, target EMA,
    and output latent shape are inherited unchanged from Phase 4.2.  The target
    encoder remains geometry-only.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        predictor_hidden_channels: int = 128,
        ema_decay: float = 0.996,
        physics_embedding_dim: int = 128,
    ) -> None:
        super().__init__(latent_channels, predictor_hidden_channels, ema_decay)
        self.physics_embedding_dim = physics_embedding_dim
        self.em_encoder = EMResponseEncoder(physics_embedding_dim)
        self.film = FiLMConditioner(physics_embedding_dim, latent_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        response: torch.Tensor,
        target_geometry: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_context = self.context_encoder(inputs)
        z_phys = self.em_encoder(response)
        conditioned_context, gamma, beta = self.film(z_context, z_phys)
        z_pred = self.predictor(conditioned_context)
        output = {
            "z_context": z_context,
            "z_phys": z_phys,
            "conditioned_context": conditioned_context,
            "film_gamma": gamma,
            "film_beta": beta,
            "z_pred": z_pred,
            "logits": self.decoder(z_pred),
        }
        if target_geometry is not None:
            output["z_target"] = self.encode_target(target_geometry)
        return output

    def trainable_parameters(self):
        return (*super().trainable_parameters(), *self.em_encoder.parameters(), *self.film.parameters())

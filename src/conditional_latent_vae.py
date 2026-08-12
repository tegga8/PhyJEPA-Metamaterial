"""Conditional latent VAE for one-to-many spectrum-to-geometry generation."""

from __future__ import annotations

import torch
from torch import nn

from src.spectrum_inverse_models import SpectrumMLPBackbone


class ConditionalLatentVAE(nn.Module):
    """Model q(z|S,G) during training and p(z|S) during generation.

    The latent is represented in standardized geometry-latent units. At
    inference the public ``sample`` path receives only a response and noise.
    """

    def __init__(self, hidden_dim: int = 256, latent_shape: tuple[int, int, int] = (64, 8, 8)) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_shape = latent_shape
        self.latent_dim = int(torch.tensor(latent_shape).prod().item())
        self.response_backbone = SpectrumMLPBackbone(hidden_dim)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.latent_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + self.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.latent_dim),
        )

    def _split_parameters(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = values.chunk(2, dim=1)
        return mean, logvar.clamp(-8.0, 4.0)

    def prior_parameters(self, response: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_parameters(self.prior(self.response_backbone(response)))

    def posterior_parameters(self, response: torch.Tensor, target_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if target_latent.ndim != 2 or target_latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected flattened standardized latent [B,{self.latent_dim}], got {tuple(target_latent.shape)}")
        features = self.response_backbone(response)
        return self._split_parameters(self.posterior(torch.cat((features, target_latent), dim=1)))

    def forward(self, response: torch.Tensor, target_latent: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prior_mean, prior_logvar = self.prior_parameters(response)
        output = {"prior_mean": prior_mean, "prior_logvar": prior_logvar}
        if target_latent is not None:
            posterior_mean, posterior_logvar = self.posterior_parameters(response, target_latent)
            if noise is None:
                noise = torch.randn_like(posterior_mean)
            sample = posterior_mean + noise * torch.exp(0.5 * posterior_logvar)
            output.update({"posterior_mean": posterior_mean, "posterior_logvar": posterior_logvar, "posterior_sample": sample})
        return output

    def sample(self, response: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Sample standardized geometry latents from response and optional noise only."""
        mean, logvar = self.prior_parameters(response)
        if noise is None:
            noise = torch.randn_like(mean)
        if noise.shape != mean.shape:
            raise ValueError(f"Expected noise shape {tuple(mean.shape)}, got {tuple(noise.shape)}")
        return mean + noise * torch.exp(0.5 * logvar)

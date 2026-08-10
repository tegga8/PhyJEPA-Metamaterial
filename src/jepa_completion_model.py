"""Compact JEPA model for Phase 4 masked geometry completion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GeometryEncoder(nn.Module):
    """Encode either [partial, mask] or complete geometry into a latent vector."""

    def __init__(self, input_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Linear(128, latent_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.projection(F.adaptive_avg_pool2d(features, 1).flatten(1))


class Predictor(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class GeometryDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, 32 * 4 * 4)
        self.features = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.project(latent).view(latent.shape[0], 32, 4, 4)
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        return self.features(features)


class JEPACompletionModel(nn.Module):
    """Context/target encoders, predictor, and geometry decoder."""

    def __init__(self, latent_dim: int = 128, predictor_hidden_dim: int = 256, ema_decay: float = 0.996) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.ema_decay = ema_decay
        self.context_encoder = GeometryEncoder(2, latent_dim)
        self.target_encoder = GeometryEncoder(1, latent_dim)
        self.predictor = Predictor(latent_dim, predictor_hidden_dim)
        self.decoder = GeometryDecoder(latent_dim)
        self.initialize_target_encoder()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

    @torch.no_grad()
    def initialize_target_encoder(self) -> None:
        context_state = self.context_encoder.state_dict()
        target_state = self.target_encoder.state_dict()
        for name, target_value in target_state.items():
            context_name = name
            source = context_state[context_name]
            if name == "features.0.weight":
                source = source.mean(dim=1, keepdim=True)
            if source.shape != target_value.shape:
                raise ValueError(f"Cannot initialize target encoder parameter {name}: {source.shape} vs {target_value.shape}")
            target_value.copy_(source)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        context_parameters = dict(self.context_encoder.named_parameters())
        for name, target_parameter in self.target_encoder.named_parameters():
            context_parameter = context_parameters[name]
            if name == "features.0.weight":
                context_parameter = context_parameter.mean(dim=1, keepdim=True)
            target_parameter.mul_(self.ema_decay).add_(context_parameter, alpha=1.0 - self.ema_decay)
        for target_buffer, context_buffer in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            target_buffer.copy_(context_buffer)

    @torch.no_grad()
    def encode_target(self, target_geometry: torch.Tensor) -> torch.Tensor:
        self.target_encoder.eval()
        return self.target_encoder(target_geometry)

    def forward(self, inputs: torch.Tensor, target_geometry: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        context = self.context_encoder(inputs)
        prediction = self.predictor(context)
        output = {"z_context": context, "z_pred": prediction, "logits": self.decoder(prediction)}
        if target_geometry is not None:
            output["z_target"] = self.encode_target(target_geometry)
        return output

    def trainable_parameters(self):
        return (*self.context_encoder.parameters(), *self.predictor.parameters(), *self.decoder.parameters())


def compose_jepa_completion(probabilities: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probabilities.shape != mask.shape or inputs.shape[1:] != (2, 16, 16):
        raise ValueError("Expected probabilities/mask [B, 1, 16, 16] and inputs [B, 2, 16, 16]")
    return inputs[:, :1] * (1.0 - mask) + probabilities * mask


def compose_binary_jepa_completion(probabilities: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return compose_jepa_completion((probabilities >= threshold).to(probabilities.dtype), inputs, mask)

"""Spatial-latent JEPA model for Phase 4.1 geometry completion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialGeometryEncoder(nn.Module):
    """Encode geometry into a coordinate-preserving [B, C, 8, 8] map."""

    def __init__(self, input_channels: int, latent_channels: int = 64) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, 3, padding=1),
            nn.GroupNorm(8, latent_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.GroupNorm(8, latent_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels or inputs.shape[-2:] != (16, 16):
            raise ValueError(f"Expected [B, {self.input_channels}, 16, 16], got {tuple(inputs.shape)}")
        output = self.features(inputs)
        if output.shape[1:] != (self.latent_channels, 8, 8):
            raise RuntimeError(f"Spatial encoder produced unexpected shape {tuple(output.shape)}")
        return output


class SpatialPredictor(nn.Module):
    """Convolutional predictor that preserves the 8x8 latent coordinates."""

    def __init__(self, latent_channels: int = 64, hidden_channels: int = 128) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.network = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (self.latent_channels, 8, 8):
            raise ValueError(f"Expected [B, {self.latent_channels}, 8, 8], got {tuple(latent.shape)}")
        return self.network(latent)


class SpatialGeometryDecoder(nn.Module):
    """Decode a spatial latent map into 16x16 geometry logits."""

    def __init__(self, latent_channels: int = 64) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.features = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (self.latent_channels, 8, 8):
            raise ValueError(f"Expected [B, {self.latent_channels}, 8, 8], got {tuple(latent.shape)}")
        features = self.features[0:2](latent)
        features = F.interpolate(features, size=(16, 16), mode="bilinear", align_corners=False)
        return self.features[2:](features)


class SpatialJEPACompletionModel(nn.Module):
    """Context/target spatial encoders, spatial predictor, and decoder."""

    def __init__(self, latent_channels: int = 64, predictor_hidden_channels: int = 128, ema_decay: float = 0.996) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.predictor_hidden_channels = predictor_hidden_channels
        self.ema_decay = ema_decay
        self.context_encoder = SpatialGeometryEncoder(2, latent_channels)
        self.target_encoder = SpatialGeometryEncoder(1, latent_channels)
        self.predictor = SpatialPredictor(latent_channels, predictor_hidden_channels)
        self.decoder = SpatialGeometryDecoder(latent_channels)
        self.initialize_target_encoder()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

    @torch.no_grad()
    def initialize_target_encoder(self) -> None:
        """Copy context weights; average the context-only input channels for target conv1."""
        context_state = self.context_encoder.state_dict()
        target_state = self.target_encoder.state_dict()
        for name, target_value in target_state.items():
            source = context_state[name]
            if name == "features.0.weight":
                source = source.mean(dim=1, keepdim=True)
            if source.shape != target_value.shape:
                raise ValueError(f"Cannot initialize target parameter {name}: {source.shape} vs {target_value.shape}")
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
        z_context = self.context_encoder(inputs)
        z_pred = self.predictor(z_context)
        output = {"z_context": z_context, "z_pred": z_pred, "logits": self.decoder(z_pred)}
        if target_geometry is not None:
            output["z_target"] = self.encode_target(target_geometry)
        return output

    def trainable_parameters(self):
        return (*self.context_encoder.parameters(), *self.predictor.parameters(), *self.decoder.parameters())


def compose_spatial_completion(probabilities: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probabilities.shape != mask.shape or inputs.shape[1:] != (2, 16, 16):
        raise ValueError("Expected probabilities/mask [B, 1, 16, 16] and inputs [B, 2, 16, 16]")
    return inputs[:, :1] * (1.0 - mask) + probabilities * mask


def compose_binary_spatial_completion(probabilities: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return compose_spatial_completion((probabilities >= threshold).to(probabilities.dtype), inputs, mask)

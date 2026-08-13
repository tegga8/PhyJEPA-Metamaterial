"""Cross-modal Physics-JEPA model with a momentum spectrum target encoder.

The online branch maps a binary geometry through ``geometry_encoder`` and the
predictor into a *predicted physical latent* ``z_pred``.  The spectrum branch
encodes the Phase-2 normalized response with the online ``spectrum_encoder``
and a momentum copy ``spectrum_target_encoder`` (EMA updated, no gradient).
The core objective makes ``z_pred`` match the momentum spectrum latent; a
small within-spectrum bootstrap term trains the online spectrum encoder
against its own momentum target so the target branch becomes physically
useful without any reconstruction loss.
"""

from __future__ import annotations

import torch
from torch import nn

from src.physics_jepa_encoders import FrequencySpectrumEncoder, GeometryLatentEncoder, PhysicsPredictor, PhysicsSpectrumEncoder


class PhysicsJEPA(nn.Module):
    """Geometry -> physical latent -> spectrum-target JEPA model."""

    def __init__(
        self,
        latent_dim: int = 64,
        ema_decay: float = 0.996,
        num_tokens: int = 12,
        token_dim: int = 64,
        alpha: float = 0.5,
        lambda_variance: float = 0.1,
        target_centering: bool = False,
        target_center_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.alpha = alpha
        self.lambda_variance = lambda_variance
        self.target_centering = target_centering
        self.target_center_decay = target_center_decay
        self.register_buffer("target_center", torch.zeros(latent_dim))
        self.geometry_encoder = GeometryLatentEncoder(latent_dim)
        self.spectrum_encoder = PhysicsSpectrumEncoder(latent_dim, num_tokens, token_dim)
        self.spectrum_target_encoder = PhysicsSpectrumEncoder(latent_dim, num_tokens, token_dim)
        self.predictor = PhysicsPredictor(latent_dim)
        self.spectrum_predictor = PhysicsPredictor(latent_dim)
        self.initialize_target_encoder()
        for parameter in self.spectrum_target_encoder.parameters():
            parameter.requires_grad_(False)
        self.spectrum_target_encoder.eval()

    @torch.no_grad()
    def initialize_target_encoder(self) -> None:
        """Hard-copy the online spectrum encoder into the momentum target."""
        source = self.spectrum_encoder.state_dict()
        target = self.spectrum_target_encoder.state_dict()
        for name, target_value in target.items():
            source_value = source[name]
            if source_value.shape != target_value.shape:
                raise ValueError(f"Cannot initialize target parameter {name}: {tuple(source_value.shape)} vs {tuple(target_value.shape)}")
            target_value.copy_(source_value)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """EMA-update the momentum target from the online spectrum encoder."""
        source = dict(self.spectrum_encoder.named_parameters())
        for name, target_parameter in self.spectrum_target_encoder.named_parameters():
            target_parameter.mul_(self.ema_decay).add_(source[name], alpha=1.0 - self.ema_decay)
        for target_buffer, source_buffer in zip(self.spectrum_target_encoder.buffers(), self.spectrum_encoder.buffers()):
            target_buffer.copy_(source_buffer)

    @torch.no_grad()
    def encode_spectrum_target(self, response: torch.Tensor) -> torch.Tensor:
        self.spectrum_target_encoder.eval()
        return self.spectrum_target_encoder(response)

    @torch.no_grad()
    def update_target_center(self, z_target_raw: torch.Tensor) -> None:
        """EMA-update the target centering buffer from raw target batch means."""
        if not self.target_centering:
            return
        batch_mean = z_target_raw.mean(dim=0)
        self.target_center.mul_(self.target_center_decay).add_(batch_mean, alpha=1.0 - self.target_center_decay)

    def forward(self, geometry: torch.Tensor, response: torch.Tensor) -> dict[str, torch.Tensor]:
        z_geometry = self.geometry_encoder(geometry)
        z_pred = self.predictor(z_geometry)
        z_online = self.spectrum_encoder(response)
        z_self = self.spectrum_predictor(z_online)
        z_target_raw = self.encode_spectrum_target(response)
        z_target = z_target_raw - self.target_center if self.target_centering else z_target_raw
        return {
            "z_geometry": z_geometry,
            "z_online": z_online,
            "z_self": z_self,
            "z_pred": z_pred,
            "z_target": z_target,
            "z_target_raw": z_target_raw,
        }

    def masking_forward(self, masked_response: torch.Tensor, full_response: torch.Tensor) -> dict[str, torch.Tensor]:
        """Masked-spectrum variant: infer the full physical latent from visible context."""
        z_visible = self.spectrum_encoder(masked_response)
        z_pred = self.spectrum_predictor(z_visible)
        z_target = self.encode_spectrum_target(full_response)
        return {"z_visible": z_visible, "z_pred": z_pred, "z_target": z_target}

    def trainable_parameters(self):
        return (
            *self.geometry_encoder.parameters(),
            *self.spectrum_encoder.parameters(),
            *self.predictor.parameters(),
            *self.spectrum_predictor.parameters(),
        )

    def target_parameter_names(self) -> list[str]:
        return [name for name, _ in self.spectrum_target_encoder.named_parameters()]


class PhysicsJEPAFrequencyRelational(nn.Module):
    """Physics-JEPA v3: frequency-aware spectrum target + small relational term.

    Label ``physics_jepa_v3_frequency_relational``.  Reuses the v2 geometry
    branch (``GeometryLatentEncoder`` + ``PhysicsPredictor``), the v2 EMA
    momentum target machinery, and the v2 variance/covariance collapse guards.
    The two v3 changes are the frequency-aware ``FrequencySpectrumEncoder`` on
    both the online and momentum spectrum branches, and the relational
    margin-ranking loss (Change B, applied at the training loop level).  The
    target branch remains EMA-updated and stop-gradient; it never receives a
    gradient from ordinary backpropagation.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        ema_decay: float = 0.996,
        channels: int = 48,
        token_dim: int = 64,
        num_heads: int = 4,
        num_harmonics: int = 8,
        kernel_size: int = 9,
        dilations: tuple[int, ...] = (1, 4, 16),
        token_stride: int = 16,
        alpha: float = 0.5,
        target_centering: bool = False,
        target_center_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.alpha = alpha
        self.target_centering = target_centering
        self.target_center_decay = target_center_decay
        self.register_buffer("target_center", torch.zeros(latent_dim))
        self.geometry_encoder = GeometryLatentEncoder(latent_dim)
        self.spectrum_encoder = FrequencySpectrumEncoder(
            latent_dim, channels, token_dim, num_heads, num_harmonics, kernel_size, dilations, token_stride
        )
        self.spectrum_target_encoder = FrequencySpectrumEncoder(
            latent_dim, channels, token_dim, num_heads, num_harmonics, kernel_size, dilations, token_stride
        )
        self.predictor = PhysicsPredictor(latent_dim)
        self.spectrum_predictor = PhysicsPredictor(latent_dim)
        self.initialize_target_encoder()
        for parameter in self.spectrum_target_encoder.parameters():
            parameter.requires_grad_(False)
        self.spectrum_target_encoder.eval()

    @torch.no_grad()
    def initialize_target_encoder(self) -> None:
        """Hard-copy the online spectrum encoder into the momentum target."""
        source = self.spectrum_encoder.state_dict()
        target = self.spectrum_target_encoder.state_dict()
        for name, target_value in target.items():
            source_value = source[name]
            if source_value.shape != target_value.shape:
                raise ValueError(f"Cannot initialize target parameter {name}: {tuple(source_value.shape)} vs {tuple(target_value.shape)}")
            target_value.copy_(source_value)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """EMA-update the momentum target from the online spectrum encoder."""
        source = dict(self.spectrum_encoder.named_parameters())
        for name, target_parameter in self.spectrum_target_encoder.named_parameters():
            target_parameter.mul_(self.ema_decay).add_(source[name], alpha=1.0 - self.ema_decay)
        for target_buffer, source_buffer in zip(self.spectrum_target_encoder.buffers(), self.spectrum_encoder.buffers()):
            target_buffer.copy_(source_buffer)

    @torch.no_grad()
    def encode_spectrum_target(self, response: torch.Tensor) -> torch.Tensor:
        self.spectrum_target_encoder.eval()
        return self.spectrum_target_encoder(response)

    @torch.no_grad()
    def update_target_center(self, z_target_raw: torch.Tensor) -> None:
        """EMA-update the target centering buffer from raw target batch means."""
        if not self.target_centering:
            return
        batch_mean = z_target_raw.mean(dim=0)
        self.target_center.mul_(self.target_center_decay).add_(batch_mean, alpha=1.0 - self.target_center_decay)

    def forward(self, geometry: torch.Tensor, response: torch.Tensor) -> dict[str, torch.Tensor]:
        z_geometry = self.geometry_encoder(geometry)
        z_pred = self.predictor(z_geometry)
        z_online = self.spectrum_encoder(response)
        z_self = self.spectrum_predictor(z_online)
        z_target_raw = self.encode_spectrum_target(response)
        z_target = z_target_raw - self.target_center if self.target_centering else z_target_raw
        return {
            "z_geometry": z_geometry,
            "z_online": z_online,
            "z_self": z_self,
            "z_pred": z_pred,
            "z_target": z_target,
            "z_target_raw": z_target_raw,
        }

    def trainable_parameters(self):
        return (
            *self.geometry_encoder.parameters(),
            *self.spectrum_encoder.parameters(),
            *self.predictor.parameters(),
            *self.spectrum_predictor.parameters(),
        )

    def target_parameter_names(self) -> list[str]:
        return [name for name, _ in self.spectrum_target_encoder.named_parameters()]

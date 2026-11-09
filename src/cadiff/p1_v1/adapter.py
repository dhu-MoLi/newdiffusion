from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cadiff.p1_v1.config import AdapterConfig


@dataclass(frozen=True)
class ResidualSlotSpec:
    index: int
    channels: int
    height: int
    width: int


SD15_DOWN_RESIDUAL_SLOTS = (
    ResidualSlotSpec(0, 320, 64, 64),
    ResidualSlotSpec(1, 320, 64, 64),
    ResidualSlotSpec(2, 320, 64, 64),
    ResidualSlotSpec(3, 320, 32, 32),
    ResidualSlotSpec(4, 640, 32, 32),
    ResidualSlotSpec(5, 640, 32, 32),
    ResidualSlotSpec(6, 640, 16, 16),
    ResidualSlotSpec(7, 1280, 16, 16),
    ResidualSlotSpec(8, 1280, 16, 16),
    ResidualSlotSpec(9, 1280, 8, 8),
    ResidualSlotSpec(10, 1280, 8, 8),
    ResidualSlotSpec(11, 1280, 8, 8),
)

_SUPPORTED_ADAPTER_CONFIG = AdapterConfig(
    type_embedding_dim=128,
    stem_channels=64,
    scale64_channels=128,
    scale32_channels=256,
    blocks_per_scale=2,
)
_SD15_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 2,
    "block_out_channels": (320, 640, 1280, 1280),
}


def _validate_adapter_config(config: AdapterConfig) -> None:
    if not isinstance(config, AdapterConfig):
        raise ValueError("config must be an AdapterConfig")
    for field in _SUPPORTED_ADAPTER_CONFIG.__dataclass_fields__:
        actual = getattr(config, field)
        expected = getattr(_SUPPORTED_ADAPTER_CONFIG, field)
        if type(actual) is not int or actual != expected:
            raise ValueError(f"{field} has an unsupported value")


def validate_sd15_unet_config(config: object) -> None:
    for field, expected in _SD15_UNET_CONFIG.items():
        try:
            actual = getattr(config, field)
        except AttributeError as error:
            raise ValueError(f"{field} is required") from error
        if field == "block_out_channels":
            if not isinstance(actual, list | tuple) or any(
                type(value) is not int for value in actual
            ):
                raise ValueError(f"{field} has an unsupported value")
            actual = tuple(actual)
        elif type(actual) is not int:
            raise ValueError(f"{field} has an unsupported value")
        if actual != expected:
            raise ValueError(f"{field} has an unsupported value")


def _sinusoidal_timestep_embedding(timesteps: Tensor, dtype: torch.dtype) -> Tensor:
    half_dim = 64
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half_dim, device=timesteps.device, dtype=dtype)
        / half_dim
    )
    angles = timesteps.to(dtype=dtype).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)


class _FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, conditioning_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.film = nn.Linear(conditioning_channels, 2 * channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, features: Tensor, conditioning: Tensor) -> Tensor:
        residual = features
        features = self.conv1(self.activation(self.norm1(features)))
        scale, shift = self.film(conditioning).chunk(2, dim=1)
        features = self.norm2(features)
        features = features * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        features = self.conv2(self.activation(features))
        return residual + features


class MultiScaleTypeAdapter(nn.Module):
    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        _validate_adapter_config(config)
        self.config = config
        conditioning_channels = config.scale32_channels
        self.conditioning_mlp = nn.Sequential(
            nn.Linear(2 * config.type_embedding_dim, conditioning_channels),
            nn.SiLU(),
            nn.Linear(conditioning_channels, conditioning_channels),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(4, config.stem_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                config.stem_channels,
                config.scale64_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        self.scale64_blocks = nn.ModuleList(
            _FiLMResidualBlock(config.scale64_channels, conditioning_channels)
            for _ in range(config.blocks_per_scale)
        )
        self.downsample = nn.Conv2d(
            config.scale64_channels,
            config.scale32_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.scale32_blocks = nn.ModuleList(
            _FiLMResidualBlock(config.scale32_channels, conditioning_channels)
            for _ in range(config.blocks_per_scale)
        )

        input_channels = (
            config.scale64_channels,
            config.scale64_channels,
            config.scale64_channels,
            config.scale32_channels,
            config.scale32_channels,
            config.scale32_channels,
        )
        output_channels = (320, 320, 320, 320, 640, 640)
        self.output_heads = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels, out_channels in zip(input_channels, output_channels, strict=True)
        )
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _validate_inputs(
        self,
        noisy_latent: Tensor,
        timesteps: Tensor,
        type_embedding: Tensor,
    ) -> None:
        if not isinstance(noisy_latent, Tensor):
            raise ValueError("noisy_latent must be a tensor")
        if noisy_latent.ndim != 4:
            raise ValueError("noisy_latent must be four-dimensional")
        if noisy_latent.shape[0] <= 0:
            raise ValueError("noisy_latent batch must be non-empty")
        if noisy_latent.shape[1] != 4:
            raise ValueError("noisy_latent must have 4 channels")
        if noisy_latent.shape[2:] != (64, 64):
            raise ValueError("noisy_latent spatial shape must be 64 x 64")
        if not noisy_latent.is_floating_point():
            raise ValueError("noisy_latent must have a floating dtype")
        parameter = self.stem[0].weight
        if noisy_latent.device != parameter.device:
            raise ValueError("noisy_latent device must match the adapter device")
        if noisy_latent.dtype != parameter.dtype:
            raise ValueError("noisy_latent dtype must match the adapter dtype")
        if not torch.isfinite(noisy_latent).all().item():
            raise ValueError("noisy_latent must contain only finite values")

        if not isinstance(timesteps, Tensor):
            raise ValueError("timesteps must be a tensor")
        if timesteps.dtype is not torch.long:
            raise ValueError("timesteps must have dtype torch.long")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be one-dimensional")
        if timesteps.shape[0] != noisy_latent.shape[0]:
            raise ValueError("timesteps batch must match noisy_latent")
        if timesteps.device != noisy_latent.device:
            raise ValueError("timesteps device must match noisy_latent")

        if not isinstance(type_embedding, Tensor):
            raise ValueError("type_embedding must be a tensor")
        if type_embedding.ndim != 2:
            raise ValueError("type_embedding must be two-dimensional")
        if type_embedding.shape[1] != self.config.type_embedding_dim:
            raise ValueError("type_embedding width must be 128")
        if type_embedding.shape[0] != noisy_latent.shape[0]:
            raise ValueError("type_embedding batch must match noisy_latent")
        if type_embedding.device != noisy_latent.device:
            raise ValueError("type_embedding device must match noisy_latent")
        if type_embedding.dtype != noisy_latent.dtype:
            raise ValueError("type_embedding dtype must match noisy_latent")
        if not torch.isfinite(type_embedding).all().item():
            raise ValueError("type_embedding must contain only finite values")

    def forward(
        self,
        noisy_latent: Tensor,
        timesteps: Tensor,
        type_embedding: Tensor,
    ) -> tuple[Tensor, ...]:
        self._validate_inputs(noisy_latent, timesteps, type_embedding)
        time_embedding = _sinusoidal_timestep_embedding(timesteps, noisy_latent.dtype)
        conditioning = self.conditioning_mlp(torch.cat((time_embedding, type_embedding), dim=1))

        scale64 = self.stem(noisy_latent)
        features = [scale64]
        for block in self.scale64_blocks:
            scale64 = block(scale64, conditioning)
            features.append(scale64)

        scale32 = self.downsample(scale64)
        features.append(scale32)
        for block in self.scale32_blocks:
            scale32 = block(scale32, conditioning)
            features.append(scale32)

        selected = tuple(
            head(feature)
            for head, feature in zip(self.output_heads, features, strict=True)
        )
        constant = tuple(
            noisy_latent.new_zeros(
                (noisy_latent.shape[0], slot.channels, slot.height, slot.width),
                requires_grad=False,
            )
            for slot in SD15_DOWN_RESIDUAL_SLOTS[6:]
        )
        return selected + constant

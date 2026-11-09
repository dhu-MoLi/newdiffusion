from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cadiff.p1_v1.conditioning import DefectConditioner
from cadiff.p2_a0.config import P2A0Config

_SD15_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 2,
    "block_out_channels": (320, 640, 1280, 1280),
}


def _validate_sd15_unet_config(config: object) -> None:
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


@dataclass(frozen=True)
class AdapterResiduals:
    down: tuple[Tensor, ...]
    mid: Tensor


def _time_embedding(timesteps: Tensor, dtype: torch.dtype) -> Tensor:
    half = 64
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=dtype)
        / half
    )
    angles = timesteps.to(dtype=dtype).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)


class _FiLMBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.film = nn.Linear(256, 2 * channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, features: Tensor, conditioning: Tensor) -> Tensor:
        residual = features
        features = self.conv1(self.activation(self.norm1(features)))
        scale, shift = self.film(conditioning).chunk(2, dim=1)
        features = self.norm2(features)
        features = features * (1.0 + scale[:, :, None, None])
        features = features + shift[:, :, None, None]
        return residual + self.conv2(self.activation(features))


class MaskedInfillAdapter(nn.Module):
    def __init__(self, config: P2A0Config) -> None:
        super().__init__()
        if not isinstance(config, P2A0Config):
            raise ValueError("config must be a P2A0Config")
        self.config = config
        self.conditioning = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
        )
        self.block64 = _FiLMBlock(128)
        self.down32 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.block32 = _FiLMBlock(256)
        self.down16 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.block16 = _FiLMBlock(512)
        self.down8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.block8 = _FiLMBlock(512)
        in_channels = (128,) * 3 + (256,) * 3 + (512,) * 6
        self.output_heads = nn.ModuleList(
            nn.ModuleList(
                nn.Conv2d(source, slot.channels, 1)
                for source, slot in zip(
                    in_channels,
                    SD15_DOWN_RESIDUAL_SLOTS,
                    strict=True,
                )
            )
            for _ in config.defect_types
        )
        self.mid_heads = nn.ModuleList(
            nn.Conv2d(512, 1280, 1) for _ in config.defect_types
        )
        for route in self.output_heads:
            for head in route:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        for head in self.mid_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def routed_parameters(self, route_index: object) -> tuple[nn.Parameter, ...]:
        if type(route_index) is not int or not 0 <= route_index < len(self.output_heads):
            raise ValueError("route_index must identify a configured defect type")
        inactive = {
            id(parameter)
            for index in range(len(self.output_heads))
            if index != route_index
            for module in (*self.output_heads[index], self.mid_heads[index])
            for parameter in module.parameters()
        }
        return tuple(
            parameter for parameter in self.parameters() if id(parameter) not in inactive
        )

    def _validate(
        self,
        noisy: Tensor,
        context: Tensor,
        mask: Tensor,
        timesteps: Tensor,
        type_embedding: Tensor,
        route_index: object,
    ) -> None:
        parameter = self.stem[0].weight
        for name, value, channels in (
            ("noisy", noisy, 4),
            ("context", context, 4),
            ("mask", mask, 1),
        ):
            if not isinstance(value, Tensor) or value.shape != (
                noisy.shape[0],
                channels,
                64,
                64,
            ):
                raise ValueError(f"{name} must have the expected latent shape")
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain only finite floating values")
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ValueError(f"{name} device and dtype must match adapter")
        if noisy.shape[0] <= 0:
            raise ValueError("batch must be non-empty")
        if not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError("mask must be binary")
        if (
            not isinstance(timesteps, Tensor)
            or timesteps.dtype is not torch.long
            or timesteps.shape != (noisy.shape[0],)
            or timesteps.device != noisy.device
        ):
            raise ValueError("timesteps must match the latent batch and device")
        if (
            not isinstance(type_embedding, Tensor)
            or type_embedding.shape != (noisy.shape[0], 128)
            or type_embedding.device != noisy.device
            or type_embedding.dtype != noisy.dtype
            or not torch.isfinite(type_embedding).all().item()
        ):
            raise ValueError("type_embedding must be finite [batch, 128]")
        if type(route_index) is not int or not 0 <= route_index < len(self.output_heads):
            raise ValueError("route_index must identify a configured defect type")

    def forward(
        self,
        noisy: Tensor,
        context: Tensor,
        mask: Tensor,
        timesteps: Tensor,
        type_embedding: Tensor,
        route_index: int,
    ) -> AdapterResiduals:
        self._validate(noisy, context, mask, timesteps, type_embedding, route_index)
        condition = self.conditioning(
            torch.cat((_time_embedding(timesteps, noisy.dtype), type_embedding), dim=1)
        )
        feature64 = self.block64(self.stem(torch.cat((noisy, context, mask), dim=1)), condition)
        feature32 = self.block32(self.down32(feature64), condition)
        feature16 = self.block16(self.down16(feature32), condition)
        feature8 = self.block8(self.down8(feature16), condition)
        features = (feature64,) * 3 + (feature32,) * 3 + (feature16,) * 3 + (feature8,) * 3
        route_heads = self.output_heads[route_index]
        down = tuple(
            head(feature)
            for head, feature in zip(route_heads, features, strict=True)
        )
        return AdapterResiduals(down=down, mid=self.mid_heads[route_index](feature8))


class P2InfillNoisePredictor(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        conditioner: DefectConditioner,
        adapter: MaskedInfillAdapter,
    ) -> None:
        super().__init__()
        if not isinstance(conditioner, DefectConditioner):
            raise TypeError("conditioner must be a DefectConditioner")
        if not isinstance(adapter, MaskedInfillAdapter):
            raise TypeError("adapter must be a MaskedInfillAdapter")
        if (
            conditioner.vocabulary.category != adapter.config.category
            or conditioner.vocabulary.labels != adapter.config.defect_types
        ):
            raise ValueError("conditioner vocabulary must match adapter config")
        try:
            _validate_sd15_unet_config(unet.config)  # type: ignore[union-attr]
        except AttributeError as error:
            raise ValueError("unet config is required") from error
        self.unet = unet
        self.conditioner = conditioner
        self.adapter = adapter
        self.unet.requires_grad_(False)
        self.unet.eval()

    def train(self, mode: bool = True) -> P2InfillNoisePredictor:
        if not isinstance(mode, bool):
            raise ValueError("training mode must be boolean")
        self.training = mode
        self.conditioner.train(mode)
        self.adapter.train(mode)
        self.unet.requires_grad_(False)
        self.unet.eval()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in (*self.adapter.parameters(), *self.conditioner.parameters()):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        return tuple(parameters)

    def gradient_parameters(self, defect_type: str) -> tuple[nn.Parameter, ...]:
        route_index = self.conditioner.vocabulary.index(defect_type)
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in (
            *self.adapter.routed_parameters(route_index),
            *self.conditioner.parameters(),
        ):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        return tuple(parameters)

    def assert_frozen_backbone(self) -> None:
        if any(parameter.requires_grad for parameter in self.unet.parameters()):
            raise AssertionError("frozen U-Net parameters must not require gradients")
        if any(module.training for module in self.unet.modules()):
            raise AssertionError("frozen U-Net and all descendants must remain in eval")
        if any(parameter.grad is not None for parameter in self.unet.parameters()):
            raise AssertionError("frozen U-Net parameters must not have gradients")

    def forward(
        self,
        noisy: Tensor,
        context: Tensor,
        mask: Tensor,
        timesteps: Tensor,
        prompt_embeds: Tensor,
        defect_type: str,
    ) -> Tensor:
        self.assert_frozen_backbone()
        if not isinstance(noisy, Tensor) or noisy.ndim != 4 or noisy.shape[1:] != (4, 64, 64):
            raise ValueError("noisy must have shape [batch, 4, 64, 64]")
        if not noisy.is_floating_point() or not torch.isfinite(noisy).all().item():
            raise ValueError("noisy must contain only finite floating values")
        if context.shape != noisy.shape or context.device != noisy.device:
            raise ValueError("context shape and device must match noisy")
        if (
            not context.is_floating_point()
            or context.dtype != noisy.dtype
            or not torch.isfinite(context).all().item()
        ):
            raise ValueError(
                "context must contain only finite floating values matching noisy dtype"
            )
        if context.requires_grad:
            raise ValueError("context must be detached")
        if mask.shape != (noisy.shape[0], 1, 64, 64) or mask.device != noisy.device:
            raise ValueError("mask must match the noisy latent batch and device")
        if (
            not mask.is_floating_point()
            or not torch.isfinite(mask).all().item()
            or not torch.all((mask == 0) | (mask == 1)).item()
        ):
            raise ValueError("mask must contain only finite binary floating values")
        if mask.requires_grad:
            raise ValueError("mask must be detached")
        if timesteps.shape != (noisy.shape[0],) or timesteps.device != noisy.device:
            raise ValueError("timesteps must match the noisy latent batch and device")
        if timesteps.dtype is not torch.long:
            raise ValueError("timesteps must have dtype torch.long")
        if (
            prompt_embeds.shape[0] != noisy.shape[0]
            or prompt_embeds.ndim != 3
            or prompt_embeds.device != noisy.device
            or prompt_embeds.dtype != noisy.dtype
        ):
            raise ValueError("prompt_embeds must match noisy batch, device and dtype")
        if not torch.isfinite(prompt_embeds).all().item():
            raise ValueError("prompt_embeds must contain only finite values")
        if prompt_embeds.requires_grad:
            raise ValueError("prompt_embeds must be detached")
        if not isinstance(defect_type, str) or not defect_type:
            raise ValueError("defect_type must be a non-empty string")
        route_index = self.conditioner.vocabulary.index(defect_type)
        trainable = self.trainable_parameters()
        if not trainable or any(
            parameter.dtype is not torch.float32
            or parameter.device != noisy.device
            or not parameter.requires_grad
            for parameter in trainable
        ):
            raise ValueError("P2 trainable parameters must be FP32 masters on noisy device")
        embedding = self.conditioner(
            (defect_type,) * noisy.shape[0],
            device=noisy.device,
            dtype=torch.float32,
        )
        residuals = self.adapter(
            noisy.float(),
            context.float(),
            mask.float(),
            timesteps,
            embedding,
            route_index,
        )
        down = tuple(value.to(dtype=noisy.dtype) for value in residuals.down)
        mid = residuals.mid.to(dtype=noisy.dtype)
        output = self.unet(
            noisy,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
        )
        try:
            prediction = output.sample
        except AttributeError as error:
            raise ValueError("U-Net output must expose sample") from error
        if prediction.shape != noisy.shape or prediction.device != noisy.device:
            raise ValueError("U-Net prediction shape and device must match noisy")
        if prediction.dtype != noisy.dtype or not torch.isfinite(prediction).all().item():
            raise ValueError("U-Net prediction must be finite and match noisy dtype")
        return prediction

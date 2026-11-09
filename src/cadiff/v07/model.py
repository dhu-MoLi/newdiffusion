"""v07 noise predictor.

Supports both the original 4-channel SD1.5 UNet and a 9-channel SD inpainting
UNet. The inpainting variant receives ``masked_image_latents`` and concatenates
``(latent, mask, masked_image_latent)`` exactly like diffusers' inpainting
pipelines.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class V07InfillPredictor(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        conditioner: nn.Module,
        adapter: nn.Module,
        *,
        inpaint: bool,
        latent_channels: int = 4,
        latent_size: int = 64,
    ) -> None:
        super().__init__()
        if type(inpaint) is not bool:
            raise TypeError("inpaint must be a boolean")
        if type(latent_channels) is not int or latent_channels <= 0:
            raise ValueError("latent_channels must be a positive integer")
        if type(latent_size) is not int or latent_size <= 0:
            raise ValueError("latent_size must be a positive integer")
        self.unet = unet
        self.conditioner = conditioner
        self.adapter = adapter
        self.inpaint = inpaint
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.unet.requires_grad_(False)
        self.unet.eval()

    # ---- training-mode plumbing -------------------------------------------------
    def train(self, mode: bool = True) -> "V07InfillPredictor":
        if not isinstance(mode, bool):
            raise ValueError("training mode must be boolean")
        self.training = mode
        self.conditioner.train(mode)
        self.adapter.train(mode)
        self.unet.requires_grad_(False)
        self.unet.eval()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        seen: set[int] = set()
        parameters: list[nn.Parameter] = []
        for parameter in (*self.adapter.parameters(), *self.conditioner.parameters()):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
        return tuple(parameters)

    def gradient_parameters(self, defect_type: str) -> tuple[nn.Parameter, ...]:
        route_index = self.conditioner.vocabulary.index(defect_type)
        seen: set[int] = set()
        parameters: list[nn.Parameter] = []
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
            raise AssertionError("frozen backbone parameters must not require gradients")
        if any(module.training for module in self.unet.modules()):
            raise AssertionError("frozen backbone must stay in eval mode")
        if any(parameter.grad is not None for parameter in self.unet.parameters()):
            raise AssertionError("frozen backbone parameters must not receive gradients")

    # ---- forward ----------------------------------------------------------------
    def _validate(
        self,
        noisy: Tensor,
        context: Tensor,
        mask: Tensor,
        timesteps: Tensor,
        prompt_embeds: Tensor,
        masked_image_latents: Tensor | None,
    ) -> None:
        expected = (
            noisy.shape[0] if isinstance(noisy, Tensor) and noisy.ndim == 4 else -1,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        if not isinstance(noisy, Tensor) or tuple(noisy.shape) != expected:
            raise ValueError(f"noisy must have shape {expected}")
        if not noisy.is_floating_point() or not torch.isfinite(noisy).all():
            raise ValueError("noisy must be finite floating point")
        if tuple(context.shape) != tuple(noisy.shape) or context.device != noisy.device:
            raise ValueError("context must match noisy")
        if tuple(mask.shape) != (noisy.shape[0], 1, *noisy.shape[2:]):
            raise ValueError("mask must be N1HW matching noisy")
        if tuple(timesteps.shape) != (noisy.shape[0],) or timesteps.dtype is not torch.long:
            raise ValueError("timesteps must be a long tensor of shape [batch]")
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != noisy.shape[0]:
            raise ValueError("prompt_embeds must match the batch")
        if prompt_embeds.device != noisy.device:
            raise ValueError("prompt_embeds must live on the noisy device")
        if self.inpaint:
            if masked_image_latents is None:
                raise ValueError("inpainting predictor requires masked_image_latents")
            if tuple(masked_image_latents.shape) != tuple(noisy.shape):
                raise ValueError("masked_image_latents must match noisy")

    def forward(
        self,
        noisy: Tensor,
        context: Tensor,
        mask: Tensor,
        timesteps: Tensor,
        prompt_embeds: Tensor,
        defect_type: str,
        masked_image_latents: Tensor | None = None,
    ) -> Tensor:
        self.assert_frozen_backbone()
        if not isinstance(defect_type, str) or not defect_type:
            raise ValueError("defect_type must be a non-empty string")
        self._validate(noisy, context, mask, timesteps, prompt_embeds, masked_image_latents)
        route_index = self.conditioner.vocabulary.index(defect_type)
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
        if self.inpaint:
            assert masked_image_latents is not None
            model_input = torch.cat(
                (
                    noisy,
                    mask.to(dtype=noisy.dtype),
                    masked_image_latents.to(dtype=noisy.dtype),
                ),
                dim=1,
            )
        else:
            model_input = noisy
        output = self.unet(
            model_input,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
        )
        try:
            prediction = output.sample
        except AttributeError as error:  # pragma: no cover - defensive
            raise ValueError("U-Net output must expose sample") from error
        if tuple(prediction.shape) != tuple(noisy.shape) or prediction.device != noisy.device:
            raise ValueError("U-Net prediction must match noisy")
        if not torch.isfinite(prediction).all():
            raise ValueError("U-Net prediction must be finite")
        return prediction


__all__ = ["V07InfillPredictor"]

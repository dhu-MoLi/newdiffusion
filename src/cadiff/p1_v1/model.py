from __future__ import annotations

import torch
from torch import Tensor, nn

from cadiff.p1_v1.adapter import MultiScaleTypeAdapter, validate_sd15_unet_config
from cadiff.p1_v1.conditioning import DefectConditioner


class P1NoisePredictor(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        conditioner: DefectConditioner,
        adapter: MultiScaleTypeAdapter,
    ) -> None:
        super().__init__()
        try:
            unet_config = unet.config
        except AttributeError as error:
            raise ValueError("unet config is required") from error
        validate_sd15_unet_config(unet_config)

        self.unet = unet
        self.conditioner = conditioner
        self.adapter = adapter
        self._mid_block_channels = unet_config.block_out_channels[-1]
        self._mid_block_size = unet_config.sample_size // (
            2 ** (len(unet_config.block_out_channels) - 1)
        )
        self.unet.requires_grad_(False)
        self.unet.eval()

    def train(self, mode: bool = True) -> P1NoisePredictor:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.adapter.train(mode)
        self.conditioner.train(mode)
        self.unet.requires_grad_(False)
        self.unet.eval()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        unique_parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in (*self.adapter.parameters(), *self.conditioner.parameters()):
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                unique_parameters.append(parameter)
        return tuple(unique_parameters)

    def assert_frozen_backbone(self) -> None:
        if any(parameter.requires_grad for parameter in self.unet.parameters()):
            raise AssertionError("frozen U-Net parameters must have requires_grad=False")
        if any(module.training for module in self.unet.modules()):
            raise AssertionError("frozen U-Net and all descendants must remain in eval mode")

    def _validate_inputs(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        prompt_embeds: Tensor,
    ) -> Tensor:
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
        if not torch.isfinite(noisy_latent).all().item():
            raise ValueError("noisy_latent must contain only finite values")

        if not isinstance(timestep, Tensor):
            raise ValueError("timestep must be a tensor")
        if timestep.dtype is not torch.long:
            raise ValueError("timestep must have dtype torch.long")
        if timestep.ndim not in (0, 1):
            raise ValueError("timestep must be zero- or one-dimensional")
        if timestep.ndim == 1 and timestep.shape[0] != noisy_latent.shape[0]:
            raise ValueError("timestep batch must match noisy_latent")
        if timestep.device != noisy_latent.device:
            raise ValueError("timestep device must match noisy_latent")

        if not isinstance(prompt_embeds, Tensor):
            raise ValueError("prompt_embeds must be a tensor")
        if prompt_embeds.ndim != 3:
            raise ValueError("prompt_embeds must be three-dimensional")
        if prompt_embeds.shape[0] != noisy_latent.shape[0]:
            raise ValueError("prompt_embeds batch must match noisy_latent")
        if prompt_embeds.shape[1] <= 0 or prompt_embeds.shape[2] <= 0:
            raise ValueError("prompt_embeds sequence and width must be non-empty")
        if not prompt_embeds.is_floating_point():
            raise ValueError("prompt_embeds must have a floating dtype")
        if prompt_embeds.dtype != noisy_latent.dtype:
            raise ValueError("prompt_embeds dtype must match noisy_latent")
        if prompt_embeds.device != noisy_latent.device:
            raise ValueError("prompt_embeds device must match noisy_latent")
        if not torch.isfinite(prompt_embeds).all().item():
            raise ValueError("prompt_embeds must contain only finite values")

        if timestep.ndim == 0:
            return timestep.expand(noisy_latent.shape[0])
        return timestep

    def _validate_defect_type(self, defect_type: str) -> None:
        if not isinstance(defect_type, str) or not defect_type:
            raise ValueError("defect_type must be a single non-empty string")
        if defect_type != "normal":
            self.conditioner.vocabulary.index(defect_type)

    @staticmethod
    def _validate_fp32_trainable_module(
        module: object,
        *,
        name: str,
        target_device: torch.device,
    ) -> tuple[nn.Parameter, ...]:
        try:
            parameters = tuple(module.parameters())  # type: ignore[union-attr]
        except AttributeError as error:
            raise ValueError(f"{name} must expose parameters") from error
        if not parameters:
            raise ValueError(f"{name} must expose trainable parameters")
        for index, parameter in enumerate(parameters):
            if not isinstance(parameter, nn.Parameter):
                raise ValueError(f"{name} parameter at index {index} is invalid")
            if not parameter.is_floating_point():
                raise ValueError(
                    f"{name} parameter at index {index} must be floating"
                )
            if not parameter.requires_grad:
                raise ValueError(
                    f"{name} parameter at index {index} must require gradients"
                )
            if parameter.dtype is not torch.float32:
                raise ValueError(
                    f"{name} parameter at index {index} must have dtype torch.float32"
                )
            if parameter.device != target_device:
                raise ValueError(
                    f"{name} parameter at index {index} device must match "
                    "noisy_latent"
                )
        return parameters

    @staticmethod
    def _validate_prediction(prediction: object, noisy_latent: Tensor) -> Tensor:
        if not isinstance(prediction, Tensor):
            raise ValueError("U-Net output sample must be a tensor")
        if prediction.shape != noisy_latent.shape:
            raise ValueError("U-Net output shape must match noisy_latent")
        if prediction.dtype != noisy_latent.dtype:
            raise ValueError("U-Net output dtype must match noisy_latent")
        if prediction.device != noisy_latent.device:
            raise ValueError("U-Net output device must match noisy_latent")
        if not torch.isfinite(prediction).all().item():
            raise ValueError("U-Net output must contain only finite values")
        return prediction

    def forward(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        prompt_embeds: Tensor,
        defect_type: str,
    ) -> Tensor:
        batched_timesteps = self._validate_inputs(noisy_latent, timestep, prompt_embeds)
        self._validate_defect_type(defect_type)

        if defect_type == "normal":
            output = self.unet(
                noisy_latent,
                timestep,
                encoder_hidden_states=prompt_embeds,
            )
        else:
            self._validate_fp32_trainable_module(
                self.adapter,
                name="adapter",
                target_device=noisy_latent.device,
            )
            self._validate_fp32_trainable_module(
                self.conditioner,
                name="conditioner",
                target_device=noisy_latent.device,
            )
            adapter_latent = noisy_latent.to(dtype=torch.float32)
            embedding = self.conditioner(
                (defect_type,) * noisy_latent.shape[0],
                device=noisy_latent.device,
                dtype=torch.float32,
            )
            adapter_residuals = self.adapter(
                adapter_latent,
                batched_timesteps,
                embedding,
            )
            residuals = tuple(
                residual.to(
                    device=noisy_latent.device,
                    dtype=noisy_latent.dtype,
                )
                for residual in adapter_residuals
            )
            mid_residual = noisy_latent.new_zeros(
                (
                    noisy_latent.shape[0],
                    self._mid_block_channels,
                    self._mid_block_size,
                    self._mid_block_size,
                ),
                requires_grad=False,
            )
            output = self.unet(
                noisy_latent,
                timestep,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=residuals,
                mid_block_additional_residual=mid_residual,
            )

        try:
            prediction = output.sample
        except AttributeError as error:
            raise ValueError("U-Net output must expose a sample") from error
        return self._validate_prediction(prediction, noisy_latent)

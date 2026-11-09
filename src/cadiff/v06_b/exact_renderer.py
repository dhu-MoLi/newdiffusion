from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from torch import Tensor

from cadiff.diffusion.components import FrozenComponents
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_b0.generation import denoise_masked_infill

from .config import V06BMaskFirstConfig


def _normal_image(value: object, *, image_size: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (image_size, image_size, 3)
        or value.dtype != np.uint8
    ):
        raise ValueError("normal must be configured-size uint8 HxWx3")
    return np.ascontiguousarray(value)


def _realization_mask(value: object, *, image_size: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (image_size, image_size)
        or not np.isin(value, (0, 1)).all()
    ):
        raise ValueError("realization_mask must be a configured-size binary HxW array")
    mask = np.ascontiguousarray(value, dtype=bool)
    if not mask.any():
        raise ValueError("realization mask is empty")
    if mask.all():
        raise ValueError("realization mask covers the full image")
    return mask


def _vae_dtype(vae: object) -> torch.dtype:
    try:
        parameters = tuple(
            parameter
            for parameter in vae.parameters()  # type: ignore[attr-defined]
            if parameter.is_floating_point()
        )
    except AttributeError as error:
        raise ValueError("VAE must be a torch module with floating parameters") from error
    if not parameters:
        raise ValueError("VAE must expose floating parameters")
    return parameters[0].dtype


class ExactMaskRenderer:
    """Run one full-DDIM render conditioned only on the formal realization mask."""

    def __init__(
        self,
        *,
        components: FrozenComponents,
        model: P2InfillNoisePredictor,
        prompts: Mapping[str, Tensor],
        config: V06BMaskFirstConfig,
        device: torch.device,
        scheduler_factory: Callable[[object], Any] = DDIMScheduler.from_config,
    ) -> None:
        if (
            not isinstance(config, V06BMaskFirstConfig)
            or config.schema_version != "v06_b_mask_first_v1"
        ):
            raise ValueError("exact renderer requires the mask-first configuration")
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        if not callable(model):
            raise ValueError("model must be callable")
        if not callable(scheduler_factory):
            raise ValueError("scheduler_factory must be callable")
        try:
            vae = components.vae
            scheduler_config = components.scheduler.config
        except AttributeError as error:
            raise ValueError("components must expose VAE and scheduler configuration") from error
        if not isinstance(prompts, Mapping) or set(prompts) != set(config.defect_types):
            raise ValueError("prompts must contain the exact configured defect vocabulary")
        validated_prompts: dict[str, Tensor] = {}
        for defect_type in config.defect_types:
            prompt = prompts[defect_type]
            if (
                not isinstance(prompt, Tensor)
                or prompt.ndim != 3
                or prompt.shape[0] != 1
                or not prompt.is_floating_point()
                or not torch.isfinite(prompt).all().item()
                or prompt.requires_grad
            ):
                raise ValueError("prompt embeddings must be detached finite floating [1,S,D]")
            validated_prompts[defect_type] = prompt
        self.components = components
        self.model = model
        self.prompts = validated_prompts
        self.config = config
        self.device = device
        self.scheduler_factory = scheduler_factory
        self._scheduler_config = scheduler_config
        self._vae_dtype = _vae_dtype(vae)

    @torch.inference_mode()
    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        realization_mask: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        normal_array = _normal_image(normal, image_size=self.config.image_size)
        mask_array = _realization_mask(
            realization_mask,
            image_size=self.config.image_size,
        )
        if defect_type not in self.config.defect_types:
            raise ValueError("defect_type is outside the configured vocabulary")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        pixels = (
            torch.from_numpy(normal_array.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self._vae_dtype)
            .div(127.5)
            .sub(1.0)
        )
        normal_latent = encode_mode(self.components.vae, pixels).detach()
        source_mask = (
            torch.from_numpy(mask_array.copy())
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        latent_mask = build_latent_masks(
            source_mask,
            tuple(normal_latent.shape[2:]),
        ).binary
        context = build_masked_context(
            normal_latent,
            latent_mask,
            torch.Generator(device=self.device).manual_seed(seed + 10_000_000),
        )
        noise = torch.randn(
            normal_latent.shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=normal_latent.dtype,
        )
        prompt = self.prompts[defect_type]
        if prompt.device != self.device or prompt.dtype != normal_latent.dtype:
            raise ValueError("prompt must match renderer device and VAE latent dtype")
        generated_latent = denoise_masked_infill(
            model=self.model,
            scheduler=self.scheduler_factory(self._scheduler_config),
            normal_latent=normal_latent,
            context=context,
            binary_mask=latent_mask,
            prompt_embeds=prompt,
            defect_type=defect_type,
            noise=noise,
            steps=self.config.ddim_steps,
        )
        decoded = decode_latent(self.components.vae, generated_latent).float()
        raw = (
            decoded[0]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .copy()
        )
        if raw.shape != normal_array.shape:
            raise ValueError("decoded raw renderer output must match normal RGB geometry")
        return np.ascontiguousarray(raw)


__all__ = ["ExactMaskRenderer"]

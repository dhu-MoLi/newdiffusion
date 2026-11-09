"""v07 renderer: same masked-infill recipe, optional 9-channel inpainting input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from torch import Tensor

from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.v06_c.can_exact_mask_r import load_can_r_modules
from cadiff.v06_c.can_mvtec2 import IMAGE_SIZE
from cadiff.v06_c.visa_exact_mask_r import CHECKPOINT_SCHEMA
from cadiff.v07.backbones import BackboneSpec, load_backbone_components, make_backbone_spec
from cadiff.v07.model import V07InfillPredictor

DDIM_STEPS = 50


@torch.inference_mode()
def denoise_masked_infill_v07(
    *,
    model: Any,
    scheduler: Any,
    normal_latent: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    masked_image_latents: Tensor | None,
    prompt_embeds: Tensor,
    defect_type: str,
    noise: Tensor,
    steps: int,
) -> Tensor:
    if normal_latent.ndim != 4 or not normal_latent.is_floating_point():
        raise ValueError("normal_latent must be a floating four-dimensional tensor")
    for name, value in (("context", context), ("noise", noise)):
        if value.shape != normal_latent.shape or value.dtype != normal_latent.dtype:
            raise ValueError(f"{name} must match normal_latent")
    if tuple(binary_mask.shape) != (
        normal_latent.shape[0],
        1,
        normal_latent.shape[2],
        normal_latent.shape[3],
    ):
        raise ValueError("binary_mask must match normal_latent")
    if masked_image_latents is not None and masked_image_latents.shape != normal_latent.shape:
        raise ValueError("masked_image_latents must match normal_latent")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")

    scheduler.set_timesteps(steps, device=normal_latent.device)
    timesteps = tuple(scheduler.timesteps)
    if len(timesteps) != steps:
        raise ValueError("scheduler did not create the requested timestep count")
    current = scheduler.add_noise(normal_latent, noise, timesteps[0])
    mask = binary_mask.to(dtype=normal_latent.dtype)
    outside_mask = 1.0 - mask
    for index, timestep in enumerate(timesteps):
        timestep_batch = torch.full(
            (normal_latent.shape[0],),
            int(timestep.item()),
            device=normal_latent.device,
            dtype=torch.long,
        )
        prediction = model(
            current,
            context,
            binary_mask,
            timestep_batch,
            prompt_embeds,
            defect_type,
            masked_image_latents=masked_image_latents,
        )
        if prediction.shape != current.shape or not torch.isfinite(prediction).all():
            raise ValueError("model prediction must be finite and match the latent")
        generated = scheduler.step(prediction, timestep, current, eta=0.0).prev_sample
        if index + 1 < len(timesteps):
            reference = scheduler.add_noise(normal_latent, noise, timesteps[index + 1])
        else:
            reference = normal_latent
        current = generated * mask + reference * outside_mask
    if not torch.isfinite(current).all():
        raise ValueError("generated latent must be finite")
    return current.detach()


class V07ExactMaskRenderer:
    """50-step DDIM infill given a placed binary mask."""

    def __init__(
        self,
        *,
        components: Any,
        model: V07InfillPredictor,
        prompt: Tensor,
        device: torch.device,
        image_size: int = IMAGE_SIZE,
        ddim_steps: int = DDIM_STEPS,
        allowed_types: tuple[str, ...] | None = None,
        inpaint: bool = False,
    ) -> None:
        self.components = components
        self.model = model
        self.prompt = prompt
        self.device = device
        self.image_size = image_size
        self.ddim_steps = ddim_steps
        self.allowed_types = allowed_types
        self.inpaint = bool(inpaint)
        self._scheduler_config = components.scheduler.config
        self._vae_dtype = next(
            parameter
            for parameter in components.vae.parameters()
            if parameter.is_floating_point()
        ).dtype

    @torch.inference_mode()
    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        realization_mask: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        if self.allowed_types is not None and defect_type not in self.allowed_types:
            raise ValueError(f"renderer vocabulary does not include {defect_type}")
        if (
            not isinstance(normal, np.ndarray)
            or normal.shape != (self.image_size, self.image_size, 3)
            or normal.dtype != np.uint8
        ):
            raise ValueError(f"normal must be {self.image_size}x{self.image_size} uint8 RGB")
        mask = np.ascontiguousarray(realization_mask, dtype=bool)
        if mask.shape != (self.image_size, self.image_size) or not mask.any() or mask.all():
            raise ValueError("realization mask must be a non-empty, non-full binary")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        pixels = (
            torch.from_numpy(np.ascontiguousarray(normal))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self._vae_dtype)
            .div(127.5)
            .sub(1.0)
        )
        normal_latent = encode_mode(self.components.vae, pixels).detach()
        source_mask = (
            torch.from_numpy(mask.copy())
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        latent_mask = build_latent_masks(
            source_mask, tuple(normal_latent.shape[2:])
        ).binary
        context = build_masked_context(
            normal_latent,
            latent_mask,
            torch.Generator(device=self.device).manual_seed(seed + 10_000_000),
        )
        masked_image_latents = None
        if self.inpaint:
            masked_image_latents = normal_latent * (
                1.0 - latent_mask.to(dtype=normal_latent.dtype)
            )
        noise = torch.randn(
            normal_latent.shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=normal_latent.dtype,
        )
        generated_latent = denoise_masked_infill_v07(
            model=self.model,
            scheduler=DDIMScheduler.from_config(self._scheduler_config),
            normal_latent=normal_latent,
            context=context,
            binary_mask=latent_mask,
            masked_image_latents=masked_image_latents,
            prompt_embeds=self.prompt,
            defect_type=defect_type,
            noise=noise,
            steps=self.ddim_steps,
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
        return np.ascontiguousarray(raw)


def load_v07_renderer(
    *,
    checkpoint: Path,
    model_root: Path,
    config: Any,
    device: str | None = None,
    spec: BackboneSpec | None = None,
) -> V07ExactMaskRenderer:
    if spec is None:
        spec = make_backbone_spec("sd15", sd15_root=model_root)
    torch_device = torch.device(device or "cuda:0")
    components = load_backbone_components(spec, torch_device, torch.float16)
    _, conditioner, adapter, _metadata = load_can_r_modules(
        checkpoint,
        config=config,
        device=torch_device,
        expected_schema=CHECKPOINT_SCHEMA,
    )
    model = V07InfillPredictor(
        components.unet,
        conditioner,
        adapter,
        inpaint=spec.inpaint,
        latent_channels=spec.latent_channels,
        latent_size=spec.latent_size,
    ).eval()
    prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
    return V07ExactMaskRenderer(
        components=components,
        model=model,
        prompt=prompt,
        device=torch_device,
        allowed_types=config.defect_types,
        inpaint=spec.inpaint,
    )


__all__ = ["V07ExactMaskRenderer", "denoise_masked_infill_v07", "load_v07_renderer"]

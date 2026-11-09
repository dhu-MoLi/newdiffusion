from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class GenerationSlot:
    slot_id: str
    normal_id: str
    mask_id: str
    seed: int
    defect_type: str


@dataclass(frozen=True)
class MaskMetrics:
    inside_l1: float
    outside_l1: float
    outside_to_inside_ratio: float
    change_concentration: float


def _unique_strings(values: tuple[str, ...], name: str) -> None:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")


def build_generation_schedule(
    *,
    normal_ids: tuple[str, ...],
    mask_ids: tuple[str, ...],
    defect_types: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[GenerationSlot, ...]:
    _unique_strings(normal_ids, "normal_ids")
    _unique_strings(mask_ids, "mask_ids")
    _unique_strings(defect_types, "defect_types")
    if len(normal_ids) != len(mask_ids):
        raise ValueError("each normal reference must have exactly one mask")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain unique values")
    slots: list[GenerationSlot] = []
    for normal_id, mask_id in zip(normal_ids, mask_ids, strict=True):
        for seed in seeds:
            for defect_type in defect_types:
                identity = f"{normal_id}\n{mask_id}\n{seed}\n{defect_type}"
                slots.append(
                    GenerationSlot(
                        slot_id=hashlib.sha256(identity.encode()).hexdigest(),
                        normal_id=normal_id,
                        mask_id=mask_id,
                        seed=seed,
                        defect_type=defect_type,
                    )
                )
    return tuple(slots)


def _finite_image(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [batch, 3, height, width]")
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain finite floating values")
    return value


def compute_mask_metrics(generated: Tensor, normal: Tensor, source_mask: Tensor) -> MaskMetrics:
    generated = _finite_image(generated, "generated")
    normal = _finite_image(normal, "normal")
    if generated.shape != normal.shape:
        raise ValueError("generated and normal shapes must match")
    if (
        not isinstance(source_mask, Tensor)
        or source_mask.shape != (generated.shape[0], 1, generated.shape[2], generated.shape[3])
        or not source_mask.is_floating_point()
        or not torch.isfinite(source_mask).all().item()
        or not torch.all((source_mask == 0) | (source_mask == 1)).item()
    ):
        raise ValueError("source_mask must be a finite binary N1HW tensor")
    if source_mask.device != generated.device or normal.device != generated.device:
        raise ValueError("metric tensors must share one device")
    mask = source_mask.to(dtype=generated.dtype)
    outside = 1.0 - mask
    if (
        not torch.all(mask.flatten(1).any(dim=1)).item()
        or not torch.all(outside.flatten(1).any(dim=1)).item()
    ):
        raise ValueError("source_mask must contain inside and outside pixels")
    difference = (generated - normal).abs().mean(dim=1, keepdim=True)
    inside_l1 = (difference * mask).sum() / mask.sum()
    outside_l1 = (difference * outside).sum() / outside.sum()
    total_change = difference.sum()
    concentration = (difference * mask).sum() / total_change.clamp_min(1e-12)
    ratio = outside_l1 / inside_l1.clamp_min(1e-12)
    values = tuple(float(value.item()) for value in (inside_l1, outside_l1, ratio, concentration))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("mask metrics must be finite and non-negative")
    return MaskMetrics(*values)


def _validate_latent_inputs(
    normal_latent: object,
    context: object,
    binary_mask: object,
    prompt_embeds: object,
    noise: object,
    defect_type: object,
    steps: object,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str, int]:
    if not isinstance(normal_latent, Tensor) or normal_latent.ndim != 4:
        raise ValueError("normal_latent must be a four-dimensional tensor")
    if not normal_latent.is_floating_point() or not torch.isfinite(normal_latent).all().item():
        raise ValueError("normal_latent must contain finite floating values")
    if not isinstance(context, Tensor) or context.shape != normal_latent.shape:
        raise ValueError("context must match normal_latent")
    if not isinstance(noise, Tensor) or noise.shape != normal_latent.shape:
        raise ValueError("noise must match normal_latent")
    if (
        context.dtype != normal_latent.dtype
        or noise.dtype != normal_latent.dtype
        or context.device != normal_latent.device
        or noise.device != normal_latent.device
        or not torch.isfinite(context).all().item()
        or not torch.isfinite(noise).all().item()
    ):
        raise ValueError("context and noise must be finite and match normal_latent")
    if (
        not isinstance(binary_mask, Tensor)
        or binary_mask.shape
        != (
            normal_latent.shape[0],
            1,
            normal_latent.shape[2],
            normal_latent.shape[3],
        )
        or binary_mask.device != normal_latent.device
        or not binary_mask.is_floating_point()
        or not torch.isfinite(binary_mask).all().item()
        or not torch.all((binary_mask == 0) | (binary_mask == 1)).item()
    ):
        raise ValueError("binary_mask must be finite, binary and match normal_latent")
    if not isinstance(prompt_embeds, Tensor) or prompt_embeds.ndim != 3:
        raise ValueError("prompt_embeds must be a three-dimensional tensor")
    if (
        prompt_embeds.shape[0] != normal_latent.shape[0]
        or prompt_embeds.device != normal_latent.device
        or prompt_embeds.dtype != normal_latent.dtype
        or not torch.isfinite(prompt_embeds).all().item()
    ):
        raise ValueError("prompt_embeds must be finite and match normal_latent")
    if not isinstance(defect_type, str) or not defect_type:
        raise ValueError("defect_type must be a non-empty string")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    return (
        normal_latent,
        context,
        binary_mask,
        prompt_embeds,
        noise,
        defect_type,
        steps,
    )


@torch.inference_mode()
def denoise_masked_infill(
    *,
    model: Any,
    scheduler: Any,
    normal_latent: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    prompt_embeds: Tensor,
    defect_type: str,
    noise: Tensor,
    steps: int,
) -> Tensor:
    (
        normal_latent,
        context,
        binary_mask,
        prompt_embeds,
        noise,
        defect_type,
        steps,
    ) = _validate_latent_inputs(
        normal_latent,
        context,
        binary_mask,
        prompt_embeds,
        noise,
        defect_type,
        steps,
    )
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
        )
        if prediction.shape != current.shape or not torch.isfinite(prediction).all().item():
            raise ValueError("model prediction must be finite and match the latent")
        generated = scheduler.step(prediction, timestep, current, eta=0.0).prev_sample
        if index + 1 < len(timesteps):
            reference = scheduler.add_noise(normal_latent, noise, timesteps[index + 1])
        else:
            reference = normal_latent
        current = generated * mask + reference * outside_mask
    if not torch.isfinite(current).all().item():
        raise ValueError("generated latent must be finite")
    return current.detach()


__all__ = [
    "GenerationSlot",
    "MaskMetrics",
    "build_generation_schedule",
    "compute_mask_metrics",
    "denoise_masked_infill",
]

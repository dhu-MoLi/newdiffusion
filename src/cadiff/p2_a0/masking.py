from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LatentMasks:
    binary: Tensor
    soft: Tensor


def _validated_source_mask(mask: object) -> Tensor:
    if not isinstance(mask, Tensor):
        raise ValueError("mask must be a tensor")
    if mask.ndim != 4:
        raise ValueError("mask must have shape [batch, 1, height, width]")
    if mask.shape[0] <= 0 or mask.shape[2] <= 0 or mask.shape[3] <= 0:
        raise ValueError("mask must have shape [batch, 1, height, width]")
    if mask.shape[1] != 1:
        raise ValueError("mask must have one channel")
    if mask.dtype is not torch.bool and not mask.is_floating_point():
        raise ValueError("mask must be floating or boolean")
    if mask.is_floating_point() and not torch.isfinite(mask).all().item():
        raise ValueError("mask must contain only finite values")
    if not torch.all((mask == 0) | (mask == 1)).item():
        raise ValueError("mask must be binary")
    return mask.detach().to(dtype=torch.float32)


def build_latent_masks(mask: Tensor, latent_hw: tuple[int, int]) -> LatentMasks:
    source = _validated_source_mask(mask)
    if (
        not isinstance(latent_hw, tuple)
        or len(latent_hw) != 2
        or any(type(value) is not int or value <= 0 for value in latent_hw)
    ):
        raise ValueError("latent_hw must contain two positive integers")
    if latent_hw[0] > source.shape[2] or latent_hw[1] > source.shape[3]:
        raise ValueError("latent_hw must not exceed the source mask size")
    binary = torch.nn.functional.adaptive_max_pool2d(source, latent_hw)
    soft = torch.nn.functional.adaptive_avg_pool2d(source, latent_hw)
    return LatentMasks(binary=binary.detach(), soft=soft.detach())


def build_masked_context(
    target: Tensor,
    binary_mask: Tensor,
    generator: torch.Generator,
) -> Tensor:
    if not isinstance(target, Tensor) or target.ndim != 4 or target.shape[0] <= 0:
        raise ValueError("target must be a non-empty four-dimensional tensor")
    if not target.is_floating_point() or not torch.isfinite(target).all().item():
        raise ValueError("target must contain only finite floating values")
    if target.requires_grad:
        raise ValueError("target must not require gradients")
    mask = _validated_source_mask(binary_mask)
    if mask.shape[0] != target.shape[0] or mask.shape[2:] != target.shape[2:]:
        raise ValueError("binary mask batch and spatial shape must match target")
    flat = mask.flatten(1)
    if not torch.all(flat.any(dim=1) & (~flat.bool()).any(dim=1)).item():
        raise ValueError("binary mask must contain both inside and outside per sample")
    if mask.device != target.device:
        raise ValueError("binary mask device must match target")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch Generator")
    if torch.device(generator.device) != target.device:
        raise ValueError("generator device must match target")
    fill = torch.randn(
        target.shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    mask = mask.to(dtype=target.dtype)
    return (target.detach() * (1.0 - mask) + fill * mask).detach()

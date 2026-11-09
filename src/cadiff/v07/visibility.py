"""Visibility objectives.

The baseline renders at 512 and the downstream detector evaluates at 256.
Small defects lose contrast after the bilinear resize. These losses ask the
renderer to preserve the anomaly's low-frequency content, i.e. the part that
survives the evaluation resize.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_latent_pair(predicted: Tensor, target: Tensor, mask: Tensor) -> None:
    if not isinstance(predicted, Tensor) or not isinstance(target, Tensor):
        raise TypeError("predicted and target must be tensors")
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("predicted and target must share a four-dimensional shape")
    expected_mask = (predicted.shape[0], 1, predicted.shape[2], predicted.shape[3])
    if tuple(mask.shape) != expected_mask:
        raise ValueError("mask must be N1HW matching the latent")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise TypeError("latent tensors must be floating point")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("latent tensors must be finite")


def latent_lowfreq_loss(
    predicted_clean: Tensor,
    target_latent: Tensor,
    soft_mask: Tensor,
    *,
    factor: int = 2,
) -> Tensor:
    """L1 on average-pooled latents inside the mask.

    ``predicted_clean`` and ``target_latent`` are expected to be in the same
    latent space. ``factor=2`` targets the 2x downsample that turns 512 into
    256, which is exactly the evaluation regime.
    """

    if type(factor) is not int or factor < 1:
        raise ValueError("factor must be a positive integer")
    _validate_latent_pair(predicted_clean, target_latent, soft_mask)
    predicted = F.avg_pool2d(predicted_clean.float(), kernel_size=factor, stride=factor)
    target = F.avg_pool2d(target_latent.float(), kernel_size=factor, stride=factor)
    weights = F.avg_pool2d(soft_mask.float(), kernel_size=factor, stride=factor)
    weights = weights.clamp(0.0, 1.0)
    denominator = weights.sum()
    if float(denominator) <= 0.0:
        return predicted_clean.float().sum() * 0.0
    channels = predicted_clean.shape[1]
    return ((predicted - target).abs() * weights).sum() / (denominator * channels)


def rgb_lowfreq_loss(
    vae: torch.nn.Module,
    predicted_clean: Tensor,
    target_image: Tensor,
    image_mask: Tensor,
    *,
    size: int = 256,
) -> Tensor:
    """L1 in RGB at the evaluation resolution, inside the defect mask.

    ``target_image`` is the anomaly image in [-1, 1] with shape [1, 3, H, W].
    The gradient flows through the frozen VAE decoder into the predicted latent.
    """

    if type(size) is not int or size < 32:
        raise ValueError("size must be an integer >= 32")
    if target_image.ndim != 4 or target_image.shape[1] != 3:
        raise ValueError("target_image must be [1, 3, H, W]")
    if image_mask.ndim != 4 or image_mask.shape[1] != 1:
        raise ValueError("image_mask must be [1, 1, H, W]")
    parameter = next(vae.parameters())
    scaling = float(vae.config.scaling_factor)  # type: ignore[attr-defined]
    decoded = vae.decode(predicted_clean.to(dtype=parameter.dtype) / scaling).sample
    decoded = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    predicted = F.interpolate(decoded, size=(size, size), mode="bilinear", align_corners=False)
    target = F.interpolate(
        target_image.float() * 0.5 + 0.5,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    mask = F.interpolate(image_mask.float(), size=(size, size), mode="area").clamp(0.0, 1.0)
    if float(mask.sum()) <= 0.0:
        return predicted.sum() * 0.0
    difference = (predicted - target).abs().mean(dim=1, keepdim=True)
    return (difference * mask).sum() / mask.sum()


def build_visibility_loss(
    kind: str,
    *,
    vae: torch.nn.Module | None = None,
    target_image: Tensor | None = None,
    image_mask: Tensor | None = None,
    factor: int = 2,
    size: int = 256,
):
    """Return a callable ``(predicted_clean, target_latent, soft_mask) -> Tensor``."""

    if kind == "none":
        return None
    if kind == "latent":
        return lambda predicted, target, soft: latent_lowfreq_loss(
            predicted, target, soft, factor=factor
        )
    if kind == "rgb":
        if vae is None or target_image is None or image_mask is None:
            raise ValueError("rgb visibility requires vae, target_image and image_mask")
        return lambda predicted, target, soft: rgb_lowfreq_loss(
            vae,
            predicted,
            target_image,
            image_mask,
            size=size,
        )
    raise ValueError(f"unknown visibility kind: {kind}")


__all__ = [
    "build_visibility_loss",
    "latent_lowfreq_loss",
    "rgb_lowfreq_loss",
]

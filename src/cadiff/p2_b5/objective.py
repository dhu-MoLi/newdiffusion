"""Boundary-safe, type-selective decoded-image structure objective."""
# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

STRUCTURE_TYPES = ("cut", "hole", "print")
_ALL_TYPES = ("crack",) + STRUCTURE_TYPES


@dataclass(frozen=True)
class StructureLoss:
    total: torch.Tensor
    gradient: torch.Tensor
    laplacian: torch.Tensor
    high_frequency: torch.Tensor
    applied: bool


def _zero_like(target: torch.Tensor) -> StructureLoss:
    zero = target.new_zeros(())
    return StructureLoss(zero, zero, zero, zero, False)


def _validate_inputs(predicted_image: torch.Tensor, target_image: torch.Tensor, image_mask: torch.Tensor, defect_type: str) -> None:
    if defect_type not in _ALL_TYPES:
        raise ValueError(f"unsupported defect_type: {defect_type!r}")
    if predicted_image.ndim != 4 or predicted_image.shape != target_image.shape:
        raise ValueError("predicted_image and target_image must have identical [B,3,H,W] shapes")
    if predicted_image.shape[1] != 3:
        raise ValueError("decoded structure images must be RGB")
    if image_mask.ndim != 4 or image_mask.shape[:1] != predicted_image.shape[:1] or image_mask.shape[1] != 1 or image_mask.shape[-2:] != predicted_image.shape[-2:]:
        raise ValueError("image_mask must have shape [B,1,H,W] matching the images")
    if image_mask.requires_grad:
        raise ValueError("structure mask must not require gradients")
    if not torch.isfinite(predicted_image).all() or not torch.isfinite(target_image).all():
        raise ValueError("decoded structure images must be finite")
    if not torch.all((image_mask == 0) | (image_mask == 1)):
        raise ValueError("structure mask must be binary")


def _five_pixel_interior(mask: torch.Tensor) -> torch.Tensor:
    """Return centres whose 5x5 neighbourhood is wholly inside the mask."""

    kernel = torch.ones((1, 1, 5, 5), device=mask.device, dtype=mask.dtype)
    return functional.conv2d(mask, kernel, padding=2) == 25


def _features(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centre = image[..., 1:-1, 1:-1]
    horizontal = (image[..., 1:-1, 2:] - image[..., 1:-1, :-2]).abs().mul(0.5)
    vertical = (image[..., 2:, 1:-1] - image[..., :-2, 1:-1]).abs().mul(0.5)
    gradient = (horizontal + vertical).mul(0.5)
    laplacian = (4 * centre - image[..., 1:-1, 2:] - image[..., 1:-1, :-2] - image[..., 2:, 1:-1] - image[..., :-2, 1:-1]).abs()
    high_frequency = (centre - functional.avg_pool2d(image, kernel_size=3, stride=1)).abs()
    return gradient, laplacian, high_frequency


def _masked_l1(predicted: torch.Tensor, target: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    selected = domain[..., 1:-1, 1:-1].expand(-1, predicted.shape[1], -1, -1)
    return (predicted - target).abs()[selected].mean()


def masked_structure_loss(
    *,
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    image_mask: torch.Tensor,
    defect_type: str,
) -> StructureLoss:
    """Compare local RGB structure on a two-pixel-eroded true-pixel domain."""

    _validate_inputs(predicted_image, target_image, image_mask, defect_type)
    if defect_type == "crack":
        return _zero_like(target_image)
    domain = _five_pixel_interior(image_mask.to(dtype=predicted_image.dtype))
    domain_features = domain[..., 1:-1, 1:-1]
    if not bool(domain_features.any().item()):
        raise ValueError("structure domain is empty after full-neighbourhood erosion")
    predicted_features = _features(predicted_image)
    target_features = _features(target_image)
    gradient, laplacian, high_frequency = (
        _masked_l1(predicted, target, domain)
        for predicted, target in zip(predicted_features, target_features, strict=True)
    )
    total = (gradient + laplacian + high_frequency) / 3.0
    return StructureLoss(total, gradient, laplacian, high_frequency, True)


def compose_total(
    base_total: torch.Tensor,
    ranking_loss: torch.Tensor,
    ranking_weight: float,
    structure_total: torch.Tensor,
    structure_weight: float,
) -> torch.Tensor:
    """The exact frozen P2-B5 optimization formula."""

    return base_total + float(ranking_weight) * ranking_loss + float(structure_weight) * structure_total

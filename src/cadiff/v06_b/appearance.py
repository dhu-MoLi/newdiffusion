from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from cadiff.p2_a0.objective import predict_clean_latent

_CHARBONNIER_EPSILON = 1e-3


@dataclass(frozen=True)
class AppearanceSchedule:
    x0_weight: Tensor
    decode_active: Tensor


@dataclass(frozen=True)
class RegionBalancedDiffusionLoss:
    epsilon_gt: Tensor
    x0_gt: Tensor
    x0_ring: Tensor
    epsilon_outside: Tensor
    predicted_clean: Tensor
    snr_weight: Tensor


@dataclass(frozen=True)
class ExactMaskDiffusionLoss:
    epsilon_gt: Tensor
    x0_gt: Tensor
    predicted_clean: Tensor


@dataclass(frozen=True)
class DefectAppearanceLoss:
    rgb: Tensor
    sobel: Tensor


def _finite_positive(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not torch.isfinite(torch.tensor(float(value))).item()
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def appearance_schedule(
    scheduler: Any,
    timesteps: Tensor,
    *,
    minimum_snr: float,
) -> AppearanceSchedule:
    minimum = _finite_positive(minimum_snr, name="minimum_snr")
    if minimum > 1.0:
        raise ValueError("minimum_snr must be in (0, 1]")
    if (
        not isinstance(timesteps, Tensor)
        or timesteps.ndim != 1
        or timesteps.dtype is not torch.long
        or timesteps.requires_grad
    ):
        raise ValueError("timesteps must be a detached one-dimensional long tensor")
    try:
        alphas = scheduler.alphas_cumprod
    except AttributeError as error:
        raise ValueError("scheduler must expose alphas_cumprod") from error
    if not isinstance(alphas, Tensor) or alphas.ndim != 1 or not alphas.numel():
        raise ValueError("scheduler alphas_cumprod must be a non-empty tensor")
    if not torch.all((timesteps >= 0) & (timesteps < alphas.numel())).item():
        raise ValueError("timesteps are outside scheduler alphas_cumprod")
    alpha = alphas.to(device=timesteps.device, dtype=torch.float32)[timesteps]
    if not torch.all((alpha > 0.0) & (alpha <= 1.0)).item():
        raise ValueError("scheduler alphas_cumprod must lie in (0, 1]")
    snr = alpha / (1.0 - alpha).clamp_min(1e-12)
    weight = snr.sqrt().clamp(max=1.0)
    active = snr >= minimum
    if not torch.isfinite(weight).all().item():
        raise ValueError("appearance schedule must be finite")
    return AppearanceSchedule(x0_weight=weight, decode_active=active)


def _charbonnier(value: Tensor) -> Tensor:
    return (value.square() + _CHARBONNIER_EPSILON**2).sqrt() - _CHARBONNIER_EPSILON


def _region_mean(error: Tensor, region: Tensor, *, name: str) -> Tensor:
    if error.ndim != 4 or region.shape != (error.shape[0], 1, *error.shape[2:]):
        raise ValueError(f"{name} region must match error spatial dimensions")
    counts = region.sum(dim=(1, 2, 3))
    if not torch.all(counts > 0.0).item():
        raise ValueError(f"{name} region must be non-empty for every sample")
    channels = error.shape[1]
    return (error * region).sum(dim=(1, 2, 3)) / (counts * channels)


def region_balanced_diffusion_loss(
    *,
    scheduler: Any,
    noisy: Tensor,
    epsilon_hat: Tensor,
    epsilon_target: Tensor,
    target_latent: Tensor,
    soft_ground_truth: Tensor,
    soft_proposal: Tensor,
    timesteps: Tensor,
) -> RegionBalancedDiffusionLoss:
    tensors = (noisy, epsilon_hat, epsilon_target, target_latent)
    if (
        any(not isinstance(value, Tensor) or value.ndim != 4 for value in tensors)
        or any(value.shape != noisy.shape for value in tensors)
        or not noisy.is_floating_point()
        or any(value.device != noisy.device for value in tensors)
        or any(not torch.isfinite(value).all().item() for value in tensors)
    ):
        raise ValueError("diffusion tensors must be matching finite floating NCHW tensors")
    expected_mask = (noisy.shape[0], 1, *noisy.shape[2:])
    if (
        not isinstance(soft_ground_truth, Tensor)
        or not isinstance(soft_proposal, Tensor)
        or soft_ground_truth.shape != expected_mask
        or soft_proposal.shape != expected_mask
        or soft_ground_truth.device != noisy.device
        or soft_proposal.device != noisy.device
        or soft_ground_truth.requires_grad
        or soft_proposal.requires_grad
        or not torch.isfinite(soft_ground_truth).all().item()
        or not torch.isfinite(soft_proposal).all().item()
        or not torch.all((soft_ground_truth >= 0.0) & (soft_ground_truth <= 1.0)).item()
        or not torch.all((soft_proposal >= 0.0) & (soft_proposal <= 1.0)).item()
    ):
        raise ValueError("soft masks must be detached finite [B,1,H,W] tensors in [0,1]")
    if torch.any(soft_ground_truth > soft_proposal + 1e-6).item():
        raise ValueError("ground truth must be a subset of proposal")
    schedule = appearance_schedule(scheduler, timesteps, minimum_snr=1e-12)
    predicted_clean = predict_clean_latent(
        scheduler,
        noisy,
        epsilon_hat,
        timesteps,
    )
    ground_truth = soft_ground_truth.float()
    proposal = soft_proposal.float()
    ring = (proposal - ground_truth).clamp_min(0.0)
    outside = 1.0 - proposal
    squared_epsilon = (epsilon_hat.float() - epsilon_target.float()).square()
    latent_error = _charbonnier(predicted_clean - target_latent.float())
    epsilon_gt = _region_mean(squared_epsilon, ground_truth, name="ground_truth").mean()
    x0_gt = (
        _region_mean(latent_error, ground_truth, name="ground_truth")
        * schedule.x0_weight
    ).mean()
    x0_ring = (
        _region_mean(latent_error, ring, name="proposal ring") * schedule.x0_weight
    ).mean()
    epsilon_outside = _region_mean(
        squared_epsilon,
        outside,
        name="proposal outside",
    ).mean()
    values = (epsilon_gt, x0_gt, x0_ring, epsilon_outside, predicted_clean)
    if not all(torch.isfinite(value).all().item() for value in values):
        raise ValueError("region-balanced diffusion loss must be finite")
    return RegionBalancedDiffusionLoss(
        epsilon_gt=epsilon_gt,
        x0_gt=x0_gt,
        x0_ring=x0_ring,
        epsilon_outside=epsilon_outside,
        predicted_clean=predicted_clean,
        snr_weight=schedule.x0_weight,
    )


def exact_mask_diffusion_loss(
    *,
    scheduler: Any,
    noisy: Tensor,
    epsilon_hat: Tensor,
    epsilon_target: Tensor,
    target_latent: Tensor,
    soft_ground_truth: Tensor,
    timesteps: Tensor,
) -> ExactMaskDiffusionLoss:
    tensors = (noisy, epsilon_hat, epsilon_target, target_latent)
    if (
        any(not isinstance(value, Tensor) or value.ndim != 4 for value in tensors)
        or any(value.shape != noisy.shape for value in tensors)
        or not noisy.is_floating_point()
        or any(value.device != noisy.device for value in tensors)
        or any(not torch.isfinite(value).all().item() for value in tensors)
    ):
        raise ValueError("diffusion tensors must be matching finite floating NCHW tensors")
    expected_mask = (noisy.shape[0], 1, *noisy.shape[2:])
    if (
        not isinstance(soft_ground_truth, Tensor)
        or soft_ground_truth.shape != expected_mask
        or soft_ground_truth.device != noisy.device
        or soft_ground_truth.requires_grad
        or not torch.isfinite(soft_ground_truth).all().item()
        or not torch.all((soft_ground_truth >= 0.0) & (soft_ground_truth <= 1.0)).item()
    ):
        raise ValueError(
            "soft_ground_truth must be a detached finite matching mask in [0,1]"
        )
    schedule = appearance_schedule(scheduler, timesteps, minimum_snr=1e-12)
    predicted_clean = predict_clean_latent(
        scheduler,
        noisy,
        epsilon_hat,
        timesteps,
    )
    ground_truth = soft_ground_truth.float()
    epsilon_gt = _region_mean(
        (epsilon_hat.float() - epsilon_target.float()).square(),
        ground_truth,
        name="ground_truth",
    ).mean()
    x0_gt = (
        _region_mean(
            _charbonnier(predicted_clean - target_latent.float()),
            ground_truth,
            name="ground_truth",
        )
        * schedule.x0_weight
    ).mean()
    values = (epsilon_gt, x0_gt, predicted_clean)
    if not all(torch.isfinite(value).all().item() for value in values):
        raise ValueError("exact-mask diffusion loss must be finite")
    return ExactMaskDiffusionLoss(
        epsilon_gt=epsilon_gt,
        x0_gt=x0_gt,
        predicted_clean=predicted_clean,
    )


def decode_latent_for_training(vae: Any, latent: Tensor) -> Tensor:
    """Decode x0 without a hard RGB clamp while keeping the VAE frozen."""

    if not isinstance(vae, nn.Module):
        raise ValueError("vae must be a frozen torch module")
    if not isinstance(latent, Tensor) or latent.ndim != 4 or not latent.is_floating_point():
        raise ValueError("latent must be a four-dimensional floating tensor")
    if not torch.isfinite(latent).all().item():
        raise ValueError("latent must contain only finite values")
    parameters = tuple(
        parameter for parameter in vae.parameters() if parameter.is_floating_point()
    )
    if not parameters:
        raise ValueError("VAE must expose floating parameters")
    device = parameters[0].device
    dtype = parameters[0].dtype
    if any(
        parameter.device != device
        or parameter.dtype != dtype
        or parameter.requires_grad
        for parameter in parameters
    ):
        raise ValueError("VAE floating parameters must be frozen on one device and dtype")
    try:
        scaling = float(vae.config.scaling_factor)
        decoded = vae.decode((latent / scaling).to(device=device, dtype=dtype)).sample
    except AttributeError as error:
        raise ValueError("VAE decode must expose config.scaling_factor and sample") from error
    if not isinstance(decoded, Tensor) or decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError("VAE decode must return an RGB [B,3,H,W] tensor")
    image = decoded.float() / 2.0 + 0.5
    if not torch.isfinite(image).all().item():
        raise ValueError("decoded training image must be finite")
    return image


def _sobel(image: Tensor) -> Tensor:
    luminance = (
        image[:, 0:1] * 0.299 + image[:, 1:2] * 0.587 + image[:, 2:3] * 0.114
    )
    horizontal = image.new_tensor(
        ((1.0, 0.0, -1.0), (2.0, 0.0, -2.0), (1.0, 0.0, -1.0))
    ).view(1, 1, 3, 3) / 8.0
    vertical = horizontal.transpose(2, 3)
    return torch.cat(
        (
            functional.conv2d(luminance, horizontal, padding=1),
            functional.conv2d(luminance, vertical, padding=1),
        ),
        dim=1,
    )


def defect_appearance_loss(
    *,
    rendered: Tensor,
    target: Tensor,
    ground_truth: Tensor,
    sample_weight: Tensor,
) -> DefectAppearanceLoss:
    if (
        not isinstance(rendered, Tensor)
        or not isinstance(target, Tensor)
        or rendered.ndim != 4
        or rendered.shape != target.shape
        or rendered.shape[1] != 3
        or not rendered.is_floating_point()
        or not target.is_floating_point()
        or rendered.device != target.device
        or target.requires_grad
        or not torch.isfinite(rendered).all().item()
        or not torch.isfinite(target).all().item()
    ):
        raise ValueError("rendered and target must be matching finite RGB tensors")
    expected_mask = (rendered.shape[0], 1, *rendered.shape[2:])
    if (
        not isinstance(ground_truth, Tensor)
        or ground_truth.shape != expected_mask
        or ground_truth.device != rendered.device
        or ground_truth.requires_grad
        or not torch.isfinite(ground_truth).all().item()
        or not torch.all((ground_truth >= 0.0) & (ground_truth <= 1.0)).item()
    ):
        raise ValueError("ground_truth must be a detached matching mask in [0,1]")
    if (
        not isinstance(sample_weight, Tensor)
        or sample_weight.shape != (rendered.shape[0],)
        or sample_weight.device != rendered.device
        or sample_weight.requires_grad
        or not torch.isfinite(sample_weight).all().item()
        or torch.any(sample_weight < 0.0).item()
    ):
        raise ValueError("sample_weight must be a detached finite non-negative batch vector")
    mask = ground_truth.float()
    rgb_per_sample = _region_mean(
        _charbonnier(rendered - target.float()),
        mask,
        name="ground_truth",
    )
    sobel_per_sample = _region_mean(
        _charbonnier(
            _sobel(rendered * mask) - _sobel(target.float() * mask)
        ),
        mask,
        name="ground_truth",
    )
    rgb = (rgb_per_sample * sample_weight).mean()
    sobel = (sobel_per_sample * sample_weight).mean()
    if not torch.isfinite(rgb).all().item() or not torch.isfinite(sobel).all().item():
        raise ValueError("appearance loss must be finite")
    return DefectAppearanceLoss(rgb=rgb, sobel=sobel)


__all__ = [
    "AppearanceSchedule",
    "DefectAppearanceLoss",
    "ExactMaskDiffusionLoss",
    "RegionBalancedDiffusionLoss",
    "appearance_schedule",
    "decode_latent_for_training",
    "defect_appearance_loss",
    "exact_mask_diffusion_loss",
    "region_balanced_diffusion_loss",
]

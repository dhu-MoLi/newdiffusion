from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class InfillLoss:
    total: Tensor
    noise: Tensor
    inside: Tensor
    outside: Tensor
    predicted_clean: Tensor


def _finite_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[0] <= 0:
        raise ValueError(f"{name} must be a non-empty four-dimensional tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite floating values")
    return value


def _positive(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def predict_clean_latent(
    scheduler: object,
    noisy: Tensor,
    epsilon_hat: Tensor,
    timesteps: Tensor,
) -> Tensor:
    noisy = _finite_tensor(noisy, name="noisy")
    epsilon_hat = _finite_tensor(epsilon_hat, name="epsilon_hat")
    if epsilon_hat.shape != noisy.shape or epsilon_hat.device != noisy.device:
        raise ValueError("epsilon_hat shape and device must match noisy")
    if not isinstance(timesteps, Tensor) or timesteps.dtype is not torch.long:
        raise ValueError("timesteps must be a torch.long tensor")
    if timesteps.shape != (noisy.shape[0],) or timesteps.device != noisy.device:
        raise ValueError("timesteps batch and device must match noisy")
    try:
        alphas = scheduler.alphas_cumprod  # type: ignore[union-attr]
    except AttributeError as error:
        raise ValueError("scheduler alphas_cumprod is required") from error
    if not isinstance(alphas, Tensor) or alphas.ndim != 1 or alphas.numel() == 0:
        raise ValueError("scheduler alphas_cumprod must be a non-empty tensor")
    if not torch.isfinite(alphas).all().item():
        raise ValueError("scheduler alphas_cumprod must be finite")
    if not torch.all((timesteps >= 0) & (timesteps < alphas.numel())).item():
        raise ValueError("timesteps are outside scheduler alphas_cumprod")
    alpha = alphas.to(device=noisy.device, dtype=torch.float32)[timesteps]
    if not torch.all((alpha > 0) & (alpha <= 1)).item():
        raise ValueError("selected scheduler alphas must be in (0, 1]")
    alpha = alpha.reshape(-1, 1, 1, 1)
    return (noisy.float() - (1.0 - alpha).sqrt() * epsilon_hat.float()) / alpha.sqrt()


def masked_infill_loss(
    *,
    scheduler: object,
    noisy: Tensor,
    epsilon_hat: Tensor,
    epsilon_target: Tensor,
    target_latent: Tensor,
    soft_mask: Tensor,
    timesteps: Tensor,
    inside_noise_weight: float,
    lambda_inside: float,
    lambda_outside: float,
) -> InfillLoss:
    noisy = _finite_tensor(noisy, name="noisy")
    epsilon_hat = _finite_tensor(epsilon_hat, name="epsilon_hat")
    epsilon_target = _finite_tensor(epsilon_target, name="epsilon_target")
    target_latent = _finite_tensor(target_latent, name="target_latent")
    for name, value in (
        ("epsilon_hat", epsilon_hat),
        ("epsilon_target", epsilon_target),
        ("target_latent", target_latent),
    ):
        if value.shape != noisy.shape or value.device != noisy.device:
            raise ValueError(f"{name} shape and device must match noisy")
        if value.dtype != noisy.dtype:
            raise ValueError(f"{name} dtype must match noisy")
    for name, value in (
        ("epsilon_target", epsilon_target),
        ("target_latent", target_latent),
    ):
        if value.requires_grad:
            raise ValueError(f"{name} must not require gradients")
    soft_mask = _finite_tensor(soft_mask, name="soft_mask")
    if soft_mask.shape != (noisy.shape[0], 1, *noisy.shape[2:]):
        raise ValueError("soft_mask must have shape [batch, 1, height, width]")
    if soft_mask.device != noisy.device:
        raise ValueError("soft_mask device must match noisy")
    if soft_mask.requires_grad:
        raise ValueError("soft_mask must not require gradients")
    if not torch.all((soft_mask >= 0) & (soft_mask <= 1)).item():
        raise ValueError("soft_mask must be in [0, 1]")
    inside_count = soft_mask.float().sum(dim=(1, 2, 3))
    outside_count = (1.0 - soft_mask.float()).sum(dim=(1, 2, 3))
    if not torch.all(inside_count > 0).item():
        raise ValueError("inside region must be non-empty")
    if not torch.all(outside_count > 0).item():
        raise ValueError("outside region must be non-empty")

    inside_noise_weight = _positive(inside_noise_weight, name="inside_noise_weight")
    lambda_inside = _positive(lambda_inside, name="lambda_inside")
    lambda_outside = _positive(lambda_outside, name="lambda_outside")
    predicted_clean = predict_clean_latent(scheduler, noisy, epsilon_hat, timesteps)
    channels = noisy.shape[1]
    soft = soft_mask.float()
    noise_weights = 1.0 + inside_noise_weight * soft
    squared_error = (epsilon_hat.float() - epsilon_target.float()).square()
    noise_loss = (squared_error * noise_weights).sum() / (noise_weights.sum() * channels)
    latent_error = (predicted_clean - target_latent.float()).abs()
    inside_loss = (latent_error * soft).sum() / (soft.sum() * channels)
    outside = 1.0 - soft
    outside_loss = (latent_error * outside).sum() / (outside.sum() * channels)
    total = noise_loss + lambda_inside * inside_loss + lambda_outside * outside_loss
    if not all(
        torch.isfinite(value).all().item()
        for value in (total, noise_loss, inside_loss, outside_loss, predicted_clean)
    ):
        raise ValueError("masked infill objective produced a non-finite value")
    return InfillLoss(
        total=total,
        noise=noise_loss,
        inside=inside_loss,
        outside=outside_loss,
        predicted_clean=predicted_clean,
    )

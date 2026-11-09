from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cadiff.p2_a0.objective import InfillLoss, masked_infill_loss
from cadiff.p2_a1.config import P2A1Config

_MAX_AMP_BACKOFFS = 17  # Includes a retry at scale 1 from the default 65536.


@dataclass(frozen=True)
class RankingStepMetrics:
    total: float
    base_total: float
    noise: float
    inside: float
    outside: float
    ranking_loss: float
    correct_negative_ratio: float
    hardest_negative: str
    grad_l2: float
    finite: bool


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    entries = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if any(not isinstance(parameter, nn.Parameter) for parameter in entries):
        raise ValueError("every optimizer entry must be an nn.Parameter")
    return entries


def _external_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if value.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    return value


def _loss(
    *,
    scheduler: object,
    noisy: Tensor,
    prediction: Tensor,
    noise: Tensor,
    target_latent: Tensor,
    soft_mask: Tensor,
    timesteps: Tensor,
    config: P2A1Config,
) -> InfillLoss:
    return masked_infill_loss(
        scheduler=scheduler,
        noisy=noisy,
        epsilon_hat=prediction,
        epsilon_target=noise,
        target_latent=target_latent,
        soft_mask=soft_mask,
        timesteps=timesteps,
        inside_noise_weight=config.inside_noise_weight,
        lambda_inside=config.lambda_inside,
        lambda_outside=config.lambda_outside,
    )


def ranking_optimizer_step(
    *,
    model: Any,
    unet: Any,
    scheduler: object,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    target_latent: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    soft_mask: Tensor,
    prompt_embeds: Tensor,
    defect_type: str,
    timesteps: Tensor,
    noise: Tensor,
    config: P2A1Config,
    _amp_backoffs: int = 0,
) -> RankingStepMetrics:
    # Release each failed autograd graph before replaying the identical batch.
    if type(_amp_backoffs) is not int or not 0 <= _amp_backoffs < _MAX_AMP_BACKOFFS:
        raise ValueError("AMP backoff count is outside the bounded recovery budget")
    for attempt in range(_amp_backoffs, _MAX_AMP_BACKOFFS):
        metrics = _ranking_optimizer_attempt(
            model=model,
            unet=unet,
            scheduler=scheduler,
            optimizer=optimizer,
            scaler=scaler,
            target_latent=target_latent,
            context=context,
            binary_mask=binary_mask,
            soft_mask=soft_mask,
            prompt_embeds=prompt_embeds,
            defect_type=defect_type,
            timesteps=timesteps,
            noise=noise,
            config=config,
            _amp_backoffs=attempt,
        )
        if metrics is not None:
            return metrics
    raise RuntimeError("ranking AMP overflow recovery exhausted")


def _ranking_optimizer_attempt(
    *,
    model: Any,
    unet: Any,
    scheduler: object,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    target_latent: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    soft_mask: Tensor,
    prompt_embeds: Tensor,
    defect_type: str,
    timesteps: Tensor,
    noise: Tensor,
    config: P2A1Config,
    _amp_backoffs: int = 0,
) -> RankingStepMetrics | None:
    if type(_amp_backoffs) is not int or _amp_backoffs < 0:
        raise ValueError("AMP backoff count must be a non-negative integer")
    if not isinstance(config, P2A1Config):
        raise ValueError("config must be a P2A1Config")
    if defect_type not in config.defect_types:
        raise ValueError("defect_type must identify a configured type")
    target_latent = _external_tensor(target_latent, name="target_latent")
    context = _external_tensor(context, name="context")
    binary_mask = _external_tensor(binary_mask, name="binary_mask")
    soft_mask = _external_tensor(soft_mask, name="soft_mask")
    prompt_embeds = _external_tensor(prompt_embeds, name="prompt_embeds")
    noise = _external_tensor(noise, name="noise")
    if target_latent.ndim != 4 or target_latent.shape[0] <= 0:
        raise ValueError("target_latent must be a non-empty four-dimensional tensor")
    if context.shape != target_latent.shape or noise.shape != target_latent.shape:
        raise ValueError("context and noise must match target_latent")
    expected_mask = (target_latent.shape[0], 1, *target_latent.shape[2:])
    if binary_mask.shape != expected_mask or soft_mask.shape != expected_mask:
        raise ValueError("masks must match target_latent")
    if any(
        value.device != target_latent.device or value.dtype != target_latent.dtype
        for value in (context, noise)
    ):
        raise ValueError("context and noise must match target_latent device and dtype")
    if any(value.device != target_latent.device for value in (binary_mask, soft_mask)):
        raise ValueError("masks must match target_latent device")
    if (
        not isinstance(timesteps, Tensor)
        or timesteps.dtype is not torch.long
        or timesteps.shape != (target_latent.shape[0],)
        or timesteps.device != target_latent.device
    ):
        raise ValueError("timesteps must match target_latent batch and device")
    if timesteps.requires_grad:
        raise ValueError("timesteps must not require gradients")
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != target_latent.shape[0]:
        raise ValueError("prompt_embeds must match target_latent batch")
    if prompt_embeds.device != target_latent.device or prompt_embeds.dtype != target_latent.dtype:
        raise ValueError("prompt_embeds must match target_latent device and dtype")
    trainable = tuple(model.trainable_parameters())
    optimized = _optimizer_parameters(optimizer)
    if not trainable or {id(value) for value in trainable} != {
        id(value) for value in optimized
    } or len(trainable) != len(optimized):
        raise ValueError("optimizer must contain exactly model trainable_parameters")

    optimizer.zero_grad(set_to_none=True)
    noisy = scheduler.add_noise(target_latent, noise, timesteps)
    correct_prediction = model(
        noisy,
        context,
        binary_mask,
        timesteps,
        prompt_embeds,
        defect_type,
    )
    correct = _loss(
        scheduler=scheduler,
        noisy=noisy,
        prediction=correct_prediction,
        noise=noise,
        target_latent=target_latent,
        soft_mask=soft_mask,
        timesteps=timesteps,
        config=config,
    )
    candidates: list[tuple[str, Tensor]] = []
    with torch.no_grad():
        for label in config.defect_types:
            if label == defect_type:
                continue
            prediction = model(
                noisy,
                context,
                binary_mask,
                timesteps,
                prompt_embeds,
                label,
            )
            losses = _loss(
                scheduler=scheduler,
                noisy=noisy,
                prediction=prediction,
                noise=noise,
                target_latent=target_latent,
                soft_mask=soft_mask,
                timesteps=timesteps,
                config=config,
            )
            candidates.append((label, losses.total.detach()))
        normal_output = unet(
            noisy,
            timesteps,
            encoder_hidden_states=prompt_embeds,
        )
        try:
            normal_prediction = normal_output.sample
        except AttributeError as error:
            raise ValueError("normal U-Net output must expose sample") from error
        normal = _loss(
            scheduler=scheduler,
            noisy=noisy,
            prediction=normal_prediction,
            noise=noise,
            target_latent=target_latent,
            soft_mask=soft_mask,
            timesteps=timesteps,
            config=config,
        )
        candidates.append(("normal", normal.total.detach()))
    if len(candidates) != len(config.defect_types):
        raise ValueError("ranking candidates must include wrong types and normal")
    hardest_label, negative = min(
        candidates,
        key=lambda item: (float(item[1].item()), item[0]),
    )
    ratio = correct.total / negative.clamp_min(1e-12)
    ranking = torch.relu(ratio - config.ranking_target_ratio)
    backward_total = correct.total + config.ranking_weight * ranking
    if not all(
        torch.isfinite(value).all().item()
        for value in (correct.total, negative, ratio, ranking, backward_total)
    ):
        raise ValueError("ranking objective produced a non-finite value")

    scaler.scale(backward_total).backward()
    scaler.unscale_(optimizer)
    required = tuple(model.gradient_parameters(defect_type))
    trainable_ids = {id(parameter) for parameter in trainable}
    required_ids = {id(parameter) for parameter in required}
    if (
        not required
        or len(required_ids) != len(required)
        or not required_ids.issubset(trainable_ids)
    ):
        raise ValueError("gradient_parameters must be a unique subset of trainables")
    gradients = tuple(parameter.grad for parameter in required)
    if any(gradient is None for gradient in gradients):
        raise ValueError("every active trainable parameter must receive a gradient")
    inactive = tuple(
        parameter for parameter in trainable if id(parameter) not in required_ids
    )
    if any(parameter.grad is not None for parameter in inactive):
        raise ValueError("inactive routed parameters must not receive gradients")
    concrete = tuple(gradient for gradient in gradients if gradient is not None)
    if any(not torch.isfinite(gradient).all().item() for gradient in concrete):
        if not scaler.is_enabled():
            raise ValueError("trainable gradients must be finite")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if not scale_after < scale_before:
            raise RuntimeError("ranking AMP overflow did not reduce loss scale")
        if _amp_backoffs >= _MAX_AMP_BACKOFFS - 1:
            raise RuntimeError("ranking AMP overflow recovery exhausted")
        return None
    grad_l2 = math.sqrt(
        sum(float(gradient.float().square().sum().item()) for gradient in concrete)
    )
    if not math.isfinite(grad_l2) or grad_l2 <= 0.0:
        raise ValueError("at least one trainable gradient must be nonzero and finite")
    scaler.step(optimizer)
    scaler.update()
    if any(not torch.isfinite(parameter).all().item() for parameter in trainable):
        raise ValueError("trainable parameters must remain finite")
    model.assert_frozen_backbone()
    if isinstance(unet, nn.Module) and any(
        parameter.grad is not None for parameter in unet.parameters()
    ):
        raise ValueError("frozen normal U-Net must not receive gradients")
    values = {
        "total": backward_total,
        "base_total": correct.total,
        "noise": correct.noise,
        "inside": correct.inside,
        "outside": correct.outside,
        "ranking_loss": ranking,
        "correct_negative_ratio": ratio,
    }
    return RankingStepMetrics(
        **{name: float(value.detach().item()) for name, value in values.items()},
        hardest_negative=hardest_label,
        grad_l2=grad_l2,
        finite=all(torch.isfinite(value).all().item() for value in values.values()),
    )

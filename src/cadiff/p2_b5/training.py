"""One optimizer step for P2-B5, preserving the P2-A1 ranking objective."""
# ruff: noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cadiff.p2_a0.objective import masked_infill_loss
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_a1.ranking import _external_tensor, _optimizer_parameters

from .config import P2B5Config
from .objective import compose_total, masked_structure_loss


@dataclass(frozen=True)
class TypeSelectiveStepMetrics:
    total: float
    base_total: float
    noise: float
    inside: float
    outside: float
    ranking_loss: float
    correct_negative_ratio: float
    hardest_negative: str
    structure_total: float
    structure_gradient: float
    structure_laplacian: float
    structure_high_frequency: float
    structure_applied: bool
    structure_weight: float
    grad_l2: float
    finite: bool


def validate_type_selective_record(record: dict[str, object], *, defect_type: str, source_config: P2A1Config, config: P2B5Config) -> None:
    """Independently validate the persisted structure-selection and total formula."""

    required = {
        "total", "base_total", "ranking_loss", "structure_total", "structure_gradient",
        "structure_laplacian", "structure_high_frequency", "structure_applied", "structure_weight",
    }
    if not required.issubset(record):
        raise ValueError("training record omits P2-B5 structure metrics")
    should_apply = defect_type in config.structure_types
    if record["structure_applied"] is not should_apply:
        raise ValueError("structure selection does not match the frozen defect types")
    numeric = tuple(record[name] for name in required - {"structure_applied"})
    if any(type(value) is not float or not math.isfinite(value) for value in numeric):
        raise ValueError("P2-B5 structure metrics must be finite floats")
    if record["structure_weight"] != config.structure_weight:
        raise ValueError("structure_weight does not match the frozen configuration")
    if not should_apply and any(record[name] != 0.0 for name in ("structure_total", "structure_gradient", "structure_laplacian", "structure_high_frequency")):
        raise ValueError("crack must have exact zero structure metrics")
    expected = float(record["base_total"]) + source_config.ranking_weight * float(record["ranking_loss"]) + config.structure_weight * float(record["structure_total"])
    if not math.isclose(float(record["total"]), expected, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("training record violates the P2-B5 total formula")


def _loss(*, scheduler: object, noisy: Tensor, prediction: Tensor, noise: Tensor, target_latent: Tensor, soft_mask: Tensor, timesteps: Tensor, config: P2A1Config):
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


def _decode_predicted_x0(vae: Any, latent: Tensor) -> Tensor:
    """Decode predicted x0 while retaining gradients only to the latent input."""

    if not isinstance(vae, nn.Module):
        raise ValueError("VAE must be a frozen nn.Module")
    parameters = tuple(
        parameter for parameter in vae.parameters() if parameter.is_floating_point()
    )
    if not parameters:
        raise ValueError("VAE must expose floating parameters")
    vae_device, vae_dtype = parameters[0].device, parameters[0].dtype
    if any(
        parameter.device != vae_device
        or parameter.dtype != vae_dtype
        or parameter.requires_grad
        for parameter in parameters
    ):
        raise ValueError("VAE floating parameters must be frozen on one device and dtype")
    try:
        scaling = float(vae.config.scaling_factor)
        vae_latent = (latent / scaling).to(device=vae_device, dtype=vae_dtype)
        decoded = vae.decode(vae_latent).sample
    except AttributeError as error:
        raise ValueError("VAE decode must expose config.scaling_factor and sample") from error
    if not isinstance(decoded, Tensor) or decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError("VAE decode must return an RGB NCHW tensor")
    image = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    if not torch.isfinite(image).all().item():
        raise ValueError("decoded predicted-x0 image must be finite")
    return image


def type_selective_optimizer_step(
    *,
    model: Any,
    unet: Any,
    vae: Any,
    scheduler: object,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    target_latent: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    soft_mask: Tensor,
    target_image: Tensor,
    image_mask: Tensor,
    prompt_embeds: Tensor,
    defect_type: str,
    timesteps: Tensor,
    noise: Tensor,
    source_config: P2A1Config,
    config: P2B5Config,
) -> TypeSelectiveStepMetrics:
    """Run the frozen base/ranking step plus the selected-type structure term."""

    if not isinstance(source_config, P2A1Config) or not isinstance(config, P2B5Config):
        raise ValueError("source_config and config must have their frozen config types")
    if defect_type not in config.defect_types or tuple(config.defect_types) != source_config.defect_types:
        raise ValueError("defect_type must identify the frozen shared vocabulary")
    target_latent = _external_tensor(target_latent, name="target_latent")
    context = _external_tensor(context, name="context")
    binary_mask = _external_tensor(binary_mask, name="binary_mask")
    soft_mask = _external_tensor(soft_mask, name="soft_mask")
    target_image = _external_tensor(target_image, name="target_image")
    image_mask = _external_tensor(image_mask, name="image_mask")
    prompt_embeds = _external_tensor(prompt_embeds, name="prompt_embeds")
    noise = _external_tensor(noise, name="noise")
    if target_latent.ndim != 4 or target_latent.shape[0] <= 0:
        raise ValueError("target_latent must be a non-empty four-dimensional tensor")
    if context.shape != target_latent.shape or noise.shape != target_latent.shape:
        raise ValueError("context and noise must match target_latent")
    expected_mask = (target_latent.shape[0], 1, *target_latent.shape[2:])
    if binary_mask.shape != expected_mask or soft_mask.shape != expected_mask:
        raise ValueError("masks must match target_latent")
    if any(value.device != target_latent.device or value.dtype != target_latent.dtype for value in (context, noise)):
        raise ValueError("context and noise must match target_latent device and dtype")
    if any(value.device != target_latent.device for value in (binary_mask, soft_mask)):
        raise ValueError("masks must match target_latent device")
    if target_image.ndim != 4 or target_image.shape[0] != target_latent.shape[0] or target_image.shape[1] != 3:
        raise ValueError("target_image must be an RGB NCHW tensor matching the batch")
    if image_mask.shape != (target_latent.shape[0], 1, *target_image.shape[-2:]):
        raise ValueError("image_mask must match target_image batch and spatial shape")
    if target_image.device != target_latent.device or image_mask.device != target_latent.device:
        raise ValueError("target_image and image_mask must match target_latent device")
    if not isinstance(timesteps, Tensor) or timesteps.dtype is not torch.long or timesteps.shape != (target_latent.shape[0],) or timesteps.device != target_latent.device or timesteps.requires_grad:
        raise ValueError("timesteps must match target_latent batch and device without gradients")
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != target_latent.shape[0] or prompt_embeds.device != target_latent.device or prompt_embeds.dtype != target_latent.dtype:
        raise ValueError("prompt_embeds must match target_latent batch, device, and dtype")
    trainable = tuple(model.trainable_parameters())
    optimized = _optimizer_parameters(optimizer)
    if not trainable or {id(value) for value in trainable} != {id(value) for value in optimized} or len(trainable) != len(optimized):
        raise ValueError("optimizer must contain exactly model trainable_parameters")

    optimizer.zero_grad(set_to_none=True)
    noisy = scheduler.add_noise(target_latent, noise, timesteps)
    correct_prediction = model(noisy, context, binary_mask, timesteps, prompt_embeds, defect_type)
    correct = _loss(scheduler=scheduler, noisy=noisy, prediction=correct_prediction, noise=noise, target_latent=target_latent, soft_mask=soft_mask, timesteps=timesteps, config=source_config)
    if defect_type == "crack":
        structure = masked_structure_loss(
            predicted_image=target_image,
            target_image=target_image,
            image_mask=image_mask,
            defect_type=defect_type,
        )
    else:
        predicted_image = _decode_predicted_x0(vae, correct.predicted_clean)
        structure = masked_structure_loss(
            predicted_image=predicted_image,
            target_image=target_image.float(),
            image_mask=image_mask,
            defect_type=defect_type,
        )
    candidates: list[tuple[str, Tensor]] = []
    with torch.no_grad():
        for label in source_config.defect_types:
            if label == defect_type:
                continue
            prediction = model(noisy, context, binary_mask, timesteps, prompt_embeds, label)
            loss = _loss(scheduler=scheduler, noisy=noisy, prediction=prediction, noise=noise, target_latent=target_latent, soft_mask=soft_mask, timesteps=timesteps, config=source_config)
            candidates.append((label, loss.total.detach()))
        normal_output = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds)
        try:
            normal_prediction = normal_output.sample
        except AttributeError as error:
            raise ValueError("normal U-Net output must expose sample") from error
        normal = _loss(scheduler=scheduler, noisy=noisy, prediction=normal_prediction, noise=noise, target_latent=target_latent, soft_mask=soft_mask, timesteps=timesteps, config=source_config)
        candidates.append(("normal", normal.total.detach()))
    if len(candidates) != len(source_config.defect_types):
        raise ValueError("ranking candidates must include wrong types and normal")
    hardest_label, negative = min(candidates, key=lambda item: (float(item[1].item()), item[0]))
    ratio = correct.total / negative.clamp_min(1e-12)
    ranking = torch.relu(ratio - source_config.ranking_target_ratio)
    backward_total = compose_total(correct.total, ranking, source_config.ranking_weight, structure.total, config.structure_weight)
    values = (correct.total, negative, ratio, ranking, structure.total, structure.gradient, structure.laplacian, structure.high_frequency, backward_total)
    if not all(torch.isfinite(value).all().item() for value in values):
        raise ValueError("P2-B5 objective produced a non-finite value")

    scaler.scale(backward_total).backward()
    scaler.unscale_(optimizer)
    required = tuple(model.gradient_parameters(defect_type))
    trainable_ids = {id(parameter) for parameter in trainable}
    required_ids = {id(parameter) for parameter in required}
    if not required or len(required_ids) != len(required) or not required_ids.issubset(trainable_ids):
        raise ValueError("gradient_parameters must be a unique subset of trainables")
    gradients = tuple(parameter.grad for parameter in required)
    if any(gradient is None for gradient in gradients):
        raise ValueError("every active trainable parameter must receive a gradient")
    inactive = tuple(parameter for parameter in trainable if id(parameter) not in required_ids)
    if any(parameter.grad is not None for parameter in inactive):
        raise ValueError("inactive routed parameters must not receive gradients")
    concrete = tuple(gradient for gradient in gradients if gradient is not None)
    if any(not torch.isfinite(gradient).all().item() for gradient in concrete):
        raise ValueError("trainable gradients must be finite")
    grad_l2 = math.sqrt(sum(float(gradient.float().square().sum().item()) for gradient in concrete))
    if not math.isfinite(grad_l2) or grad_l2 <= 0.0:
        raise ValueError("at least one trainable gradient must be nonzero and finite")
    scaler.step(optimizer)
    scaler.update()
    if any(not torch.isfinite(parameter).all().item() for parameter in trainable):
        raise ValueError("trainable parameters must remain finite")
    model.assert_frozen_backbone()
    if isinstance(unet, nn.Module) and any(parameter.grad is not None for parameter in unet.parameters()):
        raise ValueError("frozen normal U-Net must not receive gradients")
    metric_values = {
        "total": backward_total,
        "base_total": correct.total,
        "noise": correct.noise,
        "inside": correct.inside,
        "outside": correct.outside,
        "ranking_loss": ranking,
        "correct_negative_ratio": ratio,
        "structure_total": structure.total,
        "structure_gradient": structure.gradient,
        "structure_laplacian": structure.laplacian,
        "structure_high_frequency": structure.high_frequency,
    }
    return TypeSelectiveStepMetrics(
        **{name: float(value.detach().item()) for name, value in metric_values.items()},
        hardest_negative=hardest_label,
        structure_applied=structure.applied,
        structure_weight=config.structure_weight,
        grad_l2=grad_l2,
        finite=all(torch.isfinite(value).all().item() for value in metric_values.values()),
    )

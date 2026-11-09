from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cadiff.p2_a0.model import P2InfillNoisePredictor

from .appearance import (
    DefectAppearanceLoss,
    appearance_schedule,
    decode_latent_for_training,
    defect_appearance_loss,
    exact_mask_diffusion_loss,
)
from .config import V06BMaskFirstConfig

_MAX_AMP_BACKOFFS = 17  # Includes a retry at scale 1 from the default 65536.


@dataclass(frozen=True)
class ExactMaskRendererBatch:
    target_latent: Tensor
    target_image: Tensor
    context: Tensor
    binary_gt_mask: Tensor
    soft_gt_mask: Tensor
    image_gt_mask: Tensor
    prompt_embeds: Tensor
    wrong_prompt_embeds: Tensor
    defect_type: str
    wrong_defect_type: str
    timesteps: Tensor
    noise: Tensor


@dataclass(frozen=True)
class ExactRendererStepMetrics:
    total: float
    epsilon_gt: float
    x0_gt: float
    rgb_gt: float
    edge_gt: float
    type_margin: float
    wrong_semantic: float
    grad_l2: float
    finite: bool


def _frozen_module(module: object, *, name: str) -> tuple[nn.Parameter, ...]:
    if not isinstance(module, nn.Module):
        raise ValueError(f"{name} must be a frozen torch module")
    parameters = tuple(module.parameters())
    if (
        not parameters
        or any(parameter.requires_grad or parameter.grad is not None for parameter in parameters)
        or any(descendant.training for descendant in module.modules())
    ):
        raise ValueError(f"{name} must remain eval, frozen, and gradient-free")
    return parameters


def _model_trainables(model: Any) -> tuple[nn.Parameter, ...]:
    try:
        trainable = tuple(model.trainable_parameters())
        named = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    except AttributeError as error:
        raise ValueError("model must expose named and trainable parameters") from error
    if (
        not trainable
        or any(not isinstance(parameter, nn.Parameter) for parameter in trainable)
        or len({id(parameter) for parameter in trainable}) != len(trainable)
        or {id(parameter) for parameter in trainable} != {id(parameter) for _, parameter in named}
    ):
        raise ValueError("model trainables must be unique, complete Parameters")
    return trainable


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, nn.Parameter)
    )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("optimizer contains duplicate parameters")
    return parameters


def build_exact_renderer_optimizer(
    *,
    model: P2InfillNoisePredictor,
    vae: nn.Module,
    text_encoder: nn.Module,
    config: V06BMaskFirstConfig,
) -> torch.optim.Optimizer:
    if (
        not isinstance(config, V06BMaskFirstConfig)
        or config.schema_version != "v06_b_mask_first_v1"
    ):
        raise ValueError("exact renderer optimizer requires the mask-first configuration")
    try:
        unet = model.unet
    except AttributeError as error:
        raise ValueError("model must expose its frozen U-Net") from error
    frozen_parameters = (
        *_frozen_module(unet, name="U-Net"),
        *_frozen_module(vae, name="VAE"),
        *_frozen_module(text_encoder, name="text encoder"),
    )
    trainable = _model_trainables(model)
    if {id(parameter) for parameter in frozen_parameters} & {
        id(parameter) for parameter in trainable
    }:
        raise ValueError("frozen parameters must not enter renderer optimization")
    named = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if any(not name.startswith(("adapter.", "conditioner.")) for name, _ in named):
        raise ValueError("only Adapter/Conditioner trainables may be optimized")
    type_condition = tuple(
        parameter
        for name, parameter in named
        if name.startswith("conditioner.") or name.startswith("adapter.delta_")
    )
    type_ids = {id(parameter) for parameter in type_condition}
    shared = tuple(parameter for parameter in trainable if id(parameter) not in type_ids)
    if (
        not shared
        or not type_condition
        or len(type_ids) != len(type_condition)
        or len(shared) + len(type_condition) != len(trainable)
    ):
        raise ValueError("renderer optimizer groups must be non-empty, disjoint, and complete")
    rates = config.renderer_learning_rates
    return torch.optim.AdamW(
        (
            {
                "name": "shared_renderer",
                "params": shared,
                "lr": rates["shared_renderer"],
            },
            {
                "name": "type_condition",
                "params": type_condition,
                "lr": rates["type_condition"],
            },
        ),
        weight_decay=0.0,
    )


def _validate_batch(
    batch: ExactMaskRendererBatch,
    config: V06BMaskFirstConfig,
) -> None:
    if not isinstance(batch, ExactMaskRendererBatch):
        raise ValueError("batch must be an ExactMaskRendererBatch")
    if batch.defect_type not in config.defect_types:
        raise ValueError("defect type is outside the Module-B vocabulary")
    expected_wrong = config.defect_types[
        (config.defect_types.index(batch.defect_type) + 1) % len(config.defect_types)
    ]
    if batch.wrong_defect_type != expected_wrong:
        raise ValueError("wrong_defect_type must be the fixed cyclic successor")
    if (
        batch.target_latent.ndim != 4
        or batch.context.shape != batch.target_latent.shape
        or batch.noise.shape != batch.target_latent.shape
        or batch.timesteps.shape != (batch.target_latent.shape[0],)
        or batch.timesteps.dtype is not torch.long
    ):
        raise ValueError("exact renderer latent tensors have incompatible shapes")
    latent_mask_shape = (
        batch.target_latent.shape[0],
        1,
        *batch.target_latent.shape[2:],
    )
    if batch.binary_gt_mask.shape != latent_mask_shape or batch.soft_gt_mask.shape != (
        latent_mask_shape
    ):
        raise ValueError("exact renderer latent masks have incompatible shapes")
    image_mask_shape = (
        batch.target_latent.shape[0],
        1,
        *batch.target_image.shape[2:],
    )
    if (
        batch.target_image.ndim != 4
        or batch.target_image.shape[1] != 3
        or batch.image_gt_mask.shape != image_mask_shape
    ):
        raise ValueError("exact renderer image tensors have incompatible shapes")
    if (
        not torch.all((batch.binary_gt_mask == 0) | (batch.binary_gt_mask == 1)).item()
        or not torch.all((batch.image_gt_mask == 0) | (batch.image_gt_mask == 1)).item()
    ):
        raise ValueError("exact GT binary masks must be binary")
    if (
        not torch.all((batch.soft_gt_mask >= 0.0) & (batch.soft_gt_mask <= 1.0)).item()
        or torch.any(batch.soft_gt_mask > batch.binary_gt_mask + 1e-6).item()
    ):
        raise ValueError("soft exact GT mask must lie inside the binary exact GT mask")
    ordinary = (
        batch.target_latent,
        batch.target_image,
        batch.context,
        batch.binary_gt_mask,
        batch.soft_gt_mask,
        batch.image_gt_mask,
        batch.prompt_embeds,
        batch.wrong_prompt_embeds,
        batch.noise,
    )
    if any(
        not value.is_floating_point()
        or value.requires_grad
        or not torch.isfinite(value).all().item()
        for value in ordinary
    ):
        raise ValueError("exact renderer inputs must be detached finite floating tensors")
    device = batch.target_latent.device
    if any(value.device != device for value in ordinary) or batch.timesteps.device != device:
        raise ValueError("exact renderer inputs must share one device")


def _zero_from(value: Tensor) -> Tensor:
    return value.sum() * 0.0


def exact_renderer_optimizer_step(
    *,
    model: P2InfillNoisePredictor,
    vae: nn.Module,
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: ExactMaskRendererBatch,
    config: V06BMaskFirstConfig,
    _amp_backoffs: int = 0,
) -> ExactRendererStepMetrics:
    # Release each failed autograd graph before replaying the identical batch.
    if type(_amp_backoffs) is not int or not 0 <= _amp_backoffs < _MAX_AMP_BACKOFFS:
        raise ValueError("AMP backoff count is outside the bounded recovery budget")
    for attempt in range(_amp_backoffs, _MAX_AMP_BACKOFFS):
        metrics = _exact_renderer_optimizer_attempt(
            model=model,
            vae=vae,
            scheduler=scheduler,
            optimizer=optimizer,
            scaler=scaler,
            batch=batch,
            config=config,
            _amp_backoffs=attempt,
        )
        if metrics is not None:
            return metrics
    raise RuntimeError("renderer AMP overflow recovery exhausted")


def _exact_renderer_optimizer_attempt(
    *,
    model: P2InfillNoisePredictor,
    vae: nn.Module,
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: ExactMaskRendererBatch,
    config: V06BMaskFirstConfig,
    _amp_backoffs: int = 0,
) -> ExactRendererStepMetrics | None:
    if (
        not isinstance(config, V06BMaskFirstConfig)
        or config.schema_version != "v06_b_mask_first_v1"
    ):
        raise ValueError("exact renderer step requires the mask-first configuration")
    if not isinstance(scaler, torch.amp.GradScaler):
        raise ValueError("scaler must be a torch.amp.GradScaler")
    if not isinstance(_amp_backoffs, int) or _amp_backoffs < 0:
        raise ValueError("AMP backoff count must be a non-negative integer")
    _validate_batch(batch, config)
    try:
        unet = model.unet
    except AttributeError as error:
        raise ValueError("model must expose its frozen U-Net") from error
    _frozen_module(unet, name="U-Net")
    _frozen_module(vae, name="VAE")
    trainable = _model_trainables(model)
    optimized = _optimizer_parameters(optimizer)
    if len(optimized) != len(trainable) or {id(parameter) for parameter in optimized} != {
        id(parameter) for parameter in trainable
    }:
        raise ValueError("optimizer must contain exactly renderer trainables")
    expected_names = ("shared_renderer", "type_condition")
    if tuple(group.get("name") for group in optimizer.param_groups) != expected_names:
        raise ValueError("exact renderer optimizer must contain the exact named groups")
    if any(
        not math.isclose(
            float(group["lr"]),
            config.renderer_learning_rates[str(group["name"])],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for group in optimizer.param_groups
    ):
        raise ValueError("renderer optimizer learning rates differ from configuration")

    optimizer.zero_grad(set_to_none=True)
    noisy = scheduler.add_noise(batch.target_latent, batch.noise, batch.timesteps)
    prediction = model(
        noisy,
        batch.context,
        batch.binary_gt_mask,
        batch.timesteps,
        batch.prompt_embeds,
        batch.defect_type,
    )
    region = exact_mask_diffusion_loss(
        scheduler=scheduler,
        noisy=noisy,
        epsilon_hat=prediction,
        epsilon_target=batch.noise,
        target_latent=batch.target_latent,
        soft_ground_truth=batch.soft_gt_mask,
        timesteps=batch.timesteps,
    )
    schedule = appearance_schedule(
        scheduler,
        batch.timesteps,
        minimum_snr=config.appearance_min_snr,
    )
    appearance_weight = schedule.x0_weight * schedule.decode_active.to(
        dtype=schedule.x0_weight.dtype
    )
    if schedule.decode_active.any().item():
        predicted_image = decode_latent_for_training(vae, region.predicted_clean)
        appearance = defect_appearance_loss(
            rendered=predicted_image,
            target=batch.target_image,
            ground_truth=batch.image_gt_mask,
            sample_weight=appearance_weight,
        )
    else:
        appearance = DefectAppearanceLoss(
            rgb=_zero_from(prediction),
            sobel=_zero_from(prediction),
        )
    weights = config.renderer_loss_weights
    correct = (
        weights["epsilon_gt"] * region.epsilon_gt
        + weights["x0_gt"] * region.x0_gt
        + weights["rgb_gt"] * appearance.rgb
        + weights["edge_gt"] * appearance.sobel
    )

    with torch.no_grad():
        wrong_prediction = model(
            noisy,
            batch.context,
            batch.binary_gt_mask,
            batch.timesteps,
            batch.wrong_prompt_embeds,
            batch.wrong_defect_type,
        )
        wrong_region = exact_mask_diffusion_loss(
            scheduler=scheduler,
            noisy=noisy,
            epsilon_hat=wrong_prediction,
            epsilon_target=batch.noise,
            target_latent=batch.target_latent,
            soft_ground_truth=batch.soft_gt_mask,
            timesteps=batch.timesteps,
        )
        if schedule.decode_active.any().item():
            wrong_image = decode_latent_for_training(vae, wrong_region.predicted_clean)
            wrong_appearance = defect_appearance_loss(
                rendered=wrong_image,
                target=batch.target_image,
                ground_truth=batch.image_gt_mask,
                sample_weight=appearance_weight,
            )
        else:
            wrong_appearance = DefectAppearanceLoss(
                rgb=_zero_from(wrong_prediction),
                sobel=_zero_from(wrong_prediction),
            )
        wrong = (
            weights["epsilon_gt"] * wrong_region.epsilon_gt
            + weights["x0_gt"] * wrong_region.x0_gt
            + weights["rgb_gt"] * wrong_appearance.rgb
            + weights["edge_gt"] * wrong_appearance.sobel
        ).detach()
    type_margin = torch.relu(correct / wrong.detach().clamp_min(1e-4) - config.ranking_target_ratio)
    total = correct + weights["type_margin"] * type_margin
    values = {
        "total": total,
        "epsilon_gt": region.epsilon_gt,
        "x0_gt": region.x0_gt,
        "rgb_gt": appearance.rgb,
        "edge_gt": appearance.sobel,
        "type_margin": type_margin,
        "wrong_semantic": wrong,
    }
    if any(value.ndim != 0 or not torch.isfinite(value).item() for value in values.values()):
        raise ValueError("exact renderer objective must contain finite scalars")
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    gradients = tuple(parameter.grad for parameter in trainable if parameter.grad is not None)
    if not gradients:
        raise ValueError("renderer gradients must be present")
    try:
        required = tuple(model.gradient_parameters(batch.defect_type))
    except AttributeError as error:
        raise ValueError("model must expose active-route gradient parameters") from error
    if not required or any(parameter.grad is None for parameter in required):
        raise ValueError("active renderer route must receive gradients")
    if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        if not scaler.is_enabled():
            raise ValueError("renderer gradients must be present and finite")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if not scale_after < scale_before:
            raise RuntimeError("renderer AMP overflow did not reduce loss scale")
        if _amp_backoffs >= _MAX_AMP_BACKOFFS - 1:
            raise RuntimeError("renderer AMP overflow recovery exhausted")
        return None

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        config.renderer_gradient_clip,
    )
    if not torch.isfinite(grad_norm).item() or float(grad_norm.item()) <= 0.0:
        raise ValueError("renderer gradient norm must be finite and positive")
    scaler.step(optimizer)
    scaler.update()
    model.assert_frozen_backbone()
    _frozen_module(vae, name="VAE")
    numeric = {name: float(value.detach().item()) for name, value in values.items()}
    numeric["grad_l2"] = float(grad_norm.detach().item())
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("exact renderer metrics must be finite")
    return ExactRendererStepMetrics(
        total=numeric["total"],
        epsilon_gt=numeric["epsilon_gt"],
        x0_gt=numeric["x0_gt"],
        rgb_gt=numeric["rgb_gt"],
        edge_gt=numeric["edge_gt"],
        type_margin=numeric["type_margin"],
        wrong_semantic=numeric["wrong_semantic"],
        grad_l2=numeric["grad_l2"],
        finite=True,
    )


__all__ = [
    "ExactMaskRendererBatch",
    "ExactRendererStepMetrics",
    "build_exact_renderer_optimizer",
    "exact_renderer_optimizer_step",
]

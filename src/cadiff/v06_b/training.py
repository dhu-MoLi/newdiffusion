from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from cadiff.p2_a0.objective import masked_infill_loss
from cadiff.p2_a1.config import P2A1Config

from .appearance import (
    DefectAppearanceLoss,
    appearance_schedule,
    decode_latent_for_training,
    defect_appearance_loss,
    region_balanced_diffusion_loss,
)
from .config import V06BConfig
from .realization import (
    RealizationHead,
    compose_module_b_loss,
    realization_head_loss,
    renderer_realization_loss,
)
from .rollout import decode_latent_with_grad, truncated_denoise_rollout


@dataclass(frozen=True)
class DiffusionBatch:
    target_latent: Tensor
    context: Tensor
    binary_mask: Tensor
    soft_mask: Tensor
    prompt_embeds: Tensor
    defect_type: str
    timesteps: Tensor
    noise: Tensor


@dataclass(frozen=True)
class ProposalConditionedDiffusionBatch:
    target_latent: Tensor
    target_image: Tensor
    context: Tensor
    binary_proposal: Tensor
    soft_proposal: Tensor
    soft_ground_truth: Tensor
    proposal: Tensor
    ground_truth: Tensor
    prompt_embeds: Tensor
    wrong_prompt_embeds: Tensor
    defect_type: str
    wrong_defect_type: str
    timesteps: Tensor
    noise: Tensor


@dataclass(frozen=True)
class RealizationBatch:
    normal_latent: Tensor
    normal_image: Tensor
    context: Tensor
    binary_proposal: Tensor
    proposal: Tensor
    ground_truth: Tensor
    prompt_embeds: Tensor
    defect_type: str
    noise: Tensor


@dataclass(frozen=True)
class ModuleBStepMetrics:
    total: float
    diffusion: float
    visible: float
    ring: float
    mask: float
    consistency: float
    grad_l2: float
    finite: bool
    epsilon_gt: float = 0.0
    x0_gt: float = 0.0
    x0_ring: float = 0.0
    epsilon_outside: float = 0.0
    rgb_gt: float = 0.0
    sobel_gt: float = 0.0
    type_margin: float = 0.0
    wrong_semantic: float = 0.0
    appearance_active: float = 0.0
    renderer_grad_l2: float = 0.0
    head_grad_l2: float = 0.0


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, nn.Parameter)
    )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("optimizer contains duplicate parameters")
    return parameters


def _trainable_parameters(model: Any, head: RealizationHead) -> tuple[nn.Parameter, ...]:
    try:
        renderer_parameters = tuple(model.trainable_parameters())
    except AttributeError as error:
        raise ValueError("model must expose trainable_parameters") from error
    head_parameters = tuple(head.parameters())
    parameters = renderer_parameters + head_parameters
    if (
        not renderer_parameters
        or not head_parameters
        or len({id(parameter) for parameter in parameters}) != len(parameters)
        or any(not parameter.requires_grad for parameter in parameters)
    ):
        raise ValueError("renderer and realization head must expose distinct trainable parameters")
    return parameters


def _frozen_vae(vae: object) -> None:
    if not isinstance(vae, nn.Module):
        raise ValueError("vae must be a frozen torch module")
    parameters = tuple(vae.parameters())
    if not parameters or any(parameter.requires_grad for parameter in parameters):
        raise ValueError("vae parameters must be frozen")
    if any(module.training for module in vae.modules()):
        raise ValueError("vae and all descendants must remain in eval mode")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("frozen vae parameters must not have gradients")


def build_module_b_optimizer(
    *, model: Any, head: RealizationHead, config: V06BConfig
) -> torch.optim.Optimizer:
    if not isinstance(config, V06BConfig) or config.schema_version != "v06_b_causal_renderer_v2":
        raise ValueError("grouped optimizer requires the v2 appearance configuration")
    if not isinstance(head, RealizationHead):
        raise ValueError("head must be a RealizationHead")
    try:
        renderer = tuple(model.trainable_parameters())
        named = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    except AttributeError as error:
        raise ValueError("model must expose named and trainable parameters") from error
    renderer_ids = {id(parameter) for parameter in renderer}
    if (
        not renderer
        or len(renderer_ids) != len(renderer)
        or renderer_ids != {id(parameter) for _, parameter in named}
    ):
        raise ValueError("named renderer parameters must exactly match trainable_parameters")
    type_condition = tuple(
        parameter
        for name, parameter in named
        if name.startswith("conditioner.") or name.startswith("adapter.delta_")
    )
    type_ids = {id(parameter) for parameter in type_condition}
    shared = tuple(parameter for parameter in renderer if id(parameter) not in type_ids)
    head_parameters = tuple(head.parameters())
    all_parameters = shared + type_condition + head_parameters
    if (
        not shared
        or not type_condition
        or not head_parameters
        or len({id(parameter) for parameter in all_parameters}) != len(all_parameters)
    ):
        raise ValueError("optimizer groups must be non-empty, disjoint, and complete")
    rates = config.learning_rates
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
            {
                "name": "realization_head",
                "params": head_parameters,
                "lr": rates["realization_head"],
            },
        ),
        weight_decay=0.0,
    )


def cyclic_wrong_type(defect_type: str, defect_types: tuple[str, ...]) -> str:
    if defect_type not in defect_types or len(defect_types) < 2:
        raise ValueError("defect_type must belong to a multi-type vocabulary")
    return defect_types[(defect_types.index(defect_type) + 1) % len(defect_types)]


def _validate_v2_batch(
    batch: ProposalConditionedDiffusionBatch,
    realization: RealizationBatch,
    config: V06BConfig,
) -> None:
    if batch.defect_type != realization.defect_type:
        raise ValueError("both streams must use the same balanced defect type")
    if batch.defect_type not in config.defect_types:
        raise ValueError("defect type is outside the Module-B vocabulary")
    expected_wrong = cyclic_wrong_type(batch.defect_type, config.defect_types)
    if batch.wrong_defect_type != expected_wrong:
        raise ValueError("wrong_defect_type must be the fixed cyclic successor")
    if (
        batch.target_latent.ndim != 4
        or batch.context.shape != batch.target_latent.shape
        or batch.noise.shape != batch.target_latent.shape
        or batch.timesteps.shape != (batch.target_latent.shape[0],)
        or batch.timesteps.dtype is not torch.long
    ):
        raise ValueError("proposal-conditioned latent tensors have incompatible shapes")
    latent_mask_shape = (
        batch.target_latent.shape[0],
        1,
        *batch.target_latent.shape[2:],
    )
    if any(
        value.shape != latent_mask_shape
        for value in (
            batch.binary_proposal,
            batch.soft_proposal,
            batch.soft_ground_truth,
        )
    ):
        raise ValueError("proposal-conditioned latent masks have incompatible shapes")
    image_mask_shape = (
        batch.target_latent.shape[0],
        1,
        *batch.target_image.shape[2:],
    )
    if (
        batch.target_image.ndim != 4
        or batch.target_image.shape[1] != 3
        or batch.proposal.shape != image_mask_shape
        or batch.ground_truth.shape != image_mask_shape
    ):
        raise ValueError("proposal-conditioned image tensors have incompatible shapes")
    if (
        torch.any(batch.ground_truth.float() > batch.proposal.float()).item()
        or torch.any(batch.soft_ground_truth > batch.soft_proposal + 1e-6).item()
    ):
        raise ValueError("ground truth must be a subset of proposal at both resolutions")
    if not torch.all((batch.binary_proposal == 0) | (batch.binary_proposal == 1)).item():
        raise ValueError("binary_proposal must be binary")
    ordinary = (
        batch.target_latent,
        batch.target_image,
        batch.context,
        batch.binary_proposal,
        batch.soft_proposal,
        batch.soft_ground_truth,
        batch.proposal,
        batch.ground_truth,
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
        raise ValueError("proposal-conditioned inputs must be detached finite tensors")


def _zero_from(value: Tensor) -> Tensor:
    return value.sum() * 0.0


def _v2_optimizer_step(
    *,
    model: Any,
    vae: nn.Module,
    scheduler: Any,
    head: RealizationHead,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    diffusion_batch: ProposalConditionedDiffusionBatch,
    realization_batch: RealizationBatch,
    config: V06BConfig,
) -> ModuleBStepMetrics:
    _validate_v2_batch(diffusion_batch, realization_batch, config)
    _frozen_vae(vae)
    parameters = _trainable_parameters(model, head)
    optimized = _optimizer_parameters(optimizer)
    if {id(value) for value in parameters} != {id(value) for value in optimized} or len(
        parameters
    ) != len(optimized):
        raise ValueError("optimizer must contain exactly renderer and realization-head parameters")
    expected_names = ("shared_renderer", "type_condition", "realization_head")
    if tuple(group.get("name") for group in optimizer.param_groups) != expected_names:
        raise ValueError("v2 optimizer must contain the exact named parameter groups")
    if any(
        not math.isclose(
            float(group["lr"]),
            config.learning_rates[str(group["name"])],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for group in optimizer.param_groups
    ):
        raise ValueError("v2 optimizer learning rates differ from configuration")

    optimizer.zero_grad(set_to_none=True)
    real = diffusion_batch
    noisy = scheduler.add_noise(real.target_latent, real.noise, real.timesteps)
    prediction = model(
        noisy,
        real.context,
        real.binary_proposal,
        real.timesteps,
        real.prompt_embeds,
        real.defect_type,
    )
    region = region_balanced_diffusion_loss(
        scheduler=scheduler,
        noisy=noisy,
        epsilon_hat=prediction,
        epsilon_target=real.noise,
        target_latent=real.target_latent,
        soft_ground_truth=real.soft_ground_truth,
        soft_proposal=real.soft_proposal,
        timesteps=real.timesteps,
    )
    schedule = appearance_schedule(
        scheduler,
        real.timesteps,
        minimum_snr=config.appearance_min_snr,
    )
    appearance_weight = region.snr_weight * schedule.decode_active.to(
        dtype=region.snr_weight.dtype
    )
    if schedule.decode_active.any().item():
        predicted_image = decode_latent_for_training(vae, region.predicted_clean)
        appearance = defect_appearance_loss(
            rendered=predicted_image,
            target=real.target_image,
            ground_truth=real.ground_truth,
            sample_weight=appearance_weight,
        )
    else:
        appearance = DefectAppearanceLoss(
            rgb=_zero_from(prediction),
            sobel=_zero_from(prediction),
        )

    weights = config.loss_weights
    correct_semantic = (
        weights["epsilon_gt"] * region.epsilon_gt
        + weights["x0_gt"] * region.x0_gt
        + weights["rgb_gt"] * appearance.rgb
        + weights["sobel_gt"] * appearance.sobel
    )
    with torch.no_grad():
        wrong_prediction = model(
            noisy,
            real.context,
            real.binary_proposal,
            real.timesteps,
            real.wrong_prompt_embeds,
            real.wrong_defect_type,
        )
        wrong_region = region_balanced_diffusion_loss(
            scheduler=scheduler,
            noisy=noisy,
            epsilon_hat=wrong_prediction,
            epsilon_target=real.noise,
            target_latent=real.target_latent,
            soft_ground_truth=real.soft_ground_truth,
            soft_proposal=real.soft_proposal,
            timesteps=real.timesteps,
        )
        if schedule.decode_active.any().item():
            wrong_image = decode_latent_for_training(vae, wrong_region.predicted_clean)
            wrong_appearance = defect_appearance_loss(
                rendered=wrong_image,
                target=real.target_image,
                ground_truth=real.ground_truth,
                sample_weight=appearance_weight,
            )
            wrong_rgb = wrong_appearance.rgb
            wrong_sobel = wrong_appearance.sobel
        else:
            wrong_rgb = _zero_from(wrong_prediction)
            wrong_sobel = _zero_from(wrong_prediction)
        wrong_semantic = (
            weights["epsilon_gt"] * wrong_region.epsilon_gt
            + weights["x0_gt"] * wrong_region.x0_gt
            + weights["rgb_gt"] * wrong_rgb
            + weights["sobel_gt"] * wrong_sobel
        ).detach()
    type_margin = torch.relu(
        correct_semantic / wrong_semantic.clamp_min(1e-4)
        - config.ranking_target_ratio
    )
    diffusion = (
        weights["epsilon_gt"] * region.epsilon_gt
        + weights["x0_gt"] * region.x0_gt
        + weights["x0_ring"] * region.x0_ring
        + weights["epsilon_outside"] * region.epsilon_outside
    )
    real_total = (
        diffusion
        + weights["rgb_gt"] * appearance.rgb
        + weights["sobel_gt"] * appearance.sobel
        + weights["type_margin"] * type_margin
    )
    real_values = {
        "diffusion": diffusion.detach(),
        "epsilon_gt": region.epsilon_gt.detach(),
        "x0_gt": region.x0_gt.detach(),
        "x0_ring": region.x0_ring.detach(),
        "epsilon_outside": region.epsilon_outside.detach(),
        "rgb_gt": appearance.rgb.detach(),
        "sobel_gt": appearance.sobel.detach(),
        "type_margin": type_margin.detach(),
        "wrong_semantic": wrong_semantic.detach(),
        "appearance_active": schedule.decode_active.float().mean().detach(),
        "real_total": real_total.detach(),
    }
    if not all(torch.isfinite(value).all().item() for value in real_values.values()):
        raise ValueError("proposal-conditioned real objective produced non-finite values")
    scaler.scale(real_total).backward()
    del prediction, region, appearance, real_total

    realization = realization_batch
    rollout = truncated_denoise_rollout(
        model=model,
        scheduler=scheduler,
        normal_latent=realization.normal_latent,
        context=realization.context,
        binary_mask=realization.binary_proposal,
        prompt_embeds=realization.prompt_embeds,
        defect_type=realization.defect_type,
        noise=realization.noise,
        steps=config.rollout_steps,
        backprop_steps=config.backprop_steps,
        checkpoint_tracked_steps=config.checkpoint_tracked_steps,
    )
    rendered_image = decode_latent_with_grad(vae, rollout.latent)
    head_prediction = head(
        generated_latent=rollout.latent,
        normal_latent=realization.normal_latent,
        generation_feature=rollout.generation_feature,
        proposal=realization.proposal,
    )
    renderer_losses = renderer_realization_loss(
        rendered=rendered_image,
        normal=realization.normal_image,
        ground_truth=realization.ground_truth,
        proposal=realization.proposal,
        visibility_margin=config.visibility_margin,
    )
    head_losses = realization_head_loss(
        logits=head_prediction.logits,
        rendered=rendered_image,
        normal=realization.normal_image,
        ground_truth=realization.ground_truth,
        proposal=realization.proposal,
    )
    normal_total = (
        weights["visible"] * renderer_losses.visible
        + weights["ring"] * renderer_losses.ring
        + weights["mask"] * head_losses.mask
        + weights["consistency"] * head_losses.consistency
    )
    normal_values = (
        renderer_losses.visible,
        renderer_losses.ring,
        head_losses.mask,
        head_losses.consistency,
        normal_total,
    )
    if not all(value.ndim == 0 and torch.isfinite(value).all().item() for value in normal_values):
        raise ValueError("normal-reference objective produced a non-finite scalar")
    scaler.scale(normal_total).backward()
    total_value = real_values["real_total"] + normal_total.detach()

    scaler.unscale_(optimizer)
    renderer_parameters = tuple(model.trainable_parameters())
    head_parameters = tuple(head.parameters())
    gradients = tuple(
        parameter.grad
        for parameter in (*renderer_parameters, *head_parameters)
        if parameter.grad is not None
    )
    if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise ValueError("Module-B gradients must be present and finite")
    required_renderer = tuple(model.gradient_parameters(real.defect_type))
    if not required_renderer or any(parameter.grad is None for parameter in required_renderer):
        raise ValueError("active renderer route must receive gradients")
    if not any(parameter.grad is not None for parameter in head_parameters):
        raise ValueError("realization head must receive gradients")
    renderer_grad_norm = torch.nn.utils.clip_grad_norm_(
        renderer_parameters,
        config.gradient_clips["renderer"],
    )
    head_grad_norm = torch.nn.utils.clip_grad_norm_(
        head_parameters,
        config.gradient_clips["head"],
    )
    if (
        not torch.isfinite(renderer_grad_norm).item()
        or not torch.isfinite(head_grad_norm).item()
        or float(renderer_grad_norm.item()) <= 0.0
        or float(head_grad_norm.item()) <= 0.0
    ):
        raise ValueError("renderer and head gradient norms must be finite and positive")
    scaler.step(optimizer)
    scaler.update()
    model.assert_frozen_backbone()
    _frozen_vae(vae)

    numeric = {
        name: float(value.item())
        for name, value in {
            **real_values,
            "total": total_value,
            "visible": renderer_losses.visible.detach(),
            "ring": renderer_losses.ring.detach(),
            "mask": head_losses.mask.detach(),
            "consistency": head_losses.consistency.detach(),
            "renderer_grad_l2": renderer_grad_norm.detach(),
            "head_grad_l2": head_grad_norm.detach(),
        }.items()
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("Module-B metrics must be finite")
    grad_l2 = math.hypot(numeric["renderer_grad_l2"], numeric["head_grad_l2"])
    return ModuleBStepMetrics(
        total=numeric["total"],
        diffusion=numeric["diffusion"],
        visible=numeric["visible"],
        ring=numeric["ring"],
        mask=numeric["mask"],
        consistency=numeric["consistency"],
        grad_l2=grad_l2,
        finite=True,
        epsilon_gt=numeric["epsilon_gt"],
        x0_gt=numeric["x0_gt"],
        x0_ring=numeric["x0_ring"],
        epsilon_outside=numeric["epsilon_outside"],
        rgb_gt=numeric["rgb_gt"],
        sobel_gt=numeric["sobel_gt"],
        type_margin=numeric["type_margin"],
        wrong_semantic=numeric["wrong_semantic"],
        appearance_active=numeric["appearance_active"],
        renderer_grad_l2=numeric["renderer_grad_l2"],
        head_grad_l2=numeric["head_grad_l2"],
    )
def module_b_optimizer_step(
    *,
    model: Any,
    vae: nn.Module,
    scheduler: Any,
    head: RealizationHead,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    diffusion_batch: DiffusionBatch | ProposalConditionedDiffusionBatch,
    realization_batch: RealizationBatch,
    source_config: P2A1Config,
    config: V06BConfig,
) -> ModuleBStepMetrics:
    """Run one real-defect diffusion stream plus one normal-reference realization stream."""

    if not isinstance(source_config, P2A1Config):
        raise ValueError("source_config must be a P2A1Config")
    if not isinstance(config, V06BConfig):
        raise ValueError("config must be a V06BConfig")
    if not isinstance(head, RealizationHead):
        raise ValueError("head must be a RealizationHead")
    if not isinstance(realization_batch, RealizationBatch):
        raise ValueError("realization_batch must be a RealizationBatch")
    if not isinstance(scaler, torch.amp.GradScaler):
        raise ValueError("scaler must be a torch.amp.GradScaler")
    if isinstance(diffusion_batch, ProposalConditionedDiffusionBatch):
        if config.schema_version != "v06_b_causal_renderer_v2":
            raise ValueError("proposal-conditioned batches require the v2 configuration")
        return _v2_optimizer_step(
            model=model,
            vae=vae,
            scheduler=scheduler,
            head=head,
            optimizer=optimizer,
            scaler=scaler,
            diffusion_batch=diffusion_batch,
            realization_batch=realization_batch,
            config=config,
        )
    if not isinstance(diffusion_batch, DiffusionBatch):
        raise ValueError("diffusion_batch has an unsupported type")
    if diffusion_batch.defect_type != realization_batch.defect_type:
        raise ValueError("both streams must use the same balanced defect type")
    if diffusion_batch.defect_type not in config.defect_types:
        raise ValueError("defect type is outside the Module-B vocabulary")
    _frozen_vae(vae)
    parameters = _trainable_parameters(model, head)
    optimized = _optimizer_parameters(optimizer)
    if {id(value) for value in parameters} != {id(value) for value in optimized} or len(
        parameters
    ) != len(optimized):
        raise ValueError("optimizer must contain exactly renderer and realization-head parameters")

    optimizer.zero_grad(set_to_none=True)
    real = diffusion_batch
    noisy = scheduler.add_noise(real.target_latent, real.noise, real.timesteps)
    prediction = model(
        noisy,
        real.context,
        real.binary_mask,
        real.timesteps,
        real.prompt_embeds,
        real.defect_type,
    )
    diffusion = masked_infill_loss(
        scheduler=scheduler,
        noisy=noisy,
        epsilon_hat=prediction,
        epsilon_target=real.noise,
        target_latent=real.target_latent,
        soft_mask=real.soft_mask,
        timesteps=real.timesteps,
        inside_noise_weight=source_config.inside_noise_weight,
        lambda_inside=source_config.lambda_inside,
        lambda_outside=source_config.lambda_outside,
    ).total

    realization = realization_batch
    rollout = truncated_denoise_rollout(
        model=model,
        scheduler=scheduler,
        normal_latent=realization.normal_latent,
        context=realization.context,
        binary_mask=realization.binary_proposal,
        prompt_embeds=realization.prompt_embeds,
        defect_type=realization.defect_type,
        noise=realization.noise,
        steps=config.rollout_steps,
        backprop_steps=config.backprop_steps,
        checkpoint_tracked_steps=config.checkpoint_tracked_steps,
    )
    rendered_image = decode_latent_with_grad(vae, rollout.latent)
    if rendered_image.shape != realization.normal_image.shape:
        raise ValueError("decoded rollout image must match normal_image")
    head_prediction = head(
        generated_latent=rollout.latent,
        normal_latent=realization.normal_latent,
        generation_feature=rollout.generation_feature,
        proposal=realization.proposal,
    )
    renderer_losses = renderer_realization_loss(
        rendered=rendered_image,
        normal=realization.normal_image,
        ground_truth=realization.ground_truth,
        proposal=realization.proposal,
        visibility_margin=config.visibility_margin,
    )
    head_losses = realization_head_loss(
        logits=head_prediction.logits,
        rendered=rendered_image,
        normal=realization.normal_image,
        ground_truth=realization.ground_truth,
        proposal=realization.proposal,
    )
    weights = config.loss_weights
    combined = compose_module_b_loss(
        diffusion=diffusion,
        renderer=renderer_losses,
        head=head_losses,
        diffusion_weight=weights["diffusion"],
        visible_weight=weights["visible"],
        ring_weight=weights["ring"],
        mask_weight=weights["mask"],
        consistency_weight=weights["consistency"],
    )
    tensors = (
        combined.total,
        combined.diffusion,
        combined.visible,
        combined.ring,
        combined.mask,
        combined.consistency,
    )
    if not all(value.ndim == 0 and torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("Module-B objective produced a non-finite scalar")

    scaler.scale(combined.total).backward()
    scaler.unscale_(optimizer)
    gradients = tuple(
        parameter.grad for parameter in parameters if parameter.grad is not None
    )
    if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise ValueError("Module-B gradients must be present and finite")
    required_renderer = tuple(model.gradient_parameters(real.defect_type))
    if not required_renderer or any(parameter.grad is None for parameter in required_renderer):
        raise ValueError("active renderer route must receive gradients")
    if not any(parameter.grad is not None for parameter in head.parameters()):
        raise ValueError("realization head must receive gradients")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
    if not torch.isfinite(grad_norm).item() or float(grad_norm.item()) <= 0.0:
        raise ValueError("Module-B gradient norm must be finite and positive")
    scaler.step(optimizer)
    scaler.update()

    model.assert_frozen_backbone()
    _frozen_vae(vae)
    numeric = tuple(float(value.detach().item()) for value in tensors)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Module-B metrics must be finite")
    return ModuleBStepMetrics(
        total=numeric[0],
        diffusion=numeric[1],
        visible=numeric[2],
        ring=numeric[3],
        mask=numeric[4],
        consistency=numeric[5],
        grad_l2=float(grad_norm.detach().item()),
        finite=True,
    )


__all__ = [
    "DiffusionBatch",
    "ModuleBStepMetrics",
    "ProposalConditionedDiffusionBatch",
    "RealizationBatch",
    "build_module_b_optimizer",
    "module_b_optimizer_step",
]

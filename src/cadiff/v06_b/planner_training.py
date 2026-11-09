from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import V06BMaskFirstConfig
from .planner import PlannerNetwork, PlannerPrediction


@dataclass(frozen=True)
class PlannerBatch:
    normal_rgb: Tensor
    proposal: Tensor
    target: Tensor
    type_indices: Tensor


@dataclass(frozen=True)
class PlannerLoss:
    total: Tensor
    bce: Tensor
    dice: Tensor


@dataclass(frozen=True)
class PlannerStepMetrics:
    total: float
    bce: float
    dice: float
    grad_l2: float
    finite: bool


def _binary_training_masks(
    target: object,
    proposal: object,
    *,
    shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    if (
        not isinstance(target, Tensor)
        or not isinstance(proposal, Tensor)
        or target.shape != shape
        or proposal.shape != shape
        or target.device != proposal.device
    ):
        raise ValueError("target and proposal must match planner prediction geometry")
    if (
        not torch.isfinite(target).all().item()
        or not torch.isfinite(proposal).all().item()
        or not torch.all((target == 0) | (target == 1)).item()
        or not torch.all((proposal == 0) | (proposal == 1)).item()
    ):
        raise ValueError("target and proposal must be finite binary tensors")
    target_float = target.float()
    proposal_float = proposal.float()
    if torch.any(target_float > proposal_float).item():
        raise ValueError("target must be a subset of proposal")
    target_count = target_float.sum(dim=(1, 2, 3))
    proposal_count = proposal_float.sum(dim=(1, 2, 3))
    if not torch.all(target_count > 0).item():
        raise ValueError("target must be non-empty for every sample")
    if not torch.all(proposal_count > 0).item():
        raise ValueError("proposal must be non-empty for every sample")
    if not torch.all(proposal_count < proposal[0].numel()).item():
        raise ValueError("proposal must not cover the full image")
    return target_float, proposal_float


def planner_loss(
    prediction: PlannerPrediction,
    *,
    target: Tensor,
    proposal: Tensor,
    dice_weight: float,
) -> PlannerLoss:
    if not isinstance(prediction, PlannerPrediction):
        raise ValueError("prediction must be a PlannerPrediction")
    if (
        prediction.logits.ndim != 4
        or prediction.logits.shape[0] <= 0
        or prediction.logits.shape[1] != 1
        or prediction.q.shape != prediction.logits.shape
        or prediction.q.device != prediction.logits.device
        or not prediction.logits.is_floating_point()
        or prediction.q.dtype != prediction.logits.dtype
        or not torch.isfinite(prediction.logits).all().item()
        or not torch.isfinite(prediction.q).all().item()
        or torch.any((prediction.q < 0.0) | (prediction.q > 1.0)).item()
    ):
        raise ValueError("planner prediction must contain matching finite logits and q")
    if (
        isinstance(dice_weight, bool)
        or not isinstance(dice_weight, int | float)
        or not math.isfinite(float(dice_weight))
        or float(dice_weight) < 0.0
    ):
        raise ValueError("dice_weight must be finite and non-negative")
    target_float, proposal_float = _binary_training_masks(
        target,
        proposal,
        shape=tuple(prediction.logits.shape),
    )
    if target.device != prediction.logits.device:
        raise ValueError("target and proposal must share the prediction device")
    if torch.any(prediction.q * (1.0 - proposal_float) != 0.0).item():
        raise ValueError("planner q must be exactly zero outside proposal")

    inside = proposal.bool()
    bce = functional.binary_cross_entropy_with_logits(
        prediction.logits[inside], target_float[inside]
    )
    dice = 1.0 - (2.0 * (prediction.q * target_float).sum() + 1.0) / (
        prediction.q.sum() + target_float.sum() + 1.0
    )
    total = bce + float(dice_weight) * dice
    losses = PlannerLoss(total=total, bce=bce, dice=dice)
    if any(
        value.ndim != 0 or not torch.isfinite(value).all().item()
        for value in (losses.total, losses.bce, losses.dice)
    ):
        raise ValueError("planner objective must produce finite scalars")
    return losses


def build_planner_optimizer(
    *,
    network: PlannerNetwork,
    config: V06BMaskFirstConfig,
) -> torch.optim.AdamW:
    if not isinstance(network, PlannerNetwork):
        raise ValueError("network must be a PlannerNetwork")
    if (
        not isinstance(config, V06BMaskFirstConfig)
        or config.schema_version != "v06_b_mask_first_v1"
    ):
        raise ValueError("config must be the mask-first Module-B configuration")
    parameters = tuple(network.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("planner network parameters must all be trainable")
    return torch.optim.AdamW(
        parameters,
        lr=config.planner_learning_rate,
        weight_decay=0.0,
    )


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    entries = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, nn.Parameter)
    )
    if len(entries) != sum(len(group["params"]) for group in optimizer.param_groups):
        raise ValueError("optimizer entries must all be parameters")
    return entries


def planner_optimizer_step(
    *,
    network: PlannerNetwork,
    optimizer: torch.optim.Optimizer,
    batch: PlannerBatch,
    config: V06BMaskFirstConfig,
) -> PlannerStepMetrics:
    if not isinstance(network, PlannerNetwork):
        raise ValueError("network must be a PlannerNetwork")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("planner optimizer must be AdamW")
    if not isinstance(batch, PlannerBatch):
        raise ValueError("batch must be a PlannerBatch")
    if (
        not isinstance(config, V06BMaskFirstConfig)
        or config.schema_version != "v06_b_mask_first_v1"
    ):
        raise ValueError("config must be the mask-first Module-B configuration")
    parameters = tuple(network.parameters())
    optimized = _optimizer_parameters(optimizer)
    if (
        not parameters
        or len({id(parameter) for parameter in parameters}) != len(parameters)
        or len(optimized) != len(parameters)
        or {id(parameter) for parameter in optimized}
        != {id(parameter) for parameter in parameters}
    ):
        raise ValueError("optimizer must contain exactly planner network parameters")
    if any(group["lr"] != config.planner_learning_rate for group in optimizer.param_groups):
        raise ValueError("optimizer learning rate must match planner configuration")
    batch_tensors = (
        batch.normal_rgb,
        batch.proposal,
        batch.target,
        batch.type_indices,
    )
    if any(not isinstance(value, Tensor) or value.requires_grad for value in batch_tensors):
        raise ValueError("planner batch tensors must be detached")

    optimizer.zero_grad(set_to_none=True)
    prediction = network(batch.normal_rgb, batch.proposal, batch.type_indices)
    losses = planner_loss(
        prediction,
        target=batch.target,
        proposal=batch.proposal,
        dice_weight=config.planner_dice_weight,
    )
    losses.total.backward()
    gradients = tuple(parameter.grad for parameter in parameters)
    if any(gradient is None for gradient in gradients):
        raise ValueError("every planner parameter must receive a gradient")
    concrete = tuple(gradient for gradient in gradients if gradient is not None)
    if any(not torch.isfinite(gradient).all().item() for gradient in concrete):
        raise ValueError("planner gradients must be finite")
    torch.nn.utils.clip_grad_norm_(parameters, config.planner_gradient_clip)
    clipped_grad_l2 = math.sqrt(
        sum(float(gradient.float().square().sum().item()) for gradient in concrete)
    )
    if not math.isfinite(clipped_grad_l2) or clipped_grad_l2 <= 0.0:
        raise ValueError("planner gradient norm must be finite and positive")
    optimizer.step()
    if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
        raise ValueError("planner parameters must remain finite")
    values = (losses.total, losses.bce, losses.dice)
    numeric = tuple(float(value.detach().item()) for value in values)
    return PlannerStepMetrics(
        total=numeric[0],
        bce=numeric[1],
        dice=numeric[2],
        grad_l2=clipped_grad_l2,
        finite=all(math.isfinite(value) for value in numeric),
    )


__all__ = [
    "PlannerBatch",
    "PlannerLoss",
    "PlannerStepMetrics",
    "build_planner_optimizer",
    "planner_loss",
    "planner_optimizer_step",
]

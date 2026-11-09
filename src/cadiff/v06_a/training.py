"""CPU-testable epsilon objective for Module A mask diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .diffusion import MaskDiffusionSchedule
from .model import A1MaskUNet, validate_a1_model_contract


@dataclass(frozen=True)
class A1DiffusionBatch:
    z0: Tensor
    valid: Tensor
    sdf: Tensor
    u_x: Tensor
    u_y: Tensor
    type_ids: Tensor
    normal_rgb: Tensor | None = None

    def __post_init__(self) -> None:
        z0 = self.z0
        if (
            not isinstance(z0, Tensor)
            or z0.ndim != 4
            or z0.shape[0] <= 0
            or z0.shape[1:] != (1, 128, 128)
            or not z0.is_floating_point()
            or not torch.isfinite(z0).all().item()
        ):
            raise ValueError("z0 must be a finite floating [B,1,128,128] tensor")
        if (
            not isinstance(self.valid, Tensor)
            or self.valid.dtype != torch.bool
            or self.valid.shape != z0.shape
            or self.valid.device != z0.device
        ):
            raise ValueError("valid must be a bool tensor matching z0")
        if not self.valid.flatten(1).any(dim=1).all().item():
            raise ValueError("each batch item requires a non-empty valid canvas")
        if not torch.all((z0 == -1) | (z0 == 1)).item():
            raise ValueError("z0 must use the frozen {-1,+1} binary encoding")
        if not torch.all(z0[~self.valid] == -1).item():
            raise ValueError("z0 must equal -1 outside the valid canvas")
        positive_counts = ((z0 == 1) & self.valid).flatten(1).sum(dim=1)
        valid_counts = self.valid.flatten(1).sum(dim=1)
        if torch.any((positive_counts == 0) | (positive_counts == valid_counts)).item():
            raise ValueError(
                "each sample positive target must be non-empty and non-full within valid"
            )
        for value, name in ((self.sdf, "sdf"), (self.u_x, "u_x"), (self.u_y, "u_y")):
            if (
                not isinstance(value, Tensor)
                or value.shape != z0.shape
                or value.dtype != z0.dtype
                or value.device != z0.device
                or not torch.isfinite(value).all().item()
            ):
                raise ValueError(f"{name} must be finite and match z0")
        if (
            not isinstance(self.type_ids, Tensor)
            or self.type_ids.dtype != torch.long
            or self.type_ids.shape != (z0.shape[0],)
            or self.type_ids.device != z0.device
            or not torch.all(self.type_ids >= 0).item()
        ):
            raise ValueError("type_ids must be a matching non-negative long tensor")
        if self.normal_rgb is not None and (
            not isinstance(self.normal_rgb, Tensor)
            or self.normal_rgb.shape != (z0.shape[0], 3, 512, 512)
            or self.normal_rgb.dtype != z0.dtype
            or self.normal_rgb.device != z0.device
            or not torch.isfinite(self.normal_rgb).all().item()
        ):
            raise ValueError("normal_rgb must be finite [B,3,512,512] and match z0")


@dataclass(frozen=True)
class A1StepRandomness:
    timesteps: Tensor
    noise: Tensor


@dataclass(frozen=True)
class A1LossResult:
    loss: Tensor
    z_k: Tensor
    predicted_noise: Tensor
    target_noise: Tensor
    timesteps: Tensor


def sample_a1_step_randomness(
    *,
    batch: A1DiffusionBatch,
    schedule: MaskDiffusionSchedule,
    generator: torch.Generator,
) -> A1StepRandomness:
    if not isinstance(batch, A1DiffusionBatch):
        raise TypeError("batch must be an A1DiffusionBatch")
    if not isinstance(schedule, MaskDiffusionSchedule):
        raise TypeError("schedule must be a MaskDiffusionSchedule")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch Generator")
    timesteps = torch.randint(
        0,
        schedule.diffusion_steps,
        (batch.z0.shape[0],),
        dtype=torch.long,
        device=batch.z0.device,
        generator=generator,
    )
    noise = torch.randn(
        batch.z0.shape,
        dtype=batch.z0.dtype,
        device=batch.z0.device,
        generator=generator,
    )
    return A1StepRandomness(timesteps=timesteps, noise=noise)


def _validate_randomness(randomness: A1StepRandomness, batch: A1DiffusionBatch) -> None:
    if not isinstance(randomness, A1StepRandomness):
        raise TypeError("randomness must be A1StepRandomness")
    if (
        not isinstance(randomness.timesteps, Tensor)
        or randomness.timesteps.dtype != torch.long
        or randomness.timesteps.shape != (batch.z0.shape[0],)
        or randomness.timesteps.device != batch.z0.device
    ):
        raise ValueError("randomness timesteps must match the batch")
    if (
        not isinstance(randomness.noise, Tensor)
        or randomness.noise.shape != batch.z0.shape
        or randomness.noise.dtype != batch.z0.dtype
        or randomness.noise.device != batch.z0.device
        or not torch.isfinite(randomness.noise).all().item()
    ):
        raise ValueError("randomness noise must be finite and match z0")


def compute_a1_diffusion_loss(
    *,
    model: A1MaskUNet,
    schedule: MaskDiffusionSchedule,
    batch: A1DiffusionBatch,
    randomness: A1StepRandomness,
) -> A1LossResult:
    """Compute epsilon MSE over V only; RGB remains absent for GV and required for SV."""

    validate_a1_model_contract(model)
    if not isinstance(schedule, MaskDiffusionSchedule):
        raise TypeError("schedule must be a MaskDiffusionSchedule")
    if schedule.diffusion_steps != model.diffusion_steps:
        raise ValueError("schedule/model diffusion_steps mismatch")
    if not isinstance(batch, A1DiffusionBatch):
        raise TypeError("batch must be an A1DiffusionBatch")
    _validate_randomness(randomness, batch)
    if not torch.all(
        (0 <= randomness.timesteps) & (randomness.timesteps < schedule.diffusion_steps)
    ).item():
        raise ValueError("randomness timesteps are outside the diffusion schedule")

    z_k = schedule.q_sample(
        z0=batch.z0,
        timesteps=randomness.timesteps,
        noise=randomness.noise,
        valid=batch.valid,
    )
    spatial = torch.cat(
        (z_k, batch.valid.to(batch.z0.dtype), batch.sdf, batch.u_x, batch.u_y), dim=1
    )
    predicted_noise = model(
        spatial,
        type_ids=batch.type_ids,
        timesteps=randomness.timesteps,
        normal_rgb=batch.normal_rgb,
    )
    if (
        not isinstance(predicted_noise, Tensor)
        or predicted_noise.shape != batch.z0.shape
        or predicted_noise.dtype != batch.z0.dtype
        or predicted_noise.device != batch.z0.device
        or not torch.isfinite(predicted_noise).all().item()
    ):
        raise ValueError("model output must be finite and match z0")
    valid_weight = batch.valid.to(batch.z0.dtype)
    denominator = valid_weight.sum()
    if denominator.item() <= 0:
        raise ValueError("diffusion loss requires a non-empty valid canvas")
    loss = ((predicted_noise - randomness.noise).square() * valid_weight).sum() / denominator
    return A1LossResult(
        loss=loss,
        z_k=z_k,
        predicted_noise=predicted_noise,
        target_noise=randomness.noise,
        timesteps=randomness.timesteps,
    )

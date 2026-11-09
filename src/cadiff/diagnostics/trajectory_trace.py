from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cadiff.diffusion.components import FrozenComponents
from cadiff.diffusion.ddim_inversion import (
    TrajectoryState,
    _make_scheduler_pair,
    _validate_trajectory,
    encode_prompt,
)


@dataclass(frozen=True)
class TraceStateSummary:
    timestep: int
    latent_l1_norm: float | None
    latent_l2_norm: float | None
    previous_step_l1: float | None
    finite: bool


@dataclass(frozen=True)
class CycleStepSummary:
    source_timestep: int
    target_timestep: int
    latent_l1: float | None
    latent_l2: float | None
    finite: bool


@dataclass(frozen=True)
class TracedReconstruction:
    final_latent: torch.Tensor
    states: tuple[TraceStateSummary, ...]
    cycle: tuple[CycleStepSummary, ...]


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _norms(tensor: torch.Tensor) -> tuple[float | None, float | None]:
    if not _finite(tensor):
        return None, None
    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    l1 = float(value.abs().mean().item())
    l2 = float(value.square().mean().sqrt().item())
    if not math.isfinite(l1) or not math.isfinite(l2):
        return None, None
    return l1, l2


def _distance(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[float | None, float | None]:
    if left.shape != right.shape:
        raise ValueError("trace tensors must preserve latent shape")
    if not _finite(left) or not _finite(right):
        return None, None
    difference = left.detach().to("cpu", torch.float32) - right.detach().to(
        "cpu", torch.float32
    )
    l1 = float(difference.abs().mean().item())
    l2 = float(difference.square().mean().sqrt().item())
    if not math.isfinite(l1) or not math.isfinite(l2):
        return None, None
    return l1, l2


def _state_summary(
    timestep: int,
    latent: torch.Tensor,
    previous: torch.Tensor | None,
) -> TraceStateSummary:
    l1, l2 = _norms(latent)
    previous_l1 = None
    if previous is not None:
        previous_l1, _ = _distance(latent, previous)
    return TraceStateSummary(timestep, l1, l2, previous_l1, l1 is not None and l2 is not None)


def _cycle_summary(
    source_timestep: int,
    target: TrajectoryState,
    reconstructed: torch.Tensor,
) -> CycleStepSummary:
    l1, l2 = _distance(reconstructed, target.latent)
    return CycleStepSummary(
        source_timestep,
        target.timestep,
        l1,
        l2,
        l1 is not None and l2 is not None,
    )


@torch.inference_mode()
def reconstruct_with_trace(
    components: FrozenComponents,
    trajectory: list[TrajectoryState],
    prompt: str,
) -> TracedReconstruction:
    scheduler, _ = _make_scheduler_pair(components.scheduler)
    _validate_trajectory(trajectory, scheduler)
    device = components.unet.device
    dtype = components.unet.dtype
    prompt_embeds = encode_prompt(components, prompt, device)
    current = trajectory[-1].latent.to(device=device, dtype=dtype)
    states = [_state_summary(trajectory[-1].timestep, current, None)]
    cycle: list[CycleStepSummary] = []
    for index in range(len(trajectory) - 1, 0, -1):
        source = trajectory[index]
        target = trajectory[index - 1]
        timestep = torch.tensor(source.timestep, device=device)
        prediction = components.unet(
            current, timestep, encoder_hidden_states=prompt_embeds
        ).sample
        previous = current
        current = scheduler.step(prediction, timestep, current, eta=0.0).prev_sample
        states.append(_state_summary(target.timestep, current, previous))
        cycle.append(_cycle_summary(source.timestep, target, current))
    return TracedReconstruction(current, tuple(states), tuple(cycle))

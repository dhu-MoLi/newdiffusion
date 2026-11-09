from __future__ import annotations

from dataclasses import dataclass

import torch

from cadiff.diagnostics.trajectory_trace import (
    CycleStepSummary,
    TraceStateSummary,
    _cycle_summary,
    _state_summary,
)
from cadiff.diffusion.components import FrozenComponents
from cadiff.diffusion.ddim_inversion import (
    TrajectoryState,
    _cpu_snapshot,
    _make_scheduler_pair,
    _validate_latent,
    _validate_prompt,
    _validate_steps,
    _validate_trajectory,
    encode_prompt,
)


@dataclass(frozen=True)
class NoisePredictionState:
    timestep: int
    prediction: torch.Tensor


@dataclass(frozen=True)
class NoiseReplayTrajectory:
    latents: tuple[TrajectoryState, ...]
    predictions: tuple[NoisePredictionState, ...]


@dataclass(frozen=True)
class NoiseReplayReconstruction:
    final_latent: torch.Tensor
    states: tuple[TraceStateSummary, ...]
    cycle: tuple[CycleStepSummary, ...]


def _validate_model_prediction(prediction: torch.Tensor, latent: torch.Tensor) -> None:
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("noise prediction must be a torch.Tensor")
    if prediction.shape != latent.shape:
        raise ValueError("noise prediction must match latent shape")
    if prediction.dtype != latent.dtype or not prediction.is_floating_point():
        raise TypeError("noise prediction must match floating latent dtype")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("noise prediction must contain only finite values")


@torch.inference_mode()
def invert_with_noise_replay(
    components: FrozenComponents,
    latent: torch.Tensor,
    prompt: str,
    steps: int,
) -> NoiseReplayTrajectory:
    _validate_latent(latent)
    _validate_prompt(prompt)
    _validate_steps(steps, components.scheduler)
    _, inverse = _make_scheduler_pair(components.scheduler)
    inverse.set_timesteps(steps, device=latent.device)
    prompt_embeds = encode_prompt(components, prompt, latent.device)

    current = latent
    latents = [TrajectoryState(-1, _cpu_snapshot(current))]
    predictions: list[NoisePredictionState] = []
    for timestep in inverse.timesteps:
        prediction = components.unet(
            current, timestep, encoder_hidden_states=prompt_embeds
        ).sample
        _validate_model_prediction(prediction, current)
        exact_timestep = int(timestep.item())
        predictions.append(NoisePredictionState(exact_timestep, _cpu_snapshot(prediction)))
        current = inverse.step(prediction, timestep, current).prev_sample
        latents.append(TrajectoryState(exact_timestep, _cpu_snapshot(current)))
    return NoiseReplayTrajectory(tuple(latents), tuple(predictions))


def _validate_noise_replay(
    trajectory: NoiseReplayTrajectory,
    scheduler: object,
) -> None:
    if not isinstance(trajectory, NoiseReplayTrajectory):
        raise TypeError("trajectory must be a NoiseReplayTrajectory")
    if not isinstance(trajectory.latents, tuple) or not isinstance(
        trajectory.predictions, tuple
    ):
        raise TypeError("noise replay states must be tuples")
    _validate_trajectory(list(trajectory.latents), scheduler)  # type: ignore[arg-type]
    if len(trajectory.predictions) != len(trajectory.latents) - 1:
        raise ValueError("noise predictions must match every inversion step")
    if not all(isinstance(state, NoisePredictionState) for state in trajectory.predictions):
        raise TypeError("noise predictions must be NoisePredictionState values")
    if any(type(state.timestep) is not int for state in trajectory.predictions):
        raise TypeError("noise prediction timesteps must be exact integers")
    expected = [state.timestep for state in trajectory.latents[1:]]
    actual = [state.timestep for state in trajectory.predictions]
    if actual != expected:
        raise ValueError("noise prediction timesteps do not match latent timesteps")
    initial = trajectory.latents[0].latent
    for state in trajectory.predictions:
        prediction = state.prediction
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("noise prediction must be a torch.Tensor")
        if prediction.device.type != "cpu" or prediction.requires_grad:
            raise ValueError("noise predictions must be detached CPU snapshots")
        _validate_model_prediction(prediction, initial)


@torch.inference_mode()
def reconstruct_with_noise_replay(
    components: FrozenComponents,
    trajectory: NoiseReplayTrajectory,
) -> NoiseReplayReconstruction:
    scheduler, _ = _make_scheduler_pair(components.scheduler)
    _validate_noise_replay(trajectory, scheduler)
    device = components.unet.device
    dtype = components.unet.dtype
    current = trajectory.latents[-1].latent.to(device=device, dtype=dtype)
    states = [_state_summary(trajectory.latents[-1].timestep, current, None)]
    cycle: list[CycleStepSummary] = []
    for index in range(len(trajectory.predictions) - 1, -1, -1):
        saved = trajectory.predictions[index]
        target = trajectory.latents[index]
        timestep = torch.tensor(saved.timestep, device=device)
        prediction = saved.prediction.to(device=device, dtype=dtype)
        previous = current
        current = scheduler.step(prediction, timestep, current, eta=0.0).prev_sample
        states.append(_state_summary(target.timestep, current, previous))
        cycle.append(_cycle_summary(saved.timestep, target, current))
    return NoiseReplayReconstruction(current, tuple(states), tuple(cycle))

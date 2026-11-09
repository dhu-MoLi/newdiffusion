from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler

from cadiff.diffusion.components import FrozenComponents


@dataclass(frozen=True)
class TrajectoryState:
    timestep: int
    latent: torch.Tensor


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    if hasattr(left, "shape") or hasattr(right, "shape"):
        return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    return left == right


def assert_scheduler_pair(
    forward: DDIMScheduler, inverse: DDIMInverseScheduler
) -> None:
    forward_config = dict(forward.config)
    inverse_config = dict(inverse.config)
    if forward_config.get("thresholding", False):
        raise ValueError("scheduler mismatch: forward thresholding is unsupported by inverse")
    if forward_config.get("timestep_spacing") not in {"leading", "trailing"}:
        raise ValueError("scheduler mismatch: inverse does not support timestep_spacing")

    common_keys = sorted(
        key
        for key in forward_config.keys() & inverse_config.keys()
        if not key.startswith("_")
    )
    mismatches = [
        key
        for key in common_keys
        if not _same_value(forward_config[key], inverse_config[key])
    ]
    if not torch.equal(forward.alphas_cumprod, inverse.alphas_cumprod):
        mismatches.append("alphas_cumprod")
    if mismatches:
        raise ValueError(f"scheduler mismatch: {mismatches}")


def _scheduler_kwargs(
    scheduler: DDIMScheduler, scheduler_type: type[DDIMScheduler | DDIMInverseScheduler]
) -> dict[str, Any]:
    supported = {
        name
        for name, parameter in inspect.signature(scheduler_type.__init__).parameters.items()
        if name != "self" and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    return {key: value for key, value in dict(scheduler.config).items() if key in supported}


def _make_scheduler_pair(
    scheduler: DDIMScheduler,
) -> tuple[DDIMScheduler, DDIMInverseScheduler]:
    forward = DDIMScheduler(**_scheduler_kwargs(scheduler, DDIMScheduler))
    inverse = DDIMInverseScheduler(**_scheduler_kwargs(scheduler, DDIMInverseScheduler))
    assert_scheduler_pair(forward, inverse)
    return forward, inverse


def _validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")


def _validate_latent(latent: torch.Tensor, *, name: str = "latent") -> None:
    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if latent.ndim != 4:
        raise ValueError(f"{name} must be a floating 4D tensor")
    if not latent.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(latent).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_steps(steps: int, scheduler: DDIMScheduler) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be a positive integer")
    maximum = int(scheduler.config.num_train_timesteps)
    if not 0 < steps <= maximum:
        raise ValueError(f"steps must be in [1, {maximum}]")


@torch.inference_mode()
def encode_prompt(
    components: FrozenComponents, prompt: str, device: torch.device
) -> torch.Tensor:
    _validate_prompt(prompt)
    tokens = components.tokenizer(
        prompt,
        padding="max_length",
        max_length=components.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    return components.text_encoder(input_ids)[0]


def _cpu_snapshot(latent: torch.Tensor) -> torch.Tensor:
    return latent.detach().to(device="cpu").clone()


@torch.inference_mode()
def invert(
    components: FrozenComponents,
    latent: torch.Tensor,
    prompt: str,
    steps: int,
) -> list[TrajectoryState]:
    _validate_latent(latent)
    _validate_prompt(prompt)
    _validate_steps(steps, components.scheduler)
    _, inverse = _make_scheduler_pair(components.scheduler)
    inverse.set_timesteps(steps, device=latent.device)
    prompt_embeds = encode_prompt(components, prompt, latent.device)

    current = latent
    trajectory = [TrajectoryState(-1, _cpu_snapshot(current))]
    for timestep in inverse.timesteps:
        prediction = components.unet(
            current, timestep, encoder_hidden_states=prompt_embeds
        ).sample
        current = inverse.step(prediction, timestep, current).prev_sample
        trajectory.append(TrajectoryState(int(timestep.item()), _cpu_snapshot(current)))
    return trajectory


def _validate_trajectory(
    trajectory: list[TrajectoryState], scheduler: DDIMScheduler
) -> None:
    if not isinstance(trajectory, list):
        raise TypeError("trajectory must be a list")
    if len(trajectory) < 2:
        raise ValueError("trajectory must contain at least an initial state and one step")
    if not all(isinstance(state, TrajectoryState) for state in trajectory):
        raise TypeError("trajectory entries must be TrajectoryState values")
    if any(type(state.timestep) is not int for state in trajectory):
        raise TypeError("trajectory timesteps must be exact integers")
    if trajectory[0].timestep != -1:
        raise ValueError("trajectory must start with timestep -1")

    initial = trajectory[0].latent
    _validate_latent(initial, name="trajectory latent")
    if initial.device.type != "cpu" or initial.requires_grad:
        raise ValueError("trajectory snapshots must be detached CPU tensors")
    for state in trajectory[1:]:
        _validate_latent(state.latent, name="trajectory latent")
        if state.latent.device.type != "cpu" or state.latent.requires_grad:
            raise ValueError("trajectory snapshots must be detached CPU tensors")
        if state.latent.shape != initial.shape or state.latent.dtype != initial.dtype:
            raise ValueError("trajectory snapshots must preserve latent shape and dtype")

    stored = [state.timestep for state in trajectory[1:]]
    if any(left >= right for left, right in zip(stored[:-1], stored[1:], strict=True)):
        raise ValueError("trajectory timesteps must be strictly ascending")

    scheduler.set_timesteps(len(trajectory) - 1, device=initial.device)
    expected = list(reversed([int(timestep.item()) for timestep in scheduler.timesteps]))
    if stored != expected:
        raise ValueError("trajectory timesteps do not match the forward scheduler")


@torch.inference_mode()
def reconstruct(
    components: FrozenComponents,
    trajectory: list[TrajectoryState],
    prompt: str,
) -> torch.Tensor:
    _validate_prompt(prompt)
    scheduler, _ = _make_scheduler_pair(components.scheduler)
    _validate_trajectory(trajectory, scheduler)

    device = components.unet.device
    dtype = components.unet.dtype
    prompt_embeds = encode_prompt(components, prompt, device)
    current = trajectory[-1].latent.to(device=device, dtype=dtype)
    for state in reversed(trajectory[1:]):
        timestep = torch.tensor(state.timestep, device=device)
        prediction = components.unet(
            current, timestep, encoder_hidden_states=prompt_embeds
        ).sample
        current = scheduler.step(prediction, timestep, current, eta=0.0).prev_sample
    return current

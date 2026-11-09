"""Valid-canvas-gated binary mask diffusion for Module A RC1."""

from __future__ import annotations

import math

import torch
from torch import Tensor


class MaskDiffusionSchedule:
    """Cosine mask schedule with an explicit deterministic DDIM budget."""

    schedule_name = "cosine_s0.008_beta_max0.999"
    cosine_offset = 0.008
    beta_max = 0.999

    def __init__(self, *, diffusion_steps: int = 100) -> None:
        if (
            isinstance(diffusion_steps, bool)
            or not isinstance(diffusion_steps, int)
            or diffusion_steps <= 0
        ):
            raise ValueError("diffusion_steps must be a positive integer")
        self.diffusion_steps = diffusion_steps
        boundaries = torch.linspace(0, diffusion_steps, diffusion_steps + 1, dtype=torch.float64)
        alpha_boundaries = torch.cos(
            ((boundaries / diffusion_steps + self.cosine_offset) / (1 + self.cosine_offset))
            * math.pi
            / 2
        ).square()
        alpha_boundaries = alpha_boundaries / alpha_boundaries[0]
        betas = (1.0 - alpha_boundaries[1:] / alpha_boundaries[:-1]).clamp(max=self.beta_max)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def sampling_timesteps(self, *, sampling_steps: int = 50) -> Tensor:
        if (
            isinstance(sampling_steps, bool)
            or not isinstance(sampling_steps, int)
            or sampling_steps <= 0
            or sampling_steps > self.diffusion_steps
        ):
            raise ValueError(
                "sampling_steps must be a positive integer not exceeding diffusion_steps"
            )
        timesteps = (
            torch.linspace(self.diffusion_steps - 1, 0, sampling_steps, dtype=torch.float64)
            .round()
            .to(torch.long)
        )
        if timesteps.unique().numel() != sampling_steps:
            raise RuntimeError("DDIM timestep construction produced duplicates")
        return timesteps

    def _validate_state(self, state: Tensor, valid: Tensor, *, name: str) -> None:
        if (
            not isinstance(state, Tensor)
            or state.ndim != 4
            or state.shape[0] <= 0
            or state.shape[1] != 1
            or not state.is_floating_point()
            or not torch.isfinite(state).all().item()
        ):
            raise ValueError(f"{name} must be a finite floating [B,1,H,W] tensor")
        if (
            not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
            or valid.shape != state.shape
            or valid.device != state.device
        ):
            raise ValueError("valid must be a bool tensor matching the diffusion state")
        if not valid.flatten(1).any(dim=1).all().item():
            raise ValueError("each diffusion sample requires a non-empty valid canvas")

    def q_sample(
        self,
        *,
        z0: Tensor,
        timesteps: Tensor,
        noise: Tensor,
        valid: Tensor,
    ) -> Tensor:
        self._validate_state(z0, valid, name="z0")
        if (
            not isinstance(noise, Tensor)
            or noise.shape != z0.shape
            or noise.dtype != z0.dtype
            or noise.device != z0.device
            or not torch.isfinite(noise).all().item()
        ):
            raise ValueError("noise must be finite and match z0 shape, dtype, and device")
        if (
            not isinstance(timesteps, Tensor)
            or timesteps.dtype != torch.long
            or timesteps.shape != (z0.shape[0],)
            or timesteps.device != z0.device
            or not torch.all((0 <= timesteps) & (timesteps < self.diffusion_steps)).item()
        ):
            raise ValueError("timesteps must be an in-range matching long tensor")
        alpha = self.alphas_cumprod.to(device=z0.device, dtype=z0.dtype)[timesteps]
        alpha = alpha.view(-1, 1, 1, 1)
        inside = alpha.sqrt() * z0 + (1.0 - alpha).sqrt() * noise
        return torch.where(valid, inside, -torch.ones_like(inside))

    def ddim_step(
        self,
        *,
        noisy: Tensor,
        predicted_noise: Tensor,
        timestep: int,
        previous_timestep: int,
        valid: Tensor,
    ) -> Tensor:
        self._validate_state(noisy, valid, name="noisy")
        if (
            not isinstance(predicted_noise, Tensor)
            or predicted_noise.shape != noisy.shape
            or predicted_noise.dtype != noisy.dtype
            or predicted_noise.device != noisy.device
            or not torch.isfinite(predicted_noise).all().item()
        ):
            raise ValueError("predicted_noise must be finite and match noisy")
        if (
            isinstance(timestep, bool)
            or not isinstance(timestep, int)
            or not 0 <= timestep < self.diffusion_steps
        ):
            raise ValueError("timestep is outside the diffusion schedule")
        if (
            isinstance(previous_timestep, bool)
            or not isinstance(previous_timestep, int)
            or previous_timestep < -1
            or previous_timestep >= timestep
        ):
            raise ValueError("previous_timestep must be smaller than timestep or -1")
        alphas = self.alphas_cumprod.to(device=noisy.device, dtype=noisy.dtype)
        alpha = alphas[timestep]
        previous_alpha = (
            torch.ones((), device=noisy.device, dtype=noisy.dtype)
            if previous_timestep == -1
            else alphas[previous_timestep]
        )
        predicted_clean = (noisy - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        previous = (
            previous_alpha.sqrt() * predicted_clean
            + (1.0 - previous_alpha).sqrt() * predicted_noise
        )
        return torch.where(valid, previous, -torch.ones_like(previous))

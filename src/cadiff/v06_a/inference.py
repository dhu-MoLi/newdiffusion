"""Deterministic single-candidate A1 proposal inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .diffusion import MaskDiffusionSchedule
from .model import A1MaskUNet, validate_a1_model_contract


class InvalidProposalError(ValueError):
    """One frozen rollout yielded an invalid proposal; callers must not retry/select."""


@dataclass(frozen=True)
class A1InferenceCondition:
    valid: Tensor
    sdf: Tensor
    u_x: Tensor
    u_y: Tensor
    type_id: int
    normal_rgb: Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.valid, Tensor)
            or self.valid.dtype != torch.bool
            or self.valid.shape != (1, 1, 128, 128)
        ):
            raise ValueError("valid must be a bool [1,1,128,128] single-candidate tensor")
        if not self.valid.any().item():
            raise ValueError("inference requires a non-empty valid canvas")
        for value, name in ((self.sdf, "sdf"), (self.u_x, "u_x"), (self.u_y, "u_y")):
            if (
                not isinstance(value, Tensor)
                or value.shape != (1, 1, 128, 128)
                or not value.is_floating_point()
                or value.device != self.valid.device
                or not torch.isfinite(value).all().item()
            ):
                raise ValueError(f"{name} must be finite floating [1,1,128,128]")
        if self.sdf.dtype != self.u_x.dtype or self.sdf.dtype != self.u_y.dtype:
            raise ValueError("sdf and coordinate tensors must share one dtype")
        if isinstance(self.type_id, bool) or not isinstance(self.type_id, int) or self.type_id < 0:
            raise ValueError("type_id must be a non-negative integer")
        if self.normal_rgb is not None and (
            not isinstance(self.normal_rgb, Tensor)
            or self.normal_rgb.shape != (1, 3, 512, 512)
            or self.normal_rgb.dtype != self.sdf.dtype
            or self.normal_rgb.device != self.valid.device
            or not torch.isfinite(self.normal_rgb).all().item()
        ):
            raise ValueError("normal_rgb must be finite [1,3,512,512] and match conditions")


@dataclass(frozen=True)
class A1ProposalSample:
    mask_128: Tensor
    mask_512: Tensor
    logits_128: Tensor
    seed: int
    ddim_timesteps: tuple[int, ...]


class A1ProposalSampler:
    """Run one configured eta=0 DDIM trajectory from one seeded noise tensor."""

    def __init__(
        self,
        *,
        model: A1MaskUNet,
        schedule: MaskDiffusionSchedule,
        sampling_steps: int = 50,
    ) -> None:
        validate_a1_model_contract(model)
        if not isinstance(schedule, MaskDiffusionSchedule):
            raise TypeError("schedule must be a MaskDiffusionSchedule")
        if schedule.diffusion_steps != model.diffusion_steps:
            raise ValueError("schedule/model diffusion_steps mismatch")
        schedule.sampling_timesteps(sampling_steps=sampling_steps)
        try:
            parameter = next(model.parameters())
        except StopIteration as error:
            raise ValueError("model must expose a parameter for dtype/device validation") from error
        if not parameter.is_floating_point():
            raise ValueError("model parameters must use a floating dtype")
        self.model = model
        self.schedule = schedule
        self.sampling_steps = sampling_steps

    def sample(self, *, condition: A1InferenceCondition, seed: int) -> A1ProposalSample:
        validate_a1_model_contract(self.model)
        if self.schedule.diffusion_steps != self.model.diffusion_steps:
            raise ValueError("schedule/model diffusion_steps mismatch")
        if not isinstance(condition, A1InferenceCondition):
            raise TypeError("condition must be A1InferenceCondition")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be a non-negative signed 64-bit integer")
        parameter = next(self.model.parameters())
        if condition.sdf.device != parameter.device or condition.sdf.dtype != parameter.dtype:
            raise ValueError("inference conditions must match model dtype and device")

        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(seed)
        initial = torch.randn(
            condition.sdf.shape,
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )
        state = torch.where(condition.valid, initial, -torch.ones_like(initial))
        type_ids = torch.tensor([condition.type_id], dtype=torch.long, device=parameter.device)
        timesteps = self.schedule.sampling_timesteps(sampling_steps=self.sampling_steps)
        trajectory = tuple(int(value) for value in timesteps.tolist())

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for index, timestep in enumerate(trajectory):
                    previous_timestep = trajectory[index + 1] if index + 1 < len(trajectory) else -1
                    timestep_tensor = torch.tensor(
                        [timestep], dtype=torch.long, device=parameter.device
                    )
                    spatial = torch.cat(
                        (
                            state,
                            condition.valid.to(parameter.dtype),
                            condition.sdf,
                            condition.u_x,
                            condition.u_y,
                        ),
                        dim=1,
                    )
                    predicted_noise = self.model(
                        spatial,
                        type_ids=type_ids,
                        timesteps=timestep_tensor,
                        normal_rgb=condition.normal_rgb,
                    )
                    if (
                        not isinstance(predicted_noise, Tensor)
                        or predicted_noise.shape != state.shape
                        or predicted_noise.dtype != state.dtype
                        or predicted_noise.device != state.device
                        or not torch.isfinite(predicted_noise).all().item()
                    ):
                        raise ValueError("model output must be finite and match diffusion state")
                    state = self.schedule.ddim_step(
                        noisy=state,
                        predicted_noise=predicted_noise,
                        timestep=timestep,
                        previous_timestep=previous_timestep,
                        valid=condition.valid,
                    )
        finally:
            self.model.train(was_training)

        mask_128 = (state[0, 0] > 0).detach().clone()
        valid_128 = condition.valid[0, 0]
        if not mask_128.any().item():
            raise InvalidProposalError("proposal must be non-empty after the single rollout")
        if torch.all(mask_128[valid_128]).item():
            raise InvalidProposalError(
                "proposal must be non-full within valid after the single rollout"
            )
        if not torch.all(mask_128 <= valid_128).item():
            raise InvalidProposalError("proposal must remain a subset of the valid canvas")
        mask_512 = mask_128.repeat_interleave(4, 0).repeat_interleave(4, 1)
        return A1ProposalSample(
            mask_128=mask_128,
            mask_512=mask_512,
            logits_128=state[0, 0].detach().clone(),
            seed=seed,
            ddim_timesteps=trajectory,
        )

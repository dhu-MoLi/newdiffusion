from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from cadiff.p2_b0.generation import _validate_latent_inputs


@dataclass(frozen=True)
class RolloutResult:
    latent: Tensor
    generation_feature: Tensor


def _conv_out(model: object) -> nn.Module:
    try:
        convolution = model.unet.conv_out  # type: ignore[union-attr]
    except AttributeError as error:
        raise ValueError("model.unet.conv_out is required for F64 capture") from error
    if not isinstance(convolution, nn.Module):
        raise ValueError("model.unet.conv_out must be a torch module")
    return convolution


def truncated_denoise_rollout(
    *,
    model: Any,
    scheduler: Any,
    normal_latent: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    prompt_embeds: Tensor,
    defect_type: str,
    noise: Tensor,
    steps: int = 20,
    backprop_steps: int = 4,
    checkpoint_tracked_steps: bool = False,
) -> RolloutResult:
    """Run a masked DDIM rollout with a bounded differentiable tail."""

    if torch.is_inference_mode_enabled():
        raise RuntimeError("truncated rollout cannot run inside inference_mode")
    (
        normal_latent,
        context,
        binary_mask,
        prompt_embeds,
        noise,
        defect_type,
        steps,
    ) = _validate_latent_inputs(
        normal_latent,
        context,
        binary_mask,
        prompt_embeds,
        noise,
        defect_type,
        steps,
    )
    if (
        isinstance(backprop_steps, bool)
        or not isinstance(backprop_steps, int)
        or not 0 <= backprop_steps <= steps
    ):
        raise ValueError("backprop_steps must be between zero and steps")
    if not isinstance(checkpoint_tracked_steps, bool):
        raise ValueError("checkpoint_tracked_steps must be boolean")
    external = (normal_latent, context, binary_mask, prompt_embeds, noise)
    if any(value.requires_grad for value in external):
        raise ValueError("rollout inputs must be detached")
    if any(value.is_inference() for value in external):
        raise ValueError("rollout inputs must be ordinary tensors; clone inference outputs")

    convolution = _conv_out(model)
    scheduler.set_timesteps(steps, device=normal_latent.device)
    timesteps = tuple(scheduler.timesteps)
    if len(timesteps) != steps:
        raise ValueError("scheduler did not create the requested timestep count")
    current = scheduler.add_noise(normal_latent, noise, timesteps[0]).detach()
    mask = binary_mask.to(dtype=normal_latent.dtype)
    outside_mask = 1.0 - mask
    tracked_start = steps - backprop_steps
    captured: list[Tensor] = []

    def capture_feature(_module: nn.Module, arguments: tuple[object, ...]) -> None:
        if not arguments or not isinstance(arguments[0], Tensor):
            raise ValueError("U-Net conv_out pre-hook did not receive a tensor")
        captured.append(arguments[0])

    for index, timestep in enumerate(timesteps):
        tracked = index >= tracked_start
        final_step = index == steps - 1
        timestep_batch = torch.full(
            (normal_latent.shape[0],),
            int(timestep.item()),
            device=normal_latent.device,
            dtype=torch.long,
        )

        def predict(
            sample: Tensor,
            current_timestep_batch: Tensor = timestep_batch,
        ) -> Tensor:
            return model(
                sample,
                context,
                binary_mask,
                current_timestep_batch,
                prompt_embeds,
                defect_type,
            )

        handle = None
        try:
            if final_step:
                handle = convolution.register_forward_pre_hook(capture_feature)
            with torch.set_grad_enabled(tracked):
                if tracked and checkpoint_tracked_steps and not final_step:
                    prediction = checkpoint(predict, current, use_reentrant=False)
                else:
                    prediction = predict(current)
                if prediction.shape != current.shape or not torch.isfinite(prediction).all().item():
                    raise ValueError("model prediction must be finite and match the latent")
                generated = scheduler.step(
                    prediction, timestep, current, eta=0.0
                ).prev_sample
                if index + 1 < steps:
                    reference = scheduler.add_noise(
                        normal_latent, noise, timesteps[index + 1]
                    )
                else:
                    reference = normal_latent
                current = generated * mask + reference * outside_mask
        finally:
            if handle is not None:
                handle.remove()

        if index == tracked_start - 1:
            current = current.detach()

    if len(captured) != 1:
        raise ValueError("final U-Net conv_out feature must be captured exactly once")
    feature = captured[0]
    if (
        feature.ndim != 4
        or feature.shape[0] != current.shape[0]
        or feature.shape[2:] != current.shape[2:]
        or not torch.isfinite(feature).all().item()
    ):
        raise ValueError("captured generation feature must be finite [B,C,H,W]")
    if backprop_steps > 0 and (not current.requires_grad or not feature.requires_grad):
        raise ValueError("differentiable rollout tail did not retain its graph")
    if backprop_steps == 0 and (current.requires_grad or feature.requires_grad):
        raise ValueError("inference rollout unexpectedly retained a gradient graph")
    if not torch.isfinite(current).all().item():
        raise ValueError("generated latent must be finite")
    return RolloutResult(latent=current, generation_feature=feature)


def decode_latent_with_grad(vae: Any, latent: Tensor) -> Tensor:
    """Decode a latent while retaining gradients to the latent, not the frozen VAE."""

    if not isinstance(vae, nn.Module):
        raise ValueError("VAE must be a frozen torch module")
    if not isinstance(latent, Tensor) or latent.ndim != 4 or not latent.is_floating_point():
        raise ValueError("latent must be a four-dimensional floating tensor")
    if not torch.isfinite(latent).all().item():
        raise ValueError("latent must contain only finite values")
    parameters = tuple(
        parameter for parameter in vae.parameters() if parameter.is_floating_point()
    )
    if not parameters:
        raise ValueError("VAE must expose floating parameters")
    vae_device = parameters[0].device
    vae_dtype = parameters[0].dtype
    if any(
        parameter.device != vae_device
        or parameter.dtype != vae_dtype
        or parameter.requires_grad
        for parameter in parameters
    ):
        raise ValueError("VAE floating parameters must be frozen on one device and dtype")
    try:
        scaling_factor = float(vae.config.scaling_factor)
        decoded = vae.decode(
            (latent / scaling_factor).to(device=vae_device, dtype=vae_dtype)
        ).sample
    except AttributeError as error:
        raise ValueError("VAE decode must expose config.scaling_factor and sample") from error
    if not isinstance(decoded, Tensor) or decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError("VAE decode must return an RGB [B,3,H,W] tensor")
    image = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    if not torch.isfinite(image).all().item():
        raise ValueError("decoded image must be finite")
    return image

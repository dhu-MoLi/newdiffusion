from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cadiff.p1_v1.model import P1NoisePredictor


@dataclass(frozen=True)
class StepNoise:
    timesteps: Tensor
    noise: Tensor


@dataclass(frozen=True)
class StepMetrics:
    defect_type: str
    anomaly_mse: float
    normal_mse: float | None
    trainable_grad_l2: float
    zero_head_grad_nonzero: bool
    anomaly_gradient_hash: str


def _require_finite_floating_tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")
    if value.ndim == 0 or value.shape[0] <= 0:
        raise ValueError(f"{name} must have a non-empty batch")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _read_num_train_timesteps(scheduler: object) -> int:
    try:
        value = scheduler.config.num_train_timesteps  # type: ignore[union-attr]
    except AttributeError as error:
        raise ValueError("scheduler num_train_timesteps is required") from error
    if type(value) is not int or value <= 0:
        raise ValueError("scheduler num_train_timesteps must be a positive integer")
    return value


def encode_latents(
    *,
    vae: nn.Module,
    images: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Sample and scale a VAE posterior without constructing an autograd graph."""
    images = _require_finite_floating_tensor(images, name="images")
    if not isinstance(vae, nn.Module):
        raise TypeError("vae must be a torch module")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch Generator")
    try:
        scaling_factor = vae.config.scaling_factor  # type: ignore[union-attr]
    except AttributeError as error:
        raise ValueError("VAE scaling_factor is required") from error
    if (
        isinstance(scaling_factor, bool)
        or not isinstance(scaling_factor, int | float)
        or not math.isfinite(float(scaling_factor))
        or float(scaling_factor) <= 0.0
    ):
        raise ValueError("VAE scaling_factor must be finite and positive")

    with torch.no_grad():
        try:
            encoded = vae.encode(images)  # type: ignore[attr-defined]
            latent_dist = encoded.latent_dist
            latents = latent_dist.sample(generator=generator)
        except AttributeError as error:
            raise ValueError("VAE encode must expose a sampleable latent_dist") from error
        latents = latents * float(scaling_factor)

    latents = _require_finite_floating_tensor(latents, name="VAE latents")
    if latents.shape[0] != images.shape[0]:
        raise ValueError("VAE latent batch must match images")
    if latents.device != images.device:
        raise ValueError("VAE latents must remain on the image device")
    if latents.requires_grad:
        raise ValueError("VAE latents must not require gradients")
    return latents


def sample_step_noise(
    *,
    latents: Tensor,
    scheduler: object,
    generator: torch.Generator,
) -> StepNoise:
    """Sample exact integer diffusion timesteps and Gaussian noise deterministically."""
    latents = _require_finite_floating_tensor(latents, name="latents")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch Generator")
    num_train_timesteps = _read_num_train_timesteps(scheduler)
    timesteps = torch.randint(
        0,
        num_train_timesteps,
        (latents.shape[0],),
        generator=generator,
        device=latents.device,
        dtype=torch.long,
    )
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=latents.device,
        dtype=latents.dtype,
    )
    return StepNoise(timesteps=timesteps, noise=noise)


def capture_parameter_versions(module: nn.Module) -> tuple[int, ...]:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch module")
    return tuple(parameter._version for parameter in module.parameters())


def assert_parameter_versions_unchanged(
    module: nn.Module,
    expected: tuple[int, ...],
) -> None:
    if not isinstance(expected, tuple) or any(type(value) is not int for value in expected):
        raise TypeError("expected parameter versions must be a tuple of integers")
    actual = capture_parameter_versions(module)
    if len(actual) != len(expected):
        raise AssertionError("parameter count changed while checking versions")
    changed = tuple(
        index
        for index, pair in enumerate(zip(actual, expected, strict=True))
        if pair[0] != pair[1]
    )
    if changed:
        raise AssertionError(f"parameter version changed at indices {changed}")


def _validate_step_noise(
    step_noise: StepNoise,
    *,
    latents: Tensor,
    scheduler: object,
) -> None:
    if not isinstance(step_noise, StepNoise):
        raise TypeError("step_noise must be a StepNoise")
    timesteps = step_noise.timesteps
    if not isinstance(timesteps, Tensor):
        raise ValueError("step_noise timesteps must be a tensor")
    if timesteps.dtype is not torch.long:
        raise ValueError("step_noise timesteps must have dtype torch.long")
    if timesteps.shape != (latents.shape[0],):
        raise ValueError("step_noise timestep batch must match latents")
    if timesteps.device != latents.device:
        raise ValueError("step_noise timesteps must be on the latent device")
    num_train_timesteps = _read_num_train_timesteps(scheduler)
    if not torch.all((0 <= timesteps) & (timesteps < num_train_timesteps)).item():
        raise ValueError("step_noise timesteps are outside the scheduler range")

    noise = _require_finite_floating_tensor(step_noise.noise, name="step_noise noise")
    if noise.requires_grad:
        raise ValueError("step_noise noise must not require gradients")
    if noise.shape != latents.shape:
        raise ValueError("step_noise noise shape must match latents")
    if noise.dtype != latents.dtype:
        raise ValueError("step_noise noise dtype must match latents")
    if noise.device != latents.device:
        raise ValueError("step_noise noise must be on the latent device")


def _assert_module_frozen_eval(module: nn.Module, *, name: str) -> None:
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"{name} parameters must be frozen")
    if any(child.training for child in module.modules()):
        raise ValueError(f"{name} and all descendants must remain in eval mode")
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise ValueError(f"{name} parameters must not have gradients")


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    parameters: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, nn.Parameter):
                raise ValueError("optimizer entries must be parameters")
            parameters.append(parameter)
    return tuple(parameters)


def _assert_parameters_finite(
    parameters: tuple[nn.Parameter, ...], *, name: str
) -> None:
    for index, parameter in enumerate(parameters):
        if not torch.isfinite(parameter).all().item():
            raise ValueError(f"{name} parameter must be finite at index {index}")


def _validate_optimizer_boundary(
    model: P1NoisePredictor,
    optimizer: torch.optim.Optimizer,
    *,
    target_device: torch.device,
) -> tuple[nn.Parameter, ...]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    expected = tuple(model.trainable_parameters())
    if not expected:
        raise ValueError("model must expose trainable parameters")
    expected_ids = tuple(id(parameter) for parameter in expected)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("model trainable_parameters must be unique")
    if any(not isinstance(parameter, nn.Parameter) for parameter in expected):
        raise ValueError("model trainable_parameters must contain parameters")
    if any(not parameter.requires_grad for parameter in expected):
        raise ValueError("model trainable_parameters must require gradients")
    for index, parameter in enumerate(expected):
        if parameter.dtype is not torch.float32:
            raise ValueError(
                "trainable parameter at index "
                f"{index} must have dtype torch.float32"
            )
        if parameter.device != target_device:
            raise ValueError(
                f"trainable parameter at index {index} device must match "
                "anomaly_images"
            )
    _assert_parameters_finite(expected, name="trainable")

    optimized = _optimizer_parameters(optimizer)
    optimized_ids = tuple(id(parameter) for parameter in optimized)
    if any(not parameter.requires_grad for parameter in optimized):
        raise ValueError("optimizer contains a frozen parameter")
    if len(set(optimized_ids)) != len(optimized_ids):
        raise ValueError("optimizer must not contain duplicate parameters")
    if len(optimized_ids) != len(expected_ids) or set(optimized_ids) != set(expected_ids):
        raise ValueError("optimizer must contain exactly model trainable_parameters")
    return expected


def _validate_scheduler_output(
    value: object,
    *,
    latents: Tensor,
    name: str,
) -> Tensor:
    output = _require_finite_floating_tensor(value, name=name)
    if output.shape != latents.shape:
        raise ValueError(f"{name} shape must match latents")
    if output.dtype != latents.dtype:
        raise ValueError(f"{name} dtype must match latents")
    if output.device != latents.device:
        raise ValueError(f"{name} device must match latents")
    return output


def _validate_prediction(value: object, *, noisy_latents: Tensor, name: str) -> Tensor:
    prediction = _require_finite_floating_tensor(value, name=name)
    if prediction.shape != noisy_latents.shape:
        raise ValueError(f"{name} shape must match noisy latents")
    if prediction.dtype != noisy_latents.dtype:
        raise ValueError(f"{name} dtype must match noisy latents")
    if prediction.device != noisy_latents.device:
        raise ValueError(f"{name} device must match noisy latents")
    return prediction


def _fp32_mse(prediction: Tensor, target: Tensor, *, name: str) -> Tensor:
    mse = (prediction.float() - target.float()).square().mean()
    if mse.ndim != 0 or not torch.isfinite(mse).item():
        raise ValueError(f"{name} must be a finite scalar")
    return mse


def _validate_trainable_gradients(
    parameters: tuple[nn.Parameter, ...],
) -> tuple[Tensor, ...]:
    gradients: list[Tensor] = []
    for index, parameter in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            raise ValueError(f"missing trainable gradient at index {index}")
        if not torch.isfinite(gradient).all().item():
            raise ValueError(f"trainable gradient must be finite at index {index}")
        gradients.append(gradient)
    return tuple(gradients)


def _gradient_hash(gradients: tuple[Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for gradient in gradients:
        normalized = gradient.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(str(tuple(normalized.shape)).encode("ascii"))
        digest.update(normalized.numpy().tobytes())
    return digest.hexdigest()


def _zero_head_grad_nonzero(model: P1NoisePredictor) -> bool:
    try:
        parameters = tuple(model.adapter.output_heads.parameters())
    except AttributeError as error:
        raise ValueError("model adapter must expose output_heads") from error
    if not parameters:
        raise ValueError("model adapter output_heads must contain parameters")
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in parameters
    )


def run_optimizer_step(
    *,
    vae: nn.Module,
    scheduler: object,
    model: P1NoisePredictor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    anomaly_images: Tensor,
    normal_images: Tensor | None,
    prompt_embeds: Tensor,
    defect_type: str,
    generator: torch.Generator,
    step_noise: StepNoise | None = None,
) -> StepMetrics:
    """Run one audited anomaly-only optimizer step plus an optional normal control."""
    anomaly_images = _require_finite_floating_tensor(
        anomaly_images, name="anomaly_images"
    )
    prompt_embeds = _require_finite_floating_tensor(prompt_embeds, name="prompt_embeds")
    if prompt_embeds.requires_grad:
        raise ValueError("prompt_embeds must not require gradients")
    if step_noise is not None:
        if not isinstance(step_noise, StepNoise):
            raise TypeError("step_noise must be a StepNoise")
        if isinstance(step_noise.noise, Tensor) and step_noise.noise.requires_grad:
            raise ValueError("step_noise noise must not require gradients")
    if prompt_embeds.shape[0] != anomaly_images.shape[0]:
        raise ValueError("prompt_embeds batch must match anomaly_images")
    if not isinstance(defect_type, str) or not defect_type or defect_type == "normal":
        raise ValueError("defect_type must name one anomaly type")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(vae, nn.Module):
        raise TypeError("vae must be a torch module")
    if not isinstance(scaler, torch.amp.GradScaler):
        raise TypeError("scaler must be a torch.amp.GradScaler")
    if normal_images is not None:
        normal_images = _require_finite_floating_tensor(normal_images, name="normal_images")
        if normal_images.shape[0] != anomaly_images.shape[0]:
            raise ValueError("normal control must be 1:1 with the anomaly batch")

    _assert_module_frozen_eval(vae, name="VAE")
    model.train(True)
    model.assert_frozen_backbone()
    _assert_module_frozen_eval(model.unet, name="frozen U-Net")
    trainable_parameters = _validate_optimizer_boundary(
        model,
        optimizer,
        target_device=anomaly_images.device,
    )
    backbone_versions = capture_parameter_versions(model.unet)

    optimizer.zero_grad(set_to_none=True)
    anomaly_latents = encode_latents(
        vae=vae,
        images=anomaly_images,
        generator=generator,
    )
    if step_noise is None:
        step_noise = sample_step_noise(
            latents=anomaly_latents,
            scheduler=scheduler,
            generator=generator,
        )
    _validate_step_noise(step_noise, latents=anomaly_latents, scheduler=scheduler)
    try:
        anomaly_noisy = scheduler.add_noise(
            anomaly_latents,
            step_noise.noise,
            step_noise.timesteps,
        )
    except AttributeError as error:
        raise ValueError("scheduler must expose add_noise") from error
    anomaly_noisy = _validate_scheduler_output(
        anomaly_noisy,
        latents=anomaly_latents,
        name="anomaly noisy latents",
    )
    anomaly_prediction = _validate_prediction(
        model(
            anomaly_noisy,
            step_noise.timesteps,
            prompt_embeds,
            defect_type,
        ),
        noisy_latents=anomaly_noisy,
        name="anomaly prediction",
    )
    anomaly_mse = _fp32_mse(
        anomaly_prediction,
        step_noise.noise,
        name="anomaly MSE",
    )

    normal_mse: Tensor | None = None
    if normal_images is not None:
        generator_state = generator.get_state().clone()
        try:
            with torch.no_grad():
                normal_latents = encode_latents(
                    vae=vae,
                    images=normal_images,
                    generator=generator,
                )
                if normal_latents.shape != anomaly_latents.shape:
                    raise ValueError(
                        "normal and anomaly latent shapes must match for 1:1 control"
                    )
                if normal_latents.dtype != anomaly_latents.dtype:
                    raise ValueError("normal and anomaly latent dtypes must match")
                if normal_latents.device != anomaly_latents.device:
                    raise ValueError("normal and anomaly latents must share a device")
                normal_noisy = scheduler.add_noise(
                    normal_latents,
                    step_noise.noise,
                    step_noise.timesteps,
                )
                normal_noisy = _validate_scheduler_output(
                    normal_noisy,
                    latents=normal_latents,
                    name="normal noisy latents",
                )
                normal_prediction = _validate_prediction(
                    model(
                        normal_noisy,
                        step_noise.timesteps,
                        prompt_embeds,
                        "normal",
                    ),
                    noisy_latents=normal_noisy,
                    name="normal prediction",
                )
                normal_mse = _fp32_mse(
                    normal_prediction,
                    step_noise.noise,
                    name="normal MSE",
                )
        finally:
            generator.set_state(generator_state)

    scaled_loss = scaler.scale(anomaly_mse)
    if not torch.isfinite(scaled_loss).item():
        raise ValueError("scaled anomaly MSE must be finite before backward")
    scaled_loss.backward()
    scaler.unscale_(optimizer)

    try:
        gradients = _validate_trainable_gradients(trainable_parameters)
        _assert_module_frozen_eval(model.unet, name="frozen U-Net")
        _assert_module_frozen_eval(vae, name="VAE")
        model.assert_frozen_backbone()
        assert_parameter_versions_unchanged(model.unet, backbone_versions)

        squared_l2 = sum(
            gradient.detach().float().square().sum().item() for gradient in gradients
        )
        trainable_grad_l2 = math.sqrt(squared_l2)
        if not math.isfinite(trainable_grad_l2):
            raise ValueError("trainable gradient L2 must be finite")
        zero_head_grad_nonzero = _zero_head_grad_nonzero(model)
        if not zero_head_grad_nonzero:
            raise ValueError("zero-head must receive a nonzero gradient")
        anomaly_gradient_hash = _gradient_hash(gradients)
        anomaly_mse_value = float(anomaly_mse.detach().item())
        normal_mse_value = None if normal_mse is None else float(normal_mse.item())
        if not math.isfinite(anomaly_mse_value) or (
            normal_mse_value is not None and not math.isfinite(normal_mse_value)
        ):
            raise ValueError("step metrics must be finite before optimizer step")
        _assert_parameters_finite(trainable_parameters, name="trainable")

        scaler.step(optimizer)
    except Exception:
        if scaler.is_enabled():
            has_nonfinite_gradient = any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all().item()
                for parameter in trainable_parameters
            )
            if has_nonfinite_gradient:
                scaler.update()
            else:
                scaler.update(new_scale=scaler.get_scale())
        raise
    scaler.update()

    if any(not torch.isfinite(parameter).all().item() for parameter in trainable_parameters):
        raise ValueError("optimizer produced a non-finite trainable parameter")
    _assert_module_frozen_eval(model.unet, name="frozen U-Net")
    model.assert_frozen_backbone()
    assert_parameter_versions_unchanged(model.unet, backbone_versions)

    return StepMetrics(
        defect_type=defect_type,
        anomaly_mse=anomaly_mse_value,
        normal_mse=normal_mse_value,
        trainable_grad_l2=trainable_grad_l2,
        zero_head_grad_nonzero=zero_head_grad_nonzero,
        anomaly_gradient_hash=anomaly_gradient_hash,
    )

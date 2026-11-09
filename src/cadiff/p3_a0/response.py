from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from cadiff.diffusion.components import FrozenComponents
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import encode_mode
from cadiff.p3_a0.contracts import ResponseJob
from cadiff.p3_a0.dataset import ResponseInput


@dataclass(frozen=True)
class RawNormalizationStats:
    finite: bool
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    q01: float
    q99: float
    dynamic_range: float
    degenerate: bool


@dataclass(frozen=True)
class RawTimestepStats:
    timestep: int
    finite: bool
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    q01: float
    q99: float
    dynamic_range: float
    degenerate: bool
    noised_latent_sha256: str


@dataclass(frozen=True)
class ResponseMap:
    job: ResponseJob
    response: np.ndarray
    response_sha256: str
    image_latent_sha256: str
    noise_sha256: str
    timestep_stats: tuple[RawTimestepStats, ...]


@dataclass(frozen=True)
class ResponseBatch:
    maps: tuple[ResponseMap, ...]
    unet_forward_count: int
    parameters_unchanged: bool


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def robust_normalize(
    response: Tensor,
    latent_valid: Tensor,
    *,
    low: float,
    high: float,
    epsilon: float,
) -> tuple[Tensor, RawNormalizationStats]:
    if not isinstance(response, Tensor) or not response.is_floating_point():
        raise TypeError("response must be a floating torch.Tensor")
    if not isinstance(latent_valid, Tensor) or latent_valid.dtype is not torch.bool:
        raise TypeError("latent_valid must be a boolean torch.Tensor")
    if response.shape != latent_valid.shape or response.ndim != 4:
        raise ValueError("response and latent_valid must have the same 4D shape")
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("normalization quantiles must satisfy 0 <= low < high <= 1")
    if epsilon <= 0.0:
        raise ValueError("normalization epsilon must be positive")
    if not bool(torch.isfinite(response).all()):
        raise ValueError("response must contain only finite values")
    if not bool(latent_valid.any()):
        raise ValueError("latent_valid must contain at least one valid pixel")

    values = response[latent_valid].to(torch.float32)
    q_low = torch.quantile(values, low)
    q_high = torch.quantile(values, high)
    dynamic_range = q_high - q_low
    degenerate = bool(dynamic_range.item() <= epsilon)

    normalized = torch.zeros_like(response, dtype=torch.float32)
    if not degenerate:
        scaled = ((response.to(torch.float32) - q_low) / dynamic_range).clamp(0.0, 1.0)
        normalized = torch.where(latent_valid, scaled, normalized)

    stats = RawNormalizationStats(
        finite=True,
        minimum=float(values.min().item()),
        maximum=float(values.max().item()),
        mean=float(values.mean().item()),
        standard_deviation=float(values.std(unbiased=False).item()),
        q01=float(q_low.item()),
        q99=float(q_high.item()),
        dynamic_range=float(dynamic_range.item()),
        degenerate=degenerate,
    )
    return normalized, stats


def _parameters(components: FrozenComponents) -> tuple[Tensor, ...]:
    return tuple(
        parameter
        for module in (components.vae, components.text_encoder, components.unet)
        for parameter in module.parameters()
    )


def _validate_parameters(parameters: tuple[Tensor, ...]) -> dict[int, int]:
    if any(parameter.requires_grad for parameter in parameters):
        raise ValueError("frozen component parameters must not require gradients")
    return {id(parameter): parameter._version for parameter in parameters}


def _assert_parameters_unchanged(
    components: FrozenComponents,
    parameters: tuple[Tensor, ...],
    versions: dict[int, int],
) -> None:
    current = _parameters(components)
    if (
        len(current) != len(parameters)
        or any(parameter.requires_grad for parameter in current)
        or any(
            current_parameter is not original_parameter
            or current_parameter._version != versions[id(original_parameter)]
            for current_parameter, original_parameter in zip(current, parameters, strict=True)
        )
    ):
        raise RuntimeError("frozen component parameters changed during response inference")


def _validate_request(
    *,
    components: FrozenComponents,
    inputs: tuple[ResponseInput, ...],
    jobs: tuple[ResponseJob, ...],
    defect_prompts: Mapping[str, str],
    timesteps: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[tuple[str, str, int], ResponseJob]:
    if not isinstance(components, FrozenComponents):
        raise TypeError("components must be FrozenComponents")
    if not isinstance(inputs, tuple) or any(not isinstance(item, ResponseInput) for item in inputs):
        raise TypeError("inputs must contain only ResponseInput values")
    if not isinstance(jobs, tuple) or any(not isinstance(job, ResponseJob) for job in jobs):
        raise TypeError("jobs must contain only ResponseJob values")
    if not inputs:
        raise ValueError("inputs must be non-empty")
    if not defect_prompts or any(
        not isinstance(key, str) or not key or not isinstance(value, str)
        for key, value in defect_prompts.items()
    ):
        raise ValueError("defect_prompts must map non-empty defect types to prompt strings")
    if (
        not timesteps
        or any(type(value) is not int or value < 0 for value in timesteps)
        or len(set(timesteps)) != len(timesteps)
    ):
        raise ValueError("timesteps must be unique non-negative integers")
    if (
        not seeds
        or any(type(value) is not int or value < 0 for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")

    input_by_id = {item.sample_id: item for item in inputs}
    if len(input_by_id) != len(inputs):
        raise ValueError("response input sample IDs must be unique")
    defect_types = tuple(defect_prompts)
    expected = {
        (item.sample_id, defect_type, seed)
        for item in inputs
        for defect_type in defect_types
        for seed in seeds
    }
    jobs_by_key: dict[tuple[str, str, int], ResponseJob] = {}
    record_ids: set[str] = set()
    for job in jobs:
        key = (job.sample_id, job.prompt_defect_type, job.seed)
        source = input_by_id.get(job.sample_id)
        if (
            source is None
            or job.cohort != source.cohort
            or job.source_defect_type != source.source_defect_type
        ):
            raise ValueError("response job identity does not match its input")
        if key in jobs_by_key or job.record_id in record_ids:
            raise ValueError("response jobs must be unique")
        jobs_by_key[key] = job
        record_ids.add(job.record_id)
    if set(jobs_by_key) != expected:
        raise ValueError("jobs must cover the complete input, prompt, and seed matrix")
    return jobs_by_key


def _validate_input_tensor(item: ResponseInput) -> None:
    if (
        item.pixels.shape != (3, 512, 512)
        or not item.pixels.is_floating_point()
        or not bool(torch.isfinite(item.pixels).all())
    ):
        raise ValueError("response input pixels must be finite 3x512x512 floating tensors")
    if item.valid.shape != (1, 512, 512) or item.valid.dtype is not torch.bool:
        raise ValueError("response input valid region must be a boolean 1x512x512 tensor")
    if not bool(item.valid.any()):
        raise ValueError("response input valid region must be non-empty")


def _timestep_stats(
    timestep: int, stats: RawNormalizationStats, noised_latent_sha256: str
) -> RawTimestepStats:
    return RawTimestepStats(
        timestep=timestep,
        finite=stats.finite,
        minimum=stats.minimum,
        maximum=stats.maximum,
        mean=stats.mean,
        standard_deviation=stats.standard_deviation,
        q01=stats.q01,
        q99=stats.q99,
        dynamic_range=stats.dynamic_range,
        degenerate=stats.degenerate,
        noised_latent_sha256=noised_latent_sha256,
    )


@torch.inference_mode()
def run_paired_prompt_responses(
    *,
    components: FrozenComponents,
    inputs: tuple[ResponseInput, ...],
    jobs: tuple[ResponseJob, ...],
    normal_prompt: str,
    defect_prompts: Mapping[str, str],
    timesteps: tuple[int, ...],
    seeds: tuple[int, ...],
    device: torch.device,
) -> ResponseBatch:
    jobs_by_key = _validate_request(
        components=components,
        inputs=inputs,
        jobs=jobs,
        defect_prompts=defect_prompts,
        timesteps=timesteps,
        seeds=seeds,
    )
    if not isinstance(normal_prompt, str):
        raise TypeError("normal_prompt must be a string")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")

    parameters = _parameters(components)
    versions = _validate_parameters(parameters)
    defect_types = tuple(defect_prompts)
    normal_embed = encode_prompt(components, normal_prompt, device)
    defect_embeds = {
        defect_type: encode_prompt(components, defect_prompts[defect_type], device)
        for defect_type in defect_types
    }

    completed: dict[tuple[str, str, int], ResponseMap] = {}
    unet_forward_count = 0
    for item in inputs:
        _validate_input_tensor(item)
        valid = item.valid.unsqueeze(0).to(device=device)
        z0 = encode_mode(
            components.vae,
            item.pixels.unsqueeze(0).to(device=device, dtype=torch.float16),
        )
        image_latent_sha256 = _tensor_sha256(z0)
        latent_valid = F.interpolate(
            valid.to(torch.float32), size=z0.shape[-2:], mode="nearest"
        ).to(torch.bool)

        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                z0.shape,
                generator=generator,
                device=device,
                dtype=z0.dtype,
            )
            noise_sha256 = _tensor_sha256(noise)
            normalized_by_type: dict[str, list[Tensor]] = {
                defect_type: [] for defect_type in defect_types
            }
            stats_by_type: dict[str, list[RawTimestepStats]] = {
                defect_type: [] for defect_type in defect_types
            }

            for timestep in timesteps:
                t = torch.tensor([timestep], device=device, dtype=torch.long)
                zt = components.scheduler.add_noise(z0, noise, t)
                noised_latent_sha256 = _tensor_sha256(zt)
                epsilon_normal = components.unet(zt, t, encoder_hidden_states=normal_embed).sample
                unet_forward_count += 1
                for defect_type in defect_types:
                    epsilon_defect = components.unet(
                        zt,
                        t,
                        encoder_hidden_states=defect_embeds[defect_type],
                    ).sample
                    unet_forward_count += 1
                    raw = (epsilon_defect - epsilon_normal).abs().mean(dim=1, keepdim=True)
                    normalized, stats = robust_normalize(
                        raw,
                        latent_valid,
                        low=0.01,
                        high=0.99,
                        epsilon=1.0e-8,
                    )
                    restored = F.interpolate(
                        normalized,
                        size=item.valid.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    restored = restored * valid
                    normalized_by_type[defect_type].append(restored)
                    stats_by_type[defect_type].append(
                        _timestep_stats(timestep, stats, noised_latent_sha256)
                    )

            for defect_type in defect_types:
                averaged = (
                    torch.stack(normalized_by_type[defect_type], dim=0)
                    .mean(dim=0)
                    .squeeze(0)
                    .squeeze(0)
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                response = averaged.numpy()
                response.setflags(write=False)
                key = (item.sample_id, defect_type, seed)
                completed[key] = ResponseMap(
                    job=jobs_by_key[key],
                    response=response,
                    response_sha256=hashlib.sha256(response.tobytes()).hexdigest(),
                    image_latent_sha256=image_latent_sha256,
                    noise_sha256=noise_sha256,
                    timestep_stats=tuple(stats_by_type[defect_type]),
                )

    _assert_parameters_unchanged(components, parameters, versions)
    ordered = tuple(completed[(job.sample_id, job.prompt_defect_type, job.seed)] for job in jobs)
    return ResponseBatch(
        maps=ordered,
        unet_forward_count=unet_forward_count,
        parameters_unchanged=True,
    )

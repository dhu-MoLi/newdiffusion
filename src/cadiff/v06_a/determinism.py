"""Explicit deterministic-execution policy for formal Module-A CUDA attempts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

import torch

from .model import A1MaskUNet, require_supported_variant

if TYPE_CHECKING:
    from .config import V06AExperimentConfig

_CUDA_POLICY_LOCK = RLock()


@dataclass(frozen=True)
class A1DeterminismProfile:
    """Recorded strategy; it deliberately promises no cross-machine byte identity."""

    device_type: str = "cuda"
    cuda_policy_status: str = "active"
    local_seeded_model_init: bool = True
    local_seeded_data_schedule: bool = True
    local_seeded_diffusion_noise: bool = True
    torch_deterministic_algorithms: bool = True
    torch_deterministic_warn_only: bool = False
    cudnn_deterministic: bool | None = True
    cudnn_benchmark: bool | None = False
    cuda_matmul_allow_tf32: bool | None = False
    cudnn_allow_tf32: bool | None = False
    float32_matmul_precision: str = "highest"
    cublas_workspace_config: str | None = ":4096:8"
    cross_machine_byte_identity_promised: bool = False


@dataclass(frozen=True)
class A1DeterminismPreflight:
    device_type: str
    device: str
    device_name: str | None
    status: str
    error: str | None
    silent_fallback_used: bool


class CudaDeterminismPreflightError(RuntimeError):
    """Formal CUDA execution cannot proceed under the recorded deterministic policy."""

    def __init__(self, report: A1DeterminismPreflight) -> None:
        super().__init__(f"CUDA deterministic preflight {report.status}: {report.error}")
        self.report = report


def _failed_report(device: torch.device, error: str) -> A1DeterminismPreflight:
    return A1DeterminismPreflight(
        device_type="cuda",
        device=str(device),
        device_name=None,
        status="failed",
        error=error,
        silent_fallback_used=False,
    )


def _profile_is_active(profile: A1DeterminismProfile) -> bool:
    return (
        profile == A1DeterminismProfile()
        and torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled()
        and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32
        and torch.get_float32_matmul_precision() == "highest"
    )


@contextmanager
def cuda_determinism_scope(
    device: torch.device,
) -> Iterator[A1DeterminismProfile]:
    """Apply the fixed CUDA policy for a whole preflight-plus-training scope."""

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise TypeError("cuda_determinism_scope requires one torch CUDA device")
    with _CUDA_POLICY_LOCK:
        profile = A1DeterminismProfile()
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != profile.cublas_workspace_config:
            raise CudaDeterminismPreflightError(
                _failed_report(
                    device,
                    "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization",
                )
            )
        if not torch.cuda.is_available():
            raise CudaDeterminismPreflightError(
                _failed_report(device, "requested CUDA device is unavailable")
            )

        previous = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision(),
        )
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        try:
            yield profile
        finally:
            torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
            torch.backends.cudnn.deterministic = previous[2]
            torch.backends.cudnn.benchmark = previous[3]
            torch.backends.cuda.matmul.allow_tf32 = previous[4]
            torch.backends.cudnn.allow_tf32 = previous[5]
            torch.set_float32_matmul_precision(previous[6])


@contextmanager
def formal_cuda_bootstrap_scope(
    device: torch.device,
) -> Iterator[A1DeterminismProfile]:
    """Fail closed unless the formal CUDA policy precedes process initialization."""

    if not isinstance(device, torch.device) or device.type != "cuda":
        raise TypeError("formal_cuda_bootstrap_scope requires one torch CUDA device")
    with _CUDA_POLICY_LOCK:
        profile = A1DeterminismProfile()
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != profile.cublas_workspace_config:
            raise CudaDeterminismPreflightError(
                _failed_report(
                    device,
                    "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization",
                )
            )
        if torch.cuda.is_initialized():
            raise CudaDeterminismPreflightError(
                _failed_report(
                    device,
                    "CUDA is already initialized; formal bootstrap requires a fresh process",
                )
            )
        with cuda_determinism_scope(device) as active_profile:
            yield active_profile


def _cuda_kernel_probe(
    *,
    config: V06AExperimentConfig,
    variant: str,
    device: torch.device,
) -> None:
    """Exercise the exact A1 topology, batch shape, backward path, and Adam update."""

    cuda_index = torch.cuda.current_device() if device.index is None else device.index
    with torch.random.fork_rng(devices=[cuda_index], device_type="cuda"):
        torch.random.default_generator.manual_seed(config.training_seed)
        with torch.cuda.device(cuda_index):
            torch.cuda.manual_seed(config.training_seed)
        model = A1MaskUNet.from_experiment(config, variant=variant).to(device)
        batch = config.batch_size
        spatial = torch.zeros(batch, 5, 128, 128, device=device)
        spatial[:, 1] = 1.0
        type_ids = torch.arange(batch, device=device, dtype=torch.long) % len(config.defect_types)
        timesteps = torch.zeros(batch, device=device, dtype=torch.long)
        normal_rgb = (
            torch.zeros(batch, 3, 512, 512, device=device) if variant == "a1_sv_plain" else None
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        output = model(
            spatial,
            type_ids=type_ids,
            timesteps=timesteps,
            normal_rgb=normal_rgb,
        )
        output.square().mean().backward()
        optimizer.step()
        torch.cuda.synchronize(device)


def run_cuda_determinism_preflight(
    *,
    config: V06AExperimentConfig,
    variant: str,
    device: torch.device,
    profile: A1DeterminismProfile,
) -> A1DeterminismPreflight:
    """Fail before formal training if the exact CUDA path lacks deterministic kernels."""

    variant = require_supported_variant(variant)
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise TypeError("CUDA deterministic preflight requires one torch CUDA device")
    try:
        device_name = torch.cuda.get_device_name(device)
    except Exception as error:
        raise CudaDeterminismPreflightError(
            _failed_report(device, f"cannot resolve CUDA device: {error}")
        ) from error
    if not _profile_is_active(profile):
        report = A1DeterminismPreflight(
            "cuda",
            str(device),
            device_name,
            "failed",
            "determinism profile is not active exactly as recorded",
            False,
        )
        raise CudaDeterminismPreflightError(report)
    try:
        _cuda_kernel_probe(config=config, variant=variant, device=device)
    except Exception as error:
        report = A1DeterminismPreflight(
            "cuda",
            str(device),
            device_name,
            "failed",
            f"{type(error).__name__}: {error}",
            False,
        )
        raise CudaDeterminismPreflightError(report) from error
    return A1DeterminismPreflight(
        "cuda",
        str(device),
        device_name,
        "passed",
        None,
        False,
    )


def cpu_determinism_preflight() -> A1DeterminismPreflight:
    """Record that the CUDA-specific preflight was intentionally not applicable."""

    return A1DeterminismPreflight(
        "cpu",
        "cpu",
        None,
        "not_applicable_cpu",
        None,
        False,
    )


def cpu_determinism_profile() -> A1DeterminismProfile:
    """Record actual CPU process state without claiming CUDA policy activation."""

    return A1DeterminismProfile(
        device_type="cpu",
        cuda_policy_status="not_applicable",
        torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        torch_deterministic_warn_only=(torch.is_deterministic_algorithms_warn_only_enabled()),
        cudnn_deterministic=None,
        cudnn_benchmark=None,
        cuda_matmul_allow_tf32=None,
        cudnn_allow_tf32=None,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace_config=None,
    )

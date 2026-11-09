from __future__ import annotations

import json
import math
import os
import random
import re
import stat
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cadiff.config import ProjectConfig
from cadiff.diffusion.components import REQUIRED, load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p1_v1.adapter import MultiScaleTypeAdapter
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p1_v1.config import Smoke0Config
from cadiff.p1_v1.dataset import (
    TrainingIndex,
    TrainingItem,
    load_training_image,
    load_training_index,
    select_smoke0_items,
)
from cadiff.p1_v1.model import P1NoisePredictor
from cadiff.p1_v1.training import (
    StepMetrics,
    assert_parameter_versions_unchanged,
    capture_parameter_versions,
    encode_latents,
    run_optimizer_step,
    sample_step_noise,
)
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic

_REPORT_SCHEMA = "p1_v1_smoke0_report_v1"
_PROVENANCE_SCHEMA = "p1_v1_smoke0_provenance_v1"
_CUBLAS_VALUES = frozenset({":16:8", ":4096:8"})
_EXACT_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}
_MIN_3090_MEMORY = 23 * 1024**3
_MAX_3090_MEMORY = 25 * 1024**3
_ARTIFACT_FILENAMES = frozenset({"provenance.json", "report.json"})
_MAX_EXCEPTION_TYPE_LENGTH = 128
_MAX_EXCEPTION_MESSAGE_LENGTH = 256
_QUOTED_OR_TOKEN_VALUE = r'''(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)'''
_CREDENTIAL_KEY = (
    r"(?:[a-z0-9]+(?:[_-][a-z0-9]+)*[_-])?"
    r"(?:token|secret|password|passwd|api[_-]?key|access[_-]?key)"
)
_SECRET_ASSIGNMENT = re.compile(
    rf"(?<![a-z0-9_-]){_CREDENTIAL_KEY}\s*[:=]\s*{_QUOTED_OR_TOKEN_VALUE}",
    flags=re.IGNORECASE,
)
_AUTHORIZATION_VALUE = re.compile(
    rf"\b(bearer|basic)\s+{_QUOTED_OR_TOKEN_VALUE}",
    flags=re.IGNORECASE,
)
_URI = re.compile(
    r"\b[a-z][a-z0-9+.-]*://[^\s,;\"']+",
    flags=re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r'''(?:"(?:[a-z]:[\\/]|\\\\)[^"]*"|'(?:[a-z]:[\\/]|\\\\)[^']*'|'''
    r"(?:[a-z]:[\\/]|\\\\)[^\s,;]+)",
    flags=re.IGNORECASE,
)
_POSIX_ABSOLUTE_PATH = re.compile(
    r'''(?:"/[^"\r\n]*"|'/[^'\r\n]*'|(?<![a-z0-9:])/[^\s,;]+)''',
    flags=re.IGNORECASE,
)
_SENSITIVE_TAIL_PATTERNS = (
    _URI,
    _AUTHORIZATION_VALUE,
    _SECRET_ASSIGNMENT,
    _WINDOWS_ABSOLUTE_PATH,
    _POSIX_ABSOLUTE_PATH,
)


@dataclass(frozen=True)
class Smoke0Context:
    paths: ProjectConfig
    experiment: Smoke0Config
    manifest: Path
    lock: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class _GpuContract:
    name: str
    total_memory_bytes: int


@dataclass(frozen=True)
class _Smoke0Preflight:
    output: Path
    runs_root: Path
    git_head: str
    environment: dict[str, str]
    gpu: _GpuContract
    training_index: TrainingIndex
    normal: TrainingItem
    anomalies: tuple[TrainingItem, ...]
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]
    input_hashes: dict[str, object]


@dataclass(frozen=True)
class _Smoke0Execution:
    normal_bypass_equal: bool
    zero_init_defect_equal: bool
    optimizer_steps: tuple[StepMetrics, ...]
    base_parameter_versions_unchanged: bool
    base_gradients_clear: bool
    peak_memory_bytes: int
    determinism: dict[str, object]


@dataclass(frozen=True)
class _FinalIntegrity:
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]
    git_head: str


@dataclass(frozen=True)
class _PinnedOutput:
    path: Path
    runs_root: Path
    fd: int
    device: int
    inode: int
    proc_path: Path
    owned_artifacts: set[str] = field(default_factory=set, compare=False, repr=False)


class _Smoke0Failure(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        exception_type: str | None = None,
        exception_message: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.exception_type = exception_type
        self.exception_message = exception_message


def _safe_exception_type(error: Exception) -> str:
    error_type = type(error)
    rendered = f"{error_type.__module__}.{error_type.__qualname__}"
    safe = re.sub(r"[^A-Za-z0-9_.]", "_", rendered)
    return (safe or "Exception")[:_MAX_EXCEPTION_TYPE_LENGTH]


def _redact_sensitive_tail(message: str) -> str:
    starts = tuple(
        match.start()
        for pattern in _SENSITIVE_TAIL_PATTERNS
        if (match := pattern.search(message)) is not None
    )
    if not starts:
        return message
    safe_prefix = message[: min(starts)].rstrip(" :=,;")
    return f"{safe_prefix} <redacted>".strip()


def _safe_exception_message(error: Exception) -> str:
    try:
        raw_message = str(error)
    except BaseException:
        raw_message = "<message unavailable>"
    ascii_message = "".join(
        character if 0x20 <= ord(character) <= 0x7E else " "
        if character.isspace()
        else "?"
        for character in raw_message
    )
    rendered = " ".join(ascii_message.split())
    rendered = _redact_sensitive_tail(rendered)
    return (rendered or "<empty>")[:_MAX_EXCEPTION_MESSAGE_LENGTH]


def _absolute(path: Path, cwd: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else cwd / candidate


def _strictly_below(
    candidate: Path,
    root: Path,
    *,
    label: str,
    root_label: str | None = None,
) -> None:
    rendered_root = root_label or root.name
    if candidate == root:
        raise ValueError(f"{label} must be strictly inside {rendered_root}")
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be strictly inside {rendered_root}") from error


def _validate_environment(cwd: Path) -> dict[str, str]:
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas not in _CUBLAS_VALUES:
        allowed = ", ".join(sorted(_CUBLAS_VALUES))
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set before Python starts to one of: "
            f"{allowed}"
        )
    values = {"CUBLAS_WORKSPACE_CONFIG": cublas}
    for name, expected in _EXACT_ENVIRONMENT.items():
        actual = os.environ.get(name)
        if actual != expected:
            raise RuntimeError(f"{name} must be exactly {expected!r} before Python starts")
        values[name] = actual

    raw_hf_home = os.environ.get("HF_HOME")
    if not raw_hf_home:
        raise RuntimeError("HF_HOME must be set before Python starts")
    hf_home_path = _absolute(Path(raw_hf_home).expanduser(), cwd).resolve(strict=False)
    _strictly_below(hf_home_path, cwd, label="HF_HOME")
    values["HF_HOME"] = str(hf_home_path)
    return values


def _validate_gpu_contract() -> _GpuContract:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for Smoke 0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke 0 requires exactly one visible CUDA device")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 3090":
        raise RuntimeError(
            "Smoke 0 requires exactly 'NVIDIA GeForce RTX 3090'; "
            f"got {name!r}"
        )
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if type(total_memory) is not int or not (
        _MIN_3090_MEMORY <= total_memory <= _MAX_3090_MEMORY
    ):
        raise RuntimeError("RTX 3090 memory must be in the 24 GiB device class")
    return _GpuContract(name=name, total_memory_bytes=total_memory)


def _read_git_state(cwd: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return head, bool(status)


def _validated_output(context: Smoke0Context, cwd: Path) -> tuple[Path, Path]:
    runs_root = _absolute(context.paths.runs_root, cwd).resolve(strict=True)
    if not runs_root.is_dir():
        raise NotADirectoryError(f"runs_root must be a directory: {runs_root}")
    requested = _absolute(context.output, cwd)
    resolved = requested.resolve(strict=False)
    _strictly_below(resolved, runs_root, label="output", root_label="runs_root")
    if os.path.lexists(requested):
        raise FileExistsError(f"output already exists: {requested}")
    return resolved, runs_root


def _validated_input_file(path: Path, root: Path, cwd: Path, *, label: str) -> Path:
    requested = _absolute(path, cwd)
    if requested.is_symlink():
        raise ValueError(f"{label} symlink is not allowed: {requested}")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"{label} root must be a directory: {resolved_root}")
    resolved = requested.resolve(strict=True)
    _strictly_below(resolved, resolved_root, label=label)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def _regular_file(path: Path, cwd: Path, *, label: str) -> Path:
    requested = _absolute(path, cwd)
    if requested.is_symlink():
        raise ValueError(f"{label} symlink is not allowed: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def _hash_declared_inputs(
    context: Smoke0Context,
    *,
    cwd: Path,
    manifest: Path,
    lock: Path,
) -> dict[str, object]:
    if not isinstance(context.config_paths, dict) or not context.config_paths:
        raise ValueError("config_paths must be a non-empty dictionary")
    config_hashes: dict[str, str] = {}
    for name, path in sorted(context.config_paths.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("config_paths keys must be non-empty strings")
        resolved = _regular_file(path, cwd, label=f"config {name}")
        config_hashes[name] = sha256_file(resolved)
    return {
        "config_sha256": config_hashes,
        "manifest_sha256": sha256_file(manifest),
        "lock_sha256": sha256_file(lock),
    }


def _file_snapshot(path: Path) -> dict[str, int | str]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    before_identity = (before.st_size, before.st_mtime_ns)
    after_identity = (after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while hashing: {path}")
    return {
        "sha256": digest,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def _snapshot_selected_sources(
    normal: TrainingItem,
    anomalies: tuple[TrainingItem, ...],
) -> dict[str, dict[str, int | str]]:
    selected = (normal, *anomalies)
    snapshot: dict[str, dict[str, int | str]] = {}
    for item in selected:
        if item.image_relpath in snapshot:
            raise ValueError(f"duplicate selected source relpath: {item.image_relpath}")
        snapshot[item.image_relpath] = _file_snapshot(item.image_path)
    return snapshot


def _snapshot_model_files(model_root: Path) -> dict[str, dict[str, int | str]]:
    root = Path(model_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"model_root must be a directory: {root}")
    snapshot: dict[str, dict[str, int | str]] = {}
    for relative in REQUIRED:
        resolved = (root / relative).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"model file escapes model_root: {relative}") from error
        if not resolved.is_file():
            raise ValueError(f"model file must be regular: {relative}")
        snapshot[relative] = _file_snapshot(resolved)
    return snapshot


def validate_smoke0_preflight(context: Smoke0Context) -> _Smoke0Preflight:
    if not isinstance(context, Smoke0Context):
        raise TypeError("context must be a Smoke0Context")
    cwd = Path(context.cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise NotADirectoryError(f"cwd must be a directory: {cwd}")

    environment = _validate_environment(cwd)
    gpu = _validate_gpu_contract()

    git_head, git_dirty = _read_git_state(cwd)
    if git_dirty:
        raise RuntimeError("Smoke 0 requires a clean Git worktree")

    output, runs_root = _validated_output(context, cwd)
    manifest = _validated_input_file(
        context.manifest,
        cwd / "manifests",
        cwd,
        label="manifest",
    )
    lock = _validated_input_file(
        context.lock,
        cwd / "requirements",
        cwd,
        label="lock",
    )

    input_hashes = _hash_declared_inputs(
        context,
        cwd=cwd,
        manifest=manifest,
        lock=lock,
    )
    training_index = load_training_index(
        manifest,
        context.paths.dataset_root,
        context.experiment,
    )
    normal, anomalies = select_smoke0_items(training_index)
    actual_types = tuple(item.defect_type for item in anomalies)
    if actual_types != context.experiment.defect_types:
        raise ValueError("Smoke 0 selection must contain every configured defect type")

    source_snapshot = _snapshot_selected_sources(normal, anomalies)
    model_snapshot = _snapshot_model_files(context.paths.model_root)
    input_hashes["vocabulary_sha256"] = training_index.vocabulary_sha256
    return _Smoke0Preflight(
        output=output,
        runs_root=runs_root,
        git_head=git_head,
        environment=environment,
        gpu=gpu,
        training_index=training_index,
        normal=normal,
        anomalies=anomalies,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
        input_hashes=input_hashes,
    )


def _set_determinism(seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _batch_pixels(item: TrainingItem, *, image_size: int, device: torch.device) -> torch.Tensor:
    loaded = load_training_image(item, image_size)
    return loaded.pixels.unsqueeze(0).to(device=device, dtype=torch.float16)


def _capture_base_versions(components: object) -> dict[str, tuple[int, ...]]:
    return {
        "vae": capture_parameter_versions(components.vae),
        "text_encoder": capture_parameter_versions(components.text_encoder),
        "unet": capture_parameter_versions(components.unet),
    }


def _assert_base_unchanged(
    components: object,
    expected: dict[str, tuple[int, ...]],
) -> None:
    modules = {
        "vae": components.vae,
        "text_encoder": components.text_encoder,
        "unet": components.unet,
    }
    for name, module in modules.items():
        assert_parameter_versions_unchanged(module, expected[name])
        if any(parameter.grad is not None for parameter in module.parameters()):
            raise AssertionError(f"frozen {name} parameters must not have gradients")


def _direct_prediction(
    unet: object,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    output = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds)
    try:
        prediction = output.sample
    except AttributeError as error:
        raise ValueError("direct U-Net output must expose sample") from error
    if not isinstance(prediction, torch.Tensor) or not torch.isfinite(prediction).all().item():
        raise ValueError("direct U-Net prediction must be a finite tensor")
    return prediction


def _encode_training_condition(
    components: object,
    condition: str,
    device: torch.device,
) -> torch.Tensor:
    encoded = encode_prompt(components, condition, device)
    if not isinstance(encoded, torch.Tensor):
        raise ValueError("encoded condition must be a tensor")
    prepared = encoded.detach().clone()
    if (
        prepared.is_inference()
        or prepared.requires_grad
        or not torch.isfinite(prepared).all().item()
    ):
        raise ValueError("encoded condition must be finite, frozen, and non-inference")
    return prepared


def _execute_smoke0(
    context: Smoke0Context,
    preflight: _Smoke0Preflight,
) -> _Smoke0Execution:
    determinism = _set_determinism(context.experiment.seed)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    try:
        components = load_frozen_components(
            context.paths.model_root,
            device,
            torch.float16,
        )
    except Exception as error:
        raise _Smoke0Failure(
            "component_load_failure",
            exception_type=_safe_exception_type(error),
            exception_message=_safe_exception_message(error),
        ) from error
    prompt_embeds = _encode_training_condition(
        components,
        context.experiment.normal_prompt,
        device,
    )

    vocabulary = DefectVocabulary.from_config(
        context.experiment.category,
        context.experiment.defect_types,
    )
    if vocabulary.sha256 != preflight.training_index.vocabulary_sha256:
        raise _Smoke0Failure("vocabulary_changed_during_run")
    conditioner = DefectConditioner(
        vocabulary,
        context.experiment.adapter.type_embedding_dim,
    ).to(device=device, dtype=torch.float32)
    adapter = MultiScaleTypeAdapter(context.experiment.adapter).to(
        device=device,
        dtype=torch.float32,
    )
    model = P1NoisePredictor(components.unet, conditioner, adapter)
    base_versions = _capture_base_versions(components)

    generator = torch.Generator(device=device).manual_seed(context.experiment.seed)
    normal_images = _batch_pixels(
        preflight.normal,
        image_size=context.experiment.image_size,
        device=device,
    )
    with torch.no_grad():
        normal_latents = encode_latents(
            vae=components.vae,
            images=normal_images,
            generator=generator,
        )
        equality_noise = sample_step_noise(
            latents=normal_latents,
            scheduler=components.scheduler,
            generator=generator,
        )
        noisy = components.scheduler.add_noise(
            normal_latents,
            equality_noise.noise,
            equality_noise.timesteps,
        )
        direct = _direct_prediction(
            components.unet,
            noisy,
            equality_noise.timesteps,
            prompt_embeds,
        )
        normal_prediction = model(
            noisy,
            equality_noise.timesteps,
            prompt_embeds,
            "normal",
        )
        normal_equal = torch.equal(direct, normal_prediction)
        defect_equalities = tuple(
            torch.equal(
                direct,
                model(noisy, equality_noise.timesteps, prompt_embeds, defect_type),
            )
            for defect_type in context.experiment.defect_types
        )
    if not normal_equal:
        raise _Smoke0Failure("normal_bypass_mismatch")
    if not all(defect_equalities):
        raise _Smoke0Failure("zero_init_defect_mismatch")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=context.experiment.learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda")
    step_metrics: list[StepMetrics] = []
    for anomaly in preflight.anomalies:
        anomaly_images = _batch_pixels(
            anomaly,
            image_size=context.experiment.image_size,
            device=device,
        )
        step_metrics.append(
            run_optimizer_step(
                vae=components.vae,
                scheduler=components.scheduler,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                anomaly_images=anomaly_images,
                normal_images=normal_images,
                prompt_embeds=prompt_embeds,
                defect_type=anomaly.defect_type,
                generator=generator,
            )
        )

    _assert_base_unchanged(components, base_versions)
    torch.cuda.synchronize(device)
    peak_memory = torch.cuda.max_memory_allocated(device)
    return _Smoke0Execution(
        normal_bypass_equal=normal_equal,
        zero_init_defect_equal=all(defect_equalities),
        optimizer_steps=tuple(step_metrics),
        base_parameter_versions_unchanged=True,
        base_gradients_clear=True,
        peak_memory_bytes=peak_memory,
        determinism=determinism,
    )


def _validate_execution(
    execution: _Smoke0Execution,
    context: Smoke0Context,
    preflight: _Smoke0Preflight,
) -> None:
    if not isinstance(execution, _Smoke0Execution):
        raise _Smoke0Failure("invalid_execution_result")
    if execution.normal_bypass_equal is not True:
        raise _Smoke0Failure("normal_bypass_mismatch")
    if execution.zero_init_defect_equal is not True:
        raise _Smoke0Failure("zero_init_defect_mismatch")
    if execution.base_parameter_versions_unchanged is not True:
        raise _Smoke0Failure("base_parameter_changed")
    if execution.base_gradients_clear is not True:
        raise _Smoke0Failure("base_gradient_present")
    labels = tuple(metric.defect_type for metric in execution.optimizer_steps)
    if labels != context.experiment.defect_types:
        raise _Smoke0Failure("optimizer_step_labels_invalid")
    for metric in execution.optimizer_steps:
        if not isinstance(metric, StepMetrics):
            raise _Smoke0Failure("optimizer_step_invalid")
        scalars = (
            metric.anomaly_mse,
            metric.normal_mse,
            metric.trainable_grad_l2,
        )
        if any(
            value is None
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in scalars
        ):
            raise _Smoke0Failure("non_finite_metric")
        if metric.zero_head_grad_nonzero is not True:
            raise _Smoke0Failure("zero_head_gradient_missing")
        if (
            not isinstance(metric.anomaly_gradient_hash, str)
            or len(metric.anomaly_gradient_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in metric.anomaly_gradient_hash
            )
        ):
            raise _Smoke0Failure("gradient_hash_invalid")
    if (
        type(execution.peak_memory_bytes) is not int
        or execution.peak_memory_bytes <= 0
        or execution.peak_memory_bytes >= preflight.gpu.total_memory_bytes
    ):
        raise _Smoke0Failure("peak_memory_invalid")


def _verify_final_integrity(
    context: Smoke0Context,
    preflight: _Smoke0Preflight,
    output: _PinnedOutput,
) -> _FinalIntegrity:
    _validate_pinned_output(output)
    try:
        sources = _snapshot_selected_sources(preflight.normal, preflight.anomalies)
    except Exception as error:
        raise _Smoke0Failure("source_snapshot_changed") from error
    if sources != preflight.source_snapshot:
        raise _Smoke0Failure("source_snapshot_changed")

    try:
        models = _snapshot_model_files(context.paths.model_root)
    except Exception as error:
        raise _Smoke0Failure("model_snapshot_changed") from error
    if models != preflight.model_snapshot:
        raise _Smoke0Failure("model_snapshot_changed")

    try:
        git_head, git_dirty = _read_git_state(context.cwd)
    except (OSError, subprocess.SubprocessError) as error:
        raise _Smoke0Failure("git_changed_during_run") from error
    if git_dirty or git_head != preflight.git_head:
        raise _Smoke0Failure("git_changed_during_run")
    _validate_pinned_output(output)
    return _FinalIntegrity(
        source_snapshot=sources,
        model_snapshot=models,
        git_head=git_head,
    )


def _report_input_hashes(preflight: _Smoke0Preflight) -> dict[str, object]:
    return {
        **preflight.input_hashes,
        "model_files": preflight.model_snapshot,
        "selected_sources": preflight.source_snapshot,
    }


def _success_report(
    context: Smoke0Context,
    preflight: _Smoke0Preflight,
    execution: _Smoke0Execution,
) -> dict[str, object]:
    return {
        "schema_version": _REPORT_SCHEMA,
        "status": "passed",
        "category": context.experiment.category,
        "defect_types": list(context.experiment.defect_types),
        "normal_bypass_equal": execution.normal_bypass_equal,
        "zero_init_defect_equal": execution.zero_init_defect_equal,
        "optimizer_steps": [asdict(metric) for metric in execution.optimizer_steps],
        "base_parameter_versions_unchanged": (
            execution.base_parameter_versions_unchanged
        ),
        "base_gradients_clear": execution.base_gradients_clear,
        "peak_memory_bytes": execution.peak_memory_bytes,
        "gpu_total_memory_bytes": preflight.gpu.total_memory_bytes,
        "git_commit": preflight.git_head,
        "input_hashes": _report_input_hashes(preflight),
    }


def _provenance_payload(
    base: dict[str, Any],
    context: Smoke0Context,
    preflight: _Smoke0Preflight,
    execution: _Smoke0Execution,
    final: _FinalIntegrity,
) -> dict[str, Any]:
    return {
        **base,
        "schema_version": _PROVENANCE_SCHEMA,
        "category": context.experiment.category,
        "defect_types": list(context.experiment.defect_types),
        "environment": preflight.environment,
        "determinism": execution.determinism,
        "gpu_contract": {
            "name": preflight.gpu.name,
            "total_memory_bytes": preflight.gpu.total_memory_bytes,
        },
        "input_hashes": _report_input_hashes(preflight),
        "selected_sample_ids": [
            preflight.normal.sample_id,
            *(item.sample_id for item in preflight.anomalies),
        ],
        "integrity": {
            "git_start": preflight.git_head,
            "git_end": final.git_head,
            "git_dirty_end": False,
            "source_before": preflight.source_snapshot,
            "source_after": final.source_snapshot,
            "model_before": preflight.model_snapshot,
            "model_after": final.model_snapshot,
        },
    }


def _require_strict_json(payload: dict[str, object]) -> None:
    json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _failure_reason(error: Exception) -> str:
    if isinstance(error, _Smoke0Failure):
        return error.reason
    if isinstance(error, OSError | ValueError | RuntimeError | AssertionError | TypeError):
        return "runtime_failure"
    return "unexpected_failure"


def _pin_output(preflight: _Smoke0Preflight) -> _PinnedOutput:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise _Smoke0Failure("stable_output_handle_unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(preflight.output, flags)
    try:
        identity = os.fstat(fd)
        pinned = _PinnedOutput(
            path=preflight.output,
            runs_root=preflight.runs_root,
            fd=fd,
            device=identity.st_dev,
            inode=identity.st_ino,
            proc_path=Path("/proc/self/fd") / str(fd),
        )
        _validate_pinned_output(pinned)
        return pinned
    except Exception:
        os.close(fd)
        raise


def _validate_pinned_descriptor(output: _PinnedOutput) -> None:
    identity = os.fstat(output.fd)
    if (identity.st_dev, identity.st_ino) != (output.device, output.inode):
        raise _Smoke0Failure("output_changed_during_run")


def _validate_pinned_output(output: _PinnedOutput) -> None:
    _validate_pinned_descriptor(output)
    try:
        public_identity = os.stat(output.path, follow_symlinks=False)
        handle_path = output.proc_path.resolve(strict=True)
    except OSError as error:
        raise _Smoke0Failure("output_changed_during_run") from error
    if not stat.S_ISDIR(public_identity.st_mode):
        raise _Smoke0Failure("output_changed_during_run")
    if (public_identity.st_dev, public_identity.st_ino) != (
        output.device,
        output.inode,
    ):
        raise _Smoke0Failure("output_changed_during_run")
    try:
        _strictly_below(
            handle_path,
            output.runs_root,
            label="output handle",
            root_label="runs_root",
        )
        public_path = output.path.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise _Smoke0Failure("output_changed_during_run") from error
    if handle_path != output.path or public_path != output.path:
        raise _Smoke0Failure("output_changed_during_run")


def _is_owned_temp(name: str, artifact: str) -> bool:
    return name.startswith(f".{artifact}.") and name.endswith(".tmp")


def _is_owned_artifact_name(output: _PinnedOutput, name: str) -> bool:
    return name in output.owned_artifacts or any(
        _is_owned_temp(name, artifact) for artifact in output.owned_artifacts
    )


def _owned_file_entries(output: _PinnedOutput) -> list[str]:
    entries: list[str] = []
    for name in os.listdir(output.fd):
        if not _is_owned_artifact_name(output, name):
            continue
        try:
            mode = os.stat(name, dir_fd=output.fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            entries.append(name)
    return entries


def _cleanup_pinned_artifacts(output: _PinnedOutput) -> None:
    _validate_pinned_descriptor(output)
    for name in _owned_file_entries(output):
        try:
            os.unlink(name, dir_fd=output.fd)
        except (FileNotFoundError, IsADirectoryError):
            pass
    if _owned_file_entries(output):
        raise _Smoke0Failure("output_artifact_cleanup_failed")


def _write_pinned_json(
    output: _PinnedOutput,
    filename: str,
    payload: dict[str, Any],
) -> None:
    if filename not in _ARTIFACT_FILENAMES:
        raise ValueError("unsupported Smoke 0 artifact filename")
    _validate_pinned_output(output)
    output.owned_artifacts.add(filename)
    try:
        write_json_atomic(output.proc_path / filename, payload)
        _validate_pinned_output(output)
    except _Smoke0Failure as error:
        if error.reason == "output_changed_during_run":
            try:
                _cleanup_pinned_artifacts(output)
            except Exception:
                pass
        raise


def _write_pinned_failure(output: _PinnedOutput, error: Exception) -> None:
    payload: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "status": "failed",
        "reason": _failure_reason(error),
    }
    if isinstance(error, _Smoke0Failure):
        if error.exception_type is not None:
            payload["exception_type"] = error.exception_type
        if error.exception_message is not None:
            payload["exception_message"] = error.exception_message
    _require_strict_json(payload)
    _write_pinned_json(output, "report.json", payload)


def run_smoke0(context: Smoke0Context) -> int:
    preflight: _Smoke0Preflight | None = None
    pinned_output: _PinnedOutput | None = None
    exit_code = 1
    try:
        preflight = validate_smoke0_preflight(context)
        preflight.output.mkdir(parents=True, exist_ok=False)
        pinned_output = _pin_output(preflight)

        execution = _execute_smoke0(context, preflight)
        _validate_pinned_output(pinned_output)
        _validate_execution(execution, context, preflight)
        base_provenance = collect_provenance(context.paths, cwd=context.cwd)
        final = _verify_final_integrity(context, preflight, pinned_output)
        provenance = _provenance_payload(
            base_provenance,
            context,
            preflight,
            execution,
            final,
        )
        report = _success_report(context, preflight, execution)
        _require_strict_json(provenance)
        _require_strict_json(report)
        _write_pinned_json(pinned_output, "provenance.json", provenance)
        _verify_final_integrity(context, preflight, pinned_output)
        _write_pinned_json(pinned_output, "report.json", report)
        _verify_final_integrity(context, preflight, pinned_output)
        exit_code = 0
    except Exception as error:
        if pinned_output is not None:
            try:
                _validate_pinned_output(pinned_output)
                _write_pinned_failure(pinned_output, error)
            except Exception:
                pass
    finally:
        if pinned_output is not None:
            try:
                _validate_pinned_output(pinned_output)
            except Exception:
                exit_code = 1
                try:
                    _cleanup_pinned_artifacts(pinned_output)
                except Exception:
                    pass
            try:
                os.close(pinned_output.fd)
            except OSError:
                # Artifact durability/integrity checks have already certified the
                # result; this directory descriptor owns no buffered artifact data.
                pass
    return exit_code


__all__ = ["Smoke0Context", "run_smoke0", "validate_smoke0_preflight"]

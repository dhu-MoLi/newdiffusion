from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from cadiff.config import ProjectConfig
from cadiff.data.geometry import resize_pad_image
from cadiff.data.mvtec_manifest import (
    ManifestRecord,
    build_mvtec_manifest,
    exact_evenly_spaced,
    write_manifest,
)
from cadiff.diffusion.components import REQUIRED, FrozenComponents, load_frozen_components
from cadiff.diffusion.ddim_inversion import invert, reconstruct
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode, image_to_tensor
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic

P0_CATEGORIES = ("bottle", "grid", "transistor")
P0_GATE_REVISION = "gate_a_v1"
P0_SAMPLE_COUNT = 12
CUBLAS_WORKSPACE_CONFIG_VALUES = frozenset((":4096:8", ":16:8"))


@dataclass(frozen=True)
class ExperimentConfig:
    split_id: str
    categories: tuple[str, ...]
    normal_prompt_template: str
    dtype: str
    scheduler: str
    guidance_scale: float
    seed: int


@dataclass(frozen=True)
class GateConfig:
    gate_revision: str
    sample_count_per_category: int
    selection: str
    median_extra_l1_max: float
    p95_extra_l1_max: float
    allow_non_finite: bool


@dataclass(frozen=True)
class ImageMetrics:
    category: str
    image_relpath: str
    vae_l1: float
    ddim_l1: float
    extra_l1: float
    vae_psnr: float
    ddim_psnr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "vae_l1": self.vae_l1,
            "ddim_l1": self.ddim_l1,
            "extra_l1": self.extra_l1,
            "vae_psnr": self.vae_psnr,
            "ddim_psnr": self.ddim_psnr,
        }


class ReconstructionRunner(Protocol):
    def reconstruct(
        self, record: ManifestRecord, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


def _strict_mapping(path: Path, allowed: set[str]) -> Mapping[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a YAML mapping")
    unknown = set(raw) - allowed
    missing = allowed - set(raw)
    if unknown or missing:
        raise ValueError(f"{path} has unknown={sorted(unknown)} missing={sorted(missing)}")
    return raw


def _exact_int(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{field} must be an exact {qualifier}integer")
    return value


def _finite_float(value: object, field: str, *, minimum: float | None = None) -> float:
    if type(value) not in {float, int} or isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} is outside its permitted range")
    return result


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = _strict_mapping(
        path,
        {
            "split_id",
            "eval_mode",
            "categories",
            "normal_prompt_template",
            "dtype",
            "scheduler",
            "guidance_scale",
            "seed",
        },
    )
    categories = raw["categories"]
    if (
        not isinstance(categories, list)
        or not categories
        or any(type(item) is not str or not item for item in categories)
    ):
        raise ValueError("categories must be a non-empty list of strings")
    category_tuple = tuple(categories)
    if len(set(category_tuple)) != len(category_tuple):
        raise ValueError("categories must be unique")
    if category_tuple != P0_CATEGORIES:
        raise ValueError("P0 categories must be [bottle, grid, transistor] in that order")
    prompt = raw["normal_prompt_template"]
    if not isinstance(prompt, str) or prompt.count("{category}") != 1:
        raise ValueError("normal_prompt_template must contain exactly one {category}")
    if raw["dtype"] != "float16" or raw["scheduler"] != "ddim":
        raise ValueError("P0 requires dtype=float16 and scheduler=ddim")
    if raw["eval_mode"] not in {"dev_aware", "frozen_transfer"}:
        raise ValueError("eval_mode is invalid")
    split_id = raw["split_id"]
    if not isinstance(split_id, str) or not split_id:
        raise ValueError("split_id must be a non-empty string")
    guidance_scale = _finite_float(raw["guidance_scale"], "guidance_scale", minimum=0.0)
    if guidance_scale != 1.0:
        raise ValueError("P0 requires guidance_scale=1.0")
    return ExperimentConfig(
        split_id,
        category_tuple,
        prompt,
        "float16",
        "ddim",
        guidance_scale,
        _exact_int(raw["seed"], "seed"),
    )


def load_gate_config(path: Path) -> GateConfig:
    raw = _strict_mapping(
        path,
        {
            "gate_revision",
            "sample_count_per_category",
            "selection",
            "median_extra_l1_max",
            "p95_extra_l1_max",
            "allow_non_finite",
        },
    )
    if (
        raw["selection"] != "lexicographic_even_spacing"
        or raw["allow_non_finite"] is not False
        or raw["gate_revision"] != P0_GATE_REVISION
        or raw["sample_count_per_category"] != P0_SAMPLE_COUNT
    ):
        raise ValueError("P0 gate requires deterministic finite normal selection")
    revision = raw["gate_revision"]
    if not isinstance(revision, str) or not revision:
        raise ValueError("gate_revision must be a non-empty string")
    median = _finite_float(raw["median_extra_l1_max"], "median_extra_l1_max", minimum=0.0)
    p95 = _finite_float(raw["p95_extra_l1_max"], "p95_extra_l1_max", minimum=0.0)
    return GateConfig(
        revision,
        _exact_int(raw["sample_count_per_category"], "sample_count_per_category", positive=True),
        "lexicographic_even_spacing",
        median,
        p95,
        False,
    )


def assert_clean_git(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cwd, check=True, capture_output=True, text=True
    )
    if result.stdout.strip():
        raise RuntimeError("P0 requires a clean git worktree")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def ensure_output_directory(output: Path, runs_root: Path, *, overwrite: bool) -> Path:
    root = runs_root.resolve(strict=True)
    requested = output.expanduser().absolute()
    resolved = requested.resolve()
    if resolved == root:
        raise ValueError("output must not equal runs_root itself")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("output must be strictly inside runs_root") from error
    if requested != resolved:
        raise ValueError("output canonical path must equal the requested output path")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("output must be a directory")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError("refusing to overwrite non-empty output directory")
        # The root containment and canonical equality checks above are deliberately before cleanup.
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_manifest_output(
    manifest_output: Path,
    paths: ProjectConfig,
    protected_paths: Iterable[Path],
) -> Path:
    manifests_root = (paths.runs_root.resolve(strict=True).parent / "manifests").resolve(
        strict=True
    )
    requested = manifest_output.expanduser().absolute()
    resolved = requested.resolve()
    if requested != resolved or resolved.suffix != ".jsonl":
        raise ValueError("manifest must be a canonical non-symlink .jsonl path")
    try:
        relative = resolved.relative_to(manifests_root)
    except ValueError as error:
        raise ValueError("manifest must be strictly inside project manifests directory") from error
    if not relative.parts:
        raise ValueError("manifest must not equal manifests directory")
    protected = [paths.dataset_root, paths.model_root, paths.runs_root, *protected_paths]
    if any(resolved == candidate.resolve() for candidate in protected):
        raise ValueError("manifest must not overlap a protected input or output")
    return resolved


def _set_determinism(seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
        "cublas_workspace_config": validate_cublas_workspace_config(),
    }


def validate_cublas_workspace_config() -> str:
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value not in CUBLAS_WORKSPACE_CONFIG_VALUES:
        allowed = ", ".join(sorted(CUBLAS_WORKSPACE_CONFIG_VALUES))
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG must be set before Python starts to one of: {allowed}"
        )
    return value


def _validate_p0_contract(experiment: ExperimentConfig, gate: GateConfig) -> None:
    if experiment.categories != P0_CATEGORIES:
        raise ValueError("P0 requires ordered bottle, grid, transistor categories")
    if gate.gate_revision != P0_GATE_REVISION:
        raise ValueError("P0 requires the fixed Gate A revision")
    if gate.selection != "lexicographic_even_spacing" or gate.sample_count_per_category != 12:
        raise ValueError("P0 requires 12 lexicographically spaced samples per category")
    if gate.allow_non_finite:
        raise ValueError("P0 never allows non-finite metrics")


def select_normal_train(
    records: Iterable[ManifestRecord], experiment: ExperimentConfig, gate: GateConfig
) -> list[ManifestRecord]:
    selected: list[ManifestRecord] = []
    for category in experiment.categories:
        candidates = [
            record
            for record in records
            if (
                record.category == category
                and record.defect_type == "good"
                and record.roles == ("normal_train",)
            )
        ]
        if len(candidates) < gate.sample_count_per_category:
            raise ValueError(
                f"{category} has fewer than {gate.sample_count_per_category} "
                "train/good normal images"
            )
        chosen = exact_evenly_spaced(candidates, gate.sample_count_per_category)
        selected.extend(chosen)  # type: ignore[arg-type]
    if len(selected) != len(experiment.categories) * gate.sample_count_per_category:
        raise ValueError("selected normal record count is invalid")
    if len({record.image_relpath for record in selected}) != len(selected):
        raise ValueError("selected normal records must be unique")
    return selected


def _validate_metric_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating NCHW tensor")
    if value.shape[1] != 3 or torch.any(torch.isfinite(value) & ((value < 0) | (value > 1))):
        raise ValueError(f"{name} must be RGB values in [0, 1] when finite")
    return value.to(dtype=torch.float32)


def masked_reconstruction_metrics(
    reference: torch.Tensor,
    vae: torch.Tensor,
    ddim: torch.Tensor,
    valid_region: torch.Tensor,
) -> ImageMetrics:
    reference = _validate_metric_tensor(reference, "reference")
    vae = _validate_metric_tensor(vae, "vae")
    ddim = _validate_metric_tensor(ddim, "ddim")
    if reference.shape != vae.shape or reference.shape != ddim.shape:
        raise ValueError("metric tensors must share a shape")
    if (
        not isinstance(valid_region, torch.Tensor)
        or valid_region.ndim != 4
        or valid_region.shape[1] != 1
        or valid_region.shape[0] != reference.shape[0]
        or valid_region.shape[-2:] != reference.shape[-2:]
    ):
        raise ValueError("valid_region must be N1HW and match the image batch and geometry")
    if not bool(torch.all((valid_region == 0) | (valid_region == 255))):
        raise ValueError("valid_region must be binary 0/255")
    valid = valid_region == 255
    if not bool(valid.any()):
        raise ValueError("valid_region has no valid pixels")
    expanded = valid.expand(-1, reference.shape[1], -1, -1)

    if not all(torch.isfinite(value).all() for value in (reference, vae, ddim)):
        return ImageMetrics("", "", math.nan, math.nan, math.nan, math.nan, math.nan)

    def l1(value: torch.Tensor) -> float:
        return float(torch.abs(value - reference)[expanded].mean().item())

    def psnr(value: torch.Tensor) -> float:
        mse = float(((value - reference)[expanded] ** 2).mean().item())
        return float(10.0 * math.log10(1.0 / max(mse, 1e-12)))

    vae_l1 = l1(vae)
    ddim_l1 = l1(ddim)
    return ImageMetrics("", "", vae_l1, ddim_l1, ddim_l1 - vae_l1, psnr(vae), psnr(ddim))


def evaluate_gate(metrics: list[ImageMetrics], gate: GateConfig) -> dict[str, Any]:
    if not metrics:
        raise ValueError("cannot evaluate an empty gate")
    values = np.asarray([item.extra_l1 for item in metrics], dtype=np.float64)
    all_finite = bool(np.isfinite(values).all()) and all(
        math.isfinite(value) for item in metrics for value in item.as_dict().values()
    )
    if all_finite:
        overall: dict[str, float | None] = {
            "median_extra_l1": float(np.median(values)),
            "p95_extra_l1": float(np.quantile(values, 0.95, method="linear")),
        }
    else:
        overall = {"median_extra_l1": None, "p95_extra_l1": None}
    categories: dict[str, list[ImageMetrics]] = defaultdict(list)
    for item in metrics:
        categories[item.category].append(item)
    by_category = {}
    for category, items in sorted(categories.items()):
        category_values = np.asarray([item.extra_l1 for item in items], dtype=np.float64)
        category_finite = bool(np.isfinite(category_values).all())
        by_category[category] = {
            "count": len(items),
            "finite": category_finite,
            "median_extra_l1": float(np.median(category_values)) if category_finite else None,
            "p95_extra_l1": (
                float(np.quantile(category_values, 0.95, method="linear"))
                if category_finite
                else None
            ),
        }
    passed = (
        all_finite
        and overall["median_extra_l1"] is not None
        and overall["p95_extra_l1"] is not None
        and overall["median_extra_l1"] <= gate.median_extra_l1_max
        and overall["p95_extra_l1"] <= gate.p95_extra_l1_max
    )
    return {
        "schema_version": "p0_gate_a_metrics_v1",
        "gate_revision": gate.gate_revision,
        "thresholds": {
            "median_extra_l1_max": gate.median_extra_l1_max,
            "p95_extra_l1_max": gate.p95_extra_l1_max,
        },
        "quantile_method": "linear",
        "images": [
            {
                "category": item.category,
                "image_relpath": item.image_relpath,
                **{
                    key: value if math.isfinite(value) else None
                    for key, value in item.as_dict().items()
                },
                "finite": all(math.isfinite(value) for value in item.as_dict().values()),
            }
            for item in metrics
        ],
        "categories": by_category,
        "overall": overall,
        "all_finite": all_finite,
        "failure_reason": "non_finite_metrics" if not all_finite else None,
        "passed": passed,
    }


def tree_snapshot(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    entries = []
    paths = sorted(
        resolved.rglob("*"), key=lambda candidate: candidate.relative_to(resolved).as_posix()
    )
    for path in paths:
        stat = path.lstat()
        entries.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "type": "dir" if path.is_dir() else "file",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "root": str(resolved),
        "count": len(entries),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _model_weight_snapshot(root: Path) -> dict[str, dict[str, int]]:
    weights = (
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    )
    return {
        relative: {
            "size": (root / relative).stat().st_size,
            "mtime_ns": (root / relative).stat().st_mtime_ns,
        }
        for relative in weights
    }


class RealRunner:
    def __init__(self, config: ProjectConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.device = torch.device("cuda:0")
        self.components: FrozenComponents = load_frozen_components(
            config.model_root, self.device, torch.float16
        )
        self.config = config

    def reconstruct(
        self, record: ManifestRecord, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.config.dataset_root / record.image_relpath
        with Image.open(source) as image:
            prepared, valid, _ = resize_pad_image(image, self.config.image_size)
        image_tensor = image_to_tensor(prepared, self.device, torch.float16)
        reference = (image_tensor / 2 + 0.5).to(torch.float32)
        valid_tensor = pil_to_tensor(valid).unsqueeze(0).to(self.device)
        latent = encode_mode(self.components.vae, image_tensor)
        vae = decode_latent(self.components.vae, latent)
        trajectory = invert(self.components, latent, prompt, self.config.num_inversion_steps)
        ddim_latent = reconstruct(self.components, trajectory, prompt)
        ddim = decode_latent(self.components.vae, ddim_latent)
        return reference, vae, ddim, valid_tensor


def _triptych(reference: torch.Tensor, vae: torch.Tensor, ddim: torch.Tensor, output: Path) -> None:
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (reference, vae, ddim)):
        raise ValueError("triptych requires finite image tensors")
    panels = [
        tensor.detach().to("cpu", torch.float32)[0].clamp(0, 1) for tensor in (reference, vae, ddim)
    ]
    combined = torch.cat(panels, dim=2)
    array = (combined.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(output, format="PNG")


def _manifest_sha(records: list[ManifestRecord]) -> str:
    canonical = "".join(
        json.dumps(asdict(record), sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _integrity_failure(output: Path, stage: str, expected: object, actual: object) -> None:
    write_json_atomic(
        output / "integrity_failure.json",
        {
            "schema_version": "p0_integrity_failure_v1",
            "stage": stage,
            "expected": expected,
            "actual": actual,
        },
    )
    raise RuntimeError(f"P0 integrity failure at {stage}")


def _image_accounting(
    selected: list[ManifestRecord], metrics: list[ImageMetrics]
) -> list[dict[str, object]]:
    completed = {item.image_relpath: item for item in metrics}
    metric_names = ("vae_l1", "ddim_l1", "extra_l1", "vae_psnr", "ddim_psnr")
    images: list[dict[str, object]] = []
    for record in selected:
        metric = completed.get(record.image_relpath)
        if metric is None:
            images.append(
                {
                    "category": record.category,
                    "image_relpath": record.image_relpath,
                    **{name: None for name in metric_names},
                    "finite": None,
                    "status": "unrun",
                    "reason": "short_circuited_non_finite",
                }
            )
            continue
        values = metric.as_dict()
        finite = all(math.isfinite(value) for value in values.values())
        images.append(
            {
                "category": record.category,
                "image_relpath": record.image_relpath,
                **{name: value if math.isfinite(value) else None for name, value in values.items()},
                "finite": finite,
                "status": "complete" if finite else "nonfinite",
                "reason": None if finite else "non_finite_metrics",
            }
        )
    return images


def run_p0(
    paths: ProjectConfig,
    experiment: ExperimentConfig,
    gate: GateConfig,
    output: Path,
    manifest_output: Path,
    lock_file: Path,
    config_paths: Mapping[str, Path],
    *,
    overwrite: bool = False,
    runner_factory: Callable[[ProjectConfig], ReconstructionRunner] = RealRunner,
    cwd: Path | None = None,
) -> int:
    cwd = (cwd or Path.cwd()).resolve()
    _validate_p0_contract(experiment, gate)
    validate_cublas_workspace_config()
    start_commit = assert_clean_git(cwd)
    manifest_output = validate_manifest_output(
        manifest_output, paths, [lock_file, *config_paths.values()]
    )
    output = ensure_output_directory(output, paths.runs_root, overwrite=overwrite)
    data_before = tree_snapshot(paths.dataset_root)
    visa_root = paths.dataset_root.parent / "VisA"
    visa_before = tree_snapshot(visa_root) if visa_root.is_dir() else None
    model_before = _model_weight_snapshot(paths.model_root)
    immutable_hashes = {
        "portable_lock_sha256": sha256_file(lock_file),
        "config_sha256": {name: sha256_file(path) for name, path in sorted(config_paths.items())},
        "model_required_sha256": {
            relative: sha256_file(paths.model_root / relative) for relative in REQUIRED
        },
    }
    manifest = build_mvtec_manifest(paths.dataset_root, experiment.split_id)
    manifest_sha = _manifest_sha(manifest)
    written_manifest_sha = write_manifest(manifest, manifest_output)
    if written_manifest_sha != manifest_sha:
        _integrity_failure(output, "manifest_write", manifest_sha, written_manifest_sha)
    selected = select_normal_train(manifest, experiment, gate)
    if len(experiment.categories) != 3 or len(selected) != 36:
        raise ValueError("P0 must select exactly 36 normal_train images")
    determinism = _set_determinism(experiment.seed)
    runner = runner_factory(paths)
    metrics: list[ImageMetrics] = []
    reconstructions = output / "reconstructions"
    reconstructions.mkdir()
    for record in selected:
        prompt = experiment.normal_prompt_template.format(category=record.category)
        reference, vae, ddim, valid = runner.reconstruct(record, prompt)
        measured = masked_reconstruction_metrics(reference, vae, ddim, valid)
        metric = ImageMetrics(record.category, record.image_relpath, **measured.as_dict())
        metrics.append(metric)
        if not all(math.isfinite(value) for value in metric.as_dict().values()):
            print(
                json.dumps(
                    {
                        "event": "p0_progress",
                        "image_relpath": record.image_relpath,
                        "finite": False,
                        "status": "short_circuited",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            break
        _triptych(
            reference,
            vae,
            ddim,
            reconstructions / f"{record.category}_{Path(record.image_relpath).stem}.png",
        )
        print(
            json.dumps(
                {"event": "p0_progress", "image_relpath": record.image_relpath}, sort_keys=True
            ),
            flush=True,
        )
    after_manifest = build_mvtec_manifest(paths.dataset_root, experiment.split_id)
    after_manifest_sha = _manifest_sha(after_manifest)
    if manifest_sha != after_manifest_sha:
        _integrity_failure(output, "manifest", manifest_sha, after_manifest_sha)
    visa_after = tree_snapshot(visa_root) if visa_before else None
    if (
        data_before != tree_snapshot(paths.dataset_root)
        or visa_before != visa_after
        or model_before != _model_weight_snapshot(paths.model_root)
    ):
        _integrity_failure(
            output,
            "read_only_snapshot",
            {"mvtec": data_before, "visa": visa_before, "model": model_before},
            {
                "mvtec": tree_snapshot(paths.dataset_root),
                "visa": visa_after,
                "model": _model_weight_snapshot(paths.model_root),
            },
        )
    end_commit = assert_clean_git(cwd)
    if end_commit != start_commit:
        _integrity_failure(output, "git_identity", start_commit, end_commit)
    report = evaluate_gate(metrics, gate)
    report["images"] = _image_accounting(selected, metrics)
    metrics_path = output / "gate_a_metrics.json"
    write_json_atomic(metrics_path, report)
    provenance = collect_provenance(paths, cwd=cwd)
    provenance.update(
        {
            "git_commit": end_commit,
            "git_dirty": False,
            "portable_lock_sha256": immutable_hashes["portable_lock_sha256"],
            "manifest_sha256": manifest_sha,
            "config_sha256": immutable_hashes["config_sha256"],
            "model_required_sha256": immutable_hashes["model_required_sha256"],
            "input_snapshots": {
                "mvtec": data_before,
                "visa": visa_before,
                "model_weights": model_before,
            },
            "selection": [
                {"image_relpath": item.image_relpath, "image_sha256": item.image_sha256}
                for item in selected
            ],
            "gate_summary": {
                "passed": report["passed"],
                "all_finite": report["all_finite"],
                "overall": report["overall"],
            },
            "gate_a_metrics_sha256": sha256_file(metrics_path),
            "output": str(output),
            "seed": experiment.seed,
            "determinism": determinism,
            "peak_cuda_memory_allocated": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
            "python_executable": sys.executable,
        }
    )
    write_json_atomic(output / "provenance.json", provenance)
    return 0 if report["passed"] else 2

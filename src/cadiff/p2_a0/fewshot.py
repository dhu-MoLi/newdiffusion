from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import Tensor

from cadiff.config import ProjectConfig
from cadiff.data.geometry import resize_pad_image, resize_pad_mask
from cadiff.diffusion.components import load_frozen_components, validate_model_layout
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.fewshot_config import P2A0FewshotConfig
from cadiff.p2_a0.masking import LatentMasks, build_latent_masks, build_masked_context
from cadiff.p2_a0.model import MaskedInfillAdapter, P2InfillNoisePredictor
from cadiff.p2_a0.objective import masked_infill_loss
from cadiff.p2_a0.probe import (
    _assert_parameter_versions_unchanged,
    _capture_parameter_versions,
    _git_state,
    _model_snapshot,
    _output_identity,
    _regular_input,
    _snapshot,
    _strict_object,
    _tensor_sha256,
    _tensor_to_pil,
    _validate_gpu_contract,
    _validate_hf_home,
    _validated_output,
    _verify_output_identity,
    capacity_optimizer_step,
)
from cadiff.p2_a0.single_type_config import P2A0SingleTypeConfig
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic


@dataclass(frozen=True)
class FewshotSample:
    sample_id: str
    defect_type: str
    role: str
    image_relpath: str
    mask_relpath: str
    image_path: Path
    mask_path: Path
    image_sha256: str
    mask_sha256: str
    pixels: Tensor
    mask: Tensor


@dataclass(frozen=True)
class FewshotSplit:
    train: tuple[FewshotSample, ...]
    evaluation: tuple[FewshotSample, ...]


@dataclass(frozen=True)
class PreparedSample:
    sample: FewshotSample
    target: Tensor
    context: Tensor
    masks: LatentMasks


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    defect_type: str
    image_relpath: str
    mask_relpath: str
    wrong_type: str
    correct: dict[str, float]
    wrong: dict[str, float]
    normal: dict[str, float]
    prediction_sha256: dict[str, str]


@dataclass(frozen=True)
class FewshotContext:
    paths: ProjectConfig
    experiment: P2A0FewshotConfig | P2A0SingleTypeConfig
    manifest: Path
    lock: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class FewshotPreflight:
    context: FewshotContext
    split: FewshotSplit
    git_head: str
    input_hashes: dict[str, str]
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]


_EXPECTED_COUNTS = {
    "fewshot_train": {"broken_large": 6, "broken_small": 7, "contamination": 7},
    "open_dev_report": {"broken_large": 14, "broken_small": 15, "contamination": 14},
}
_EXPECTED_COUNTS_BY_CATEGORY = {
    "bottle": _EXPECTED_COUNTS,
    "hazelnut": {
        "fewshot_train": {"crack": 6, "cut": 5, "hole": 6, "print": 5},
        "open_dev_report": {"crack": 12, "cut": 12, "hole": 12, "print": 12},
    },
    "cable": {
        "fewshot_train": {
            "bent_wire": 4,
            "cut_inner_insulation": 4,
            "cut_outer_insulation": 3,
            "poke_insulation": 3,
        },
        "open_dev_report": {
            "bent_wire": 9,
            "cut_inner_insulation": 10,
            "cut_outer_insulation": 7,
            "poke_insulation": 7,
        },
    },
}
_METRIC_KEYS = ("total", "noise", "inside", "outside")
_OUTSIDE_NORMAL_RATIO_LIMIT = 1.10


def _active_defect_types(
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> tuple[str, ...]:
    if isinstance(config, P2A0SingleTypeConfig):
        return config.active_defect_types
    return config.defect_types


def _expected_counts(
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> dict[str, dict[str, int]]:
    if isinstance(config, P2A0SingleTypeConfig):
        return {
            config.train_role: config.expected_train_counts,
            config.eval_role: config.expected_eval_counts,
        }
    try:
        return _EXPECTED_COUNTS_BY_CATEGORY[config.category]
    except KeyError as error:
        raise ValueError("few-shot category has no frozen expected counts") from error


def _artifact_schema(
    config: P2A0FewshotConfig | P2A0SingleTypeConfig, artifact: str
) -> str:
    prefix = (
        "p2_a0_single_type"
        if isinstance(config, P2A0SingleTypeConfig)
        else "p2_a0_fewshot"
    )
    return f"{prefix}_{artifact}_v1"


def _contained_file(root: Path, relpath: str, *, name: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"{name} relpath must be a non-empty string")
    path = (root / relpath).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes dataset root: {path}") from error
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    return path


def _load_sample(
    raw: dict[str, object],
    root: Path,
    role: str,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> FewshotSample:
    defect_type = raw.get("defect_type")
    image_relpath = raw.get("image_relpath")
    mask_relpath = raw.get("mask_relpath")
    image_sha256 = raw.get("image_sha256")
    mask_sha256 = raw.get("mask_sha256")
    if defect_type not in config.defect_types:
        raise ValueError(f"unsupported defect_type: {defect_type}")
    for field, value in (
        ("image_relpath", image_relpath),
        ("mask_relpath", mask_relpath),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    for field, value in (("image_sha256", image_sha256), ("mask_sha256", mask_sha256)):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{field} must be a SHA256 string")
    image_path = _contained_file(root, image_relpath, name="image")
    mask_path = _contained_file(root, mask_relpath, name="mask")
    if sha256_file(image_path) != image_sha256:
        raise ValueError("fewshot image SHA256 mismatch")
    if sha256_file(mask_path) != mask_sha256:
        raise ValueError("fewshot mask SHA256 mismatch")
    with Image.open(image_path) as image:
        transformed, _, geometry = resize_pad_image(image, config.image_size)
    with Image.open(mask_path) as mask_image:
        transformed_mask = resize_pad_mask(mask_image, geometry)
    pixels = (
        torch.from_numpy(np.array(transformed, dtype=np.uint8, copy=True))
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(127.5)
        .sub_(1.0)
        .contiguous()
    )
    mask = (
        torch.from_numpy(np.array(transformed_mask, dtype=np.uint8, copy=True))
        .ge(128)
        .to(dtype=torch.float32)
        .unsqueeze(0)
        .contiguous()
    )
    sample_id = hashlib.sha256(
        f"{config.category}\n{defect_type}\n{image_relpath}".encode()
    ).hexdigest()
    return FewshotSample(
        sample_id=sample_id,
        defect_type=defect_type,
        role=role,
        image_relpath=image_relpath,
        mask_relpath=mask_relpath,
        image_path=image_path,
        mask_path=mask_path,
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        pixels=pixels,
        mask=mask,
    )


def load_fewshot_split(
    manifest: Path,
    dataset_root: Path,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> FewshotSplit:
    if manifest.is_symlink():
        raise ValueError(f"manifest symlink is not allowed: {manifest}")
    manifest = manifest.resolve(strict=True)
    if not manifest.is_file():
        raise ValueError(f"manifest must be a regular file: {manifest}")
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dataset root must be a directory: {root}")
    selected: dict[str, list[FewshotSample]] = {
        config.train_role: [],
        config.eval_role: [],
    }
    raw_seen: set[str] = set()
    resolved_seen: set[Path] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = _strict_object(line, line_number)
            image_relpath = raw.get("image_relpath")
            if not isinstance(image_relpath, str) or not image_relpath:
                raise ValueError(f"manifest line {line_number} image_relpath must be non-empty")
            if image_relpath in raw_seen:
                raise ValueError(f"duplicate image_relpath: {image_relpath}")
            raw_seen.add(image_relpath)
            roles = raw.get("roles")
            if raw.get("category") != config.category or not isinstance(roles, list):
                continue
            matched = [role for role in selected if role in roles]
            if not matched:
                continue
            if len(matched) != 1:
                raise ValueError("sample cannot belong to both train and evaluation roles")
            if raw.get("defect_type") not in _active_defect_types(config):
                continue
            item = _load_sample(raw, root, matched[0], config)
            if item.image_path in resolved_seen:
                raise ValueError(f"duplicate resolved image path: {item.image_path}")
            resolved_seen.add(item.image_path)
            selected[matched[0]].append(item)
    active_types = _active_defect_types(config)
    order = {label: index for index, label in enumerate(active_types)}
    for role, expected in _expected_counts(config).items():
        actual = {label: 0 for label in active_types}
        for item in selected[role]:
            actual[item.defect_type] += 1
        if actual != expected:
            raise ValueError(f"{role} counts must be {expected}, got {actual}")
        selected[role].sort(key=lambda item: (order[item.defect_type], item.image_relpath))
    return FewshotSplit(
        train=tuple(selected[config.train_role]),
        evaluation=tuple(selected[config.eval_role]),
    )


def balanced_training_order(
    samples: tuple[FewshotSample, ...],
    steps: int,
    defect_types: tuple[str, ...],
) -> tuple[FewshotSample, ...]:
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    groups = {
        label: tuple(
            sorted(
                (item for item in samples if item.defect_type == label),
                key=lambda item: item.image_relpath,
            )
        )
        for label in defect_types
    }
    if any(not group for group in groups.values()):
        raise ValueError("every defect type must have at least one training sample")
    offsets = {label: 0 for label in defect_types}
    result: list[FewshotSample] = []
    for step in range(steps):
        label = defect_types[step % len(defect_types)]
        group = groups[label]
        result.append(group[offsets[label] % len(group)])
        offsets[label] += 1
    return tuple(result)


def _resampled_training_steps(
    schedule: tuple[FewshotSample, ...],
    prepared_by_id: dict[str, PreparedSample],
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> Iterator[tuple[int, FewshotSample, PreparedSample, int, str]]:
    for step, sample in enumerate(schedule, start=1):
        try:
            original = prepared_by_id[sample.sample_id]
        except KeyError as error:
            raise ValueError(
                f"training sample is missing prepared latent: {sample.sample_id}"
            ) from error
        context_seed = config.context_fill_seed + 20_000_000 + step
        context = build_masked_context(
            original.target,
            original.masks.binary,
            torch.Generator(device=original.target.device).manual_seed(context_seed),
        )
        prepared = PreparedSample(
            sample=original.sample,
            target=original.target,
            context=context,
            masks=original.masks,
        )
        yield step, sample, prepared, context_seed, _tensor_sha256(context)


def _wrong_type(
    defect_type: str, config: P2A0FewshotConfig | P2A0SingleTypeConfig
) -> str:
    if config.wrong_type_policy != "cyclic_next":
        raise ValueError("unsupported wrong_type_policy")
    try:
        index = config.defect_types.index(defect_type)
    except ValueError as error:
        raise ValueError(f"unknown defect type: {defect_type}") from error
    return config.defect_types[(index + 1) % len(config.defect_types)]


@torch.no_grad()
def _evaluate_prepared_sample(
    *,
    model: Any,
    unet: Any,
    scheduler: object,
    prepared: PreparedSample,
    prompt: Tensor,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
    sample_index: int,
) -> EvaluationRecord:
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    wrong_type = _wrong_type(prepared.sample.defect_type, config)
    accumulated = {
        condition: {metric: [] for metric in _METRIC_KEYS}
        for condition in ("correct", "wrong", "normal")
    }
    prediction_hashes = {condition: [] for condition in ("correct", "wrong", "normal")}
    for timestep in config.eval_timesteps:
        timesteps = torch.tensor([timestep], device=prepared.target.device, dtype=torch.long)
        generator = torch.Generator(device=prepared.target.device).manual_seed(
            config.noise_seed + 1_000_000 + sample_index * 10_000 + timestep
        )
        noise = torch.randn(
            prepared.target.shape,
            generator=generator,
            device=prepared.target.device,
            dtype=prepared.target.dtype,
        )
        noisy = scheduler.add_noise(prepared.target, noise, timesteps)  # type: ignore[union-attr]
        predictions = {
            "correct": model(
                noisy,
                prepared.context,
                prepared.masks.binary,
                timesteps,
                prompt,
                prepared.sample.defect_type,
            ),
            "wrong": model(
                noisy,
                prepared.context,
                prepared.masks.binary,
                timesteps,
                prompt,
                wrong_type,
            ),
            "normal": unet(noisy, timesteps, encoder_hidden_states=prompt).sample,
        }
        for condition, prediction in predictions.items():
            losses = masked_infill_loss(
                scheduler=scheduler,
                noisy=noisy,
                epsilon_hat=prediction,
                epsilon_target=noise,
                target_latent=prepared.target,
                soft_mask=prepared.masks.soft,
                timesteps=timesteps,
                inside_noise_weight=config.inside_noise_weight,
                lambda_inside=config.lambda_inside,
                lambda_outside=config.lambda_outside,
            )
            for metric in _METRIC_KEYS:
                accumulated[condition][metric].append(float(getattr(losses, metric).item()))
            prediction_hashes[condition].append(_tensor_sha256(prediction))
    metrics = {
        condition: {
            metric: float(np.mean(values, dtype=np.float64))
            for metric, values in by_metric.items()
        }
        for condition, by_metric in accumulated.items()
    }
    hashes = {
        condition: hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()
        for condition, values in prediction_hashes.items()
    }
    return EvaluationRecord(
        sample_id=prepared.sample.sample_id,
        defect_type=prepared.sample.defect_type,
        image_relpath=prepared.sample.image_relpath,
        mask_relpath=prepared.sample.mask_relpath,
        wrong_type=wrong_type,
        correct=metrics["correct"],
        wrong=metrics["wrong"],
        normal=metrics["normal"],
        prediction_sha256=hashes,
    )


def _save_checkpoint(
    output: Path,
    *,
    adapter: torch.nn.Module,
    conditioner: torch.nn.Module,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
    git_head: str,
    vocabulary_sha256: str,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"checkpoint output already exists: {output}")
    if len(git_head) != 40 or len(vocabulary_sha256) != 64:
        raise ValueError("checkpoint origin hashes are malformed")
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, int | str]] = {}
    for name, module in (("adapter", adapter), ("conditioner", conditioner)):
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in module.state_dict().items()
        }
        if not state or any(not torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError(f"{name} checkpoint state must be non-empty and finite")
        path = output / f"{name}.safetensors"
        save_file(state, path)
        files[path.name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    metadata: dict[str, Any] = {
        "schema_version": _artifact_schema(config, "checkpoint"),
        "git_head": git_head,
        "vocabulary_sha256": vocabulary_sha256,
        "config": asdict(config),
        "files": files,
    }
    write_json_atomic(output / "metadata.json", metadata)
    return metadata


def _validate_metric_map(metrics: dict[str, float], *, name: str) -> None:
    if set(metrics) != set(_METRIC_KEYS):
        raise ValueError(f"{name} metrics must contain {_METRIC_KEYS}")
    if any(
        type(value) is not float or not math.isfinite(value) or value < 0
        for value in metrics.values()
    ):
        raise ValueError(f"{name} metrics must be finite non-negative floats")


def summarize_evaluation(
    records: tuple[EvaluationRecord, ...],
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> dict[str, Any]:
    if not records:
        raise ValueError("evaluation records must be non-empty")
    per_type: dict[str, Any] = {}
    counts: dict[str, int] = {}
    gates: list[bool] = []
    for label in _active_defect_types(config):
        selected = tuple(record for record in records if record.defect_type == label)
        if not selected:
            raise ValueError(f"evaluation records missing defect type: {label}")
        counts[label] = len(selected)
        medians: dict[str, dict[str, float]] = {}
        for condition in ("correct", "wrong", "normal"):
            maps = [getattr(record, condition) for record in selected]
            for index, metrics in enumerate(maps):
                _validate_metric_map(metrics, name=f"{label}/{condition}/{index}")
            medians[condition] = {
                metric: float(statistics.median(values[metric] for values in maps))
                for metric in _METRIC_KEYS
            }
        correct_wrong_ratio = medians["correct"]["total"] / max(
            medians["wrong"]["total"], 1e-12
        )
        correct_normal_ratio = medians["correct"]["total"] / max(
            medians["normal"]["total"], 1e-12
        )
        outside_normal_ratio = medians["correct"]["outside"] / max(
            medians["normal"]["outside"], 1e-12
        )
        better_wrong = correct_wrong_ratio < config.correct_wrong_ratio_limit
        better_normal = correct_normal_ratio < config.correct_normal_ratio_limit
        outside_stable = outside_normal_ratio <= _OUTSIDE_NORMAL_RATIO_LIMIT
        type_separated = all(
            record.prediction_sha256["correct"]
            not in {
                record.prediction_sha256["wrong"],
                record.prediction_sha256["normal"],
            }
            for record in selected
        )
        correct_normal_separated = all(
            record.prediction_sha256["correct"]
            != record.prediction_sha256["normal"]
            for record in selected
        )
        if isinstance(config, P2A0SingleTypeConfig):
            gates.extend((better_normal, outside_stable, correct_normal_separated))
        else:
            gates.extend((better_wrong, better_normal, outside_stable, type_separated))
        type_summary = {
            **medians,
            "correct_wrong_total_ratio": correct_wrong_ratio,
            "correct_normal_total_ratio": correct_normal_ratio,
            "outside_normal_ratio": outside_normal_ratio,
            "correct_better_than_wrong": better_wrong,
            "correct_better_than_normal": better_normal,
            "outside_stable": outside_stable,
            "type_separated": type_separated,
        }
        if isinstance(config, P2A0SingleTypeConfig):
            type_summary["correct_normal_separated"] = correct_normal_separated
        per_type[label] = type_summary
    overall = {
        condition: {
            metric: float(
                statistics.median(getattr(record, condition)[metric] for record in records)
            )
            for metric in _METRIC_KEYS
        }
        for condition in ("correct", "wrong", "normal")
    }
    thresholds = {
        "correct_normal_total_ratio_lt": config.correct_normal_ratio_limit,
        "outside_normal_ratio_lte": _OUTSIDE_NORMAL_RATIO_LIMIT,
    }
    if not isinstance(config, P2A0SingleTypeConfig):
        thresholds["correct_wrong_total_ratio_lt"] = config.correct_wrong_ratio_limit
    result = {
        "counts": counts,
        "per_type": per_type,
        "overall": overall,
        "gate_thresholds": thresholds,
        "gate_passed": all(gates),
    }
    if isinstance(config, P2A0SingleTypeConfig):
        result["gate_policy"] = config.gate_policy
    return result


def _validate_evaluation_records(
    records: tuple[EvaluationRecord, ...],
    split: FewshotSplit,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> None:
    if len(records) != len(split.evaluation):
        raise ValueError(
            f"evaluation record count must be {len(split.evaluation)}, got {len(records)}"
        )
    for index, (record, sample) in enumerate(
        zip(records, split.evaluation, strict=True)
    ):
        if not isinstance(record, EvaluationRecord):
            raise ValueError(f"evaluation record {index} has an invalid type")
        if (
            record.sample_id != sample.sample_id
            or record.defect_type != sample.defect_type
            or record.image_relpath != sample.image_relpath
            or record.mask_relpath != sample.mask_relpath
        ):
            raise ValueError(f"evaluation record {index} does not match the frozen split")
        if record.wrong_type != _wrong_type(record.defect_type, config):
            raise ValueError(f"evaluation record {index} has the wrong control label")
        for condition in ("correct", "wrong", "normal"):
            _validate_metric_map(
                getattr(record, condition),
                name=f"evaluation/{index}/{condition}",
            )
        if set(record.prediction_sha256) != {"correct", "wrong", "normal"}:
            raise ValueError(f"evaluation record {index} has invalid prediction hashes")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in record.prediction_sha256.values()
        ):
            raise ValueError(f"evaluation record {index} has malformed prediction hashes")


def _validate_training_records(
    records: tuple[dict[str, Any], ...],
    split: FewshotSplit,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
) -> None:
    if len(records) != config.steps:
        raise ValueError(f"training record count must be {config.steps}, got {len(records)}")
    schedule = balanced_training_order(
        split.train, config.steps, _active_defect_types(config)
    )
    expected_keys = {
        "step",
        "sample_id",
        "defect_type",
        "timestep",
        "total",
        "noise",
        "inside",
        "outside",
        "grad_l2",
        "finite",
    }
    expected_keys.update(("context_seed", "context_sha256"))
    context_hashes: set[str] = set()
    for index, (record, sample) in enumerate(zip(records, schedule, strict=True), start=1):
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ValueError(f"training record {index} has an invalid schema")
        if (
            record["step"] != index
            or record["sample_id"] != sample.sample_id
            or record["defect_type"] != sample.defect_type
        ):
            raise ValueError(f"training record {index} does not match the frozen schedule")
        timestep = record["timestep"]
        if type(timestep) is not int or not 0 <= timestep < 1000:
            raise ValueError(f"training record {index} has an invalid timestep")
        expected_context_seed = config.context_fill_seed + 20_000_000 + index
        if (
            type(record["context_seed"]) is not int
            or record["context_seed"] != expected_context_seed
        ):
            raise ValueError(f"training record {index} has an invalid context_seed")
        context_sha256 = record["context_sha256"]
        if (
            not isinstance(context_sha256, str)
            or len(context_sha256) != 64
            or any(character not in "0123456789abcdef" for character in context_sha256)
        ):
            raise ValueError(f"training record {index} has an invalid context_sha256")
        context_hashes.add(context_sha256)
        for field in ("total", "noise", "inside", "outside", "grad_l2"):
            value = record[field]
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"training record {index} has invalid {field}")
        if record["grad_l2"] <= 0.0 or record["finite"] is not True:
            raise ValueError(f"training record {index} is not finite with nonzero gradients")
    if len(context_hashes) != config.steps:
        raise ValueError("training contexts must be unique per step")


def _source_snapshot(split: FewshotSplit) -> dict[str, dict[str, int | str]]:
    return {
        relative: _snapshot(path)
        for item in (*split.train, *split.evaluation)
        for relative, path in (
            (item.image_relpath, item.image_path),
            (item.mask_relpath, item.mask_path),
        )
    }


def validate_fewshot_preflight(context: FewshotContext) -> FewshotPreflight:
    _validate_gpu_contract()
    _validate_hf_home(context.cwd)
    head, dirty = _git_state(context.cwd)
    if dirty:
        raise RuntimeError("Git worktree must be clean")
    _validated_output(context.output, context.paths.runs_root)
    validate_model_layout(context.paths.model_root)
    manifest = _regular_input(context.manifest, name="manifest")
    lock = _regular_input(context.lock, name="lock")
    config_paths = {
        name: _regular_input(path, name=f"config {name}")
        for name, path in sorted(context.config_paths.items())
    }
    split = load_fewshot_split(manifest, context.paths.dataset_root, context.experiment)
    return FewshotPreflight(
        context=context,
        split=split,
        git_head=head,
        input_hashes={
            "manifest": sha256_file(manifest),
            "lock": sha256_file(lock),
            **{f"config:{name}": sha256_file(path) for name, path in config_paths.items()},
        },
        source_snapshot=_source_snapshot(split),
        model_snapshot=_model_snapshot(context.paths.model_root),
    )


def _write_jsonl_atomic(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                )
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_integrity(preflight: FewshotPreflight, identity: tuple[int, int]) -> str:
    head, dirty = _git_state(preflight.context.cwd)
    if dirty or head != preflight.git_head:
        raise RuntimeError("Git changed during fewshot run")
    if _source_snapshot(preflight.split) != preflight.source_snapshot:
        raise RuntimeError("fewshot source changed during run")
    if _model_snapshot(preflight.context.paths.model_root) != preflight.model_snapshot:
        raise RuntimeError("model files changed during fewshot run")
    current_hashes = {
        "manifest": sha256_file(preflight.context.manifest.resolve(strict=True)),
        "lock": sha256_file(preflight.context.lock.resolve(strict=True)),
        **{
            f"config:{name}": sha256_file(path.resolve(strict=True))
            for name, path in sorted(preflight.context.config_paths.items())
        },
    }
    if current_hashes != preflight.input_hashes:
        raise RuntimeError("fewshot input hash changed during run")
    _verify_output_identity(preflight, identity)  # type: ignore[arg-type]
    return head


@torch.no_grad()
def _prepare_samples(
    *,
    vae: Any,
    samples: tuple[FewshotSample, ...],
    device: torch.device,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
    seed_offset: int,
) -> tuple[PreparedSample, ...]:
    prepared: list[PreparedSample] = []
    for index, sample in enumerate(samples):
        pixels = sample.pixels.unsqueeze(0).to(device=device, dtype=torch.float16)
        target = encode_mode(vae, pixels).detach().clone()
        masks = build_latent_masks(
            sample.mask.unsqueeze(0).to(device=device),
            (target.shape[2], target.shape[3]),
        )
        context = build_masked_context(
            target,
            masks.binary,
            torch.Generator(device=device).manual_seed(
                config.context_fill_seed + seed_offset + index
            ),
        )
        prepared.append(
            PreparedSample(
                sample=sample,
                target=target,
                context=context,
                masks=masks,
            )
        )
    return tuple(prepared)


@torch.no_grad()
def _visualize_sample(
    *,
    model: P2InfillNoisePredictor,
    unet: Any,
    vae: Any,
    scheduler: object,
    prepared: PreparedSample,
    prompt: Tensor,
    config: P2A0FewshotConfig | P2A0SingleTypeConfig,
    sample_index: int,
    output: Path,
) -> tuple[str, str]:
    timestep_value = config.eval_timesteps[len(config.eval_timesteps) // 2]
    timesteps = torch.tensor(
        [timestep_value], device=prepared.target.device, dtype=torch.long
    )
    generator = torch.Generator(device=prepared.target.device).manual_seed(
        config.noise_seed + 1_000_000 + sample_index * 10_000 + timestep_value
    )
    noise = torch.randn(
        prepared.target.shape,
        generator=generator,
        device=prepared.target.device,
        dtype=prepared.target.dtype,
    )
    noisy = scheduler.add_noise(prepared.target, noise, timesteps)  # type: ignore[union-attr]
    predictions = {
        "correct": model(
            noisy,
            prepared.context,
            prepared.masks.binary,
            timesteps,
            prompt,
            prepared.sample.defect_type,
        ),
        "wrong": model(
            noisy,
            prepared.context,
            prepared.masks.binary,
            timesteps,
            prompt,
            _wrong_type(prepared.sample.defect_type, config),
        ),
        "normal": unet(noisy, timesteps, encoder_hidden_states=prompt).sample,
    }
    decoded: dict[str, Tensor] = {}
    for condition, prediction in predictions.items():
        losses = masked_infill_loss(
            scheduler=scheduler,
            noisy=noisy,
            epsilon_hat=prediction,
            epsilon_target=noise,
            target_latent=prepared.target,
            soft_mask=prepared.masks.soft,
            timesteps=timesteps,
            inside_noise_weight=config.inside_noise_weight,
            lambda_inside=config.lambda_inside,
            lambda_outside=config.lambda_outside,
        )
        composite = (
            prepared.masks.binary * losses.predicted_clean
            + (1.0 - prepared.masks.binary) * prepared.target.float()
        ).to(dtype=prepared.target.dtype)
        decoded[condition] = decode_latent(vae, composite)[0]
    cells = [
        _tensor_to_pil(
            ((prepared.sample.pixels.float().cpu() + 1.0) / 2.0).clamp(0, 1)
        ),
        _tensor_to_pil(decode_latent(vae, prepared.target)[0]),
        _tensor_to_pil(decode_latent(vae, prepared.context)[0]),
        _tensor_to_pil(decoded["correct"]),
        _tensor_to_pil(decoded["wrong"]),
        _tensor_to_pil(decoded["normal"]),
    ]
    visual_root = output / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    grid = Image.new("RGB", (config.image_size * len(cells), config.image_size))
    for index, cell in enumerate(cells):
        grid.paste(cell, (index * config.image_size, 0))
    grid_path = visual_root / f"{prepared.sample.defect_type}_grid.png"
    grid.save(grid_path)
    mask_path = visual_root / f"{prepared.sample.defect_type}_mask.png"
    Image.fromarray(
        (prepared.sample.mask[0].numpy() * 255).astype(np.uint8), mode="L"
    ).save(mask_path)
    return str(grid_path.relative_to(output)), str(mask_path.relative_to(output))


def _execute_fewshot(preflight: FewshotPreflight) -> dict[str, Any]:
    context = preflight.context
    config = context.experiment
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats()
    components = load_frozen_components(context.paths.model_root, device, torch.float16)
    unet_versions = _capture_parameter_versions(components.unet)
    prompt = encode_prompt(components, config.normal_prompt, device).detach().clone()
    train_prepared = _prepare_samples(
        vae=components.vae,
        samples=preflight.split.train,
        device=device,
        config=config,
        seed_offset=0,
    )
    evaluation_prepared = _prepare_samples(
        vae=components.vae,
        samples=preflight.split.evaluation,
        device=device,
        config=config,
        seed_offset=len(train_prepared),
    )
    prepared_by_id = {item.sample.sample_id: item for item in train_prepared}
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    conditioner = DefectConditioner(vocabulary, 128).to(device=device, dtype=torch.float32)
    adapter = MaskedInfillAdapter(config).to(device=device, dtype=torch.float32)
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    schedule = balanced_training_order(
        preflight.split.train,
        config.steps,
        _active_defect_types(config),
    )
    generator = torch.Generator(device=device).manual_seed(config.noise_seed + 10_000_000)
    training: list[dict[str, Any]] = []
    num_train_timesteps = int(components.scheduler.alphas_cumprod.numel())
    training_steps = _resampled_training_steps(schedule, prepared_by_id, config)
    for step, sample, prepared, context_seed, context_sha256 in training_steps:
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (1,),
            generator=generator,
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn(
            prepared.target.shape,
            generator=generator,
            device=device,
            dtype=prepared.target.dtype,
        )
        metrics = capacity_optimizer_step(
            model=model,
            scheduler=components.scheduler,
            optimizer=optimizer,
            scaler=scaler,
            target_latent=prepared.target,
            context=prepared.context,
            binary_mask=prepared.masks.binary,
            soft_mask=prepared.masks.soft,
            prompt_embeds=prompt,
            defect_type=sample.defect_type,
            timesteps=timesteps,
            noise=noise,
            config=config,
        )
        training_record = {
            "step": step,
            "sample_id": sample.sample_id,
            "defect_type": sample.defect_type,
            "timestep": int(timesteps.item()),
            **asdict(metrics),
        }
        training_record.update(
            {"context_seed": context_seed, "context_sha256": context_sha256}
        )
        training.append(training_record)
        if step == 1 or step % 100 == 0 or step == config.steps:
            print(
                json.dumps(
                    {
                        "stage": "training",
                        "step": step,
                        "steps": config.steps,
                        "total": metrics.total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval()
    records: list[EvaluationRecord] = []
    visuals: list[str] = []
    visualized: set[str] = set()
    for index, prepared in enumerate(evaluation_prepared):
        records.append(
            _evaluate_prepared_sample(
                model=model,
                unet=components.unet,
                scheduler=components.scheduler,
                prepared=prepared,
                prompt=prompt,
                config=config,
                sample_index=index,
            )
        )
        if prepared.sample.defect_type not in visualized:
            visualized.add(prepared.sample.defect_type)
            visuals.extend(
                _visualize_sample(
                    model=model,
                    unet=components.unet,
                    vae=components.vae,
                    scheduler=components.scheduler,
                    prepared=prepared,
                    prompt=prompt,
                    config=config,
                    sample_index=index,
                    output=context.output,
                )
            )
        if (index + 1) % 10 == 0 or index + 1 == len(evaluation_prepared):
            print(
                json.dumps(
                    {
                        "stage": "evaluation",
                        "complete": index + 1,
                        "total": len(evaluation_prepared),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    checkpoint = _save_checkpoint(
        context.output / "checkpoint",
        adapter=adapter,
        conditioner=conditioner,
        config=config,
        git_head=preflight.git_head,
        vocabulary_sha256=vocabulary.sha256,
    )
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated()
    _assert_parameter_versions_unchanged(components.unet, unet_versions)
    if any(parameter.grad is not None for parameter in components.unet.parameters()):
        raise RuntimeError("frozen U-Net accumulated gradients")
    return {
        "records": tuple(records),
        "training": tuple(training),
        "checkpoint": checkpoint,
        "visuals": visuals,
        "peak_vram_bytes": peak,
    }


def run_fewshot(context: FewshotContext) -> int:
    try:
        preflight = validate_fewshot_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_fewshot(preflight)
        records = tuple(result["records"])
        _validate_evaluation_records(records, preflight.split, context.experiment)
        training = tuple(result["training"])
        _validate_training_records(training, preflight.split, context.experiment)
        summary = summarize_evaluation(records, context.experiment)
        head_end = _verify_integrity(preflight, identity)
        _write_jsonl_atomic(
            context.output / "metrics.jsonl",
            tuple(asdict(record) for record in records),
        )
        _verify_integrity(preflight, identity)
        _write_jsonl_atomic(
            context.output / "training_metrics.jsonl",
            tuple(dict(record) for record in training),
        )
        _verify_integrity(preflight, identity)
        report = {
            "schema_version": _artifact_schema(context.experiment, "report"),
            "status": "complete",
            "reason": None,
            "gate_passed": summary["gate_passed"],
            "summary": summary,
            "checkpoint": result["checkpoint"],
            "visuals": result["visuals"],
            "peak_vram_bytes": result["peak_vram_bytes"],
            "input_hashes": preflight.input_hashes,
        }
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": _artifact_schema(context.experiment, "provenance"),
                "git_head_start": preflight.git_head,
                "git_head_end": head_end,
                "git_dirty": False,
                "input_hashes": preflight.input_hashes,
                "source_snapshot_before": preflight.source_snapshot,
                "source_snapshot_after": preflight.source_snapshot,
                "model_snapshot_before": preflight.model_snapshot,
                "model_snapshot_after": preflight.model_snapshot,
            }
        )
        write_json_atomic(context.output / "provenance.json", provenance)
        _verify_integrity(preflight, identity)
        write_json_atomic(context.output / "report.json", report)
        _verify_integrity(preflight, identity)
        return 0 if summary["gate_passed"] else 2
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": _artifact_schema(context.experiment, "report"),
            "status": "failed",
            "reason": "runtime_or_integrity_failure",
        }
        try:
            _verify_output_identity(preflight, identity)  # type: ignore[arg-type]
        except (OSError, RuntimeError):
            pass
        else:
            try:
                write_json_atomic(context.output / "report.json", failed)
            except OSError:
                pass
        print(json.dumps(failed, sort_keys=True))
        return 1

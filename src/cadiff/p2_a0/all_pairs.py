from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from cadiff.config import ProjectConfig, load_config
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.fewshot import (
    FewshotContext,
    FewshotSplit,
    PreparedSample,
    _assert_parameter_versions_unchanged,
    _capture_parameter_versions,
    _git_state,
    _model_snapshot,
    _output_identity,
    _prepare_samples,
    _snapshot,
    _source_snapshot,
    _tensor_sha256,
    _verify_output_identity,
    _write_jsonl_atomic,
    validate_fewshot_preflight,
)
from cadiff.p2_a0.fewshot_config import P2A0FewshotConfig, load_p2_a0_fewshot_config
from cadiff.p2_a0.model import MaskedInfillAdapter, P2InfillNoisePredictor
from cadiff.p2_a0.objective import masked_infill_loss
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic

METRIC_KEYS = ("total", "noise", "inside", "outside")
NOISE_BANK_OFFSETS = (0, 50_000_000)
SCHEMA_VERSION = "p2_a0_type_routed_all_pairs_v1"
SOURCE_RUN_RELATIVE = Path(
    "p2_a0/bottle/fewshot/context_resampling/type_routed_heads/"
    "seed-20260817-attempt1"
)
SOURCE_CHECKPOINT_HEAD = "7ac8d2c7eb948e5ff6d84c8f7bc6a6347de7ef67"
SOURCE_VOCABULARY_SHA256 = (
    "d7432094e45c5ea99fafc2e4e8e28e9db16cb221f74968db2245f2017e791f80"
)
SOURCE_HASHES = {
    "checkpoint/metadata.json": (
        "0b6e541d23cd62aa34387ef4f7fe4b2f09b423936ac9aa377c086f6ed37e2b29"
    ),
    "checkpoint/adapter.safetensors": (
        "b911a7c983829c2a953cd5dcb9b9c59a7da410cf54f4af24ff9fc5e29a82bf71"
    ),
    "checkpoint/conditioner.safetensors": (
        "6128af9099309a0db60e332378b8747110bd26ca9a41f73aac19dcb5b03fd048"
    ),
    "report.json": "db0e9049a15806e42f260047f01bdb0d941a38e16dbe6e17fe21c21bb096adf3",
    "provenance.json": (
        "6343736bd5145b37662d359810befafa26ba3d0c9b85f055bfd8242c4b41fa0c"
    ),
    "metrics.jsonl": "be1a25325f7583999be0cf9acba044b101724f182798f72bdf0120f08831a104",
    "training_metrics.jsonl": (
        "dfaa31f32072cd993b3f7dc6d44fd10d593562db2c5800b0239461f45e03bb9a"
    ),
}


@dataclass(frozen=True)
class AllPairsContext:
    paths: ProjectConfig
    experiment: P2A0FewshotConfig
    manifest: Path
    lock: Path
    source_run: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class AllPairsPreflight:
    context: AllPairsContext
    split: FewshotSplit
    git_head: str
    input_hashes: dict[str, str]
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]
    reference_snapshot: dict[str, dict[str, int | str]]
    reference_hashes: dict[str, str]
    checkpoint_metadata: dict[str, Any]


def _validate_labels(labels: tuple[str, ...]) -> None:
    if (
        not isinstance(labels, tuple)
        or len(labels) < 2
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("labels must be a unique non-empty tuple")


def _validate_metrics(value: object, *, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(METRIC_KEYS):
        raise ValueError(f"{name} must contain the exact metric keys")
    if any(
        type(metric) is not float or not math.isfinite(metric) or metric < 0.0
        for metric in value.values()
    ):
        raise ValueError(f"{name} metrics must be finite non-negative floats")
    return value


def _validate_records(
    records: tuple[dict[str, Any], ...],
    labels: tuple[str, ...],
    bank_offsets: tuple[int, ...],
) -> None:
    _validate_labels(labels)
    if (
        not isinstance(bank_offsets, tuple)
        or not bank_offsets
        or any(type(offset) is not int or offset < 0 for offset in bank_offsets)
        or len(set(bank_offsets)) != len(bank_offsets)
    ):
        raise ValueError("bank_offsets must be unique non-negative integers")
    if not records:
        raise ValueError("records must be non-empty")
    expected_conditions = {*labels, "normal"}
    seen: set[tuple[int, str]] = set()
    samples_by_bank: dict[int, set[str]] = {
        index: set() for index in range(len(bank_offsets))
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        bank_index = record.get("bank_index")
        if type(bank_index) is not int or bank_index not in samples_by_bank:
            raise ValueError(f"record {index} has an invalid bank_index")
        if record.get("noise_seed_offset") != bank_offsets[bank_index]:
            raise ValueError(f"record {index} has an invalid noise_seed_offset")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"record {index} has an invalid sample_id")
        identity = (bank_index, sample_id)
        if identity in seen:
            raise ValueError(f"duplicate sample-bank record: {identity}")
        seen.add(identity)
        samples_by_bank[bank_index].add(sample_id)
        true_type = record.get("true_type")
        if true_type not in labels:
            raise ValueError(f"record {index} has an invalid true_type")
        mean_metrics = record.get("mean_metrics")
        if not isinstance(mean_metrics, dict) or set(mean_metrics) != expected_conditions:
            raise ValueError(f"record {index} must contain every condition")
        for condition, metrics in mean_metrics.items():
            _validate_metrics(metrics, name=f"record {index}/{condition}")
        expected_top1 = min(
            labels,
            key=lambda label: (mean_metrics[label]["total"], label),
        )
        if record.get("top1_type") != expected_top1:
            raise ValueError(f"record {index} top1_type is inconsistent")
        per_timestep = record.get("per_timestep")
        if not isinstance(per_timestep, dict) or not per_timestep:
            raise ValueError(f"record {index} per_timestep must be non-empty")
        for timestep, condition_metrics in per_timestep.items():
            if not isinstance(timestep, str) or not timestep.isdigit():
                raise ValueError(f"record {index} has an invalid timestep key")
            if (
                not isinstance(condition_metrics, dict)
                or set(condition_metrics) != expected_conditions
            ):
                raise ValueError(f"record {index}/{timestep} must contain every condition")
            for condition, metrics in condition_metrics.items():
                _validate_metrics(
                    metrics,
                    name=f"record {index}/{timestep}/{condition}",
                )
    expected_samples = next(iter(samples_by_bank.values()))
    if not expected_samples or any(
        sample_ids != expected_samples for sample_ids in samples_by_bank.values()
    ):
        raise ValueError("every noise bank must contain the same samples")
    for bank_index in samples_by_bank:
        bank_records = [record for record in records if record["bank_index"] == bank_index]
        if any(
            not any(record["true_type"] == label for record in bank_records)
            for label in labels
        ):
            raise ValueError("every bank must contain every true type")


def _paired_stats(
    rows: tuple[dict[str, Any], ...],
    true_type: str,
    wrong_type: str,
) -> dict[str, int | float]:
    deltas = tuple(
        row["mean_metrics"][wrong_type]["total"]
        - row["mean_metrics"][true_type]["total"]
        for row in rows
    )
    wins = sum(delta > 1e-12 for delta in deltas)
    losses = sum(delta < -1e-12 for delta in deltas)
    ties = len(deltas) - wins - losses
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / len(deltas),
        "median_delta": float(statistics.median(deltas)),
        "mean_delta": float(statistics.fmean(deltas)),
    }


def _summarize_rows(
    rows: tuple[dict[str, Any], ...], labels: tuple[str, ...]
) -> dict[str, Any]:
    per_true_type: dict[str, Any] = {}
    confusion_matrix: dict[str, dict[str, int]] = {}
    for true_type in labels:
        selected = tuple(row for row in rows if row["true_type"] == true_type)
        if not selected:
            raise ValueError(f"records are missing true type: {true_type}")
        confusion = {
            label: sum(row["top1_type"] == label for row in selected) for label in labels
        }
        medians = {
            condition: float(
                statistics.median(
                    row["mean_metrics"][condition]["total"] for row in selected
                )
            )
            for condition in (*labels, "normal")
        }
        correct = medians[true_type]
        paired = {
            wrong_type: _paired_stats(selected, true_type, wrong_type)
            for wrong_type in labels
            if wrong_type != true_type
        }
        per_timestep: dict[str, Any] = {}
        timestep_keys = tuple(sorted(selected[0]["per_timestep"], key=int))
        for timestep in timestep_keys:
            timestep_medians = {
                condition: float(
                    statistics.median(
                        row["per_timestep"][timestep][condition]["total"]
                        for row in selected
                    )
                )
                for condition in labels
            }
            per_timestep[timestep] = {
                "median_total": timestep_medians,
                "correct_is_lowest": true_type
                == min(labels, key=lambda label: (timestep_medians[label], label)),
            }
        per_true_type[true_type] = {
            "count": len(selected),
            "median_total": medians,
            "correct_normal_ratio": correct / max(medians["normal"], 1e-12),
            "correct_wrong_ratios": {
                wrong_type: correct / max(medians[wrong_type], 1e-12)
                for wrong_type in labels
                if wrong_type != true_type
            },
            "paired_against": paired,
            "top1_correct": confusion[true_type],
            "top1_accuracy": confusion[true_type] / len(selected),
            "top1_confusion": confusion,
            "per_timestep": per_timestep,
            "all_pairwise_median_advantages_positive": all(
                stats["median_delta"] > 0.0 for stats in paired.values()
            ),
        }
        confusion_matrix[true_type] = confusion
    return {
        "record_count": len(rows),
        "per_true_type": per_true_type,
        "confusion_matrix": confusion_matrix,
    }


def summarize_all_pairs(
    records: tuple[dict[str, Any], ...],
    labels: tuple[str, ...],
    bank_offsets: tuple[int, ...] = NOISE_BANK_OFFSETS,
) -> dict[str, Any]:
    _validate_records(records, labels, bank_offsets)
    banks = {
        str(bank_index): {
            "noise_seed_offset": bank_offsets[bank_index],
            **_summarize_rows(
                tuple(record for record in records if record["bank_index"] == bank_index),
                labels,
            ),
        }
        for bank_index in range(len(bank_offsets))
    }
    return {
        "noise_seed_offsets": list(bank_offsets),
        "banks": banks,
        "combined": _summarize_rows(records, labels),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_execution_records(
    records: tuple[dict[str, Any], ...],
    *,
    split: FewshotSplit,
    config: P2A0FewshotConfig,
) -> None:
    labels = tuple(config.defect_types)
    _validate_records(records, labels, NOISE_BANK_OFFSETS)
    expected_keys = {
        "bank_index",
        "noise_seed_offset",
        "noise_seeds",
        "sample_id",
        "true_type",
        "image_relpath",
        "mask_relpath",
        "context_seed",
        "context_sha256",
        "mean_metrics",
        "per_timestep",
        "prediction_sha256",
        "top1_type",
    }
    expected_count = len(split.evaluation) * len(NOISE_BANK_OFFSETS)
    if len(records) != expected_count:
        raise ValueError("all-pairs evaluation record count is invalid")
    for record_index, record in enumerate(records):
        if set(record) != expected_keys:
            raise ValueError(f"record {record_index} has an invalid exact schema")
        bank_index, sample_index = divmod(record_index, len(split.evaluation))
        sample = split.evaluation[sample_index]
        offset = NOISE_BANK_OFFSETS[bank_index]
        expected_identity = {
            "bank_index": bank_index,
            "noise_seed_offset": offset,
            "sample_id": sample.sample_id,
            "true_type": sample.defect_type,
            "image_relpath": sample.image_relpath,
            "mask_relpath": sample.mask_relpath,
        }
        if any(record[field] != value for field, value in expected_identity.items()):
            raise ValueError(f"record {record_index} does not match the frozen split")
        expected_context_seed = config.context_fill_seed + len(split.train) + sample_index
        if (
            type(record["context_seed"]) is not int
            or record["context_seed"] != expected_context_seed
        ):
            raise ValueError(f"record {record_index} has an invalid context_seed")
        if not _is_sha256(record["context_sha256"]):
            raise ValueError(f"record {record_index} has an invalid context_sha256")
        expected_noise_seeds = [
            config.noise_seed
            + 1_000_000
            + offset
            + sample_index * 10_000
            + timestep
            for timestep in config.eval_timesteps
        ]
        if record["noise_seeds"] != expected_noise_seeds:
            raise ValueError(f"record {record_index} has invalid noise seeds")
        if set(record["per_timestep"]) != {
            str(timestep) for timestep in config.eval_timesteps
        }:
            raise ValueError(f"record {record_index} has invalid eval timesteps")
        prediction_hashes = record["prediction_sha256"]
        if (
            not isinstance(prediction_hashes, dict)
            or set(prediction_hashes) != {*labels, "normal"}
            or any(not _is_sha256(value) for value in prediction_hashes.values())
        ):
            raise ValueError(f"record {record_index} has invalid prediction hashes")


@torch.no_grad()
def evaluate_all_pairs(
    *,
    model: Any,
    unet: Any,
    scheduler: Any,
    prepared: PreparedSample,
    prompt: Tensor,
    config: Any,
    sample_index: int,
    bank_index: int,
    noise_seed_offset: int,
    context_seed_offset: int,
) -> dict[str, Any]:
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    if type(bank_index) is not int or bank_index < 0:
        raise ValueError("bank_index must be a non-negative integer")
    if type(noise_seed_offset) is not int or noise_seed_offset < 0:
        raise ValueError("noise_seed_offset must be a non-negative integer")
    if type(context_seed_offset) is not int or context_seed_offset < 0:
        raise ValueError("context_seed_offset must be a non-negative integer")
    labels = tuple(config.defect_types)
    _validate_labels(labels)
    conditions = (*labels, "normal")
    accumulated = {
        condition: {metric: [] for metric in METRIC_KEYS} for condition in conditions
    }
    hashes = {condition: [] for condition in conditions}
    per_timestep: dict[str, dict[str, dict[str, float]]] = {}
    noise_seeds: list[int] = []
    for timestep in config.eval_timesteps:
        seed = (
            config.noise_seed
            + 1_000_000
            + noise_seed_offset
            + sample_index * 10_000
            + timestep
        )
        noise_seeds.append(seed)
        timesteps = torch.tensor(
            [timestep], device=prepared.target.device, dtype=torch.long
        )
        generator = torch.Generator(device=prepared.target.device).manual_seed(seed)
        noise = torch.randn(
            prepared.target.shape,
            generator=generator,
            device=prepared.target.device,
            dtype=prepared.target.dtype,
        )
        noisy = scheduler.add_noise(prepared.target, noise, timesteps)
        predictions = {
            label: model(
                noisy,
                prepared.context,
                prepared.masks.binary,
                timesteps,
                prompt,
                label,
            )
            for label in labels
        }
        predictions["normal"] = unet(
            noisy,
            timesteps,
            encoder_hidden_states=prompt,
        ).sample
        timestep_metrics: dict[str, dict[str, float]] = {}
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
            values = {
                metric: float(getattr(losses, metric).item()) for metric in METRIC_KEYS
            }
            _validate_metrics(values, name=f"{condition}/{timestep}")
            timestep_metrics[condition] = values
            for metric, value in values.items():
                accumulated[condition][metric].append(value)
            hashes[condition].append(_tensor_sha256(prediction))
        per_timestep[str(timestep)] = timestep_metrics
    mean_metrics = {
        condition: {
            metric: float(statistics.fmean(values))
            for metric, values in by_metric.items()
        }
        for condition, by_metric in accumulated.items()
    }
    aggregate_hashes = {
        condition: hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()
        for condition, values in hashes.items()
    }
    top1 = min(labels, key=lambda label: (mean_metrics[label]["total"], label))
    return {
        "bank_index": bank_index,
        "noise_seed_offset": noise_seed_offset,
        "noise_seeds": noise_seeds,
        "sample_id": prepared.sample.sample_id,
        "true_type": prepared.sample.defect_type,
        "image_relpath": prepared.sample.image_relpath,
        "mask_relpath": prepared.sample.mask_relpath,
        "context_seed": config.context_fill_seed + context_seed_offset + sample_index,
        "context_sha256": _tensor_sha256(prepared.context),
        "mean_metrics": mean_metrics,
        "per_timestep": per_timestep,
        "prediction_sha256": aggregate_hashes,
        "top1_type": top1,
    }


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _regular_frozen_file(path: Path, *, root: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"frozen source-run symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"frozen source-run file escaped its root: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"frozen source-run path must be a regular file: {path}")
    return resolved


def validate_frozen_source(
    source_run: Path,
    *,
    runs_root: Path,
    expected_relative_path: str | Path,
    expected_hashes: dict[str, str],
    expected_checkpoint_head: str,
    expected_config: dict[str, Any],
    expected_vocabulary_sha256: str,
) -> dict[str, Any]:
    root = runs_root.resolve(strict=True)
    if source_run.is_symlink():
        raise ValueError("frozen source-run symlink is forbidden")
    source = source_run.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("frozen source-run must be a directory")
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise ValueError("frozen source-run escaped runs_root") from error
    if relative != Path(expected_relative_path):
        raise ValueError("frozen source-run path does not match the certified origin")
    required = set(SOURCE_HASHES)
    if set(expected_hashes) != required:
        raise ValueError("frozen source-run hash manifest has an invalid file set")
    snapshot: dict[str, dict[str, int | str]] = {}
    actual_hashes: dict[str, str] = {}
    for relative_path in sorted(expected_hashes):
        candidate = _regular_frozen_file(source / relative_path, root=source)
        snapshot[relative_path] = _snapshot(candidate)
        actual_hashes[relative_path] = sha256_file(candidate)
        if actual_hashes[relative_path] != expected_hashes[relative_path]:
            raise RuntimeError(f"frozen source-run hash changed: {relative_path}")
    metadata = _strict_json_object(source / "checkpoint/metadata.json")
    if metadata.get("schema_version") != "p2_a0_fewshot_checkpoint_v1":
        raise ValueError("frozen checkpoint schema is invalid")
    if metadata.get("git_head") != expected_checkpoint_head:
        raise ValueError("frozen checkpoint origin HEAD is invalid")
    canonical_expected_config = json.loads(
        json.dumps(expected_config, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    if metadata.get("config") != canonical_expected_config:
        raise ValueError("frozen checkpoint config is invalid")
    if metadata.get("vocabulary_sha256") != expected_vocabulary_sha256:
        raise ValueError("frozen checkpoint vocabulary is invalid")
    files = metadata.get("files")
    expected_checkpoint_files = {
        "adapter.safetensors": source / "checkpoint/adapter.safetensors",
        "conditioner.safetensors": source / "checkpoint/conditioner.safetensors",
    }
    if not isinstance(files, dict) or set(files) != set(expected_checkpoint_files):
        raise ValueError("frozen checkpoint file manifest is invalid")
    for name, path in expected_checkpoint_files.items():
        record = files[name]
        if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
            raise ValueError(f"frozen checkpoint metadata is invalid: {name}")
        if record != {"sha256": sha256_file(path), "size": path.stat().st_size}:
            raise ValueError(f"frozen checkpoint metadata does not match: {name}")
    report = _strict_json_object(source / "report.json")
    provenance = _strict_json_object(source / "provenance.json")
    if report.get("status") != "complete" or report.get("gate_passed") is not True:
        raise ValueError("frozen source-run report is not a successful run")
    if report.get("checkpoint") != metadata:
        raise ValueError("frozen source-run report checkpoint is inconsistent")
    if provenance.get("git_dirty") is not False:
        raise ValueError("frozen source-run provenance is dirty")
    if any(
        provenance.get(field) != expected_checkpoint_head
        for field in ("git_commit", "git_head_start", "git_head_end")
    ):
        raise ValueError("frozen source-run provenance origin is inconsistent")
    return {
        "metadata": metadata,
        "hashes": actual_hashes,
        "snapshot": snapshot,
    }


def validate_all_pairs_preflight(context: AllPairsContext) -> AllPairsPreflight:
    base = validate_fewshot_preflight(
        FewshotContext(
            paths=context.paths,
            experiment=context.experiment,
            manifest=context.manifest,
            lock=context.lock,
            output=context.output,
            config_paths=context.config_paths,
            cwd=context.cwd,
        )
    )
    frozen = validate_frozen_source(
        context.source_run,
        runs_root=context.paths.runs_root,
        expected_relative_path=SOURCE_RUN_RELATIVE,
        expected_hashes=SOURCE_HASHES,
        expected_checkpoint_head=SOURCE_CHECKPOINT_HEAD,
        expected_config=asdict(context.experiment),
        expected_vocabulary_sha256=SOURCE_VOCABULARY_SHA256,
    )
    return AllPairsPreflight(
        context=context,
        split=base.split,
        git_head=base.git_head,
        input_hashes={
            **base.input_hashes,
            **{f"source-run:{key}": value for key, value in frozen["hashes"].items()},
        },
        source_snapshot=base.source_snapshot,
        model_snapshot=base.model_snapshot,
        reference_snapshot=frozen["snapshot"],
        reference_hashes=frozen["hashes"],
        checkpoint_metadata=frozen["metadata"],
    )


def _load_checkpoint(
    source_run: Path,
    *,
    config: P2A0FewshotConfig,
    device: torch.device,
) -> tuple[DefectVocabulary, DefectConditioner, MaskedInfillAdapter, dict[str, Any]]:
    metadata = _strict_json_object(source_run / "checkpoint/metadata.json")
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    conditioner = DefectConditioner(vocabulary, 128).to(device=device)
    adapter = MaskedInfillAdapter(config).to(device=device)
    conditioner.load_state_dict(
        load_file(source_run / "checkpoint/conditioner.safetensors", device="cpu"),
        strict=True,
    )
    adapter.load_state_dict(
        load_file(source_run / "checkpoint/adapter.safetensors", device="cpu"),
        strict=True,
    )
    conditioner.eval()
    adapter.eval()
    return vocabulary, conditioner, adapter, metadata


def _execute_all_pairs(preflight: AllPairsPreflight) -> dict[str, Any]:
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
    prepared = _prepare_samples(
        vae=components.vae,
        samples=preflight.split.evaluation,
        device=device,
        config=config,
        seed_offset=len(preflight.split.train),
    )
    _, conditioner, adapter, metadata = _load_checkpoint(
        context.source_run,
        config=config,
        device=device,
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for bank_index, offset in enumerate(NOISE_BANK_OFFSETS):
            for sample_index, sample in enumerate(prepared):
                records.append(
                    evaluate_all_pairs(
                        model=model,
                        unet=components.unet,
                        scheduler=components.scheduler,
                        prepared=sample,
                        prompt=prompt,
                        config=config,
                        sample_index=sample_index,
                        bank_index=bank_index,
                        noise_seed_offset=offset,
                        context_seed_offset=len(preflight.split.train),
                    )
                )
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated()
    _assert_parameter_versions_unchanged(components.unet, unet_versions)
    if any(parameter.grad is not None for parameter in components.unet.parameters()):
        raise RuntimeError("frozen U-Net accumulated gradients")
    if any(parameter.grad is not None for parameter in model.trainable_parameters()):
        raise RuntimeError("read-only trainables accumulated gradients")
    return {
        "records": tuple(records),
        "peak_vram_bytes": peak,
        "checkpoint_metadata": metadata,
    }


def _verify_all_pairs_integrity(
    preflight: AllPairsPreflight, identity: tuple[int, int]
) -> str:
    head, dirty = _git_state(preflight.context.cwd)
    if dirty or head != preflight.git_head:
        raise RuntimeError("Git changed during all-pairs diagnostic")
    if _source_snapshot(preflight.split) != preflight.source_snapshot:
        raise RuntimeError("evaluation source changed during all-pairs diagnostic")
    if _model_snapshot(preflight.context.paths.model_root) != preflight.model_snapshot:
        raise RuntimeError("model files changed during all-pairs diagnostic")
    current_reference = {
        relative: _snapshot(preflight.context.source_run / relative)
        for relative in sorted(preflight.reference_hashes)
    }
    if current_reference != preflight.reference_snapshot:
        raise RuntimeError("frozen source-run changed during all-pairs diagnostic")
    current_input_hashes = {
        "manifest": sha256_file(preflight.context.manifest.resolve(strict=True)),
        "lock": sha256_file(preflight.context.lock.resolve(strict=True)),
        **{
            f"config:{name}": sha256_file(path.resolve(strict=True))
            for name, path in sorted(preflight.context.config_paths.items())
        },
        **{
            f"source-run:{relative}": sha256_file(
                (preflight.context.source_run / relative).resolve(strict=True)
            )
            for relative in sorted(preflight.reference_hashes)
        },
    }
    if current_input_hashes != preflight.input_hashes:
        raise RuntimeError("all-pairs input hash changed during diagnostic")
    _verify_output_identity(preflight, identity)  # type: ignore[arg-type]
    return head


def _remove_partial_artifacts(output: Path) -> None:
    for name in ("metrics.jsonl", "provenance.json", "report.json"):
        candidate = output / name
        if candidate.is_symlink() or candidate.is_file():
            candidate.unlink(missing_ok=True)


def run_all_pairs(context: AllPairsContext) -> int:
    try:
        preflight = validate_all_pairs_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_all_pairs(preflight)
        records = tuple(result["records"])
        _validate_execution_records(
            records,
            split=preflight.split,
            config=context.experiment,
        )
        summary = summarize_all_pairs(records, tuple(context.experiment.defect_types))
        head_end = _verify_all_pairs_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "metrics.jsonl", records)
        _verify_all_pairs_integrity(preflight, identity)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "reason": None,
            "diagnostic_only": True,
            "training_performed": False,
            "checkpoint_written": False,
            "noise_seed_offsets": list(NOISE_BANK_OFFSETS),
            "evaluation_records": len(records),
            "summary": summary,
            "peak_vram_bytes": result["peak_vram_bytes"],
            "checkpoint_origin": preflight.checkpoint_metadata,
            "input_hashes": preflight.input_hashes,
        }
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": SCHEMA_VERSION,
                "git_head_start": preflight.git_head,
                "git_head_end": head_end,
                "git_dirty": False,
                "diagnostic_only": True,
                "training_performed": False,
                "checkpoint_written": False,
                "noise_seed_offsets": list(NOISE_BANK_OFFSETS),
                "input_hashes": preflight.input_hashes,
                "source_snapshot_before": preflight.source_snapshot,
                "source_snapshot_after": preflight.source_snapshot,
                "model_snapshot_before": preflight.model_snapshot,
                "model_snapshot_after": preflight.model_snapshot,
                "reference_snapshot_before": preflight.reference_snapshot,
                "reference_snapshot_after": preflight.reference_snapshot,
            }
        )
        write_json_atomic(context.output / "provenance.json", provenance)
        _verify_all_pairs_integrity(preflight, identity)
        write_json_atomic(context.output / "report.json", report)
        _verify_all_pairs_integrity(preflight, identity)
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "reason": "runtime_or_integrity_failure",
            "diagnostic_only": True,
        }
        try:
            _verify_output_identity(preflight, identity)  # type: ignore[arg-type]
        except (OSError, RuntimeError):
            pass
        else:
            try:
                _remove_partial_artifacts(context.output)
                write_json_atomic(context.output / "report.json", failed)
            except OSError:
                pass
        print(json.dumps(failed, sort_keys=True))
        return 1


def build_all_pairs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only two-bank all-pairs diagnostic for Type-Routed Heads"
    )
    parser.add_argument("--paths", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_all_pairs_parser().parse_args(argv)
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        return run_all_pairs(
            AllPairsContext(
                paths=load_config(paths_path),
                experiment=load_p2_a0_fewshot_config(experiment_path),
                manifest=Path(args.manifest),
                lock=Path(args.lock),
                source_run=Path(args.source_run),
                output=Path(args.output),
                config_paths={"experiment": experiment_path, "paths": paths_path},
                cwd=Path.cwd(),
            )
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1

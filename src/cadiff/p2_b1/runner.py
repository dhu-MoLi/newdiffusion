from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from cadiff.config import ProjectConfig, load_config
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.dataset import load_training_image, load_training_index
from cadiff.p2_a0.all_pairs import _strict_json_object
from cadiff.p2_a0.fewshot import (
    FewshotPreflight,
    _assert_parameter_versions_unchanged,
    _capture_parameter_versions,
    _output_identity,
    _snapshot,
    _tensor_sha256,
    _tensor_to_pil,
    _verify_integrity,
    _verify_output_identity,
    _write_jsonl_atomic,
)
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_a1.runner import (
    P2A1Context,
    _load_p2_a1_checkpoint,
    validate_p2_a1_preflight,
)
from cadiff.p2_b0.generation import compute_mask_metrics, denoise_masked_infill
from cadiff.p2_b0.runner import (
    _checkpoint_snapshot,
    _cleanup_partial,
    _contained_artifact,
    _real_child,
    _save_sample,
    _validated_source_run,
)
from cadiff.p2_b1.config import P2B1Config, load_p2_b1_config
from cadiff.p2_b1.generation import (
    HeldoutGenerationSlot,
    build_heldout_generation_schedule,
    select_heldout_probe_inputs,
)
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic


@dataclass(frozen=True)
class P2B1Context:
    paths: ProjectConfig | Any
    experiment: P2B1Config
    source_experiment: P2A1Config | Any
    manifest: Path
    lock: Path
    source_run: Path
    baseline_run: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class P2B1Preflight:
    context: P2B1Context
    base: FewshotPreflight
    normals: tuple[Any, ...]
    masks: tuple[Any, ...]
    normal_ids: tuple[str, ...]
    mask_ids: tuple[str, ...]
    source_mask_types: tuple[str, ...]
    git_head: str
    input_hashes: dict[str, str]
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]
    checkpoint_snapshot: dict[str, dict[str, int | str]]
    baseline_snapshot: dict[str, dict[str, int | str]]
    baseline_summary: dict[str, float]

_RECORD_KEYS = {
    "schema_version",
    "slot_id",
    "normal_id",
    "mask_id",
    "source_mask_type",
    "seed",
    "defect_type",
    "source_normal_sha256",
    "source_mask_sha256",
    "image_relpath",
    "mask_relpath",
    "image_sha256",
    "mask_sha256",
    "generated_tensor_sha256",
    "inside_l1",
    "outside_l1",
    "outside_to_inside_ratio",
    "change_concentration",
    "composite_outside_l1",
    "finite",
}
_PAIRWISE_KEYS = {
    "schema_version",
    "context_id",
    "normal_id",
    "mask_id",
    "source_mask_type",
    "seed",
    "left_type",
    "right_type",
    "masked_l1",
    "finite",
}
_SHA_FIELDS = (
    "source_normal_sha256",
    "source_mask_sha256",
    "image_sha256",
    "mask_sha256",
    "generated_tensor_sha256",
)
_METRIC_FIELDS = (
    "inside_l1",
    "outside_l1",
    "outside_to_inside_ratio",
    "change_concentration",
    "composite_outside_l1",
)


def _sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    return value


def _validate_generation_records(
    records: tuple[dict[str, Any], ...],
    pairwise: tuple[dict[str, Any], ...],
    schedule: Sequence[HeldoutGenerationSlot],
    config: P2B1Config,
) -> None:
    if len(records) != len(schedule):
        raise ValueError("generation record count does not match the schedule")
    seen_paths: set[str] = set()
    source_hashes: dict[tuple[str, str, str], tuple[str, str]] = {}
    for index, (record, slot) in enumerate(zip(records, schedule, strict=True)):
        if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
            raise ValueError(f"generation record {index} has an invalid exact schema")
        expected = {
            "slot_id": slot.slot_id,
            "normal_id": slot.normal_id,
            "mask_id": slot.mask_id,
            "source_mask_type": slot.source_mask_type,
            "seed": slot.seed,
            "defect_type": slot.defect_type,
        }
        if any(record[field] != value for field, value in expected.items()):
            raise ValueError(f"generation record {index} does not match the schedule")
        if type(record["seed"]) is not int:
            raise ValueError(f"generation record {index} seed must be an integer")
        if record["schema_version"] != "p2_b1_generation_record_v1":
            raise ValueError(f"generation record {index} schema is incompatible")
        for field in ("image_relpath", "mask_relpath"):
            value = record[field]
            if not isinstance(value, str) or not value.startswith("samples/"):
                raise ValueError(f"generation record {index} has invalid {field}")
            if value in seen_paths:
                raise ValueError(f"generation record {index} duplicates an artifact path")
            seen_paths.add(value)
        for field in _SHA_FIELDS:
            _sha256(record[field], f"generation record {index} {field}")
        source_key = (slot.normal_id, slot.mask_id, slot.source_mask_type)
        hashes = (record["source_normal_sha256"], record["source_mask_sha256"])
        if source_key in source_hashes and source_hashes[source_key] != hashes:
            raise ValueError(f"generation record {index} source hashes are inconsistent")
        source_hashes[source_key] = hashes
        for field in _METRIC_FIELDS:
            _finite_float(record[field], f"generation record {index} {field}")
        if record["finite"] is not True:
            raise ValueError(f"generation record {index} is not finite")
        if record["change_concentration"] > 1.0:
            raise ValueError(f"generation record {index} change_concentration exceeds one")
        if record["composite_outside_l1"] > 1e-7:
            raise ValueError(f"generation record {index} composite outside changed")

    expected_count = (
        config.normal_reference_count
        * len(config.sample_seeds)
        * math.comb(len(config.defect_types), 2)
    )
    if len(pairwise) != expected_count:
        raise ValueError("pairwise record count is invalid")
    schedule_contexts = {
        (slot.normal_id, slot.mask_id, slot.source_mask_type, slot.seed) for slot in schedule
    }
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for index, record in enumerate(pairwise):
        if not isinstance(record, dict) or set(record) != _PAIRWISE_KEYS:
            raise ValueError(f"pairwise record {index} has an invalid exact schema")
        if record["schema_version"] != "p2_b1_pairwise_record_v1":
            raise ValueError(f"pairwise record {index} schema is incompatible")
        if type(record["seed"]) is not int or record["seed"] not in config.sample_seeds:
            raise ValueError(f"pairwise record {index} seed is invalid")
        if record["source_mask_type"] not in config.defect_types:
            raise ValueError(f"pairwise record {index} source mask type is invalid")
        if record["finite"] is not True:
            raise ValueError(f"pairwise record {index} is not finite")
        _finite_float(record["masked_l1"], f"pairwise record {index} masked_l1")
        key = (
            record["normal_id"],
            record["mask_id"],
            record["source_mask_type"],
            record["seed"],
        )
        if key not in schedule_contexts:
            raise ValueError(
                f"pairwise record {index} source mask type or context is invalid"
            )
        expected_context_id = hashlib.sha256(
            (
                f"{record['normal_id']}\n{record['mask_id']}\n"
                f"{record['source_mask_type']}\n{record['seed']}"
            ).encode()
        ).hexdigest()
        if record["context_id"] != expected_context_id:
            raise ValueError(f"pairwise record {index} context_id is invalid")
        groups.setdefault(key, []).append(record)
    if set(groups) != schedule_contexts:
        raise ValueError("pairwise source mask type or context does not match the schedule")
    expected_pairs = set(itertools.combinations(config.defect_types, 2))
    for key, values in groups.items():
        actual_pairs = {(value["left_type"], value["right_type"]) for value in values}
        if actual_pairs != expected_pairs or len(values) != len(expected_pairs):
            raise ValueError(f"pairwise type pairing is invalid for {key}")
        if len({value["context_id"] for value in values}) != 1:
            raise ValueError(f"pairwise context_id is inconsistent for {key}")


def _metric_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "record_count": len(records),
        **{
            f"{field}_median": float(statistics.median(record[field] for record in records))
            for field in _METRIC_FIELDS
        },
    }


def summarize_generation(
    records: tuple[dict[str, Any], ...],
    pairwise: tuple[dict[str, Any], ...],
    config: P2B1Config,
) -> dict[str, Any]:
    per_route = {
        defect_type: _metric_summary(
            [record for record in records if record["defect_type"] == defect_type]
        )
        for defect_type in config.defect_types
    }
    per_source = {
        defect_type: _metric_summary(
            [record for record in records if record["source_mask_type"] == defect_type]
        )
        for defect_type in config.defect_types
    }
    per_pair: dict[str, dict[str, float | int]] = {}
    seed_medians: dict[tuple[str, str], list[float]] = {}
    for left, right in itertools.combinations(config.defect_types, 2):
        values = [
            record
            for record in pairwise
            if record["left_type"] == left and record["right_type"] == right
        ]
        per_pair[f"{left}|{right}"] = {
            "record_count": len(values),
            "masked_l1_median": float(statistics.median(v["masked_l1"] for v in values)),
        }
        seed_medians[(left, right)] = [
            float(
                statistics.median(
                    value["masked_l1"] for value in values if value["seed"] == seed
                )
            )
            for seed in config.sample_seeds
        ]
    inside = float(statistics.median(record["inside_l1"] for record in records))
    outside = float(statistics.median(record["outside_l1"] for record in records))
    separation = float(statistics.median(record["masked_l1"] for record in pairwise))
    spread = max(max(values) - min(values) for values in seed_medians.values())
    return {
        "diagnostic_only": True,
        "record_count": len(records),
        "pairwise_record_count": len(pairwise),
        "inside_l1_median": inside,
        "outside_l1_median": outside,
        "inside_to_outside_ratio": inside / max(outside, 1e-12),
        "inside_change_exceeds_outside": inside > outside,
        "pairwise_masked_l1_median": separation,
        "route_noncollapse_observed": separation > 1e-6,
        "cross_seed_pairwise_median_spread": float(spread),
        "per_route": per_route,
        "per_source_mask_type": per_source,
        "per_pair": per_pair,
    }


def compare_with_baseline(
    summary: dict[str, Any], baseline_summary: dict[str, Any]
) -> dict[str, float]:
    pairs = (
        ("inside_l1_median", "inside_l1_ratio_to_p2_b0"),
        ("outside_l1_median", "outside_l1_ratio_to_p2_b0"),
        ("pairwise_masked_l1_median", "pairwise_l1_ratio_to_p2_b0"),
    )
    result: dict[str, float] = {}
    for source, target in pairs:
        current = _finite_float(summary.get(source), f"P2-B1 summary {source}")
        baseline = _finite_float(
            baseline_summary.get(source), f"P2-B0 baseline summary {source}"
        )
        if baseline <= 0.0:
            raise ValueError(f"P2-B0 baseline summary {source} must be positive")
        result[target] = current / baseline
    return result


def _validated_child_run(run: Path, runs_root: Path, name: str) -> Path:
    lexical_runs = Path(os.path.abspath(runs_root))
    lexical_run = Path(os.path.abspath(run))
    try:
        relative = lexical_run.relative_to(lexical_runs)
    except ValueError as error:
        raise ValueError(f"{name} must be inside runs_root") from error
    if not relative.parts:
        raise ValueError(f"{name} must be strictly inside runs_root")
    current = lexical_runs
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} path must not contain symlinks: {current}")
    resolved_runs = lexical_runs.resolve(strict=True)
    resolved_run = lexical_run.resolve(strict=True)
    if not resolved_run.is_dir() or not resolved_run.is_relative_to(resolved_runs):
        raise ValueError(f"{name} must resolve inside runs_root")
    return resolved_run


def _validated_baseline_run(
    baseline_run: Path, runs_root: Path
) -> tuple[Path, frozenset[str], frozenset[str]]:
    resolved = _validated_child_run(baseline_run, runs_root, "baseline_run")
    report_path = _real_child(resolved, "report.json", directory=False)
    report = _strict_json_object(report_path)
    if (
        report.get("schema_version") != "p2_b0_generation_report_v1"
        or report.get("status") != "complete"
        or report.get("reason") is not None
        or report.get("training_performed") is not False
        or report.get("checkpoint_written") is not False
        or report.get("evaluation_mode") != "diagnostic_only"
    ):
        raise ValueError("baseline_run is not the accepted completed P2-B0 diagnostic")
    results: list[frozenset[str]] = []
    for field in ("normal_ids", "mask_ids"):
        value = report.get(field)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError(f"baseline report {field} must contain unique strings")
        results.append(frozenset(value))
    return resolved, results[0], results[1]


def validate_p2_b1_preflight(context: P2B1Context) -> P2B1Preflight:
    if not isinstance(context, P2B1Context):
        raise ValueError("context must be a P2B1Context")
    if not isinstance(context.experiment, P2B1Config):
        raise ValueError("experiment must be a P2B1Config")
    if not isinstance(context.source_experiment, P2A1Config):
        raise ValueError("source_experiment must be a P2A1Config")
    if (
        context.experiment.category != context.source_experiment.category
        or context.experiment.defect_types != context.source_experiment.defect_types
        or context.experiment.image_size != context.source_experiment.image_size
        or context.experiment.dtype != context.source_experiment.dtype
        or context.experiment.normal_prompt != context.source_experiment.normal_prompt
    ):
        raise ValueError("generation and source experiment contracts must match")
    base = validate_p2_a1_preflight(
        P2A1Context(
            paths=context.paths,
            experiment=context.source_experiment,
            manifest=context.manifest,
            lock=context.lock,
            output=context.output,
            config_paths=context.config_paths,
            cwd=context.cwd,
        )
    )
    source_run, checkpoint = _validated_source_run(
        context.source_run, context.paths.runs_root
    )
    checkpoint_snapshot = _checkpoint_snapshot(checkpoint)
    baseline_run, excluded_normals, excluded_masks = _validated_baseline_run(
        context.baseline_run, context.paths.runs_root
    )
    baseline_report = _real_child(baseline_run, "report.json", directory=False)
    baseline_report_data = _strict_json_object(baseline_report)
    raw_baseline_summary = baseline_report_data.get("summary")
    if not isinstance(raw_baseline_summary, dict):
        raise ValueError("baseline report summary must be an object")
    baseline_summary = {
        field: _finite_float(raw_baseline_summary.get(field), f"baseline summary {field}")
        for field in (
            "inside_l1_median",
            "outside_l1_median",
            "pairwise_masked_l1_median",
        )
    }
    baseline_snapshot = {"report.json": _snapshot(baseline_report)}
    index = load_training_index(
        context.manifest,
        context.paths.dataset_root,
        context.source_experiment,  # type: ignore[arg-type]
    )
    normals, masks = select_heldout_probe_inputs(
        normals=index.normals,
        heldout_masks=base.split.evaluation,
        defect_types=context.experiment.defect_types,
        masks_per_type=context.experiment.masks_per_type,
        normal_count=context.experiment.normal_reference_count,
        excluded_normal_ids=excluded_normals,
        excluded_mask_ids=excluded_masks,
    )
    source_snapshot = {
        **base.source_snapshot,
        **{str(item.image_path): _snapshot(item.image_path) for item in normals},
    }
    input_hashes = dict(base.input_hashes)
    input_hashes["source_report"] = sha256_file(source_run / "report.json")
    input_hashes["baseline_report"] = sha256_file(baseline_report)
    for name, value in checkpoint_snapshot.items():
        input_hashes[f"source_checkpoint/{name}"] = str(value["sha256"])
    return P2B1Preflight(
        context=context,
        base=base,
        normals=normals,
        masks=masks,
        normal_ids=tuple(item.sample_id for item in normals),
        mask_ids=tuple(item.sample_id for item in masks),
        source_mask_types=tuple(item.defect_type for item in masks),
        git_head=base.git_head,
        input_hashes=input_hashes,
        source_snapshot=source_snapshot,
        model_snapshot=base.model_snapshot,
        checkpoint_snapshot=checkpoint_snapshot,
        baseline_snapshot=baseline_snapshot,
        baseline_summary=baseline_summary,
    )


def _verify_p2_b1_integrity(preflight: P2B1Preflight, identity: tuple[int, int]) -> str:
    head = _verify_integrity(preflight.base, identity)
    for path_text, before in preflight.source_snapshot.items():
        path = Path(path_text)
        if path.is_absolute() and _snapshot(path) != before:
            raise RuntimeError(f"source snapshot changed during generation: {path_text}")
    source_run, checkpoint = _validated_source_run(
        preflight.context.source_run, preflight.context.paths.runs_root
    )
    if sha256_file(source_run / "report.json") != preflight.input_hashes["source_report"]:
        raise RuntimeError("source report changed during generation")
    if _checkpoint_snapshot(checkpoint) != preflight.checkpoint_snapshot:
        raise RuntimeError("source checkpoint changed during generation")
    baseline_run, _, _ = _validated_baseline_run(
        preflight.context.baseline_run, preflight.context.paths.runs_root
    )
    baseline_report = _real_child(baseline_run, "report.json", directory=False)
    if (
        {"report.json": _snapshot(baseline_report)} != preflight.baseline_snapshot
        or sha256_file(baseline_report) != preflight.input_hashes["baseline_report"]
    ):
        raise RuntimeError("P2-B0 baseline report changed during generation")
    return head


def _pairwise_records(
    *,
    normal_id: str,
    mask_id: str,
    source_mask_type: str,
    seed: int,
    generated: dict[str, torch.Tensor],
    source_mask: torch.Tensor,
    defect_types: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    context_id = hashlib.sha256(
        f"{normal_id}\n{mask_id}\n{source_mask_type}\n{seed}".encode()
    ).hexdigest()
    mask = source_mask.to(dtype=torch.float32)
    result: list[dict[str, Any]] = []
    for left, right in itertools.combinations(defect_types, 2):
        difference = (
            (generated[left].float() - generated[right].float())
            .abs()
            .mean(dim=1, keepdim=True)
        )
        masked_l1 = float(((difference * mask).sum() / mask.sum()).item())
        result.append(
            {
                "schema_version": "p2_b1_pairwise_record_v1",
                "context_id": context_id,
                "normal_id": normal_id,
                "mask_id": mask_id,
                "source_mask_type": source_mask_type,
                "seed": seed,
                "left_type": left,
                "right_type": right,
                "masked_l1": masked_l1,
                "finite": math.isfinite(masked_l1),
            }
        )
    return tuple(result)


def _prepare_normal_tensors(
    pixels: torch.Tensor,
    device: torch.device,
    *,
    vae_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(pixels, torch.Tensor)
        or pixels.ndim != 3
        or pixels.shape[0] != 3
        or not pixels.is_floating_point()
        or not torch.isfinite(pixels).all().item()
    ):
        raise ValueError("normal pixels must be a finite floating CHW RGB tensor")
    normal_image = (
        ((pixels.float() + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .unsqueeze(0)
        .to(device=device)
    )
    if not torch.empty((), dtype=vae_dtype).is_floating_point():
        raise ValueError("VAE dtype must be floating")
    vae_pixels = pixels.unsqueeze(0).to(device=device, dtype=vae_dtype)
    return vae_pixels, normal_image


@torch.no_grad()
def _execute_p2_b1(preflight: P2B1Preflight, identity: tuple[int, int]) -> dict[str, Any]:
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
    _, checkpoint = _validated_source_run(context.source_run, context.paths.runs_root)
    _, conditioner, adapter, _ = _load_p2_a1_checkpoint(
        checkpoint, config=context.source_experiment, device=device
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).eval()
    schedule = build_heldout_generation_schedule(
        normal_ids=preflight.normal_ids,
        mask_ids=preflight.mask_ids,
        source_mask_types=preflight.source_mask_types,
        defect_types=config.defect_types,
        seeds=config.sample_seeds,
    )
    slot_by_key = {
        (slot.normal_id, slot.mask_id, slot.source_mask_type, slot.seed, slot.defect_type): slot
        for slot in schedule
    }
    records: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    for normal_item, mask_item in zip(preflight.normals, preflight.masks, strict=True):
        loaded = load_training_image(normal_item, config.image_size)
        normal_pixels, normal_image = _prepare_normal_tensors(loaded.pixels, device)
        normal_latent = encode_mode(components.vae, normal_pixels).detach()
        source_mask = mask_item.mask.unsqueeze(0).to(device=device, dtype=torch.float32)
        masks = build_latent_masks(source_mask, normal_latent.shape[2:])
        for seed in config.sample_seeds:
            context_latent = build_masked_context(
                normal_latent,
                masks.binary,
                torch.Generator(device=device).manual_seed(seed + 10_000_000),
            )
            noise = torch.randn(
                normal_latent.shape,
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
                dtype=normal_latent.dtype,
            )
            generated_by_type: dict[str, torch.Tensor] = {}
            for defect_type in config.defect_types:
                scheduler = DDIMScheduler.from_config(components.scheduler.config)
                generated_latent = denoise_masked_infill(
                    model=model,
                    scheduler=scheduler,
                    normal_latent=normal_latent,
                    context=context_latent,
                    binary_mask=masks.binary,
                    prompt_embeds=prompt,
                    defect_type=defect_type,
                    noise=noise,
                    steps=config.denoise_steps,
                )
                raw_image = decode_latent(components.vae, generated_latent).float()
                raw_metrics = compute_mask_metrics(raw_image, normal_image, source_mask)
                composite = raw_image * source_mask + normal_image * (1.0 - source_mask)
                composite_metrics = compute_mask_metrics(composite, normal_image, source_mask)
                slot = slot_by_key[
                    (
                        normal_item.sample_id,
                        mask_item.sample_id,
                        mask_item.defect_type,
                        seed,
                        defect_type,
                    )
                ]
                image_relpath, mask_relpath, image_sha, mask_sha = _save_sample(
                    output=context.output,
                    slot_id=slot.slot_id,
                    image=composite,
                    mask=source_mask,
                )
                generated_by_type[defect_type] = composite.detach().clone()
                records.append(
                    {
                        "schema_version": "p2_b1_generation_record_v1",
                        **asdict(slot),
                        "source_normal_sha256": normal_item.image_sha256,
                        "source_mask_sha256": mask_item.mask_sha256,
                        "image_relpath": image_relpath,
                        "mask_relpath": mask_relpath,
                        "image_sha256": image_sha,
                        "mask_sha256": mask_sha,
                        "generated_tensor_sha256": _tensor_sha256(composite),
                        **asdict(raw_metrics),
                        "composite_outside_l1": composite_metrics.outside_l1,
                        "finite": True,
                    }
                )
            pairwise.extend(
                _pairwise_records(
                    normal_id=normal_item.sample_id,
                    mask_id=mask_item.sample_id,
                    source_mask_type=mask_item.defect_type,
                    seed=seed,
                    generated=generated_by_type,
                    source_mask=source_mask,
                    defect_types=config.defect_types,
                )
            )
            print(
                json.dumps(
                    {
                        "stage": "heldout_generation",
                        "records": len(records),
                        "total": len(schedule),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _assert_parameter_versions_unchanged(components.unet, unet_versions)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only heldout generation accumulated gradients")
    _verify_p2_b1_integrity(preflight, identity)
    torch.cuda.synchronize(device)
    return {
        "records": tuple(records),
        "pairwise": tuple(pairwise),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }


def _validate_sample_artifacts(
    preflight: P2B1Preflight | Any, records: tuple[dict[str, Any], ...]
) -> None:
    output = preflight.context.output
    image_size = preflight.context.experiment.image_size
    samples = _real_child(output, "samples", directory=True)
    normals = {item.sample_id: item for item in preflight.normals}
    masks = {item.sample_id: item for item in preflight.masks}
    expected_paths: set[Path] = set()
    normal_arrays: dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    mask_arrays: dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    for index, record in enumerate(records):
        normal = normals.get(record["normal_id"])
        mask_item = masks.get(record["mask_id"])
        if normal is None or mask_item is None:
            raise ValueError(f"sample record {index} references an unknown source")
        if record["source_mask_type"] != mask_item.defect_type:
            raise ValueError(f"sample record {index} source mask type mismatch")
        if record["source_normal_sha256"] != normal.image_sha256:
            raise ValueError(f"sample record {index} source normal hash mismatch")
        if record["source_mask_sha256"] != mask_item.mask_sha256:
            raise ValueError(f"sample record {index} source mask hash mismatch")
        image_path = _contained_artifact(output, record["image_relpath"])
        mask_path = _contained_artifact(output, record["mask_relpath"])
        expected_paths.update((image_path, mask_path))
        if sha256_file(image_path) != record["image_sha256"]:
            raise ValueError(f"sample artifact image hash mismatch at record {index}")
        if sha256_file(mask_path) != record["mask_sha256"]:
            raise ValueError(f"sample artifact mask hash mismatch at record {index}")
        with Image.open(image_path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (image_size, image_size):
                raise ValueError(f"sample image contract mismatch at record {index}")
            image_array = np.asarray(image, dtype=np.uint8)
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            if mask_image.mode != "L" or mask_image.size != (image_size, image_size):
                raise ValueError(f"sample mask contract mismatch at record {index}")
            saved_mask = np.asarray(mask_image, dtype=np.uint8)
        if mask_item.sample_id not in mask_arrays:
            mask_arrays[mask_item.sample_id] = (
                mask_item.mask[0].detach().cpu().numpy() * 255
            ).astype(np.uint8)
        expected_mask = mask_arrays[mask_item.sample_id]
        if not np.array_equal(saved_mask, expected_mask):
            raise ValueError(f"sample artifact differs from heldout source mask at record {index}")
        if normal.sample_id not in normal_arrays:
            loaded = load_training_image(normal, image_size)
            normal_image = ((loaded.pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)
            normal_arrays[normal.sample_id] = np.asarray(
                _tensor_to_pil(normal_image), dtype=np.uint8
            )
        outside = expected_mask == 0
        if not np.array_equal(image_array[outside], normal_arrays[normal.sample_id][outside]):
            raise ValueError(f"sample artifact changed mask exterior at record {index}")
    actual_paths = {path.resolve(strict=True) for path in samples.iterdir()}
    if actual_paths != expected_paths:
        raise ValueError("samples directory contains unexpected artifacts")


def run_p2_b1(context: P2B1Context | Any) -> int:
    try:
        preflight = validate_p2_b1_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_p2_b1(preflight, identity)
        records = tuple(result["records"])
        pairwise = tuple(result["pairwise"])
        schedule = build_heldout_generation_schedule(
            normal_ids=preflight.normal_ids,
            mask_ids=preflight.mask_ids,
            source_mask_types=preflight.source_mask_types,
            defect_types=context.experiment.defect_types,
            seeds=context.experiment.sample_seeds,
        )
        _validate_generation_records(records, pairwise, schedule, context.experiment)
        _validate_sample_artifacts(preflight, records)
        summary = summarize_generation(records, pairwise, context.experiment)
        summary["comparison_to_p2_b0"] = compare_with_baseline(
            summary, preflight.baseline_summary
        )
        head_end = _verify_p2_b1_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "metrics.jsonl", records)
        _verify_p2_b1_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "pairwise_metrics.jsonl", pairwise)
        _verify_p2_b1_integrity(preflight, identity)
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": "p2_b1_generation_provenance_v1",
                "git_head_start": preflight.git_head,
                "git_head_end": head_end,
                "git_dirty": False,
                "training_performed": False,
                "checkpoint_written": False,
                "evaluation_mode": "diagnostic_only",
                "input_hashes": preflight.input_hashes,
                "source_snapshot_before": preflight.source_snapshot,
                "source_snapshot_after": preflight.source_snapshot,
                "model_snapshot_before": preflight.model_snapshot,
                "model_snapshot_after": preflight.model_snapshot,
                "checkpoint_snapshot_before": preflight.checkpoint_snapshot,
                "checkpoint_snapshot_after": preflight.checkpoint_snapshot,
                "baseline_snapshot_before": preflight.baseline_snapshot,
                "baseline_snapshot_after": preflight.baseline_snapshot,
            }
        )
        report = {
            "schema_version": "p2_b1_generation_report_v1",
            "status": "complete",
            "reason": None,
            "training_performed": False,
            "checkpoint_written": False,
            "evaluation_mode": "diagnostic_only",
            "summary": summary,
            "peak_vram_bytes": result["peak_vram_bytes"],
            "input_hashes": preflight.input_hashes,
            "normal_ids": list(preflight.normal_ids),
            "mask_ids": list(preflight.mask_ids),
            "source_mask_types": list(preflight.source_mask_types),
        }
        write_json_atomic(context.output / "provenance.json", provenance)
        _verify_p2_b1_integrity(preflight, identity)
        write_json_atomic(context.output / "report.json", report)
        _verify_p2_b1_integrity(preflight, identity)
        _validate_sample_artifacts(preflight, records)
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": "p2_b1_generation_report_v1",
            "status": "failed",
            "reason": "runtime_or_integrity_failure",
        }
        try:
            _verify_output_identity(preflight.base, identity)  # type: ignore[arg-type]
        except (OSError, RuntimeError):
            pass
        else:
            try:
                _cleanup_partial(context.output)
                write_json_atomic(context.output / "report.json", failed)
            except OSError:
                pass
        print(json.dumps(failed, sort_keys=True))
        return 1


def build_p2_b1_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the P2-B1 Hazelnut heldout probe")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_p2_b1_parser().parse_args(argv)
    paths_path = Path(args.paths)
    generation_path = Path(args.generation)
    source_experiment_path = Path(args.source_experiment)
    try:
        return run_p2_b1(
            P2B1Context(
                paths=load_config(paths_path),
                experiment=load_p2_b1_config(generation_path),
                source_experiment=load_p2_a1_config(source_experiment_path),
                manifest=Path(args.manifest),
                lock=Path(args.lock),
                source_run=Path(args.source_run),
                baseline_run=Path(args.baseline_run),
                output=Path(args.output),
                config_paths={
                    "generation": generation_path,
                    "paths": paths_path,
                    "source_experiment": source_experiment_path,
                },
                cwd=Path.cwd(),
            )
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1


__all__ = [
    "P2B1Context",
    "_prepare_normal_tensors",
    "_validate_generation_records",
    "_validate_sample_artifacts",
    "_validated_baseline_run",
    "build_p2_b1_parser",
    "compare_with_baseline",
    "run_p2_b1",
    "summarize_generation",
]

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from cadiff.config import ProjectConfig, load_config
from cadiff.data.mvtec_manifest import exact_evenly_spaced
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
from cadiff.p2_b0.config import P2B0Config, load_p2_b0_config
from cadiff.p2_b0.generation import (
    GenerationSlot,
    build_generation_schedule,
    compute_mask_metrics,
    denoise_masked_infill,
)
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic


@dataclass(frozen=True)
class P2B0Context:
    paths: ProjectConfig | Any
    experiment: P2B0Config
    source_experiment: P2A1Config | Any
    manifest: Path
    lock: Path
    source_run: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class P2B0Preflight:
    context: P2B0Context
    base: FewshotPreflight
    normals: tuple[Any, ...]
    masks: tuple[Any, ...]
    normal_ids: tuple[str, ...]
    mask_ids: tuple[str, ...]
    git_head: str
    input_hashes: dict[str, str]
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]
    checkpoint_snapshot: dict[str, dict[str, int | str]]


_RECORD_KEYS = {
    "schema_version",
    "slot_id",
    "normal_id",
    "mask_id",
    "seed",
    "defect_type",
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
    "seed",
    "left_type",
    "right_type",
    "masked_l1",
    "finite",
}
_SHA_FIELDS = ("image_sha256", "mask_sha256", "generated_tensor_sha256")
_METRIC_FIELDS = (
    "inside_l1",
    "outside_l1",
    "outside_to_inside_ratio",
    "change_concentration",
    "composite_outside_l1",
)


def _sample_id(value: Any) -> str:
    sample_id = getattr(value, "sample_id", None)
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("probe samples must have non-empty sample_id values")
    return sample_id


def _ordered(items: tuple[Any, ...], name: str) -> list[Any]:
    if not items:
        raise ValueError(f"{name} must be non-empty")
    ordered = sorted(items, key=lambda item: str(getattr(item, "image_relpath", "")))
    identifiers = [_sample_id(item) for item in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} sample IDs must be unique")
    return ordered


def select_probe_inputs(
    *,
    normals: tuple[Any, ...],
    train_masks: tuple[Any, ...],
    heldout_masks: tuple[Any, ...],
    count: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if type(count) is not int or count < 2:
        raise ValueError("count must be an integer of at least two")
    ordered_normals = _ordered(normals, "normals")
    ordered_train = _ordered(train_masks, "train_masks")
    heldout_ids = {_sample_id(item) for item in _ordered(heldout_masks, "heldout_masks")}
    train_ids = {_sample_id(item) for item in ordered_train}
    if train_ids & heldout_ids:
        raise ValueError("train and heldout mask samples must be disjoint")
    if len(ordered_normals) < count or len(ordered_train) < count:
        raise ValueError("probe input pools are smaller than count")
    selected_normals = exact_evenly_spaced(ordered_normals, count)
    selected_masks = exact_evenly_spaced(ordered_train, count)
    return tuple(selected_normals), tuple(selected_masks)


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
    schedule: tuple[GenerationSlot, ...],
    config: P2B0Config,
) -> None:
    if len(records) != len(schedule):
        raise ValueError("generation record count does not match the schedule")
    seen_paths: set[str] = set()
    for index, (record, slot) in enumerate(zip(records, schedule, strict=True)):
        if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
            raise ValueError(f"generation record {index} has an invalid exact schema")
        expected = {
            "slot_id": slot.slot_id,
            "normal_id": slot.normal_id,
            "mask_id": slot.mask_id,
            "seed": slot.seed,
            "defect_type": slot.defect_type,
        }
        if any(record[field] != value for field, value in expected.items()):
            raise ValueError(
                f"generation record {index} does not match the schedule seed or pairing"
            )
        if type(record["seed"]) is not int:
            raise ValueError(f"generation record {index} seed must be an integer")
        if record["schema_version"] != "p2_b0_generation_record_v1":
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
        for field in _METRIC_FIELDS:
            _finite_float(record[field], f"generation record {index} {field}")
        if record["finite"] is not True:
            raise ValueError(f"generation record {index} is not finite")
        if record["change_concentration"] > 1.0:
            raise ValueError(f"generation record {index} change_concentration exceeds one")
        if record["composite_outside_l1"] > 1e-7:
            raise ValueError(f"generation record {index} composite outside changed")

    expected_pair_count = (
        config.normal_reference_count
        * len(config.sample_seeds)
        * math.comb(len(config.defect_types), 2)
    )
    if len(pairwise) != expected_pair_count:
        raise ValueError("pairwise record count is invalid")
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for index, record in enumerate(pairwise):
        if not isinstance(record, dict) or set(record) != _PAIRWISE_KEYS:
            raise ValueError(f"pairwise record {index} has an invalid exact schema")
        if record["schema_version"] != "p2_b0_pairwise_record_v1":
            raise ValueError(f"pairwise record {index} schema is incompatible")
        if type(record["seed"]) is not int or record["seed"] not in config.sample_seeds:
            raise ValueError(f"pairwise record {index} seed is invalid")
        if (
            not isinstance(record["context_id"], str)
            or not record["context_id"]
            or not isinstance(record["normal_id"], str)
            or not isinstance(record["mask_id"], str)
        ):
            raise ValueError(f"pairwise record {index} identity is invalid")
        if record["finite"] is not True:
            raise ValueError(f"pairwise record {index} is not finite")
        _finite_float(record["masked_l1"], f"pairwise record {index} masked_l1")
        key = (record["normal_id"], record["mask_id"], record["seed"])
        groups.setdefault(key, []).append(record)
    expected_contexts = {(slot.normal_id, slot.mask_id, slot.seed) for slot in schedule}
    if set(groups) != expected_contexts:
        raise ValueError("pairwise contexts do not match the generation schedule")
    expected_pairs = set(itertools.combinations(config.defect_types, 2))
    for key, values in groups.items():
        actual_pairs = {(value["left_type"], value["right_type"]) for value in values}
        if actual_pairs != expected_pairs or len(values) != len(expected_pairs):
            raise ValueError(f"pairwise type pairing is invalid for {key}")
        if len({value["context_id"] for value in values}) != 1:
            raise ValueError(f"pairwise context_id is inconsistent for {key}")


def summarize_generation(
    records: tuple[dict[str, Any], ...],
    pairwise: tuple[dict[str, Any], ...],
    config: P2B0Config,
) -> dict[str, Any]:
    per_type: dict[str, dict[str, float]] = {}
    for defect_type in config.defect_types:
        selected = [record for record in records if record["defect_type"] == defect_type]
        if not selected:
            raise ValueError(f"no generation records for {defect_type}")
        per_type[defect_type] = {
            f"{field}_median": float(statistics.median(record[field] for record in selected))
            for field in _METRIC_FIELDS
        }
    inside = float(statistics.median(record["inside_l1"] for record in records))
    outside = float(statistics.median(record["outside_l1"] for record in records))
    separation = float(statistics.median(record["masked_l1"] for record in pairwise))
    return {
        "diagnostic_only": True,
        "record_count": len(records),
        "pairwise_record_count": len(pairwise),
        "inside_l1_median": inside,
        "outside_l1_median": outside,
        "inside_change_exceeds_outside": inside > outside,
        "pairwise_masked_l1_median": separation,
        "route_noncollapse_observed": separation > 1e-6,
        "per_type": per_type,
    }


def _real_child(root: Path, name: str, *, directory: bool) -> Path:
    if Path(name).name != name:
        raise ValueError("artifact name must be a direct child")
    candidate = root / name
    if candidate.is_symlink():
        raise ValueError(f"artifact symlink is not allowed: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != root.resolve(strict=True):
        raise ValueError(f"artifact escapes its root: {candidate}")
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"artifact must be a regular {kind}: {candidate}")
    return resolved


def _validated_source_run(source_run: Path, runs_root: Path) -> tuple[Path, Path]:
    lexical_runs = Path(os.path.abspath(runs_root))
    lexical_source = Path(os.path.abspath(source_run))
    try:
        lexical_relative = lexical_source.relative_to(lexical_runs)
    except ValueError as error:
        raise ValueError("source_run must be inside runs_root") from error
    current = lexical_runs
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source_run path must not contain symlinks: {current}")
    resolved_runs = lexical_runs.resolve(strict=True)
    resolved_source = lexical_source.resolve(strict=True)
    if not resolved_runs.is_dir() or not resolved_source.is_dir():
        raise ValueError("runs_root and source_run must be directories")
    try:
        relative = resolved_source.relative_to(resolved_runs)
    except ValueError as error:
        raise ValueError("source_run must be inside runs_root") from error
    if not relative.parts:
        raise ValueError("source_run must be strictly inside runs_root")
    report_path = _real_child(resolved_source, "report.json", directory=False)
    report = _strict_json_object(report_path)
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("evaluation_mode") != "diagnostic_only"
        or report.get("training_performed") is not True
        or report.get("checkpoint_written") is not True
    ):
        raise ValueError("source_run is not a completed Hazelnut diagnostic run")
    checkpoint = _real_child(resolved_source, "checkpoint", directory=True)
    return resolved_source, checkpoint


def _checkpoint_snapshot(checkpoint: Path) -> dict[str, dict[str, int | str]]:
    expected = ("adapter.safetensors", "conditioner.safetensors", "metadata.json")
    return {name: _snapshot(_real_child(checkpoint, name, directory=False)) for name in expected}


def validate_p2_b0_preflight(context: P2B0Context) -> P2B0Preflight:
    if not isinstance(context, P2B0Context):
        raise ValueError("context must be a P2B0Context")
    if not isinstance(context.experiment, P2B0Config):
        raise ValueError("experiment must be a P2B0Config")
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
    _, checkpoint = _validated_source_run(context.source_run, context.paths.runs_root)
    checkpoint_snapshot = _checkpoint_snapshot(checkpoint)
    index = load_training_index(
        context.manifest,
        context.paths.dataset_root,
        context.source_experiment,  # type: ignore[arg-type]
    )
    normals, masks = select_probe_inputs(
        normals=index.normals,
        train_masks=base.split.train,
        heldout_masks=base.split.evaluation,
        count=context.experiment.normal_reference_count,
    )
    normal_snapshot = {str(item.image_path): _snapshot(item.image_path) for item in normals}
    source_snapshot = {**base.source_snapshot, **normal_snapshot}
    input_hashes = dict(base.input_hashes)
    input_hashes["source_report"] = sha256_file(context.source_run / "report.json")
    for name, value in checkpoint_snapshot.items():
        input_hashes[f"source_checkpoint/{name}"] = str(value["sha256"])
    return P2B0Preflight(
        context=context,
        base=base,
        normals=normals,
        masks=masks,
        normal_ids=tuple(_sample_id(item) for item in normals),
        mask_ids=tuple(_sample_id(item) for item in masks),
        git_head=base.git_head,
        input_hashes=input_hashes,
        source_snapshot=source_snapshot,
        model_snapshot=base.model_snapshot,
        checkpoint_snapshot=checkpoint_snapshot,
    )


def _verify_p2_b0_integrity(preflight: P2B0Preflight, identity: tuple[int, int]) -> str:
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
    return head


def _save_sample(
    *, output: Path, slot_id: str, image: torch.Tensor, mask: torch.Tensor
) -> tuple[str, str, str, str]:
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    image_relpath = f"samples/{slot_id}_image.png"
    mask_relpath = f"samples/{slot_id}_mask.png"
    image_path = output / image_relpath
    mask_path = output / mask_relpath
    if image_path.exists() or mask_path.exists():
        raise FileExistsError(f"sample artifact already exists for {slot_id}")
    _tensor_to_pil(image[0]).save(image_path)
    mask_array = (mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_array, mode="L").save(mask_path)
    return image_relpath, mask_relpath, sha256_file(image_path), sha256_file(mask_path)


def _pairwise_records(
    *,
    normal_id: str,
    mask_id: str,
    seed: int,
    generated: dict[str, torch.Tensor],
    source_mask: torch.Tensor,
    defect_types: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    context_id = hashlib.sha256(f"{normal_id}\n{mask_id}\n{seed}".encode()).hexdigest()
    mask = source_mask.to(dtype=torch.float32)
    result: list[dict[str, Any]] = []
    for left, right in itertools.combinations(defect_types, 2):
        difference = (
            (generated[left].float() - generated[right].float()).abs().mean(dim=1, keepdim=True)
        )
        masked_l1 = float(((difference * mask).sum() / mask.sum()).item())
        result.append(
            {
                "schema_version": "p2_b0_pairwise_record_v1",
                "context_id": context_id,
                "normal_id": normal_id,
                "mask_id": mask_id,
                "seed": seed,
                "left_type": left,
                "right_type": right,
                "masked_l1": masked_l1,
                "finite": math.isfinite(masked_l1),
            }
        )
    return tuple(result)


@torch.no_grad()
def _execute_p2_b0(preflight: P2B0Preflight, identity: tuple[int, int]) -> dict[str, Any]:
    context = preflight.context
    config = context.experiment
    source_config = context.source_experiment
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
        checkpoint, config=source_config, device=device
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).eval()
    schedule = build_generation_schedule(
        normal_ids=preflight.normal_ids,
        mask_ids=preflight.mask_ids,
        defect_types=config.defect_types,
        seeds=config.sample_seeds,
    )
    slot_by_key = {
        (slot.normal_id, slot.mask_id, slot.seed, slot.defect_type): slot for slot in schedule
    }
    records: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    for normal_item, mask_item in zip(preflight.normals, preflight.masks, strict=True):
        loaded = load_training_image(normal_item, config.image_size)
        normal_pixels = loaded.pixels.unsqueeze(0).to(device=device, dtype=torch.float16)
        normal_image = ((normal_pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)
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
                slot = slot_by_key[(normal_item.sample_id, mask_item.sample_id, seed, defect_type)]
                image_relpath, mask_relpath, image_sha, mask_sha = _save_sample(
                    output=context.output,
                    slot_id=slot.slot_id,
                    image=composite,
                    mask=source_mask,
                )
                generated_by_type[defect_type] = composite.detach().clone()
                records.append(
                    {
                        "schema_version": "p2_b0_generation_record_v1",
                        **asdict(slot),
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
                    seed=seed,
                    generated=generated_by_type,
                    source_mask=source_mask,
                    defect_types=config.defect_types,
                )
            )
            print(
                json.dumps(
                    {
                        "stage": "generation",
                        "records": len(records),
                        "total": len(schedule),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _assert_parameter_versions_unchanged(components.unet, unet_versions)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only generation accumulated gradients")
    _verify_p2_b0_integrity(preflight, identity)
    torch.cuda.synchronize(device)
    return {
        "records": tuple(records),
        "pairwise": tuple(pairwise),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }


def _contained_artifact(output: Path, relpath: object) -> Path:
    if not isinstance(relpath, str) or not relpath.startswith("samples/"):
        raise ValueError("sample artifact relative path is invalid")
    candidate = output / relpath
    if candidate.is_symlink():
        raise ValueError(f"sample artifact symlink is not allowed: {relpath}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(output.resolve(strict=True)):
        raise ValueError(f"sample artifact escapes output: {relpath}")
    return resolved


def _validate_sample_artifacts(
    output: Path, records: tuple[dict[str, Any], ...], *, image_size: int
) -> None:
    samples = _real_child(output, "samples", directory=True)
    expected_paths: set[Path] = set()
    for index, record in enumerate(records):
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
        with Image.open(mask_path) as mask:
            mask.load()
            values = np.unique(np.asarray(mask, dtype=np.uint8))
            if (
                mask.mode != "L"
                or mask.size != (image_size, image_size)
                or {int(value) for value in values} != {0, 255}
            ):
                raise ValueError(f"sample mask contract mismatch at record {index}")
    actual_paths = {path.resolve(strict=True) for path in samples.iterdir()}
    if actual_paths != expected_paths:
        raise ValueError("samples directory contains unexpected artifacts")


def _cleanup_partial(output: Path) -> None:
    for name in (
        "metrics.jsonl",
        "pairwise_metrics.jsonl",
        "provenance.json",
        "report.json",
    ):
        candidate = output / name
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink(missing_ok=True)
    samples = output / "samples"
    if samples.is_symlink():
        samples.unlink(missing_ok=True)
    elif samples.is_dir():
        shutil.rmtree(samples)


def run_p2_b0(context: P2B0Context | Any) -> int:
    try:
        preflight = validate_p2_b0_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_p2_b0(preflight, identity)
        records = tuple(result["records"])
        pairwise = tuple(result["pairwise"])
        schedule = build_generation_schedule(
            normal_ids=preflight.normal_ids,
            mask_ids=preflight.mask_ids,
            defect_types=context.experiment.defect_types,
            seeds=context.experiment.sample_seeds,
        )
        _validate_generation_records(records, pairwise, schedule, context.experiment)
        _validate_sample_artifacts(
            context.output, records, image_size=context.experiment.image_size
        )
        summary = summarize_generation(records, pairwise, context.experiment)
        head_end = _verify_p2_b0_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "metrics.jsonl", records)
        _verify_p2_b0_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "pairwise_metrics.jsonl", pairwise)
        _verify_p2_b0_integrity(preflight, identity)
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": "p2_b0_generation_provenance_v1",
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
            }
        )
        report = {
            "schema_version": "p2_b0_generation_report_v1",
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
        }
        write_json_atomic(context.output / "provenance.json", provenance)
        _verify_p2_b0_integrity(preflight, identity)
        write_json_atomic(context.output / "report.json", report)
        _verify_p2_b0_integrity(preflight, identity)
        _validate_sample_artifacts(
            context.output, records, image_size=context.experiment.image_size
        )
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": "p2_b0_generation_report_v1",
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


def build_p2_b0_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the P2-B0 Hazelnut generation probe")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_p2_b0_parser().parse_args(argv)
    paths_path = Path(args.paths)
    generation_path = Path(args.generation)
    source_experiment_path = Path(args.source_experiment)
    try:
        return run_p2_b0(
            P2B0Context(
                paths=load_config(paths_path),
                experiment=load_p2_b0_config(generation_path),
                source_experiment=load_p2_a1_config(source_experiment_path),
                manifest=Path(args.manifest),
                lock=Path(args.lock),
                source_run=Path(args.source_run),
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
    "P2B0Context",
    "_validate_generation_records",
    "_validate_sample_artifacts",
    "build_p2_b0_parser",
    "run_p2_b0",
    "select_probe_inputs",
    "summarize_generation",
]

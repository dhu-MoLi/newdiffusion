from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from cadiff.config import ProjectConfig, load_config
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.dataset import load_training_image
from cadiff.p2_a0.all_pairs import _strict_json_object
from cadiff.p2_a0.fewshot import (
    _output_identity,
    _snapshot,
    _tensor_sha256,
    _tensor_to_pil,
    _verify_output_identity,
    _write_jsonl_atomic,
)
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_a1.runner import _load_p2_a1_checkpoint
from cadiff.p2_b0.generation import compute_mask_metrics, denoise_masked_infill
from cadiff.p2_b0.runner import (
    _contained_artifact,
    _real_child,
    _validated_source_run,
)
from cadiff.p2_b1.config import P2B1Config, load_p2_b1_config
from cadiff.p2_b1.generation import build_heldout_generation_schedule
from cadiff.p2_b1.runner import (
    P2B1Context,
    P2B1Preflight,
    _prepare_normal_tensors,
    _validate_generation_records,
    _validate_sample_artifacts,
    _validated_child_run,
    _verify_p2_b1_integrity,
    validate_p2_b1_preflight,
)
from cadiff.p2_b1.type_prompt import (
    BlindAssignment,
    TypePromptConfig,
    TypePromptJob,
    build_blind_assignments,
    load_type_prompt_config,
    select_type_prompt_jobs,
)
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic


@dataclass(frozen=True)
class TypePromptContext:
    paths: ProjectConfig | Any
    generation: P2B1Config | Any
    probe: TypePromptConfig
    source_experiment: P2A1Config | Any
    manifest: Path
    lock: Path
    source_run: Path
    baseline_run: Path
    reference_run: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class ReferenceRun:
    path: Path
    report: dict[str, Any]
    provenance: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    pairwise: tuple[dict[str, Any], ...]
    snapshot: dict[str, dict[str, int | str]]


@dataclass(frozen=True)
class TypePromptPreflight:
    context: TypePromptContext
    base: P2B1Preflight
    reference: ReferenceRun
    jobs: tuple[TypePromptJob, ...]
    assignments: tuple[BlindAssignment, ...]
    git_head: str
    input_hashes: dict[str, str]
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]
    checkpoint_snapshot: dict[str, dict[str, int | str]]
    reference_snapshot: dict[str, dict[str, int | str]]


_PROMPT_RECORD_KEYS = {
    "schema_version",
    *TypePromptJob.__dataclass_fields__,
    "normal_prompt",
    "type_prompt_sha256",
    "prompt_embedding_sha256",
    "normal_latent_sha256",
    "context_latent_sha256",
    "noise_sha256",
    "image_relpath",
    "mask_relpath",
    "real_reference_relpath",
    "image_sha256",
    "mask_sha256",
    "real_reference_sha256",
    "generated_tensor_sha256",
    "inside_l1",
    "outside_l1",
    "outside_to_inside_ratio",
    "change_concentration",
    "composite_outside_l1",
    "baseline_candidate_inside_l1",
    "finite",
}
_PROMPT_SHA_FIELDS = (
    "pair_id",
    "baseline_slot_id",
    "source_normal_sha256",
    "source_mask_sha256",
    "baseline_image_sha256",
    "baseline_mask_sha256",
    "type_prompt_sha256",
    "prompt_embedding_sha256",
    "normal_latent_sha256",
    "context_latent_sha256",
    "noise_sha256",
    "image_sha256",
    "mask_sha256",
    "real_reference_sha256",
    "generated_tensor_sha256",
)
_PROMPT_METRICS = (
    "inside_l1",
    "outside_l1",
    "outside_to_inside_ratio",
    "change_concentration",
    "composite_outside_l1",
    "baseline_candidate_inside_l1",
)


def _strict_jsonl_objects(path: Path) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL line {line_number} in {path}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=hook,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid strict JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} in {path} must be an object")
        result.append(value)
    if not result:
        raise ValueError(f"JSONL artifact must not be empty: {path}")
    return tuple(result)


def _contained_output_artifact(output: Path, relpath: object, prefix: str) -> Path:
    if not isinstance(relpath, str) or not relpath.startswith(f"{prefix}/"):
        raise ValueError(f"{prefix} artifact relative path is invalid")
    lexical_output = Path(os.path.abspath(output))
    lexical_path = Path(os.path.abspath(lexical_output / relpath))
    try:
        relative = lexical_path.relative_to(lexical_output)
    except ValueError as error:
        raise ValueError(f"{prefix} artifact escapes output") from error
    current = lexical_output
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{prefix} artifact path must not contain symlinks")
    resolved_output = lexical_output.resolve(strict=True)
    resolved_path = lexical_path.resolve(strict=True)
    if not resolved_path.is_file() or not resolved_path.is_relative_to(resolved_output):
        raise ValueError(f"{prefix} artifact must be a regular file inside output")
    return resolved_path


def _validated_reference_run(
    run: Path, runs_root: Path, config: TypePromptConfig
) -> ReferenceRun:
    resolved = _validated_child_run(run, runs_root, "reference_run")
    paths: dict[str, Path] = {}
    snapshot: dict[str, dict[str, int | str]] = {}
    for name, expected_hash in config.reference_hashes.items():
        path = _real_child(resolved, name, directory=False)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"reference_run hash mismatch for {name}")
        paths[name] = path
        snapshot[name] = _snapshot(path)
    report = _strict_json_object(paths["report.json"])
    provenance = _strict_json_object(paths["provenance.json"])
    if (
        report.get("schema_version") != "p2_b1_generation_report_v1"
        or report.get("status") != "complete"
        or report.get("reason") is not None
        or report.get("training_performed") is not False
        or report.get("checkpoint_written") is not False
        or report.get("evaluation_mode") != "diagnostic_only"
    ):
        raise ValueError("reference_run report is not the accepted P2-B1 diagnostic")
    if (
        provenance.get("schema_version") != "p2_b1_generation_provenance_v1"
        or provenance.get("git_head_start") != config.reference_git_head
        or provenance.get("git_head_end") != config.reference_git_head
        or provenance.get("git_dirty") is not False
        or provenance.get("training_performed") is not False
        or provenance.get("checkpoint_written") is not False
    ):
        raise ValueError("reference_run provenance does not authenticate the accepted HEAD")
    return ReferenceRun(
        path=resolved,
        report=report,
        provenance=provenance,
        records=_strict_jsonl_objects(paths["metrics.jsonl"]),
        pairwise=_strict_jsonl_objects(paths["pairwise_metrics.jsonl"]),
        snapshot=snapshot,
    )


def _bind_reference_to_current_preflight(
    reference: ReferenceRun, base: P2B1Preflight | Any
) -> None:
    provenance = reference.provenance
    reference_inputs = provenance.get("input_hashes")
    report_inputs = reference.report.get("input_hashes")
    if not isinstance(reference_inputs, dict) or report_inputs != reference_inputs:
        raise ValueError("reference_run input hashes are missing or inconsistent")
    if any(base.input_hashes.get(key) != value for key, value in reference_inputs.items()):
        raise ValueError("reference_run input hashes differ from the current preflight")
    expected = {
        "source_snapshot_before": base.source_snapshot,
        "source_snapshot_after": base.source_snapshot,
        "model_snapshot_before": base.model_snapshot,
        "model_snapshot_after": base.model_snapshot,
        "checkpoint_snapshot_before": base.checkpoint_snapshot,
        "checkpoint_snapshot_after": base.checkpoint_snapshot,
        "baseline_snapshot_before": base.baseline_snapshot,
        "baseline_snapshot_after": base.baseline_snapshot,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"reference_run {field} differs from the current preflight")


def _capture_inference_state(
    modules: Mapping[str, torch.nn.Module],
) -> dict[str, tuple[int, int, str, str, tuple[int, ...]]]:
    state: dict[str, tuple[int, int, str, str, tuple[int, ...]]] = {}
    for module_name, module in sorted(modules.items()):
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"{module_name} must be a torch module")
        tensors = tuple(("parameter", name, value) for name, value in module.named_parameters())
        tensors += tuple(("buffer", name, value) for name, value in module.named_buffers())
        for kind, name, value in tensors:
            state[f"{module_name}.{kind}.{name}"] = (
                id(value),
                value._version,
                str(value.dtype),
                str(value.device),
                tuple(value.shape),
            )
    if not state:
        raise ValueError("inference state must contain parameters or buffers")
    return state


def _assert_inference_state_unchanged(
    modules: Mapping[str, torch.nn.Module],
    expected: Mapping[str, tuple[int, int, str, str, tuple[int, ...]]],
) -> None:
    if _capture_inference_state(modules) != dict(expected):
        raise RuntimeError("inference-only module state changed during Type-Prompt probe")
    if any(
        parameter.grad is not None
        for module in modules.values()
        for parameter in module.parameters()
    ):
        raise RuntimeError("inference-only Type-Prompt probe accumulated gradients")


def _validate_exact_regular_children(directory: Path, expected_names: set[str]) -> None:
    resolved = directory.resolve(strict=True)
    if directory.is_symlink() or not resolved.is_dir():
        raise ValueError(f"artifact directory must be a real directory: {directory}")
    actual_names: set[str] = set()
    for child in directory.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError(f"artifact directory contains a symlink or non-file: {child.name}")
        actual_names.add(child.name)
    if actual_names != expected_names:
        raise ValueError("artifact directory contains unexpected lexical children")


def _cleanup_type_prompt_partial(output: Path) -> None:
    owned = {
        "samples",
        "references",
        "comparisons",
        "private",
        "blind_package",
        "records.jsonl",
        "blind_manifest.jsonl",
        "reveal_key.json",
        "provenance.json",
        "report.json",
    }
    for name in owned:
        child = output / name
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except FileNotFoundError:
            continue


def validate_type_prompt_preflight(context: TypePromptContext) -> TypePromptPreflight:
    if not isinstance(context, TypePromptContext):
        raise ValueError("context must be a TypePromptContext")
    if not isinstance(context.generation, P2B1Config):
        raise ValueError("generation must be a P2B1Config")
    if not isinstance(context.probe, TypePromptConfig):
        raise ValueError("probe must be a TypePromptConfig")
    if not isinstance(context.source_experiment, P2A1Config):
        raise ValueError("source_experiment must be a P2A1Config")
    if (
        context.probe.category != context.generation.category
        or context.probe.normal_prompt != context.generation.normal_prompt
        or tuple(context.probe.prompts) != context.generation.defect_types
    ):
        raise ValueError("Type-Prompt and P2-B1 generation contracts must match")
    base = validate_p2_b1_preflight(
        P2B1Context(
            paths=context.paths,
            experiment=context.generation,
            source_experiment=context.source_experiment,
            manifest=context.manifest,
            lock=context.lock,
            source_run=context.source_run,
            baseline_run=context.baseline_run,
            output=context.output,
            config_paths=context.config_paths,
            cwd=context.cwd,
        )
    )
    reference = _validated_reference_run(
        context.reference_run, context.paths.runs_root, context.probe
    )
    schedule = build_heldout_generation_schedule(
        normal_ids=base.normal_ids,
        mask_ids=base.mask_ids,
        source_mask_types=base.source_mask_types,
        defect_types=context.generation.defect_types,
        seeds=context.generation.sample_seeds,
    )
    _validate_generation_records(
        reference.records, reference.pairwise, schedule, context.generation
    )
    reference_artifacts = SimpleNamespace(
        context=SimpleNamespace(output=reference.path, experiment=context.generation),
        normals=base.normals,
        masks=base.masks,
    )
    _validate_sample_artifacts(reference_artifacts, reference.records)
    if (
        reference.report.get("normal_ids") != list(base.normal_ids)
        or reference.report.get("mask_ids") != list(base.mask_ids)
        or reference.report.get("source_mask_types") != list(base.source_mask_types)
    ):
        raise ValueError("reference_run report selection differs from reconstructed P2-B1")
    if (
        reference.provenance.get("checkpoint_snapshot_before") != base.checkpoint_snapshot
        or reference.provenance.get("checkpoint_snapshot_after") != base.checkpoint_snapshot
    ):
        raise ValueError("reference_run checkpoint snapshot differs from the frozen source")
    _bind_reference_to_current_preflight(reference, base)
    jobs = select_type_prompt_jobs(reference.records, context.probe)
    reference_snapshot = dict(reference.snapshot)
    for job in jobs:
        for field in ("baseline_image_relpath", "baseline_mask_relpath"):
            path = _contained_artifact(reference.path, getattr(job, field))
            expected = getattr(job, field.replace("relpath", "sha256"))
            if sha256_file(path) != expected:
                raise ValueError(f"reference_run selected artifact hash mismatch for {job.pair_id}")
            reference_snapshot[str(path)] = _snapshot(path)
    input_hashes = dict(base.input_hashes)
    for name, digest in context.probe.reference_hashes.items():
        input_hashes[f"reference_run/{name}"] = digest
    input_hashes["type_prompt_config"] = sha256_file(context.config_paths["probe"])
    return TypePromptPreflight(
        context=context,
        base=base,
        reference=reference,
        jobs=jobs,
        assignments=build_blind_assignments(jobs, context.probe.blind_seed),
        git_head=base.git_head,
        input_hashes=input_hashes,
        source_snapshot=base.source_snapshot,
        model_snapshot=base.model_snapshot,
        checkpoint_snapshot=base.checkpoint_snapshot,
        reference_snapshot=reference_snapshot,
    )


def _verify_type_prompt_integrity(
    preflight: TypePromptPreflight, identity: tuple[int, int]
) -> str:
    head = _verify_p2_b1_integrity(preflight.base, identity)
    reference = _validated_reference_run(
        preflight.context.reference_run,
        preflight.context.paths.runs_root,
        preflight.context.probe,
    )
    current = dict(reference.snapshot)
    for path_text in preflight.reference_snapshot:
        path = Path(path_text)
        if path.is_absolute():
            current[path_text] = _snapshot(path)
    if current != preflight.reference_snapshot:
        raise RuntimeError("P2-B1 reference artifacts changed during Type-Prompt probe")
    return head


def _baseline_candidate_inside_l1(
    candidate: torch.Tensor, baseline_path: Path, source_mask: torch.Tensor
) -> float:
    with Image.open(baseline_path) as image:
        image.load()
        baseline_array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    baseline = (
        torch.from_numpy(baseline_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=candidate.device, dtype=torch.float32)
        / 255.0
    )
    difference = (candidate.float() - baseline).abs().mean(dim=1, keepdim=True)
    value = float(((difference * source_mask).sum() / source_mask.sum()).item())
    if not math.isfinite(value):
        raise ValueError("baseline/candidate masked L1 must be finite")
    return value


def _save_type_prompt_sample(
    *, output: Path, pair_id: str, image: torch.Tensor, mask: torch.Tensor
) -> tuple[str, str, str, str]:
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    image_relpath = f"samples/{pair_id}_image.png"
    mask_relpath = f"samples/{pair_id}_mask.png"
    image_path = output / image_relpath
    mask_path = output / mask_relpath
    if image_path.exists() or mask_path.exists():
        raise FileExistsError(f"sample artifact already exists for {pair_id}")
    _tensor_to_pil(image[0]).save(image_path)
    mask_array = (mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_array).save(mask_path)
    return image_relpath, mask_relpath, sha256_file(image_path), sha256_file(mask_path)


@torch.inference_mode()
def _execute_type_prompt(
    preflight: TypePromptPreflight,
    identity: tuple[int, int],
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    context = preflight.context
    generation = context.generation
    target_device = device or torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(generation.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(generation.seed)
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(False):
        components = load_frozen_components(
            context.paths.model_root, target_device, torch.float16
        )
        _, checkpoint = _validated_source_run(context.source_run, context.paths.runs_root)
        _, conditioner, adapter, _ = _load_p2_a1_checkpoint(
            checkpoint, config=context.source_experiment, device=target_device
        )
        model = P2InfillNoisePredictor(components.unet, conditioner, adapter).eval()
    guarded_modules = {
        "unet": components.unet,
        "vae": components.vae,
        "text_encoder": components.text_encoder,
        "conditioner": conditioner,
        "adapter": adapter,
    }
    inference_state = _capture_inference_state(guarded_modules)
    prompts = {
        defect_type: encode_prompt(components, prompt, target_device).detach().clone()
        for defect_type, prompt in context.probe.prompts.items()
    }
    normals = {item.sample_id: item for item in preflight.base.normals}
    masks_by_id = {item.sample_id: item for item in preflight.base.masks}
    normal_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    context_cache: dict[tuple[str, str, int], tuple[Any, ...]] = {}
    real_cache: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []
    for job in preflight.jobs:
        normal_item = normals[job.normal_id]
        mask_item = masks_by_id[job.mask_id]
        if job.normal_id not in normal_cache:
            loaded = load_training_image(normal_item, generation.image_size)
            normal_pixels, normal_image = _prepare_normal_tensors(
                loaded.pixels, target_device
            )
            normal_cache[job.normal_id] = (
                encode_mode(components.vae, normal_pixels).detach(),
                normal_image,
            )
        normal_latent, normal_image = normal_cache[job.normal_id]
        cache_key = (job.normal_id, job.mask_id, job.seed)
        if cache_key not in context_cache:
            source_mask = mask_item.mask.unsqueeze(0).to(
                device=target_device, dtype=torch.float32
            )
            latent_masks = build_latent_masks(source_mask, normal_latent.shape[2:])
            context_latent = build_masked_context(
                normal_latent,
                latent_masks.binary,
                torch.Generator(device=target_device).manual_seed(job.seed + 10_000_000),
            )
            noise = torch.randn(
                normal_latent.shape,
                generator=torch.Generator(device=target_device).manual_seed(job.seed),
                device=target_device,
                dtype=normal_latent.dtype,
            )
            context_cache[cache_key] = (
                source_mask,
                latent_masks.binary,
                context_latent,
                noise,
            )
        source_mask, binary_mask, context_latent, noise = context_cache[cache_key]
        scheduler = DDIMScheduler.from_config(components.scheduler.config)
        generated_latent = denoise_masked_infill(
            model=model,
            scheduler=scheduler,
            normal_latent=normal_latent,
            context=context_latent,
            binary_mask=binary_mask,
            prompt_embeds=prompts[job.defect_type],
            defect_type=job.defect_type,
            noise=noise,
            steps=generation.denoise_steps,
        )
        raw_image = decode_latent(components.vae, generated_latent).float()
        raw_metrics = compute_mask_metrics(raw_image, normal_image, source_mask)
        composite = raw_image * source_mask + normal_image * (1.0 - source_mask)
        composite_metrics = compute_mask_metrics(composite, normal_image, source_mask)
        image_relpath, mask_relpath, image_sha, mask_sha = _save_type_prompt_sample(
            output=context.output,
            pair_id=job.pair_id,
            image=composite,
            mask=source_mask,
        )
        references = context.output / "references"
        references.mkdir(parents=True, exist_ok=True)
        real_relpath = f"references/{job.pair_id}_real.png"
        real_path = context.output / real_relpath
        if real_path.exists():
            raise FileExistsError(f"real reference already exists for {job.pair_id}")
        if job.mask_id not in real_cache:
            real_loaded = load_training_image(mask_item, generation.image_size)
            real_cache[job.mask_id] = ((real_loaded.pixels.float() + 1.0) / 2.0).clamp(0, 1)
        _tensor_to_pil(real_cache[job.mask_id]).save(real_path)
        baseline_path = _contained_artifact(
            preflight.reference.path, job.baseline_image_relpath
        )
        records.append(
            {
                "schema_version": "p2_b1_type_prompt_record_v1",
                **asdict(job),
                "normal_prompt": context.probe.normal_prompt,
                "type_prompt_sha256": hashlib.sha256(job.type_prompt.encode()).hexdigest(),
                "prompt_embedding_sha256": _tensor_sha256(prompts[job.defect_type]),
                "normal_latent_sha256": _tensor_sha256(normal_latent),
                "context_latent_sha256": _tensor_sha256(context_latent),
                "noise_sha256": _tensor_sha256(noise),
                "image_relpath": image_relpath,
                "mask_relpath": mask_relpath,
                "real_reference_relpath": real_relpath,
                "image_sha256": image_sha,
                "mask_sha256": mask_sha,
                "real_reference_sha256": sha256_file(real_path),
                "generated_tensor_sha256": _tensor_sha256(composite),
                **asdict(raw_metrics),
                "composite_outside_l1": composite_metrics.outside_l1,
                "baseline_candidate_inside_l1": _baseline_candidate_inside_l1(
                    composite, baseline_path, source_mask
                ),
                "finite": True,
            }
        )
        print(
            json.dumps(
                {"stage": "type_prompt_generation", "records": len(records), "total": 16},
                sort_keys=True,
            ),
            flush=True,
        )
    _assert_inference_state_unchanged(guarded_modules, inference_state)
    _verify_type_prompt_integrity(preflight, identity)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = 0
    return {"records": tuple(records), "peak_vram_bytes": peak}


def _validate_prompt_records(
    records: Sequence[object],
    jobs: Sequence[TypePromptJob],
    config: TypePromptConfig,
) -> None:
    if len(records) != config.sample_count or len(jobs) != config.sample_count:
        raise ValueError("Type-Prompt records must match the exact 16-job schedule")
    prompt_embedding_by_type: dict[str, str] = {}
    normal_latent_by_id: dict[str, str] = {}
    for index, (value, job) in enumerate(zip(records, jobs, strict=True)):
        if not isinstance(value, dict) or set(value) != _PROMPT_RECORD_KEYS:
            raise ValueError(f"Type-Prompt record {index} has an invalid exact schema")
        expected_identity = asdict(job)
        if any(value.get(field) != expected for field, expected in expected_identity.items()):
            raise ValueError(f"Type-Prompt record {index} differs from its job identity")
        if value["schema_version"] != "p2_b1_type_prompt_record_v1":
            raise ValueError(f"Type-Prompt record {index} has an invalid schema version")
        if value["normal_prompt"] != config.normal_prompt:
            raise ValueError(f"Type-Prompt record {index} normal prompt mismatch")
        if value["type_prompt"] == config.normal_prompt:
            raise ValueError(f"Type-Prompt record {index} did not change the prompt")
        expected_prompt_sha = hashlib.sha256(job.type_prompt.encode()).hexdigest()
        if value["type_prompt_sha256"] != expected_prompt_sha:
            raise ValueError(f"Type-Prompt record {index} prompt hash mismatch")
        for field in _PROMPT_SHA_FIELDS:
            digest = value[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"Type-Prompt record {index} {field} must be a SHA256")
        if type(value["seed"]) is not int or value["finite"] is not True:
            raise ValueError(f"Type-Prompt record {index} seed/finite contract mismatch")
        for field in _PROMPT_METRICS:
            metric = value[field]
            if type(metric) is not float or not math.isfinite(metric) or metric < 0.0:
                raise ValueError(f"Type-Prompt record {index} {field} must be finite")
        if value["composite_outside_l1"] != 0.0:
            raise ValueError(f"Type-Prompt record {index} changed the mask exterior")
        if value["mask_sha256"] != job.baseline_mask_sha256:
            raise ValueError(f"Type-Prompt record {index} differs from the source mask")
        prompt_hash = str(value["prompt_embedding_sha256"])
        previous_prompt_hash = prompt_embedding_by_type.setdefault(
            job.defect_type, prompt_hash
        )
        if previous_prompt_hash != prompt_hash:
            raise ValueError(
                f"Type-Prompt record {index} has an inconsistent prompt embedding"
            )
        latent_hash = str(value["normal_latent_sha256"])
        previous_latent_hash = normal_latent_by_id.setdefault(job.normal_id, latent_hash)
        if previous_latent_hash != latent_hash:
            raise ValueError(
                f"Type-Prompt record {index} has an inconsistent normal latent"
            )
    if len(set(prompt_embedding_by_type.values())) != len(config.prompts):
        raise ValueError("Type-Prompt embeddings must be distinct across defect types")


def _build_blind_records(
    jobs: Sequence[TypePromptJob],
    records: Sequence[dict[str, Any]],
    assignments: Sequence[BlindAssignment],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    job_by_pair = {item.pair_id: item for item in jobs}
    record_by_pair = {str(item["pair_id"]): item for item in records}
    public: list[dict[str, Any]] = []
    reveal: list[dict[str, Any]] = []
    for assignment in assignments:
        job = job_by_pair[assignment.pair_id]
        record = record_by_pair[assignment.pair_id]
        source_sha = {
            "baseline": job.baseline_image_sha256,
            "type_prompt": record["image_sha256"],
        }
        public.append(
            {
                "schema_version": "p2_b1_blind_pair_v1",
                "pair_id": job.pair_id,
                "defect_type": job.defect_type,
                "seed": job.seed,
                "panel_a_relpath": f"comparisons/{job.pair_id}_A.png",
                "panel_b_relpath": f"comparisons/{job.pair_id}_B.png",
                "real_reference_relpath": f"comparisons/{job.pair_id}_reference.png",
                "mask_relpath": f"comparisons/{job.pair_id}_mask.png",
                "panel_a_sha256": source_sha[assignment.panel_a_source],
                "panel_b_sha256": source_sha[assignment.panel_b_source],
                "real_reference_sha256": record["real_reference_sha256"],
                "mask_sha256": record["mask_sha256"],
            }
        )
        reveal.append(
            {
                "schema_version": "p2_b1_type_prompt_reveal_v1",
                "pair_id": job.pair_id,
                "defect_type": job.defect_type,
                "type_prompt_panel": assignment.type_prompt_panel,
                "prompt": job.type_prompt,
                "prompt_sha256": hashlib.sha256(job.type_prompt.encode()).hexdigest(),
            }
        )
    return tuple(public), tuple(reveal)


def _materialize_blind_artifacts(
    preflight: TypePromptPreflight,
    records: Sequence[dict[str, Any]],
    public: Sequence[dict[str, Any]],
) -> None:
    output = preflight.context.output
    package = output / "blind_package"
    package.mkdir(parents=True, exist_ok=False)
    comparisons = package / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=False)
    record_by_pair = {str(record["pair_id"]): record for record in records}
    assignment_by_pair = {item.pair_id: item for item in preflight.assignments}
    for row in public:
        pair_id = str(row["pair_id"])
        record = record_by_pair[pair_id]
        job = next(item for item in preflight.jobs if item.pair_id == pair_id)
        assignment = assignment_by_pair[pair_id]
        sources = {
            "baseline": _contained_artifact(
                preflight.reference.path, job.baseline_image_relpath
            ),
            "type_prompt": _contained_artifact(output, str(record["image_relpath"])),
        }
        copies = (
            (sources[assignment.panel_a_source], str(row["panel_a_relpath"])),
            (sources[assignment.panel_b_source], str(row["panel_b_relpath"])),
            (
                _contained_output_artifact(
                    output, str(record["real_reference_relpath"]), "references"
                ),
                str(row["real_reference_relpath"]),
            ),
            (
                _contained_artifact(output, str(record["mask_relpath"])),
                str(row["mask_relpath"]),
            ),
        )
        for source, relpath in copies:
            destination = package / relpath
            if destination.exists():
                raise FileExistsError(f"blind artifact already exists: {relpath}")
            shutil.copyfile(source, destination)


def _validate_real_reference_artifacts(
    preflight: TypePromptPreflight | Any,
    records: Sequence[dict[str, Any]],
) -> None:
    output = preflight.context.output
    image_size = preflight.context.generation.image_size
    references = _real_child(output, "references", directory=True)
    masks_by_id = {item.sample_id: item for item in preflight.base.masks}
    expected_paths: set[Path] = set()
    expected_arrays: dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
    for index, record in enumerate(records):
        mask_item = masks_by_id.get(record["mask_id"])
        if mask_item is None:
            raise ValueError(f"real reference record {index} uses an unknown mask")
        path = _contained_output_artifact(
            output, str(record["real_reference_relpath"]), "references"
        )
        if path.parent != references:
            raise ValueError(f"real reference record {index} is outside references")
        expected_paths.add(path)
        if sha256_file(path) != record["real_reference_sha256"]:
            raise ValueError(f"real reference hash mismatch at record {index}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (image_size, image_size):
                raise ValueError(f"real reference contract mismatch at record {index}")
            actual = np.asarray(image, dtype=np.uint8)
        if mask_item.sample_id not in expected_arrays:
            loaded = load_training_image(mask_item, image_size)
            transformed = ((loaded.pixels.float() + 1.0) / 2.0).clamp(0, 1)
            expected_arrays[mask_item.sample_id] = np.asarray(
                _tensor_to_pil(transformed), dtype=np.uint8
            )
        if not np.array_equal(actual, expected_arrays[mask_item.sample_id]):
            raise ValueError(f"real reference differs from heldout anomaly at record {index}")
    _validate_exact_regular_children(references, {path.name for path in expected_paths})


def _validate_prompt_artifacts(
    preflight: TypePromptPreflight,
    records: Sequence[dict[str, Any]],
    public: Sequence[dict[str, Any]],
) -> None:
    wrapper = SimpleNamespace(
        context=SimpleNamespace(
            output=preflight.context.output,
            experiment=preflight.context.generation,
        ),
        normals=preflight.base.normals,
        masks=preflight.base.masks,
    )
    _validate_sample_artifacts(wrapper, tuple(records))
    _validate_real_reference_artifacts(preflight, records)
    if len(public) != 16:
        raise ValueError("blind manifest must contain exactly 16 rows")
    package = _real_child(preflight.context.output, "blind_package", directory=True)
    comparisons = _real_child(package, "comparisons", directory=True)
    manifest = _real_child(package, "blind_manifest.jsonl", directory=False)
    if _strict_jsonl_objects(manifest) != tuple(public):
        raise ValueError("blind package manifest differs from the public schedule")
    if {path.name for path in package.iterdir()} != {
        "comparisons",
        "blind_manifest.jsonl",
    }:
        raise ValueError("blind package must contain only comparisons and public manifest")
    expected_names: set[str] = set()
    for index, row in enumerate(public):
        for rel_field, sha_field in (
            ("panel_a_relpath", "panel_a_sha256"),
            ("panel_b_relpath", "panel_b_sha256"),
            ("real_reference_relpath", "real_reference_sha256"),
            ("mask_relpath", "mask_sha256"),
        ):
            path = _contained_output_artifact(
                package, str(row[rel_field]), "comparisons"
            )
            expected_names.add(path.name)
            if sha256_file(path) != row[sha_field]:
                raise ValueError(f"blind artifact hash mismatch at row {index}")
        for field in ("panel_a_relpath", "panel_b_relpath", "real_reference_relpath"):
            with Image.open(package / str(row[field])) as image:
                image.load()
                if image.mode != "RGB" or image.size != (512, 512):
                    raise ValueError(f"blind RGB artifact contract mismatch at row {index}")
        with Image.open(package / str(row["mask_relpath"])) as mask:
            mask.load()
            if mask.mode != "L" or mask.size != (512, 512):
                raise ValueError(f"blind mask contract mismatch at row {index}")
    _validate_exact_regular_children(comparisons, expected_names)


def run_type_prompt_probe(context: TypePromptContext | Any) -> int:
    try:
        preflight = validate_type_prompt_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_type_prompt(preflight, identity)
        records = tuple(result["records"])
        _validate_prompt_records(records, preflight.jobs, context.probe)
        public, reveal = _build_blind_records(
            preflight.jobs, records, preflight.assignments
        )
        _materialize_blind_artifacts(preflight, records, public)
        _write_jsonl_atomic(
            context.output / "blind_package" / "blind_manifest.jsonl", public
        )
        _validate_prompt_artifacts(preflight, records, public)
        head_end = _verify_type_prompt_integrity(preflight, identity)
        private = context.output / "private"
        private.mkdir(parents=True, exist_ok=False)
        _write_jsonl_atomic(private / "records.jsonl", records)
        _verify_type_prompt_integrity(preflight, identity)
        write_json_atomic(
            private / "reveal_key.json",
            {
                "schema_version": "p2_b1_type_prompt_reveal_key_v1",
                "blind_seed": context.probe.blind_seed,
                "records": list(reveal),
            },
        )
        _verify_type_prompt_integrity(preflight, identity)
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": "p2_b1_type_prompt_provenance_v1",
                "git_head_start": preflight.git_head,
                "git_head_end": head_end,
                "git_dirty": False,
                "training_performed": False,
                "checkpoint_written": False,
                "evaluation_mode": "inference_only_diagnostic",
                "input_hashes": preflight.input_hashes,
                "source_snapshot_before": preflight.source_snapshot,
                "source_snapshot_after": preflight.source_snapshot,
                "model_snapshot_before": preflight.model_snapshot,
                "model_snapshot_after": preflight.model_snapshot,
                "checkpoint_snapshot_before": preflight.checkpoint_snapshot,
                "checkpoint_snapshot_after": preflight.checkpoint_snapshot,
                "reference_snapshot_before": preflight.reference_snapshot,
                "reference_snapshot_after": preflight.reference_snapshot,
            }
        )
        report = {
            "schema_version": "p2_b1_type_prompt_report_v1",
            "status": "complete",
            "reason": None,
            "visual_status": "pending_human_review",
            "training_performed": False,
            "checkpoint_written": False,
            "evaluation_mode": "inference_only_diagnostic",
            "record_count": len(records),
            "blind_pair_count": len(public),
            "blind_package_relpath": "blind_package",
            "peak_vram_bytes": result["peak_vram_bytes"],
            "input_hashes": preflight.input_hashes,
        }
        write_json_atomic(context.output / "provenance.json", provenance)
        _verify_type_prompt_integrity(preflight, identity)
        write_json_atomic(context.output / "report.json", report)
        _verify_type_prompt_integrity(preflight, identity)
        _validate_prompt_artifacts(preflight, records, public)
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": "p2_b1_type_prompt_report_v1",
            "status": "failed",
            "reason": "runtime_or_integrity_failure",
        }
        try:
            _verify_output_identity(preflight.base.base, identity)
        except (OSError, RuntimeError):
            pass
        else:
            try:
                _cleanup_type_prompt_partial(context.output)
                write_json_atomic(context.output / "report.json", failed)
            except OSError:
                pass
        print(json.dumps(failed, sort_keys=True))
        return 1


def build_type_prompt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the P2-B1 Type-Prompt probe")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_type_prompt_parser().parse_args(argv)
    paths_path = Path(args.paths)
    generation_path = Path(args.generation)
    probe_path = Path(args.probe)
    source_experiment_path = Path(args.source_experiment)
    try:
        return run_type_prompt_probe(
            TypePromptContext(
                paths=load_config(paths_path),
                generation=load_p2_b1_config(generation_path),
                probe=load_type_prompt_config(probe_path),
                source_experiment=load_p2_a1_config(source_experiment_path),
                manifest=Path(args.manifest),
                lock=Path(args.lock),
                source_run=Path(args.source_run),
                baseline_run=Path(args.baseline_run),
                reference_run=Path(args.reference_run),
                output=Path(args.output),
                config_paths={
                    "generation": generation_path,
                    "paths": paths_path,
                    "probe": probe_path,
                    "source_experiment": source_experiment_path,
                },
                cwd=Path.cwd(),
            )
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1


__all__ = [
    "TypePromptContext",
    "TypePromptPreflight",
    "_build_blind_records",
    "_execute_type_prompt",
    "_validate_prompt_artifacts",
    "_validate_prompt_records",
    "_validate_real_reference_artifacts",
    "_validated_reference_run",
    "build_type_prompt_parser",
    "run_type_prompt_probe",
    "validate_type_prompt_preflight",
]

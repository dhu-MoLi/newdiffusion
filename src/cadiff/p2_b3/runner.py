from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import stat
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from safetensors.torch import load_file as load_safetensors

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
)
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_a1.runner import _load_p2_a1_checkpoint
from cadiff.p2_b0.generation import denoise_masked_infill
from cadiff.p2_b1.config import P2B1Config, load_p2_b1_config
from cadiff.p2_b1.generation import build_heldout_generation_schedule
from cadiff.p2_b1.runner import (
    P2B1Context,
    _prepare_normal_tensors,
    _validate_generation_records,
    _validate_sample_artifacts,
    _verify_p2_b1_integrity,
    validate_p2_b1_preflight,
)
from cadiff.p2_b1.type_prompt_runner import (
    _assert_inference_state_unchanged,
    _bind_reference_to_current_preflight,
    _capture_inference_state,
    _validated_reference_run,
)
from cadiff.p2_b3.config import P2B3Config, load_p2_b3_config
from cadiff.p2_b3.contracts import (
    CheckpointProbeJob,
    build_blind_assignments,
    build_blind_records,
    select_checkpoint_probe_jobs,
    validate_paired_records,
)
from cadiff.p2_b3.validator import (
    _assert_named_directory,
    _open_directory_at,
    _open_root_authority,
    _publish_bytes_at,
    _write_json_at,
    _write_jsonl_at,
    materialize_blind_package,
    validate_p2_b3_artifact_tree,
)
from cadiff.provenance import sha256_file

_STAGE_SCHEMA = "p2_b3_stage_event_v1"
_STAGES = frozenset(
    {
        "load_components",
        "encode_prompts",
        "load_A",
        "load_B",
        "capture_state",
        "prepare_job",
        "sample_A",
        "sample_B",
        "validate",
        "publish",
        "finalize",
    }
)


def _emit_stage_event(
    stage: str, event: str, error: BaseException | None = None
) -> None:
    if stage not in _STAGES or event not in {"start", "ok", "error"}:
        raise ValueError("P2-B3 stage event is invalid")
    payload = {
        "schema_version": _STAGE_SCHEMA,
        "stage": stage,
        "event": event,
    }
    if error is not None:
        exception_type = type(error).__name__
        if (
            len(exception_type) > 64
            or not exception_type.isascii()
            or not exception_type.replace("_", "").isalnum()
        ):
            exception_type = "Exception"
        payload["exception_type"] = exception_type
        payload["reason"] = f"{stage}_failed"
    try:
        print(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )
    except OSError:
        pass


@contextmanager
def _stage(stage: str):
    _emit_stage_event(stage, "start")
    try:
        yield
    except Exception as error:
        _emit_stage_event(stage, "error", error)
        raise
    else:
        _emit_stage_event(stage, "ok")

_CHECKPOINT_FILES = {
    "adapter.safetensors",
    "conditioner.safetensors",
    "metadata.json",
}
_OWNED_TOP_LEVEL = {
    "paired",
    "blind_package",
    "private",
    "records.jsonl",
    "provenance.json",
    "report.json",
}


@dataclass(frozen=True)
class P2B3Context:
    paths: ProjectConfig | Any
    generation: P2B1Config | Any
    experiment: P2B3Config
    source_experiment: P2A1Config | Any
    manifest: Path
    lock: Path
    source_run: Path
    baseline_run: Path
    reference_run: Path
    baseline_checkpoint: Path
    candidate_checkpoint: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class AuthenticatedCheckpoint:
    role: str
    path: Path
    hashes: Mapping[str, str]
    metadata: Mapping[str, Any]
    snapshot: Mapping[str, Mapping[str, int | str]]


@dataclass(frozen=True)
class P2B3Preflight:
    context: P2B3Context
    base: Any
    jobs: tuple[CheckpointProbeJob, ...]
    baseline_checkpoint: AuthenticatedCheckpoint
    candidate_checkpoint: AuthenticatedCheckpoint
    git_head: str
    input_hashes: Mapping[str, str]
    reference_path: Path
    reference_snapshot: Mapping[str, Mapping[str, int | str]]
    candidate_run_snapshot: Mapping[str, Mapping[str, int | str]]


@dataclass(frozen=True)
class _InferenceRuntime:
    components: Any
    prompts: Mapping[str, torch.Tensor]
    baseline_model: torch.nn.Module
    candidate_model: torch.nn.Module
    baseline_conditioner: torch.nn.Module
    candidate_conditioner: torch.nn.Module
    baseline_adapter: torch.nn.Module
    candidate_adapter: torch.nn.Module
    baseline_state_hashes: Mapping[str, str]
    candidate_state_hashes: Mapping[str, str]
    guarded_modules: Mapping[str, torch.nn.Module]
    inference_state: Mapping[str, tuple[int, int, str, str, tuple[int, ...]]]


@dataclass
class _PinnedOutput:
    path: Path
    runs_root: Path
    fd: int
    device: int
    inode: int
    proc_path: Path


class _InstrumentedScheduler:
    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self.set_timesteps_calls = 0
        self.step_timesteps: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        self.set_timesteps_calls += 1
        self._scheduler.set_timesteps(steps, device=device)

    def step(
        self,
        prediction: torch.Tensor,
        timestep: torch.Tensor,
        current: torch.Tensor,
        *,
        eta: float,
    ) -> Any:
        self.step_timesteps.append(int(timestep.item()))
        return self._scheduler.step(prediction, timestep, current, eta=eta)


def _reject_lexical_symlinks(root: Path, candidate: Path) -> None:
    lexical_root = Path(os.path.abspath(root))
    lexical_candidate = Path(os.path.abspath(candidate))
    try:
        relative = lexical_candidate.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError("checkpoint lexical path escapes runs_root") from error
    current = lexical_root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        try:
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise ValueError(f"checkpoint lexical node is unavailable: {current}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"checkpoint lexical ancestor is a symlink: {current}")


def _checkpoint_snapshot(checkpoint: Path) -> dict[str, dict[str, int | str]]:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("checkpoint must be a real directory")
    children = {child.name for child in checkpoint.iterdir()}
    if children != _CHECKPOINT_FILES:
        raise ValueError("checkpoint must contain exactly three accepted files")
    snapshot: dict[str, dict[str, int | str]] = {}
    for name in sorted(_CHECKPOINT_FILES):
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint child must be a regular file: {name}")
        snapshot[name] = _snapshot(path)
    return snapshot


def _authenticate_checkpoint(
    checkpoint: Path,
    runs_root: Path,
    *,
    role: str,
    config: P2B3Config,
    source_config: P2A1Config,
) -> AuthenticatedCheckpoint:
    if role not in {"baseline", "candidate"}:
        raise ValueError("checkpoint role must be baseline or candidate")
    _reject_lexical_symlinks(runs_root, checkpoint)
    lexical = Path(os.path.abspath(checkpoint))
    expected_relpath = getattr(config, f"{role}_checkpoint_relpath")
    expected_lexical = Path(os.path.abspath(runs_root / expected_relpath))
    if lexical != expected_lexical:
        raise ValueError(f"{role} checkpoint does not match the exact lexical path")
    root = runs_root.resolve(strict=True)
    resolved = checkpoint.resolve(strict=True)
    if resolved != (root / expected_relpath).resolve(strict=True):
        raise ValueError(f"{role} checkpoint does not match the frozen resolved path")
    if not resolved.is_relative_to(root):
        raise ValueError(f"{role} checkpoint escapes runs_root")
    snapshot = _checkpoint_snapshot(resolved)
    expected_hashes = dict(getattr(config, f"{role}_checkpoint_hashes"))
    for name, expected_hash in expected_hashes.items():
        if sha256_file(resolved / name) != expected_hash:
            raise ValueError(f"{role} checkpoint hash mismatch for {name}")
    metadata = _strict_json_object(resolved / "metadata.json")
    expected_source_config = json.loads(
        json.dumps(asdict(source_config), ensure_ascii=False, sort_keys=True)
    )
    if metadata.get("schema_version") != "p2_a1_factorized_checkpoint_v1":
        raise ValueError(f"{role} checkpoint schema is incompatible")
    if metadata.get("config") != expected_source_config:
        raise ValueError(f"{role} checkpoint config is incompatible with P2-A1")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != {
        "adapter.safetensors", "conditioner.safetensors"
    }:
        raise ValueError(f"{role} checkpoint metadata files schema is incompatible")
    for name in ("adapter.safetensors", "conditioner.safetensors"):
        item = files[name]
        if not isinstance(item, dict) or set(item) != {"sha256", "size"}:
            raise ValueError(f"{role} checkpoint metadata file entry is invalid")
        if item["sha256"] != expected_hashes[name]:
            raise ValueError(f"{role} checkpoint metadata hash mismatch for {name}")
        if item["size"] != snapshot[name]["size"]:
            raise ValueError(f"{role} checkpoint metadata size mismatch for {name}")
    vocabulary = metadata.get("vocabulary_sha256")
    git_head = metadata.get("git_head")
    if not isinstance(vocabulary, str) or len(vocabulary) != 64:
        raise ValueError(f"{role} checkpoint vocabulary hash is invalid")
    if not isinstance(git_head, str) or len(git_head) != 40:
        raise ValueError(f"{role} checkpoint Git identity is invalid")
    return AuthenticatedCheckpoint(
        role=role,
        path=lexical,
        hashes=MappingProxyType(expected_hashes),
        metadata=MappingProxyType(metadata),
        snapshot=MappingProxyType(snapshot),
    )


def _authenticate_candidate_run(
    candidate: AuthenticatedCheckpoint,
    baseline: AuthenticatedCheckpoint,
    config: P2B3Config,
) -> Mapping[str, Mapping[str, int | str]]:
    run = candidate.path.parent
    artifacts: dict[str, Path] = {}
    snapshot: dict[str, Mapping[str, int | str]] = {}
    for name in ("report.json", "provenance.json"):
        path = run / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"accepted candidate run {name} must be a regular file")
        if sha256_file(path) != config.candidate_run_hashes[name]:
            raise ValueError(f"accepted candidate run {name} hash mismatch")
        artifacts[name] = path
        snapshot[name] = _snapshot(path)
    report = _strict_json_object(artifacts["report.json"])
    provenance = _strict_json_object(artifacts["provenance.json"])
    if config.schema_version == "p2_b6_texture_checkpoint_paired_probe_v1":
        expected_structure = {
            "erosion_kernel": 5,
            "space": "decoded_predicted_x0_rgb",
            "types": ["cut", "hole", "print"],
            "weight": 0.1,
        }
        if (
            report.get("schema_version")
            != "p2_b5_type_selective_texture_report_v1"
            or report.get("status") != "complete"
            or report.get("reason") is not None
            or report.get("training_performed") is not True
            or report.get("checkpoint_written") is not True
            or report.get("evaluation_mode") != "diagnostic_only"
            or report.get("steps") != 200
            or report.get("structure") != expected_structure
            or report.get("checkpoint") != dict(candidate.metadata)
        ):
            raise ValueError(
                "candidate checkpoint is not bound to the accepted P2-B5 report"
            )
        if (
            provenance.get("schema_version")
            != "p2_b5_type_selective_texture_provenance_v1"
            or provenance.get("training_performed") is not True
            or provenance.get("checkpoint_written") is not True
            or provenance.get("git_dirty") is not False
            or provenance.get("git_head_start") != candidate.metadata.get("git_head")
            or provenance.get("git_head_end") != candidate.metadata.get("git_head")
            or provenance.get("structure") != expected_structure
        ):
            raise ValueError(
                "candidate checkpoint is not bound to accepted P2-B5 provenance"
            )
        expected_source_lineage = {
            "checkpoint": str(baseline.path),
            "hashes": dict(baseline.hashes),
            "metadata": dict(baseline.metadata),
        }
        if any(
            document.get("source_checkpoint_lineage") != expected_source_lineage
            for document in (report, provenance)
        ):
            raise ValueError("accepted P2-B5 source checkpoint lineage is invalid")
        return MappingProxyType(snapshot)
    if config.schema_version != "p2_b3_checkpoint_paired_probe_v1":
        raise ValueError("candidate run profile is unsupported")
    if (
        report.get("schema_version") != "p2_b2_type_prompt_training_report_v1"
        or report.get("status") != "complete"
        or report.get("reason") is not None
        or report.get("training_performed") is not True
        or report.get("checkpoint_written") is not True
        or report.get("evaluation_mode") != "diagnostic_only"
    ):
        raise ValueError("candidate checkpoint is not bound to the accepted P2-B2 report")
    if (
        provenance.get("schema_version")
        != "p2_b2_type_prompt_training_provenance_v1"
        or provenance.get("training_performed") is not True
        or provenance.get("checkpoint_written") is not True
        or provenance.get("git_dirty") is not False
        or provenance.get("git_head_start") != candidate.metadata.get("git_head")
        or provenance.get("git_head_end") != candidate.metadata.get("git_head")
    ):
        raise ValueError("candidate checkpoint is not bound to accepted P2-B2 provenance")
    current_snapshot = dict(candidate.snapshot)
    for document in (report, provenance):
        if (
            document.get("new_checkpoint_snapshot_before") != current_snapshot
            or document.get("new_checkpoint_snapshot_after") != current_snapshot
        ):
            raise ValueError("accepted P2-B2 checkpoint snapshot differs from candidate")
        source_lineage = document.get("source_checkpoint_lineage")
        if (
            not isinstance(source_lineage, dict)
            or source_lineage.get("checkpoint") != str(baseline.path)
            or source_lineage.get("hashes") != dict(baseline.hashes)
            or source_lineage.get("metadata") != dict(baseline.metadata)
        ):
            raise ValueError("accepted P2-B2 source checkpoint lineage is invalid")
    if report.get("checkpoint") != dict(candidate.metadata):
        raise ValueError("accepted P2-B2 report checkpoint metadata differs from candidate")
    expected_prompts: dict[str, dict[str, str]] = {}
    reported_prompts = report.get("type_prompts")
    if not isinstance(reported_prompts, dict) or set(reported_prompts) != set(config.prompts):
        raise ValueError("accepted P2-B2 prompt lineage schema is invalid")
    for label, text in config.prompts.items():
        item = reported_prompts[label]
        if not isinstance(item, dict) or set(item) != {
            "text", "prompt_sha256", "prompt_embedding_sha256"
        }:
            raise ValueError("accepted P2-B2 prompt lineage entry is invalid")
        expected = {
            "text": text,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_embedding_sha256": item["prompt_embedding_sha256"],
        }
        embedding_sha = item["prompt_embedding_sha256"]
        if (
            not isinstance(embedding_sha, str)
            or len(embedding_sha) != 64
            or any(character not in "0123456789abcdef" for character in embedding_sha)
        ):
            raise ValueError("accepted P2-B2 prompt embedding hash is invalid")
        if item != expected:
            raise ValueError("accepted P2-B2 prompt text/hash lineage is invalid")
        expected_prompts[label] = expected
    if provenance.get("type_prompts") != expected_prompts:
        raise ValueError("accepted P2-B2 report/provenance prompt lineages differ")
    return MappingProxyType(snapshot)


def validate_p2_b3_preflight(context: P2B3Context) -> P2B3Preflight:
    if not isinstance(context, P2B3Context):
        raise ValueError("context must be a P2B3Context")
    if not isinstance(context.experiment, P2B3Config):
        raise ValueError("experiment must be a P2B3Config")
    if not isinstance(context.generation, P2B1Config):
        raise ValueError("generation must be a P2B1Config")
    if not isinstance(context.source_experiment, P2A1Config):
        raise ValueError("source_experiment must be a P2A1Config")
    if (
        context.experiment.category != context.generation.category
        or context.experiment.defect_types != context.generation.defect_types
        or context.experiment.denoise_steps != context.generation.denoise_steps
    ):
        raise ValueError("P2-B3 and P2-B1 generation contracts are incompatible")
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
    baseline = _authenticate_checkpoint(
        context.baseline_checkpoint,
        context.paths.runs_root,
        role="baseline",
        config=context.experiment,
        source_config=context.source_experiment,
    )
    candidate = _authenticate_checkpoint(
        context.candidate_checkpoint,
        context.paths.runs_root,
        role="candidate",
        config=context.experiment,
        source_config=context.source_experiment,
    )
    if baseline.path == candidate.path or baseline.hashes == candidate.hashes:
        raise ValueError("checkpoint probe requires two distinct authenticated checkpoints")
    if baseline.metadata.get("vocabulary_sha256") != candidate.metadata.get(
        "vocabulary_sha256"
    ):
        raise ValueError("checkpoint vocabularies differ")
    if (
        context.experiment.schema_version == "p2_b3_checkpoint_paired_probe_v1"
        and dict(base.checkpoint_snapshot) != dict(baseline.snapshot)
    ):
        raise ValueError("P2-B1 reconstructed baseline checkpoint snapshot differs")
    candidate_run_snapshot = _authenticate_candidate_run(
        candidate, baseline, context.experiment
    )
    reference_path = Path(os.path.abspath(context.reference_run))
    expected_reference = Path(
        os.path.abspath(context.paths.runs_root / context.experiment.reference_run_relpath)
    )
    if reference_path != expected_reference:
        raise ValueError("reference_run does not match the exact accepted P2-B1 path")
    reference = _validated_reference_run(
        reference_path,
        context.paths.runs_root,
        SimpleNamespace(
            reference_hashes=context.experiment.reference_hashes,
            reference_git_head=context.experiment.reference_git_head,
        ),
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
    _validate_sample_artifacts(
        SimpleNamespace(
            context=SimpleNamespace(
                output=reference.path, experiment=context.generation
            ),
            normals=base.normals,
            masks=base.masks,
        ),
        reference.records,
    )
    _bind_reference_to_current_preflight(reference, base)
    jobs = select_checkpoint_probe_jobs(reference.records, context.experiment)
    reference_snapshot: dict[str, Mapping[str, int | str]] = dict(reference.snapshot)
    for job in jobs:
        for relpath in (job.accepted_image_relpath, job.accepted_mask_relpath):
            path = reference.path / relpath
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(
                reference.path
            ):
                raise ValueError("selected P2-B1 reference artifact is invalid")
            reference_snapshot[str(path.resolve())] = _snapshot(path)
    input_hashes = dict(base.input_hashes)
    for name, path in context.config_paths.items():
        input_hashes[f"config/{name}"] = sha256_file(path)
    for name, digest in context.experiment.reference_hashes.items():
        input_hashes[f"reference_run/{name}"] = digest
    for role in ("baseline", "candidate"):
        for name, digest in getattr(context.experiment, f"{role}_checkpoint_hashes").items():
            input_hashes[f"{role}_checkpoint/{name}"] = digest
    return P2B3Preflight(
        context=context,
        base=base,
        jobs=jobs,
        baseline_checkpoint=baseline,
        candidate_checkpoint=candidate,
        git_head=base.git_head,
        input_hashes=MappingProxyType(input_hashes),
        reference_path=reference.path,
        reference_snapshot=MappingProxyType(reference_snapshot),
        candidate_run_snapshot=candidate_run_snapshot,
    )


def _canonical_sha(value: object, name: str) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ensure_ascii=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _freeze_inference_module(module: torch.nn.Module, name: str) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"{name} must be a torch module")
    module.eval()
    module.requires_grad_(False)
    if module.training or any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in module.parameters()
    ):
        raise RuntimeError(f"{name} is not frozen eval inference state")
    return module


def _prepare_checkpoint_inference_module(
    module: torch.nn.Module, name: str
) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"{name} must be a torch module")
    module.eval()
    if module.training or any(
        parameter.dtype is not torch.float32
        or not parameter.requires_grad
        or parameter.grad is not None
        for parameter in module.parameters()
    ):
        raise RuntimeError(f"{name} must retain clean FP32 master parameters")
    return module


def _assert_disjoint_checkpoint_models(
    baseline_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    baseline_conditioner: torch.nn.Module,
    candidate_conditioner: torch.nn.Module,
    baseline_adapter: torch.nn.Module,
    candidate_adapter: torch.nn.Module,
) -> None:
    modules = (
        baseline_model,
        candidate_model,
        baseline_conditioner,
        candidate_conditioner,
        baseline_adapter,
        candidate_adapter,
    )
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("checkpoint model, conditioner and adapter must be distinct objects")
    if (
        not hasattr(baseline_model, "unet")
        or baseline_model.unet is not candidate_model.unet
        or baseline_model.conditioner is not baseline_conditioner
        or candidate_model.conditioner is not candidate_conditioner
        or baseline_model.adapter is not baseline_adapter
        or candidate_model.adapter is not candidate_adapter
    ):
        raise ValueError("checkpoint wrappers must share only the exact frozen U-Net")
    shared_unet = baseline_model.unet
    if shared_unet.training or any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in shared_unet.parameters()
    ):
        raise ValueError("shared frozen U-Net must remain eval with no gradients")
    baseline_parameters = {
        id(parameter)
        for module in (baseline_conditioner, baseline_adapter)
        for parameter in module.parameters()
    }
    candidate_parameters = {
        id(parameter)
        for module in (candidate_conditioner, candidate_adapter)
        for parameter in module.parameters()
    }
    if not baseline_parameters or not candidate_parameters:
        raise ValueError("checkpoint models must expose authenticated parameters")
    if baseline_parameters & candidate_parameters:
        raise ValueError("checkpoint model parameter identities must be disjoint")
    for module in (
        baseline_conditioner,
        candidate_conditioner,
        baseline_adapter,
        candidate_adapter,
    ):
        if module.training or any(
            parameter.dtype is not torch.float32
            or not parameter.requires_grad
            or parameter.grad is not None
            for parameter in module.parameters()
        ):
            raise ValueError(
                "checkpoint modules must retain clean FP32 master parameters"
            )


def _canonical_tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    entries: list[dict[str, object]] = []
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("canonical state must map names to tensors")
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "tensor_sha256": _tensor_sha256(tensor),
            }
        )
    if not entries:
        raise ValueError("canonical state must not be empty")
    return _canonical_sha(entries, "canonical tensor state")


def _canonical_module_state_sha256(module: torch.nn.Module) -> str:
    if not isinstance(module, torch.nn.Module):
        raise ValueError("canonical module state requires a torch module")
    return _canonical_tensor_state_sha256(module.state_dict())


def _canonical_checkpoint_state_sha256(checkpoint: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for role, name in (
        ("conditioner", "conditioner.safetensors"),
        ("adapter", "adapter.safetensors"),
    ):
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint {name} must remain a regular file")
        result[role] = _canonical_tensor_state_sha256(load_safetensors(path, device="cpu"))
    return result


def _assert_checkpoint_state_binding(
    authenticated: AuthenticatedCheckpoint | Any,
    conditioner: torch.nn.Module,
    adapter: torch.nn.Module,
) -> dict[str, str]:
    expected = _canonical_checkpoint_state_sha256(authenticated.path)
    actual = {
        "conditioner": _canonical_module_state_sha256(conditioner),
        "adapter": _canonical_module_state_sha256(adapter),
    }
    for name in ("conditioner", "adapter"):
        if actual[name] != expected[name]:
            raise ValueError(
                f"{authenticated.role} loaded {name} state hash is not bound to checkpoint"
            )
    return actual


def _load_checkpoint_model(
    authenticated: AuthenticatedCheckpoint | Any,
    *,
    source_config: P2A1Config | Any,
    unet: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Mapping[str, str]]:
    _, conditioner, adapter, metadata = _load_p2_a1_checkpoint(
        authenticated.path, config=source_config, device=device
    )
    if metadata != authenticated.metadata:
        raise ValueError(f"{authenticated.role} loaded checkpoint metadata mismatch")
    conditioner = _prepare_checkpoint_inference_module(
        conditioner, f"{authenticated.role} conditioner"
    )
    adapter = _prepare_checkpoint_inference_module(
        adapter, f"{authenticated.role} adapter"
    )
    state_hashes = _assert_checkpoint_state_binding(authenticated, conditioner, adapter)
    model = P2InfillNoisePredictor(unet, conditioner, adapter).eval()
    if model.training:
        raise RuntimeError(f"{authenticated.role} predictor is not eval")
    return model, conditioner, adapter, MappingProxyType(state_hashes)


def _create_paired_output(authority: Path | int) -> tuple[int, int]:
    with ExitStack() as stack:
        root_fd = _open_root_authority(stack, authority, "paired output root")
        try:
            os.mkdir("paired", dir_fd=root_fd)
        except OSError as error:
            raise ValueError("paired directory cannot be created exclusively") from error
        paired_fd = _open_directory_at(stack, root_fd, "paired", "paired output")
        identity = os.fstat(paired_fd)
        _assert_named_directory(root_fd, "paired", paired_fd, "paired output")
        return identity.st_dev, identity.st_ino


def _publish_paired_png(
    authority: Path | int,
    paired_identity: tuple[int, int],
    name: str,
    image: Image.Image,
) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    with ExitStack() as stack:
        root_fd = _open_root_authority(stack, authority, "paired output root")
        paired_fd = _open_directory_at(stack, root_fd, "paired", "paired output")
        opened = os.fstat(paired_fd)
        if (opened.st_dev, opened.st_ino) != paired_identity:
            raise ValueError("paired output descriptor changed")
        _publish_bytes_at(paired_fd, name, payload)
        _assert_named_directory(root_fd, "paired", paired_fd, "paired output")
    return hashlib.sha256(payload).hexdigest()


def _save_rgb_atomic(
    authority: Path | int,
    paired_identity: tuple[int, int],
    name: str,
    tensor: torch.Tensor,
) -> str:
    return _publish_paired_png(
        authority, paired_identity, name, _tensor_to_pil(tensor[0])
    )


def _save_mask_atomic(
    authority: Path | int,
    paired_identity: tuple[int, int],
    name: str,
    tensor: torch.Tensor,
) -> str:
    array = (tensor[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    return _publish_paired_png(
        authority, paired_identity, name, Image.fromarray(array)
    )


def _decode_and_composite(
    vae: torch.nn.Module,
    prediction: torch.Tensor,
    source_mask: torch.Tensor,
    normal_image: torch.Tensor,
) -> torch.Tensor:
    raw = decode_latent(vae, prediction).float()
    return raw * source_mask + normal_image * (1.0 - source_mask)


@contextmanager
def _forbid_training_apis():
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("P2-B3 is inference-only; training/checkpoint APIs are forbidden")

    original_backward = torch.Tensor.backward
    original_autograd_backward = torch.autograd.backward
    original_save = torch.save
    optimizer_names = (
        "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "LBFGS",
        "NAdam", "RAdam", "RMSprop", "Rprop", "SGD", "SparseAdam",
    )
    original_optimizers = {
        name: getattr(torch.optim, name)
        for name in optimizer_names
        if hasattr(torch.optim, name)
    }
    original_scalers = []
    for namespace in (torch.amp, torch.cuda.amp):
        if hasattr(namespace, "GradScaler"):
            original_scalers.append((namespace, namespace.GradScaler))
    try:
        torch.Tensor.backward = forbidden  # type: ignore[method-assign]
        torch.autograd.backward = forbidden  # type: ignore[assignment]
        torch.save = forbidden  # type: ignore[assignment]
        for name in original_optimizers:
            setattr(torch.optim, name, forbidden)
        for namespace, _original in original_scalers:
            namespace.GradScaler = forbidden
        yield
    finally:
        torch.Tensor.backward = original_backward  # type: ignore[method-assign]
        torch.autograd.backward = original_autograd_backward
        torch.save = original_save
        for name, original in original_optimizers.items():
            setattr(torch.optim, name, original)
        for namespace, original in original_scalers:
            namespace.GradScaler = original


def _load_inference_runtime(
    preflight: P2B3Preflight | Any,
    target_device: torch.device,
) -> _InferenceRuntime:
    context = preflight.context
    config = context.experiment
    with _stage("load_components"):
        with torch.inference_mode(False), torch.no_grad():
            components = load_frozen_components(
                context.paths.model_root, target_device, torch.float16
            )
            base_modules = {
                "unet": _freeze_inference_module(components.unet, "unet"),
                "vae": _freeze_inference_module(components.vae, "vae"),
                "text_encoder": _freeze_inference_module(
                    components.text_encoder, "text_encoder"
                ),
            }
    with _stage("encode_prompts"):
        with torch.inference_mode():
            prompts = {
                defect_type: encode_prompt(
                    components, prompt, target_device
                ).detach().clone()
                for defect_type, prompt in config.prompts.items()
            }
    with _stage("load_A"):
        with torch.inference_mode(False), torch.no_grad():
            (
                baseline_model,
                baseline_conditioner,
                baseline_adapter,
                baseline_state_hashes,
            ) = _load_checkpoint_model(
                preflight.baseline_checkpoint,
                source_config=context.source_experiment,
                unet=components.unet,
                device=target_device,
            )
    with _stage("load_B"):
        with torch.inference_mode(False), torch.no_grad():
            (
                candidate_model,
                candidate_conditioner,
                candidate_adapter,
                candidate_state_hashes,
            ) = _load_checkpoint_model(
                preflight.candidate_checkpoint,
                source_config=context.source_experiment,
                unet=components.unet,
                device=target_device,
            )
    with _stage("validate"):
        with torch.inference_mode(False), torch.no_grad():
            _assert_disjoint_checkpoint_models(
                baseline_model,
                candidate_model,
                baseline_conditioner,
                candidate_conditioner,
                baseline_adapter,
                candidate_adapter,
            )
    guarded_modules = MappingProxyType(
        {
            **base_modules,
            "baseline_conditioner": baseline_conditioner,
            "baseline_adapter": baseline_adapter,
            "candidate_conditioner": candidate_conditioner,
            "candidate_adapter": candidate_adapter,
        }
    )
    with _stage("capture_state"):
        with torch.inference_mode(False), torch.no_grad():
            inference_state = _capture_inference_state(guarded_modules)
    return _InferenceRuntime(
        components=components,
        prompts=MappingProxyType(prompts),
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        baseline_conditioner=baseline_conditioner,
        candidate_conditioner=candidate_conditioner,
        baseline_adapter=baseline_adapter,
        candidate_adapter=candidate_adapter,
        baseline_state_hashes=baseline_state_hashes,
        candidate_state_hashes=candidate_state_hashes,
        guarded_modules=guarded_modules,
        inference_state=MappingProxyType(inference_state),
    )


def _execute_checkpoint_probe_inner(
    preflight: P2B3Preflight | Any,
    artifact_root: Path | int,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    context = preflight.context
    config = context.experiment
    generation = context.generation
    target_device = device or torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(generation.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(generation.seed)
        torch.cuda.reset_peak_memory_stats()
    runtime = _load_inference_runtime(preflight, target_device)
    components = runtime.components
    prompts = runtime.prompts
    baseline_model = runtime.baseline_model
    candidate_model = runtime.candidate_model
    baseline_conditioner = runtime.baseline_conditioner
    candidate_conditioner = runtime.candidate_conditioner
    baseline_adapter = runtime.baseline_adapter
    candidate_adapter = runtime.candidate_adapter
    baseline_state_hashes = runtime.baseline_state_hashes
    candidate_state_hashes = runtime.candidate_state_hashes
    guarded_modules = runtime.guarded_modules
    inference_state = runtime.inference_state
    normals = {item.sample_id: item for item in preflight.base.normals}
    masks = {item.sample_id: item for item in preflight.base.masks}
    normal_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    causal_cache: dict[tuple[str, str, int], tuple[torch.Tensor, ...]] = {}
    scheduler_config = dict(components.scheduler.config)
    expected_scheduler = DDIMScheduler.from_config(scheduler_config)
    expected_scheduler.set_timesteps(config.denoise_steps, device=target_device)
    expected_timesteps = [int(item.item()) for item in expected_scheduler.timesteps]
    if len(expected_timesteps) != config.denoise_steps:
        raise ValueError("frozen scheduler did not produce the required timestep count")
    records: list[dict[str, Any]] = []
    actual_traces: list[Mapping[str, object]] = []
    paired_identity = _create_paired_output(artifact_root)
    for job in preflight.jobs:
        normal_item = normals[job.normal_id]
        mask_item = masks[job.mask_id]
        with _stage("prepare_job"):
            if job.normal_id not in normal_cache:
                loaded = load_training_image(normal_item, generation.image_size)
                try:
                    vae_dtype = next(components.vae.parameters()).dtype
                except StopIteration as error:
                    raise ValueError("VAE must expose floating parameters") from error
                vae_pixels, normal_image = _prepare_normal_tensors(
                    loaded.pixels,
                    target_device,
                    vae_dtype=vae_dtype,
                )
                with torch.inference_mode():
                    normal_latent = encode_mode(
                        components.vae, vae_pixels
                    ).detach()
                normal_cache[job.normal_id] = (normal_latent, normal_image)
            normal_latent, normal_image = normal_cache[job.normal_id]
            cache_key = (job.normal_id, job.mask_id, job.seed)
            if cache_key not in causal_cache:
                source_mask = mask_item.mask.unsqueeze(0).to(
                    device=target_device, dtype=torch.float32
                )
                latent_masks = build_latent_masks(
                    source_mask, normal_latent.shape[2:]
                )
                context_latent = build_masked_context(
                    normal_latent,
                    latent_masks.binary,
                    torch.Generator(device=target_device).manual_seed(
                        job.seed + 10_000_000
                    ),
                )
                noise = torch.randn(
                    normal_latent.shape,
                    generator=torch.Generator(device=target_device).manual_seed(
                        job.seed
                    ),
                    device=target_device,
                    dtype=normal_latent.dtype,
                )
                causal_cache[cache_key] = (
                    source_mask,
                    latent_masks.binary,
                    latent_masks.soft,
                    context_latent,
                    noise,
                )
            source_mask, binary_mask, soft_mask, context_latent, noise = (
                causal_cache[cache_key]
            )
            trusted_reference_path = (
                preflight.reference_path / job.accepted_image_relpath
            )
            if (
                trusted_reference_path.is_symlink()
                or not trusted_reference_path.is_file()
                or sha256_file(trusted_reference_path)
                != job.accepted_image_sha256
            ):
                raise ValueError(
                    "accepted P2-B1 reference image changed before execution"
                )
            with Image.open(trusted_reference_path) as trusted_reference:
                trusted_reference.load()
                if (
                    trusted_reference.mode != "RGB"
                    or trusted_reference.size
                    != (generation.image_size, generation.image_size)
                ):
                    raise ValueError(
                        "accepted P2-B1 reference image contract is invalid"
                    )
                reference_array = np.asarray(
                    trusted_reference, dtype=np.uint8
                ).copy()
            reference_image = (
                torch.from_numpy(reference_array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=target_device, dtype=torch.float32)
                / 255.0
            )
            prompt = prompts[job.defect_type]
        conditions: dict[str, dict[str, Any]] = {}
        for role, model, authenticated in (
            ("baseline", baseline_model, preflight.baseline_checkpoint),
            ("candidate", candidate_model, preflight.candidate_checkpoint),
        ):
            scheduler = _InstrumentedScheduler(
                DDIMScheduler.from_config(scheduler_config)
            )
            model_forward_calls = 0

            def count_model_forward(
                _module: torch.nn.Module,
                _inputs: tuple[object, ...],
                _output: object,
            ) -> None:
                nonlocal model_forward_calls
                model_forward_calls += 1

            with _stage("sample_A" if role == "baseline" else "sample_B"):
                with torch.inference_mode():
                    hook = model.register_forward_hook(count_model_forward)
                    try:
                        prediction = denoise_masked_infill(
                            model=model,
                            scheduler=scheduler,
                            normal_latent=normal_latent,
                            context=context_latent,
                            binary_mask=binary_mask,
                            prompt_embeds=prompt,
                            defect_type=job.defect_type,
                            noise=noise,
                            steps=config.denoise_steps,
                        )
                    finally:
                        hook.remove()
                    actual_timesteps = [
                        int(item.item()) for item in scheduler.timesteps
                    ]
                    if (
                        scheduler.set_timesteps_calls != 1
                        or scheduler.step_timesteps != expected_timesteps
                        or actual_timesteps != expected_timesteps
                        or model_forward_calls != config.denoise_steps
                    ):
                        raise RuntimeError(
                            f"{role} did not execute the authenticated full "
                            "scheduler/model trace"
                        )
                    sampler_sha = _canonical_sha(
                        {
                            "scheduler": "DDIMScheduler",
                            "config": scheduler_config,
                            "timesteps": actual_timesteps,
                            "set_timesteps_calls": scheduler.set_timesteps_calls,
                            "step_calls": len(scheduler.step_timesteps),
                            "model_forward_calls": model_forward_calls,
                        },
                        "instrumented sampler trace",
                    )
                    timestep_sha = _canonical_sha(
                        actual_timesteps, "instrumented timestep schedule"
                    )
                    composite = _decode_and_composite(
                        components.vae,
                        prediction,
                        source_mask,
                        normal_image,
                    )
            image_relpath = f"paired/{job.pair_id}_{role}.png"
            with _stage("publish"):
                image_sha = _save_rgb_atomic(
                    artifact_root,
                    paired_identity,
                    Path(image_relpath).name,
                    composite,
                )
            conditions[role] = {
                "checkpoint_role": role,
                "checkpoint_relpath": getattr(config, f"{role}_checkpoint_relpath"),
                "checkpoint_hashes": dict(authenticated.hashes),
                "data_role": "heldout",
                "normal_id": job.normal_id,
                "mask_id": job.mask_id,
                "seed": job.seed,
                "source_normal_sha256": job.source_normal_sha256,
                "source_mask_sha256": job.source_mask_sha256,
                "reference_image_sha256": "",
                "prompt_sha256": hashlib.sha256(job.type_prompt.encode()).hexdigest(),
                "prompt_embedding_sha256": _tensor_sha256(prompt),
                "normal_latent_sha256": _tensor_sha256(normal_latent),
                "context_latent_sha256": _tensor_sha256(context_latent),
                "binary_mask_sha256": _tensor_sha256(binary_mask),
                "soft_mask_sha256": _tensor_sha256(soft_mask),
                "noise_sha256": _tensor_sha256(noise),
                "sampler_sha256": sampler_sha,
                "timestep_schedule_sha256": timestep_sha,
                "scheduler_config": scheduler_config,
                "scheduler_timesteps": actual_timesteps,
                "scheduler_set_timesteps_calls": scheduler.set_timesteps_calls,
                "scheduler_step_calls": len(scheduler.step_timesteps),
                "model_forward_calls": model_forward_calls,
                "conditioner_state_sha256": (
                    baseline_state_hashes if role == "baseline" else candidate_state_hashes
                )["conditioner"],
                "adapter_state_sha256": (
                    baseline_state_hashes if role == "baseline" else candidate_state_hashes
                )["adapter"],
                "model_binding_sha256": _canonical_sha(
                    {
                        "checkpoint_hashes": dict(authenticated.hashes),
                        "state_hashes": dict(
                            baseline_state_hashes
                            if role == "baseline"
                            else candidate_state_hashes
                        ),
                    },
                    f"{role} model checkpoint binding",
                ),
                "prediction_sha256": _tensor_sha256(prediction),
                "image_relpath": image_relpath,
                "image_sha256": image_sha,
                "composite_outside_l1": 0.0,
                "sampler_calls": 1,
            }
            actual_traces.append(
                MappingProxyType(
                    {
                        "pair_id": job.pair_id,
                        "checkpoint_role": role,
                        "checkpoint_state_identity": MappingProxyType(
                            dict(
                                baseline_state_hashes
                                if role == "baseline"
                                else candidate_state_hashes
                            )
                        ),
                        "set_timesteps_calls": scheduler.set_timesteps_calls,
                        "scheduler_steps": tuple(scheduler.step_timesteps),
                        "model_forward_calls": model_forward_calls,
                        "prediction_sha256": conditions[role][
                            "prediction_sha256"
                        ],
                        "output_sha256": image_sha,
                    }
                )
            )
        reference_relpath = f"paired/{job.pair_id}_reference.png"
        mask_relpath = f"paired/{job.pair_id}_mask.png"
        with _stage("publish"):
            reference_sha = _save_rgb_atomic(
                artifact_root,
                paired_identity,
                Path(reference_relpath).name,
                reference_image,
            )
            mask_sha = _save_mask_atomic(
                artifact_root,
                paired_identity,
                Path(mask_relpath).name,
                source_mask,
            )
        for condition in conditions.values():
            condition["reference_image_sha256"] = reference_sha
        records.append(
            {
                "schema_version": "p2_b3_paired_record_v1",
                "pair_id": job.pair_id,
                "reference_slot_id": job.reference_slot_id,
                "normal_id": job.normal_id,
                "mask_id": job.mask_id,
                "source_mask_type": job.source_mask_type,
                "seed": job.seed,
                "defect_type": job.defect_type,
                "source_normal_sha256": job.source_normal_sha256,
                "source_mask_sha256": job.source_mask_sha256,
                "reference_image_relpath": reference_relpath,
                "reference_image_sha256": reference_sha,
                "mask_relpath": mask_relpath,
                "mask_sha256": mask_sha,
                "prompt_sha256": hashlib.sha256(job.type_prompt.encode()).hexdigest(),
                "baseline": conditions["baseline"],
                "candidate": conditions["candidate"],
            }
        )
    with _stage("validate"):
        with torch.inference_mode(False), torch.no_grad():
            _assert_inference_state_unchanged(
                guarded_modules, inference_state
            )
    with _stage("finalize"):
        peak = 0
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
            peak = torch.cuda.max_memory_allocated(target_device)
        result = {
            "records": tuple(records),
            "peak_vram_bytes": peak,
            "runtime_authority": {
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "baseline_conditioner": baseline_conditioner,
                "candidate_conditioner": candidate_conditioner,
                "baseline_adapter": baseline_adapter,
                "candidate_adapter": candidate_adapter,
                "baseline_state_hashes": dict(baseline_state_hashes),
                "candidate_state_hashes": dict(candidate_state_hashes),
                "scheduler_config": scheduler_config,
                "scheduler_timesteps": expected_timesteps,
                "actual_traces": tuple(actual_traces),
            },
        }
    return result


def _execute_checkpoint_probe(
    preflight: P2B3Preflight | Any,
    artifact_root: Path | int,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    with _forbid_training_apis():
        return _execute_checkpoint_probe_inner(
            preflight, artifact_root, device=device
        )


def _verify_snapshot(
    path: Path, expected: Mapping[str, Mapping[str, int | str]], label: str
) -> None:
    try:
        actual = _checkpoint_snapshot(path)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        raise RuntimeError(f"{label} changed during P2-B3") from error
    if actual != dict(expected):
        raise RuntimeError(f"{label} changed during P2-B3")


def _checkpoint_effect_summary(
    records: Sequence[Mapping[str, object]],
) -> tuple[bool, int]:
    values = tuple(records)
    identical = sum(
        record["baseline"]["image_sha256"]
        == record["candidate"]["image_sha256"]
        for record in values
    )
    return identical < len(values), identical


def _verify_p2_b3_integrity(
    preflight: P2B3Preflight, identity: tuple[int, int]
) -> str:
    head = _verify_p2_b1_integrity(preflight.base, identity)
    for authenticated in (
        preflight.baseline_checkpoint, preflight.candidate_checkpoint
    ):
        _reject_lexical_symlinks(
            preflight.context.paths.runs_root, authenticated.path
        )
        _verify_snapshot(
            authenticated.path, authenticated.snapshot,
            f"{authenticated.role} checkpoint",
        )
    current_reference: dict[str, Mapping[str, int | str]] = {}
    for name in preflight.reference_snapshot:
        path = Path(name)
        if not path.is_absolute():
            path = preflight.reference_path / name
        current_reference[name] = _snapshot(path)
    if current_reference != dict(preflight.reference_snapshot):
        raise RuntimeError("accepted P2-B1 reference changed during P2-B3")
    current_candidate_run = {
        name: _snapshot(preflight.candidate_checkpoint.path.parent / name)
        for name in preflight.candidate_run_snapshot
    }
    if current_candidate_run != dict(preflight.candidate_run_snapshot):
        raise RuntimeError("accepted P2-B2 report/provenance changed during P2-B3")
    return head


def _pin_output(context: P2B3Context) -> _PinnedOutput:
    flags = os.O_RDONLY
    for name in ("O_DIRECTORY", "O_NOFOLLOW"):
        if not hasattr(os, name):
            raise RuntimeError("stable output descriptor is unavailable")
        flags |= getattr(os, name)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(context.output, flags)
    identity = os.fstat(fd)
    pinned = _PinnedOutput(
        path=Path(os.path.abspath(context.output)),
        runs_root=Path(os.path.abspath(context.paths.runs_root)),
        fd=fd,
        device=identity.st_dev,
        inode=identity.st_ino,
        proc_path=Path("/proc/self/fd") / str(fd),
    )
    _validate_pinned_output(pinned)
    return pinned


def _validate_pinned_output(output: _PinnedOutput) -> None:
    handle = os.fstat(output.fd)
    if (handle.st_dev, handle.st_ino) != (output.device, output.inode):
        raise RuntimeError("output changed during P2-B3")
    public = os.stat(output.path, follow_symlinks=False)
    if not stat.S_ISDIR(public.st_mode):
        raise RuntimeError("output changed during P2-B3")
    if (public.st_dev, public.st_ino) != (output.device, output.inode):
        raise RuntimeError("output changed during P2-B3")
    if output.path.resolve(strict=True) != output.proc_path.resolve(strict=True):
        raise RuntimeError("output changed during P2-B3")
    if not output.path.is_relative_to(output.runs_root.resolve(strict=True)):
        raise RuntimeError("output escapes runs_root")


def _cleanup_owned(output: _PinnedOutput) -> None:
    def remove_at(parent_fd: int, name: str) -> None:
        try:
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(entry.st_mode):
            os.unlink(name, dir_fd=parent_fd)
            return
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        child_fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(child_fd)
            if (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino):
                raise RuntimeError("owned directory changed during cleanup")
            for child_name in os.listdir(child_fd):
                remove_at(child_fd, child_name)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                raise RuntimeError("owned directory changed during cleanup")
        finally:
            os.close(child_fd)
        os.rmdir(name, dir_fd=parent_fd)

    for name in _OWNED_TOP_LEVEL:
        remove_at(output.fd, name)


def _write_failed(output: _PinnedOutput) -> None:
    payload = {
        "schema_version": "p2_b3_checkpoint_paired_probe_report_v1",
        "status": "failed",
        "reason": "runtime_or_integrity_failure",
    }
    json.dumps(payload, sort_keys=True, allow_nan=False)
    _write_json_at(output.fd, "report.json", payload)
    _validate_pinned_output(output)


def run_p2_b3(context: P2B3Context | Any) -> int:
    pinned: _PinnedOutput | None = None
    try:
        with _stage("validate"):
            preflight = validate_p2_b3_preflight(context)
        with _stage("publish"):
            context.output.mkdir(parents=True, exist_ok=False)
            pinned = _pin_output(context)
            identity = _output_identity(context.output)
        result = _execute_checkpoint_probe(preflight, pinned.fd)
        _validate_pinned_output(pinned)
        records = tuple(result["records"])
        with _stage("validate"):
            validate_paired_records(records, preflight.jobs, context.experiment)
        assignments = build_blind_assignments(
            preflight.jobs, context.experiment.blind_seed
        )
        blind_ids = {
            job.pair_id: secrets.token_hex(32) for job in preflight.jobs
        }
        public, reveal = build_blind_records(
            records, assignments, blind_ids=blind_ids
        )
        with _stage("publish"):
            _write_jsonl_at(pinned.fd, "records.jsonl", records)
            _validate_pinned_output(pinned)
            materialize_blind_package(
                pinned.fd,
                records,
                public,
                reveal,
                blind_seed=context.experiment.blind_seed,
            )
            _validate_pinned_output(pinned)
        with _stage("finalize"):
            head_end = _verify_p2_b3_integrity(preflight, identity)
        prompt_embeddings = {
            str(record["defect_type"]): str(
                record["baseline"]["prompt_embedding_sha256"]
            )
            for record in records
        }
        type_prompts = {
            label: {
                "text": text,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_embedding_sha256": prompt_embeddings[label],
            }
            for label, text in context.experiment.prompts.items()
        }
        checkpoint_effect_detected, identical_pair_count = (
            _checkpoint_effect_summary(records)
        )
        report_lineages = {
            role: {
                "path": str(getattr(preflight, f"{role}_checkpoint").path),
                "hashes": dict(getattr(preflight, f"{role}_checkpoint").hashes),
                "snapshot": dict(getattr(preflight, f"{role}_checkpoint").snapshot),
            }
            for role in ("baseline", "candidate")
        }
        provenance = {
            "schema_version": "p2_b3_checkpoint_paired_probe_provenance_v1",
            "git_head_start": preflight.git_head,
            "git_head_end": head_end,
            "git_dirty": False,
            "visual_status": "pending_human_review",
            "training_performed": False,
            "checkpoint_written": False,
            "checkpoint_effect_detected": checkpoint_effect_detected,
            "identical_pair_count": identical_pair_count,
            "evaluation_mode": context.experiment.evaluation_mode,
            "causal_difference": context.experiment.causal_difference,
            "input_hashes": dict(preflight.input_hashes),
            "source_snapshot_before": preflight.base.source_snapshot,
            "source_snapshot_after": preflight.base.source_snapshot,
            "model_snapshot_before": preflight.base.model_snapshot,
            "model_snapshot_after": preflight.base.model_snapshot,
            "reference_snapshot_before": dict(preflight.reference_snapshot),
            "reference_snapshot_after": dict(preflight.reference_snapshot),
            "baseline_checkpoint_lineage": {
                "path": str(preflight.baseline_checkpoint.path),
                "hashes": dict(preflight.baseline_checkpoint.hashes),
                "snapshot_before": dict(preflight.baseline_checkpoint.snapshot),
                "snapshot_after": dict(preflight.baseline_checkpoint.snapshot),
            },
            "candidate_checkpoint_lineage": {
                "path": str(preflight.candidate_checkpoint.path),
                "hashes": dict(preflight.candidate_checkpoint.hashes),
                "snapshot_before": dict(preflight.candidate_checkpoint.snapshot),
                "snapshot_after": dict(preflight.candidate_checkpoint.snapshot),
            },
            "candidate_run_lineage": {
                "path": str(preflight.candidate_checkpoint.path.parent),
                "hashes": dict(context.experiment.candidate_run_hashes),
                "snapshot_before": dict(preflight.candidate_run_snapshot),
                "snapshot_after": dict(preflight.candidate_run_snapshot),
            },
            "type_prompts": type_prompts,
        }
        report = {
            "schema_version": "p2_b3_checkpoint_paired_probe_report_v1",
            "status": "complete",
            "reason": None,
            "visual_status": "pending_human_review",
            "training_performed": False,
            "checkpoint_written": False,
            "evaluation_mode": context.experiment.evaluation_mode,
            "causal_difference": context.experiment.causal_difference,
            "record_count": len(records),
            "blind_pair_count": len(public),
            "checkpoint_effect_detected": checkpoint_effect_detected,
            "identical_pair_count": identical_pair_count,
            "blind_package_relpath": "blind_package",
            "peak_vram_bytes": result["peak_vram_bytes"],
            "input_hashes": dict(preflight.input_hashes),
            "checkpoint_lineages": report_lineages,
            "promotion_rule": {
                "candidate_semantic_improves_at_least_types": 3,
                "required_improvements": ["print", "hole"],
                "no_realism_regression": True,
                "no_boundary_regression": True,
            },
        }
        json.dumps(provenance, sort_keys=True, allow_nan=False)
        json.dumps(report, sort_keys=True, allow_nan=False)
        with _stage("publish"):
            _write_json_at(pinned.fd, "provenance.json", provenance)
            _validate_pinned_output(pinned)
            _verify_p2_b3_integrity(preflight, identity)
            _write_json_at(pinned.fd, "report.json", report)
            _validate_pinned_output(pinned)
            _verify_p2_b3_integrity(preflight, identity)
        with _stage("validate"):
            validate_p2_b3_artifact_tree(
                pinned.fd,
                preflight,
                runtime_authority=result["runtime_authority"],
                image_size=context.generation.image_size,
            )
            _validate_pinned_output(pinned)
        with _stage("finalize"):
            _verify_p2_b3_integrity(preflight, identity)
            _validate_pinned_output(pinned)
        return 0
    except (AssertionError, FileNotFoundError, NotADirectoryError, OSError,
            RuntimeError, TypeError, ValueError):
        if pinned is not None:
            try:
                _cleanup_owned(pinned)
                _validate_pinned_output(pinned)
                _write_failed(pinned)
            except (OSError, RuntimeError, ValueError):
                try:
                    _cleanup_owned(pinned)
                except (OSError, RuntimeError):
                    pass
        return 1
    finally:
        if pinned is not None:
            try:
                os.close(pinned.fd)
            except OSError:
                pass


def build_p2_b3_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the P2-B3 checkpoint-paired semantic probe"
    )
    parser.add_argument("--paths", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_p2_b3_parser().parse_args(argv)
    paths_path = Path(args.paths)
    generation_path = Path(args.generation)
    experiment_path = Path(args.experiment)
    source_experiment_path = Path(args.source_experiment)
    try:
        return run_p2_b3(
            P2B3Context(
                paths=load_config(paths_path),
                generation=load_p2_b1_config(generation_path),
                experiment=load_p2_b3_config(experiment_path),
                source_experiment=load_p2_a1_config(source_experiment_path),
                manifest=Path(args.manifest),
                lock=Path(args.lock),
                source_run=Path(args.source_run),
                baseline_run=Path(args.baseline_run),
                reference_run=Path(args.reference_run),
                baseline_checkpoint=Path(args.baseline_checkpoint),
                candidate_checkpoint=Path(args.candidate_checkpoint),
                output=Path(args.output),
                config_paths={
                    "paths": paths_path,
                    "generation": generation_path,
                    "experiment": experiment_path,
                    "source_experiment": source_experiment_path,
                },
                cwd=Path.cwd(),
            )
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True))
        return 1


__all__ = [
    "AuthenticatedCheckpoint",
    "P2B3Context",
    "P2B3Preflight",
    "_authenticate_checkpoint",
    "_execute_checkpoint_probe",
    "_forbid_training_apis",
    "build_p2_b3_parser",
    "run_p2_b3",
    "validate_p2_b3_preflight",
]

from __future__ import annotations

import gc
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from cadiff.config import ProjectConfig
from cadiff.p2_a0.probe import (
    _git_state,
    _regular_input,
    _validate_gpu_contract,
    _validate_hf_home,
    _validated_output,
)
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_b2.config import P2B2Config
from cadiff.provenance import sha256_file, write_json_atomic

from .config import P2C0Config
from .contracts import GenerationJob, build_generation_schedule
from .dataset import DownstreamData, load_downstream_data
from .generation import (
    FrozenP2B2Renderer,
    GenerationPools,
    GenerationSummary,
    load_generation_pools,
    materialize_synthetic_dataset,
)
from .model import DraemSegmenter
from .training import ArmResult, decide_promotion, train_arm


@dataclass(frozen=True)
class P2C0Context:
    paths: ProjectConfig
    experiment: P2C0Config
    source_experiment: P2A1Config
    prompt_experiment: P2B2Config
    renderer_manifest: Path
    manifest: Path
    lock: Path
    output: Path
    config_paths: Mapping[str, Path]
    cwd: Path


@dataclass(frozen=True)
class P2C0Preflight:
    context: P2C0Context
    output: Path
    git_head: str
    checkpoint: Path
    checkpoint_hashes: dict[str, str]
    input_hashes: dict[str, str]
    pools: GenerationPools
    jobs: tuple[GenerationJob, ...]


@dataclass(frozen=True)
class P2C0Backend:
    generate: Callable[[P2C0Preflight, Path, torch.device], GenerationSummary]
    load_data: Callable[[P2C0Preflight, Path], DownstreamData]
    train: Callable[..., ArmResult]


_ARM_ORDER = ("real_fs", "gen_only", "real_plus_gen", "cutpaste")


def _strict_yaml(path: Path) -> dict[str, Any]:
    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader: Loader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"renderer manifest has invalid or duplicate key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)
    except yaml.YAMLError as error:
        raise ValueError("renderer manifest is invalid YAML") from error
    if not isinstance(value, dict):
        raise ValueError("renderer manifest must be a mapping")
    return value


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _renderer_authority(
    path: Path, runs_root: Path
) -> tuple[Path, dict[str, str], dict[str, str]]:
    raw = _strict_yaml(path)
    if (
        raw.get("schema_version") != "p2_renderer_freeze_v1"
        or raw.get("category") != "hazelnut"
        or raw.get("status") != "frozen_default"
    ):
        raise ValueError("renderer freeze identity is invalid")
    checkpoint_raw = raw.get("checkpoint")
    source_raw = raw.get("source_run")
    if not isinstance(checkpoint_raw, dict) or not isinstance(source_raw, dict):
        raise ValueError("renderer checkpoint and source run must be mappings")
    checkpoint = (runs_root / str(checkpoint_raw.get("runs_root_relpath"))).resolve(
        strict=True
    )
    checkpoint.relative_to(runs_root.resolve(strict=True))
    files = checkpoint_raw.get("files")
    if not isinstance(files, dict) or set(files) != {
        "adapter.safetensors",
        "conditioner.safetensors",
        "metadata.json",
    }:
        raise ValueError("renderer checkpoint layout is invalid")
    checkpoint_hashes: dict[str, str] = {}
    for name, authority in files.items():
        if not isinstance(authority, dict) or not _valid_sha(authority.get("sha256")):
            raise ValueError("renderer checkpoint hash authority is invalid")
        candidate = checkpoint / name
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("renderer checkpoint file is not regular")
        stat = candidate.stat()
        if stat.st_size != authority.get("size") or sha256_file(candidate) != authority["sha256"]:
            raise ValueError(f"renderer checkpoint mismatch: {name}")
        checkpoint_hashes[name] = str(authority["sha256"])
    source_root = (runs_root / str(source_raw.get("runs_root_relpath"))).resolve(strict=True)
    source_root.relative_to(runs_root.resolve(strict=True))
    source_hashes = source_raw.get("hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("renderer source run hashes are missing")
    verified_source: dict[str, str] = {}
    for relpath, expected in source_hashes.items():
        if not isinstance(relpath, str) or not _valid_sha(expected):
            raise ValueError("renderer source run hash authority is invalid")
        candidate = (source_root / relpath).resolve(strict=True)
        candidate.relative_to(source_root)
        if candidate.is_symlink() or not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"renderer source run mismatch: {relpath}")
        verified_source[f"renderer_source/{relpath}"] = expected
    return checkpoint, checkpoint_hashes, verified_source


def validate_p2_c0_preflight(context: P2C0Context) -> P2C0Preflight:
    _validate_gpu_contract()
    _validate_hf_home(context.cwd)
    head, dirty = _git_state(context.cwd)
    if dirty:
        raise RuntimeError("Git worktree must be clean")
    output = _validated_output(context.output, context.paths.runs_root)
    manifest = _regular_input(context.manifest, name="manifest")
    lock = _regular_input(context.lock, name="lock")
    renderer_manifest = _regular_input(context.renderer_manifest, name="renderer manifest")
    checkpoint, checkpoint_hashes, source_hashes = _renderer_authority(
        renderer_manifest, context.paths.runs_root
    )
    pools = load_generation_pools(
        manifest=manifest,
        dataset_root=context.paths.dataset_root,
        source_config=context.source_experiment,
        normal_count=context.experiment.synthetic_per_type,
        image_size=context.experiment.image_size,
    )
    jobs = build_generation_schedule(
        normal_ids=tuple(pools.normals),
        masks=pools.training_masks,
        defect_types=context.experiment.defect_types,
        samples_per_type=context.experiment.synthetic_per_type,
        seed=context.experiment.generation_seed,
    )
    input_hashes = {
        "manifest": sha256_file(manifest),
        "lock": sha256_file(lock),
        "renderer_manifest": sha256_file(renderer_manifest),
        **{
            f"config/{name}": sha256_file(_regular_input(path, name=f"{name} config"))
            for name, path in context.config_paths.items()
        },
        **{f"renderer_checkpoint/{name}": digest for name, digest in checkpoint_hashes.items()},
        **source_hashes,
    }
    return P2C0Preflight(
        context=context,
        output=output,
        git_head=head,
        checkpoint=checkpoint,
        checkpoint_hashes=checkpoint_hashes,
        input_hashes=input_hashes,
        pools=pools,
        jobs=jobs,
    )


def _generate(
    preflight: P2C0Preflight, output: Path, device: torch.device
) -> GenerationSummary:
    context = preflight.context
    renderer = FrozenP2B2Renderer(
        model_root=context.paths.model_root,
        checkpoint=preflight.checkpoint,
        checkpoint_hashes=preflight.checkpoint_hashes,
        source_config=context.source_experiment,
        prompts=context.prompt_experiment.prompts,
        device=device,
        denoise_steps=50,
    )
    try:
        return materialize_synthetic_dataset(
            jobs=preflight.jobs,
            normals=preflight.pools.normals,
            masks=preflight.pools.masks,
            output=output,
            render=renderer,
            checkpoint_hashes=preflight.checkpoint_hashes,
        )
    finally:
        del renderer
        gc.collect()
        torch.cuda.empty_cache()


def _load_data(preflight: P2C0Preflight, synthetic_root: Path) -> DownstreamData:
    context = preflight.context
    return load_downstream_data(
        manifest=context.manifest,
        dataset_root=context.paths.dataset_root,
        source_config=context.source_experiment,
        synthetic_root=synthetic_root,
        image_size=context.experiment.downstream_image_size,
        seed=context.experiment.generation_seed,
    )


def default_backend() -> P2C0Backend:
    return P2C0Backend(generate=_generate, load_data=_load_data, train=train_arm)


def _result_payload(result: ArmResult) -> dict[str, Any]:
    return {
        "steps": result.steps,
        "final_loss": result.final_loss,
        "pixel": asdict(result.pixel),
        "image": asdict(result.image),
        "checkpoint_relpath": str(
            result.checkpoint_path.parent.name + "/" + result.checkpoint_path.name
        ),
        "checkpoint_sha256": result.checkpoint_sha256,
    }


def _promotion(results: Mapping[str, ArmResult], config: P2C0Config) -> bool:
    real_fs = results["real_fs"].metrics_dict()
    mixed = results["real_plus_gen"].metrics_dict()
    keys = ("pixel_ap", "au_pro", "image_auroc")
    return decide_promotion(
        real_fs={key: real_fs[key] for key in keys},
        real_plus_gen={key: mixed[key] for key in keys},
        pixel_ap_delta=config.promotion_pixel_ap_delta,
        au_pro_delta=config.promotion_au_pro_delta,
        maximum_image_auroc_drop=config.maximum_image_auroc_drop,
    )


def _execute_p2_c0(
    preflight: P2C0Preflight,
    *,
    backend: P2C0Backend,
    device: torch.device,
) -> dict[str, Any]:
    context = preflight.context
    synthetic_root = preflight.output / "synthetic"
    generation = backend.generate(preflight, synthetic_root, device)
    if generation.record_count != context.experiment.synthetic_total:
        raise RuntimeError("synthetic generation count is incomplete")
    data = backend.load_data(preflight, synthetic_root)
    if tuple(data.arms) != _ARM_ORDER:
        raise RuntimeError("downstream arms must have the frozen order")
    torch.manual_seed(context.experiment.generation_seed)
    template = DraemSegmenter(base_channels=16)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in template.state_dict().items()
    }
    del template
    results: dict[str, ArmResult] = {}
    for arm_name in _ARM_ORDER:
        result = backend.train(
            arm=data.arms[arm_name],
            evaluation=data.evaluation,
            model_factory=lambda: DraemSegmenter(base_channels=16),
            initial_state=initial_state,
            steps=context.experiment.training_steps_per_arm,
            batch_size=context.experiment.batch_size,
            learning_rate=context.experiment.learning_rate,
            focal_gamma=context.experiment.focal_gamma,
            max_fpr=context.experiment.max_fpr,
            device=device,
            checkpoint_path=preflight.output / arm_name / "model.safetensors",
        )
        if result.name != arm_name or result.steps != context.experiment.training_steps_per_arm:
            raise RuntimeError("downstream arm result violates the equal-budget contract")
        results[arm_name] = result
    promotion = _promotion(results, context.experiment)
    return {
        "schema_version": "p2_c0_hazelnut_downstream_report_v1",
        "status": "complete",
        "reason": None,
        "evaluation_mode": context.experiment.evaluation_mode,
        "generation": asdict(generation),
        "data_counts": data.counts,
        "arms": {name: _result_payload(results[name]) for name in _ARM_ORDER},
        "promotion": {
            "gate_passed": promotion,
            "next_step": "three_seed_confirmation" if promotion else "analyze_downstream_failure",
        },
        "git_head": preflight.git_head,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }


def _smoke_p2_c0(
    preflight: P2C0Preflight,
    *,
    backend: P2C0Backend,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    by_type: dict[str, GenerationJob] = {}
    for job in preflight.jobs:
        by_type.setdefault(job.defect_type, job)
    if tuple(by_type) != preflight.context.experiment.defect_types:
        raise RuntimeError("GPU smoke requires one frozen generation job per type")
    smoke_preflight = replace(preflight, jobs=tuple(by_type.values()))
    output.mkdir(parents=True, exist_ok=False)
    generation = backend.generate(smoke_preflight, output / "synthetic", device)
    if generation.record_count != len(preflight.context.experiment.defect_types):
        raise RuntimeError("GPU smoke generation is incomplete")
    data = backend.load_data(smoke_preflight, output / "synthetic")
    torch.manual_seed(preflight.context.experiment.generation_seed)
    template = DraemSegmenter(base_channels=16)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in template.state_dict().items()
    }
    del template
    smoke_normals = tuple(item for item in data.evaluation if not item.mask.bool().any())[:4]
    smoke_anomalies = tuple(item for item in data.evaluation if item.mask.bool().any())[:4]
    smoke_evaluation = smoke_normals + smoke_anomalies
    if not smoke_normals or not smoke_anomalies:
        raise RuntimeError("GPU smoke requires normal and anomaly evaluations")
    completed: list[str] = []
    for arm_name in _ARM_ORDER:
        result = backend.train(
            arm=data.arms[arm_name],
            evaluation=smoke_evaluation,
            model_factory=lambda: DraemSegmenter(base_channels=16),
            initial_state=initial_state,
            steps=1,
            batch_size=preflight.context.experiment.batch_size,
            learning_rate=preflight.context.experiment.learning_rate,
            focal_gamma=preflight.context.experiment.focal_gamma,
            max_fpr=preflight.context.experiment.max_fpr,
            device=device,
            checkpoint_path=output / arm_name / "model.safetensors",
        )
        if result.name != arm_name or result.steps != 1:
            raise RuntimeError("GPU smoke did not execute the requested arm")
        completed.append(arm_name)
    return {"generated": generation.record_count, "arms": completed}


def run_p2_c0(context: P2C0Context, *, backend: P2C0Backend | None = None) -> int:
    output: Path | None = None
    try:
        preflight = validate_p2_c0_preflight(context)
        selected_backend = default_backend() if backend is None else backend
        if backend is None:
            with tempfile.TemporaryDirectory(prefix="p2-c0-gpu-smoke-") as temporary:
                _smoke_p2_c0(
                    preflight,
                    backend=selected_backend,
                    device=torch.device("cuda:0"),
                    output=Path(temporary) / "smoke",
                )
        output = preflight.output
        output.mkdir(parents=True, exist_ok=False)
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        report = _execute_p2_c0(
            preflight, backend=selected_backend, device=device
        )
        write_json_atomic(output / "metrics.json", report["arms"])
        write_json_atomic(
            output / "provenance.json",
            {
                "schema_version": "p2_c0_hazelnut_downstream_provenance_v1",
                "git_head": preflight.git_head,
                "input_hashes": preflight.input_hashes,
                "renderer_checkpoint": str(preflight.checkpoint),
                "renderer_checkpoint_hashes": preflight.checkpoint_hashes,
                "training_performed": True,
                "checkpoint_written": True,
            },
        )
        write_json_atomic(output / "report.json", report)
        return 0
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        if output is not None:
            if output.exists() and output.is_dir() and not output.is_symlink():
                shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=False)
            write_json_atomic(
                output / "report.json",
                {
                    "schema_version": "p2_c0_hazelnut_downstream_report_v1",
                    "status": "failed",
                    "reason": type(error).__name__,
                },
            )
        return 1


__all__ = [
    "P2C0Backend",
    "P2C0Context",
    "P2C0Preflight",
    "default_backend",
    "run_p2_c0",
    "validate_p2_c0_preflight",
]

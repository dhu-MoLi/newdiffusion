from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from cadiff.config import ProjectConfig, load_config
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p2_a0.all_pairs import _strict_json_object, evaluate_all_pairs
from cadiff.p2_a0.fewshot import (
    FewshotPreflight,
    FewshotSplit,
    _assert_parameter_versions_unchanged,
    _capture_parameter_versions,
    _output_identity,
    _prepare_samples,
    _resampled_training_steps,
    _snapshot,
    _tensor_sha256,
    _verify_integrity,
    _verify_output_identity,
    _visualize_sample,
    _write_jsonl_atomic,
    balanced_training_order,
)
from cadiff.p2_a0.masking import build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_a1.evaluation import NOISE_BANK_OFFSETS, summarize_p2_a1_evaluation
from cadiff.p2_a1.ranking import ranking_optimizer_step
from cadiff.p2_a1.runner import (
    P2A1Context,
    _assert_new_artifact_child,
    _cleanup_partial,
    _load_p2_a1_checkpoint,
    _save_p2_a1_checkpoint,
    _verify_artifact_child,
    validate_p2_a1_preflight,
)
from cadiff.p2_b2.config import P2B2Config, load_p2_b2_experiment_config
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic


@dataclass(frozen=True)
class P2B2Context:
    paths: ProjectConfig | Any
    experiment: P2B2Config
    source_experiment: P2A1Config
    manifest: Path
    lock: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class P2B2Preflight:
    context: P2B2Context
    base: FewshotPreflight
    source_checkpoint: Path
    source_checkpoint_snapshot: dict[str, dict[str, int | str]]


@dataclass(frozen=True)
class TrainingTraceIdentity:
    step: int
    sample_id: str
    defect_type: str
    timestep: int
    context_seed: int
    context_sha256: str
    noise_sha256: str
    prompt_sha256: str
    prompt_embedding_sha256: str


def _encode_type_prompts(
    components: object, config: P2B2Config, device: torch.device
) -> Mapping[str, Tensor]:
    if not isinstance(config, P2B2Config):
        raise ValueError("prompt bank requires a P2B2Config")
    prompts = {
        label: encode_prompt(components, config.prompts[label], device)
        .detach()
        .clone()
        for label in config.defect_types
    }
    if any(
        not value.is_floating_point()
        or value.requires_grad
        or not torch.isfinite(value).all().item()
        for value in prompts.values()
    ):
        raise ValueError("type prompt embeddings must be finite detached tensors")
    if len({_tensor_sha256(value) for value in prompts.values()}) != len(prompts):
        raise ValueError("type prompt embeddings must be distinct")
    return prompts


def _freeze_prompt_lineage(
    config: P2B2Config, prompts: Mapping[str, Tensor]
) -> Mapping[str, Mapping[str, str]]:
    lineage = _prompt_lineage(config, prompts)
    _validate_prompt_lineage(config, lineage)
    return MappingProxyType(
        {
            label: MappingProxyType(dict(lineage[label]))
            for label in config.defect_types
        }
    )


def _validate_prompt_bank_lineage(
    config: P2B2Config,
    prompts: Mapping[str, Tensor],
    expected_lineage: Mapping[str, Mapping[str, str]],
) -> None:
    _validate_prompt_lineage(config, expected_lineage)
    actual_lineage = _prompt_lineage(config, prompts)
    _validate_prompt_lineage(config, actual_lineage)
    if actual_lineage != expected_lineage:
        raise ValueError("prompt bank differs from the independent prompt authority")


def _prompt_identity(
    config: P2B2Config, prompts: Mapping[str, Tensor], defect_type: str
) -> tuple[str, str]:
    if not isinstance(config, P2B2Config):
        raise ValueError("prompt identity requires a P2B2Config")
    if set(prompts) != set(config.defect_types) or len(prompts) != len(
        config.defect_types
    ):
        raise ValueError("prompt bank must contain the exact configured vocabulary")
    if defect_type not in config.defect_types:
        raise ValueError("prompt identity defect_type is not configured")
    embedding = prompts[defect_type]
    return (
        hashlib.sha256(config.prompts[defect_type].encode()).hexdigest(),
        _tensor_sha256(embedding),
    )


def _run_ranking_step(*, prepared: Any, vae: Any, **kwargs: Any) -> Any:
    """Narrow extension seam; the frozen P2-B2 objective ignores image-space inputs."""

    del prepared, vae
    return ranking_optimizer_step(**kwargs)


def _validate_training_records(
    records: Sequence[object],
    split: FewshotSplit,
    config: P2B2Config,
    source_config: P2A1Config,
    expected_trace: Sequence[TrainingTraceIdentity],
) -> None:
    if len(records) != config.steps:
        raise ValueError(f"training record count must be {config.steps}")
    if (
        source_config.category != config.category
        or source_config.defect_types != config.defect_types
        or source_config.learning_rate != config.learning_rate
        or source_config.type_rank != config.type_rank
    ):
        raise ValueError("source experiment is incompatible with P2-B2")
    schedule = balanced_training_order(split.train, config.steps, config.defect_types)
    expected_keys = {
        "step",
        "sample_id",
        "defect_type",
        "timestep",
        "total",
        "base_total",
        "noise",
        "inside",
        "outside",
        "ranking_loss",
        "correct_negative_ratio",
        "hardest_negative",
        "grad_l2",
        "finite",
        "context_seed",
        "context_sha256",
        "noise_sha256",
        "prompt_sha256",
        "prompt_embedding_sha256",
    }
    if len(expected_trace) != config.steps:
        raise ValueError("independent trace authority must match configured steps")
    for index, (value, sample, expected) in enumerate(
        zip(records, schedule, expected_trace, strict=True), 1
    ):
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(f"training record {index} has an invalid exact schema")
        if (
            value["step"] != index
            or value["sample_id"] != sample.sample_id
            or value["defect_type"] != sample.defect_type
        ):
            raise ValueError(f"training record {index} does not match the schedule")
        expected_identity = asdict(expected)
        if any(
            value[field] != expected_value
            for field, expected_value in expected_identity.items()
        ):
            raise ValueError(
                f"training record {index} differs from the independent trace authority"
            )
        for field in (
            "context_sha256",
            "noise_sha256",
            "prompt_sha256",
            "prompt_embedding_sha256",
        ):
            digest = value[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"training record {index} has an invalid {field}")
        for field in (
            "total",
            "base_total",
            "noise",
            "inside",
            "outside",
            "ranking_loss",
            "correct_negative_ratio",
            "grad_l2",
        ):
            metric = value[field]
            if type(metric) is not float or not math.isfinite(metric) or metric < 0.0:
                raise ValueError(f"training record {index} has invalid {field}")
        ratio = value["correct_negative_ratio"]
        ranking = max(0.0, ratio - source_config.ranking_target_ratio)
        total = value["base_total"] + source_config.ranking_weight * ranking
        if not math.isclose(value["ranking_loss"], ranking, abs_tol=1e-6):
            raise ValueError(f"training record {index} violates the ranking formula")
        if not math.isclose(value["total"], total, abs_tol=1e-6):
            raise ValueError(f"training record {index} violates the total formula")
        hardest = value["hardest_negative"]
        if hardest not in {*config.defect_types, "normal"} - {sample.defect_type}:
            raise ValueError(f"training record {index} has invalid hardest_negative")
        if value["grad_l2"] <= 0.0 or value["finite"] is not True:
            raise ValueError(f"training record {index} is not finite")
    context_hashes = {str(value["context_sha256"]) for value in records}  # type: ignore[index]
    if len(context_hashes) != config.steps:
        raise ValueError("training contexts must be unique per step")


def _validate_prompt_lineage(
    config: P2B2Config, lineage: Mapping[str, Mapping[str, str]]
) -> None:
    if set(lineage) != set(config.defect_types) or len(lineage) != len(config.defect_types):
        raise ValueError("prompt lineage must contain the exact configured vocabulary")
    embedding_hashes: set[str] = set()
    for label in config.defect_types:
        row = lineage[label]
        if set(row) != {"text", "prompt_sha256", "prompt_embedding_sha256"}:
            raise ValueError(f"prompt lineage has an invalid exact schema for {label}")
        text = config.prompts[label]
        if row["text"] != text or row["prompt_sha256"] != hashlib.sha256(
            text.encode()
        ).hexdigest():
            raise ValueError(f"prompt lineage text identity mismatch for {label}")
        embedding_sha = row["prompt_embedding_sha256"]
        if (
            len(embedding_sha) != 64
            or any(character not in "0123456789abcdef" for character in embedding_sha)
        ):
            raise ValueError(f"prompt lineage embedding hash is invalid for {label}")
        embedding_hashes.add(embedding_sha)
    if len(embedding_hashes) != len(config.defect_types):
        raise ValueError("prompt lineage embeddings must be distinct")


def _build_expected_training_trace(
    *,
    schedule: tuple[Any, ...],
    prepared_by_id: Mapping[str, Any],
    config: P2B2Config,
    source_config: P2A1Config,
    prompt_lineage: Mapping[str, Mapping[str, str]],
    num_train_timesteps: int,
) -> tuple[TrainingTraceIdentity, ...]:
    if len(schedule) != config.steps:
        raise ValueError("trace schedule must match configured steps")
    if type(num_train_timesteps) is not int or num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be a positive integer")
    _validate_prompt_lineage(config, prompt_lineage)
    first = prepared_by_id.get(schedule[0].sample_id) if schedule else None
    if first is None:
        raise ValueError("trace authority requires prepared training samples")
    device = first.target.device
    generator = torch.Generator(device=device).manual_seed(
        source_config.noise_seed + 10_000_000
    )
    expected: list[TrainingTraceIdentity] = []
    for step, sample in enumerate(schedule, 1):
        try:
            prepared = prepared_by_id[sample.sample_id]
        except KeyError as error:
            raise ValueError(
                f"trace authority is missing prepared sample {sample.sample_id}"
            ) from error
        if prepared.target.device != device:
            raise ValueError("trace authority prepared samples must share one device")
        context_seed = source_config.context_fill_seed + 20_000_000 + step
        context = build_masked_context(
            prepared.target,
            prepared.masks.binary,
            torch.Generator(device=device).manual_seed(context_seed),
        )
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
        prompt = prompt_lineage[sample.defect_type]
        expected.append(
            TrainingTraceIdentity(
                step=step,
                sample_id=sample.sample_id,
                defect_type=sample.defect_type,
                timestep=int(timesteps.item()),
                context_seed=context_seed,
                context_sha256=_tensor_sha256(context),
                noise_sha256=_tensor_sha256(noise),
                prompt_sha256=prompt["prompt_sha256"],
                prompt_embedding_sha256=prompt["prompt_embedding_sha256"],
            )
        )
    return tuple(expected)


def _load_accepted_trainables(
    checkpoint: Path, config: P2A1Config, device: torch.device
) -> tuple[Any, nn.Module, nn.Module, dict[str, Any]]:
    loaded = _load_p2_a1_checkpoint(checkpoint, config=config, device=device)
    vocabulary, conditioner, adapter, metadata = loaded
    if metadata.get("schema_version") != "p2_a1_factorized_checkpoint_v1":
        raise ValueError("accepted source checkpoint schema is incompatible")
    return vocabulary, conditioner, adapter, metadata


def _build_optimizer(
    model: Any, frozen_modules: Sequence[nn.Module], learning_rate: float
) -> torch.optim.Optimizer:
    if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    frozen_parameters = tuple(
        parameter for module in frozen_modules for parameter in module.parameters()
    )
    if any(module.training for module in frozen_modules) or any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in frozen_parameters
    ):
        raise ValueError("frozen components must be eval, grad-free, and non-trainable")
    trainable = tuple(model.trainable_parameters())
    if (
        not trainable
        or any(not isinstance(parameter, nn.Parameter) for parameter in trainable)
        or len({id(parameter) for parameter in trainable}) != len(trainable)
        or any(not parameter.requires_grad for parameter in trainable)
    ):
        raise ValueError("model trainable_parameters must be unique trainable Parameters")
    frozen_ids = {id(parameter) for parameter in frozen_parameters}
    if any(id(parameter) in frozen_ids for parameter in trainable):
        raise ValueError("frozen parameter entered the trainable boundary")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    optimized = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if optimized != trainable:
        raise ValueError("optimizer must contain exactly existing model trainables")
    return optimizer


def _checkpoint_snapshot(checkpoint: Path) -> dict[str, dict[str, int | str]]:
    expected = {"adapter.safetensors", "conditioner.safetensors", "metadata.json"}
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("source checkpoint must be a real directory")
    children = {path.name for path in checkpoint.iterdir()}
    if children != expected:
        raise ValueError("source checkpoint must contain exactly three accepted files")
    snapshot: dict[str, dict[str, int | str]] = {}
    for name in sorted(expected):
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source checkpoint file must be regular: {name}")
        snapshot[name] = _snapshot(path)
    return snapshot


def _reject_lexical_symlinks(root: Path, candidate: Path) -> None:
    lexical_root = Path(os.path.abspath(root))
    lexical_candidate = Path(os.path.abspath(candidate))
    try:
        relative = lexical_candidate.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError("source checkpoint lexical path escapes runs_root") from error
    current = lexical_root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        try:
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise ValueError(
                f"source checkpoint lexical node is unavailable: {current}"
            ) from error
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"source checkpoint lexical ancestor is a symlink: {current}"
            )


def _verify_new_checkpoint(
    checkpoint: Path, expected: Mapping[str, Mapping[str, int | str]]
) -> dict[str, dict[str, int | str]]:
    try:
        actual = _checkpoint_snapshot(checkpoint)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        raise RuntimeError("new checkpoint changed during P2-B2") from error
    if actual != expected:
        raise RuntimeError("new checkpoint changed during P2-B2")
    return actual


def _canonical_json(value: object, *, name: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite canonical JSON") from error


def _validated_source_checkpoint(
    checkpoint: Path,
    runs_root: Path,
    config: P2B2Config,
    source_config: P2A1Config,
) -> dict[str, dict[str, int | str]]:
    _reject_lexical_symlinks(runs_root, checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("source checkpoint must be a real directory")
    root = runs_root.resolve(strict=True)
    expected = (root / config.source_checkpoint_relpath).resolve(strict=True)
    resolved = checkpoint.resolve(strict=True)
    if resolved != expected or not resolved.is_relative_to(root):
        raise ValueError("source checkpoint does not match the frozen accepted path")
    snapshot = _checkpoint_snapshot(resolved)
    for name, expected_hash in config.source_checkpoint_hashes.items():
        if sha256_file(resolved / name) != expected_hash:
            raise ValueError(f"source checkpoint hash mismatch for {name}")
    metadata = _strict_json_object(resolved / "metadata.json")
    expected_config = json.loads(
        json.dumps(asdict(source_config), ensure_ascii=False, sort_keys=True)
    )
    if metadata.get("schema_version") != "p2_a1_factorized_checkpoint_v1":
        raise ValueError("source checkpoint schema is incompatible with P2-A1")
    if metadata.get("config") != expected_config:
        raise ValueError("source checkpoint config does not match the accepted P2-A1 run")
    return snapshot


def _verify_source_checkpoint(
    checkpoint: Path, expected: Mapping[str, Mapping[str, int | str]]
) -> None:
    try:
        actual = _checkpoint_snapshot(checkpoint)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        raise RuntimeError("source checkpoint changed during P2-B2") from error
    if actual != expected:
        raise RuntimeError("source checkpoint changed during P2-B2")


def _validate_source_experiment(config: P2B2Config, source: P2A1Config) -> None:
    expected = {
        "schema_version": "p2_a1_factorized_v1",
        "category": config.category,
        "defect_types": config.defect_types,
        "steps": 1200,
        "learning_rate": config.learning_rate,
        "type_rank": config.type_rank,
    }
    for field, value in expected.items():
        if getattr(source, field) != value:
            raise ValueError(f"source experiment {field} is incompatible with P2-B2")


def validate_p2_b2_preflight(context: P2B2Context) -> P2B2Preflight:
    if not isinstance(context, P2B2Context):
        raise ValueError("context must be a P2B2Context")
    if not isinstance(context.experiment, P2B2Config):
        raise ValueError("experiment must be a P2B2Config")
    if not isinstance(context.source_experiment, P2A1Config):
        raise ValueError("source_experiment must be a P2A1Config")
    _validate_source_experiment(context.experiment, context.source_experiment)
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
    checkpoint = context.paths.runs_root / context.experiment.source_checkpoint_relpath
    snapshot = _validated_source_checkpoint(
        checkpoint,
        context.paths.runs_root,
        context.experiment,
        context.source_experiment,
    )
    return P2B2Preflight(
        context=context,
        base=base,
        source_checkpoint=Path(os.path.abspath(checkpoint)),
        source_checkpoint_snapshot=snapshot,
    )


def _verify_p2_b2_integrity(
    preflight: P2B2Preflight, identity: tuple[int, int]
) -> str:
    head = _verify_integrity(preflight.base, identity)
    _reject_lexical_symlinks(
        preflight.context.paths.runs_root, preflight.source_checkpoint
    )
    _verify_source_checkpoint(
        preflight.source_checkpoint, preflight.source_checkpoint_snapshot
    )
    return head


def _evaluate_prompted_partition(
    *,
    model: Any,
    unet: Any,
    scheduler: Any,
    prepared: tuple[Any, ...],
    prompts: Mapping[str, Tensor],
    config: P2A1Config,
    context_seed_offset: int,
) -> tuple[dict[str, Any], ...]:
    if type(context_seed_offset) is not int or context_seed_offset < 0:
        raise ValueError("context_seed_offset must be a non-negative integer")
    if set(prompts) != set(config.defect_types):
        raise ValueError("evaluation prompt bank must match the source vocabulary")
    records: list[dict[str, Any]] = []
    for bank_index, noise_seed_offset in enumerate(NOISE_BANK_OFFSETS):
        for sample_index, sample in enumerate(prepared):
            defect_type = sample.sample.defect_type
            if defect_type not in config.defect_types:
                raise ValueError("evaluation sample has an unknown defect type")
            records.append(
                evaluate_all_pairs(
                    model=model,
                    unet=unet,
                    scheduler=scheduler,
                    prepared=sample,
                    prompt=prompts[defect_type],
                    config=config,
                    sample_index=sample_index,
                    bank_index=bank_index,
                    noise_seed_offset=noise_seed_offset,
                    context_seed_offset=context_seed_offset,
                )
            )
    return tuple(records)


def _evaluate_reloaded_p2_b2(
    *,
    checkpoint: Path,
    config: P2A1Config,
    device: torch.device,
    unet: Any,
    vae: Any,
    scheduler: Any,
    prompts: Mapping[str, Tensor],
    train_prepared: tuple[Any, ...],
    heldout_prepared: tuple[Any, ...],
    output: Path,
    expected_checkpoint_metadata: Mapping[str, Any],
    expected_checkpoint_snapshot: Mapping[str, Mapping[str, int | str]],
) -> dict[str, Any]:
    _verify_new_checkpoint(checkpoint, expected_checkpoint_snapshot)
    _, conditioner, adapter, metadata = _load_p2_a1_checkpoint(
        checkpoint, config=config, device=device
    )
    if _canonical_json(metadata, name="reloaded checkpoint metadata") != _canonical_json(
        expected_checkpoint_metadata, name="saved checkpoint metadata"
    ):
        raise ValueError("reloaded checkpoint metadata differs from saved metadata")
    _verify_new_checkpoint(checkpoint, expected_checkpoint_snapshot)
    model = P2InfillNoisePredictor(unet, conditioner, adapter).eval()
    with torch.inference_mode():
        train_records = _evaluate_prompted_partition(
            model=model,
            unet=unet,
            scheduler=scheduler,
            prepared=train_prepared,
            prompts=prompts,
            config=config,
            context_seed_offset=0,
        )
        heldout_records = _evaluate_prompted_partition(
            model=model,
            unet=unet,
            scheduler=scheduler,
            prepared=heldout_prepared,
            prompts=prompts,
            config=config,
            context_seed_offset=len(train_prepared),
        )
        visuals: list[str] = []
        visualized: set[str] = set()
        for index, prepared in enumerate(heldout_prepared):
            label = prepared.sample.defect_type
            if label not in visualized:
                visualized.add(label)
                visuals.extend(
                    _visualize_sample(
                        model=model,
                        unet=unet,
                        vae=vae,
                        scheduler=scheduler,
                        prepared=prepared,
                        prompt=prompts[label],
                        config=config,
                        sample_index=index,
                        output=output,
                    )
                )
        if visualized != set(config.defect_types):
            raise ValueError("diagnostic visuals must cover every defect type")
    _verify_new_checkpoint(checkpoint, expected_checkpoint_snapshot)
    if any(parameter.grad is not None for parameter in model.trainable_parameters()):
        raise RuntimeError("reloaded P2-B2 checkpoint accumulated gradients")
    return {
        "train_records": train_records,
        "heldout_records": heldout_records,
        "checkpoint_metadata": metadata,
        "visuals": visuals,
    }


def _execute_p2_b2(
    preflight: P2B2Preflight | Any, identity: tuple[int, int]
) -> dict[str, Any]:
    context = preflight.context
    config = context.experiment
    source_config = context.source_experiment
    _validate_source_experiment(config, source_config)
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(source_config.seed)
    torch.cuda.manual_seed_all(source_config.seed)
    torch.cuda.reset_peak_memory_stats()
    components = load_frozen_components(context.paths.model_root, device, torch.float16)
    frozen_modules = (components.unet, components.vae, components.text_encoder)
    frozen_versions = {
        name: _capture_parameter_versions(module)
        for name, module in zip(
            ("unet", "vae", "text_encoder"), frozen_modules, strict=True
        )
    }
    prompts = _encode_type_prompts(components, config, device)
    expected_prompt_lineage = _freeze_prompt_lineage(config, prompts)
    _validate_prompt_bank_lineage(config, prompts, expected_prompt_lineage)
    train_prepared = _prepare_samples(
        vae=components.vae,
        samples=preflight.base.split.train,
        device=device,
        config=source_config,
        seed_offset=0,
    )
    heldout_prepared = _prepare_samples(
        vae=components.vae,
        samples=preflight.base.split.evaluation,
        device=device,
        config=source_config,
        seed_offset=len(train_prepared),
    )
    prepared_by_id = {item.sample.sample_id: item for item in train_prepared}
    vocabulary, conditioner, adapter, source_metadata = _load_accepted_trainables(
        preflight.source_checkpoint, source_config, device
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).train()
    optimizer = _build_optimizer(model, frozen_modules, config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    schedule = balanced_training_order(
        preflight.base.split.train, config.steps, config.defect_types
    )
    generator = torch.Generator(device=device).manual_seed(
        source_config.noise_seed + 10_000_000
    )
    training: list[dict[str, Any]] = []
    num_train_timesteps = int(components.scheduler.alphas_cumprod.numel())
    expected_training_trace = _build_expected_training_trace(
        schedule=schedule,
        prepared_by_id=prepared_by_id,
        config=config,
        source_config=source_config,
        prompt_lineage=expected_prompt_lineage,
        num_train_timesteps=num_train_timesteps,
    )
    for step, sample, prepared, context_seed, context_sha256 in _resampled_training_steps(
        schedule, prepared_by_id, source_config
    ):
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
        prompt = prompts[sample.defect_type]
        metrics = _run_ranking_step(
            prepared=prepared,
            vae=components.vae,
            model=model,
            unet=components.unet,
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
            config=source_config,
        )
        prompt_sha, prompt_embedding_sha = _prompt_identity(
            config, prompts, sample.defect_type
        )
        training.append(
            {
                "step": step,
                "sample_id": sample.sample_id,
                "defect_type": sample.defect_type,
                "timestep": int(timesteps.item()),
                **asdict(metrics),
                "context_seed": context_seed,
                "context_sha256": context_sha256,
                "noise_sha256": _tensor_sha256(noise),
                "prompt_sha256": prompt_sha,
                "prompt_embedding_sha256": prompt_embedding_sha,
            }
        )
        if step == 1 or step % 100 == 0 or step == config.steps:
            print(
                json.dumps(
                    {
                        "stage": "type_prompt_training",
                        "step": step,
                        "steps": config.steps,
                        "defect_type": sample.defect_type,
                        "total": metrics.total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _validate_training_records(
        training,
        preflight.base.split,
        config,
        source_config,
        expected_training_trace,
    )
    model.eval()
    _assert_new_artifact_child(preflight.base, identity, "checkpoint")
    checkpoint = _save_p2_a1_checkpoint(
        context.output / "checkpoint",
        adapter=adapter,
        conditioner=conditioner,
        config=source_config,
        git_head=preflight.base.git_head,
        vocabulary_sha256=vocabulary.sha256,
    )
    new_checkpoint_snapshot = _checkpoint_snapshot(context.output / "checkpoint")
    _verify_new_checkpoint(context.output / "checkpoint", new_checkpoint_snapshot)
    _verify_artifact_child(context.output, "checkpoint")
    _verify_p2_b2_integrity(preflight, identity)
    _assert_new_artifact_child(preflight.base, identity, "visuals")
    del optimizer, scaler, model, adapter, conditioner
    evaluation = _evaluate_reloaded_p2_b2(
        checkpoint=context.output / "checkpoint",
        config=source_config,
        device=device,
        unet=components.unet,
        vae=components.vae,
        scheduler=components.scheduler,
        prompts=prompts,
        train_prepared=train_prepared,
        heldout_prepared=heldout_prepared,
        output=context.output,
        expected_checkpoint_metadata=checkpoint,
        expected_checkpoint_snapshot=new_checkpoint_snapshot,
    )
    _verify_new_checkpoint(context.output / "checkpoint", new_checkpoint_snapshot)
    _verify_artifact_child(context.output, "checkpoint")
    _verify_artifact_child(context.output, "visuals")
    _verify_p2_b2_integrity(preflight, identity)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated()
    for name, module in zip(
        ("unet", "vae", "text_encoder"), frozen_modules, strict=True
    ):
        _assert_parameter_versions_unchanged(module, frozen_versions[name])
        if module.training or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in module.parameters()
        ):
            raise RuntimeError(f"frozen {name} state changed during P2-B2")
    return {
        "train_records": evaluation["train_records"],
        "heldout_records": evaluation["heldout_records"],
        "training": tuple(training),
        "checkpoint": checkpoint,
        "new_checkpoint_snapshot": new_checkpoint_snapshot,
        "reloaded_checkpoint_metadata": evaluation["checkpoint_metadata"],
        "source_checkpoint_metadata": source_metadata,
        "prompt_lineage": expected_prompt_lineage,
        "expected_training_trace": expected_training_trace,
        "visuals": evaluation["visuals"],
        "peak_vram_bytes": peak,
    }


def _prompt_lineage(
    config: P2B2Config, prompts: Mapping[str, Tensor]
) -> dict[str, dict[str, str]]:
    return {
        label: {
            "text": config.prompts[label],
            "prompt_sha256": _prompt_identity(config, prompts, label)[0],
            "prompt_embedding_sha256": _prompt_identity(config, prompts, label)[1],
        }
        for label in config.defect_types
    }


def run_p2_b2(context: P2B2Context | Any) -> int:
    try:
        preflight = validate_p2_b2_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_p2_b2(preflight, identity)
        new_checkpoint_path = context.output / "checkpoint"
        new_checkpoint_snapshot = result["new_checkpoint_snapshot"]
        _verify_new_checkpoint(new_checkpoint_path, new_checkpoint_snapshot)
        if _canonical_json(
            result["checkpoint"], name="saved checkpoint metadata"
        ) != _canonical_json(
            result["reloaded_checkpoint_metadata"],
            name="reloaded checkpoint metadata",
        ):
            raise ValueError("reloaded checkpoint metadata differs from saved metadata")
        training = tuple(result["training"])
        expected_training_trace = tuple(result["expected_training_trace"])
        _validate_training_records(
            training,
            preflight.base.split,
            context.experiment,
            context.source_experiment,
            expected_training_trace,
        )
        train_records = tuple(result["train_records"])
        heldout_records = tuple(result["heldout_records"])
        summary = summarize_p2_a1_evaluation(
            train_records,
            heldout_records,
            split=preflight.base.split,
            config=context.source_experiment,
        )
        _write_jsonl_atomic(
            context.output / "metrics.jsonl", (*train_records, *heldout_records)
        )
        _verify_new_checkpoint(new_checkpoint_path, new_checkpoint_snapshot)
        _verify_p2_b2_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "training_metrics.jsonl", training)
        checkpoint_snapshot_after = _verify_new_checkpoint(
            new_checkpoint_path, new_checkpoint_snapshot
        )
        head_end = _verify_p2_b2_integrity(preflight, identity)
        prompt_lineage = result["prompt_lineage"]
        _validate_prompt_lineage(context.experiment, prompt_lineage)
        serializable_prompt_lineage = {
            label: dict(prompt_lineage[label])
            for label in context.experiment.defect_types
        }
        source_lineage = {
            "checkpoint": str(preflight.source_checkpoint),
            "hashes": dict(context.experiment.source_checkpoint_hashes),
            "metadata": result["source_checkpoint_metadata"],
        }
        report = {
            "schema_version": "p2_b2_type_prompt_training_report_v1",
            "status": "complete",
            "reason": None,
            "training_performed": True,
            "checkpoint_written": True,
            "evaluation_mode": context.experiment.evaluation_mode,
            "steps": context.experiment.steps,
            "partition_records": {
                "train": len(train_records),
                "heldout": len(heldout_records),
            },
            "evaluation_summary": summary,
            "scientific_metrics_role": "diagnostic_only",
            "checkpoint": result["checkpoint"],
            "new_checkpoint_snapshot_before": new_checkpoint_snapshot,
            "new_checkpoint_snapshot_after": checkpoint_snapshot_after,
            "source_checkpoint_lineage": source_lineage,
            "type_prompts": serializable_prompt_lineage,
            "visuals": result["visuals"],
            "peak_vram_bytes": result["peak_vram_bytes"],
            "input_hashes": preflight.base.input_hashes,
        }
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": "p2_b2_type_prompt_training_provenance_v1",
                "git_head_start": preflight.base.git_head,
                "git_head_end": head_end,
                "git_dirty": False,
                "training_performed": True,
                "checkpoint_written": True,
                "evaluation_mode": context.experiment.evaluation_mode,
                "source_checkpoint_lineage": source_lineage,
                "source_checkpoint_snapshot_before": (
                    preflight.source_checkpoint_snapshot
                ),
                "source_checkpoint_snapshot_after": (
                    preflight.source_checkpoint_snapshot
                ),
                "new_checkpoint_snapshot_before": new_checkpoint_snapshot,
                "new_checkpoint_snapshot_after": checkpoint_snapshot_after,
                "type_prompts": serializable_prompt_lineage,
                "input_hashes": preflight.base.input_hashes,
                "source_snapshot_before": preflight.base.source_snapshot,
                "source_snapshot_after": preflight.base.source_snapshot,
                "model_snapshot_before": preflight.base.model_snapshot,
                "model_snapshot_after": preflight.base.model_snapshot,
            }
        )
        _verify_new_checkpoint(new_checkpoint_path, new_checkpoint_snapshot)
        write_json_atomic(context.output / "provenance.json", provenance)
        _verify_new_checkpoint(new_checkpoint_path, new_checkpoint_snapshot)
        _verify_p2_b2_integrity(preflight, identity)
        _verify_new_checkpoint(new_checkpoint_path, new_checkpoint_snapshot)
        write_json_atomic(context.output / "report.json", report)
        _verify_new_checkpoint(new_checkpoint_path, new_checkpoint_snapshot)
        _verify_p2_b2_integrity(preflight, identity)
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": "p2_b2_type_prompt_training_report_v1",
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


def build_p2_b2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue accepted P2-A1 weights with P2-B2 Type-Prompt training"
    )
    parser.add_argument("--paths", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_p2_b2_parser().parse_args(argv)
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    source_experiment_path = Path(args.source_experiment)
    try:
        return run_p2_b2(
            P2B2Context(
                paths=load_config(paths_path),
                experiment=load_p2_b2_experiment_config(experiment_path),
                source_experiment=load_p2_a1_config(source_experiment_path),
                manifest=Path(args.manifest),
                lock=Path(args.lock),
                output=Path(args.output),
                config_paths={
                    "experiment": experiment_path,
                    "source_experiment": source_experiment_path,
                    "paths": paths_path,
                },
                cwd=Path.cwd(),
            )
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1


__all__ = []

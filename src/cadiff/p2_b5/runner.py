"""P2-B5 preflight and command wiring.

The execution path deliberately reuses the frozen P2-A1 few-shot preflight and
the P2-B2 checkpoint verifier; it never creates a fresh trainable model.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cadiff.config import ProjectConfig, load_config
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_a1.runner import P2A1Context, validate_p2_a1_preflight
from cadiff.p2_b2 import runner as b2_runner
from cadiff.p2_b2.config import P2B2Config
from cadiff.p2_b2.runner import _validated_source_checkpoint
from cadiff.provenance import collect_provenance, write_json_atomic

from .config import P2B5Config, load_p2_b5_config
from .training import type_selective_optimizer_step, validate_type_selective_record


@dataclass(frozen=True)
class P2B5Context:
    paths: ProjectConfig | Any
    experiment: P2B5Config
    source_experiment: P2A1Config
    manifest: Path
    lock: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class P2B5Preflight:
    context: P2B5Context
    base: Any
    source_checkpoint: Path
    source_checkpoint_snapshot: dict[str, dict[str, int | str]]


def _b2_checkpoint_view(config: P2B5Config) -> P2B2Config:
    """Adapt only the shared checkpoint contract to its established verifier."""

    return P2B2Config(
        schema_version="p2_b2_type_prompt_training_v1",
        category=config.category,
        defect_types=config.defect_types,
        prompts=config.prompts,
        steps=config.steps,
        learning_rate=config.learning_rate,
        type_rank=config.type_rank,
        source_checkpoint_relpath=config.source_checkpoint_relpath,
        source_checkpoint_hashes=config.source_checkpoint_hashes,
        evaluation_mode=config.evaluation_mode,
    )


def _validate_inheritance(config: P2B5Config, source: P2A1Config) -> None:
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
            raise ValueError(f"source experiment {field} is incompatible with P2-B5")


def validate_p2_b5_preflight(context: P2B5Context) -> P2B5Preflight:
    if not isinstance(context, P2B5Context):
        raise ValueError("context must be a P2B5Context")
    if not isinstance(context.experiment, P2B5Config) or not isinstance(context.source_experiment, P2A1Config):
        raise ValueError("P2-B5 requires its frozen experiment and P2-A1 source config")
    _validate_inheritance(context.experiment, context.source_experiment)
    base = validate_p2_a1_preflight(P2A1Context(
        paths=context.paths, experiment=context.source_experiment, manifest=context.manifest,
        lock=context.lock, output=context.output, config_paths=context.config_paths, cwd=context.cwd,
    ))
    checkpoint = context.paths.runs_root / context.experiment.source_checkpoint_relpath
    snapshot = _validated_source_checkpoint(checkpoint, context.paths.runs_root, _b2_checkpoint_view(context.experiment), context.source_experiment)
    return P2B5Preflight(context, base, Path(os.path.abspath(checkpoint)), snapshot)


def _validate_training_records(
    records: tuple[dict[str, Any], ...], preflight: P2B5Preflight
) -> None:
    config = preflight.context.experiment
    source = preflight.context.source_experiment
    schedule = b2_runner.balanced_training_order(
        preflight.base.split.train, config.steps, config.defect_types
    )
    if len(records) != config.steps:
        raise ValueError("training record count must equal the frozen P2-B5 steps")
    for index, (record, sample) in enumerate(zip(records, schedule, strict=True), 1):
        if record.get("step") != index or record.get("sample_id") != sample.sample_id:
            raise ValueError("training record does not match the balanced frozen schedule")
        if record.get("defect_type") != sample.defect_type:
            raise ValueError("training record defect type does not match the schedule")
        validate_type_selective_record(
            record, defect_type=sample.defect_type, source_config=source, config=config
        )
        for field in ("context_sha256", "noise_sha256", "prompt_sha256", "prompt_embedding_sha256"):
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"training record has invalid {field}")
        if record.get("finite") is not True or not isinstance(record.get("grad_l2"), float):
            raise ValueError("training record must preserve finite gradient metrics")


_STRUCTURE_RECORD_FIELDS = frozenset(
    {
        "structure_total",
        "structure_gradient",
        "structure_laplacian",
        "structure_high_frequency",
        "structure_applied",
        "structure_weight",
    }
)


def _project_base_records(
    records: tuple[dict[str, Any], ...], source_config: P2A1Config
) -> tuple[dict[str, Any], ...]:
    """Restore the exact B2 record schema solely for its independent trace check."""

    projected: list[dict[str, Any]] = []
    for record in records:
        if not _STRUCTURE_RECORD_FIELDS.issubset(record):
            raise ValueError("P2-B5 record omits required structure fields")
        base = {key: value for key, value in record.items() if key not in _STRUCTURE_RECORD_FIELDS}
        if not isinstance(base.get("base_total"), float) or not isinstance(base.get("ranking_loss"), float):
            raise ValueError("P2-B5 base metrics must be finite floats")
        base["total"] = base["base_total"] + source_config.ranking_weight * base["ranking_loss"]
        projected.append(base)
    return tuple(projected)


def _execute_p2_b5(preflight: P2B5Preflight, identity: tuple[int, int]) -> dict[str, Any]:
    """Reuse B2's complete load/train/save/delete/reload/evaluate path narrowly.

    The only substituted operation is the optimizer step; all frozen component,
    source-checkpoint, artifact, visual, and reloaded-evaluation code remains the
    established P2-B2 implementation.
    """

    context = preflight.context
    b2_context = b2_runner.P2B2Context(
        paths=context.paths,
        experiment=_b2_checkpoint_view(context.experiment),
        source_experiment=context.source_experiment,
        manifest=context.manifest,
        lock=context.lock,
        output=context.output,
        config_paths=context.config_paths,
        cwd=context.cwd,
    )
    b2_preflight = b2_runner.P2B2Preflight(
        context=b2_context,
        base=preflight.base,
        source_checkpoint=preflight.source_checkpoint,
        source_checkpoint_snapshot=preflight.source_checkpoint_snapshot,
    )
    original_step = b2_runner._run_ranking_step
    original_validate = b2_runner._validate_training_records

    def _step(*, prepared: Any, vae: Any, **kwargs: Any) -> Any:
        original_config = kwargs.pop("config", None)
        if not isinstance(original_config, P2A1Config):
            raise ValueError("B2 execution seam must supply its P2-A1 objective config")
        if original_config is not context.source_experiment:
            raise ValueError("B2 execution seam changed the frozen P2-A1 objective config")
        try:
            pixels = prepared.sample.pixels
            mask = prepared.sample.mask
            target_latent = kwargs["target_latent"]
        except (AttributeError, KeyError) as error:
            raise ValueError("prepared sample must expose pixels and mask") from error
        if not isinstance(pixels, torch.Tensor) or not isinstance(mask, torch.Tensor) or not isinstance(target_latent, torch.Tensor):
            raise ValueError("prepared image-space structure inputs must be tensors")
        target_image = (
            ((pixels.float() + 1.0) / 2.0)
            .clamp(0.0, 1.0)
            .unsqueeze(0)
            .to(device=target_latent.device)
        )
        image_mask = mask.unsqueeze(0).to(device=target_latent.device, dtype=torch.float32)
        return type_selective_optimizer_step(
            **kwargs,
            vae=vae,
            target_image=target_image,
            image_mask=image_mask,
            source_config=original_config,
            config=context.experiment,
        )

    def _validate(
        records: tuple[dict[str, Any], ...],
        split: object,
        config: object,
        source_config: object,
        expected_trace: object,
    ) -> None:
        if config is not b2_context.experiment or source_config is not context.source_experiment:
            raise ValueError("B2 validator seam changed its frozen config objects")
        projected = _project_base_records(tuple(records), source_config)
        original_validate(projected, split, config, source_config, expected_trace)
        _validate_training_records(tuple(records), preflight)

    b2_runner._run_ranking_step = _step
    b2_runner._validate_training_records = _validate
    try:
        return b2_runner._execute_p2_b2(b2_preflight, identity)
    finally:
        b2_runner._run_ranking_step = original_step
        b2_runner._validate_training_records = original_validate


def run_p2_b5(context: P2B5Context | Any) -> int:
    """Run P2-B5 with B2's atomic artifact and cleanup contract."""

    try:
        preflight = validate_p2_b5_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = b2_runner._output_identity(context.output)
    try:
        result = _execute_p2_b5(preflight, identity)
        checkpoint_path = context.output / "checkpoint"
        snapshot = result["new_checkpoint_snapshot"]
        b2_runner._verify_new_checkpoint(checkpoint_path, snapshot)
        if b2_runner._canonical_json(result["checkpoint"], name="saved checkpoint") != b2_runner._canonical_json(result["reloaded_checkpoint_metadata"], name="reloaded checkpoint"):
            raise ValueError("reloaded checkpoint metadata differs from saved metadata")
        training = tuple(result["training"])
        _validate_training_records(training, preflight)
        train_records, heldout_records = tuple(result["train_records"]), tuple(result["heldout_records"])
        summary = b2_runner.summarize_p2_a1_evaluation(train_records, heldout_records, split=preflight.base.split, config=context.source_experiment)
        b2_runner._write_jsonl_atomic(context.output / "metrics.jsonl", (*train_records, *heldout_records))
        b2_runner._write_jsonl_atomic(context.output / "training_metrics.jsonl", training)
        source_lineage = {"checkpoint": str(preflight.source_checkpoint), "hashes": dict(context.experiment.source_checkpoint_hashes), "metadata": result["source_checkpoint_metadata"]}
        report = {
            "schema_version": "p2_b5_type_selective_texture_report_v1", "status": "complete", "reason": None,
            "training_performed": True, "checkpoint_written": True, "evaluation_mode": context.experiment.evaluation_mode,
            "steps": context.experiment.steps, "partition_records": {"train": len(train_records), "heldout": len(heldout_records)},
            "evaluation_summary": summary, "scientific_metrics_role": "diagnostic_only", "checkpoint": result["checkpoint"],
            "source_checkpoint_lineage": source_lineage, "visuals": result["visuals"], "peak_vram_bytes": result["peak_vram_bytes"],
            "structure": {
                "weight": context.experiment.structure_weight,
                "types": list(context.experiment.structure_types),
                "space": context.experiment.structure_space,
                "erosion_kernel": context.experiment.structure_erosion_kernel,
            },
            "input_hashes": preflight.base.input_hashes,
        }
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update({"schema_version": "p2_b5_type_selective_texture_provenance_v1", "git_head_start": preflight.base.git_head, "git_head_end": b2_runner._verify_p2_b2_integrity(preflight, identity), "git_dirty": False, "training_performed": True, "checkpoint_written": True, "source_checkpoint_lineage": source_lineage, "structure": report["structure"]})
        write_json_atomic(context.output / "provenance.json", provenance)
        write_json_atomic(context.output / "report.json", report)
        b2_runner._verify_new_checkpoint(checkpoint_path, snapshot)
        b2_runner._verify_p2_b2_integrity(preflight, identity)
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {"schema_version": "p2_b5_type_selective_texture_report_v1", "status": "failed", "reason": "runtime_or_integrity_failure"}
        try:
            b2_runner._verify_output_identity(preflight.base, identity)
            b2_runner._cleanup_partial(context.output)
            write_json_atomic(context.output / "report.json", failed)
        except (OSError, RuntimeError):
            pass
        print(json.dumps(failed, sort_keys=True))
        return 1


def build_p2_b5_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2-B5 type-selective texture continuation")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_p2_b5_parser().parse_args(argv)
    paths_path, experiment_path, source_path = Path(args.paths), Path(args.experiment), Path(args.source_experiment)
    try:
        return run_p2_b5(P2B5Context(
            paths=load_config(paths_path), experiment=load_p2_b5_config(experiment_path),
            source_experiment=load_p2_a1_config(source_path), manifest=Path(args.manifest), lock=Path(args.lock),
            output=Path(args.output), config_paths={"paths": paths_path, "experiment": experiment_path, "source_experiment": source_path}, cwd=Path.cwd(),
        ))
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1

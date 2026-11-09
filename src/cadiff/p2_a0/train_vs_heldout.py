from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cadiff.config import ProjectConfig, load_config
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p2_a0.all_pairs import (
    NOISE_BANK_OFFSETS,
    AllPairsContext,
    AllPairsPreflight,
    _load_checkpoint,
    _validate_execution_records,
    _verify_all_pairs_integrity,
    evaluate_all_pairs,
    summarize_all_pairs,
    validate_all_pairs_preflight,
)
from cadiff.p2_a0.fewshot import (
    FewshotSplit,
    _assert_parameter_versions_unchanged,
    _capture_parameter_versions,
    _output_identity,
    _prepare_samples,
    _verify_output_identity,
    _write_jsonl_atomic,
)
from cadiff.p2_a0.fewshot_config import P2A0FewshotConfig, load_p2_a0_fewshot_config
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.provenance import collect_provenance, write_json_atomic

SCHEMA_VERSION = "p2_a0_type_routed_train_vs_heldout_all_pairs_v1"


@dataclass(frozen=True)
class TrainVsHeldoutContext:
    paths: ProjectConfig
    experiment: P2A0FewshotConfig
    manifest: Path
    lock: Path
    source_run: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


def _metric(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def summarize_train_vs_heldout(
    train: dict[str, Any],
    heldout: dict[str, Any],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(labels, tuple) or not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must be a unique non-empty tuple")
    try:
        train_types = train["combined"]["per_true_type"]
        heldout_types = heldout["combined"]["per_true_type"]
    except (KeyError, TypeError) as error:
        raise ValueError("split summary schema is invalid") from error
    if set(train_types) != set(labels) or set(heldout_types) != set(labels):
        raise ValueError("split summaries must contain every label")
    result: dict[str, Any] = {}
    for label in labels:
        train_type = train_types[label]
        heldout_type = heldout_types[label]
        train_accuracy = _metric(
            train_type.get("top1_accuracy"), name=f"train/{label}/top1_accuracy"
        )
        heldout_accuracy = _metric(
            heldout_type.get("top1_accuracy"), name=f"heldout/{label}/top1_accuracy"
        )
        train_normal = _metric(
            train_type.get("correct_normal_ratio"),
            name=f"train/{label}/correct_normal_ratio",
        )
        heldout_normal = _metric(
            heldout_type.get("correct_normal_ratio"),
            name=f"heldout/{label}/correct_normal_ratio",
        )
        train_paired = train_type.get("paired_against")
        heldout_paired = heldout_type.get("paired_against")
        expected_wrong = set(labels) - {label}
        if (
            not isinstance(train_paired, dict)
            or not isinstance(heldout_paired, dict)
            or set(train_paired) != expected_wrong
            or set(heldout_paired) != expected_wrong
        ):
            raise ValueError(f"paired summary is invalid for {label}")
        paired: dict[str, Any] = {}
        for wrong in labels:
            if wrong == label:
                continue
            train_win = _metric(
                train_paired[wrong].get("win_rate"),
                name=f"train/{label}/{wrong}/win_rate",
            )
            heldout_win = _metric(
                heldout_paired[wrong].get("win_rate"),
                name=f"heldout/{label}/{wrong}/win_rate",
            )
            train_delta = _metric(
                train_paired[wrong].get("median_delta"),
                name=f"train/{label}/{wrong}/median_delta",
            )
            heldout_delta = _metric(
                heldout_paired[wrong].get("median_delta"),
                name=f"heldout/{label}/{wrong}/median_delta",
            )
            paired[wrong] = {
                "train_win_rate": train_win,
                "heldout_win_rate": heldout_win,
                "win_rate_drop": train_win - heldout_win,
                "train_median_delta": train_delta,
                "heldout_median_delta": heldout_delta,
                "median_delta_drop": train_delta - heldout_delta,
            }
        result[label] = {
            "train_top1_accuracy": train_accuracy,
            "heldout_top1_accuracy": heldout_accuracy,
            "top1_accuracy_drop": train_accuracy - heldout_accuracy,
            "train_correct_normal_ratio": train_normal,
            "heldout_correct_normal_ratio": heldout_normal,
            "correct_normal_ratio_increase": heldout_normal - train_normal,
            "paired_against": paired,
        }
    return result


def _evaluate_partition(
    *,
    model: Any,
    unet: Any,
    scheduler: Any,
    prepared: tuple[Any, ...],
    prompt: torch.Tensor,
    config: P2A0FewshotConfig,
    context_seed_offset: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        evaluate_all_pairs(
            model=model,
            unet=unet,
            scheduler=scheduler,
            prepared=sample,
            prompt=prompt,
            config=config,
            sample_index=sample_index,
            bank_index=bank_index,
            noise_seed_offset=offset,
            context_seed_offset=context_seed_offset,
        )
        for bank_index, offset in enumerate(NOISE_BANK_OFFSETS)
        for sample_index, sample in enumerate(prepared)
    )


def _execute_train_vs_heldout(preflight: AllPairsPreflight) -> dict[str, Any]:
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
    heldout_prepared = _prepare_samples(
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
    with torch.inference_mode():
        train_records = _evaluate_partition(
            model=model,
            unet=components.unet,
            scheduler=components.scheduler,
            prepared=train_prepared,
            prompt=prompt,
            config=config,
            context_seed_offset=0,
        )
        heldout_records = _evaluate_partition(
            model=model,
            unet=components.unet,
            scheduler=components.scheduler,
            prepared=heldout_prepared,
            prompt=prompt,
            config=config,
            context_seed_offset=len(preflight.split.train),
        )
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated()
    _assert_parameter_versions_unchanged(components.unet, unet_versions)
    if any(parameter.grad is not None for parameter in components.unet.parameters()):
        raise RuntimeError("frozen U-Net accumulated gradients")
    if any(parameter.grad is not None for parameter in model.trainable_parameters()):
        raise RuntimeError("read-only trainables accumulated gradients")
    return {
        "train_records": train_records,
        "heldout_records": heldout_records,
        "peak_vram_bytes": peak,
        "checkpoint_metadata": metadata,
    }


def _validate_partition_records(
    records: tuple[dict[str, Any], ...],
    *,
    samples: tuple[Any, ...],
    context_seed_offset: int,
    config: P2A0FewshotConfig,
) -> None:
    _validate_execution_records(
        records,
        split=FewshotSplit(
            train=tuple(object() for _ in range(context_seed_offset)),  # type: ignore[arg-type]
            evaluation=samples,
        ),
        config=config,
    )


def _cleanup(output: Path) -> None:
    for name in (
        "train_metrics.jsonl",
        "heldout_metrics.jsonl",
        "provenance.json",
        "report.json",
    ):
        path = output / name
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)


def run_train_vs_heldout(context: TrainVsHeldoutContext | AllPairsContext) -> int:
    try:
        preflight = validate_all_pairs_preflight(context)  # type: ignore[arg-type]
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_train_vs_heldout(preflight)
        train_records = tuple(result["train_records"])
        heldout_records = tuple(result["heldout_records"])
        _validate_partition_records(
            train_records,
            samples=preflight.split.train,
            context_seed_offset=0,
            config=context.experiment,
        )
        _validate_partition_records(
            heldout_records,
            samples=preflight.split.evaluation,
            context_seed_offset=len(preflight.split.train),
            config=context.experiment,
        )
        train_summary = summarize_all_pairs(
            train_records, tuple(context.experiment.defect_types)
        )
        heldout_summary = summarize_all_pairs(
            heldout_records, tuple(context.experiment.defect_types)
        )
        generalization = summarize_train_vs_heldout(
            train_summary,
            heldout_summary,
            tuple(context.experiment.defect_types),
        )
        head_end = _verify_all_pairs_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "train_metrics.jsonl", train_records)
        _verify_all_pairs_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "heldout_metrics.jsonl", heldout_records)
        _verify_all_pairs_integrity(preflight, identity)
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "reason": None,
            "diagnostic_only": True,
            "training_performed": False,
            "checkpoint_written": False,
            "noise_seed_offsets": list(NOISE_BANK_OFFSETS),
            "partition_records": {
                "train": len(train_records),
                "heldout": len(heldout_records),
            },
            "summary": {
                "train": train_summary,
                "heldout": heldout_summary,
                "generalization": generalization,
            },
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
                _cleanup(context.output)
                write_json_atomic(context.output / "report.json", failed)
            except OSError:
                pass
        print(json.dumps(failed, sort_keys=True))
        return 1


def build_train_vs_heldout_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Type-Routed Heads train-vs-heldout all-pairs diagnostic"
    )
    parser.add_argument("--paths", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_train_vs_heldout_parser().parse_args(argv)
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        return run_train_vs_heldout(
            TrainVsHeldoutContext(
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

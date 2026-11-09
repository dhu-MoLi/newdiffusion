from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from cadiff.config import ProjectConfig, load_config
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.all_pairs import _strict_json_object
from cadiff.p2_a0.fewshot import (
    FewshotContext,
    FewshotPreflight,
    FewshotSplit,
    _active_defect_types,
    _assert_parameter_versions_unchanged,
    _capture_parameter_versions,
    _output_identity,
    _prepare_samples,
    _resampled_training_steps,
    _verify_integrity,
    _verify_output_identity,
    _visualize_sample,
    _write_jsonl_atomic,
    balanced_training_order,
    validate_fewshot_preflight,
)
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_a1.evaluation import (
    NOISE_BANK_OFFSETS,
    evaluate_partition,
    summarize_p2_a1_evaluation,
)
from cadiff.p2_a1.model import FactorizedMaskedInfillAdapter
from cadiff.p2_a1.ranking import ranking_optimizer_step
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic


@dataclass(frozen=True)
class P2A1Context:
    paths: ProjectConfig
    experiment: P2A1Config
    manifest: Path
    lock: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


def _evaluation_mode(config: P2A1Config) -> str:
    if not isinstance(config, P2A1Config):
        raise ValueError("evaluation mode requires a P2A1Config")
    if config.category == "bottle":
        return "frozen_gate"
    if config.category in {"hazelnut", "cable"}:
        return "diagnostic_only"
    raise ValueError("P2-A1 evaluation category is unsupported")


def validate_p2_a1_preflight(context: P2A1Context) -> FewshotPreflight:
    if not isinstance(context, P2A1Context):
        raise ValueError("context must be a P2A1Context")
    if not isinstance(context.experiment, P2A1Config):
        raise ValueError("experiment must be a P2A1Config")
    return validate_fewshot_preflight(
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


def _save_p2_a1_checkpoint(
    output: Path,
    *,
    adapter: torch.nn.Module,
    conditioner: torch.nn.Module,
    config: P2A1Config,
    git_head: str,
    vocabulary_sha256: str,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"checkpoint output already exists: {output}")
    if not isinstance(config, P2A1Config):
        raise ValueError("checkpoint config must be a P2A1Config")
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
        "schema_version": "p2_a1_factorized_checkpoint_v1",
        "git_head": git_head,
        "vocabulary_sha256": vocabulary_sha256,
        "config": asdict(config),
        "files": files,
    }
    write_json_atomic(output / "metadata.json", metadata)
    return metadata


def _load_p2_a1_checkpoint(
    checkpoint: Path,
    *,
    config: P2A1Config,
    device: torch.device,
) -> tuple[DefectVocabulary, DefectConditioner, FactorizedMaskedInfillAdapter, dict[str, Any]]:
    metadata = _strict_json_object(checkpoint / "metadata.json")
    if metadata.get("schema_version") != "p2_a1_factorized_checkpoint_v1":
        raise ValueError("checkpoint schema is incompatible with P2-A1")
    if metadata.get("config") != json.loads(
        json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, allow_nan=False)
    ):
        raise ValueError("checkpoint config does not match P2-A1")
    files = metadata.get("files")
    expected = {
        "adapter.safetensors": checkpoint / "adapter.safetensors",
        "conditioner.safetensors": checkpoint / "conditioner.safetensors",
    }
    if not isinstance(files, dict) or set(files) != set(expected):
        raise ValueError("checkpoint file manifest is invalid")
    for name, path in expected.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint file is not regular: {name}")
        if files[name] != {"sha256": sha256_file(path), "size": path.stat().st_size}:
            raise ValueError(f"checkpoint file hash or size changed: {name}")
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    if metadata.get("vocabulary_sha256") != vocabulary.sha256:
        raise ValueError("checkpoint vocabulary does not match P2-A1")
    conditioner = DefectConditioner(vocabulary, 128).to(device=device, dtype=torch.float32)
    adapter = FactorizedMaskedInfillAdapter(config).to(
        device=device, dtype=torch.float32
    )
    conditioner.load_state_dict(
        load_file(checkpoint / "conditioner.safetensors", device="cpu"), strict=True
    )
    adapter.load_state_dict(
        load_file(checkpoint / "adapter.safetensors", device="cpu"), strict=True
    )
    conditioner.eval()
    adapter.eval()
    return vocabulary, conditioner, adapter, metadata


def _validate_ranking_records(
    records: tuple[dict[str, Any], ...],
    split: FewshotSplit,
    config: P2A1Config,
) -> None:
    if len(records) != config.steps:
        raise ValueError(f"training record count must be {config.steps}")
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
    }
    context_hashes: set[str] = set()
    for index, (record, sample) in enumerate(zip(records, schedule, strict=True), start=1):
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ValueError(f"training record {index} has an invalid exact schema")
        if (
            record["step"] != index
            or record["sample_id"] != sample.sample_id
            or record["defect_type"] != sample.defect_type
        ):
            raise ValueError(f"training record {index} does not match the schedule")
        timestep = record["timestep"]
        if type(timestep) is not int or not 0 <= timestep < 1000:
            raise ValueError(f"training record {index} has an invalid timestep")
        expected_seed = config.context_fill_seed + 20_000_000 + index
        if type(record["context_seed"]) is not int or record["context_seed"] != expected_seed:
            raise ValueError(f"training record {index} has an invalid context_seed")
        context_sha = record["context_sha256"]
        if (
            not isinstance(context_sha, str)
            or len(context_sha) != 64
            or any(character not in "0123456789abcdef" for character in context_sha)
        ):
            raise ValueError(f"training record {index} has an invalid context_sha256")
        context_hashes.add(context_sha)
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
            value = record[field]
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"training record {index} has invalid {field}")
        if record["correct_negative_ratio"] <= 0.0:
            raise ValueError(f"training record {index} has invalid correct_negative_ratio")
        expected_ranking = max(
            0.0, record["correct_negative_ratio"] - config.ranking_target_ratio
        )
        if not math.isclose(record["ranking_loss"], expected_ranking, abs_tol=1e-6):
            raise ValueError(f"training record {index} violates the ranking formula")
        expected_total = record["base_total"] + config.ranking_weight * expected_ranking
        if not math.isclose(record["total"], expected_total, abs_tol=1e-6):
            raise ValueError(f"training record {index} violates the total formula")
        hardest = record["hardest_negative"]
        if hardest not in {*config.defect_types, "normal"} - {sample.defect_type}:
            raise ValueError(f"training record {index} has invalid hardest_negative")
        if record["grad_l2"] <= 0.0 or record["finite"] is not True:
            raise ValueError(f"training record {index} is not finite")
    if len(context_hashes) != config.steps:
        raise ValueError("training contexts must be unique per step")


def _assert_new_artifact_child(
    preflight: FewshotPreflight, identity: tuple[int, int], name: str
) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("artifact child name must be a direct child")
    _verify_integrity(preflight, identity)
    output = preflight.context.output.resolve(strict=True)
    candidate = preflight.context.output / name
    if candidate.is_symlink() or candidate.exists():
        raise RuntimeError(f"artifact child must be absent before writing: {name}")
    if candidate.parent.resolve(strict=True) != output:
        raise RuntimeError(f"artifact child escapes output: {name}")
    return candidate


def _verify_artifact_child(output: Path, name: str) -> None:
    root = output / name
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"artifact child is not a real directory: {name}")
    resolved_output = output.resolve(strict=True)
    if not root.resolve(strict=True).is_relative_to(resolved_output):
        raise RuntimeError(f"artifact child escapes output: {name}")
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or not (candidate.is_dir() or candidate.is_file()):
            raise RuntimeError(f"artifact child contains an unsafe node: {name}")
        if not candidate.resolve(strict=True).is_relative_to(resolved_output):
            raise RuntimeError(f"artifact child contains an escaping node: {name}")


def _evaluate_reloaded_p2_a1(
    *,
    checkpoint: Path,
    config: P2A1Config,
    device: torch.device,
    unet: Any,
    scheduler: Any,
    prompt: torch.Tensor,
    train_prepared: tuple[Any, ...],
    heldout_prepared: tuple[Any, ...],
) -> dict[str, Any]:
    _, conditioner, adapter, metadata = _load_p2_a1_checkpoint(
        checkpoint, config=config, device=device
    )
    model = P2InfillNoisePredictor(unet, conditioner, adapter).eval()
    with torch.inference_mode():
        train_records = evaluate_partition(
            model=model,
            unet=unet,
            scheduler=scheduler,
            prepared=train_prepared,
            prompt=prompt,
            config=config,
            context_seed_offset=0,
        )
        heldout_records = evaluate_partition(
            model=model,
            unet=unet,
            scheduler=scheduler,
            prepared=heldout_prepared,
            prompt=prompt,
            config=config,
            context_seed_offset=len(train_prepared),
        )
    if any(parameter.grad is not None for parameter in model.trainable_parameters()):
        raise RuntimeError("reloaded read-only trainables accumulated gradients")
    return {
        "train_records": train_records,
        "heldout_records": heldout_records,
        "checkpoint_metadata": metadata,
    }


def _execute_p2_a1(
    preflight: FewshotPreflight, identity: tuple[int, int]
) -> dict[str, Any]:
    context = preflight.context
    config = context.experiment
    if not isinstance(config, P2A1Config):
        raise ValueError("P2-A1 execution requires a P2A1Config")
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
    adapter = FactorizedMaskedInfillAdapter(config).to(
        device=device, dtype=torch.float32
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    schedule = balanced_training_order(
        preflight.split.train, config.steps, _active_defect_types(config)
    )
    generator = torch.Generator(device=device).manual_seed(config.noise_seed + 10_000_000)
    training: list[dict[str, Any]] = []
    num_train_timesteps = int(components.scheduler.alphas_cumprod.numel())
    for step, sample, prepared, context_seed, context_sha256 in _resampled_training_steps(
        schedule, prepared_by_id, config
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
        metrics = ranking_optimizer_step(
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
            config=config,
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
            }
        )
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
    _assert_new_artifact_child(preflight, identity, "checkpoint")
    checkpoint = _save_p2_a1_checkpoint(
        context.output / "checkpoint",
        adapter=adapter,
        conditioner=conditioner,
        config=config,
        git_head=preflight.git_head,
        vocabulary_sha256=vocabulary.sha256,
    )
    _verify_artifact_child(context.output, "checkpoint")
    _verify_integrity(preflight, identity)
    visuals: list[str] = []
    visualized: set[str] = set()
    _assert_new_artifact_child(preflight, identity, "visuals")
    for index, prepared in enumerate(evaluation_prepared):
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
    _verify_artifact_child(context.output, "visuals")
    _verify_integrity(preflight, identity)
    del optimizer, scaler, model, adapter, conditioner
    evaluation = _evaluate_reloaded_p2_a1(
        checkpoint=context.output / "checkpoint",
        config=config,
        device=device,
        unet=components.unet,
        scheduler=components.scheduler,
        prompt=prompt,
        train_prepared=train_prepared,
        heldout_prepared=evaluation_prepared,
    )
    _verify_artifact_child(context.output, "checkpoint")
    _verify_artifact_child(context.output, "visuals")
    _verify_integrity(preflight, identity)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated()
    _assert_parameter_versions_unchanged(components.unet, unet_versions)
    if any(parameter.grad is not None for parameter in components.unet.parameters()):
        raise RuntimeError("frozen U-Net accumulated gradients")
    return {
        "train_records": evaluation["train_records"],
        "heldout_records": evaluation["heldout_records"],
        "training": tuple(training),
        "checkpoint": checkpoint,
        "visuals": visuals,
        "peak_vram_bytes": peak,
    }


def _cleanup_partial(output: Path) -> None:
    for name in (
        "metrics.jsonl",
        "training_metrics.jsonl",
        "provenance.json",
        "report.json",
    ):
        candidate = output / name
        if candidate.is_file() or candidate.is_symlink():
            candidate.unlink(missing_ok=True)
    for name in ("checkpoint", "visuals"):
        candidate = output / name
        if candidate.is_symlink():
            candidate.unlink(missing_ok=True)
        elif candidate.is_dir():
            shutil.rmtree(candidate)


def run_p2_a1(context: P2A1Context | Any) -> int:
    try:
        preflight = validate_p2_a1_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(json.dumps({"reason": "preflight_failure", "status": "failed"}, sort_keys=True))
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    identity = _output_identity(context.output)
    try:
        result = _execute_p2_a1(preflight, identity)
        train_records = tuple(result["train_records"])
        heldout_records = tuple(result["heldout_records"])
        training = tuple(result["training"])
        _validate_ranking_records(training, preflight.split, context.experiment)
        summary = summarize_p2_a1_evaluation(
            train_records,
            heldout_records,
            split=preflight.split,
            config=context.experiment,
        )
        head_end = _verify_integrity(preflight, identity)
        _write_jsonl_atomic(
            context.output / "metrics.jsonl", (*train_records, *heldout_records)
        )
        _verify_integrity(preflight, identity)
        _write_jsonl_atomic(context.output / "training_metrics.jsonl", training)
        _verify_integrity(preflight, identity)
        report = {
            "schema_version": "p2_a1_factorized_report_v1",
            "status": "complete",
            "reason": None,
            "training_performed": True,
            "checkpoint_written": True,
            "evaluation_mode": _evaluation_mode(context.experiment),
            "noise_seed_offsets": list(NOISE_BANK_OFFSETS),
            "partition_records": {
                "train": len(train_records),
                "heldout": len(heldout_records),
            },
            "evaluation_summary": summary,
            "scientific_gate": summary["decision"],
            "checkpoint": result["checkpoint"],
            "visuals": result["visuals"],
            "peak_vram_bytes": result["peak_vram_bytes"],
            "input_hashes": preflight.input_hashes,
        }
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": "p2_a1_factorized_provenance_v1",
                "git_head_start": preflight.git_head,
                "git_head_end": head_end,
                "git_dirty": False,
                "training_performed": True,
                "checkpoint_written": True,
                "evaluation_mode": _evaluation_mode(context.experiment),
                "noise_seed_offsets": list(NOISE_BANK_OFFSETS),
                "scientific_gate": summary["decision"],
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
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": "p2_a1_factorized_report_v1",
            "status": "failed",
            "reason": "runtime_or_integrity_failure",
        }
        try:
            _verify_output_identity(preflight, identity)  # type: ignore[arg-type]
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


def build_p2_a1_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train P2-A1 factorized type routing")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_p2_a1_parser().parse_args(argv)
    paths_path = Path(args.paths)
    experiment_path = Path(args.experiment)
    try:
        return run_p2_a1(
            P2A1Context(
                paths=load_config(paths_path),
                experiment=load_p2_a1_config(experiment_path),
                manifest=Path(args.manifest),
                lock=Path(args.lock),
                output=Path(args.output),
                config_paths={"experiment": experiment_path, "paths": paths_path},
                cwd=Path.cwd(),
            )
        )
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error), "exit_code": 1}, sort_keys=True), flush=True)
        return 1

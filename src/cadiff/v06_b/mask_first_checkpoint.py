from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from cadiff.provenance import sha256_file, write_json_atomic

from .config import (
    CABLE_SCHEME2_A2_TO_H_OUTPUT,
    CABLE_SCHEME2_A_TO_H_OUTPUT,
    CABLE_SCHEME2_OUTPUT,
    V06BMaskFirstConfig,
    is_cable_scheme2,
)
from .planner import PlannerNetwork
from .planner_threshold import ThresholdCalibration
from .source import SourceRendererManifest

CHECKPOINT_SCHEMA = "v06_b_mask_first_checkpoint_v1"
INFERENCE_SCHEMA = "v06_b_mask_first_inference_v1"
CALIBRATION_POLICY = "fewshot_train_geometry_support_macro_dice_v1"
PLANNER_SEED_MODE = "deterministic_invariant_v1"
_P2_B2_RENDERER_ID = "p2_b2_hazelnut_type_prompt_attempt1"
_CABLE_P2_B2_RENDERER_ID = "p2_b2_cable_type_prompt_scheme2"


_CONTINUE_FREE_CONFIG_KEYS = frozenset(
    {
        "seed",
        "output_relpath",
        "panel_seeds",
        "blind_panel_seed",
    }
)


def architecture_config_payload(config: V06BMaskFirstConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    return {
        key: value
        for key, value in payload.items()
        if key not in _CONTINUE_FREE_CONFIG_KEYS
    }


def validate_mask_first_renderer_continue_configs(
    continue_config: V06BMaskFirstConfig,
    parent_config: V06BMaskFirstConfig,
) -> None:
    if not is_cable_scheme2(continue_config) or not is_cable_scheme2(parent_config):
        raise ValueError("renderer continue is only authorized for Cable scheme 2")
    if architecture_config_payload(continue_config) != architecture_config_payload(
        parent_config
    ):
        raise ValueError("continue config must match the parent mask-first architecture")
    if continue_config.seed == parent_config.seed:
        raise ValueError("continue seed must differ so renderer noise is not repeated")
    if continue_config.output_relpath == parent_config.output_relpath:
        raise ValueError("continue output must not overwrite the parent run")


def validate_mask_first_a_to_h_configs(
    child_config: V06BMaskFirstConfig,
    parent_config: V06BMaskFirstConfig,
) -> None:
    if not is_cable_scheme2(child_config) or not is_cable_scheme2(parent_config):
        raise ValueError("A→H is only authorized for Cable scheme 2")
    if parent_config.output_relpath != CABLE_SCHEME2_OUTPUT:
        raise ValueError("A→H parent must be the frozen Cable scheme-2 mask-first run")
    if child_config.output_relpath != CABLE_SCHEME2_A_TO_H_OUTPUT:
        raise ValueError("A→H output must be the authorized scheme2-a-to-h relpath")
    if architecture_config_payload(child_config) != architecture_config_payload(parent_config):
        raise ValueError("A→H config must match the parent mask-first architecture")
    if child_config.seed == parent_config.seed:
        raise ValueError("A→H seed must differ from the parent run")


def validate_mask_first_a2_to_h_configs(
    child_config: V06BMaskFirstConfig,
    parent_config: V06BMaskFirstConfig,
) -> None:
    if not is_cable_scheme2(child_config) or not is_cable_scheme2(parent_config):
        raise ValueError("a2→H is only authorized for Cable scheme 2")
    if parent_config.output_relpath != CABLE_SCHEME2_OUTPUT:
        raise ValueError("a2→H parent must be the frozen Cable scheme-2 mask-first run")
    if child_config.output_relpath != CABLE_SCHEME2_A2_TO_H_OUTPUT:
        raise ValueError("a2→H output must be the authorized scheme2-a2-to-h relpath")
    if architecture_config_payload(child_config) != architecture_config_payload(parent_config):
        raise ValueError("a2→H config must match the parent mask-first architecture")
    if child_config.seed == parent_config.seed:
        raise ValueError("a2→H seed must differ from the parent run")


def renderer_initialization_label(config: V06BMaskFirstConfig) -> str:
    if is_cable_scheme2(config):
        return "p2_b2_cable_trained_default"
    if config.category == "cable":
        return "p2_b2_hazelnut_adapter_architecture"
    if config.category != "hazelnut":
        raise ValueError("mask-first category must be hazelnut or cable")
    return "p2_b2_frozen_default"


def type_condition_initialization_label(config: V06BMaskFirstConfig) -> str:
    if is_cable_scheme2(config):
        return "p2_b2_cable_trained_default"
    if config.category == "cable":
        return "fresh_cable_vocabulary"
    return "p2_b2_frozen_default"


_SOURCE_FILES = (
    "adapter.safetensors",
    "conditioner.safetensors",
    "metadata.json",
)
_TENSOR_FILES = {
    "planner.safetensors": "planner",
    "adapter.safetensors": "adapter",
    "conditioner.safetensors": "conditioner",
}
_PAYLOAD_FILES = (*_TENSOR_FILES, "inference_config.json")
_CHECKPOINT_LAYOUT = {*_PAYLOAD_FILES, "metadata.json"}


@dataclass(frozen=True)
class ResolvedMaskFirstInference:
    realization_threshold: float
    calibration_policy: str
    support_record_ids: tuple[str, ...]
    support_digest: str
    planner_seed_mode: Literal["deterministic_invariant_v1"]
    defect_types: tuple[str, ...]
    ddim_steps: int


@dataclass(frozen=True)
class LoadedMaskFirstCheckpoint:
    metadata: Mapping[str, Any]
    inference: ResolvedMaskFirstInference


def _hash(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal hash")
    return value


def _config_payload(config: V06BMaskFirstConfig) -> dict[str, Any]:
    if (
        not isinstance(config, V06BMaskFirstConfig)
        or config.schema_version != "v06_b_mask_first_v1"
    ):
        raise ValueError("config must be a v06_b_mask_first_v1 configuration")
    return {
        "schema_version": config.schema_version,
        "category": config.category,
        "defect_types": list(config.defect_types),
        "source_renderer_manifest": config.source_renderer_manifest,
        "source_experiment_config": config.source_experiment_config,
        "source_model_config": config.source_model_config,
        "image_size": config.image_size,
        "proposal_radius": config.proposal_radius,
        "support_corner_width": config.support_corner_width,
        "planner_base_channels": config.planner_base_channels,
        "planner_type_embedding_dim": config.planner_type_embedding_dim,
        "planner_seed_noise_channels": config.planner_seed_noise_channels,
        "planner_training_steps": config.planner_training_steps,
        "planner_learning_rate": config.planner_learning_rate,
        "planner_gradient_clip": config.planner_gradient_clip,
        "planner_dice_weight": config.planner_dice_weight,
        "planner_calibration_normal_count": config.planner_calibration_normal_count,
        "planner_threshold_candidates": list(config.planner_threshold_candidates),
        "renderer_training_steps": config.renderer_training_steps,
        "renderer_learning_rates": dict(config.renderer_learning_rates),
        "renderer_gradient_clip": config.renderer_gradient_clip,
        "renderer_loss_weights": dict(config.renderer_loss_weights),
        "appearance_min_snr": config.appearance_min_snr,
        "ranking_target_ratio": config.ranking_target_ratio,
        "ddim_steps": config.ddim_steps,
        "seed": config.seed,
        "panel_normal_count": config.panel_normal_count,
        "panel_seeds": list(config.panel_seeds),
        "blind_panel_seed": config.blind_panel_seed,
        "output_relpath": config.output_relpath,
    }


def _source_payload(
    source: SourceRendererManifest,
    *,
    config: V06BMaskFirstConfig,
) -> dict[str, Any]:
    if not isinstance(source, SourceRendererManifest):
        raise ValueError("source must be an authenticated SourceRendererManifest")
    expected_id = (
        _CABLE_P2_B2_RENDERER_ID if is_cable_scheme2(config) else _P2_B2_RENDERER_ID
    )
    expected_category = "cable" if is_cable_scheme2(config) else "hazelnut"
    if source.renderer_id != expected_id:
        raise ValueError("source renderer must match the Cable scheme or frozen P2-B2")
    if source.category != expected_category:
        raise ValueError("architecture source category does not match the Cable scheme")
    if config.category not in {"hazelnut", "cable"}:
        raise ValueError("mask-first category must be hazelnut or cable")
    _hash(source.manifest_sha256, name="source manifest", length=64)
    _hash(source.source_git_head, name="source git_head", length=40)
    if not isinstance(source.checkpoint_hashes, Mapping) or set(
        source.checkpoint_hashes
    ) != set(_SOURCE_FILES):
        raise ValueError("source checkpoint hashes must cover the exact P2-B2 files")
    checkpoint_hashes = {
        filename: _hash(
            source.checkpoint_hashes[filename],
            name=f"source {filename}",
            length=64,
        )
        for filename in _SOURCE_FILES
    }
    return {
        "renderer_id": source.renderer_id,
        "manifest_sha256": source.manifest_sha256,
        "source_git_head": source.source_git_head,
        "checkpoint_hashes": checkpoint_hashes,
    }


def _state(module: object, *, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(module, nn.Module):
        raise ValueError(f"{name} must be a torch module")
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in module.state_dict().items()
    }
    if not state:
        raise ValueError(f"{name} state must be non-empty")
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError(f"{name} state must be finite")
    return state


def _support_ids(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(record_id, str) or not record_id for record_id in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("support record IDs must be unique non-empty strings")
    return value


def _support_digest(support_record_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        list(support_record_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inference_payload(
    threshold: ThresholdCalibration,
    *,
    config: V06BMaskFirstConfig,
) -> dict[str, Any]:
    if not isinstance(threshold, ThresholdCalibration):
        raise ValueError("threshold must be a ThresholdCalibration")
    if (
        isinstance(threshold.threshold, bool)
        or not isinstance(threshold.threshold, int | float)
        or not math.isfinite(float(threshold.threshold))
        or float(threshold.threshold) not in config.planner_threshold_candidates
    ):
        raise ValueError("realization threshold must be a configured calibrated candidate")
    if (
        isinstance(threshold.macro_dice, bool)
        or not isinstance(threshold.macro_dice, int | float)
        or not math.isfinite(float(threshold.macro_dice))
        or not 0.0 <= float(threshold.macro_dice) <= 1.0
    ):
        raise ValueError("calibration macro Dice must be finite in [0,1]")
    support_ids = _support_ids(threshold.support_record_ids)
    return {
        "schema_version": INFERENCE_SCHEMA,
        "realization_threshold": float(threshold.threshold),
        "calibration_policy": CALIBRATION_POLICY,
        "support_record_ids": list(support_ids),
        "support_digest": _support_digest(support_ids),
        "planner_seed_mode": PLANNER_SEED_MODE,
        "defect_types": list(config.defect_types),
        "ddim_steps": config.ddim_steps,
    }


def save_mask_first_checkpoint(
    output: Path,
    *,
    planner: nn.Module,
    adapter: nn.Module,
    conditioner: nn.Module,
    config: V06BMaskFirstConfig,
    threshold: ThresholdCalibration,
    source: SourceRendererManifest,
    git_head: str,
) -> dict[str, Any]:
    if not isinstance(output, Path):
        raise ValueError("output must be a pathlib.Path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"checkpoint output already exists: {output}")
    _hash(git_head, name="git_head", length=40)
    config_snapshot = _config_payload(config)
    source_snapshot = _source_payload(source, config=config)
    inference = _inference_payload(threshold, config=config)
    modules = {
        "planner": planner,
        "adapter": adapter,
        "conditioner": conditioner,
    }
    states = {name: _state(module, name=name) for name, module in modules.items()}

    output.mkdir(parents=True, exist_ok=False)
    for filename, module_name in _TENSOR_FILES.items():
        save_file(states[module_name], output / filename)
    write_json_atomic(output / "inference_config.json", inference)
    files = {
        filename: {
            "sha256": sha256_file(output / filename),
            "size": (output / filename).stat().st_size,
        }
        for filename in _PAYLOAD_FILES
    }
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "git_head": git_head,
        "planner_initialization": "fresh",
        "renderer_initialization": renderer_initialization_label(config),
        "source_renderer": source_snapshot,
        "training_steps": {
            "planner": config.planner_training_steps,
            "renderer": config.renderer_training_steps,
        },
        "config": config_snapshot,
        "files": files,
    }
    write_json_atomic(output / "metadata.json", metadata)
    return metadata


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return loaded


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    config: V06BMaskFirstConfig,
    source: SourceRendererManifest,
) -> None:
    expected_keys = {
        "schema_version",
        "git_head",
        "planner_initialization",
        "renderer_initialization",
        "source_renderer",
        "training_steps",
        "config",
        "files",
    }
    if set(metadata) != expected_keys or metadata.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint metadata schema is incompatible with mask-first v1")
    _hash(metadata.get("git_head"), name="checkpoint git_head", length=40)
    if metadata.get("planner_initialization") != "fresh":
        raise ValueError("planner initialization must be fresh")
    expected_renderer = renderer_initialization_label(config)
    if metadata.get("renderer_initialization") != expected_renderer:
        raise ValueError(f"renderer initialization must be {expected_renderer}")
    if metadata.get("source_renderer") != _source_payload(source, config=config):
        raise ValueError("checkpoint source renderer lineage does not match")
    if metadata.get("training_steps") != {
        "planner": config.planner_training_steps,
        "renderer": config.renderer_training_steps,
    }:
        raise ValueError("checkpoint training steps differ from the configuration")
    if metadata.get("config") != _config_payload(config):
        raise ValueError("checkpoint config does not match mask-first v1")


def _validate_payload_files(checkpoint: Path, metadata: dict[str, Any]) -> None:
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != set(_PAYLOAD_FILES):
        raise ValueError("checkpoint payload manifest is invalid")
    for filename in _PAYLOAD_FILES:
        path = checkpoint / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint payload must be regular: {filename}")
        expected = {"sha256": sha256_file(path), "size": path.stat().st_size}
        if files[filename] != expected:
            raise ValueError(f"checkpoint file hash or size changed: {filename}")


def _resolve_inference(
    payload: dict[str, Any],
    *,
    config: V06BMaskFirstConfig,
) -> ResolvedMaskFirstInference:
    expected_keys = {
        "schema_version",
        "realization_threshold",
        "calibration_policy",
        "support_record_ids",
        "support_digest",
        "planner_seed_mode",
        "defect_types",
        "ddim_steps",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != INFERENCE_SCHEMA:
        raise ValueError("inference config schema is incompatible with mask-first v1")
    threshold = payload.get("realization_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or float(threshold) not in config.planner_threshold_candidates
    ):
        raise ValueError("realization threshold is not a configured calibrated candidate")
    raw_support_ids = payload.get("support_record_ids")
    if not isinstance(raw_support_ids, list):
        raise ValueError("support record IDs must be a JSON list")
    support_ids = _support_ids(tuple(raw_support_ids))
    support_digest = _hash(
        payload.get("support_digest"),
        name="support digest",
        length=64,
    )
    if support_digest != _support_digest(support_ids):
        raise ValueError("support digest does not match support record IDs")
    if payload.get("calibration_policy") != CALIBRATION_POLICY:
        raise ValueError("calibration policy is not the certified support-only policy")
    if payload.get("planner_seed_mode") != PLANNER_SEED_MODE:
        raise ValueError("planner seed mode is incompatible")
    if payload.get("defect_types") != list(config.defect_types):
        raise ValueError("inference defect types differ from the configuration")
    if payload.get("ddim_steps") != config.ddim_steps:
        raise ValueError("inference DDIM steps differ from the configured full rollout")
    return ResolvedMaskFirstInference(
        realization_threshold=float(threshold),
        calibration_policy=CALIBRATION_POLICY,
        support_record_ids=support_ids,
        support_digest=support_digest,
        planner_seed_mode=PLANNER_SEED_MODE,
        defect_types=config.defect_types,
        ddim_steps=config.ddim_steps,
    )


def load_mask_first_checkpoint(
    checkpoint: Path,
    *,
    planner: PlannerNetwork,
    adapter: nn.Module,
    conditioner: nn.Module,
    config: V06BMaskFirstConfig,
    source: SourceRendererManifest,
) -> LoadedMaskFirstCheckpoint:
    if not isinstance(checkpoint, Path):
        raise ValueError("checkpoint must be a pathlib.Path")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("checkpoint must be a regular directory")
    if {path.name for path in checkpoint.iterdir()} != _CHECKPOINT_LAYOUT:
        raise ValueError("checkpoint has an invalid exact file layout")
    if not isinstance(planner, PlannerNetwork):
        raise ValueError("planner must be a freshly instantiated PlannerNetwork")
    if not isinstance(adapter, nn.Module) or not isinstance(conditioner, nn.Module):
        raise ValueError("adapter and conditioner must be torch modules")
    metadata = _json_object(checkpoint / "metadata.json", name="checkpoint metadata")
    _validate_metadata(metadata, config=config, source=source)
    _validate_payload_files(checkpoint, metadata)
    inference_payload = _json_object(
        checkpoint / "inference_config.json",
        name="inference config",
    )
    inference = _resolve_inference(inference_payload, config=config)

    modules = {
        "planner": planner,
        "adapter": adapter,
        "conditioner": conditioner,
    }
    for filename, module_name in _TENSOR_FILES.items():
        modules[module_name].load_state_dict(
            load_file(checkpoint / filename, device="cpu"),
            strict=True,
        )
    return LoadedMaskFirstCheckpoint(
        metadata=MappingProxyType(metadata),
        inference=inference,
    )


__all__ = [
    "LoadedMaskFirstCheckpoint",
    "ResolvedMaskFirstInference",
    "architecture_config_payload",
    "load_mask_first_checkpoint",
    "renderer_initialization_label",
    "save_mask_first_checkpoint",
    "type_condition_initialization_label",
    "validate_mask_first_a2_to_h_configs",
    "validate_mask_first_a_to_h_configs",
    "validate_mask_first_renderer_continue_configs",
]

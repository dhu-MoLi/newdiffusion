from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from cadiff.provenance import sha256_file, write_json_atomic

from .config import V06BConfig

CHECKPOINT_SCHEMA = "v06_b_causal_renderer_checkpoint_v1"
_TENSOR_FILES = {
    "adapter.safetensors": "adapter",
    "conditioner.safetensors": "conditioner",
    "realization_head.safetensors": "realization_head",
}


def config_payload(config: V06BConfig) -> dict[str, Any]:
    if not isinstance(config, V06BConfig):
        raise ValueError("config must be a V06BConfig")
    payload: dict[str, Any] = {
        "schema_version": config.schema_version,
        "category": config.category,
        "defect_types": list(config.defect_types),
        "source_renderer_manifest": config.source_renderer_manifest,
        "source_experiment_config": config.source_experiment_config,
        "source_model_config": config.source_model_config,
        "image_size": config.image_size,
        "proposal_radius": config.proposal_radius,
        "support_corner_width": config.support_corner_width,
        "rollout_steps": config.rollout_steps,
        "backprop_steps": config.backprop_steps,
        "checkpoint_tracked_steps": config.checkpoint_tracked_steps,
        "feature_channels": config.feature_channels,
        "head_hidden_channels": config.head_hidden_channels,
        "realization_threshold": config.realization_threshold,
        "visibility_margin": config.visibility_margin,
        "training_steps": config.training_steps,
        "seed": config.seed,
        "checkpoint_every": config.checkpoint_every,
        "panel_normal_count": config.panel_normal_count,
        "panel_seeds": list(config.panel_seeds),
        "output_relpath": config.output_relpath,
    }
    if config.schema_version == "v06_b_causal_renderer_v1":
        payload.update(
            {
                "loss_weights": dict(config.loss_weights),
                "learning_rate": config.learning_rate,
                "gradient_clip": config.gradient_clip,
            }
        )
    elif config.schema_version == "v06_b_causal_renderer_v2":
        payload.update(
            {
                "prompts": dict(config.prompts),
                "parent_experiment_config": config.parent_experiment_config,
                "parent_checkpoint_relpath": config.parent_checkpoint_relpath,
                "parent_checkpoint_step": config.parent_checkpoint_step,
                "appearance_min_snr": config.appearance_min_snr,
                "ranking_target_ratio": config.ranking_target_ratio,
                "amp_initial_scale": config.amp_initial_scale,
                "loss_weights": dict(config.loss_weights),
                "learning_rates": dict(config.learning_rates),
                "gradient_clips": dict(config.gradient_clips),
                "blind_panel_seed": config.blind_panel_seed,
            }
        )
    else:
        raise ValueError("unsupported Module-B checkpoint configuration")
    return payload


def _hash(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal hash")
    return value


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


def save_v06_b_checkpoint(
    output: Path,
    *,
    adapter: nn.Module,
    conditioner: nn.Module,
    realization_head: nn.Module,
    config: V06BConfig,
    git_head: str,
    source_renderer_manifest_sha256: str,
    step: int,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"checkpoint output already exists: {output}")
    _hash(git_head, name="git_head", length=40)
    _hash(
        source_renderer_manifest_sha256,
        name="source_renderer_manifest_sha256",
        length=64,
    )
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    modules = {
        "adapter": adapter,
        "conditioner": conditioner,
        "realization_head": realization_head,
    }
    states = {name: _state(module, name=name) for name, module in modules.items()}

    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, int | str]] = {}
    for filename, module_name in _TENSOR_FILES.items():
        path = output / filename
        save_file(states[module_name], path)
        files[filename] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "git_head": git_head,
        "source_renderer_manifest_sha256": source_renderer_manifest_sha256,
        "step": step,
        "config": config_payload(config),
        "files": files,
    }
    write_json_atomic(output / "metadata.json", metadata)
    return metadata


def _metadata(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "metadata.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint metadata must be a regular file")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("checkpoint metadata must be valid UTF-8 JSON") from error
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return loaded


def load_v06_b_checkpoint(
    checkpoint: Path,
    *,
    adapter: nn.Module,
    conditioner: nn.Module,
    realization_head: nn.Module,
    config: V06BConfig,
    source_renderer_manifest_sha256: str,
    expected_step: int,
) -> dict[str, Any]:
    _hash(
        source_renderer_manifest_sha256,
        name="source_renderer_manifest_sha256",
        length=64,
    )
    if (
        isinstance(expected_step, bool)
        or not isinstance(expected_step, int)
        or expected_step <= 0
    ):
        raise ValueError("expected_step must be a positive integer")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("checkpoint must be a regular directory")
    expected_layout = {*_TENSOR_FILES, "metadata.json"}
    actual_layout = {path.name for path in checkpoint.iterdir()}
    if actual_layout != expected_layout:
        raise ValueError("checkpoint has an invalid exact file layout")
    metadata = _metadata(checkpoint)
    expected_keys = {
        "schema_version",
        "git_head",
        "source_renderer_manifest_sha256",
        "step",
        "config",
        "files",
    }
    if set(metadata) != expected_keys or metadata.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is incompatible with v0.6-B")
    if metadata.get("config") != config_payload(config):
        raise ValueError("checkpoint config does not match v0.6-B")
    _hash(metadata.get("git_head"), name="git_head", length=40)
    metadata_source = _hash(
        metadata.get("source_renderer_manifest_sha256"),
        name="source_renderer_manifest_sha256",
        length=64,
    )
    if metadata_source != source_renderer_manifest_sha256:
        raise ValueError("checkpoint source renderer manifest does not match")
    metadata_step = metadata.get("step")
    if isinstance(metadata_step, bool) or not isinstance(metadata_step, int) or metadata_step <= 0:
        raise ValueError("checkpoint step must be a positive integer")
    if metadata_step != expected_step:
        raise ValueError("checkpoint does not have the expected step")
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != set(_TENSOR_FILES):
        raise ValueError("checkpoint tensor manifest is invalid")
    for filename in _TENSOR_FILES:
        path = checkpoint / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint tensor must be regular: {filename}")
        expected = {"sha256": sha256_file(path), "size": path.stat().st_size}
        if files[filename] != expected:
            raise ValueError(f"checkpoint file hash or size changed: {filename}")

    modules = {
        "adapter": adapter,
        "conditioner": conditioner,
        "realization_head": realization_head,
    }
    for filename, module_name in _TENSOR_FILES.items():
        module = modules[module_name]
        if not isinstance(module, nn.Module):
            raise ValueError(f"{module_name} must be a torch module")
        module.load_state_dict(load_file(checkpoint / filename, device="cpu"), strict=True)
    return metadata


def load_v06_b_parent_checkpoint(
    checkpoint: Path,
    *,
    adapter: nn.Module,
    conditioner: nn.Module,
    realization_head: nn.Module,
    parent_config: V06BConfig,
    config: V06BConfig,
    source_renderer_manifest_sha256: str,
) -> dict[str, Any]:
    """Strictly load the declared v1 parent weights into a fresh v2 optimizer state."""

    if config.schema_version != "v06_b_causal_renderer_v2":
        raise ValueError("parent warm-start requires a v2 configuration")
    if parent_config.schema_version != "v06_b_causal_renderer_v1":
        raise ValueError("parent warm-start requires the frozen v1 parent configuration")
    if config.parent_checkpoint_step != parent_config.final_step:
        raise ValueError("parent checkpoint step differs from the parent final step")
    expected_relpath = f"{parent_config.output_relpath}/checkpoint"
    if config.parent_checkpoint_relpath != expected_relpath:
        raise ValueError("parent checkpoint relpath differs from the parent output")
    compatible_fields = (
        "category",
        "defect_types",
        "source_renderer_manifest",
        "source_experiment_config",
        "source_model_config",
        "image_size",
        "proposal_radius",
        "support_corner_width",
        "rollout_steps",
        "backprop_steps",
        "checkpoint_tracked_steps",
        "feature_channels",
        "head_hidden_channels",
        "realization_threshold",
        "visibility_margin",
    )
    if any(
        getattr(config, name) != getattr(parent_config, name)
        for name in compatible_fields
    ):
        raise ValueError("v2 architecture or causal-mask contract differs from its parent")
    return load_v06_b_checkpoint(
        checkpoint,
        adapter=adapter,
        conditioner=conditioner,
        realization_head=realization_head,
        config=parent_config,
        source_renderer_manifest_sha256=source_renderer_manifest_sha256,
        expected_step=config.parent_checkpoint_step,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "config_payload",
    "load_v06_b_checkpoint",
    "load_v06_b_parent_checkpoint",
    "save_v06_b_checkpoint",
]

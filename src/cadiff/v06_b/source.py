from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import torch
import yaml
from torch import nn

from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_a1.model import FactorizedMaskedInfillAdapter
from cadiff.p2_a1.runner import _load_p2_a1_checkpoint
from cadiff.provenance import sha256_file

_SOURCE_FILES = (
    "adapter.safetensors",
    "conditioner.safetensors",
    "metadata.json",
)
_P2_B2_RENDERER_ID = "p2_b2_hazelnut_type_prompt_attempt1"
_P2_B2_CHECKPOINT_RELPATH = (
    "p2_b2/hazelnut/type-prompt-training/seed-20260817-attempt1/checkpoint"
)
_CABLE_P2_B2_RENDERER_ID = "p2_b2_cable_type_prompt_scheme2"
_CABLE_P2_B2_CHECKPOINT_RELPATH = (
    "p2_b2/cable/type-prompt-training/seed-20260817/checkpoint"
)


@dataclass(frozen=True)
class SourceRendererManifest:
    renderer_id: str
    category: str
    checkpoint: Path
    checkpoint_hashes: Mapping[str, str]
    source_git_head: str
    manifest_sha256: str


@dataclass(frozen=True)
class FreshP2B2Renderer:
    vocabulary: DefectVocabulary
    conditioner: DefectConditioner
    adapter: FactorizedMaskedInfillAdapter
    model: P2InfillNoisePredictor
    source: SourceRendererManifest
    checkpoint_metadata: Mapping[str, Any]


def _hex(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal hash")
    return value


def _relative(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ValueError(f"{name} must be a normalized relative path")
    return value


def load_source_renderer_manifest(
    path: Path,
    *,
    runs_root: Path,
    category: str,
) -> SourceRendererManifest:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source renderer manifest must be a regular file")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        raise ValueError("source renderer manifest must be valid UTF-8 YAML") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("source renderer manifest must be a mapping")
    expected_keys = {
        "schema_version",
        "renderer_id",
        "category",
        "status",
        "decision_date",
        "scope",
        "checkpoint",
        "source_run",
        "decision_evidence",
        "decision",
    }
    if set(loaded) != expected_keys:
        raise ValueError("source renderer manifest has an invalid exact schema")
    if loaded.get("schema_version") != "p2_renderer_freeze_v1":
        raise ValueError("source renderer manifest schema is incompatible")
    if loaded.get("category") != category or loaded.get("status") != "frozen_default":
        raise ValueError("source renderer must be the frozen category default")
    renderer_id = loaded.get("renderer_id")
    if not isinstance(renderer_id, str) or not renderer_id:
        raise ValueError("source renderer_id must be non-empty")
    checkpoint_data = loaded.get("checkpoint")
    if not isinstance(checkpoint_data, Mapping) or set(checkpoint_data) != {
        "runs_root_relpath",
        "source_git_head",
        "files",
    }:
        raise ValueError("source checkpoint manifest is invalid")
    relpath = _relative(
        checkpoint_data["runs_root_relpath"], name="checkpoint.runs_root_relpath"
    )
    root = runs_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("runs_root must be a directory")
    checkpoint_entry = root / relpath
    if checkpoint_entry.is_symlink() or not checkpoint_entry.is_dir():
        raise ValueError("source checkpoint must be a regular directory")
    checkpoint = checkpoint_entry.resolve(strict=True)
    try:
        checkpoint.relative_to(root)
    except ValueError as error:
        raise ValueError("source checkpoint escapes runs_root") from error
    if {entry.name for entry in checkpoint.iterdir()} != set(_SOURCE_FILES):
        raise ValueError("source checkpoint has an invalid exact file layout")

    file_data = checkpoint_data.get("files")
    if not isinstance(file_data, Mapping) or set(file_data) != set(_SOURCE_FILES):
        raise ValueError("source checkpoint file manifest is invalid")
    hashes: dict[str, str] = {}
    for filename in _SOURCE_FILES:
        spec = file_data[filename]
        if not isinstance(spec, Mapping) or set(spec) != {"sha256", "size"}:
            raise ValueError(f"source checkpoint file spec is invalid: {filename}")
        expected_hash = _hex(spec["sha256"], name=f"{filename}.sha256", length=64)
        expected_size = spec["size"]
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise ValueError(f"{filename}.size must be a positive integer")
        file_path = checkpoint / filename
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"source checkpoint file must be regular: {filename}")
        if file_path.stat().st_size != expected_size or sha256_file(file_path) != expected_hash:
            raise ValueError(f"source checkpoint hash or size mismatch: {filename}")
        hashes[filename] = expected_hash
    source_git_head = _hex(
        checkpoint_data["source_git_head"], name="source_git_head", length=40
    )
    try:
        checkpoint_metadata = json.loads(
            (checkpoint / "metadata.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("source checkpoint metadata must be valid UTF-8 JSON") from error
    if (
        not isinstance(checkpoint_metadata, Mapping)
        or checkpoint_metadata.get("git_head") != source_git_head
    ):
        raise ValueError("source checkpoint git_head differs from the freeze manifest")
    return SourceRendererManifest(
        renderer_id=renderer_id,
        category=category,
        checkpoint=checkpoint,
        checkpoint_hashes=MappingProxyType(hashes),
        source_git_head=source_git_head,
        manifest_sha256=sha256_file(path),
    )


def load_fresh_p2_b2_renderer(
    *,
    manifest_path: Path,
    runs_root: Path,
    category: str,
    source_config: P2A1Config,
    unet: nn.Module,
    device: torch.device,
) -> FreshP2B2Renderer:
    if not isinstance(source_config, P2A1Config):
        raise ValueError("source_config must be the P2-A1 model configuration")
    if not isinstance(unet, nn.Module):
        raise ValueError("unet must be a frozen torch module")
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    if category == "cable":
        if (
            source_config.category != "cable"
            or source_config.defect_types
            != (
                "bent_wire",
                "cut_inner_insulation",
                "cut_outer_insulation",
                "poke_insulation",
            )
        ):
            raise ValueError("fresh Cable renderer requires the Cable P2-A1 vocabulary")
        source = load_source_renderer_manifest(
            manifest_path,
            runs_root=runs_root,
            category=category,
        )
        expected_checkpoint = runs_root.resolve(strict=True) / _CABLE_P2_B2_CHECKPOINT_RELPATH
        if (
            source.renderer_id != _CABLE_P2_B2_RENDERER_ID
            or source.checkpoint != expected_checkpoint
        ):
            raise ValueError("renderer source must be the Cable P2-B2 scheme-2 checkpoint")
        vocabulary, conditioner, adapter, metadata = _load_p2_a1_checkpoint(
            source.checkpoint,
            config=source_config,
            device=device,
        )
        revalidated = load_source_renderer_manifest(
            manifest_path,
            runs_root=runs_root,
            category=category,
        )
        if revalidated != source or metadata.get("git_head") != source.source_git_head:
            raise ValueError("Cable P2-B2 source changed while loading")
        model = P2InfillNoisePredictor(unet, conditioner, adapter).train()
        return FreshP2B2Renderer(
            vocabulary=vocabulary,
            conditioner=conditioner,
            adapter=adapter,
            model=model,
            source=source,
            checkpoint_metadata=metadata,
        )
    if (
        category != "hazelnut"
        or source_config.category != category
        or source_config.defect_types != ("crack", "cut", "hole", "print")
    ):
        raise ValueError("fresh renderer requires the frozen P2-B2 Hazelnut vocabulary")
    source = load_source_renderer_manifest(
        manifest_path,
        runs_root=runs_root,
        category=category,
    )
    expected_checkpoint = runs_root.resolve(strict=True) / _P2_B2_CHECKPOINT_RELPATH
    if (
        source.renderer_id != _P2_B2_RENDERER_ID
        or source.checkpoint != expected_checkpoint
    ):
        raise ValueError("renderer source must be the frozen P2-B2 attempt1")
    vocabulary, conditioner, adapter, metadata = _load_p2_a1_checkpoint(
        source.checkpoint,
        config=source_config,
        device=device,
    )
    revalidated = load_source_renderer_manifest(
        manifest_path,
        runs_root=runs_root,
        category=category,
    )
    if revalidated != source or metadata.get("git_head") != source.source_git_head:
        raise ValueError("frozen P2-B2 source changed while loading")
    model = P2InfillNoisePredictor(unet, conditioner, adapter).train()
    return FreshP2B2Renderer(
        vocabulary=vocabulary,
        conditioner=conditioner,
        adapter=adapter,
        model=model,
        source=source,
        checkpoint_metadata=metadata,
    )


def bind_fresh_type_vocabulary(
    loaded: FreshP2B2Renderer,
    *,
    category: str,
    defect_types: tuple[str, ...],
    unet: nn.Module,
    device: torch.device,
) -> FreshP2B2Renderer:
    """Keep the frozen P2-B2 adapter weights; rebuild Cable type embeddings from scratch."""

    if (
        category == loaded.vocabulary.category
        and defect_types == loaded.vocabulary.labels
    ):
        return loaded
    if category != "cable":
        raise ValueError("only Cable may rebind the frozen P2-B2 adapter vocabulary")
    vocabulary = DefectVocabulary.from_config(category, defect_types)
    conditioner = DefectConditioner(vocabulary, 128).to(device=device, dtype=torch.float32)
    adapter = loaded.adapter
    if isinstance(adapter, FactorizedMaskedInfillAdapter) and isinstance(
        adapter.config, P2A1Config
    ):
        cable_config = replace(
            adapter.config,
            category=category,
            defect_types=defect_types,
            normal_prompt="a photo of a normal cable",
            capacity_defect_type=defect_types[0],
        )
        rebound = FactorizedMaskedInfillAdapter(cable_config).to(
            device=device, dtype=torch.float32
        )
        rebound.load_state_dict(adapter.state_dict())
        adapter = rebound
    model = P2InfillNoisePredictor(unet, conditioner, adapter).train()
    return FreshP2B2Renderer(
        vocabulary=vocabulary,
        conditioner=conditioner,
        adapter=adapter,
        model=model,
        source=loaded.source,
        checkpoint_metadata=loaded.checkpoint_metadata,
    )


__all__ = [
    "FreshP2B2Renderer",
    "SourceRendererManifest",
    "bind_fresh_type_vocabulary",
    "load_fresh_p2_b2_renderer",
    "load_source_renderer_manifest",
]

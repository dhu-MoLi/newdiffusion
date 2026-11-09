"""Strict, self-describing A1 checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from .config import V06AExperimentConfig
from .diffusion import MaskDiffusionSchedule
from .model import (
    SPATIAL_INPUT_CHANNELS,
    A1MaskUNet,
    require_supported_variant,
    validate_a1_model_contract,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_METADATA_KEYS = {
    "schema_version",
    "category",
    "variant",
    "defect_types",
    "target_kind",
    "spatial_input_channels",
    "architecture",
    "architecture_scales",
    "type_conditioning",
    "timestep_conditioning",
    "normal_conditioning",
    "image_size",
    "proposal_size",
    "base_channels",
    "type_embedding_dim",
    "timestep_embedding_dim",
    "normal_feature_channels",
    "diffusion_schedule",
    "diffusion_steps",
    "training_steps",
    "experiment_config_sha256",
    "model_construction_signature",
    "weights_sha256",
}


@dataclass(frozen=True)
class A1CheckpointMetadata:
    schema_version: str
    category: str
    variant: str
    defect_types: tuple[str, ...]
    target_kind: str
    spatial_input_channels: tuple[str, ...]
    architecture: str
    architecture_scales: tuple[int, ...]
    type_conditioning: str
    timestep_conditioning: str
    normal_conditioning: str
    image_size: int
    proposal_size: int
    base_channels: int
    type_embedding_dim: int
    timestep_embedding_dim: int
    normal_feature_channels: int
    diffusion_schedule: str
    diffusion_steps: int
    training_steps: int
    experiment_config_sha256: str
    model_construction_signature: str
    weights_sha256: str


def a1_experiment_config_sha256(config: V06AExperimentConfig) -> str:
    """Hash scientific/model configuration while excluding caller-owned placement."""

    if not isinstance(config, V06AExperimentConfig):
        raise TypeError("config must be a V06AExperimentConfig")
    values = asdict(config)
    # Artifact placement is supplied to ``run_training`` and has no scientific
    # effect.  Retain the legacy YAML field for compatibility without letting
    # it change the model/config signature.
    values.pop("output_relpath")
    values.pop("sampling_steps")
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_config(
    model: A1MaskUNet,
    config: V06AExperimentConfig,
    *,
    variant: str,
) -> None:
    if not isinstance(model, A1MaskUNet):
        raise TypeError("model must be an A1MaskUNet")
    if not isinstance(config, V06AExperimentConfig):
        raise TypeError("config must be V06AExperimentConfig")
    variant = require_supported_variant(variant)
    validate_a1_model_contract(
        model,
        expected_variant=variant,
        expected_category=config.category,
        expected_defect_types=config.defect_types,
    )
    expected = {
        "category": config.category,
        "defect_types": config.defect_types,
        "variant": variant,
        "base_channels": config.base_channels,
        "type_embedding_dim": config.type_embedding_dim,
        "timestep_embedding_dim": config.timestep_embedding_dim,
        "normal_feature_channels": config.normal_feature_channels,
        "diffusion_steps": config.diffusion_steps,
        "proposal_size": config.proposal_size,
        "image_size": config.image_size,
        "spatial_input_channels": SPATIAL_INPUT_CHANNELS,
    }
    mismatches = {
        key: (getattr(model, key, None), value)
        for key, value in expected.items()
        if getattr(model, key, None) != value
    }
    if mismatches:
        raise ValueError(f"model/config metadata mismatch: {mismatches}")


def _expected_metadata(
    *,
    model: A1MaskUNet,
    config: V06AExperimentConfig,
    variant: str,
    weights_sha256: str,
) -> A1CheckpointMetadata:
    _validate_model_config(model, config, variant=variant)
    return A1CheckpointMetadata(
        schema_version="v06_a_checkpoint_rc2",
        category=config.category,
        variant=variant,
        defect_types=config.defect_types,
        target_kind="plain_geometry_transfer",
        spatial_input_channels=SPATIAL_INPUT_CHANNELS,
        architecture="three_scale_unet",
        architecture_scales=(128, 64, 32),
        type_conditioning="film_gating",
        timestep_conditioning="film_gating",
        normal_conditioning=("none" if variant == "a1_gv" else "multiscale_rgb_fusion"),
        image_size=config.image_size,
        proposal_size=config.proposal_size,
        base_channels=config.base_channels,
        type_embedding_dim=config.type_embedding_dim,
        timestep_embedding_dim=config.timestep_embedding_dim,
        normal_feature_channels=config.normal_feature_channels,
        diffusion_schedule=MaskDiffusionSchedule.schedule_name,
        diffusion_steps=config.diffusion_steps,
        training_steps=config.training_steps,
        experiment_config_sha256=a1_experiment_config_sha256(config),
        model_construction_signature=validate_a1_model_contract(model),
        weights_sha256=weights_sha256,
    )


def _metadata_mapping(metadata: A1CheckpointMetadata) -> dict[str, object]:
    mapping = asdict(metadata)
    for key in ("defect_types", "spatial_input_channels", "architecture_scales"):
        mapping[key] = list(mapping[key])  # type: ignore[arg-type]
    return mapping


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint metadata key: {key}")
        result[key] = value
    return result


def _read_metadata(path: Path) -> A1CheckpointMetadata:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint metadata: {error}") from error
    if not isinstance(raw, dict) or set(raw) != _METADATA_KEYS:
        raise ValueError("checkpoint metadata schema mismatch")
    tuple_fields: dict[str, type] = {
        "defect_types": str,
        "spatial_input_channels": str,
        "architecture_scales": int,
    }
    for key, item_type in tuple_fields.items():
        value = raw[key]
        if (
            not isinstance(value, list)
            or any(type(item) is not item_type for item in value)
            or not value
        ):
            raise ValueError(f"checkpoint metadata {key} has invalid type")
        raw[key] = tuple(value)
    string_fields = (
        _METADATA_KEYS
        - set(tuple_fields)
        - {
            "image_size",
            "proposal_size",
            "base_channels",
            "type_embedding_dim",
            "timestep_embedding_dim",
            "normal_feature_channels",
            "diffusion_steps",
            "training_steps",
        }
    )
    if any(not isinstance(raw[key], str) for key in string_fields):
        raise ValueError("checkpoint metadata string field has invalid type")
    integer_fields = _METADATA_KEYS - set(tuple_fields) - string_fields
    if any(isinstance(raw[key], bool) or not isinstance(raw[key], int) for key in integer_fields):
        raise ValueError("checkpoint metadata integer field has invalid type")
    for key in (
        "experiment_config_sha256",
        "model_construction_signature",
        "weights_sha256",
    ):
        if _SHA256.fullmatch(raw[key]) is None:
            raise ValueError(f"checkpoint metadata {key} is not SHA-256")
    try:
        return A1CheckpointMetadata(**raw)
    except TypeError as error:
        raise ValueError(f"invalid checkpoint metadata: {error}") from error


def save_a1_checkpoint(
    *,
    checkpoint: Path,
    model: A1MaskUNet,
    config: V06AExperimentConfig,
) -> A1CheckpointMetadata:
    """Create one category/variant-specific safetensors checkpoint."""

    if not isinstance(checkpoint, Path):
        raise TypeError("checkpoint must be a pathlib.Path")
    variant = require_supported_variant(getattr(model, "variant", None))
    _validate_model_config(model, config, variant=variant)
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(f"checkpoint path already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint.name}.tmp-", dir=checkpoint.parent))
    published = False
    try:
        weights_path = temporary / "model.safetensors"
        state = {
            name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
        }
        save_safetensors(state, weights_path)
        metadata = _expected_metadata(
            model=model,
            config=config,
            variant=variant,
            weights_sha256=_sha256(weights_path),
        )
        (temporary / "metadata.json").write_text(
            json.dumps(_metadata_mapping(metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if checkpoint.exists() or checkpoint.is_symlink():
            raise FileExistsError(f"checkpoint path appeared during save: {checkpoint}")
        temporary.rename(checkpoint)
        published = True
        return metadata
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def load_a1_checkpoint(
    *,
    checkpoint: Path,
    model: A1MaskUNet,
    config: V06AExperimentConfig,
    expected_variant: str,
) -> A1CheckpointMetadata:
    """Validate every frozen contract field and weight hash before strict loading."""

    if not isinstance(checkpoint, Path):
        raise TypeError("checkpoint must be a pathlib.Path")
    expected_variant = require_supported_variant(expected_variant)
    expected = _expected_metadata(
        model=model,
        config=config,
        variant=expected_variant,
        weights_sha256="",
    )
    metadata = _read_metadata(checkpoint / "metadata.json")
    if replace(metadata, weights_sha256="") != expected:
        raise ValueError("checkpoint metadata mismatch for requested model/config")
    weights_path = checkpoint / "model.safetensors"
    try:
        observed_sha256 = _sha256(weights_path)
    except OSError as error:
        raise ValueError(f"cannot read checkpoint weights: {error}") from error
    if observed_sha256 != metadata.weights_sha256:
        raise ValueError("checkpoint weights SHA-256 mismatch")

    parameter = next(model.parameters())
    try:
        state = load_safetensors(weights_path, device=str(parameter.device))
    except Exception as error:
        raise ValueError(f"invalid safetensors checkpoint: {error}") from error
    expected_state = model.state_dict()
    if set(state) != set(expected_state) or any(
        state[name].shape != expected_state[name].shape
        or state[name].dtype != expected_state[name].dtype
        for name in expected_state
    ):
        raise ValueError("checkpoint state dict does not exactly match the model")
    model.load_state_dict(state, strict=True)
    return metadata

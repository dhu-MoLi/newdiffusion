from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import yaml

_TYPES = ("crack", "cut", "hole", "print")
_PROMPTS = {
    "crack": "a photo of a hazelnut with thin cracks in the shell and fractured edges",
    "cut": "a photo of a hazelnut with a sharp cut in the shell",
    "hole": "a photo of a hazelnut with a puncture hole in the shell",
    "print": (
        "a photo of a hazelnut with dark printed characters and structured ink strokes "
        "on the shell surface"
    ),
}
_SOURCE_HASHES = {
    "adapter.safetensors": (
        "06cad68ee60e42479aa42fe2728024add0476fc3f7311c41be7904bbdd83ceb3"
    ),
    "conditioner.safetensors": (
        "0003cb5c597d8fdb35812b98e09c4c3bce26745f10424f5b636f61f3291038d8"
    ),
    "metadata.json": (
        "49d9c1c641e28e90ad3b3a28565863719be22dce3c34030985abe3c8ca8c6a67"
    ),
}


@dataclass(frozen=True)
class P2B2Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    prompts: Mapping[str, str]
    steps: int
    learning_rate: float
    type_rank: int
    source_checkpoint_relpath: str
    source_checkpoint_hashes: Mapping[str, str]
    evaluation_mode: str


def load_p2_b2_config(path: str | Path) -> P2B2Config:
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid P2-B2 YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("P2-B2 configuration must be a YAML mapping")
    expected_keys = frozenset(P2B2Config.__dataclass_fields__)
    if set(loaded) != expected_keys:
        missing = sorted(expected_keys - set(loaded))
        extra = sorted(repr(key) for key in set(loaded) - expected_keys)
        raise ValueError(f"P2-B2 schema mismatch: missing={missing}, extra={extra}")
    defect_types = loaded["defect_types"]
    prompts = loaded["prompts"]
    hashes = loaded["source_checkpoint_hashes"]
    if not isinstance(defect_types, list) or tuple(defect_types) != _TYPES:
        raise ValueError("defect_types must equal the frozen Hazelnut vocabulary")
    if not isinstance(prompts, Mapping) or tuple(prompts) != _TYPES:
        raise ValueError("prompts must contain the exact ordered Hazelnut vocabulary")
    prompt_values = dict(prompts)
    if prompt_values != _PROMPTS:
        raise ValueError("prompts must equal the frozen P2-B2 Type-Prompt mapping")
    if not isinstance(hashes, Mapping) or dict(hashes) != _SOURCE_HASHES:
        raise ValueError("source_checkpoint_hashes must equal the accepted P2-A1 hashes")
    relpath = loaded["source_checkpoint_relpath"]
    if not isinstance(relpath, str) or not relpath:
        raise ValueError("source_checkpoint_relpath must be a non-empty string")
    pure = PurePosixPath(relpath)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relpath:
        raise ValueError("source_checkpoint_relpath must be a normalized relative path")
    config = P2B2Config(
        schema_version=str(loaded["schema_version"]),
        category=str(loaded["category"]),
        defect_types=tuple(defect_types),
        prompts=MappingProxyType(prompt_values),
        steps=loaded["steps"],
        learning_rate=loaded["learning_rate"],
        type_rank=loaded["type_rank"],
        source_checkpoint_relpath=relpath,
        source_checkpoint_hashes=MappingProxyType(dict(hashes)),
        evaluation_mode=str(loaded["evaluation_mode"]),
    )
    expected = {
        "schema_version": "p2_b2_type_prompt_training_v1",
        "category": "hazelnut",
        "steps": 400,
        "learning_rate": 1e-4,
        "type_rank": 8,
        "source_checkpoint_relpath": (
            "p2_a1/hazelnut/factorized/seed-20260817-attempt1/checkpoint"
        ),
        "evaluation_mode": "diagnostic_only",
    }
    for field, value in expected.items():
        actual = getattr(config, field)
        if actual != value or (
            field == "learning_rate"
            and (type(actual) is not float or not math.isfinite(actual))
        ):
            raise ValueError(f"{field} must equal the frozen P2-B2 value {value!r}")
    if type(config.steps) is not int or type(config.type_rank) is not int:
        raise ValueError("steps and type_rank must be integers")
    return config


_CABLE_TYPES = (
    "bent_wire",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "poke_insulation",
)
_CABLE_PROMPTS = {
    "bent_wire": "a photo of a cable with a bent conductor wire",
    "cut_inner_insulation": "a photo of a cable with a cut in the inner insulation",
    "cut_outer_insulation": "a photo of a cable with a cut in the outer insulation",
    "poke_insulation": "a photo of a cable with a poke puncture through the insulation",
}
_CABLE_SOURCE_RELPATH = "p2_a1/cable/factorized/seed-20260817/checkpoint"


def _hex_hash(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase sha256 hex digest")
    return value


def load_p2_b2_cable_config(path: str | Path) -> P2B2Config:
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid P2-B2 YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("P2-B2 configuration must be a YAML mapping")
    expected_keys = frozenset(P2B2Config.__dataclass_fields__)
    if set(loaded) != expected_keys:
        missing = sorted(expected_keys - set(loaded))
        extra = sorted(repr(key) for key in set(loaded) - expected_keys)
        raise ValueError(f"P2-B2 schema mismatch: missing={missing}, extra={extra}")
    defect_types = loaded["defect_types"]
    prompts = loaded["prompts"]
    hashes = loaded["source_checkpoint_hashes"]
    if not isinstance(defect_types, list) or tuple(defect_types) != _CABLE_TYPES:
        raise ValueError("defect_types must equal the frozen Cable vocabulary")
    if not isinstance(prompts, Mapping) or tuple(prompts) != _CABLE_TYPES:
        raise ValueError("prompts must contain the exact ordered Cable vocabulary")
    prompt_values = dict(prompts)
    if prompt_values != _CABLE_PROMPTS:
        raise ValueError("prompts must equal the Cable Type-Prompt mapping")
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "adapter.safetensors",
        "conditioner.safetensors",
        "metadata.json",
    }:
        raise ValueError("source_checkpoint_hashes must cover the P2-A1 checkpoint files")
    hash_values = {
        name: _hex_hash(hashes[name], name=name) for name in sorted(hashes)
    }
    relpath = loaded["source_checkpoint_relpath"]
    if not isinstance(relpath, str) or not relpath:
        raise ValueError("source_checkpoint_relpath must be a non-empty string")
    pure = PurePosixPath(relpath)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relpath:
        raise ValueError("source_checkpoint_relpath must be a normalized relative path")
    if relpath != _CABLE_SOURCE_RELPATH:
        raise ValueError("source_checkpoint_relpath must be the Cable P2-A1 checkpoint")
    config = P2B2Config(
        schema_version=str(loaded["schema_version"]),
        category=str(loaded["category"]),
        defect_types=tuple(defect_types),
        prompts=MappingProxyType(prompt_values),
        steps=loaded["steps"],
        learning_rate=loaded["learning_rate"],
        type_rank=loaded["type_rank"],
        source_checkpoint_relpath=relpath,
        source_checkpoint_hashes=MappingProxyType(hash_values),
        evaluation_mode=str(loaded["evaluation_mode"]),
    )
    expected = {
        "schema_version": "p2_b2_type_prompt_training_v1",
        "category": "cable",
        "steps": 400,
        "learning_rate": 1e-4,
        "type_rank": 8,
        "source_checkpoint_relpath": _CABLE_SOURCE_RELPATH,
        "evaluation_mode": "diagnostic_only",
    }
    for field, value in expected.items():
        actual = getattr(config, field)
        if actual != value or (
            field == "learning_rate"
            and (type(actual) is not float or not math.isfinite(actual))
        ):
            raise ValueError(f"{field} must equal the Cable P2-B2 value {value!r}")
    if type(config.steps) is not int or type(config.type_rank) is not int:
        raise ValueError("steps and type_rank must be integers")
    return config


def load_p2_b2_experiment_config(path: str | Path) -> P2B2Config:
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid P2-B2 YAML: {error}") from error
    if isinstance(loaded, Mapping) and loaded.get("category") == "cable":
        return load_p2_b2_cable_config(path)
    return load_p2_b2_config(path)


__all__ = [
    "P2B2Config",
    "load_p2_b2_cable_config",
    "load_p2_b2_config",
    "load_p2_b2_experiment_config",
]

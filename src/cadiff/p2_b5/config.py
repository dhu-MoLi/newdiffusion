"""Strict configuration for the bounded P2-B5 continuation."""
# ruff: noqa: E501, UP038

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml

_SCHEMA = "p2_b5_type_selective_texture_v1"
_TYPES = ("crack", "cut", "hole", "print")
_STRUCTURE_TYPES = ("cut", "hole", "print")
_STRUCTURE_SPACE = "decoded_predicted_x0_rgb"
_STRUCTURE_EROSION_KERNEL = 5
_PROMPTS = {
    "crack": "a photo of a hazelnut with thin cracks in the shell and fractured edges",
    "cut": "a photo of a hazelnut with a sharp cut in the shell",
    "hole": "a photo of a hazelnut with a puncture hole in the shell",
    "print": "a photo of a hazelnut with dark printed characters and structured ink strokes on the shell surface",
}
_SOURCE_REL = "p2_b2/hazelnut/type-prompt-training/seed-20260817-attempt1/checkpoint"
_SOURCE_HASHES = {
    "adapter.safetensors": "520fdbd577ba6da80855ecf82b9627e53a41455bf804f0d0c713af817649af6a",
    "conditioner.safetensors": "4247fed4526671e6e17da7adf78b4b15b30fd873f65be2b074c6e450e49fb073",
    "metadata.json": "80f7f6e2575c827a89bbaf98a946f8668c36db9087d9bcbb9f4769ec042a5da8",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True)
class P2B5Config:
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
    structure_weight: float
    structure_types: tuple[str, ...]
    structure_space: str
    structure_erosion_kernel: int


def _expect_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _expect_exact_keys(data: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "category",
        "defect_types",
        "prompts",
        "steps",
        "learning_rate",
        "type_rank",
        "source_checkpoint_relpath",
        "source_checkpoint_hashes",
        "evaluation_mode",
        "structure_weight",
        "structure_types",
        "structure_space",
        "structure_erosion_kernel",
    }
    if set(data) != expected:
        raise ValueError(f"configuration keys must be exactly {sorted(expected)}")


def _strict_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_p2_b5_config(path: Path) -> P2B5Config:
    """Load the frozen P2-B5 experiment config without implicit defaults."""

    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}") from exc
    data = _expect_mapping(raw, "configuration")
    _expect_exact_keys(data)

    defect_types = tuple(data["defect_types"])
    if defect_types != _TYPES:
        raise ValueError(f"defect_types must be exactly {_TYPES}")
    prompts_raw = _expect_mapping(data["prompts"], "prompts")
    prompts = {key: value for key, value in prompts_raw.items()}
    if dict(prompts) != _PROMPTS:
        raise ValueError("prompts must equal the frozen P2-B2 Type-Prompt mapping")
    steps = _strict_positive_int(data["steps"], "steps")
    if steps != 200:
        raise ValueError("steps must be frozen at 200")
    learning_rate = data["learning_rate"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or float(learning_rate) != 1e-4:
        raise ValueError("learning_rate must be frozen at 1e-4")
    type_rank = _strict_positive_int(data["type_rank"], "type_rank")
    if type_rank != 8:
        raise ValueError("type_rank must be frozen at 8")
    source_relpath = data["source_checkpoint_relpath"]
    if not isinstance(source_relpath, str) or PurePosixPath(source_relpath).is_absolute() or source_relpath != _SOURCE_REL:
        raise ValueError("source_checkpoint_relpath must be the frozen P2-B2 checkpoint")
    hashes = _expect_mapping(data["source_checkpoint_hashes"], "source_checkpoint_hashes")
    if dict(hashes) != _SOURCE_HASHES or not all(isinstance(value, str) for value in hashes.values()):
        raise ValueError("source_checkpoint_hashes must match the frozen P2-B2 checkpoint")
    structure_weight = data["structure_weight"]
    if isinstance(structure_weight, bool) or not isinstance(structure_weight, (int, float)) or float(structure_weight) != 0.1:
        raise ValueError("structure_weight must be frozen at 0.1")
    structure_types = tuple(data["structure_types"])
    if structure_types != _STRUCTURE_TYPES:
        raise ValueError(f"structure_types must be exactly {_STRUCTURE_TYPES}")
    if data["structure_space"] != _STRUCTURE_SPACE:
        raise ValueError(f"structure_space must be exactly {_STRUCTURE_SPACE!r}")
    erosion_kernel = _strict_positive_int(
        data["structure_erosion_kernel"], "structure_erosion_kernel"
    )
    if erosion_kernel != _STRUCTURE_EROSION_KERNEL:
        raise ValueError(
            f"structure_erosion_kernel must be exactly {_STRUCTURE_EROSION_KERNEL}"
        )
    if data["schema_version"] != _SCHEMA or data["category"] != "hazelnut" or data["evaluation_mode"] != "diagnostic_only":
        raise ValueError("schema_version, category, or evaluation_mode differs from the frozen contract")
    return P2B5Config(
        schema_version=_SCHEMA,
        category="hazelnut",
        defect_types=defect_types,
        prompts=MappingProxyType(prompts),
        steps=steps,
        learning_rate=float(learning_rate),
        type_rank=type_rank,
        source_checkpoint_relpath=source_relpath,
        source_checkpoint_hashes=MappingProxyType(dict(hashes)),
        evaluation_mode="diagnostic_only",
        structure_weight=float(structure_weight),
        structure_types=structure_types,
        structure_space=_STRUCTURE_SPACE,
        structure_erosion_kernel=erosion_kernel,
    )

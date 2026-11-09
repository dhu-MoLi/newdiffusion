from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AdapterConfig:
    type_embedding_dim: int
    stem_channels: int
    scale64_channels: int
    scale32_channels: int
    blocks_per_scale: int


@dataclass(frozen=True)
class Smoke0Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    normal_prompt: str
    image_size: int
    dtype: str
    seed: int
    learning_rate: float
    adapter: AdapterConfig


_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "category",
        "defect_types",
        "normal_prompt",
        "image_size",
        "dtype",
        "seed",
        "learning_rate",
        "adapter",
    }
)
_ADAPTER_KEYS = frozenset(
    {
        "type_embedding_dim",
        "stem_channels",
        "scale64_channels",
        "scale32_channels",
        "blocks_per_scale",
    }
)
_DEFECT_TYPES = ("broken_large", "broken_small", "contamination")


def _mapping(value: object, field: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _exact_keys(mapping: Mapping[object, object], expected: frozenset[str], field: str) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = actual - expected
    if missing:
        raise ValueError(f"{field} is missing required keys: {', '.join(missing)}")
    if extra:
        rendered_extra = ", ".join(sorted(repr(key) for key in extra))
        raise ValueError(f"{field} has unexpected keys: {rendered_extra}")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _defect_types(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("defect_types must be a non-empty list")
    labels = tuple(_string(label, "defect_types") for label in value)
    if "normal" in labels:
        raise ValueError("defect_types must not contain normal")
    if len(set(labels)) != len(labels):
        raise ValueError("defect_types must not contain duplicate labels")
    if labels != tuple(sorted(labels)):
        raise ValueError("defect_types must be lexicographically sorted")
    if labels != _DEFECT_TYPES:
        raise ValueError("defect_types must match the fixed bottle vocabulary")
    return labels


def _adapter_config(value: object) -> AdapterConfig:
    raw = _mapping(value, "adapter")
    _exact_keys(raw, _ADAPTER_KEYS, "adapter")
    fields = {name: _integer(raw[name], f"adapter.{name}") for name in _ADAPTER_KEYS}
    config = AdapterConfig(**fields)
    expected = AdapterConfig(128, 64, 128, 256, 2)
    if config != expected:
        for field in _ADAPTER_KEYS:
            if getattr(config, field) != getattr(expected, field):
                raise ValueError(f"adapter.{field} has an unsupported value")
        raise AssertionError("unreachable")
    return config


def load_smoke0_config(path: Path) -> Smoke0Config:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    raw_mapping = _mapping(raw, "configuration")
    _exact_keys(raw_mapping, _REQUIRED_KEYS, "configuration")

    schema_version = _string(raw_mapping["schema_version"], "schema_version")
    if schema_version != "p1_v1_smoke0_v1":
        raise ValueError("schema_version must be p1_v1_smoke0_v1")
    category = _string(raw_mapping["category"], "category")
    if category != "bottle":
        raise ValueError("category must be bottle")
    normal_prompt = _string(raw_mapping["normal_prompt"], "normal_prompt")
    if normal_prompt != "a photo of a normal bottle":
        raise ValueError("normal_prompt must be the fixed bottle prompt")
    image_size = _integer(raw_mapping["image_size"], "image_size")
    if image_size != 512:
        raise ValueError("image_size must be 512")
    dtype = _string(raw_mapping["dtype"], "dtype")
    if dtype != "float16":
        raise ValueError("dtype must be float16")
    seed = _integer(raw_mapping["seed"], "seed")
    if seed != 20260817:
        raise ValueError("seed must be 20260817")
    learning_rate = raw_mapping["learning_rate"]
    if type(learning_rate) is not float or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive float")
    if learning_rate != 1e-4:
        raise ValueError("learning_rate must be 0.0001")

    return Smoke0Config(
        schema_version=schema_version,
        category=category,
        defect_types=_defect_types(raw_mapping["defect_types"]),
        normal_prompt=normal_prompt,
        image_size=image_size,
        dtype=dtype,
        seed=seed,
        learning_rate=learning_rate,
        adapter=_adapter_config(raw_mapping["adapter"]),
    )

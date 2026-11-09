from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class P2B0Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    normal_prompt: str
    image_size: int
    dtype: str
    seed: int
    denoise_steps: int
    normal_reference_count: int
    sample_seeds: tuple[int, ...]


_KEYS = frozenset(P2B0Config.__dataclass_fields__)


def _exact_mapping(value: object) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    missing = sorted(_KEYS - set(value))
    extra = sorted(repr(key) for key in set(value) - _KEYS)
    if missing:
        raise ValueError(f"configuration is missing required keys: {', '.join(missing)}")
    if extra:
        raise ValueError(f"configuration has unexpected keys: {', '.join(extra)}")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty list of strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty list of non-negative integers")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must contain unique values")
    return result


def load_p2_b0_config(path: str | Path) -> P2B0Config:
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    raw = _exact_mapping(loaded)
    config = P2B0Config(
        schema_version=_string(raw["schema_version"], "schema_version"),
        category=_string(raw["category"], "category"),
        defect_types=_string_tuple(raw["defect_types"], "defect_types"),
        normal_prompt=_string(raw["normal_prompt"], "normal_prompt"),
        image_size=_positive_integer(raw["image_size"], "image_size"),
        dtype=_string(raw["dtype"], "dtype"),
        seed=_positive_integer(raw["seed"], "seed"),
        denoise_steps=_positive_integer(raw["denoise_steps"], "denoise_steps"),
        normal_reference_count=_positive_integer(
            raw["normal_reference_count"], "normal_reference_count"
        ),
        sample_seeds=_integer_tuple(raw["sample_seeds"], "sample_seeds"),
    )
    expected = {
        "schema_version": "p2_b0_generation_probe_v1",
        "category": "hazelnut",
        "defect_types": ("crack", "cut", "hole", "print"),
        "normal_prompt": "a photo of a normal hazelnut",
        "image_size": 512,
        "dtype": "float16",
        "seed": 20260817,
        "denoise_steps": 50,
        "normal_reference_count": 3,
        "sample_seeds": (202608171, 202608172, 202608173, 202608174),
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"{field} must equal the frozen P2-B0 value {value!r}")
    return config


__all__ = ["P2B0Config", "load_p2_b0_config"]

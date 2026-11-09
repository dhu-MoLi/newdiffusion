from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class P2B1Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    normal_prompt: str
    image_size: int
    dtype: str
    seed: int
    denoise_steps: int
    masks_per_type: int
    normal_reference_count: int
    sample_seeds: tuple[int, ...]


_EXPECTED: dict[str, object] = {
    "schema_version": "p2_b1_heldout_generation_probe_v1",
    "category": "hazelnut",
    "defect_types": ("crack", "cut", "hole", "print"),
    "normal_prompt": "a photo of a normal hazelnut",
    "image_size": 512,
    "dtype": "float16",
    "seed": 20260817,
    "denoise_steps": 50,
    "masks_per_type": 2,
    "normal_reference_count": 8,
    "sample_seeds": (202608175, 202608176),
}


def _tuple(value: object, *, strings: bool, field: str) -> tuple[object, ...]:
    valid = isinstance(value, list) and bool(value)
    if valid and strings:
        valid = all(isinstance(item, str) and bool(item) for item in value)
    elif valid:
        valid = all(type(item) is int and item >= 0 for item in value)
    if not valid or len(set(value)) != len(value):
        kind = "strings" if strings else "non-negative integers"
        raise ValueError(f"{field} must be a non-empty unique list of {kind}")
    return tuple(value)


def load_p2_b1_config(path: str | Path) -> P2B1Config:
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    keys = frozenset(P2B1Config.__dataclass_fields__)
    missing = sorted(keys - set(loaded))
    extra = sorted(repr(key) for key in set(loaded) - keys)
    if missing or extra:
        raise ValueError(f"configuration schema mismatch: missing={missing}, extra={extra}")
    values: dict[str, object] = {}
    for field in ("schema_version", "category", "normal_prompt", "dtype"):
        value = loaded[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
        values[field] = value
    for field in (
        "image_size",
        "seed",
        "denoise_steps",
        "masks_per_type",
        "normal_reference_count",
    ):
        value = loaded[field]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        values[field] = value
    values["defect_types"] = _tuple(loaded["defect_types"], strings=True, field="defect_types")
    values["sample_seeds"] = _tuple(
        loaded["sample_seeds"], strings=False, field="sample_seeds"
    )
    config = P2B1Config(**values)  # type: ignore[arg-type]
    for field, expected in _EXPECTED.items():
        if getattr(config, field) != expected:
            raise ValueError(f"{field} must equal the frozen P2-B1 value {expected!r}")
    return config


__all__ = ["P2B1Config", "load_p2_b1_config"]

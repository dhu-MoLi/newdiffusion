from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TYPES = ("crack", "cut", "hole", "print")
_PROMPTS = {
    "crack": "a photo of a hazelnut with thin cracks in the shell and fractured edges",
    "cut": "a photo of a hazelnut with a sharp cut in the shell",
    "hole": "a photo of a hazelnut with a puncture hole in the shell",
    "print": (
        "a photo of a hazelnut with dark printed characters and structured ink "
        "strokes on the shell surface"
    ),
}


@dataclass(frozen=True)
class GateThresholds:
    minimum_passing_types: int
    area_envelope_fraction: float
    roi_precision: float
    roi_mass_advantage: float
    seed_l1: float
    seed_spearman: float
    auxiliary_margin: float


@dataclass(frozen=True)
class P3A0Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    normal_prompt: str
    defect_prompts: dict[str, str]
    image_size: int
    dtype: str
    timesteps: tuple[int, ...]
    seeds: tuple[int, ...]
    normal_count: int
    calibration_counts: dict[str, int]
    auxiliary_counts: dict[str, int]
    quantiles: tuple[float, float]
    normalization_epsilon: float
    roi_border_width: int
    otsu_bins: int
    roi_max_fraction: float
    area_envelope: tuple[float, float, float]
    random_baseline_count: int
    max_fpr: float
    gate: GateThresholds


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader whose mapping constructor rejects duplicate keys."""


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _strict_yaml(path: Path) -> Mapping[str, Any]:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid P3-A0 YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("P3-A0 configuration must be a YAML mapping")
    return loaded


def _require_exact_type(value: object, expected_type: type, field: str) -> None:
    if type(value) is not expected_type:
        raise ValueError(f"{field} must have exact {expected_type.__name__} type")


def _validate_scalar_types(raw: Mapping[str, Any], gate_raw: Mapping[object, object]) -> None:
    for field in ("schema_version", "category", "normal_prompt", "dtype"):
        _require_exact_type(raw[field], str, field)
    for field in (
        "image_size",
        "normal_count",
        "roi_border_width",
        "otsu_bins",
        "random_baseline_count",
    ):
        _require_exact_type(raw[field], int, field)
    for field in ("normalization_epsilon", "roi_max_fraction", "max_fpr"):
        _require_exact_type(raw[field], float, field)
    for field in ("defect_types", "timesteps", "seeds"):
        value = raw[field]
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        expected_type = str if field == "defect_types" else int
        for item in value:
            _require_exact_type(item, expected_type, field)
    for field, expected_length in (("quantiles", 2), ("area_envelope", 3)):
        value = raw[field]
        if not isinstance(value, list) or len(value) != expected_length:
            raise ValueError(f"{field} must be a list of length {expected_length}")
        for item in value:
            _require_exact_type(item, float, field)
    if not isinstance(raw["defect_prompts"], Mapping):
        raise ValueError("defect_prompts must be a mapping")
    for key, value in raw["defect_prompts"].items():
        _require_exact_type(key, str, "defect_prompts key")
        _require_exact_type(value, str, "defect_prompts value")
    for field in ("calibration_counts", "auxiliary_counts"):
        value = raw[field]
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be a mapping")
        for key, count in value.items():
            _require_exact_type(key, str, f"{field} key")
            _require_exact_type(count, int, f"{field} value")
    _require_exact_type(gate_raw["minimum_passing_types"], int, "gate.minimum_passing_types")
    for field in (
        "area_envelope_fraction",
        "roi_precision",
        "roi_mass_advantage",
        "seed_l1",
        "seed_spearman",
        "auxiliary_margin",
    ):
        _require_exact_type(gate_raw[field], float, f"gate.{field}")


def load_p3_a0_config(path: Path) -> P3A0Config:
    raw = _strict_yaml(path)
    keys = frozenset(P3A0Config.__dataclass_fields__)
    if set(raw) != keys:
        missing = sorted(keys - set(raw))
        extra = sorted(repr(key) for key in set(raw) - keys)
        raise ValueError(f"P3-A0 schema mismatch: missing={missing}, extra={extra}")
    gate_raw = raw["gate"]
    if not isinstance(gate_raw, Mapping):
        raise ValueError("gate must be a mapping")
    gate_keys = frozenset(GateThresholds.__dataclass_fields__)
    if set(gate_raw) != gate_keys:
        raise ValueError("gate must match the frozen threshold schema")
    _validate_scalar_types(raw, gate_raw)
    config = P3A0Config(
        schema_version=raw["schema_version"],
        category=raw["category"],
        defect_types=tuple(raw["defect_types"]) if isinstance(raw["defect_types"], list) else (),
        normal_prompt=raw["normal_prompt"],
        defect_prompts=dict(raw["defect_prompts"])
        if isinstance(raw["defect_prompts"], Mapping)
        else {},
        image_size=raw["image_size"],
        dtype=raw["dtype"],
        timesteps=tuple(raw["timesteps"]) if isinstance(raw["timesteps"], list) else (),
        seeds=tuple(raw["seeds"]) if isinstance(raw["seeds"], list) else (),
        normal_count=raw["normal_count"],
        calibration_counts=dict(raw["calibration_counts"])
        if isinstance(raw["calibration_counts"], Mapping)
        else {},
        auxiliary_counts=dict(raw["auxiliary_counts"])
        if isinstance(raw["auxiliary_counts"], Mapping)
        else {},
        quantiles=tuple(raw["quantiles"]) if isinstance(raw["quantiles"], list) else (),
        normalization_epsilon=raw["normalization_epsilon"],
        roi_border_width=raw["roi_border_width"],
        otsu_bins=raw["otsu_bins"],
        roi_max_fraction=raw["roi_max_fraction"],
        area_envelope=tuple(raw["area_envelope"]) if isinstance(raw["area_envelope"], list) else (),
        random_baseline_count=raw["random_baseline_count"],
        max_fpr=raw["max_fpr"],
        gate=GateThresholds(**dict(gate_raw)),
    )
    expected = {
        "schema_version": "p3_a0_hazelnut_spatial_response_v1",
        "category": "hazelnut",
        "defect_types": _TYPES,
        "normal_prompt": "a photo of a normal hazelnut",
        "defect_prompts": _PROMPTS,
        "image_size": 512,
        "dtype": "float16",
        "timesteps": (100, 400, 700),
        "seeds": (202608175, 202608176),
        "normal_count": 8,
        "calibration_counts": {"crack": 6, "cut": 5, "hole": 6, "print": 5},
        "auxiliary_counts": {defect_type: 12 for defect_type in _TYPES},
        "quantiles": (0.01, 0.99),
        "normalization_epsilon": 1.0e-8,
        "roi_border_width": 16,
        "otsu_bins": 256,
        "roi_max_fraction": 0.90,
        "area_envelope": (0.50, 2.00, 0.50),
        "random_baseline_count": 16,
        "max_fpr": 0.30,
        "gate": GateThresholds(
            minimum_passing_types=3,
            area_envelope_fraction=0.75,
            roi_precision=0.80,
            roi_mass_advantage=0.10,
            seed_l1=0.001,
            seed_spearman=0.20,
            auxiliary_margin=0.01,
        ),
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"{field} must match the frozen P3-A0 contract")
    return config

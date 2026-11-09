from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class P2A0Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    normal_prompt: str
    image_size: int
    dtype: str
    seed: int
    learning_rate: float
    steps: int
    capacity_defect_type: str
    context_fill_seed: int
    noise_seed: int
    eval_timesteps: tuple[int, ...]
    inside_noise_weight: float
    lambda_inside: float
    lambda_outside: float


_KEYS = frozenset(P2A0Config.__dataclass_fields__)
_DEFECT_TYPES = ("broken_large", "broken_small", "contamination")


def _mapping(value: object) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    return value


def _exact_keys(raw: Mapping[object, object]) -> None:
    actual = set(raw)
    missing = sorted(_KEYS - actual)
    extra = actual - _KEYS
    if missing:
        raise ValueError(f"configuration is missing required keys: {', '.join(missing)}")
    if extra:
        rendered = ", ".join(sorted(repr(key) for key in extra))
        raise ValueError(f"configuration has unexpected keys: {rendered}")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return value


def load_p2_a0_config(path: Path) -> P2A0Config:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    raw = _mapping(loaded)
    _exact_keys(raw)

    schema_version = _string(raw["schema_version"], "schema_version")
    category = _string(raw["category"], "category")
    defect_types_raw = raw["defect_types"]
    if not isinstance(defect_types_raw, list) or any(
        not isinstance(label, str) or not label for label in defect_types_raw
    ):
        raise ValueError("defect_types must be a list of non-empty strings")
    defect_types = tuple(defect_types_raw)
    normal_prompt = _string(raw["normal_prompt"], "normal_prompt")
    image_size = _integer(raw["image_size"], "image_size")
    dtype = _string(raw["dtype"], "dtype")
    seed = _integer(raw["seed"], "seed")
    learning_rate = _positive_float(raw["learning_rate"], "learning_rate")
    steps = _integer(raw["steps"], "steps")
    capacity_defect_type = _string(raw["capacity_defect_type"], "capacity_defect_type")
    context_fill_seed = _integer(raw["context_fill_seed"], "context_fill_seed")
    noise_seed = _integer(raw["noise_seed"], "noise_seed")
    timesteps_raw = raw["eval_timesteps"]
    if not isinstance(timesteps_raw, list) or any(
        type(value) is not int for value in timesteps_raw
    ):
        raise ValueError("eval_timesteps must be a list of integers")
    eval_timesteps = tuple(timesteps_raw)
    inside_noise_weight = _positive_float(raw["inside_noise_weight"], "inside_noise_weight")
    lambda_inside = _positive_float(raw["lambda_inside"], "lambda_inside")
    lambda_outside = _positive_float(raw["lambda_outside"], "lambda_outside")

    expected = {
        "schema_version": "p2_a0_capacity_v1",
        "category": "bottle",
        "defect_types": _DEFECT_TYPES,
        "normal_prompt": "a photo of a normal bottle",
        "image_size": 512,
        "dtype": "float16",
        "seed": 20260817,
        "learning_rate": 1e-4,
        "steps": 200,
        "capacity_defect_type": "broken_large",
        "context_fill_seed": 202608170,
        "noise_seed": 202608171,
        "eval_timesteps": (100, 400, 700),
        "inside_noise_weight": 4.0,
        "lambda_inside": 1.0,
        "lambda_outside": 0.1,
    }
    actual = {
        "schema_version": schema_version,
        "category": category,
        "defect_types": defect_types,
        "normal_prompt": normal_prompt,
        "image_size": image_size,
        "dtype": dtype,
        "seed": seed,
        "learning_rate": learning_rate,
        "steps": steps,
        "capacity_defect_type": capacity_defect_type,
        "context_fill_seed": context_fill_seed,
        "noise_seed": noise_seed,
        "eval_timesteps": eval_timesteps,
        "inside_noise_weight": inside_noise_weight,
        "lambda_inside": lambda_inside,
        "lambda_outside": lambda_outside,
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            rendered = list(expected_value) if field == "eval_timesteps" else expected_value
            if field == "defect_types":
                raise ValueError("defect_types must match the fixed bottle vocabulary")
            raise ValueError(f"{field} must be {rendered}")
    return P2A0Config(**actual)

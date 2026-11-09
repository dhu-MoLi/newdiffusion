from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from cadiff.p2_a0.fewshot_config import P2A0FewshotConfig


@dataclass(frozen=True)
class P2A1Config(P2A0FewshotConfig):
    type_rank: int
    ranking_target_ratio: float
    ranking_weight: float


_KEYS = frozenset(P2A1Config.__dataclass_fields__)
_CATEGORY_PROFILES: dict[str, dict[str, object]] = {
    "bottle": {
        "defect_types": ("broken_large", "broken_small", "contamination"),
        "normal_prompt": "a photo of a normal bottle",
        "capacity_defect_type": "broken_large",
    },
    "hazelnut": {
        "defect_types": ("crack", "cut", "hole", "print"),
        "normal_prompt": "a photo of a normal hazelnut",
        "capacity_defect_type": "crack",
    },
    "cable": {
        "defect_types": (
            "bent_wire",
            "cut_inner_insulation",
            "cut_outer_insulation",
            "poke_insulation",
        ),
        "normal_prompt": "a photo of a normal cable",
        "capacity_defect_type": "bent_wire",
    },
}


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


def load_p2_a1_config(path: Path) -> P2A1Config:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML configuration: {error}") from error
    raw = _mapping(loaded)
    _exact_keys(raw)
    defect_types_raw = raw["defect_types"]
    if not isinstance(defect_types_raw, list) or any(
        not isinstance(label, str) or not label for label in defect_types_raw
    ):
        raise ValueError("defect_types must be a list of non-empty strings")
    timesteps_raw = raw["eval_timesteps"]
    if not isinstance(timesteps_raw, list) or any(
        type(value) is not int for value in timesteps_raw
    ):
        raise ValueError("eval_timesteps must be a list of integers")
    actual: dict[str, object] = {
        "schema_version": _string(raw["schema_version"], "schema_version"),
        "category": _string(raw["category"], "category"),
        "defect_types": tuple(defect_types_raw),
        "normal_prompt": _string(raw["normal_prompt"], "normal_prompt"),
        "image_size": _integer(raw["image_size"], "image_size"),
        "dtype": _string(raw["dtype"], "dtype"),
        "seed": _integer(raw["seed"], "seed"),
        "learning_rate": _positive_float(raw["learning_rate"], "learning_rate"),
        "steps": _integer(raw["steps"], "steps"),
        "capacity_defect_type": _string(
            raw["capacity_defect_type"], "capacity_defect_type"
        ),
        "context_fill_seed": _integer(raw["context_fill_seed"], "context_fill_seed"),
        "noise_seed": _integer(raw["noise_seed"], "noise_seed"),
        "eval_timesteps": tuple(timesteps_raw),
        "inside_noise_weight": _positive_float(
            raw["inside_noise_weight"], "inside_noise_weight"
        ),
        "lambda_inside": _positive_float(raw["lambda_inside"], "lambda_inside"),
        "lambda_outside": _positive_float(raw["lambda_outside"], "lambda_outside"),
        "train_role": _string(raw["train_role"], "train_role"),
        "eval_role": _string(raw["eval_role"], "eval_role"),
        "wrong_type_policy": _string(raw["wrong_type_policy"], "wrong_type_policy"),
        "correct_wrong_ratio_limit": _positive_float(
            raw["correct_wrong_ratio_limit"], "correct_wrong_ratio_limit"
        ),
        "correct_normal_ratio_limit": _positive_float(
            raw["correct_normal_ratio_limit"], "correct_normal_ratio_limit"
        ),
        "type_rank": _integer(raw["type_rank"], "type_rank"),
        "ranking_target_ratio": _positive_float(
            raw["ranking_target_ratio"], "ranking_target_ratio"
        ),
        "ranking_weight": _positive_float(raw["ranking_weight"], "ranking_weight"),
    }
    category = str(actual["category"])
    try:
        profile = _CATEGORY_PROFILES[category]
    except KeyError as error:
        supported = ", ".join(sorted(_CATEGORY_PROFILES))
        raise ValueError(f"category must be one of: {supported}") from error
    expected: dict[str, object] = {
        "schema_version": "p2_a1_factorized_v1",
        "image_size": 512,
        "dtype": "float16",
        "seed": 20260817,
        "learning_rate": 1e-4,
        "steps": 1200,
        "context_fill_seed": 202608170,
        "noise_seed": 202608171,
        "eval_timesteps": (100, 400, 700),
        "inside_noise_weight": 4.0,
        "lambda_inside": 1.0,
        "lambda_outside": 0.1,
        "train_role": "fewshot_train",
        "eval_role": "open_dev_report",
        "wrong_type_policy": "cyclic_next",
        "correct_wrong_ratio_limit": 1.0,
        "correct_normal_ratio_limit": 1.0,
        "type_rank": 8,
        "ranking_target_ratio": 0.95,
        "ranking_weight": 0.25,
        **profile,
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            if field == "defect_types":
                raise ValueError(
                    f"defect_types must match the fixed {category} vocabulary"
                )
            rendered = list(expected_value) if field == "eval_timesteps" else expected_value
            raise ValueError(f"{field} must be {rendered}")
    if not 0.0 < float(actual["ranking_target_ratio"]) < 1.0:
        raise ValueError("ranking_target_ratio must be between zero and one")
    return P2A1Config(**actual)  # type: ignore[arg-type]

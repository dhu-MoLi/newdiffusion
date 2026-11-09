from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from cadiff.p2_a0.fewshot_config import (
    _DEFECT_TYPES,
    P2A0FewshotConfig,
    _integer,
    _mapping,
    _positive_float,
    _string,
)


@dataclass(frozen=True)
class P2A0SingleTypeConfig(P2A0FewshotConfig):
    target_defect_type: str

    @property
    def active_defect_types(self) -> tuple[str, ...]:
        return (self.target_defect_type,)

    @property
    def expected_train_counts(self) -> dict[str, int]:
        return {self.target_defect_type: 6}

    @property
    def expected_eval_counts(self) -> dict[str, int]:
        return {self.target_defect_type: 14}

    @property
    def gate_policy(self) -> str:
        return "correct_vs_normal"


_KEYS = frozenset(P2A0SingleTypeConfig.__dataclass_fields__)


def _exact_keys(raw: Mapping[object, object]) -> None:
    actual = set(raw)
    missing = sorted(_KEYS - actual)
    extra = actual - _KEYS
    if missing:
        raise ValueError(f"configuration is missing required keys: {', '.join(missing)}")
    if extra:
        rendered = ", ".join(sorted(repr(key) for key in extra))
        raise ValueError(f"configuration has unexpected keys: {rendered}")


def load_p2_a0_single_type_config(path: Path) -> P2A0SingleTypeConfig:
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
        "target_defect_type": _string(
            raw["target_defect_type"], "target_defect_type"
        ),
    }
    expected: dict[str, object] = {
        "schema_version": "p2_a0_single_type_v1",
        "category": "bottle",
        "defect_types": _DEFECT_TYPES,
        "normal_prompt": "a photo of a normal bottle",
        "image_size": 512,
        "dtype": "float16",
        "seed": 20260817,
        "learning_rate": 1e-4,
        "steps": 400,
        "capacity_defect_type": "broken_large",
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
        "target_defect_type": "broken_large",
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            if field == "defect_types":
                raise ValueError("defect_types must match the fixed bottle vocabulary")
            rendered = list(expected_value) if field == "eval_timesteps" else expected_value
            raise ValueError(f"{field} must be {rendered}")
    return P2A0SingleTypeConfig(**actual)  # type: ignore[arg-type]


__all__ = ["P2A0SingleTypeConfig", "load_p2_a0_single_type_config"]

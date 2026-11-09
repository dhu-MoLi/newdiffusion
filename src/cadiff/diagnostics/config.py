from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

DIAGNOSTIC_SCHEMA_VERSION = "gate_a_rootcause_v1"
NOISE_REPLAY_SCHEMA_VERSION = "gate_a_noise_replay_v1"
FROZEN_METRICS_SHA256 = (
    "783923689c30d780db1fd7cfc1f31e6ad25cddc974117d3fa9cef6c990239005"
)
GENERIC_PROMPT = "a photo of a normal object"


class PromptMode(StrEnum):
    CLASS = "class"
    EMPTY = "empty"
    GENERIC = "generic"


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    phase: str
    steps: int
    prompt_mode: PromptMode
    dtype: str
    repeat: int


@dataclass(frozen=True)
class DiagnosticConfig:
    schema_version: str
    frozen_metrics_sha256: str
    generic_prompt: str
    single_factor_cases: tuple[DiagnosticCase, ...]
    maximum_combinations: int
    maximum_full_candidates: int
    bottle_regression_max: float


@dataclass(frozen=True)
class NoiseReplayConfig:
    schema_version: str
    algorithm: str
    case_id: str
    steps: int
    prompt_mode: PromptMode
    dtype: str
    guidance_scale: float
    eta: float
    replay_strength: float
    seed: int


EXPECTED_CASES = (
    DiagnosticCase(
        "d0_class_50_fp16_r1", "baseline_repeat", 50, PromptMode.CLASS, "float16", 1
    ),
    DiagnosticCase(
        "d0_class_50_fp16_r2", "baseline_repeat", 50, PromptMode.CLASS, "float16", 2
    ),
    DiagnosticCase(
        "d1_class_25_fp16", "single_factor", 25, PromptMode.CLASS, "float16", 1
    ),
    DiagnosticCase(
        "d1_class_100_fp16", "single_factor", 100, PromptMode.CLASS, "float16", 1
    ),
    DiagnosticCase(
        "d2_empty_50_fp16", "single_factor", 50, PromptMode.EMPTY, "float16", 1
    ),
    DiagnosticCase(
        "d2_generic_50_fp16", "single_factor", 50, PromptMode.GENERIC, "float16", 1
    ),
    DiagnosticCase(
        "d3_class_50_fp32", "single_factor", 50, PromptMode.CLASS, "float32", 1
    ),
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "frozen_metrics_sha256",
        "generic_prompt",
        "maximum_combinations",
        "maximum_full_candidates",
        "bottle_regression_max",
        "cases",
    }
)
CASE_KEYS = frozenset({"case_id", "phase", "steps", "prompt_mode", "dtype", "repeat"})
NOISE_REPLAY_KEYS = frozenset(
    {
        "schema_version",
        "algorithm",
        "case_id",
        "steps",
        "prompt_mode",
        "dtype",
        "guidance_scale",
        "eta",
        "replay_strength",
        "seed",
    }
)


def _strict_mapping(raw: object, expected: frozenset[str], context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a YAML mapping")
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(f"{context} has unknown={sorted(unknown)} missing={sorted(missing)}")
    return raw


def _exact_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an exact integer")
    return value


def _case(raw: object, index: int) -> DiagnosticCase:
    values = _strict_mapping(raw, CASE_KEYS, f"cases[{index}]")
    case_id = values["case_id"]
    phase = values["phase"]
    dtype = values["dtype"]
    prompt_mode = values["prompt_mode"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"cases[{index}].case_id must be a non-empty string")
    if phase not in {"baseline_repeat", "single_factor"}:
        raise ValueError(f"cases[{index}].phase is invalid")
    if dtype not in {"float16", "float32"}:
        raise ValueError(f"cases[{index}].dtype is invalid")
    try:
        parsed_prompt = PromptMode(prompt_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cases[{index}].prompt_mode is invalid") from error
    steps = _exact_int(values["steps"], f"cases[{index}].steps")
    repeat = _exact_int(values["repeat"], f"cases[{index}].repeat")
    if steps <= 0 or repeat <= 0:
        raise ValueError(f"cases[{index}] steps and repeat must be positive")
    return DiagnosticCase(case_id, str(phase), steps, parsed_prompt, str(dtype), repeat)


def load_diagnostic_config(path: Path) -> DiagnosticConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = _strict_mapping(raw, TOP_LEVEL_KEYS, "diagnostic configuration")
    if values["schema_version"] != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("schema_version must be gate_a_rootcause_v1")
    if values["frozen_metrics_sha256"] != FROZEN_METRICS_SHA256:
        raise ValueError("frozen_metrics_sha256 does not identify the audited P0 report")
    if values["generic_prompt"] != GENERIC_PROMPT:
        raise ValueError(f"generic_prompt must be {GENERIC_PROMPT!r}")
    maximum_combinations = _exact_int(values["maximum_combinations"], "maximum_combinations")
    maximum_full_candidates = _exact_int(
        values["maximum_full_candidates"], "maximum_full_candidates"
    )
    if maximum_combinations != 3 or maximum_full_candidates != 2:
        raise ValueError("diagnostic candidate limits must remain 3 combinations and 2 full runs")
    regression = values["bottle_regression_max"]
    if type(regression) not in {float, int} or isinstance(regression, bool):
        raise ValueError("bottle_regression_max must be a finite number")
    bottle_regression_max = float(regression)
    if not math.isfinite(bottle_regression_max) or bottle_regression_max != 0.005:
        raise ValueError("bottle_regression_max must be exactly 0.005")
    raw_cases = values["cases"]
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a YAML list")
    cases = tuple(_case(item, index) for index, item in enumerate(raw_cases))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case_id values must be unique")
    if cases != EXPECTED_CASES:
        raise ValueError("cases must match the frozen D0-D3 diagnostic matrix")
    return DiagnosticConfig(
        DIAGNOSTIC_SCHEMA_VERSION,
        FROZEN_METRICS_SHA256,
        GENERIC_PROMPT,
        cases,
        maximum_combinations,
        maximum_full_candidates,
        bottle_regression_max,
    )


def _fixed_number(value: object, expected: float, field: str) -> float:
    if type(value) not in {float, int} or isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed != expected:
        raise ValueError(f"{field} must be exactly {expected}")
    return parsed


def load_noise_replay_config(path: Path) -> NoiseReplayConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = _strict_mapping(
        raw, NOISE_REPLAY_KEYS, "noise replay configuration"
    )
    fixed_strings = {
        "schema_version": NOISE_REPLAY_SCHEMA_VERSION,
        "algorithm": "trajectory_consistent_noise_replay",
        "case_id": "d5_noise_replay_50_class_fp16",
        "prompt_mode": "class",
        "dtype": "float16",
    }
    for field, expected in fixed_strings.items():
        if values[field] != expected:
            raise ValueError(f"{field} must be {expected!r}")
    steps = _exact_int(values["steps"], "steps")
    seed = _exact_int(values["seed"], "seed")
    if steps != 50:
        raise ValueError("steps must be exactly 50")
    if seed != 20260817:
        raise ValueError("seed must be exactly 20260817")
    return NoiseReplayConfig(
        NOISE_REPLAY_SCHEMA_VERSION,
        "trajectory_consistent_noise_replay",
        "d5_noise_replay_50_class_fp16",
        steps,
        PromptMode.CLASS,
        "float16",
        _fixed_number(values["guidance_scale"], 1.0, "guidance_scale"),
        _fixed_number(values["eta"], 0.0, "eta"),
        _fixed_number(values["replay_strength"], 1.0, "replay_strength"),
        seed,
    )

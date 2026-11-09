from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

_SCHEMA_VERSION = "p3_a0_hazelnut_spatial_response_v1"
_TYPES = ("crack", "cut", "hole", "print")
_SEEDS = (202608175, 202608176)
_NORMAL_COUNT = 8
_AUXILIARY_COUNT = 12


@dataclass(frozen=True)
class ResponseJob:
    record_id: str
    cohort: Literal["normal_reference", "auxiliary_real"]
    sample_id: str
    source_defect_type: str | None
    prompt_defect_type: str
    seed: int


@dataclass(frozen=True)
class ResponseSchedule:
    normal: tuple[ResponseJob, ...]
    auxiliary: tuple[ResponseJob, ...]


def _validate_ids(ids: tuple[str, ...], expected_count: int, field: str) -> None:
    if (
        len(ids) != expected_count
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(f"{field} must contain {expected_count} unique non-empty IDs")


def _record_id(
    cohort: Literal["normal_reference", "auxiliary_real"],
    sample_id: str,
    source_defect_type: str | None,
    prompt_defect_type: str,
    seed: int,
) -> str:
    identity = "\n".join(
        (
            _SCHEMA_VERSION,
            cohort,
            sample_id,
            source_defect_type if source_defect_type is not None else "none",
            prompt_defect_type,
            str(seed),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _job(
    cohort: Literal["normal_reference", "auxiliary_real"],
    sample_id: str,
    source_defect_type: str | None,
    prompt_defect_type: str,
    seed: int,
) -> ResponseJob:
    return ResponseJob(
        record_id=_record_id(cohort, sample_id, source_defect_type, prompt_defect_type, seed),
        cohort=cohort,
        sample_id=sample_id,
        source_defect_type=source_defect_type,
        prompt_defect_type=prompt_defect_type,
        seed=seed,
    )


def build_response_schedule(
    normal_ids: tuple[str, ...],
    auxiliary_ids_by_type: Mapping[str, tuple[str, ...]],
    defect_types: tuple[str, ...],
    seeds: tuple[int, ...],
) -> ResponseSchedule:
    if defect_types != _TYPES:
        raise ValueError("defect_types must match the frozen P3-A0 vocabulary and order")
    if seeds != _SEEDS:
        raise ValueError("seeds must match the frozen P3-A0 order")
    _validate_ids(normal_ids, _NORMAL_COUNT, "normal_ids")
    if set(auxiliary_ids_by_type) != set(_TYPES):
        raise ValueError("auxiliary IDs must contain exactly the frozen defect types")
    for source_type in _TYPES:
        ids = auxiliary_ids_by_type[source_type]
        if not isinstance(ids, tuple):
            raise ValueError("auxiliary IDs must be tuples")
        _validate_ids(ids, _AUXILIARY_COUNT, f"auxiliary IDs for {source_type}")

    normal = tuple(
        _job("normal_reference", sample_id, None, prompt_type, seed)
        for sample_id in normal_ids
        for prompt_type in defect_types
        for seed in seeds
    )
    auxiliary = tuple(
        _job("auxiliary_real", sample_id, source_type, prompt_type, seed)
        for source_type in defect_types
        for sample_id in auxiliary_ids_by_type[source_type]
        for prompt_type in defect_types
        for seed in seeds
    )
    if len({job.record_id for job in (*normal, *auxiliary)}) != len(normal) + len(auxiliary):
        raise ValueError("response schedule record IDs must be unique")
    return ResponseSchedule(normal=normal, auxiliary=auxiliary)

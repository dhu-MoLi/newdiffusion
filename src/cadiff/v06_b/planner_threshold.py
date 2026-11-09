from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Literal

import numpy as np

from .config import DEFECT_TYPES


@dataclass(frozen=True, eq=False)
class ThresholdRecord:
    record_id: str
    role: Literal["fewshot_train_geometry_support"]
    defect_type: str
    source_id: str
    target_id: str
    q: np.ndarray
    target: np.ndarray
    proposal: np.ndarray


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    macro_dice: float
    support_record_ids: tuple[str, ...]


def _threshold(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
    ):
        raise ValueError("threshold must lie strictly between zero and one")
    return float(value)


def _binary_mask(value: object, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must be an HxW numpy array matching q")
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return value.astype(bool, copy=True)


def _q_array(value: object) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.size == 0
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or np.any((value < 0.0) | (value > 1.0))
    ):
        raise ValueError("q must be a non-empty finite floating HxW array in [0,1]")
    return np.ascontiguousarray(value)


def finalize_realization(
    q: np.ndarray,
    proposal: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    q_array = _q_array(q)
    proposal_array = _binary_mask(
        proposal,
        name="proposal",
        shape=tuple(q_array.shape),
    )
    threshold_value = _threshold(threshold)
    final = np.ascontiguousarray(q_array >= threshold_value, dtype=bool)
    if not final.any():
        raise ValueError("realization mask is empty")
    if final.all():
        raise ValueError("realization mask covers the full image")
    if (final & ~proposal_array).any():
        raise ValueError("realization mask escaped proposal")
    return final


def _validated_records(
    records: object,
    *,
    calibration_target_ids: object,
    defect_types: tuple[str, ...] = DEFECT_TYPES,
) -> tuple[ThresholdRecord, ...]:
    if (
        not isinstance(calibration_target_ids, tuple)
        or not calibration_target_ids
        or any(
            not isinstance(target_id, str) or not target_id
            for target_id in calibration_target_ids
        )
        or len(set(calibration_target_ids)) != len(calibration_target_ids)
    ):
        raise ValueError("calibration target authority must contain unique non-empty IDs")
    if (
        not isinstance(records, tuple)
        or not records
        or any(not isinstance(record, ThresholdRecord) for record in records)
    ):
        raise ValueError("records must be a non-empty tuple of ThresholdRecord values")
    record_ids = tuple(record.record_id for record in records)
    if (
        any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
        or len(set(record_ids)) != len(record_ids)
    ):
        raise ValueError("records must have unique non-empty record_id values")
    if any(record.role != "fewshot_train_geometry_support" for record in records):
        raise ValueError("records must have role fewshot_train_geometry_support")
    if {record.defect_type for record in records} != set(defect_types):
        raise ValueError("records must cover all four configured defect types")
    if any(
        not isinstance(record.source_id, str)
        or not record.source_id
        or not isinstance(record.target_id, str)
        or not record.target_id
        for record in records
    ):
        raise ValueError("support source and target IDs must be non-empty strings")
    authority = set(calibration_target_ids)
    represented = {record.target_id for record in records}
    if not represented.issubset(authority):
        raise ValueError("record target is outside calibration target authority")
    if represented != authority:
        raise ValueError("every calibration target must be represented")

    for record in records:
        q = _q_array(record.q)
        target = _binary_mask(record.target, name="target", shape=tuple(q.shape))
        proposal = _binary_mask(record.proposal, name="proposal", shape=tuple(q.shape))
        if (target & ~proposal).any():
            raise ValueError("target must be a subset of proposal")
        if not target.any():
            raise ValueError("target must be non-empty")
        if target.all():
            raise ValueError("target must not cover the full image")
        if not proposal.any():
            raise ValueError("proposal must be non-empty")
        if proposal.all():
            raise ValueError("proposal must not cover the full image")
        if np.any(q[~proposal] != 0.0):
            raise ValueError("support q must be exactly zero outside proposal")
    return records


def calibrate_realization_threshold(
    records: tuple[ThresholdRecord, ...],
    *,
    candidates: tuple[float, ...],
    calibration_target_ids: tuple[str, ...],
    defect_types: tuple[str, ...] = DEFECT_TYPES,
) -> ThresholdCalibration:
    validated = _validated_records(
        records,
        calibration_target_ids=calibration_target_ids,
        defect_types=defect_types,
    )
    if (
        not isinstance(candidates, tuple)
        or not candidates
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("candidates must be a non-empty tuple of unique thresholds")
    candidate_values = tuple(_threshold(candidate) for candidate in candidates)
    legal: list[tuple[float, Decimal, int, float]] = []
    for index, candidate in enumerate(candidate_values):
        by_type: dict[str, list[float]] = {label: [] for label in defect_types}
        for record in validated:
            try:
                final = finalize_realization(
                    record.q,
                    record.proposal,
                    threshold=candidate,
                )
            except ValueError:
                break
            target = record.target.astype(bool, copy=False)
            intersection = int(np.count_nonzero(final & target))
            denominator = int(np.count_nonzero(final)) + int(np.count_nonzero(target))
            by_type[record.defect_type].append(2.0 * intersection / denominator)
        else:
            per_type = tuple(
                sum(by_type[label]) / len(by_type[label])
                for label in defect_types
            )
            macro_dice = sum(per_type) / len(per_type)
            distance = abs(Decimal(str(candidate)) - Decimal("0.50"))
            legal.append((macro_dice, distance, index, candidate))
    if not legal:
        raise ValueError("no legal realization threshold candidate")
    best = legal[0]
    for candidate in legal[1:]:
        if candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[1] < best[1]
        ):
            best = candidate
    return ThresholdCalibration(
        threshold=best[3],
        macro_dice=best[0],
        support_record_ids=tuple(record.record_id for record in validated),
    )


__all__ = [
    "ThresholdCalibration",
    "ThresholdRecord",
    "calibrate_realization_threshold",
    "finalize_realization",
]

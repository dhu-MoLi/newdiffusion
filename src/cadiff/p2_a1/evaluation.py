from __future__ import annotations

import math
from typing import Any

import torch

from cadiff.p2_a0.all_pairs import (
    NOISE_BANK_OFFSETS,
    _validate_execution_records,
    evaluate_all_pairs,
    summarize_all_pairs,
)
from cadiff.p2_a0.fewshot import FewshotSplit
from cadiff.p2_a0.train_vs_heldout import summarize_train_vs_heldout
from cadiff.p2_a1.config import P2A1Config

HELDOUT_TOP1_THRESHOLDS = {
    "broken_large": 0.90,
    "broken_small": 0.60,
    "contamination": 0.80,
}
HELDOUT_CORRECT_NORMAL_RATIO_LIMIT = 0.98


def _ordered_type_diagnostic(
    train: dict[str, Any],
    heldout: dict[str, Any],
    labels: tuple[str, ...],
    normal_ratio_limit: float,
    *,
    schema_version: str,
) -> dict[str, Any]:
    expected_labels = set(labels)
    if len(labels) != len(expected_labels):
        raise ValueError("P2-A1 diagnostic labels must be unique")
    if (
        type(normal_ratio_limit) is not float
        or not math.isfinite(normal_ratio_limit)
        or normal_ratio_limit <= 0.0
    ):
        raise ValueError("P2-A1 normal ratio limit must be finite and positive")
    try:
        train_types = train["combined"]["per_true_type"]
        heldout_combined = heldout["combined"]
        heldout_types = heldout_combined["per_true_type"]
        confusion = heldout_combined["confusion_matrix"]
    except (KeyError, TypeError) as error:
        raise ValueError("P2-A1 diagnostic summary schema is invalid") from error
    if (
        set(train_types) != expected_labels
        or set(heldout_types) != expected_labels
        or set(confusion) != expected_labels
    ):
        raise ValueError("P2-A1 diagnostic summaries must contain every frozen label")

    predicted_labels: set[str] = set()
    per_type: dict[str, Any] = {}
    for label in labels:
        row = confusion[label]
        if not isinstance(row, dict) or set(row) != expected_labels:
            raise ValueError(f"P2-A1 diagnostic confusion row is invalid for {label}")
        for predicted, count in row.items():
            if type(count) is not int or count < 0:
                raise ValueError(
                    f"P2-A1 diagnostic confusion count is invalid for {label}"
                )
            if count > 0:
                predicted_labels.add(predicted)
        top1 = heldout_types[label].get("top1_accuracy")
        normal_ratio = heldout_types[label].get("correct_normal_ratio")
        if (
            type(top1) is not float
            or not 0.0 <= top1 <= 1.0
            or type(normal_ratio) is not float
            or not math.isfinite(normal_ratio)
            or normal_ratio < 0.0
        ):
            raise ValueError(f"P2-A1 diagnostic metrics are invalid for {label}")
        per_type[label] = {
            "heldout_top1_accuracy": top1,
            "heldout_correct_normal_ratio": normal_ratio,
            "beats_normal": normal_ratio < normal_ratio_limit,
        }

    all_types_beat_normal = all(item["beats_normal"] for item in per_type.values())
    predicted_label_coverage = len(predicted_labels)
    route_noncollapse_observed = predicted_label_coverage >= 2
    return {
        "schema_version": schema_version,
        "diagnostic_only": True,
        "gate_passed": None,
        "promotion": "not_applicable",
        "internal_top1_blocking": False,
        "normal_ratio_limit_exclusive": normal_ratio_limit,
        "all_types_beat_normal": all_types_beat_normal,
        "predicted_label_coverage": predicted_label_coverage,
        "predicted_labels": sorted(predicted_labels),
        "route_noncollapse_observed": route_noncollapse_observed,
        "generation_probe_ready": (
            all_types_beat_normal and route_noncollapse_observed
        ),
        "per_type": per_type,
    }


def evaluate_partition(
    *,
    model: Any,
    unet: Any,
    scheduler: Any,
    prepared: tuple[Any, ...],
    prompt: torch.Tensor,
    config: P2A1Config,
    context_seed_offset: int,
) -> tuple[dict[str, Any], ...]:
    if type(context_seed_offset) is not int or context_seed_offset < 0:
        raise ValueError("context_seed_offset must be a non-negative integer")
    return tuple(
        evaluate_all_pairs(
            model=model,
            unet=unet,
            scheduler=scheduler,
            prepared=sample,
            prompt=prompt,
            config=config,
            sample_index=sample_index,
            bank_index=bank_index,
            noise_seed_offset=noise_seed_offset,
            context_seed_offset=context_seed_offset,
        )
        for bank_index, noise_seed_offset in enumerate(NOISE_BANK_OFFSETS)
        for sample_index, sample in enumerate(prepared)
    )


def validate_partition_records(
    records: tuple[dict[str, Any], ...],
    *,
    split: FewshotSplit,
    partition: str,
    config: P2A1Config,
) -> None:
    if partition == "train":
        validation_split = FewshotSplit(train=(), evaluation=split.train)
    elif partition == "heldout":
        validation_split = split
    else:
        raise ValueError("partition must be train or heldout")
    _validate_execution_records(records, split=validation_split, config=config)


def _decision_from_summaries(
    train: dict[str, Any],
    heldout: dict[str, Any],
    labels: tuple[str, ...],
    *,
    category: str = "bottle",
    normal_ratio_limit: float = 1.0,
) -> dict[str, Any]:
    if category == "hazelnut":
        expected = {"crack", "cut", "hole", "print"}
        if set(labels) != expected or len(labels) != len(expected):
            raise ValueError("P2-A1 hazelnut labels do not match the frozen diagnostic")
        return _ordered_type_diagnostic(
            train,
            heldout,
            labels,
            normal_ratio_limit,
            schema_version="p2_a1_hazelnut_diagnostic_v1",
        )
    if category == "cable":
        expected = {
            "bent_wire",
            "cut_inner_insulation",
            "cut_outer_insulation",
            "poke_insulation",
        }
        if set(labels) != expected or len(labels) != len(expected):
            raise ValueError("P2-A1 cable labels do not match the frozen diagnostic")
        return _ordered_type_diagnostic(
            train,
            heldout,
            labels,
            normal_ratio_limit,
            schema_version="p2_a1_cable_diagnostic_v1",
        )
    if category != "bottle":
        raise ValueError("P2-A1 decision category is unsupported")
    if set(labels) != set(HELDOUT_TOP1_THRESHOLDS) or len(labels) != len(
        HELDOUT_TOP1_THRESHOLDS
    ):
        raise ValueError("P2-A1 decision labels do not match the frozen gate")
    try:
        train_types = train["combined"]["per_true_type"]
        heldout_types = heldout["combined"]["per_true_type"]
    except (KeyError, TypeError) as error:
        raise ValueError("P2-A1 split summary schema is invalid") from error
    if set(train_types) != set(labels) or set(heldout_types) != set(labels):
        raise ValueError("P2-A1 split summaries must contain every frozen label")
    per_type: dict[str, Any] = {}
    for label in labels:
        top1 = heldout_types[label].get("top1_accuracy")
        normal_ratio = heldout_types[label].get("correct_normal_ratio")
        if (
            type(top1) is not float
            or not 0.0 <= top1 <= 1.0
            or type(normal_ratio) is not float
            or not math.isfinite(normal_ratio)
            or normal_ratio < 0.0
        ):
            raise ValueError(f"P2-A1 heldout metrics are invalid for {label}")
        minimum = HELDOUT_TOP1_THRESHOLDS[label]
        top1_met = top1 >= minimum
        ratio_met = normal_ratio < HELDOUT_CORRECT_NORMAL_RATIO_LIMIT
        per_type[label] = {
            "heldout_top1_accuracy": top1,
            "heldout_top1_minimum": minimum,
            "heldout_top1_met": top1_met,
            "heldout_correct_normal_ratio": normal_ratio,
            "heldout_correct_normal_ratio_limit_exclusive": (
                HELDOUT_CORRECT_NORMAL_RATIO_LIMIT
            ),
            "heldout_correct_normal_ratio_met": ratio_met,
            "gate_met": top1_met and ratio_met,
        }
    gate_passed = all(result["gate_met"] for result in per_type.values())
    return {
        "schema_version": "p2_a1_factorized_decision_v1",
        "gate_passed": gate_passed,
        "promotion": "promote" if gate_passed else "blocked",
        "per_type": per_type,
    }


def summarize_p2_a1_evaluation(
    train_records: tuple[dict[str, Any], ...],
    heldout_records: tuple[dict[str, Any], ...],
    *,
    split: FewshotSplit,
    config: P2A1Config,
) -> dict[str, Any]:
    validate_partition_records(
        train_records, split=split, partition="train", config=config
    )
    validate_partition_records(
        heldout_records, split=split, partition="heldout", config=config
    )
    labels = tuple(config.defect_types)
    train = summarize_all_pairs(train_records, labels)
    heldout = summarize_all_pairs(heldout_records, labels)
    return {
        "train": train,
        "heldout": heldout,
        "generalization": summarize_train_vs_heldout(train, heldout, labels),
        "decision": _decision_from_summaries(
            train,
            heldout,
            labels,
            category=config.category,
            normal_ratio_limit=config.correct_normal_ratio_limit,
        ),
    }

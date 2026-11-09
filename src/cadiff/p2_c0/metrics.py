from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    auroc: float
    average_precision: float
    f1_max: float


@dataclass(frozen=True)
class PixelMetrics:
    auroc: float
    average_precision: float
    f1_max: float
    au_pro: float


def _validated_binary(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if scores.size == 0 or scores.shape != labels.shape:
        raise ValueError("scores and labels must be aligned and non-empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    labels = labels.astype(np.uint8)
    if np.unique(labels).size != 2:
        raise ValueError("both classes are required")
    return scores, labels


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> BinaryMetrics:
    scores, labels = _validated_binary(scores, labels)
    positives = int(labels.sum())
    negatives = labels.size - positives
    ranks = _average_ranks(scores)
    auroc = float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels, dtype=np.float64)
    predicted_positives = np.arange(1, labels.size + 1, dtype=np.float64)
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    average_precision = float(precision[sorted_labels == 1].sum() / positives)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return BinaryMetrics(
        auroc=auroc,
        average_precision=average_precision,
        f1_max=float(f1.max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for row, column in zip(*np.nonzero(mask), strict=True):
        if seen[row, column]:
            continue
        stack = [(int(row), int(column))]
        seen[row, column] = True
        coordinates: list[int] = []
        while stack:
            current_row, current_column = stack.pop()
            coordinates.append(current_row * width + current_column)
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not seen[next_row, next_column]
                ):
                    seen[next_row, next_column] = True
                    stack.append((next_row, next_column))
        result.append(np.asarray(coordinates, dtype=np.int64))
    return result


def _au_pro(scores: np.ndarray, masks: np.ndarray, max_fpr: float) -> float:
    components: list[tuple[int, np.ndarray]] = []
    for image_index, mask in enumerate(masks):
        components.extend((image_index, item) for item in _components(mask.astype(bool)))
    if not components:
        raise ValueError("at least one positive region is required")
    negatives = masks == 0
    negative_count = int(negatives.sum())
    if negative_count == 0:
        raise ValueError("negative pixels are required for AU-PRO")
    component_weights = np.zeros(scores.shape, dtype=np.float64)
    component_count = len(components)
    for image_index, region in components:
        component_weights[image_index].reshape(-1)[region] = 1.0 / (
            component_count * region.size
        )
    flat_scores = scores.reshape(-1)
    order = np.argsort(-flat_scores, kind="mergesort")
    ordered_scores = flat_scores[order]
    cumulative_false_positives = np.cumsum(
        negatives.reshape(-1)[order], dtype=np.float64
    )
    cumulative_pro = np.cumsum(component_weights.reshape(-1)[order])
    score_group_ends = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    fprs = cumulative_false_positives[score_group_ends] / negative_count
    pros = cumulative_pro[score_group_ends]
    last_at_fpr = np.r_[fprs[1:] != fprs[:-1], True]
    fprs = fprs[last_at_fpr]
    pros = pros[last_at_fpr]
    if fprs[0] > 0.0:
        fprs = np.insert(fprs, 0, 0.0)
        pros = np.insert(pros, 0, 0.0)
    if fprs[-1] < max_fpr:
        fprs = np.append(fprs, max_fpr)
        pros = np.append(pros, pros[-1])
    elif max_fpr not in fprs:
        index = int(np.searchsorted(fprs, max_fpr))
        interpolated = float(
            np.interp(max_fpr, fprs[index - 1 : index + 1], pros[index - 1 : index + 1])
        )
        fprs = np.insert(fprs, index, max_fpr)
        pros = np.insert(pros, index, interpolated)
    keep = fprs <= max_fpr
    return float(np.trapz(pros[keep], fprs[keep]) / max_fpr)


def pixel_metrics(
    score_maps: np.ndarray, masks: np.ndarray, *, max_fpr: float
) -> PixelMetrics:
    scores = np.asarray(score_maps, dtype=np.float64)
    labels = np.asarray(masks)
    if scores.ndim != 3 or scores.shape != labels.shape:
        raise ValueError("score_maps and masks must be aligned N x H x W arrays")
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    binary = binary_metrics(scores.reshape(-1), labels.reshape(-1))
    return PixelMetrics(
        auroc=binary.auroc,
        average_precision=binary.average_precision,
        f1_max=binary.f1_max,
        au_pro=_au_pro(scores, labels.astype(np.uint8), max_fpr),
    )

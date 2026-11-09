"""Diagnostic placement for masks that already cover most of a merged TokenCut.

Official scale maps the source TokenCut onto one target object, so a stuck-together
pair shrinks into a single product and the clip deletes the second instance.
This arm multiplies that scale by sqrt(2) and keeps the unclipped template, including
pixels outside the target TokenCut. It is not official H and is not wired into generate().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cadiff.v06_b.a2_component_place import (
    ANGLES_DEG,
    place_shape_max_overlap_clip,
    rotate_binary_mask,
    sample_area_factor,
)
from cadiff.v06_b.a2_shape_instantiate import (
    EmptyFeasiblePlacement,
    object_scale,
    scale_binary_mask,
)

PAIR_INSTANCE_COUNT = 2


@dataclass(frozen=True)
class PairOutsidePlacement:
    mask: NDArray[np.bool_]
    on_object: NDArray[np.bool_]
    outside: NDArray[np.bool_]
    scale: float
    factor: float
    object_scale: float
    angle_deg: int


def pair_instance_scale(
    source_support: np.ndarray,
    target_support: np.ndarray,
    factor: float,
) -> float:
    """Official object scale times sqrt(2), then the usual few-shot area factor."""

    if isinstance(factor, bool) or not isinstance(factor, int | float):
        raise ValueError("factor must be a positive finite number")
    value = float(factor)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("factor must be a positive finite number")
    return object_scale(source_support, target_support) * math.sqrt(PAIR_INSTANCE_COUNT) * value


def place_pair_outside(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    *,
    seed: int,
    fewshot_areas: tuple[int, ...] | list[int],
    min_shape_pixels: int = 64,
) -> PairOutsidePlacement:
    """Place a two-instance template so one side meets the target and the other hangs out."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(min_shape_pixels) is not int or min_shape_pixels < 1:
        raise ValueError("min_shape_pixels must be a positive integer")
    gt = np.asarray(source_gt, dtype=bool)
    source = np.asarray(source_support, dtype=bool)
    target = np.asarray(target_support, dtype=bool)
    if gt.ndim != 2 or gt.shape[0] != gt.shape[1] or gt.shape != source.shape or gt.shape != target.shape:
        raise ValueError("source GT, source support, and target support must share one square grid")
    if not gt.any():
        raise EmptyFeasiblePlacement("source GT is empty")
    if not target.any():
        raise EmptyFeasiblePlacement("target TokenCut is empty")
    rng = np.random.RandomState(seed)
    factor = sample_area_factor(int(gt.sum()), fewshot_areas, seed=seed, rng=rng)
    base = object_scale(source, target)
    scale = base * math.sqrt(PAIR_INSTANCE_COUNT) * factor
    canvas = int(gt.shape[0])
    try:
        scaled = scale_binary_mask(gt, scale, size=canvas)
    except EmptyFeasiblePlacement as exc:
        raise EmptyFeasiblePlacement("pair-scaled GT does not fit the canvas") from exc
    hits: list[tuple[int, NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]] = []
    for angle in ANGLES_DEG:
        try:
            template = rotate_binary_mask(scaled, angle)
            placed = place_shape_max_overlap_clip(
                template,
                target,
                seed=seed,
                min_pixels=min_shape_pixels,
            )
        except EmptyFeasiblePlacement:
            continue
        unclipped = placed.unclipped
        if unclipped is None:
            continue
        mask = np.ascontiguousarray(unclipped, dtype=bool)
        if not mask.any() or mask.all():
            continue
        on_object = np.ascontiguousarray(mask & target, dtype=bool)
        outside = np.ascontiguousarray(mask & ~target, dtype=bool)
        if int(on_object.sum()) < min_shape_pixels or not outside.any():
            continue
        hits.append((int(angle), mask, on_object, outside))
    if not hits:
        raise EmptyFeasiblePlacement("pair template did not hang outside the target object")
    angle, mask, on_object, outside = hits[int(rng.randint(0, len(hits)))]
    return PairOutsidePlacement(
        mask=mask,
        on_object=on_object,
        outside=outside,
        scale=float(scale),
        factor=float(factor),
        object_scale=float(base),
        angle_deg=int(angle),
    )


__all__ = [
    "PAIR_INSTANCE_COUNT",
    "PairOutsidePlacement",
    "pair_instance_scale",
    "place_pair_outside",
]

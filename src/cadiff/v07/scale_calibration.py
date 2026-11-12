"""Scale-calibrated defect placement (v08 component).

The baseline scales a few-shot mask by the object-size ratio and then forces it
inside the A2 proposal; large or structural defects get shrunk to 1/3-1/5 of
their real area (measured: transistor/misplaced 40.4% of image -> 7.8%).

This module instead:

1. samples the defect area ratio relative to the object support from the
   few-shot masks (``area_i / source_support_area``);
2. scales the source mask so that ``scaled_area == ratio * target_support_area``;
3. places it in the proposal first, and, when it does not fit, in the whole
   object support (proposal is only a material-similarity region, large defects
   legitimately cross it);
4. shrinks only as a last resort, and records ``used_site_fallback`` when the
   placement had to leave the proposal.
"""

from __future__ import annotations

import numpy as np

from cadiff.v06_b.a2_component_place import (
    ANGLES_DEG,
    ComponentPlacement,
    EmptyFeasiblePlacement,
    place_shape_max_overlap_clip,
    rotate_binary_mask,
)
from cadiff.v06_b.a2_shape_instantiate import (
    place_shape_inside_proposal,
    scale_binary_mask,
)

CANVAS = 512


def _place_in_region(
    template_source: np.ndarray,
    region: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    min_shape_pixels: int,
    weighted: bool,
    rng: np.random.RandomState,
) -> tuple[int, np.ndarray, object] | None:
    angles = list(ANGLES_DEG)
    rng.shuffle(angles)
    for angle in angles:
        try:
            template = rotate_binary_mask(template_source, angle)
            placed = place_shape_inside_proposal(
                template,
                region,
                weights,
                seed=seed,
                weight_by_similarity=weighted,
            )
        except EmptyFeasiblePlacement:
            continue
        if int(placed.mask.sum()) < min_shape_pixels:
            continue
        return int(angle), np.ascontiguousarray(template), placed
    return None


def instantiate_scale_calibrated(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    fewshot_areas: tuple[int, ...] | list[int],
    min_shape_pixels: int = 64,
    location_prior: object | None = None,
    defect_type: str | None = None,
    target_valid: np.ndarray | None = None,
    max_shrink_steps: int = 5,
) -> tuple[np.ndarray, ComponentPlacement]:
    valid = (
        np.ascontiguousarray(target_valid, dtype=bool)
        if target_valid is not None
        else np.ones_like(target_support, dtype=bool)
    )
    support_mask = np.ascontiguousarray(
        np.asarray(target_support, dtype=bool) & valid, dtype=bool
    )
    proposal_mask = np.ascontiguousarray(
        np.asarray(proposal, dtype=bool) & support_mask, dtype=bool
    )
    if not support_mask.any():
        raise EmptyFeasiblePlacement("empty target support")
    gt = np.asarray(source_gt, dtype=bool)
    gt_area = int(gt.sum())
    if gt_area == 0:
        raise EmptyFeasiblePlacement("empty source mask")
    source_support_area = max(int(np.asarray(source_support, dtype=bool).sum()), 1)
    target_support_area = max(int(support_mask.sum()), 1)

    rng = np.random.RandomState(seed)
    ratios = [float(area) / float(source_support_area) for area in fewshot_areas]
    ratios = [value for value in ratios if value > 0]
    ratio = float(ratios[int(rng.randint(0, len(ratios)))]) if ratios else gt_area / source_support_area
    target_area = ratio * target_support_area
    upper = 0.92 * target_support_area
    target_area = float(np.clip(target_area, float(min_shape_pixels), max(upper, float(min_shape_pixels))))
    scale = float(np.sqrt(target_area / float(gt_area)))

    if location_prior is not None:
        weights = np.asarray(
            location_prior.heatmap(support_mask, defect_type), dtype=np.float32
        )
        weights = np.maximum(weights, 0.0)
        weighted = True
    else:
        weights = np.asarray(similarity, dtype=np.float32)
        weighted = False

    regions = [("proposal", proposal_mask), ("support", support_mask)]
    for step in range(max_shrink_steps + 1):
        candidate_scale = scale * (0.9**step)
        try:
            scaled = scale_binary_mask(gt, candidate_scale, size=CANVAS)
        except EmptyFeasiblePlacement:
            continue
        if scaled is None or not scaled.any():
            continue
        for name, region in regions:
            if not region.any():
                continue
            hit = _place_in_region(
                scaled,
                region,
                weights,
                seed=seed,
                min_shape_pixels=min_shape_pixels,
                weighted=weighted,
                rng=rng,
            )
            if hit is None:
                continue
            angle, template, placed = hit
            return template, ComponentPlacement(
                mask=np.ascontiguousarray(placed.mask),
                site=np.ascontiguousarray(region),
                site_index=0,
                scale=float(candidate_scale),
                angle_deg=int(angle),
                top=placed.top,
                left=placed.left,
                score=placed.score,
                feasible_count=placed.feasible_count,
                component_count=1,
                source_gt_pixels=gt_area,
                placed_pixels=int(placed.mask.sum()),
                used_clip=False,
                used_site_fallback=(name == "support"),
                factor=float(ratio),
                extra=np.zeros_like(placed.mask),
                unclipped=np.ascontiguousarray(placed.mask),
            )

    # last resort: clip inside the support
    try:
        scaled = scale_binary_mask(gt, scale, size=CANVAS)
    except EmptyFeasiblePlacement:
        scaled = gt
    for angle in list(ANGLES_DEG):
        try:
            template = rotate_binary_mask(scaled, angle)
            placed = place_shape_max_overlap_clip(
                template,
                support_mask,
                seed=seed,
                min_pixels=min_shape_pixels,
            )
        except EmptyFeasiblePlacement:
            continue
        return template, ComponentPlacement(
            mask=np.ascontiguousarray(placed.mask),
            site=np.ascontiguousarray(support_mask),
            site_index=0,
            scale=float(scale),
            angle_deg=int(angle),
            top=placed.top,
            left=placed.left,
            score=placed.score,
            feasible_count=placed.feasible_count,
            component_count=1,
            source_gt_pixels=gt_area,
            placed_pixels=int(placed.mask.sum()),
            used_clip=True,
            used_site_fallback=True,
            factor=float(ratio),
        )
    raise EmptyFeasiblePlacement("scale-calibrated placement failed")


__all__ = ["instantiate_scale_calibrated"]

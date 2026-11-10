"""Placement with proposal-component sites and an optional location prior.

This is the L0/L1 counterpart of ``instantiate_site_scale``: the site comes
from the proposal's connected components (not the object support's), and a
location prior can reweight the choice.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from cadiff.v06_b.a2_component_place import (
    ANGLES_DEG,
    ComponentPlacement,
    EmptyFeasiblePlacement,
    object_scale,
    place_shape_max_overlap_clip,
    rotate_binary_mask,
    sample_area_factor,
)
from cadiff.v06_b.a2_shape_instantiate import (
    place_shape_inside_proposal,
    scale_binary_mask,
)
from cadiff.v07.site_selection import proposal_sites, select_site


def _safe_scale(
    source_gt: np.ndarray,
    scale: float,
    site_mask: np.ndarray,
    *,
    canvas: int,
) -> np.ndarray:
    """Shrink until the scaled shape fits the selected site's bounding box."""

    site_rows, site_columns = np.nonzero(site_mask)
    if site_rows.size == 0:
        raise EmptyFeasiblePlacement("empty site")
    site_height = int(site_rows.max() - site_rows.min() + 1)
    site_width = int(site_columns.max() - site_columns.min() + 1)
    candidate = float(scale)
    for _ in range(8):
        try:
            scaled = scale_binary_mask(source_gt, candidate, size=canvas)
        except EmptyFeasiblePlacement:
            scaled = None
        if scaled is not None and scaled.any():
            rows, columns = np.nonzero(scaled)
            height = int(rows.max() - rows.min() + 1)
            width = int(columns.max() - columns.min() + 1)
            if height <= site_height and width <= site_width:
                return scaled
        candidate *= 0.8
    raise EmptyFeasiblePlacement("scaled GT does not fit the selected site")


def instantiate_site_scale_v07(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    fewshot_areas: tuple[int, ...] | list[int],
    min_shape_pixels: int = 64,
    site_method: str = "proposal_cc",
    location_prior: object | None = None,
    defect_type: str | None = None,
    site_temperature: float = 1.0,
    similarity_weight: float = 0.3,
    target_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, ComponentPlacement]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if site_method not in {"proposal_cc", "proposal_cc_prior"}:
        raise ValueError("site_method must be proposal_cc or proposal_cc_prior")
    valid = (
        np.ascontiguousarray(target_valid, dtype=bool)
        if target_valid is not None
        else np.ones_like(target_support, dtype=bool)
    )
    sites = proposal_sites(
        proposal,
        target_support,
        valid,
        similarity=similarity,
        min_pixels=min_shape_pixels,
    )
    if not sites:
        raise EmptyFeasiblePlacement("no proposal component above the area floor")
    prior_scores = None
    if site_method == "proposal_cc_prior" and location_prior is not None:
        prior_scores = [
            location_prior.score_site(site, target_support, defect_type)
            for site in sites
        ]
    site = select_site(
        sites,
        seed=seed,
        similarity_weight=similarity_weight,
        prior_scores=prior_scores,
        temperature=site_temperature,
    )
    gt = np.asarray(source_gt, dtype=bool)
    rng = np.random.RandomState(seed)
    factor = sample_area_factor(int(gt.sum()), fewshot_areas, seed=seed, rng=rng)
    scale = object_scale(source_support, target_support) * factor
    canvas = int(gt.shape[0])
    scaled = _safe_scale(gt, scale, site.mask, canvas=canvas)

    contained: list[tuple[int, np.ndarray, ComponentPlacement]] = []
    for angle in ANGLES_DEG:
        try:
            template = rotate_binary_mask(scaled, angle)
            placed = place_shape_inside_proposal(
                template,
                site.mask,
                similarity,
                seed=seed,
                weight_by_similarity=False,
            )
        except EmptyFeasiblePlacement:
            continue
        extra = np.zeros_like(placed.mask)
        contained.append(
            (
                int(angle),
                np.ascontiguousarray(template),
                ComponentPlacement(
                    mask=np.ascontiguousarray(placed.mask),
                    site=site.mask,
                    site_index=site.index,
                    scale=float(scale),
                    angle_deg=int(angle),
                    top=placed.top,
                    left=placed.left,
                    score=placed.score,
                    feasible_count=placed.feasible_count,
                    component_count=len(sites),
                    source_gt_pixels=int(gt.sum()),
                    placed_pixels=int(placed.mask.sum()),
                    used_clip=False,
                    factor=float(factor),
                    extra=extra,
                    unclipped=np.ascontiguousarray(placed.mask),
                ),
            )
        )
    contained = [
        item for item in contained if item[2].placed_pixels >= min_shape_pixels
    ]
    if contained:
        _angle, template, placed = contained[int(rng.randint(0, len(contained)))]
        if (placed.mask & ~site.mask).any() or (placed.mask & ~proposal).any():
            raise RuntimeError("site-scale placement escaped the proposal site")
        return template, placed

    clipped_hits: list[tuple[int, np.ndarray, ComponentPlacement]] = []
    for angle in ANGLES_DEG:
        try:
            template = rotate_binary_mask(scaled, angle)
            placed = place_shape_max_overlap_clip(
                template,
                site.mask,
                seed=seed,
                min_pixels=min_shape_pixels,
            )
        except EmptyFeasiblePlacement:
            continue
        clipped_hits.append((int(angle), np.ascontiguousarray(template), placed))
    if not clipped_hits:
        raise EmptyFeasiblePlacement("clip remainder is below the area floor")
    angle, template, placed = clipped_hits[int(rng.randint(0, len(clipped_hits)))]
    if (placed.mask & ~site.mask).any() or (placed.mask & ~proposal).any():
        raise RuntimeError("site-scale placement escaped the proposal site")
    return template, replace(
        placed,
        site=site.mask,
        site_index=site.index,
        scale=float(scale),
        factor=float(factor),
        angle_deg=angle,
        source_gt_pixels=int(gt.sum()),
        component_count=len(sites),
    )


__all__ = ["instantiate_site_scale_v07"]

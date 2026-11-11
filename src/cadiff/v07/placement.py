"""Placement with proposal-component sites and an optional location prior.

This is the L0/L1 counterpart of ``instantiate_site_scale``: the site comes
from the proposal's connected components (not the object support's), and a
location prior can reweight the choice.

Candidate sites are tried in score order. A site that cannot host the scaled
shape is skipped; when no proposal component fits, the whole proposal is used
and the placement is flagged with ``used_site_fallback``.
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


def _place_in_site(
    site_mask: np.ndarray,
    site_index: int,
    *,
    gt: np.ndarray,
    scale: float,
    factor: float,
    similarity: np.ndarray,
    seed: int,
    min_shape_pixels: int,
    component_count: int,
    rng: np.random.RandomState,
    fallback: bool,
) -> tuple[np.ndarray, ComponentPlacement] | None:
    canvas = int(gt.shape[0])
    try:
        scaled = _safe_scale(gt, scale, site_mask, canvas=canvas)
    except EmptyFeasiblePlacement:
        return None

    contained: list[tuple[int, np.ndarray, ComponentPlacement]] = []
    for angle in ANGLES_DEG:
        try:
            template = rotate_binary_mask(scaled, angle)
            placed = place_shape_inside_proposal(
                template,
                site_mask,
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
                    site=site_mask,
                    site_index=site_index,
                    scale=float(scale),
                    angle_deg=int(angle),
                    top=placed.top,
                    left=placed.left,
                    score=placed.score,
                    feasible_count=placed.feasible_count,
                    component_count=component_count,
                    source_gt_pixels=int(gt.sum()),
                    placed_pixels=int(placed.mask.sum()),
                    used_clip=False,
                    used_site_fallback=fallback,
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
        if (placed.mask & ~site_mask).any():
            raise RuntimeError("site-scale placement escaped the proposal site")
        return template, placed

    clipped_hits: list[tuple[int, np.ndarray, ComponentPlacement]] = []
    for angle in ANGLES_DEG:
        try:
            template = rotate_binary_mask(scaled, angle)
            placed = place_shape_max_overlap_clip(
                template,
                site_mask,
                seed=seed,
                min_pixels=min_shape_pixels,
            )
        except EmptyFeasiblePlacement:
            continue
        clipped_hits.append((int(angle), np.ascontiguousarray(template), placed))
    if not clipped_hits:
        return None
    angle, template, placed = clipped_hits[int(rng.randint(0, len(clipped_hits)))]
    if (placed.mask & ~site_mask).any():
        raise RuntimeError("site-scale placement escaped the proposal site")
    return template, replace(
        placed,
        site=site_mask,
        site_index=site_index,
        scale=float(scale),
        factor=float(factor),
        angle_deg=angle,
        source_gt_pixels=int(gt.sum()),
        component_count=component_count,
        used_site_fallback=fallback,
    )


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

    scores = [float(item.mean_similarity) * similarity_weight for item in sites]
    if prior_scores is not None:
        scores = [value + float(prior) for value, prior in zip(scores, prior_scores)]
    order = sorted(range(len(sites)), key=lambda index: scores[index], reverse=True)
    attempts = [site] + [sites[index] for index in order if sites[index] is not site]

    for candidate in attempts:
        result = _place_in_site(
            candidate.mask,
            candidate.index,
            gt=gt,
            scale=scale,
            factor=factor,
            similarity=similarity,
            seed=seed,
            min_shape_pixels=min_shape_pixels,
            component_count=len(sites),
            rng=rng,
            fallback=False,
        )
        if result is not None:
            return result

    proposal_mask = np.ascontiguousarray(np.asarray(proposal, dtype=bool))
    result = _place_in_site(
        proposal_mask,
        site.index,
        gt=gt,
        scale=scale,
        factor=factor,
        similarity=similarity,
        seed=seed,
        min_shape_pixels=min_shape_pixels,
        component_count=len(sites),
        rng=rng,
        fallback=True,
    )
    if result is None:
        raise EmptyFeasiblePlacement("no feasible site for the scaled shape")
    return result


def instantiate_site_scale_location_weighted(
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
) -> tuple[np.ndarray, ComponentPlacement]:
    """L1: place the scaled shape inside the whole proposal, with the
    translation distribution weighted by the object-frame defect-position
    histogram. The shape itself keeps the few-shot size prior."""

    if location_prior is None:
        raise ValueError("location_weighted placement requires a location prior")
    valid = (
        np.ascontiguousarray(target_valid, dtype=bool)
        if target_valid is not None
        else np.ones_like(target_support, dtype=bool)
    )
    proposal_mask = np.ascontiguousarray(
        np.asarray(proposal, dtype=bool) & np.asarray(target_support, dtype=bool) & valid,
        dtype=bool,
    )
    if not proposal_mask.any():
        raise EmptyFeasiblePlacement("empty proposal")
    gt = np.asarray(source_gt, dtype=bool)
    rng = np.random.RandomState(seed)
    factor = sample_area_factor(int(gt.sum()), fewshot_areas, seed=seed, rng=rng)
    scale = object_scale(source_support, target_support) * factor
    canvas = int(gt.shape[0])

    weights = np.asarray(
        location_prior.heatmap(target_support, defect_type), dtype=np.float32
    )
    weights = np.maximum(weights, 0.0)
    if not weights.any():
        weights = np.ones_like(weights)

    candidate_scale = float(scale)
    last_error: Exception | None = None
    for _attempt in range(7):
        try:
            scaled = _safe_scale(gt, candidate_scale, proposal_mask, canvas=canvas)
        except EmptyFeasiblePlacement as error:
            last_error = error
            candidate_scale *= 0.8
            continue
        angles = list(ANGLES_DEG)
        rng.shuffle(angles)
        for angle in angles:
            try:
                template = rotate_binary_mask(scaled, angle)
                placed = place_shape_inside_proposal(
                    template,
                    proposal_mask,
                    weights,
                    seed=seed,
                    weight_by_similarity=True,
                )
            except EmptyFeasiblePlacement as error:
                last_error = error
                continue
            if int(placed.mask.sum()) < min_shape_pixels:
                continue
            return template, ComponentPlacement(
                mask=np.ascontiguousarray(placed.mask),
                site=proposal_mask,
                site_index=0,
                scale=float(candidate_scale),
                angle_deg=int(angle),
                top=placed.top,
                left=placed.left,
                score=placed.score,
                feasible_count=placed.feasible_count,
                component_count=1,
                source_gt_pixels=int(gt.sum()),
                placed_pixels=int(placed.mask.sum()),
                used_clip=False,
                factor=float(factor),
                extra=np.zeros_like(placed.mask),
                unclipped=np.ascontiguousarray(placed.mask),
            )
        candidate_scale *= 0.8
    raise EmptyFeasiblePlacement(f"location-weighted placement failed: {last_error}")


__all__ = ["instantiate_site_scale_location_weighted", "instantiate_site_scale_v07"]

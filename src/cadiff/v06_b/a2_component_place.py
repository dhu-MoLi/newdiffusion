"""Official H (2026-09-18): TokenCut-CC site sample + log-area scale + uniform translation.

Historical instantiate_on_component (largest a2 CC, shrink grid, similarity-weighted
translation, extra-shrink to 0.01) remains for landed diagnostics. Official generate()
calls instantiate_site_scale and frozen R; it must not call the learned planner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import label
from scipy.signal import correlate

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import _material_proposal_maps, load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, _estimate_support, estimate_normal_structure
from cadiff.v06_atlas_diag.cable_a import _lookup_p1_target, build_cable_a_slots
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source
from cadiff.v06_b.a2_shape_instantiate import (
    EmptyFeasiblePlacement,
    _contained_output,
    _fail_rgb,
    _heatmap,
    _mask_u8,
    _overlay,
    _read_optional_png,
    _require_frozen_scheme2_renderer,
    _save_sheet,
    object_scale,
    place_shape_inside_proposal,
    scale_binary_mask,
)
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.cable_generate import load_cable_generate_config, load_cable_runtime

_EIGHT = np.ones((3, 3), dtype=np.uint8)
_CC_COLORS = (
    (220, 70, 70),
    (50, 170, 80),
    (50, 110, 220),
    (220, 180, 40),
    (180, 70, 180),
    (40, 190, 190),
    (220, 120, 40),
    (130, 90, 210),
)
SCALE_FACTORS = (1.0, 0.85, 0.7, 0.55, 0.4)
ANGLES_DEG = tuple(range(0, 180, 30))
FALLBACK_SCALE_FACTORS = (1.0, 0.85, 0.7, 0.55, 0.4, 0.3)
FALLBACK_ANGLES_DEG = tuple(range(0, 180, 15))
EXTRA_SCALE_FACTORS = (0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01)
MIN_COMPONENT_PIXELS = 64
MIN_SHAPE_PIXELS = 64
PREFERRED_POSE_GRIDS = (
    (SCALE_FACTORS, ANGLES_DEG, False),
    (FALLBACK_SCALE_FACTORS, FALLBACK_ANGLES_DEG, True),
)
RESCUE_POSE_GRIDS = ((EXTRA_SCALE_FACTORS, FALLBACK_ANGLES_DEG, False),)

FLOW_COLUMNS = (
    "source",
    "source_gt",
    "inpaint",
    "shape",
    "normal",
    "a2_prop",
    "a2_cc",
    "site",
    "placed",
    "overlay",
    "pixel_H",
    "raw_r",
    "final_y",
)
COMPARE_COLUMNS = (
    "a2_prop",
    "a2_cc",
    "site",
    "pixel_H",
    "placed",
    "final_y",
)
COLUMN_MEANINGS = {
    "source": "few-shot RGB，红叠加为官方 GT",
    "source_gt": "few-shot 官方 GT",
    "inpaint": "最近邻填洞后的源图（a2 用）",
    "shape": "本行采用的缩放+旋转 GT 模板",
    "normal": "目标正常图 x",
    "a2_prop": "正式 a2 整张材料区",
    "a2_cc": "a2 的 8 连通部件着色（本诊断的切开结果）",
    "site": "最大 a2 连通部件，即本次允许区",
    "placed": "放进该部件的 T(g)；额外缩小保底仍是 GT 形状，不等于 site",
    "overlay": "蓝=整张 a2，绿=最大部件，黄=placed，品红=额外缩小保底",
    "pixel_H": "像素 a2→H 的 m_final（对照）",
    "raw_r": "冻结方案 2 R",
    "final_y": "因果合成 y",
}


@dataclass(frozen=True)
class ComponentSite:
    index: int
    mask: NDArray[np.bool_]
    pixels: int
    mean_similarity: float


@dataclass(frozen=True)
class ComponentPlacement:
    mask: NDArray[np.bool_]
    site: NDArray[np.bool_]
    site_index: int
    scale: float
    angle_deg: int
    top: int
    left: int
    score: float
    feasible_count: int
    component_count: int
    source_gt_pixels: int
    placed_pixels: int
    used_site_fallback: bool = False
    used_proposal_floor: bool = False
    used_clip: bool = False
    factor: float = 1.0
    extra: NDArray[np.bool_] | None = None
    unclipped: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class TokenCutSite:
    index: int
    mask: NDArray[np.bool_]
    pixels: int


@dataclass(frozen=True)
class PoseCandidate:
    template: NDArray[np.bool_]
    scale: float
    angle_deg: int
    factor: float
    used_site_fallback: bool


def rotate_binary_mask(mask: np.ndarray, degrees: float) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1]:
        raise ValueError("mask must be a square canvas")
    if not np.isfinite(degrees):
        raise ValueError("degrees must be finite")
    if abs(degrees) % 360 < 1e-6:
        return np.ascontiguousarray(binary.copy())
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        raise EmptyFeasiblePlacement("cannot rotate an empty mask")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = binary[top:bottom, left:right]
    rotated = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255).rotate(
            -float(degrees),
            resample=Image.Resampling.NEAREST,
            expand=True,
        ),
        dtype=np.uint8,
    ) > 127
    if not rotated.any():
        raise EmptyFeasiblePlacement("rotated GT vanished")
    size = binary.shape[0]
    if rotated.shape[0] >= size or rotated.shape[1] >= size:
        raise EmptyFeasiblePlacement("rotated GT does not fit the canvas")
    canvas = np.zeros_like(binary, dtype=bool)
    center_row = 0.5 * (top + bottom - 1)
    center_col = 0.5 * (left + right - 1)
    paste_top = int(round(center_row - 0.5 * (rotated.shape[0] - 1)))
    paste_left = int(round(center_col - 0.5 * (rotated.shape[1] - 1)))
    paste_top = min(max(paste_top, 0), size - rotated.shape[0])
    paste_left = min(max(paste_left, 0), size - rotated.shape[1])
    canvas[
        paste_top : paste_top + rotated.shape[0],
        paste_left : paste_left + rotated.shape[1],
    ] = rotated
    return canvas


def label_a2_components(
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    min_pixels: int = MIN_COMPONENT_PIXELS,
) -> tuple[NDArray[np.int32], tuple[ComponentSite, ...]]:
    allowed = np.asarray(proposal, dtype=bool)
    scores = np.asarray(similarity, dtype=np.float32)
    if allowed.shape != scores.shape:
        raise ValueError("proposal and similarity must share one grid")
    labels, count = label(allowed, structure=_EIGHT)
    sites: list[ComponentSite] = []
    kept = np.zeros(allowed.shape, dtype=np.int32)
    next_index = 0
    for raw_id in range(1, count + 1):
        mask = labels == raw_id
        pixels = int(mask.sum())
        if pixels < min_pixels:
            continue
        next_index += 1
        kept[mask] = next_index
        sites.append(
            ComponentSite(
                index=next_index,
                mask=np.ascontiguousarray(mask, dtype=bool),
                pixels=pixels,
                mean_similarity=float(scores[mask].mean()) if pixels else 0.0,
            )
        )
    return kept, tuple(sites)


def color_components(labels: np.ndarray) -> np.ndarray:
    field = np.asarray(labels)
    image = np.zeros((*field.shape, 3), dtype=np.uint8)
    for index in range(1, int(field.max()) + 1):
        image[field == index] = _CC_COLORS[(index - 1) % len(_CC_COLORS)]
    return image


def has_feasible_placement(shape: np.ndarray, allowed: np.ndarray) -> bool:
    template = np.asarray(shape, dtype=bool)
    region = np.asarray(allowed, dtype=bool)
    if template.shape != region.shape:
        raise ValueError("shape and allowed must share one grid")
    rows, cols = np.nonzero(template)
    if rows.size == 0:
        return False
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = np.ascontiguousarray(template[top:bottom, left:right], dtype=np.float64)
    height, width = crop.shape
    cc_rows, cc_cols = np.nonzero(region)
    if cc_rows.size == 0:
        return False
    r0, r1 = int(cc_rows.min()), int(cc_rows.max()) + 1
    c0, c1 = int(cc_cols.min()), int(cc_cols.max()) + 1
    if height > (r1 - r0) or width > (c1 - c0):
        return False
    outside = np.ascontiguousarray(~region[r0:r1, c0:c1], dtype=np.float64)
    overlap = correlate(outside, crop, mode="valid", method="fft")
    return bool((overlap < 0.5).any())


def pick_largest_site(sites: tuple[ComponentSite, ...]) -> ComponentSite:
    if not sites:
        raise EmptyFeasiblePlacement("a2 has no component above the area floor")
    return max(sites, key=lambda site: (site.pixels, site.mean_similarity, -site.index))


def sample_area_factor(
    src_area: int | float,
    fewshot_areas: tuple[int, ...] | list[int],
    *,
    seed: int,
    rng: np.random.RandomState | None = None,
) -> float:
    source = float(src_area)
    if source <= 0:
        raise ValueError("source area must be positive")
    areas = [float(area) for area in fewshot_areas if float(area) > 0]
    if not areas:
        raise ValueError("few-shot areas are empty")
    lo = min(areas)
    hi = max(areas)
    engine = rng if rng is not None else np.random.RandomState(seed)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= 0:
        raise ValueError("few-shot areas must be positive and finite")
    if lo == hi:
        target = lo
    else:
        target = float(np.exp(engine.uniform(np.log(lo), np.log(hi))))
    return float(np.sqrt(target / source))


def candidate_tokencut_sites(
    object_mask: np.ndarray,
    proposal: np.ndarray,
    *,
    min_pixels: int = MIN_SHAPE_PIXELS,
) -> tuple[TokenCutSite, ...]:
    obj = np.asarray(object_mask, dtype=bool)
    prop = np.asarray(proposal, dtype=bool)
    if obj.shape != prop.shape:
        raise ValueError("object and proposal must share one grid")
    if type(min_pixels) is not int or min_pixels < 1:
        raise ValueError("min_pixels must be a positive integer")
    if not obj.any():
        raise EmptyFeasiblePlacement("no TokenCut site with enough proposal")
    labels, count = label(obj, structure=_EIGHT)
    ranked: list[tuple[int, int, NDArray[np.bool_]]] = []
    for raw_id in range(1, count + 1):
        mask = labels == raw_id
        ranked.append((int(mask.sum()), int(np.argmax(mask)), np.ascontiguousarray(mask)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    sites: list[TokenCutSite] = []
    for index, (_pixels, _arg, cc_mask) in enumerate(ranked, start=1):
        mask = np.ascontiguousarray(prop & cc_mask)
        pixels = int(mask.sum())
        if pixels >= min_pixels:
            sites.append(TokenCutSite(index=index, mask=mask, pixels=pixels))
    if not sites:
        raise EmptyFeasiblePlacement("no TokenCut site with enough proposal")
    return tuple(sites)


def pick_tokencut_site(sites: tuple[TokenCutSite, ...] | list[TokenCutSite], *, seed: int) -> TokenCutSite:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    chosen = tuple(sites)
    if not chosen:
        raise EmptyFeasiblePlacement("no TokenCut site with enough proposal")
    rng = np.random.RandomState(seed)
    return chosen[int(rng.randint(0, len(chosen)))]


def place_shape_max_overlap_clip(
    shape: np.ndarray,
    site: np.ndarray,
    *,
    seed: int,
    min_pixels: int,
) -> ComponentPlacement:
    template = np.asarray(shape, dtype=bool)
    allowed = np.asarray(site, dtype=bool)
    if template.shape != allowed.shape:
        raise ValueError("shape and site must share one grid")
    rows, cols = np.nonzero(template)
    if rows.size == 0:
        raise EmptyFeasiblePlacement("scaled GT is empty")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = np.ascontiguousarray(template[top:bottom, left:right], dtype=np.float64)
    height, width = crop.shape
    if height > allowed.shape[0] or width > allowed.shape[1]:
        raise EmptyFeasiblePlacement("scaled GT is larger than the canvas")
    inside = correlate(allowed.astype(np.float64), crop, mode="valid", method="fft")
    peak = float(inside.max()) if inside.size else 0.0
    if peak < float(min_pixels):
        raise EmptyFeasiblePlacement("clip remainder is below the area floor")
    ys, xs = np.nonzero(inside >= peak - 1e-6)
    rng = np.random.RandomState(seed)
    index = int(rng.choice(ys.size))
    paste_top = int(ys[index])
    paste_left = int(xs[index])
    unclipped = np.zeros_like(template, dtype=bool)
    unclipped[paste_top : paste_top + height, paste_left : paste_left + width] = crop > 0.5
    clipped = np.ascontiguousarray(unclipped & allowed)
    extra = np.ascontiguousarray(unclipped & ~allowed)
    if int(clipped.sum()) < min_pixels:
        raise EmptyFeasiblePlacement("clip remainder is below the area floor")
    return ComponentPlacement(
        mask=clipped,
        site=np.ascontiguousarray(allowed),
        site_index=-1,
        scale=1.0,
        angle_deg=0,
        top=paste_top,
        left=paste_left,
        score=peak,
        feasible_count=int(ys.size),
        component_count=1,
        source_gt_pixels=int(template.sum()),
        placed_pixels=int(clipped.sum()),
        used_clip=True,
        extra=extra,
        unclipped=unclipped,
    )


def instantiate_site_scale(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    fewshot_areas: tuple[int, ...] | list[int],
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
) -> tuple[NDArray[np.bool_], ComponentPlacement]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(min_shape_pixels) is not int or min_shape_pixels < 1:
        raise ValueError("min_shape_pixels must be a positive integer")
    sites = candidate_tokencut_sites(target_support, proposal, min_pixels=min_shape_pixels)
    site = pick_tokencut_site(sites, seed=seed)
    gt = np.asarray(source_gt, dtype=bool)
    rng = np.random.RandomState(seed)
    factor = sample_area_factor(int(gt.sum()), fewshot_areas, seed=seed, rng=rng)
    scale = object_scale(source_support, target_support) * factor
    canvas_size = int(gt.shape[0])
    try:
        scaled = scale_binary_mask(gt, scale, size=canvas_size)
    except EmptyFeasiblePlacement as exc:
        raise EmptyFeasiblePlacement("scaled GT does not fit the canvas") from exc
    contained: list[tuple[int, NDArray[np.bool_], ComponentPlacement]] = []
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
    contained = [item for item in contained if item[2].placed_pixels >= min_shape_pixels]
    if contained:
        _angle, template, placed = contained[int(rng.randint(0, len(contained)))]
        if (placed.mask & ~site.mask).any() or (placed.mask & ~np.asarray(proposal, dtype=bool)).any():
            raise RuntimeError("site-scale placement escaped the chosen TokenCut site")
        return template, placed
    clipped_hits: list[tuple[int, NDArray[np.bool_], ComponentPlacement]] = []
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
    if (placed.mask & ~site.mask).any() or (placed.mask & ~np.asarray(proposal, dtype=bool)).any():
        raise RuntimeError("site-scale placement escaped the chosen TokenCut site")
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


def _placement_from_hit(
    *,
    template: np.ndarray,
    site: ComponentSite,
    similarity: np.ndarray,
    proposal: np.ndarray,
    seed: int,
    scale: float,
    angle: int,
    source_gt: np.ndarray,
    component_count: int,
    used_site_fallback: bool,
    used_proposal_floor: bool = False,
) -> ComponentPlacement:
    placed = place_shape_inside_proposal(template, site.mask, similarity, seed=seed)
    if (placed.mask & ~site.mask).any() or (placed.mask & ~np.asarray(proposal, dtype=bool)).any():
        raise RuntimeError("component placement escaped the chosen a2 CC")
    return ComponentPlacement(
        mask=placed.mask,
        site=site.mask,
        site_index=site.index,
        scale=scale,
        angle_deg=angle,
        top=placed.top,
        left=placed.left,
        score=placed.score,
        feasible_count=placed.feasible_count,
        component_count=component_count,
        source_gt_pixels=int(np.asarray(source_gt, dtype=bool).sum()),
        placed_pixels=int(placed.mask.sum()),
        used_site_fallback=used_site_fallback,
        used_proposal_floor=used_proposal_floor,
    )


def _proposal_floor_placement(
    *,
    site: ComponentSite,
    proposal: np.ndarray,
    source_gt: np.ndarray,
    component_count: int,
) -> ComponentPlacement:
    mask = np.ascontiguousarray(site.mask, dtype=bool)
    allowed = np.asarray(proposal, dtype=bool)
    if not mask.any():
        raise EmptyFeasiblePlacement("cannot use an empty a2 component as the floor")
    if (mask & ~allowed).any():
        raise RuntimeError("proposal floor escaped a2")
    return ComponentPlacement(
        mask=mask,
        site=mask,
        site_index=site.index,
        scale=1.0,
        angle_deg=0,
        top=0,
        left=0,
        score=0.0,
        feasible_count=0,
        component_count=component_count,
        source_gt_pixels=int(np.asarray(source_gt, dtype=bool).sum()),
        placed_pixels=int(mask.sum()),
        used_site_fallback=False,
        used_proposal_floor=True,
    )


def collect_feasible_poses(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    site: np.ndarray,
    *,
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
    grids: tuple[tuple[tuple[float, ...], tuple[int, ...], bool], ...] = PREFERRED_POSE_GRIDS,
) -> tuple[tuple[PoseCandidate, ...], bool]:
    """Return feasible (scale, angle) on the first non-empty grid."""

    gt = np.asarray(source_gt, dtype=bool)
    if gt.ndim != 2 or gt.shape[0] != gt.shape[1]:
        raise ValueError("source_gt must be a square canvas")
    canvas_size = int(gt.shape[0])
    base_scale = object_scale(source_support, target_support)
    allowed = np.asarray(site, dtype=bool)
    for factors, angles, used_site_fallback in grids:
        hits: list[PoseCandidate] = []
        for factor in factors:
            scale = base_scale * float(factor)
            try:
                scaled = scale_binary_mask(gt, scale, size=canvas_size)
            except EmptyFeasiblePlacement:
                continue
            if int(scaled.sum()) < min_shape_pixels:
                continue
            for angle in angles:
                try:
                    template = rotate_binary_mask(scaled, angle)
                except EmptyFeasiblePlacement:
                    continue
                if has_feasible_placement(template, allowed):
                    hits.append(
                        PoseCandidate(
                            template=np.ascontiguousarray(template, dtype=bool),
                            scale=float(scale),
                            angle_deg=int(angle),
                            factor=float(factor),
                            used_site_fallback=used_site_fallback,
                        )
                    )
        if hits:
            return tuple(hits), used_site_fallback
    return (), False


def _sample_candidate(
    candidates: tuple[PoseCandidate, ...],
    *,
    seed: int,
    site: ComponentSite,
    similarity: np.ndarray,
    proposal: np.ndarray,
    source_gt: np.ndarray,
    component_count: int,
    used_proposal_floor: bool,
) -> tuple[NDArray[np.bool_], ComponentPlacement]:
    rng = np.random.RandomState(seed)
    chosen = candidates[int(rng.randint(0, len(candidates)))]
    placement = _placement_from_hit(
        template=chosen.template,
        site=site,
        similarity=similarity,
        proposal=proposal,
        seed=seed,
        scale=chosen.scale,
        angle=chosen.angle_deg,
        source_gt=source_gt,
        component_count=component_count,
        used_site_fallback=chosen.used_site_fallback,
        used_proposal_floor=used_proposal_floor,
    )
    return chosen.template, placement


def instantiate_on_component_greedy(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
    allow_proposal_floor: bool = True,
) -> tuple[np.ndarray, ComponentPlacement]:
    """Historical first-hit placement. Official H no longer uses this path."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(min_shape_pixels) is not int or min_shape_pixels < 1:
        raise ValueError("min_shape_pixels must be a positive integer")
    if not isinstance(allow_proposal_floor, bool):
        raise ValueError("allow_proposal_floor must be boolean")
    _labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
    if not sites:
        raise EmptyFeasiblePlacement("a2 has no component")
    largest = pick_largest_site(sites)
    candidates, _used_site_fallback = collect_feasible_poses(
        source_gt,
        source_support,
        target_support,
        largest.mask,
        min_shape_pixels=min_shape_pixels,
        grids=PREFERRED_POSE_GRIDS,
    )
    if candidates:
        first = candidates[0]
        return first.template, _placement_from_hit(
            template=first.template,
            site=largest,
            similarity=similarity,
            proposal=proposal,
            seed=seed,
            scale=first.scale,
            angle=first.angle_deg,
            source_gt=source_gt,
            component_count=len(sites),
            used_site_fallback=first.used_site_fallback,
        )
    if not allow_proposal_floor:
        raise EmptyFeasiblePlacement("shape does not fit the largest a2 component")
    return np.ascontiguousarray(np.asarray(source_gt, dtype=bool)), _proposal_floor_placement(
        site=largest,
        proposal=proposal,
        source_gt=source_gt,
        component_count=len(sites),
    )


def instantiate_on_component(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
    allow_proposal_floor: bool = True,
) -> tuple[np.ndarray, ComponentPlacement]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(min_shape_pixels) is not int or min_shape_pixels < 1:
        raise ValueError("min_shape_pixels must be a positive integer")
    if not isinstance(allow_proposal_floor, bool):
        raise ValueError("allow_proposal_floor must be boolean")
    _labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
    if not sites:
        raise EmptyFeasiblePlacement("a2 has no component")
    largest = pick_largest_site(sites)
    candidates, _used_site_fallback = collect_feasible_poses(
        source_gt,
        source_support,
        target_support,
        largest.mask,
        min_shape_pixels=min_shape_pixels,
        grids=PREFERRED_POSE_GRIDS,
    )
    used_proposal_floor = False
    if not candidates and allow_proposal_floor:
        if min_shape_pixels > 1:
            candidates, _used_site_fallback = collect_feasible_poses(
                source_gt,
                source_support,
                target_support,
                largest.mask,
                min_shape_pixels=1,
                grids=PREFERRED_POSE_GRIDS,
            )
        if not candidates:
            candidates, _used_site_fallback = collect_feasible_poses(
                source_gt,
                source_support,
                target_support,
                largest.mask,
                min_shape_pixels=1,
                grids=RESCUE_POSE_GRIDS,
            )
            used_proposal_floor = bool(candidates)
    if not candidates:
        raise EmptyFeasiblePlacement("shape does not fit the largest a2 component")
    return _sample_candidate(
        candidates,
        seed=seed,
        site=largest,
        similarity=similarity,
        proposal=proposal,
        source_gt=source_gt,
        component_count=len(sites),
        used_proposal_floor=used_proposal_floor,
    )


def run_a2_component_place_diagnostic(
    *,
    repository_root: Path,
    dataset_root: Path,
    output: Path,
    a_experiment: Path,
    generate_config: Path,
    pixel_h_root: Path | None,
    device: str | None = None,
    tile: int = 160,
    renderer: Callable[[np.ndarray, str, np.ndarray, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    destination = _contained_output(output)
    experiment = load_v06_a_experiment(a_experiment)
    manifest = load_v06_a_identity_manifest(
        repository_root / experiment.identity_manifest,
        experiment=experiment,
    )
    slots = build_cable_a_slots(manifest, experiment=experiment)
    generator = load_a2_generator(
        experiment_path=a_experiment,
        dataset_root=dataset_root,
        device=device,
    )
    if renderer is None:
        _require_frozen_scheme2_renderer(generate_config, repository_root)
        runtime = load_cable_runtime(
            load_cable_generate_config(generate_config, repository_root=repository_root),
            repository_root,
            device,
        )
        renderer = runtime.module_b.renderer
    destination.mkdir(parents=True, exist_ok=False)
    identities = destination / "identities"
    identities.mkdir()
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    compare_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    per_type: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {
        name: [] for name in experiment.defect_types
    }
    records: list[dict[str, Any]] = []
    structures: dict[str, NormalStructure] = {}

    for slot in slots:
        identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        sample_root = identities / identity
        sample_root.mkdir()
        if slot.normal_id not in structures:
            target_record = _lookup_p1_target(manifest, slot.normal_id)
            loaded_target = load_identity_record(
                target_record,
                dataset_root=dataset_root,
                config=experiment,
            )
            structures[slot.normal_id] = estimate_normal_structure(
                loaded_target, config=experiment
            )
        target = structures[slot.normal_id]
        source = generator.select_source(slot.defect_type, slot.seed)
        assert source.anomaly_gt is not None
        inpainted = neighbor_fill_source(source.rgb, source.anomaly_gt, source.valid)
        inpainted_identity = replace(
            source,
            rgb=np.array(inpainted, copy=True),
            anomaly_gt=None,
            record=replace(source.record, role="normal_fit", defect_type="good"),
        )
        source_support = _estimate_support(inpainted_identity, experiment)
        source_feat = generator._encoder.encode(inpainted)
        target_feat = generator._encoder.encode(target.normal.rgb)
        proposal, tau, similarity = _material_proposal_maps(
            source.anomaly_gt,
            source.valid,
            target,
            source_feat=source_feat,
            target_feat=target_feat,
        )
        cc_labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
        preferred = pick_largest_site(sites) if sites else None
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        fail = _fail_rgb(normal.shape)
        empty_mask = np.zeros(proposal.shape, dtype=bool)
        preferred_mask = preferred.mask if preferred is not None else empty_mask
        arrays = {
            "source": _overlay(source.rgb, source.anomaly_gt, (255, 40, 40), 0.55),
            "source_gt": _mask_u8(source.anomaly_gt),
            "inpaint": np.ascontiguousarray(inpainted),
            "shape": _mask_u8(source.anomaly_gt),
            "normal": normal,
            "a2_prop": _mask_u8(proposal),
            "a2_cc": color_components(cc_labels),
            "site": _mask_u8(preferred_mask),
            "placed": _mask_u8(empty_mask),
            "overlay": _overlay(
                _overlay(normal, proposal, (80, 160, 255), 0.28),
                preferred_mask,
                (40, 200, 80),
                0.4,
            ),
            "pixel_H": fail.copy(),
            "raw_r": fail.copy(),
            "final_y": fail.copy(),
            "pixel_y": fail.copy(),
        }
        if pixel_h_root is not None:
            pixel_h = _read_optional_png(pixel_h_root / identity / "m_final.png")
            pixel_y = _read_optional_png(pixel_h_root / identity / "final_y.png")
            if pixel_h is not None:
                arrays["pixel_H"] = pixel_h
            if pixel_y is not None:
                arrays["pixel_y"] = pixel_y
        record: dict[str, Any] = {
            "identity": identity,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": source.record.item_id,
            "seed": slot.seed,
            "a2_tau": tau,
            "proposal_pixels": int(proposal.sum()),
            "component_count": len(sites),
            "source_gt_pixels": int(source.anomaly_gt.sum()),
            "failed": False,
            "failure_reason": None,
            "site_index": None if preferred is None else preferred.index,
            "site_pixels": 0 if preferred is None else preferred.pixels,
            "used_site_fallback": False,
            "used_proposal_floor": False,
            "scale": None,
            "angle_deg": None,
            "placed_pixels": 0,
            "feasible_count": 0,
            "placement_score": None,
            "m_final_subset_of_site": None,
            "m_final_subset_of_proposal": None,
            "outside_rgb_unchanged": None,
        }
        try:
            template, placement = instantiate_on_component(
                source.anomaly_gt,
                source_support,
                target.support & target.valid,
                proposal,
                similarity,
                seed=slot.seed,
            )
            validate_realization(placement.mask, placement.site)
            arrays["shape"] = _mask_u8(template)
            arrays["site"] = _mask_u8(placement.site)
            arrays["placed"] = _mask_u8(placement.mask)
            placed_color = (
                (220, 40, 180) if placement.used_proposal_floor else (255, 220, 0)
            )
            arrays["overlay"] = _overlay(
                _overlay(
                    _overlay(normal, proposal, (80, 160, 255), 0.28),
                    placement.site,
                    (40, 200, 80),
                    0.4,
                ),
                placement.mask,
                placed_color,
                0.75,
            )
            raw = renderer(normal, slot.defect_type, placement.mask.astype(np.uint8), slot.seed)
            final_y = compose_causal(normal, raw, placement.mask)
            in_site = bool(not (placement.mask & ~placement.site).any())
            in_prop = bool(not (placement.mask & ~proposal).any())
            unchanged = bool(np.array_equal(final_y[~placement.mask], normal[~placement.mask]))
            if not in_site or not in_prop or not unchanged:
                raise ValueError("placed shape violated the single-component contract")
            arrays["raw_r"] = np.ascontiguousarray(raw)
            arrays["final_y"] = np.ascontiguousarray(final_y)
            record.update(
                site_index=placement.site_index,
                site_pixels=int(placement.site.sum()),
                scale=placement.scale,
                angle_deg=placement.angle_deg,
                placed_pixels=placement.placed_pixels,
                feasible_count=placement.feasible_count,
                placement_score=placement.score,
                m_final_subset_of_site=in_site,
                m_final_subset_of_proposal=in_prop,
                outside_rgb_unchanged=unchanged,
                paste_top=placement.top,
                paste_left=placement.left,
                used_site_fallback=placement.used_site_fallback,
                used_proposal_floor=placement.used_proposal_floor,
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        for name, array in arrays.items():
            Image.fromarray(array).save(sample_root / f"{name}.png")
        write_json_atomic(sample_root / "record.json", record)
        records.append(record)
        if record["failed"]:
            status = "FAIL " + str(record["failure_reason"])
            label_text = identity + " FAIL"
        elif record.get("used_proposal_floor"):
            status = "OK FLOOR"
            label_text = identity + " FLOOR"
        elif record.get("used_site_fallback"):
            status = "OK FALLBACK"
            label_text = identity + " FALLBACK"
        else:
            status = "OK"
            label_text = identity
        flow = {name: arrays[name] for name in FLOW_COLUMNS}
        compare = {name: arrays[name] for name in COMPARE_COLUMNS}
        flow_rows.append((label_text, flow))
        compare_rows.append((label_text, compare))
        per_type[slot.defect_type].append((label_text, flow))
        print(identity, status, flush=True)

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    _save_sheet(destination / "compare_pixel_h.png", compare_rows, COMPARE_COLUMNS, tile=tile)
    for kind, rows in per_type.items():
        _save_sheet(destination / f"flow_{kind}.png", rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": "v06_b_a2_component_place_v2",
        "diagnostic_only": True,
        "not_learned": True,
        "not_official_pass": True,
        "one_variable": "sample_feasible_pose_then_gt_in_site_rescue",
        "identity_count": len(records),
        "failed_rows": sum(row["failed"] for row in records),
        "floor_rows": sum(bool(row.get("used_proposal_floor")) for row in records),
        "fallback_rows": sum(bool(row.get("used_site_fallback")) for row in records),
        "method": "fewshot_gt_sampled_pose_inside_largest_a2_cc",
        "renderer": "frozen_scheme2",
        "proposal_floor_is_last_resort": False,
        "floor_copies_site": False,
        "pose_selection": "sample_feasible_scale_angle",
        "empty_a2_still_fails": True,
        "no_projection": True,
        "no_square_dilate": True,
        "scale_factors": list(SCALE_FACTORS),
        "angles_deg": list(ANGLES_DEG),
        "fallback_scale_factors": list(FALLBACK_SCALE_FACTORS),
        "fallback_angles_deg": list(FALLBACK_ANGLES_DEG),
        "extra_scale_factors": list(EXTRA_SCALE_FACTORS),
        "min_component_pixels": MIN_COMPONENT_PIXELS,
        "flow_columns": list(FLOW_COLUMNS),
        "compare_columns": list(COMPARE_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "pixel_h_root": None if pixel_h_root is None else str(pixel_h_root),
        "destination": str(destination),
        "visible_defects_passed": None,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_a2_component_place_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    generate_config: Path,
    output: Path,
    cwd: Path,
    pixel_h_root: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_a2_component_place_diagnostic(
        repository_root=cwd,
        dataset_root=project.dataset_root,
        output=output,
        a_experiment=a_experiment if a_experiment.is_absolute() else cwd / a_experiment,
        generate_config=(
            generate_config if generate_config.is_absolute() else cwd / generate_config
        ),
        pixel_h_root=pixel_h_root,
        device=device,
    )


__all__ = [
    "COLUMN_MEANINGS",
    "COMPARE_COLUMNS",
    "ComponentPlacement",
    "EmptyFeasiblePlacement",
    "FLOW_COLUMNS",
    "PoseCandidate",
    "TokenCutSite",
    "collect_feasible_poses",
    "has_feasible_placement",
    "instantiate_on_component",
    "instantiate_on_component_greedy",
    "instantiate_site_scale",
    "label_a2_components",
    "pick_largest_site",
    "pick_tokencut_site",
    "place_shape_max_overlap_clip",
    "sample_area_factor",
    "candidate_tokencut_sites",
    "rotate_binary_mask",
    "run_a2_component_place_diagnostic",
    "run_a2_component_place_from_cli",
]

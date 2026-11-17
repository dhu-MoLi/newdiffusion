"""Allowed region (Omega) for target-conditioned mask synthesis.

Omega answers *where* a defect of a given type may appear on a given normal
image.  It fuses a type-level position prior (object frame) with a boundary
term, then cleans the result on a coarse grid so that the region is smooth and
made of at most a few connected components -- the A2 proposal alone is far too
fragmented to serve as a generation region.

See docs/v08_omega_design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANVAS = 512


@dataclass(frozen=True)
class OmegaConfig:
    grid: int = 64
    close_radius: int = 2
    open_radius: int = 1
    max_components: int = 3
    min_component_ratio: float = 0.01
    area_min: float = 0.25
    area_max: float = 0.75
    area_multiplier: float = 2.0
    boundary_weight: float = 0.6
    structural_floor: float = 0.35
    part_weight: float = 0.5
    part_sigma_px: float = 18.0
    boundary_sigma_px: float = 24.0
    structural_dilate_px: int = 8
    max_adjust: int = 3


def _quantile_threshold(score: np.ndarray, pool: np.ndarray, ratio: float) -> float:
    values = score[pool]
    if values.size == 0:
        return float(np.max(score))
    ratio = float(np.clip(ratio, 0.01, 0.99))
    return float(np.quantile(values, 1.0 - ratio))


def _cleanup(mask: np.ndarray, support: np.ndarray, cfg: OmegaConfig) -> np.ndarray:
    import cv2

    size = cfg.grid
    small = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA) > 0.5
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * cfg.close_radius + 1, 2 * cfg.close_radius + 1)
    )
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * cfg.open_radius + 1, 2 * cfg.open_radius + 1)
    )
    small = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel_open) > 0
    support_small = cv2.resize(
        support.astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA
    ) > 0.5
    floor = max(1, int(round(cfg.min_component_ratio * support_small.sum())))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(small.astype(np.uint8), 8)
    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)]
    areas.sort(reverse=True)
    keep = np.zeros_like(small)
    for area, index in areas[: cfg.max_components]:
        if area >= floor:
            keep |= labels == index
    return np.ascontiguousarray(keep)


def build_allowed_region(
    rgb: np.ndarray,
    support: np.ndarray,
    valid: np.ndarray,
    prior: object,
    defect_type: str,
    *,
    fewshot_area_ratios: list[float],
    sdf: np.ndarray,
    cfg: OmegaConfig = OmegaConfig(),
    structural: bool = False,
    texture_mode: bool = False,
    domain_dilate_px: int = 24,
    area_ratio_override: float | None = None,
    parts: list | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Return (omega_mask, score_512, stats).

    The region is *not* restricted to the TokenCut support: on texture classes
    the support covers almost none of the real defects (measured: carpet
    0.02-0.37 GT-in-support), so the domain is the dilated support, or the
    whole valid image in texture mode.
    """

    import cv2

    support = np.ascontiguousarray(np.asarray(support, dtype=bool) & np.asarray(valid, dtype=bool))
    if not support.any():
        support = np.ascontiguousarray(np.asarray(valid, dtype=bool))
    if texture_mode:
        domain = np.ascontiguousarray(np.asarray(valid, dtype=bool))
    else:
        kernel_domain = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * domain_dilate_px + 1,) * 2
        )
        domain = (cv2.dilate(support.astype(np.uint8), kernel_domain) > 0) & np.asarray(
            valid, dtype=bool
        )
    if not domain.any():
        return np.zeros((CANVAS, CANVAS), dtype=bool), np.zeros((CANVAS, CANVAS), np.float32), {}
    density = np.asarray(prior.heatmap(support, defect_type), dtype=np.float32)
    density = np.maximum(density, 0.0)
    if density.max() > 0:
        density = density / density.max()

    depth = np.clip(np.asarray(sdf, dtype=np.float32), 0.0, None)
    boundary = np.exp(-depth / max(cfg.boundary_sigma_px, 1.0))

    junction = np.zeros_like(boundary)
    if parts:
        from scipy.ndimage import binary_erosion, distance_transform_edt

        part_boundary = np.zeros_like(support, dtype=bool)
        for part in parts:
            ring = np.asarray(part.mask, dtype=bool) & ~binary_erosion(
                np.asarray(part.mask, dtype=bool), iterations=2
            )
            part_boundary |= ring
        if part_boundary.any():
            distance = distance_transform_edt(~part_boundary)
            junction = np.exp(-distance / max(cfg.part_sigma_px, 1.0)).astype(np.float32)

    if structural:
        # structural / logical defects (large, part-crossing, displaced) extend
        # outside the object and cluster near part junctions
        score = (
            density * (1.0 + cfg.boundary_weight * boundary + cfg.part_weight * junction)
            + cfg.structural_floor * boundary
        )
        score = np.maximum(score, cfg.structural_floor * boundary)
    else:
        score = density * (1.0 + 0.25 * (1.0 - boundary) + 0.1 * junction)
    score = np.asarray(score, dtype=np.float32)

    ratios = [value for value in fewshot_area_ratios if value > 0]
    median_ratio = float(np.median(ratios)) if ratios else 0.1
    if area_ratio_override is not None:
        target_ratio = float(area_ratio_override)
    else:
        target_ratio = float(
            np.clip(cfg.area_multiplier * median_ratio, cfg.area_min, cfg.area_max)
        )

    mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    ratio = target_ratio
    used_ratio = target_ratio
    for _attempt in range(cfg.max_adjust + 1):
        threshold = _quantile_threshold(score, support, ratio)
        mask = (score >= threshold) & domain
        cleaned = _cleanup(mask, domain, cfg)
        big = cv2.resize(cleaned.astype(np.uint8), (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST) > 0
        used_ratio = float(big.sum() / max(1, int(domain.sum())))
        if cfg.area_min * 0.8 <= used_ratio <= cfg.area_max * 1.1:
            mask = big
            break
        ratio = float(np.clip(ratio * (target_ratio / max(used_ratio, 1e-3)), 0.05, 0.95))
        mask = big

    if structural and cfg.structural_dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * cfg.structural_dilate_px + 1,) * 2
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        mask &= np.asarray(valid, dtype=bool)
    mask = np.ascontiguousarray(mask, dtype=bool)

    count, _ = cv2.connectedComponents(mask.astype(np.uint8), 8)
    stats = {
        "area_ratio": float(mask.sum() / max(1, int(domain.sum()))),
        "components": int(count - 1),
        "target_ratio": target_ratio,
        "used_ratio": used_ratio,
    }
    return mask, score, stats, domain


__all__ = ["OmegaConfig", "build_allowed_region"]

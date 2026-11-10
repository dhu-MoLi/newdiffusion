"""Stage-0 proposal probe.

Decomposes the A2 entrance into measurable pieces:

    source hole / ring -> prototype -> similarity -> proposal -> TokenCut site

It deliberately does **not** score the proposal against a defect mask on a
defective target: the proposal is a normal-material region and a defect is
anomalous by construction. Location quality is measured separately with the
object frame / SDF statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cadiff.v06_a.a2 import _l2_rows, _upsample_map
from cadiff.v06_a.structure import physical_sdf_px_512, support_frame
from cadiff.v06_b.a2_component_place import (
    candidate_tokencut_sites,
    pick_tokencut_site,
)
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement
from cadiff.v06_atlas_diag.features import downsample_mask
from cadiff.v07.reconstruction import IMAGE_SIZE, ring_mask


@dataclass(frozen=True)
class ProbeResult:
    proposal: np.ndarray
    similarity: np.ndarray
    source_support: np.ndarray
    target_support: np.ndarray
    stats: dict[str, float]


@dataclass(frozen=True)
class SiteResult:
    found: bool
    mask: np.ndarray | None
    pixel_count: int
    site_index: int
    site_count: int


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def probe_from_features(
    *,
    source_features: Any,
    source_mask: np.ndarray,
    source_valid: np.ndarray,
    source_support: np.ndarray,
    target_features: Any,
    target_valid: np.ndarray,
    target_support: np.ndarray,
    quantile: float = 0.60,
) -> ProbeResult:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    for name, value in (
        ("source_mask", source_mask),
        ("source_valid", source_valid),
        ("source_support", source_support),
        ("target_valid", target_valid),
        ("target_support", target_support),
    ):
        if not isinstance(value, np.ndarray) or value.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{name} must be 512x512")
    mask = np.ascontiguousarray(source_mask & source_valid, dtype=bool)
    if not mask.any():
        raise ValueError("source mask is empty after intersecting valid")
    height = int(source_features.grid_height)
    width = int(source_features.grid_width)
    source_grid = downsample_mask(mask, height=height, width=width)
    if not source_grid.any():
        raise ValueError("source mask has no encoder tokens")
    source_desc = _l2_rows(source_features.tokens)
    prototype = _unit(source_desc[source_grid].mean(axis=0))

    target_height = int(target_features.grid_height)
    target_width = int(target_features.grid_width)
    target_desc = _l2_rows(target_features.tokens)
    grid_similarity = target_desc @ prototype
    valid_grid = downsample_mask(target_valid, height=target_height, width=target_width)
    support_grid = downsample_mask(
        target_support & target_valid,
        height=target_height,
        width=target_width,
    )
    support_similarities = grid_similarity[support_grid]
    if support_similarities.size == 0:
        raise ValueError("target support has no encoder tokens")
    tau = float(np.quantile(support_similarities, float(quantile)))
    grid_similarity = np.where(valid_grid, grid_similarity, np.float32(-1.0))
    similarity = _upsample_map(grid_similarity.astype(np.float32), size=IMAGE_SIZE)
    proposal = (similarity >= tau) & target_valid & target_support
    support_pixels = int((target_support & target_valid).sum())
    valid_pixels = int(target_valid.sum())
    proposal_pixels = int(proposal.sum())
    outside = target_support & target_valid & ~proposal
    proposal_mean = float(similarity[proposal].mean()) if proposal_pixels else float("nan")
    outside_mean = float(similarity[outside].mean()) if outside.any() else float("nan")
    proposal_components = int(
        cv2.connectedComponents(proposal.astype(np.uint8), connectivity=8)[0]
    ) - 1
    stats = {
        "tau": tau,
        "prototype_norm": float(np.linalg.norm(prototype)),
        "proposal_pixels": float(proposal_pixels),
        "proposal_components": float(proposal_components),
        "support_pixels": float(support_pixels),
        "valid_pixels": float(valid_pixels),
        "proposal_support_ratio": float(proposal_pixels / max(support_pixels, 1)),
        "support_valid_ratio": float(support_pixels / max(valid_pixels, 1)),
        "proposal_mean_similarity": proposal_mean,
        "support_outside_similarity": outside_mean,
        "similarity_gap": proposal_mean - outside_mean,
    }
    return ProbeResult(
        proposal=proposal,
        similarity=similarity,
        source_support=np.ascontiguousarray(source_support),
        target_support=np.ascontiguousarray(target_support),
        stats=stats,
    )


def fill_consistency_from_features(
    *,
    source_rgb: np.ndarray,
    filled_rgb: np.ndarray,
    hole: np.ndarray,
    valid: np.ndarray,
    features: Any,
    ring_radius: int = 12,
) -> dict[str, float]:
    ring = ring_mask(hole, valid, radius=ring_radius)
    height = int(features.grid_height)
    width = int(features.grid_width)
    hole_grid = downsample_mask(hole & valid, height=height, width=width)
    ring_grid = downsample_mask(ring, height=height, width=width) & ~hole_grid
    if not ring_grid.any():
        # A thin ring can share every encoder cell with the hole. Fall back to
        # the full normal region so the metric stays defined for small holes.
        ring_grid = downsample_mask(valid & ~hole, height=height, width=width)
    if not hole_grid.any() or not ring_grid.any():
        return {
            "fill_feature_cos": float("nan"),
            "fill_color_l1": float("nan"),
            "ring_pixels": float(int(ring.sum())),
        }
    descriptors = _l2_rows(features.tokens)
    hole_mean = _unit(descriptors[hole_grid].mean(axis=0))
    ring_mean = _unit(descriptors[ring_grid].mean(axis=0))
    cosine = float(hole_mean @ ring_mean)
    hole_pixels = filled_rgb[hole & valid].astype(np.float32)
    ring_pixels = source_rgb[ring].astype(np.float32)
    color_l1 = float(
        np.abs(hole_pixels.mean(axis=0) - ring_pixels.mean(axis=0)).mean() / 255.0
    )
    return {
        "fill_feature_cos": cosine,
        "fill_color_l1": color_l1,
        "ring_pixels": float(int(ring.sum())),
    }


def location_stats(
    *,
    mask: np.ndarray,
    support: np.ndarray,
    sdf: np.ndarray | None = None,
) -> dict[str, float]:
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or mask.dtype != bool:
        raise ValueError("mask must be a 512x512 bool array")
    if support.shape != (IMAGE_SIZE, IMAGE_SIZE) or support.dtype != bool:
        raise ValueError("support must be a 512x512 bool array")
    if not mask.any() or not support.any():
        return {
            "mask_pixels": float(int(mask.sum())),
            "support_pixels": float(int(support.sum())),
            "sdf_mean": float("nan"),
            "sdf_median": float("nan"),
            "radial_mean": float("nan"),
            "radial_median": float("nan"),
        }
    signed = physical_sdf_px_512(support) if sdf is None else sdf
    frame = support_frame(support)
    rows, columns = np.nonzero(mask)
    radial = np.sqrt(
        ((columns - frame.center_x) / frame.scale_x) ** 2
        + ((rows - frame.center_y) / frame.scale_y) ** 2
    )
    values = signed[mask]
    return {
        "mask_pixels": float(int(mask.sum())),
        "support_pixels": float(int(support.sum())),
        "sdf_mean": float(values.mean()),
        "sdf_median": float(np.median(values)),
        "radial_mean": float(radial.mean()),
        "radial_median": float(np.median(radial)),
    }


def site_probe(
    *,
    target_support: np.ndarray,
    proposal: np.ndarray,
    seed: int,
    min_pixels: int = 64,
) -> SiteResult:
    try:
        sites = candidate_tokencut_sites(
            target_support, proposal, min_pixels=min_pixels
        )
        site = pick_tokencut_site(sites, seed=seed)
    except EmptyFeasiblePlacement:
        return SiteResult(
            found=False,
            mask=None,
            pixel_count=0,
            site_index=-1,
            site_count=0,
        )
    return SiteResult(
        found=True,
        mask=np.ascontiguousarray(site.mask),
        pixel_count=int(site.pixels),
        site_index=int(site.index),
        site_count=len(sites),
    )


__all__ = [
    "ProbeResult",
    "SiteResult",
    "fill_consistency_from_features",
    "location_stats",
    "probe_from_features",
    "site_probe",
]

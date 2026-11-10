"""Proposal-component site selection (L0).

The baseline labels the *object support* connected components, so a
single-component object yields exactly one site (= the whole proposal).
Here sites are the connected components of the proposal itself, optionally
ranked by similarity and by an object-frame location prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cadiff.v06_a.structure import physical_sdf_px_512, support_frame

IMAGE_SIZE = 512


@dataclass(frozen=True)
class Site:
    index: int
    mask: np.ndarray
    pixels: int
    mean_similarity: float
    sdf_mean: float
    radial_mean: float


def _site(
    *,
    index: int,
    mask: np.ndarray,
    similarity: np.ndarray | None,
    sdf: np.ndarray | None,
    center_x: float,
    center_y: float,
    scale_x: float,
    scale_y: float,
) -> Site:
    rows, columns = np.nonzero(mask)
    radial = np.sqrt(
        ((columns - center_x) / scale_x) ** 2 + ((rows - center_y) / scale_y) ** 2
    )
    return Site(
        index=int(index),
        mask=np.ascontiguousarray(mask, dtype=bool),
        pixels=int(mask.sum()),
        mean_similarity=(
            float(similarity[mask].mean()) if similarity is not None else 0.0
        ),
        sdf_mean=float(sdf[mask].mean()) if sdf is not None else 0.0,
        radial_mean=float(radial.mean()) if radial.size else 0.0,
    )


def proposal_sites(
    proposal: np.ndarray,
    target_support: np.ndarray,
    valid: np.ndarray,
    *,
    similarity: np.ndarray | None = None,
    min_pixels: int = 64,
    sdf: np.ndarray | None = None,
    frame: object | None = None,
) -> tuple[Site, ...]:
    if proposal.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("proposal must be 512x512")
    if target_support.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("target_support must be 512x512")
    if valid.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("valid must be 512x512")
    allowed = proposal & target_support & valid
    if not allowed.any():
        return ()
    count, labels = cv2.connectedComponents(
        allowed.astype(np.uint8), connectivity=8
    )
    actual_frame = support_frame(target_support) if frame is None else frame
    actual_sdf = (
        physical_sdf_px_512(target_support)
        if sdf is None and target_support.any()
        else sdf
    )
    sites: list[Site] = []
    index = 0
    for label in range(1, count):
        mask = labels == label
        pixels = int(mask.sum())
        if pixels < min_pixels:
            continue
        index += 1
        sites.append(
            _site(
                index=index,
                mask=mask,
                similarity=similarity,
                sdf=actual_sdf,
                center_x=actual_frame.center_x,
                center_y=actual_frame.center_y,
                scale_x=actual_frame.scale_x,
                scale_y=actual_frame.scale_y,
            )
        )
    return tuple(sites)


def select_site(
    sites: tuple[Site, ...] | list[Site],
    *,
    seed: int,
    similarity_weight: float = 0.3,
    prior_scores: list[float] | None = None,
    temperature: float = 1.0,
) -> Site:
    if not sites:
        raise ValueError("no candidate site")
    if prior_scores is not None and len(prior_scores) != len(sites):
        raise ValueError("prior_scores must match sites")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be positive and finite")
    scores = np.asarray(
        [float(site.mean_similarity) for site in sites], dtype=np.float64
    ) * float(similarity_weight)
    if prior_scores is not None:
        scores = scores + np.asarray(prior_scores, dtype=np.float64)
    scores = scores / float(temperature)
    scores = scores - scores.max()
    probabilities = np.exp(scores)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        probabilities = np.full(len(sites), 1.0 / len(sites))
    else:
        probabilities = probabilities / total
    rng = np.random.RandomState(int(seed) % (2**32))
    chosen = int(rng.choice(len(sites), p=probabilities))
    return sites[chosen]


__all__ = ["Site", "proposal_sites", "select_site"]

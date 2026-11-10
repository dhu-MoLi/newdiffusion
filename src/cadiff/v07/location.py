"""Object-frame defect location prior (L1).

Few-shot GT masks are mapped into a normalised object frame (support bbox) and
smoothed into a per-type density. At generation time candidate sites are
scored/sampled by this density instead of uniformly at random.

The prior is intentionally simple and interpretable: a normalised 2D histogram
with Gaussian smoothing. It can later be replaced by a learned scorer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from cadiff.v06_a.structure import support_frame
from cadiff.v06_a.tokencut import tokencut_object_mask
from cadiff.v06_c.visa_pilot import fewshot_identity_records, load_visa_identity

IMAGE_SIZE = 512


@dataclass
class ObjectLocationPrior:
    bins: int
    extent: float
    sigma: float
    epsilon: float
    histograms: dict[str, np.ndarray]
    pooled: np.ndarray

    @classmethod
    def from_masks(
        cls,
        masks_by_type: Mapping[str, Sequence[np.ndarray]],
        supports_by_type: Mapping[str, Sequence[np.ndarray]],
        *,
        bins: int = 64,
        extent: float = 1.5,
        sigma: float = 2.0,
        epsilon: float = 1e-4,
    ) -> "ObjectLocationPrior":
        if type(bins) is not int or bins < 8:
            raise ValueError("bins must be an integer >= 8")
        if not np.isfinite(extent) or extent <= 0.0:
            raise ValueError("extent must be positive")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if set(masks_by_type) != set(supports_by_type):
            raise ValueError("masks and supports must cover the same defect types")
        histograms: dict[str, np.ndarray] = {}
        for defect_type, masks in masks_by_type.items():
            supports = supports_by_type[defect_type]
            if len(masks) != len(supports):
                raise ValueError("masks and supports must be aligned per type")
            accumulated = np.zeros((bins, bins), dtype=np.float64)
            used = 0
            for mask, support in zip(masks, supports, strict=True):
                binary = np.asarray(mask, dtype=bool) & np.asarray(support, dtype=bool)
                if not binary.any():
                    continue
                frame = support_frame(np.asarray(support, dtype=bool))
                rows, columns = np.nonzero(binary)
                u = (columns - frame.center_x) / frame.scale_x
                v = (rows - frame.center_y) / frame.scale_y
                histogram, _, _ = np.histogram2d(
                    u,
                    v,
                    bins=bins,
                    range=[[-extent, extent], [-extent, extent]],
                )
                total = float(histogram.sum())
                if total <= 0.0:
                    continue
                accumulated += histogram / total
                used += 1
            if used == 0:
                continue
            density = gaussian_filter(accumulated / used, sigma=sigma, mode="constant")
            density = np.maximum(density, 0.0)
            total = float(density.sum())
            if total <= 0.0:
                continue
            histograms[defect_type] = density / total + epsilon
        if not histograms:
            raise ValueError("no usable few-shot masks for the location prior")
        pooled = np.mean(list(histograms.values()), axis=0)
        pooled = pooled / pooled.sum() + epsilon
        return cls(
            bins=bins,
            extent=float(extent),
            sigma=float(sigma),
            epsilon=float(epsilon),
            histograms=histograms,
            pooled=pooled,
        )

    @classmethod
    def from_inventory(
        cls,
        inventory: object,
        encoder: object,
        *,
        bins: int = 64,
        extent: float = 1.5,
        sigma: float = 2.0,
    ) -> "ObjectLocationPrior":
        masks_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
        supports_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
        for record in fewshot_identity_records(inventory):
            loaded = load_visa_identity(
                record, dataset_root=inventory.source_root
            )
            if loaded.anomaly_gt is None:
                continue
            features = encoder.encode(loaded.rgb)
            support = tokencut_object_mask(features.tokens, loaded.valid)
            if not support.any():
                support = np.ascontiguousarray(loaded.valid)
            masks_by_type[record.defect_type].append(
                np.ascontiguousarray(loaded.anomaly_gt, dtype=bool)
            )
            supports_by_type[record.defect_type].append(
                np.ascontiguousarray(support, dtype=bool)
            )
        return cls.from_masks(
            masks_by_type,
            supports_by_type,
            bins=bins,
            extent=extent,
            sigma=sigma,
        )

    def _histogram(self, defect_type: str | None) -> np.ndarray:
        if defect_type is not None and defect_type in self.histograms:
            return self.histograms[defect_type]
        return self.pooled

    def _indices(
        self, rows: np.ndarray, columns: np.ndarray, frame: object
    ) -> tuple[np.ndarray, np.ndarray]:
        u = (columns - frame.center_x) / frame.scale_x
        v = (rows - frame.center_y) / frame.scale_y
        iu = np.clip(
            ((u + self.extent) / (2.0 * self.extent) * self.bins).astype(np.int64),
            0,
            self.bins - 1,
        )
        iv = np.clip(
            ((v + self.extent) / (2.0 * self.extent) * self.bins).astype(np.int64),
            0,
            self.bins - 1,
        )
        return iu, iv

    def score_mask(
        self,
        mask: np.ndarray,
        support: np.ndarray,
        defect_type: str | None = None,
    ) -> float:
        mask = np.asarray(mask, dtype=bool) & np.asarray(support, dtype=bool)
        if not mask.any():
            return float("nan")
        rows, columns = np.nonzero(mask)
        frame = support_frame(np.asarray(support, dtype=bool))
        iu, iv = self._indices(rows, columns, frame)
        density = self._histogram(defect_type)[iu, iv]
        return float(np.log(density + self.epsilon).mean())

    def score_site(
        self,
        site: object,
        support: np.ndarray,
        defect_type: str | None = None,
    ) -> float:
        return self.score_mask(site.mask, support, defect_type)

    def heatmap(
        self,
        support: np.ndarray,
        defect_type: str | None = None,
    ) -> np.ndarray:
        support = np.asarray(support, dtype=bool)
        if not support.any():
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        frame = support_frame(support)
        rows, columns = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
        iu, iv = self._indices(rows, columns, frame)
        density = self._histogram(defect_type)[iu, iv]
        return np.ascontiguousarray(density.astype(np.float32))

    def describe(self) -> dict[str, float]:
        return {
            "types": float(len(self.histograms)),
            "bins": float(self.bins),
            "extent": float(self.extent),
            "sigma": float(self.sigma),
            "pooled_max": float(self.pooled.max()),
        }


__all__ = ["ObjectLocationPrior"]

"""Parametric defect-shape prior.

The baseline copies a few-shot ground-truth mask, resizes it and tries six
rotations. This module samples a *distribution* instead: empirical area range,
aspect/solidity statistics, arbitrary rotation, elastic deformation and
optional multi-component composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


@dataclass(frozen=True)
class ShapeStats:
    areas: tuple[int, ...]
    aspect_ratios: tuple[float, ...]
    solidities: tuple[float, ...]
    component_counts: tuple[int, ...]

    @property
    def min_area(self) -> int:
        return min(self.areas)

    @property
    def max_area(self) -> int:
        return max(self.areas)


def _crop_to_content(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(binary)
    if rows.size == 0:
        raise ValueError("empty mask")
    return np.ascontiguousarray(
        binary[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    )


def mask_metrics(mask: np.ndarray) -> dict[str, float]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        raise ValueError("empty mask")
    rows, columns = np.nonzero(binary)
    height = int(rows.max() - rows.min() + 1)
    width = int(columns.max() - columns.min() + 1)
    aspect = float(width) / float(max(height, 1))
    component_count = int(cv2.connectedComponents(binary, connectivity=8)[0]) - 1
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    if largest is None:
        solidity = 0.0
    else:
        contour_area = float(cv2.contourArea(largest))
        hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0
    return {
        "area": float(area),
        "aspect": aspect,
        "solidity": solidity,
        "components": float(component_count),
    }


def compute_shape_stats(masks: Sequence[np.ndarray]) -> ShapeStats:
    metrics = [mask_metrics(mask) for mask in masks]
    return ShapeStats(
        areas=tuple(int(item["area"]) for item in metrics),
        aspect_ratios=tuple(float(item["aspect"]) for item in metrics),
        solidities=tuple(float(item["solidity"]) for item in metrics),
        component_counts=tuple(int(item["components"]) for item in metrics),
    )


def _elastic(mask: np.ndarray, alpha: float, sigma: float, rng: np.random.RandomState) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return binary
    height, width = binary.shape
    displacement_y = gaussian_filter(
        rng.uniform(-1.0, 1.0, (height, width)), sigma=sigma, mode="reflect"
    ) * alpha
    displacement_x = gaussian_filter(
        rng.uniform(-1.0, 1.0, (height, width)), sigma=sigma, mode="reflect"
    ) * alpha
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    warped = map_coordinates(
        binary.astype(np.float32),
        [grid_y + displacement_y, grid_x + displacement_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return warped > 0.5


def _rotate(mask: np.ndarray, degrees: float) -> np.ndarray:
    binary = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
    height, width = binary.shape
    center = (width / 2.0 - 0.5, height / 2.0 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        binary,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return rotated > 127


def _resize_to_area(mask: np.ndarray, target_area: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    area = int(binary.sum())
    if area == 0:
        raise ValueError("empty mask")
    scale = float(np.sqrt(max(target_area, 1.0) / float(area)))
    height, width = binary.shape
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        binary.astype(np.uint8),
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


class ParametricShapePrior:
    def __init__(
        self,
        masks_by_type: Mapping[str, Sequence[np.ndarray]],
        *,
        size: int = 512,
        elastic_alpha: float = 6.0,
        elastic_sigma: float = 8.0,
        elastic_probability: float = 0.5,
        multi_component_probability: float = 0.0,
        max_components: int = 3,
        rotation: bool = True,
        rng_seed: int = 0,
    ) -> None:
        if type(size) is not int or size < 32:
            raise ValueError("size must be an integer >= 32")
        if not masks_by_type:
            raise ValueError("masks_by_type must be non-empty")
        cleaned: dict[str, tuple[np.ndarray, ...]] = {}
        for defect_type, masks in masks_by_type.items():
            if not isinstance(defect_type, str) or not defect_type:
                raise ValueError("defect type names must be non-empty strings")
            arrays = tuple(
                np.ascontiguousarray(np.asarray(mask, dtype=bool)) for mask in masks
            )
            if not arrays:
                raise ValueError(f"no masks for defect type {defect_type}")
            if any(array.shape != (size, size) for array in arrays):
                raise ValueError(f"masks for {defect_type} must be {size}x{size}")
            if any(not array.any() for array in arrays):
                raise ValueError(f"empty mask for {defect_type}")
            cleaned[defect_type] = arrays
        if not 0.0 <= elastic_probability <= 1.0:
            raise ValueError("elastic_probability must be in [0, 1]")
        if not 0.0 <= multi_component_probability <= 1.0:
            raise ValueError("multi_component_probability must be in [0, 1]")
        if type(max_components) is not int or max_components < 1:
            raise ValueError("max_components must be a positive integer")
        self.masks_by_type = cleaned
        self.stats_by_type = {
            defect_type: compute_shape_stats(list(arrays))
            for defect_type, arrays in cleaned.items()
        }
        self.size = size
        self.elastic_alpha = float(elastic_alpha)
        self.elastic_sigma = float(elastic_sigma)
        self.elastic_probability = float(elastic_probability)
        self.multi_component_probability = float(multi_component_probability)
        self.max_components = int(max_components)
        self.rotation = bool(rotation)
        self.rng_seed = int(rng_seed)

    def fewshot_areas(self, defect_type: str) -> tuple[int, ...]:
        return self.stats_by_type[defect_type].areas

    def describe(self, defect_type: str) -> dict[str, float]:
        stats = self.stats_by_type[defect_type]
        return {
            "count": float(len(stats.areas)),
            "area_min": float(stats.min_area),
            "area_max": float(stats.max_area),
            "area_mean": float(np.mean(stats.areas)),
            "aspect_mean": float(np.mean(stats.aspect_ratios)),
            "solidity_mean": float(np.mean(stats.solidities)),
            "components_mean": float(np.mean(stats.component_counts)),
        }

    def _sample_component(
        self,
        sources: Sequence[np.ndarray],
        rng: np.random.RandomState,
        target_area: int,
    ) -> np.ndarray:
        source = sources[int(rng.randint(0, len(sources)))]
        component = _crop_to_content(source)
        component = _resize_to_area(component, target_area)
        if self.rotation:
            component = _rotate(component, float(rng.uniform(0.0, 360.0)))
            component = _crop_to_content(component)
        if rng.rand() < self.elastic_probability:
            component = _elastic(
                component,
                self.elastic_alpha,
                self.elastic_sigma,
                rng,
            )
            if component.any():
                component = _crop_to_content(component)
        return component

    def _paste(
        self,
        canvas: np.ndarray,
        component: np.ndarray,
        rng: np.random.RandomState,
        allow_overlap: bool,
    ) -> None:
        height, width = component.shape
        if height > self.size or width > self.size:
            scale = min(self.size / height, self.size / width) * 0.98
            component = cv2.resize(
                component.astype(np.uint8),
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            height, width = component.shape
        best: tuple[float, int, int] | None = None
        for _ in range(30):
            top = int(rng.randint(0, self.size - height + 1))
            left = int(rng.randint(0, self.size - width + 1))
            candidate = np.zeros_like(canvas)
            candidate[top : top + height, left : left + width] = component
            if not canvas.any():
                best = (0.0, top, left)
                break
            intersection = int((candidate & canvas).sum())
            ratio = intersection / max(1, int(candidate.sum()))
            if best is None or ratio < best[0]:
                best = (ratio, top, left)
            if ratio <= 0.1:
                break
        assert best is not None
        if not allow_overlap and best[0] > 0.35:
            return
        _, top, left = best
        canvas[top : top + height, left : left + width] |= component

    def sample(self, defect_type: str, seed: int) -> np.ndarray:
        if defect_type not in self.masks_by_type:
            raise ValueError(f"unknown defect type for shape prior: {defect_type}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        rng = np.random.RandomState((seed + self.rng_seed) % (2**32))
        stats = self.stats_by_type[defect_type]
        sources = self.masks_by_type[defect_type]
        count = 1
        if self.max_components > 1 and rng.rand() < self.multi_component_probability:
            count = 1 + int(rng.randint(0, self.max_components))
        total_area = int(
            round(
                np.exp(
                    rng.uniform(
                        np.log(max(stats.min_area, 1)),
                        np.log(max(stats.max_area, stats.min_area + 1)),
                    )
                )
            )
        )
        per_component = max(64, total_area // count)
        canvas = np.zeros((self.size, self.size), dtype=bool)
        for index in range(count):
            try:
                component = self._sample_component(sources, rng, per_component)
            except ValueError:
                continue
            self._paste(canvas, component, rng, allow_overlap=index > 0)
        if not canvas.any():
            canvas = np.ascontiguousarray(sources[0])
        return canvas


__all__ = [
    "ParametricShapePrior",
    "ShapeStats",
    "compute_shape_stats",
    "mask_metrics",
]

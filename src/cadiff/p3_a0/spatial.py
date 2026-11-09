from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cadiff.p3_a0.dataset import MaskTarget


def otsu_binary(values: np.ndarray, valid: np.ndarray, *, bins: int) -> np.ndarray:
    values_array = np.asarray(values)
    valid_array = np.asarray(valid, dtype=bool)
    if values_array.shape != valid_array.shape:
        raise ValueError("values and valid must have the same shape")
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer greater than one")

    finite_valid = valid_array & np.isfinite(values_array)
    result = np.zeros(values_array.shape, dtype=bool)
    if not finite_valid.any():
        return result
    samples = values_array[finite_valid].astype(np.float64, copy=False)
    minimum = float(samples.min())
    maximum = float(samples.max())
    if minimum == maximum:
        return result

    histogram, edges = np.histogram(samples, bins=bins, range=(minimum, maximum))
    centers = (edges[:-1] + edges[1:]) * 0.5
    weights_left = np.cumsum(histogram, dtype=np.float64)
    sums_left = np.cumsum(histogram * centers, dtype=np.float64)
    total_weight = weights_left[-1]
    total_sum = sums_left[-1]
    weights_right = total_weight - weights_left
    scores = np.zeros(bins, dtype=np.float64)
    usable = (weights_left > 0.0) & (weights_right > 0.0)
    means_left = sums_left[usable] / weights_left[usable]
    means_right = (total_sum - sums_left[usable]) / weights_right[usable]
    scores[usable] = weights_left[usable] * weights_right[usable] * (means_left - means_right) ** 2
    threshold = centers[int(np.argmax(scores))]
    return finite_valid & (values_array > threshold)


@dataclass(frozen=True)
class MorphologyDescriptor:
    area_fraction: float
    component_count: int
    largest_component_aspect_ratio: float
    perimeter_to_area: float


@dataclass(frozen=True)
class MorphologyCalibration:
    centers: dict[str, MorphologyDescriptor]
    pooled_mad: MorphologyDescriptor
    area_envelopes: dict[str, tuple[float, float]]


def _as_2d(value: object, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return array


def _valid_pair(value: object, valid: object, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    value_array = _as_2d(value, name=name)
    valid_array = _as_2d(valid, name="valid").astype(bool, copy=False)
    if value_array.shape != valid_array.shape:
        raise ValueError(f"{name} and valid must have the same shape")
    if not valid_array.any():
        raise ValueError("valid must contain at least one pixel")
    return value_array, valid_array


def product_roi(rgb: np.ndarray, valid: np.ndarray, *, border_width: int) -> np.ndarray:
    rgb_array = np.asarray(rgb)
    valid_array = _as_2d(valid, name="valid").astype(bool, copy=False)
    if rgb_array.ndim != 3 or rgb_array.shape[:2] != valid_array.shape or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3 matching valid")
    if type(border_width) is not int or border_width <= 0:
        raise ValueError("border_width must be a positive integer")
    coordinates = np.argwhere(valid_array)
    if coordinates.size == 0:
        raise ValueError("product ROI is empty")
    row_min, column_min = coordinates.min(axis=0)
    row_max, column_max = coordinates.max(axis=0)
    rows, columns = np.indices(valid_array.shape)
    border = valid_array & (
        (rows < row_min + border_width)
        | (rows > row_max - border_width)
        | (columns < column_min + border_width)
        | (columns > column_max - border_width)
    )
    background = np.median(rgb_array[border].astype(np.float64, copy=False), axis=0)
    distances = np.linalg.norm(rgb_array.astype(np.float64, copy=False) - background, axis=2)
    finite_valid = valid_array & np.isfinite(distances)
    if not finite_valid.any():
        raise ValueError("product ROI is empty")
    finite_distances = distances[finite_valid]
    if float(finite_distances.min()) == float(finite_distances.max()):
        raise ValueError("product ROI distance is constant")
    roi = otsu_binary(distances, valid_array, bins=256)
    area = int(np.count_nonzero(roi))
    if area == 0:
        raise ValueError("product ROI is empty")
    if area / int(np.count_nonzero(valid_array)) > 0.90:
        raise ValueError("product ROI occupies more than 90% of valid pixels")
    return roi


def morphology_descriptor(binary: np.ndarray, valid: np.ndarray) -> MorphologyDescriptor:
    binary_array, valid_array = _valid_pair(binary, valid, name="binary")
    foreground = binary_array.astype(bool, copy=False) & valid_array
    foreground_area = int(np.count_nonzero(foreground))
    if foreground_area == 0:
        return MorphologyDescriptor(0.0, 0, 0.0, 0.0)
    visited = np.zeros(foreground.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = foreground.shape
    for row_value, column_value in np.argwhere(foreground):
        row = int(row_value)
        column = int(column_value)
        if visited[row, column]:
            continue
        visited[row, column] = True
        stack = [(row, column)]
        component: list[tuple[int, int]] = []
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbor_row = current_row + row_offset
                    neighbor_column = current_column + column_offset
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                        and foreground[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        stack.append((neighbor_row, neighbor_column))
        components.append(component)
    largest = max(components, key=len)
    largest_rows = [row for row, _ in largest]
    largest_columns = [column for _, column in largest]
    box_height = max(largest_rows) - min(largest_rows) + 1
    box_width = max(largest_columns) - min(largest_columns) + 1
    aspect_ratio = max(box_height, box_width) / min(box_height, box_width)
    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1]
    perimeter = sum(
        int(np.count_nonzero(core & ~neighbor))
        for neighbor in (
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
        )
    )
    return MorphologyDescriptor(
        area_fraction=foreground_area / int(np.count_nonzero(valid_array)),
        component_count=len(components),
        largest_component_aspect_ratio=float(aspect_ratio),
        perimeter_to_area=perimeter / foreground_area,
    )


def _descriptor_vector(value: MorphologyDescriptor) -> np.ndarray:
    return np.array(
        [
            value.area_fraction,
            value.component_count,
            value.largest_component_aspect_ratio,
            value.perimeter_to_area,
        ],
        dtype=np.float64,
    )


def _vector_descriptor(value: np.ndarray) -> MorphologyDescriptor:
    return MorphologyDescriptor(float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def calibrate_morphology(targets: tuple[MaskTarget, ...]) -> MorphologyCalibration:
    if not isinstance(targets, tuple) or not targets:
        raise ValueError("targets must be a non-empty tuple")
    grouped: dict[str, list[MorphologyDescriptor]] = {}
    for target in targets:
        grouped.setdefault(target.defect_type, []).append(
            morphology_descriptor(target.mask, target.valid)
        )
    centers: dict[str, MorphologyDescriptor] = {}
    area_envelopes: dict[str, tuple[float, float]] = {}
    for defect_type, descriptors in grouped.items():
        vectors = np.stack([_descriptor_vector(value) for value in descriptors])
        centers[defect_type] = _vector_descriptor(np.median(vectors, axis=0))
        areas = vectors[:, 0]
        area_envelopes[defect_type] = (0.5 * float(areas.min()), min(2.0 * float(areas.max()), 0.5))
    residuals = np.stack(
        [
            np.abs(_descriptor_vector(descriptor) - _descriptor_vector(centers[defect_type]))
            for defect_type, descriptors in grouped.items()
            for descriptor in descriptors
        ]
    )
    pooled = np.maximum(np.median(residuals, axis=0), 1e-6)
    return MorphologyCalibration(centers, _vector_descriptor(pooled), area_envelopes)


def morphology_distance(
    value: MorphologyDescriptor, center: MorphologyDescriptor, scale: MorphologyDescriptor
) -> float:
    return float(
        np.mean(
            np.abs(_descriptor_vector(value) - _descriptor_vector(center))
            / _descriptor_vector(scale)
        )
    )


def normalized_l1(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    left_array, valid_array = _valid_pair(left, valid, name="left")
    right_array, _ = _valid_pair(right, valid, name="right")
    if left_array.shape != right_array.shape:
        raise ValueError("left and right must have the same shape")
    left_values = left_array[valid_array].astype(np.float64, copy=False)
    right_values = right_array[valid_array].astype(np.float64, copy=False)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("valid values must be finite")
    return float(np.mean(np.abs(left_values - right_values)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts, dtype=np.float64)
    starts = ends - counts + 1.0
    return ((starts + ends) * 0.5)[inverse]


def spearman(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    left_array, valid_array = _valid_pair(left, valid, name="left")
    right_array, _ = _valid_pair(right, valid, name="right")
    if left_array.shape != right_array.shape:
        raise ValueError("left and right must have the same shape")
    left_values = left_array[valid_array].astype(np.float64, copy=False)
    right_values = right_array[valid_array].astype(np.float64, copy=False)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("valid values must be finite")
    if float(left_values.min()) == float(left_values.max()) or float(right_values.min()) == float(
        right_values.max()
    ):
        return 0.0
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    return float(np.dot(left_centered, right_centered) / denominator)


def sobel_baseline(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb_array = np.asarray(rgb)
    valid_array = _as_2d(valid, name="valid").astype(bool, copy=False)
    if rgb_array.ndim != 3 or rgb_array.shape[:2] != valid_array.shape or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3 matching valid")
    if not valid_array.any():
        raise ValueError("valid must contain at least one pixel")
    if not np.isfinite(rgb_array).all():
        raise ValueError("rgb must contain only finite values")
    rgb_float = rgb_array.astype(np.float64, copy=False)
    grayscale = 0.299 * rgb_float[:, :, 0] + 0.587 * rgb_float[:, :, 1] + 0.114 * rgb_float[:, :, 2]
    padded = np.pad(grayscale, 1, mode="edge")
    horizontal = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    ) / 8.0
    vertical = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 8.0
    magnitude = np.sqrt(horizontal * horizontal + vertical * vertical)
    valid_values = magnitude[valid_array]
    q01 = float(np.quantile(valid_values, 0.01))
    q99 = float(np.quantile(valid_values, 0.99))
    result = np.zeros(valid_array.shape, dtype=np.float32)
    if q99 - q01 > 1e-8:
        normalized = np.clip((magnitude - q01) / (q99 - q01), 0.0, 1.0)
        result[valid_array] = normalized[valid_array].astype(np.float32)
    return result


def random_baselines(record_id: str, valid: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
    valid_array = _as_2d(valid, name="valid").astype(bool, copy=False)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")
    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    results = []
    for index in range(count):
        digest = hashlib.sha256(f"p3-a0-random\n{record_id}\n{index}".encode()).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        value = (
            np.random.default_rng(seed)
            .random(valid_array.shape, dtype=np.float64)
            .astype(np.float32)
        )
        value[~valid_array] = 0.0
        results.append(value)
    return tuple(results)


__all__ = [
    "MorphologyCalibration",
    "MorphologyDescriptor",
    "calibrate_morphology",
    "morphology_descriptor",
    "morphology_distance",
    "normalized_l1",
    "otsu_binary",
    "product_roi",
    "random_baselines",
    "sobel_baseline",
    "spearman",
]

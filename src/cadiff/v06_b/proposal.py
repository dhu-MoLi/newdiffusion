from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

DEFAULT_PROPOSAL_RADIUS = 24
DIRECT_TRANSFER_RETENTION = 0.95
BACKGROUND_DISTANCE_MARGIN = 0.03
BACKGROUND_MAD_MULTIPLIER = 6.0 * 1.4826


@dataclass(frozen=True, eq=False)
class TrainingProposal:
    """A transferred training target and its coarse permission region."""

    ground_truth: np.ndarray
    proposal: np.ndarray
    retained_fraction: float
    initial_retained_fraction: float
    recentered: bool
    row_shift: int
    column_shift: int


def _binary_mask(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return array.astype(bool, copy=True)


def _bbox(mask: np.ndarray, *, name: str) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError(f"{name} must contain at least one pixel")
    row_min, column_min = coordinates.min(axis=0)
    row_max, column_max = coordinates.max(axis=0)
    return int(row_min), int(row_max) + 1, int(column_min), int(column_max) + 1


def _window_any(mask: np.ndarray, *, radius: int, axis: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(mask, padding, mode="constant", constant_values=False)
    cumulative = np.cumsum(padded, axis=axis, dtype=np.int32)
    leading_shape = list(cumulative.shape)
    leading_shape[axis] = 1
    cumulative = np.concatenate([np.zeros(leading_shape, dtype=np.int32), cumulative], axis=axis)
    window = 2 * radius + 1
    high = [slice(None), slice(None)]
    low = [slice(None), slice(None)]
    high[axis] = slice(window, None)
    low[axis] = slice(None, -window)
    return (cumulative[tuple(high)] - cumulative[tuple(low)]) > 0


def coarsen_ground_truth(
    ground_truth: object,
    support: object,
    *,
    radius: int = DEFAULT_PROPOSAL_RADIUS,
) -> np.ndarray:
    """Square-dilate a GT mask and clip the training-only proposal to the object."""

    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    ground_truth_array = _binary_mask(ground_truth, name="ground_truth")
    support_array = _binary_mask(support, name="support")
    if ground_truth_array.shape != support_array.shape:
        raise ValueError("ground_truth and support must have the same shape")
    if not ground_truth_array.any():
        raise ValueError("ground_truth must contain at least one pixel")
    if (ground_truth_array & ~support_array).any():
        raise ValueError("ground_truth must be inside support")

    horizontal = _window_any(ground_truth_array, radius=radius, axis=1)
    proposal = _window_any(horizontal, radius=radius, axis=0) & support_array
    if not np.all(ground_truth_array <= proposal):
        raise RuntimeError("proposal construction lost ground-truth pixels")
    if proposal.all():
        raise ValueError("proposal must not cover the full image")
    return proposal


def _largest_center_component(foreground: np.ndarray, valid: np.ndarray) -> np.ndarray:
    row_start, row_stop, column_start, column_stop = _bbox(valid, name="valid")
    center_rows = range(
        row_start + (row_stop - row_start) // 4,
        row_start + 3 * (row_stop - row_start) // 4 + 1,
    )
    center_columns = range(
        column_start + (column_stop - column_start) // 4,
        column_start + 3 * (column_stop - column_start) // 4 + 1,
    )
    center_region = np.zeros(foreground.shape, dtype=bool)
    center_region[
        slice(center_rows.start, min(center_rows.stop, row_stop)),
        slice(center_columns.start, min(center_columns.stop, column_stop)),
    ] = True
    # ndimage labels 8-connected components in raster order, matching the
    # original DFS seed order (including its first-component tie break).
    from scipy.ndimage import label

    labels, count = label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    candidates = np.unique(labels[center_region])
    candidates = candidates[candidates != 0]
    if not candidates.size:
        raise ValueError("hazelnut support has no center-associated component")
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    best = candidates[np.argmax(sizes[candidates])]
    return labels == best


def _solid_object_silhouette(component: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill internal color gaps while retaining the component's row/column envelope."""

    from_left = np.maximum.accumulate(component, axis=1)
    from_right = np.maximum.accumulate(component[:, ::-1], axis=1)[:, ::-1]
    from_top = np.maximum.accumulate(component, axis=0)
    from_bottom = np.maximum.accumulate(component[::-1, :], axis=0)[::-1, :]
    return valid & from_left & from_right & from_top & from_bottom


def _recenter_if_crossing_support(
    geometry: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, float, int, int]:
    """Move a cross-object warp to the object center before the final support clip."""

    coordinates = np.argwhere(geometry)
    support_coordinates = np.argwhere(support)
    if coordinates.size == 0 or support_coordinates.size == 0:
        raise ValueError("geometry and support must be non-empty")
    initial_retained_fraction = float((geometry & support).sum() / len(coordinates))
    if initial_retained_fraction >= DIRECT_TRANSFER_RETENTION:
        return geometry, initial_retained_fraction, 0, 0
    offset = np.rint(support_coordinates.mean(axis=0) - coordinates.mean(axis=0)).astype(np.int64)
    shifted_coordinates = coordinates + offset
    height, width = geometry.shape
    inside = (
        (shifted_coordinates[:, 0] >= 0)
        & (shifted_coordinates[:, 0] < height)
        & (shifted_coordinates[:, 1] >= 0)
        & (shifted_coordinates[:, 1] < width)
    )
    shifted = np.zeros_like(geometry)
    shifted[
        shifted_coordinates[inside, 0],
        shifted_coordinates[inside, 1],
    ] = True
    shifted_count = int(shifted.sum())
    if shifted_count <= 0:
        return geometry, initial_retained_fraction, 0, 0
    shifted_retained_fraction = float((shifted & support).sum() / shifted_count)
    if shifted_retained_fraction <= initial_retained_fraction:
        return geometry, initial_retained_fraction, 0, 0
    return shifted, initial_retained_fraction, int(offset[0]), int(offset[1])


def extract_hazelnut_support(
    rgb: object,
    valid: object,
    *,
    corner_width: int = 16,
    occupancy_limit: float = 0.90,
    background_mad_multiplier: float = BACKGROUND_MAD_MULTIPLIER,
) -> np.ndarray:
    """Estimate the centered object from padded RGB geometry."""

    rgb_array = np.asarray(rgb)
    valid_array = _binary_mask(valid, name="valid")
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if rgb_array.shape[:2] != valid_array.shape:
        raise ValueError("rgb and valid must have matching spatial shape")
    if isinstance(corner_width, bool) or not isinstance(corner_width, int) or corner_width <= 0:
        raise ValueError("corner_width must be a positive integer")
    if not valid_array.any():
        raise ValueError("valid must contain at least one pixel")
    if not np.isfinite(rgb_array).all():
        raise ValueError("rgb must contain only finite values")

    row_start, row_stop, column_start, column_stop = _bbox(valid_array, name="valid")
    height = row_stop - row_start
    width = column_stop - column_start
    patch_height = min(corner_width, height)
    patch_width = min(corner_width, width)
    corner = np.zeros(valid_array.shape, dtype=bool)
    for rows in (
        slice(row_start, row_start + patch_height),
        slice(row_stop - patch_height, row_stop),
    ):
        for columns in (
            slice(column_start, column_start + patch_width),
            slice(column_stop - patch_width, column_stop),
        ):
            corner[rows, columns] = True
    corner &= valid_array
    if not corner.any():
        raise ValueError("valid corners must contain at least one pixel")

    rgb_float = rgb_array.astype(np.float64, copy=False)
    if np.any((rgb_float < 0.0) | (rgb_float > 1.0)):
        raise ValueError("rgb must be normalized to [0,1]")
    background = np.median(rgb_float[corner], axis=0)
    distances = np.linalg.norm(rgb_float - background, axis=2)
    finite_distances = distances[valid_array]
    if float(finite_distances.min()) == float(finite_distances.max()):
        raise ValueError("hazelnut support distance is constant")
    corner_distances = distances[corner]
    corner_center = float(np.median(corner_distances))
    corner_mad = float(np.median(np.abs(corner_distances - corner_center)))
    if (
        isinstance(background_mad_multiplier, bool)
        or not isinstance(background_mad_multiplier, int | float)
        or not np.isfinite(background_mad_multiplier)
        or background_mad_multiplier < 0.0
    ):
        raise ValueError("background_mad_multiplier must be a finite non-negative number")
    threshold = (
        corner_center + float(background_mad_multiplier) * corner_mad + BACKGROUND_DISTANCE_MARGIN
    )
    foreground = valid_array & (distances > threshold)
    if not foreground.any():
        raise ValueError("hazelnut support is empty")
    component = _largest_center_component(foreground, valid_array)
    support = _solid_object_silhouette(component, valid_array)
    if support.sum() / valid_array.sum() > occupancy_limit:
        raise ValueError(f"object support occupies more than {occupancy_limit:.0%} of valid pixels")
    return support


def extract_training_support(
    rgb: object,
    valid: object,
    *,
    category: str,
    corner_width: int = 16,
) -> np.ndarray:
    """Object support for B training; weaker Cable MAD preserves textured scenes."""

    if category == "hazelnut":
        return extract_hazelnut_support(rgb, valid, corner_width=corner_width)
    if category == "cable":
        return extract_hazelnut_support(
            rgb,
            valid,
            corner_width=corner_width,
            occupancy_limit=0.98,
            background_mad_multiplier=0.0,
        )
    raise ValueError("training support is implemented only for hazelnut and cable")


def transfer_training_proposal(
    source_ground_truth: object,
    source_support: object,
    target_support: object,
    *,
    proposal_radius: int = DEFAULT_PROPOSAL_RADIUS,
) -> TrainingProposal:
    """Transfer only mask geometry between unpaired object supports."""

    source_ground_truth_array = _binary_mask(source_ground_truth, name="source_ground_truth")
    source_support_array = _binary_mask(source_support, name="source_support")
    target_support_array = _binary_mask(target_support, name="target_support")
    if source_ground_truth_array.shape != source_support_array.shape:
        raise ValueError("source_ground_truth and source_support must have the same shape")
    if not source_ground_truth_array.any():
        raise ValueError("source_ground_truth must contain at least one pixel")
    if (source_ground_truth_array & ~source_support_array).any():
        raise ValueError("source ground truth must be inside source support")

    source_row_start, source_row_stop, source_column_start, source_column_stop = _bbox(
        source_support_array, name="source_support"
    )
    target_row_start, target_row_stop, target_column_start, target_column_stop = _bbox(
        target_support_array, name="target_support"
    )
    source_crop = source_ground_truth_array[
        source_row_start:source_row_stop,
        source_column_start:source_column_stop,
    ]
    target_height = target_row_stop - target_row_start
    target_width = target_column_stop - target_column_start
    resized = Image.fromarray(source_crop.astype(np.uint8) * 255).resize(
        (target_width, target_height), Image.Resampling.NEAREST
    )
    transferred = np.zeros(target_support_array.shape, dtype=bool)
    transferred[target_row_start:target_row_stop, target_column_start:target_column_stop] = (
        np.asarray(resized) >= 128
    )
    (
        transferred,
        initial_retained_fraction,
        row_shift,
        column_shift,
    ) = _recenter_if_crossing_support(transferred, target_support_array)
    mapped_count = int(transferred.sum())
    if mapped_count <= 0:
        raise ValueError("mapped ground truth is empty before target support clipping")
    transferred &= target_support_array
    if not transferred.any():
        raise ValueError("transferred ground truth is empty on target support")
    retained_fraction = float(transferred.sum() / mapped_count)
    proposal = coarsen_ground_truth(
        transferred,
        target_support_array,
        radius=proposal_radius,
    )
    return TrainingProposal(
        ground_truth=transferred,
        proposal=proposal,
        retained_fraction=retained_fraction,
        initial_retained_fraction=initial_retained_fraction,
        recentered=bool(row_shift or column_shift),
        row_shift=row_shift,
        column_shift=column_shift,
    )

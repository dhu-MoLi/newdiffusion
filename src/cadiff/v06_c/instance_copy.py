"""Duplicate the reference TokenCut object so a second instance sits beside it.

Stuck-together labels are a second product, not a texture spot. The trained renderer
repaints pixels it already sees, so a mask on empty background becomes a blur.
This arm copies the reference object's own RGB along its long axis until the copy
touches it. The mask is the union. It is not official H and is not wired into generate().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scipy.ndimage import binary_erosion, binary_fill_holes, label

from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement


@dataclass(frozen=True)
class InstanceCopy:
    image: NDArray[np.uint8]
    mask: NDArray[np.bool_]
    object_mask: NDArray[np.bool_]
    copy_mask: NDArray[np.bool_]
    shift_row: int
    shift_col: int


def product_from_background(
    rgb: np.ndarray,
    valid: np.ndarray,
    *,
    border_fraction: float = 0.06,
) -> NDArray[np.bool_]:
    """Largest region whose color leaves the photo border. Diagnostic object only."""

    image = np.asarray(rgb)
    allowed = np.asarray(valid, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be HxWx3 uint8")
    if allowed.shape != image.shape[:2]:
        raise ValueError("valid must match rgb")
    if not allowed.any():
        raise EmptyFeasiblePlacement("valid region is empty")
    radius = max(2, int(round(min(image.shape[:2]) * float(border_fraction))))
    eroded = binary_erosion(allowed, iterations=radius)
    ring = allowed & ~eroded
    if int(ring.sum()) < 32:
        ring = allowed
    background = np.median(image[ring].astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    threshold = max(
        float(np.percentile(distance[ring], 99) + 12.0),
        0.45 * float(np.percentile(distance[allowed], 90)),
    )
    raw = allowed & (distance > threshold)
    labels, count = label(raw)
    if count < 1:
        raise EmptyFeasiblePlacement("background split found no product")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    mask = binary_fill_holes(labels == int(np.argmax(counts)))
    mask = np.ascontiguousarray(mask & allowed, dtype=bool)
    if not mask.any() or mask.all():
        raise EmptyFeasiblePlacement("product mask is empty or covers the image")
    return mask


def _shift_bool(mask: np.ndarray, row: int, col: int) -> NDArray[np.bool_]:
    out = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    src_top = max(0, -row)
    src_bottom = min(height, height - row)
    src_left = max(0, -col)
    src_right = min(width, width - col)
    if src_bottom <= src_top or src_right <= src_left:
        return out
    dst_top = max(0, row)
    dst_left = max(0, col)
    out[
        dst_top : dst_top + (src_bottom - src_top),
        dst_left : dst_left + (src_right - src_left),
    ] = mask[src_top:src_bottom, src_left:src_right]
    return out


def _shift_rgb(rgb: np.ndarray, row: int, col: int) -> NDArray[np.uint8]:
    out = np.zeros_like(rgb)
    height, width = rgb.shape[:2]
    src_top = max(0, -row)
    src_bottom = min(height, height - row)
    src_left = max(0, -col)
    src_right = min(width, width - col)
    if src_bottom <= src_top or src_right <= src_left:
        return out
    dst_top = max(0, row)
    dst_left = max(0, col)
    out[
        dst_top : dst_top + (src_bottom - src_top),
        dst_left : dst_left + (src_right - src_left),
    ] = rgb[src_top:src_bottom, src_left:src_right]
    return out


def _principal_axes(mask: np.ndarray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows, cols = np.nonzero(mask)
    coords = np.stack((rows.astype(np.float64), cols.astype(np.float64)), axis=1)
    centered = coords - coords.mean(axis=0, keepdims=True)
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    major = np.asarray(vectors[:, -1], dtype=np.float64)
    minor = np.asarray(vectors[:, 0], dtype=np.float64)
    for index, axis in enumerate((major, minor)):
        if axis[1] < 0.0 or (axis[1] == 0.0 and axis[0] < 0.0):
            if index == 0:
                major = -axis
            else:
                minor = -axis
    return major, minor


def _extent_along(mask: np.ndarray, axis: np.ndarray) -> float:
    rows, cols = np.nonzero(mask)
    coords = np.stack((rows.astype(np.float64), cols.astype(np.float64)), axis=1)
    projected = coords @ np.asarray(axis, dtype=np.float64)
    return float(projected.max() - projected.min())


def _max_shift_inside(mask: np.ndarray, row_step: float, col_step: float) -> tuple[int, int]:
    """Largest integer shift along a unit step that keeps every object pixel on canvas."""

    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    limit = None
    if row_step > 1e-8:
        limit = (height - 1 - int(rows.max())) / row_step
    elif row_step < -1e-8:
        limit = int(rows.min()) / (-row_step)
    if col_step > 1e-8:
        room = (width - 1 - int(cols.max())) / col_step
        limit = room if limit is None else min(limit, room)
    elif col_step < -1e-8:
        room = int(cols.min()) / (-col_step)
        limit = room if limit is None else min(limit, room)
    if limit is None or limit < 1.0:
        return 0, 0
    return int(np.floor(limit * row_step)), int(np.floor(limit * col_step))


def duplicate_instance(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    *,
    seed: int,
    min_shape_pixels: int = 64,
    keep_inside: np.ndarray | None = None,
) -> InstanceCopy:
    """Paste a full copy that stays on canvas and sticks out from the object.

    The shift follows the long or short axis until the copy touches the original,
    and never moves farther than the canvas allows. Seed picks between the two
    fullest sides.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(min_shape_pixels) is not int or min_shape_pixels < 1:
        raise ValueError("min_shape_pixels must be a positive integer")
    image = np.asarray(rgb)
    obj = np.asarray(object_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be HxWx3 uint8")
    if obj.shape != image.shape[:2]:
        raise ValueError("object mask must match rgb")
    if not obj.any() or obj.all():
        raise EmptyFeasiblePlacement("object mask must be non-empty and not the full image")
    major, minor = _principal_axes(obj)
    hits: list[tuple[int, int, int, NDArray[np.bool_], NDArray[np.bool_], int, int]] = []
    object_pixels = int(obj.sum())
    region = None if keep_inside is None else np.asarray(keep_inside, dtype=bool)
    if region is not None and region.shape != obj.shape:
        raise ValueError("keep_inside must match the object mask")
    for axis_index, axis in enumerate((major, minor)):
        for sign in (1, -1):
            row, col = _max_shift_inside(obj, sign * float(axis[0]), sign * float(axis[1]))
            extent = _extent_along(obj, axis)
            length = float(np.hypot(row, col))
            if length > extent >= 1.0:
                scale = extent / length
                row = int(round(row * scale))
                col = int(round(col * scale))
            while row != 0 or col != 0:
                shifted = _shift_bool(obj, row, col)
                inside = region is None or not (shifted & ~region).any()
                if int(shifted.sum()) == object_pixels and inside:
                    break
                if abs(row) >= abs(col) and row != 0:
                    row -= int(np.sign(row))
                elif col != 0:
                    col -= int(np.sign(col))
                else:
                    row = 0
                    col = 0
            if row == 0 and col == 0:
                continue
            shifted = _shift_bool(obj, row, col)
            if int(shifted.sum()) != object_pixels:
                continue
            if region is not None and (shifted & ~region).any():
                continue
            extra = np.ascontiguousarray(shifted & ~obj, dtype=bool)
            if int(extra.sum()) < min_shape_pixels:
                continue
            hits.append((int(extra.sum()), axis_index, sign, shifted, extra, row, col))
    if not hits:
        raise EmptyFeasiblePlacement("duplicated object does not stay on the canvas")
    hits.sort(key=lambda item: (item[0], -item[1], item[2]), reverse=True)
    chosen = hits[1] if len(hits) > 1 and seed % 2 == 1 else hits[0]
    _extra_px, _axis_index, _sign, shifted, extra, row, col = chosen
    moved = _shift_rgb(image, row, col)
    composite = np.array(image, copy=True)
    composite[extra] = moved[extra]
    mask = np.ascontiguousarray(obj | shifted, dtype=bool)
    if not mask.any() or mask.all():
        raise EmptyFeasiblePlacement("duplicated mask is empty or covers the image")
    if np.array_equal(composite[~extra], image[~extra]) is False:
        raise RuntimeError("instance copy changed RGB outside the added object")
    return InstanceCopy(
        image=np.ascontiguousarray(composite),
        mask=mask,
        object_mask=np.ascontiguousarray(obj),
        copy_mask=extra,
        shift_row=row,
        shift_col=col,
    )


__all__ = ["InstanceCopy", "duplicate_instance", "product_from_background"]

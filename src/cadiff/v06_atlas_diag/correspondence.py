"""Dense nearest-neighbour correspondence and mask splat. No DINO import."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CollapseLimits:
    max_cycle_fail_ratio: float
    min_valid_matches: int
    max_median_displacement_px: float
    wipe_on_collapse: bool = True


@dataclass(frozen=True)
class CorrespondenceResult:
    flow_xy: NDArray[np.float32]
    match_valid: NDArray[np.bool_]
    cycle_fail_ratio: float
    valid_match_count: int
    median_displacement_px: float
    collapsed: bool
    collapse_reason: str | None


def feature_cell_centers_512(
    height: int, width: int, *, image_size: int = 512
) -> NDArray[np.float32]:
    """Return (H, W, 2) xy centers of a regular feature grid on a square canvas."""

    if height <= 0 or width <= 0 or image_size <= 0:
        raise ValueError("feature grid and image_size must be positive")
    cell_w = image_size / width
    cell_h = image_size / height
    columns = (np.arange(width, dtype=np.float32) + 0.5) * cell_w
    rows = (np.arange(height, dtype=np.float32) + 0.5) * cell_h
    grid_x, grid_y = np.meshgrid(columns, rows)
    return np.stack((grid_x, grid_y), axis=-1).astype(np.float32)


def _l2_normalize(features: NDArray[np.floating]) -> NDArray[np.float32]:
    vectors = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def match_feature_maps(
    source_features: NDArray[np.floating],
    target_features: NDArray[np.floating],
    source_support: NDArray[np.bool_],
    target_support: NDArray[np.bool_],
    *,
    limits: CollapseLimits,
    image_size: int = 512,
    smooth_radius: int = 1,
) -> CorrespondenceResult:
    """Mutual nearest-neighbour matching with cycle check and local median smooth."""

    if source_features.ndim != 3 or target_features.ndim != 3:
        raise ValueError("features must be (H, W, C)")
    if source_features.shape[:2] != source_support.shape:
        raise ValueError("source support must match source feature map")
    if target_features.shape[:2] != target_support.shape:
        raise ValueError("target support must match target feature map")
    if source_features.shape[2] != target_features.shape[2]:
        raise ValueError("source and target feature depths must match")

    source_desc = _l2_normalize(source_features)
    target_desc = _l2_normalize(target_features)
    source_h, source_w = source_desc.shape[:2]
    target_h, target_w = target_desc.shape[:2]
    source_centers = feature_cell_centers_512(source_h, source_w, image_size=image_size)
    target_centers = feature_cell_centers_512(target_h, target_w, image_size=image_size)

    source_flat = source_desc.reshape(-1, source_desc.shape[-1])
    target_flat = target_desc.reshape(-1, target_desc.shape[-1])
    similarity = source_flat @ target_flat.T
    source_active = source_support.reshape(-1)
    target_active = target_support.reshape(-1)
    similarity[~source_active] = -np.inf
    similarity[:, ~target_active] = -np.inf

    forward = np.argmax(similarity, axis=1)
    backward = np.argmax(similarity, axis=0)
    source_rows, source_cols = np.divmod(np.arange(source_h * source_w), source_w)
    target_rows, target_cols = np.divmod(forward, target_w)
    returned = backward[forward]
    returned_rows, returned_cols = np.divmod(returned, source_w)
    cycle_ok = (
        source_active
        & target_active[forward]
        & (np.abs(returned_rows - source_rows) <= 1)
        & (np.abs(returned_cols - source_cols) <= 1)
    )

    raw_flow = np.zeros((source_h, source_w, 2), dtype=np.float32)
    raw_valid = cycle_ok.reshape(source_h, source_w)
    dest_x = target_centers[target_rows, target_cols, 0].reshape(source_h, source_w)
    dest_y = target_centers[target_rows, target_cols, 1].reshape(source_h, source_w)
    raw_flow[..., 0] = dest_x - source_centers[..., 0]
    raw_flow[..., 1] = dest_y - source_centers[..., 1]
    raw_flow[~raw_valid] = 0.0

    flow = _median_smooth_flow(raw_flow, raw_valid, radius=smooth_radius)
    valid = raw_valid.copy()
    source_count = int(source_active.sum())
    valid_count = int(valid.sum())
    cycle_fail_ratio = 1.0 if source_count == 0 else 1.0 - valid_count / source_count
    if valid_count:
        displacements = np.linalg.norm(flow[valid], axis=1)
        median_disp = float(np.median(displacements))
    else:
        median_disp = float("inf")

    reason: str | None = None
    if source_count == 0:
        reason = "empty_source_support"
    elif cycle_fail_ratio > limits.max_cycle_fail_ratio:
        reason = "cycle_fail_ratio"
    elif valid_count < limits.min_valid_matches:
        reason = "too_few_matches"
    elif median_disp > limits.max_median_displacement_px:
        reason = "median_displacement"

    if reason is not None and limits.wipe_on_collapse:
        valid = np.zeros_like(valid)
        flow = np.zeros_like(flow)
        reported_count = 0
        reported_disp = float("inf")
    else:
        reported_count = valid_count
        reported_disp = median_disp

    return CorrespondenceResult(
        flow_xy=flow,
        match_valid=valid,
        cycle_fail_ratio=float(cycle_fail_ratio),
        valid_match_count=reported_count,
        median_displacement_px=reported_disp,
        collapsed=reason is not None,
        collapse_reason=reason,
    )


def _median_smooth_flow(
    flow: NDArray[np.float32], valid: NDArray[np.bool_], *, radius: int
) -> NDArray[np.float32]:
    if radius <= 0:
        return flow.copy()
    height, width, _ = flow.shape
    smoothed = flow.copy()
    for row in range(height):
        for column in range(width):
            if not valid[row, column]:
                continue
            top = max(0, row - radius)
            bottom = min(height, row + radius + 1)
            left = max(0, column - radius)
            right = min(width, column + radius + 1)
            patch_valid = valid[top:bottom, left:right]
            if not patch_valid.any():
                continue
            patch = flow[top:bottom, left:right][patch_valid]
            smoothed[row, column] = np.median(patch, axis=0)
    return smoothed


def splat_mask(
    source_mask: NDArray[np.bool_],
    result: CorrespondenceResult,
    *,
    image_size: int = 512,
    neighbor_radius: int = 0,
) -> NDArray[np.bool_]:
    """Vote source mask pixels onto the target using the feature-grid flow."""

    if source_mask.shape != (image_size, image_size):
        raise ValueError("source mask must match image_size")
    if neighbor_radius < 0:
        raise ValueError("splat neighbor radius must be non-negative")
    height, width = result.match_valid.shape
    cell_w = image_size / width
    cell_h = image_size / height
    rows, cols = np.nonzero(source_mask)
    if rows.size == 0:
        return np.zeros((image_size, image_size), dtype=bool)
    cell_col = np.clip((cols / cell_w).astype(np.int32), 0, width - 1)
    cell_row = np.clip((rows / cell_h).astype(np.int32), 0, height - 1)
    usable = result.match_valid[cell_row, cell_col]
    if neighbor_radius > 0 and not usable.all():
        cell_row, cell_col, usable = _inherit_neighbor_cells(
            cell_row,
            cell_col,
            usable,
            result.match_valid,
            radius=neighbor_radius,
        )
    if not usable.any():
        return np.zeros((image_size, image_size), dtype=bool)
    dest_x = cols[usable] + result.flow_xy[cell_row[usable], cell_col[usable], 0]
    dest_y = rows[usable] + result.flow_xy[cell_row[usable], cell_col[usable], 1]
    dest_x = np.rint(dest_x).astype(np.int32)
    dest_y = np.rint(dest_y).astype(np.int32)
    inside = (dest_x >= 0) & (dest_x < image_size) & (dest_y >= 0) & (dest_y < image_size)
    transferred = np.zeros((image_size, image_size), dtype=bool)
    transferred[dest_y[inside], dest_x[inside]] = True
    return transferred


def _inherit_neighbor_cells(
    cell_row: NDArray[np.int32],
    cell_col: NDArray[np.int32],
    usable: NDArray[np.bool_],
    match_valid: NDArray[np.bool_],
    *,
    radius: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.bool_]]:
    inherited_row = cell_row.copy()
    inherited_col = cell_col.copy()
    inherited_ok = usable.copy()
    height, width = match_valid.shape
    offsets = [
        (delta_r, delta_c)
        for delta_r in range(-radius, radius + 1)
        for delta_c in range(-radius, radius + 1)
        if not (delta_r == 0 and delta_c == 0)
    ]
    for delta_r, delta_c in offsets:
        if inherited_ok.all():
            break
        neighbor_row = np.clip(cell_row + delta_r, 0, height - 1)
        neighbor_col = np.clip(cell_col + delta_c, 0, width - 1)
        found = ~inherited_ok & match_valid[neighbor_row, neighbor_col]
        inherited_row = np.where(found, neighbor_row, inherited_row)
        inherited_col = np.where(found, neighbor_col, inherited_col)
        inherited_ok = inherited_ok | found
    return inherited_row, inherited_col, inherited_ok


def select_used_mask(
    propagated: NDArray[np.bool_],
    geometry: NDArray[np.bool_],
    *,
    collapsed: bool,
    use_kept_splat: bool,
) -> tuple[NDArray[np.bool_], bool]:
    """Return (used_mask, used_geometry_fallback)."""

    if use_kept_splat:
        if propagated.any():
            return propagated, False
        return geometry, True
    if collapsed or not propagated.any():
        return geometry, True
    return propagated, False


def geometry_fallback_mask(
    source_mask: NDArray[np.bool_],
    source_support: NDArray[np.bool_],
    target_support: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """One similarity map from source support bbox onto target support bbox."""

    if source_mask.shape != source_support.shape or source_support.shape != target_support.shape:
        raise ValueError("mask and supports must share one shape")
    source_box = _bbox(source_support)
    target_box = _bbox(target_support)
    if source_box is None or target_box is None:
        return np.zeros_like(source_mask)
    source_cx, source_cy, source_w, source_h = source_box
    target_cx, target_cy, target_w, target_h = target_box
    scale = float(np.sqrt((target_w / source_w) * (target_h / source_h)))
    rows, cols = np.nonzero(source_mask)
    dest_x = (cols - source_cx) * scale + target_cx
    dest_y = (rows - source_cy) * scale + target_cy
    dest_x = np.rint(dest_x).astype(np.int32)
    dest_y = np.rint(dest_y).astype(np.int32)
    height, width = source_mask.shape
    inside = (dest_x >= 0) & (dest_x < width) & (dest_y >= 0) & (dest_y < height)
    transferred = np.zeros_like(source_mask)
    transferred[dest_y[inside], dest_x[inside]] = True
    return transferred


def source_match_support(
    source_gt: NDArray[np.bool_],
    source_object: NDArray[np.bool_],
    source_valid: NDArray[np.bool_],
    *,
    dilate_px: int,
) -> NDArray[np.bool_]:
    """Restrict matching to a dilated source-GT band, or the full object if dilate_px=0."""

    if source_gt.shape != source_object.shape or source_object.shape != source_valid.shape:
        raise ValueError("source gt, object support, and valid mask must share one shape")
    if dilate_px < 0:
        raise ValueError("source match dilation must be non-negative")
    if dilate_px == 0:
        return source_object & source_valid
    return square_dilate(source_gt, dilate_px) & source_object & source_valid


def square_dilate(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    dilated = mask.copy()
    if radius == 0 or not mask.any():
        return dilated
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    for row, column in zip(rows, cols, strict=True):
        dilated[
            max(0, row - radius) : min(height, row + radius + 1),
            max(0, column - radius) : min(width, column + radius + 1),
        ] = True
    return dilated


def _bbox(mask: NDArray[np.bool_]) -> tuple[float, float, float, float] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    left = float(cols.min())
    right = float(cols.max()) + 1.0
    top = float(rows.min())
    bottom = float(rows.max()) + 1.0
    return ((left + right) / 2.0, (top + bottom) / 2.0, right - left, bottom - top)

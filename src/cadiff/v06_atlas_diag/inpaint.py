"""Source-only inpaint so correspondence matches structure, not the defect."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def inpaint_source(
    rgb: NDArray[np.uint8],
    hole: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    *,
    radius: int = 3,
) -> NDArray[np.uint8]:
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("inpaint expects 512x512 uint8 RGB")
    if hole.shape != (512, 512) or valid.shape != (512, 512):
        raise ValueError("hole and valid must be 512x512")
    masked = hole & valid
    filled = rgb.copy()
    if not masked.any():
        return filled
    try:
        import cv2

        hole_u8 = masked.astype(np.uint8) * 255
        return cv2.inpaint(rgb, hole_u8, radius, cv2.INPAINT_TELEA)
    except ImportError:
        return neighbor_fill_source(rgb, hole, valid)


def neighbor_fill_source(
    rgb: NDArray[np.uint8],
    hole: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    *,
    radius: int = 3,
) -> NDArray[np.uint8]:
    """Always nearest-neighbor fill. Official a2 never uses cv2 Telea."""

    del radius
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("inpaint expects 512x512 uint8 RGB")
    if hole.shape != (512, 512) or valid.shape != (512, 512):
        raise ValueError("hole and valid must be 512x512")
    masked = hole & valid
    filled = rgb.copy()
    if not masked.any():
        return filled
    return _neighbor_fill(filled, masked, valid)


def _neighbor_fill(
    rgb: NDArray[np.uint8], hole: NDArray[np.bool_], valid: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    filled = rgb.copy()
    source = valid & ~hole
    if not source.any():
        return filled
    rows, cols = np.nonzero(hole)
    source_rows, source_cols = np.nonzero(source)
    for row, column in zip(rows, cols, strict=True):
        distance = (source_rows - row) ** 2 + (source_cols - column) ** 2
        nearest = int(np.argmin(distance))
        filled[row, column] = rgb[source_rows[nearest], source_cols[nearest]]
    return filled

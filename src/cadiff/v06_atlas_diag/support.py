"""Coarse object support for search restriction. Not a part label."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cadiff.v06_a.data import LoadedIdentity
from cadiff.v06_a.structure import estimate_normal_structure


def target_support(normal: LoadedIdentity, experiment) -> NDArray[np.bool_]:
    return estimate_normal_structure(normal, config=experiment).support


def rgb_support(
    rgb: NDArray[np.uint8],
    valid: NDArray[np.bool_],
    *,
    border_width: int = 16,
    mad_k: float = 3.5,
    close_px: int = 4,
) -> NDArray[np.bool_]:
    """Estimate support from RGB and V only, same MAD idea as Module A."""

    if rgb.shape != (512, 512, 3) or valid.shape != (512, 512):
        raise ValueError("rgb_support expects 512x512 inputs")
    border = np.zeros_like(valid)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border &= valid
    if not border.any():
        return valid.copy()
    background = rgb[border].astype(np.float32)
    median = np.median(background, axis=0)
    distance = np.max(np.abs(rgb.astype(np.float32) - median), axis=2)
    scatter = float(np.median(np.abs(distance[border] - np.median(distance[border]))))
    threshold = float(np.median(distance[border]) + mad_k * max(scatter, 1.0))
    raw = valid & (distance > threshold)
    if not raw.any():
        return valid.copy()
    labels = _label_components(raw)
    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0
    if counts.max() <= 0:
        return valid.copy()
    largest = labels == int(np.argmax(counts))
    if close_px > 0:
        from .correspondence import square_dilate

        largest = square_dilate(largest, close_px) & valid
    return largest


def _label_components(mask: NDArray[np.bool_]) -> NDArray[np.int32]:
    try:
        from scipy.ndimage import label

        labels, _count = label(mask)
        return labels.astype(np.int32)
    except ImportError:
        pass
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current = 0
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or labels[row, column]:
                continue
            current += 1
            stack = [(row, column)]
            labels[row, column] = current
            while stack:
                y, x = stack.pop()
                for ny in (y - 1, y, y + 1):
                    for nx in (x - 1, x, x + 1):
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask[ny, nx]
                            and not labels[ny, nx]
                        ):
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels

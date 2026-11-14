"""Structural / object-level defect masks (v08 component 2).

Scale calibration fixes how large a defect is; this module changes *what kind*
of defect is synthesised for object-level types: merged parts (stuck_together),
cracks (cut/broken), displaced parts (misplaced) and -- later, once a background
renderer exists -- missing parts.

Operators work directly on the target object support, so they can produce the
large masks that copying a single few-shot shape cannot reach.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

CANVAS = 512
MIN_PIXELS = 64


def operator_for(defect_type: str) -> str | None:
    """Map a defect-type name to a structural operator (None = ordinary shape)."""

    name = defect_type.lower().replace("-", "_")
    if any(token in name for token in ("stuck", "together", "merge", "adhes")):
        return "merge"
    if any(token in name for token in ("misplaced", "displac", "flip", "shift")):
        return "displace"
    if any(token in name for token in ("cut", "broken", "crack", "split", "scratch")):
        return "split"
    return None


def _components(mask: np.ndarray) -> list[np.ndarray]:
    import cv2

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    parts: list[np.ndarray] = []
    for label in range(1, count):
        component = labels == label
        if int(component.sum()) >= MIN_PIXELS:
            parts.append(np.ascontiguousarray(component))
    return parts


def _disc(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    grid = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (grid[0] ** 2 + grid[1] ** 2) <= radius * radius


def _stamp(canvas: np.ndarray, centre: tuple[int, int], radius: int) -> None:
    disc = _disc(radius)
    row, col = int(centre[0]), int(centre[1])
    r0, r1 = row - radius, row + radius + 1
    c0, c1 = col - radius, col + radius + 1
    rr0, rr1 = max(0, r0), min(CANVAS, r1)
    cc0, cc1 = max(0, c0), min(CANVAS, c1)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    canvas[rr0:rr1, cc0:cc1] |= disc[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]


def _line(canvas: np.ndarray, start: np.ndarray, end: np.ndarray, radius: int) -> None:
    length = float(np.hypot(*(end - start)))
    steps = max(2, int(length))
    for t in np.linspace(0.0, 1.0, steps):
        point = start + t * (end - start)
        _stamp(canvas, (point[0], point[1]), radius)


def _support_parts(support: np.ndarray, rng: np.random.RandomState) -> list[np.ndarray]:
    parts = _components(support)
    if len(parts) >= 2:
        return parts
    # single blob: split it in two along its principal axis so that structural
    # operators still have two "parts" to work with
    rows, cols = np.nonzero(support)
    if rows.size < 2 * MIN_PIXELS:
        return parts
    centre = np.array([rows.mean(), cols.mean()])
    coords = np.stack([rows, cols], axis=1) - centre
    cov = np.cov(coords.T)
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    projection = coords @ axis
    cut = float(np.median(projection))
    first = np.zeros_like(support)
    second = np.zeros_like(support)
    first[rows[projection <= cut], cols[projection <= cut]] = True
    second[rows[projection > cut], cols[projection > cut]] = True
    first = binary_dilation(first, iterations=2) & support
    second = binary_dilation(second, iterations=2) & support
    return [np.ascontiguousarray(first), np.ascontiguousarray(second)]


def merge_mask(
    support: np.ndarray,
    valid: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray | None:
    """Bridge two adjacent object parts (stuck_together)."""

    parts = _support_parts(support & valid, rng)
    if len(parts) < 2:
        return None
    centres = [np.array(np.nonzero(part), dtype=np.float64).mean(axis=1) for part in parts]
    best = None
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            distance = float(np.hypot(*(centres[i] - centres[j])))
            if best is None or distance < best[0]:
                best = (distance, i, j)
    if best is None:
        return None
    _distance, i, j = best
    first, second = parts[i], parts[j]
    # nearest pixel pair between the two parts
    dmap = distance_transform_edt(~first)
    rows, cols = np.nonzero(second)
    if rows.size == 0:
        return None
    values = dmap[rows, cols]
    index = int(np.argmin(values))
    end_a = np.array([rows[index], cols[index]], dtype=np.float64)
    dmap2 = distance_transform_edt(~second)
    rows_a, cols_a = np.nonzero(first)
    values2 = dmap2[rows_a, cols_a]
    index2 = int(np.argmin(values2))
    end_b = np.array([rows_a[index2], cols_a[index2]], dtype=np.float64)
    canvas = np.zeros_like(support, dtype=bool)
    radius = int(rng.randint(3, 8))
    _line(canvas, end_a, end_b, radius)
    mask = canvas & binary_dilation(support & valid, iterations=radius + 2)
    if int(mask.sum()) < MIN_PIXELS:
        return None
    return np.ascontiguousarray(mask, dtype=bool)


def split_mask(
    support: np.ndarray,
    valid: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray | None:
    """Thin crack across one object part (cut / broken)."""

    parts = _support_parts(support & valid, rng)
    if not parts:
        return None
    part = parts[int(rng.randint(0, len(parts)))]
    rows, cols = np.nonzero(part)
    if rows.size < MIN_PIXELS:
        return None
    centre = np.array([rows.mean(), cols.mean()])
    angle = float(rng.uniform(0.0, np.pi))
    direction = np.array([np.cos(angle), np.sin(angle)])
    reach = float(max(rows.max() - rows.min(), cols.max() - cols.min())) * 0.75 + 4
    start = centre - direction * reach
    end = centre + direction * reach
    canvas = np.zeros_like(support, dtype=bool)
    _line(canvas, start, end, int(rng.randint(1, 3)))
    mask = canvas & part
    if int(mask.sum()) < MIN_PIXELS:
        mask = canvas & binary_dilation(part, iterations=1)
    if int(mask.sum()) < MIN_PIXELS:
        return None
    return np.ascontiguousarray(mask, dtype=bool)


def displace_mask(
    support: np.ndarray,
    valid: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray | None:
    """A part and its shifted copy (misplaced / flipped)."""

    parts = _support_parts(support & valid, rng)
    if not parts:
        return None
    parts.sort(key=lambda part: int(part.sum()))
    part = parts[int(rng.randint(0, len(parts)))]
    rows, cols = np.nonzero(part)
    if rows.size < MIN_PIXELS:
        return None
    scale = float(np.sqrt(part.sum()))
    magnitude = max(6.0, scale * float(rng.uniform(0.25, 0.7)))
    angle = float(rng.uniform(0.0, 2 * np.pi))
    shift_row = int(round(np.cos(angle) * magnitude))
    shift_col = int(round(np.sin(angle) * magnitude))
    shifted = np.zeros_like(part)
    r0, r1 = max(0, shift_row), min(CANVAS, CANVAS + shift_row)
    c0, c1 = max(0, shift_col), min(CANVAS, CANVAS + shift_col)
    shifted[r0:r1, c0:c1] = part[
        r0 - shift_row : r1 - shift_row, c0 - shift_col : c1 - shift_col
    ]
    mask = part | shifted
    mask &= binary_dilation(support & valid, iterations=8)
    if int(mask.sum()) < MIN_PIXELS:
        return None
    return np.ascontiguousarray(mask, dtype=bool)


OPERATORS: dict[str, Callable[[np.ndarray, np.ndarray, np.random.RandomState], np.ndarray | None]] = {
    "merge": merge_mask,
    "split": split_mask,
    "displace": displace_mask,
}


def make_structural_mask(
    operator: str,
    *,
    support: np.ndarray,
    valid: np.ndarray,
    seed: int,
) -> np.ndarray | None:
    handler = OPERATORS.get(operator)
    if handler is None:
        return None
    rng = np.random.RandomState(seed)
    mask = handler(np.asarray(support, dtype=bool), np.asarray(valid, dtype=bool), rng)
    if mask is None:
        return None
    mask = mask & np.asarray(valid, dtype=bool)
    if int(mask.sum()) < MIN_PIXELS:
        return None
    return np.ascontiguousarray(mask, dtype=bool)


__all__ = ["OPERATORS", "make_structural_mask", "operator_for"]

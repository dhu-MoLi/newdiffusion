"""PSMG step 1: family classification and analytic inversion.

Every ground-truth mask is described by structure-aware evidence (area, shape,
position, part occupancy) and mapped to one of six edit families *without*
looking at the defect-type name.  The same evidence is inverted into family
parameters that can be rasterised back into a mask, measuring how expressive
the atom library is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

CANVAS = 512
FAMILIES = (
    "surface_texture",
    "material_gain",
    "material_loss",
    "geometric_cut",
    "deformation",
    "displacement",
)


@dataclass
class MaskEvidence:
    area_ratio: float
    aspect: float
    bbox_solidity: float
    hull_ratio: float
    elongation: float
    n_components: int
    max_part_iou: float
    crosses_parts: float
    sdf_depth: float
    radial: float
    centroid_u: float
    centroid_v: float
    boundary_touch: float
    two_blob_gap: float
    part_pair_gap: float
    part_index: int
    second_part_index: int


def _hull_ratio(mask: np.ndarray) -> float:
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 1.0
    hull_area = sum(cv2.contourArea(cv2.convexHull(item)) for item in contours)
    area = float(mask.sum())
    return float(area / max(hull_area, 1.0))


def _principal(mask: np.ndarray) -> float:
    rows, cols = np.nonzero(mask)
    if rows.size < 3:
        return 1.0
    coords = np.stack([rows, cols], axis=1).astype(np.float64)
    coords -= coords.mean(axis=0)
    cov = np.cov(coords.T)
    values = np.sort(np.linalg.eigvalsh(cov))
    return float(np.sqrt(max(values[-1], 1e-6) / max(values[0], 1e-6)))


def _components(mask: np.ndarray, min_pixels: int = 32) -> list[np.ndarray]:
    import cv2

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    parts = []
    for label in range(1, count):
        component = labels == label
        if int(component.sum()) >= min_pixels:
            parts.append(component)
    parts.sort(key=lambda item: int(item.sum()), reverse=True)
    return parts


def extract_evidence(
    mask: np.ndarray,
    support: np.ndarray,
    valid: np.ndarray,
    parts: list,
    sdf: np.ndarray,
    frame: object,
) -> MaskEvidence:
    mask = np.ascontiguousarray(np.asarray(mask, dtype=bool) & np.asarray(support, dtype=bool))
    area = float(mask.sum())
    support_area = float(max(int(support.sum()), 1))
    rows, cols = np.nonzero(mask)
    height = int(rows.max() - rows.min() + 1) if rows.size else 1
    width = int(cols.max() - cols.min() + 1) if cols.size else 1
    components = _components(mask)
    two_gap = 0.0
    if len(components) >= 2:
        c1 = np.array(np.nonzero(components[0]), dtype=np.float64).mean(axis=1)
        c2 = np.array(np.nonzero(components[1]), dtype=np.float64).mean(axis=1)
        two_gap = float(np.hypot(*(c1 - c2)) / np.sqrt(support_area))

    part_ious = []
    part_cover = []
    for part in parts:
        inter = float((mask & part.mask).sum())
        union = float((mask | part.mask).sum())
        part_ious.append(inter / max(union, 1.0))
        part_cover.append(inter / max(area, 1.0))
    order = np.argsort(part_cover)[::-1] if part_cover else []
    dominant = int(order[0]) if len(order) else -1
    second = int(order[1]) if len(order) > 1 else -1
    crosses = float(sum(1 for value in part_cover if value > 0.1) - 1)
    crosses = max(0.0, crosses)

    pair_gap = 0.0
    if dominant >= 0 and second >= 0:
        dmap = distance_transform_edt(~parts[dominant].mask)
        gap = float(dmap[parts[second].mask].min()) if parts[second].mask.any() else 0.0
        pair_gap = gap / np.sqrt(support_area)

    from scipy.ndimage import binary_erosion

    interior = binary_erosion(support, iterations=4)
    edge = support & ~interior  # interior border band, not the outside
    boundary_touch = float((mask & edge).sum() / max(area, 1e-6))

    scale = 0.5 * (frame.scale_x + frame.scale_y)
    sdf_depth = float(sdf[mask].mean() / max(scale, 1e-6)) if rows.size else 0.0
    centroid = (float(cols.mean()), float(rows.mean())) if rows.size else (0.0, 0.0)
    u = (centroid[0] - frame.center_x) / max(frame.scale_x, 1e-6)
    v = (centroid[1] - frame.center_y) / max(frame.scale_y, 1e-6)
    radial = float(np.hypot(u, v))

    return MaskEvidence(
        area_ratio=area / support_area,
        aspect=max(height, width) / max(1.0, min(height, width)),
        bbox_solidity=area / max(1.0, height * width),
        hull_ratio=_hull_ratio(mask),
        elongation=_principal(mask),
        n_components=len(components),
        max_part_iou=float(max(part_ious)) if part_ious else 0.0,
        crosses_parts=crosses,
        sdf_depth=sdf_depth,
        radial=radial,
        centroid_u=u,
        centroid_v=v,
        boundary_touch=boundary_touch,
        two_blob_gap=two_gap,
        part_pair_gap=pair_gap,
        part_index=dominant,
        second_part_index=second,
    )


def _sigmoid(value: float, centre: float, width: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(value - centre) / max(width, 1e-6))))


def family_scores(evidence: MaskEvidence, texture_mode: bool) -> dict[str, float]:
    if texture_mode:
        scores = {name: 0.05 for name in FAMILIES}
        scores["surface_texture"] = 1.0
        if evidence.area_ratio > 0.12:
            scores["material_loss"] = 0.9
        return scores
    loss = max(
        _sigmoid(evidence.area_ratio, 0.10, 0.04),
        _sigmoid(evidence.max_part_iou, 0.60, 0.08),
    )
    cut = max(
        _sigmoid(evidence.elongation, 4.0, 1.2),
        _sigmoid(-evidence.hull_ratio, -0.35, 0.08),
    )
    displacement = _sigmoid(evidence.two_blob_gap, 0.10, 0.03)
    gain = _sigmoid(evidence.crosses_parts, 0.6, 0.3) * _sigmoid(-evidence.area_ratio, -0.35, 0.1)
    deform = _sigmoid(evidence.boundary_touch, 0.45, 0.12) * _sigmoid(
        -evidence.hull_ratio, -0.75, 0.1
    )
    surface = (
        _sigmoid(-evidence.area_ratio, -0.20, 0.08)
        * _sigmoid(-evidence.elongation, -3.0, 1.0)
        * (1.0 - _sigmoid(evidence.boundary_touch, 0.6, 0.15))
    )
    raw = {
        "surface_texture": float(surface),
        "material_gain": float(gain),
        "material_loss": float(loss),
        "geometric_cut": float(cut),
        "deformation": float(deform),
        "displacement": float(displacement),
    }
    total = sum(raw.values())
    if total <= 1e-6:
        return {name: 1.0 / len(raw) for name in raw}
    return {name: value / total for name, value in raw.items()}


def predict_family(evidence: MaskEvidence, texture_mode: bool) -> tuple[str, float]:
    scores = family_scores(evidence, texture_mode)
    best = max(scores, key=scores.get)
    return best, float(scores[best])


# --------------------------- inversion / rasterisation ---------------------------


def _ellipse(centre_rc: tuple[float, float], height: float, width: float, angle: float) -> np.ndarray:
    import cv2

    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    cv2.ellipse(
        canvas,
        (int(round(centre_rc[1])), int(round(centre_rc[0]))),
        (max(1, int(round(width / 2))), max(1, int(round(height / 2)))),
        float(np.degrees(angle)),
        0,
        360,
        1,
        -1,
    )
    return canvas.astype(bool)


def invert_and_rasterize(
    family: str,
    mask: np.ndarray,
    support: np.ndarray,
    parts: list,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros_like(mask)
    height = float(rows.max() - rows.min() + 1)
    width = float(cols.max() - cols.min() + 1)
    centre = (rows.mean(), cols.mean())

    if family == "surface_texture":
        coords = np.stack([rows, cols], axis=1).astype(np.float64)
        coords -= coords.mean(axis=0)
        if coords.shape[0] > 2:
            cov = np.cov(coords.T)
            values, vectors = np.linalg.eigh(cov)
            angle = float(np.arctan2(vectors[1, -1], vectors[0, -1]))
            # ellipse with the same second moments: semi-axis = 2*sqrt(lambda)
            axis_major = float(2.0 * np.sqrt(max(values[-1], 1e-6)))
            axis_minor = float(2.0 * np.sqrt(max(values[0], 1e-6)))
            if axis_minor > 1.0:
                return _ellipse(centre, 2 * axis_minor, 2 * axis_major, angle) & support
        return _ellipse(centre, height, width, 0.0) & support

    if family == "geometric_cut":
        import cv2

        coords = np.stack([rows, cols], axis=1).astype(np.float64)
        coords -= coords.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(coords.T))
        direction = vectors[:, -1]
        length = float(rows.size / max(1.0, width * 0.5))
        canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        p1 = (int(centre[1] - direction[1] * length / 2), int(centre[0] - direction[0] * length / 2))
        p2 = (int(centre[1] + direction[1] * length / 2), int(centre[0] + direction[0] * length / 2))
        thickness = max(1, int(round(rows.size / max(length, 1.0))))
        cv2.line(canvas, p1, p2, 1, thickness)
        return canvas.astype(bool) & binary_dilation(support, iterations=2)

    if family == "material_loss" and parts:
        best = max(parts, key=lambda part: float((mask & part.mask).sum()))
        ratio = float(mask.sum()) / max(float(best.mask.sum()), 1.0)
        ratio = float(np.clip(ratio, 0.05, 1.0))
        if ratio >= 0.85:
            return np.ascontiguousarray(best.mask & support)
        rows_p, cols_p = np.nonzero(best.mask)
        centre_p = (rows_p.mean(), cols_p.mean())
        h = float(rows_p.max() - rows_p.min() + 1) * np.sqrt(ratio)
        w = float(cols_p.max() - cols_p.min() + 1) * np.sqrt(ratio)
        return _ellipse(centre_p, h, w, 0.0) & best.mask

    if family == "displacement" and parts:
        components = _components(mask)
        if len(components) >= 2:
            first, second = components[0], components[1]
            c1 = np.array(np.nonzero(first), dtype=np.float64).mean(axis=1)
            c2 = np.array(np.nonzero(second), dtype=np.float64).mean(axis=1)
            shift = tuple(int(round(v)) for v in (c2 - c1))
            moved = np.zeros_like(first)
            r0, r1 = max(0, shift[0]), min(CANVAS, CANVAS + shift[0])
            c0, c1i = max(0, shift[1]), min(CANVAS, CANVAS + shift[1])
            if r0 < r1 and c0 < c1i:
                moved[r0:r1, c0:c1i] = first[r0 - shift[0] : r1 - shift[0], c0 - shift[1] : c1i - shift[1]]
            return np.ascontiguousarray(first | moved)

    if family == "material_gain":
        import cv2

        if parts:
            best = max(parts, key=lambda part: float((mask & part.mask).sum()))
            others = [part for part in parts if part is not best]
            if others:
                dmap = distance_transform_edt(~best.mask)
                nearest = min(
                    others,
                    key=lambda part: float(dmap[part.mask].min()) if part.mask.any() else 1e9,
                )
                bridge_area = float(mask.sum())
                rows_n, cols_n = np.nonzero(nearest.mask)
                rows_b0, cols_b0 = np.nonzero(best.mask)
                centre_n = np.array([rows_n.mean(), cols_n.mean()])
                centre_b = np.array([rows_b0.mean(), cols_b0.mean()])
                length = float(np.hypot(*(centre_n - centre_b)))
                width = max(2.0, bridge_area / max(length, 1.0))
                dmap_other = distance_transform_edt(~nearest.mask)
                rows_b, cols_b = np.nonzero(best.mask)
                values = dmap_other[rows_b, cols_b]
                index = int(np.argmin(values))
                start = np.array([rows_b[index], cols_b[index]], dtype=np.float64)
                dmap_best = distance_transform_edt(~best.mask)
                rows_o, cols_o = np.nonzero(nearest.mask)
                values_o = dmap_best[rows_o, cols_o]
                end = np.array([rows_o[int(np.argmin(values_o))], cols_o[int(np.argmin(values_o))]], dtype=np.float64)
                canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
                cv2.line(canvas, (int(start[1]), int(start[0])), (int(end[1]), int(end[0])), 1,
                         max(1, int(round(width))))
                return canvas.astype(bool) & binary_dilation(support, iterations=int(width) + 2)
        return np.ascontiguousarray(binary_dilation(mask, iterations=2) & support)

    # deformation and any fallback: boundary-fitted band around the mask
    return np.ascontiguousarray(binary_dilation(mask, iterations=3) & support)


__all__ = [
    "FAMILIES",
    "MaskEvidence",
    "extract_evidence",
    "family_scores",
    "invert_and_rasterize",
    "predict_family",
]

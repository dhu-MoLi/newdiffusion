"""Structure parser for PSMG (step 0): part decomposition + degradation.

Splits an object support into parts with spectral clustering on frozen DINOv2
patch tokens, computes a texture score, and decides whether the object has
meaningful parts at all. Texture-like classes (carpet, grid, wood, ...) must be
routed to surface editing instead of a part graph -- see docs/psmg_design_v2.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.sparse import csr_matrix

CANVAS = 512


@dataclass(frozen=True)
class PartConfig:
    grid: int = 37
    max_parts: int = 12
    min_part_tokens: int = 6
    knn: int = 10
    laplacian_threshold: float = 0.030
    strong_texture_threshold: float = 0.033
    separability_threshold: float = 1.7
    support_ratio_threshold: float = 0.85
    part_source: str = "spectral"
    sam_min_ratio: float = 0.005
    sam_max_border: float = 0.15
    sam_max_parts: int = 12
    sam_inside_ratio: float = 0.5
    merge_min_ratio: float = 0.01
    assign_remainder: bool = True
    appearance_merge_cosine: float = 0.97


@dataclass
class Part:
    index: int
    mask: np.ndarray
    area_ratio: float
    aspect: float
    solidity: float
    radial: float
    centroid: tuple[float, float]
    boundary_complexity: float
    embedding: np.ndarray


@dataclass
class StructureReport:
    category: str
    image_id: str
    k: int
    texture_score: float
    separability: float
    texture_mode: bool
    support_ratio: float
    parts: list[Part] = field(default_factory=list)
    adjacency: np.ndarray | None = None
    support: np.ndarray | None = None


def _support_grid(support: np.ndarray, grid: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(
        support.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_AREA
    )
    return resized > 0.5


def _l2(tokens: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(tokens, axis=-1, keepdims=True)
    return tokens / np.maximum(norms, 1e-12)


def _affinity(features: np.ndarray, knn: int) -> csr_matrix:
    from sklearn.neighbors import kneighbors_graph

    graph = kneighbors_graph(
        features, n_neighbors=min(knn, features.shape[0] - 1), mode="distance"
    )
    graph = graph.maximum(graph.T)
    sigma = float(np.median(graph.data)) if graph.nnz else 1.0
    sigma = max(sigma, 1e-3)
    graph.data = np.exp(-(graph.data**2) / (2 * sigma**2))
    return csr_matrix(graph)


def _spectrum(affinity: csr_matrix, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.sparse.linalg import eigsh

    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    scaled = affinity.multiply(inv_sqrt[:, None]).multiply(inv_sqrt[None, :])
    laplacian = csr_matrix(np.eye(affinity.shape[0])) - scaled
    k = min(max_k + 1, affinity.shape[0] - 2)
    if k < 2:
        return np.zeros(1), np.zeros((affinity.shape[0], 0))
    values, vectors = eigsh(laplacian, k=k, which="SM")
    order = np.argsort(values)
    return values[order], vectors[:, order]


def _part_mask(label_grid: np.ndarray, label: int, support: np.ndarray) -> np.ndarray:
    import cv2

    grid = label_grid.shape[0]
    binary = (label_grid == label).astype(np.uint8)
    resized = cv2.resize(binary, (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray((resized > 0) & support, dtype=bool)


def _descriptor(mask: np.ndarray, support: np.ndarray, embedding: np.ndarray) -> Part | None:
    rows, cols = np.nonzero(mask)
    if rows.size < 16:
        return None
    import cv2

    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    aspect = max(height, width) / max(1.0, min(height, width))
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = sum(cv2.arcLength(item, True) for item in contours)
    area = float(mask.sum())
    solidity = area / max(1.0, height * width)
    complexity = (perimeter**2) / max(4.0 * np.pi * area, 1e-6)
    support_area = max(1, int(support.sum()))
    centre = (float(cols.mean()), float(rows.mean()))
    rows_s, cols_s = np.nonzero(support)
    radial_scale = 0.5 * (rows_s.max() - rows_s.min() + 1)
    radial = float(
        np.hypot(centre[0] - cols_s.mean(), centre[1] - rows_s.mean()) / max(radial_scale, 1.0)
    )
    return Part(
        index=-1,
        mask=np.ascontiguousarray(mask),
        area_ratio=float(area / support_area),
        aspect=float(aspect),
        solidity=float(solidity),
        radial=float(radial),
        centroid=centre,
        boundary_complexity=float(complexity),
        embedding=embedding,
    )


def _texture_score(rgb: np.ndarray, support: np.ndarray) -> float:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    if not support.any():
        return 0.0
    return float(np.abs(laplacian)[support].mean())


def _border_fraction(segment: np.ndarray, width: int = 3) -> float:
    border = np.zeros_like(segment)
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True
    return float((segment & border).sum() / max(1, int(segment.sum())))


def _sam_masks(
    rgb: np.ndarray,
    support: np.ndarray,
    sam: Any,
    cfg: PartConfig,
) -> list[np.ndarray]:
    raw = sam.generate(rgb)
    support_area = max(1, int(support.sum()))
    kept: list[np.ndarray] = []
    for item in sorted(raw, key=lambda mask: int(mask["area"]), reverse=True):
        segment = np.asarray(item["segmentation"], dtype=bool)
        area = int(segment.sum())
        if area < cfg.sam_min_ratio * support_area:
            continue
        inside = float((segment & support).sum() / max(1, area))
        if inside < cfg.sam_inside_ratio:
            continue
        if _border_fraction(segment) > cfg.sam_max_border:
            continue
        # near-duplicate detection by IoU only: a part nested inside a larger
        # instance is a legitimate part (bottle label inside the bottle), not a
        # duplicate. Containment-based dedup removed exactly the parts we need.
        duplicate = False
        for existing in kept:
            inter = float((segment & existing).sum())
            union = float((segment | existing).sum())
            if union > 0 and inter / union > 0.85:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(np.ascontiguousarray(segment & support))
        if len(kept) >= cfg.sam_max_parts:
            break
    if not kept:
        return []
    import cv2

    union = np.zeros_like(support, dtype=bool)
    for segment in kept:
        union |= segment
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    foreground = (cv2.dilate(union.astype(np.uint8), kernel) > 0) & support
    if foreground.sum() < 0.5 * max(1, int(support.sum())):
        foreground = support
    filtered = []
    for segment in kept:
        area = max(1, int(segment.sum()))
        if float((segment & foreground).sum() / area) < cfg.sam_inside_ratio:
            continue
        filtered.append(np.ascontiguousarray(segment & foreground))
    return filtered or kept


def _mask_embedding(mask: np.ndarray, tokens: np.ndarray) -> np.ndarray | None:
    import cv2

    grid = tokens.shape[0]
    cells = (
        cv2.resize(mask.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_AREA) > 0.5
    )
    if not cells.any():
        return None
    value = np.asarray(tokens[cells].mean(axis=0), dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-9)


def _merge_similar_parts(
    masks: list[np.ndarray], tokens: np.ndarray, cfg: PartConfig
) -> list[np.ndarray]:
    """Merge adjacent parts that look like the same surface.

    SAM splits uniform surfaces (capsule body, toothbrush handle) into several
    slices; adjacent slices with near-identical DINOv2 embeddings are one part.
    """

    import cv2

    if cfg.appearance_merge_cosine >= 1.0 or len(masks) < 2:
        return masks
    embeddings = [_mask_embedding(mask, tokens) for mask in masks]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    while len(masks) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(masks)):
            if embeddings[i] is None:
                continue
            grown = cv2.dilate(masks[i].astype(np.uint8), kernel) > 0
            for j in range(i + 1, len(masks)):
                if embeddings[j] is None or not (grown & masks[j]).any():
                    continue
                cosine = float(embeddings[i] @ embeddings[j])
                if cosine >= cfg.appearance_merge_cosine and (
                    best is None or cosine > best[0]
                ):
                    best = (cosine, i, j)
        if best is None:
            break
        _cosine, i, j = best
        masks[i] = masks[i] | masks[j]
        masks.pop(j)
        embeddings[i] = _mask_embedding(masks[i], tokens)
        embeddings.pop(j)
    return masks


def _postprocess_masks(
    masks: list[np.ndarray], support: np.ndarray, cfg: PartConfig, tokens: np.ndarray
) -> list[np.ndarray]:
    """Merge tiny parts, cap the count, then assign uncovered support pixels.

    SAM instances do not tile the object (measured union coverage 0.735), and
    small parts (bristles, screws) explode the part count. Both are fixed here
    with generic rules -- no per-class thresholds.
    """

    import cv2

    masks = [np.ascontiguousarray(np.asarray(mask, dtype=bool) & support) for mask in masks]
    masks = [mask for mask in masks if mask.any()]
    if not masks:
        return []
    support_area = max(1, int(support.sum()))

    union = np.zeros_like(support, dtype=bool)
    for mask in masks:
        union |= mask
    uncovered = np.ascontiguousarray(support & ~union)
    keep_remainder = bool(
        cfg.assign_remainder
        and int(uncovered.sum()) >= cfg.merge_min_ratio * support_area
    )
    target = max(1, cfg.max_parts - (1 if keep_remainder else 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def merge_smallest_into_neighbor() -> bool:
        areas = np.array([int(mask.sum()) for mask in masks])
        index = int(np.argmin(areas))
        grown = cv2.dilate(masks[index].astype(np.uint8), kernel) > 0
        best_index, best_overlap = -1, -1
        for other, mask in enumerate(masks):
            if other == index:
                continue
            overlap = int((grown & mask).sum())
            if overlap > best_overlap:
                best_overlap, best_index = overlap, other
        if best_index < 0:
            return False
        masks[best_index] = masks[best_index] | masks[index]
        masks.pop(index)
        return True

    while len(masks) > 1:
        areas = np.array([int(mask.sum()) for mask in masks])
        too_small = bool((areas < cfg.merge_min_ratio * support_area).any())
        if not (too_small or len(masks) > target):
            break
        if not merge_smallest_into_neighbor():
            break

    masks = _merge_similar_parts(masks, tokens, cfg)
    while len(masks) > target and len(masks) > 1:
        if not merge_smallest_into_neighbor():
            break

    if cfg.assign_remainder:
        union = np.zeros_like(support, dtype=bool)
        for mask in masks:
            union |= mask
        remainder = np.ascontiguousarray(support & ~union)
        if int(remainder.sum()) >= cfg.merge_min_ratio * support_area:
            masks.append(remainder)
        elif int(remainder.sum()) > 0:
            masks[0] = masks[0] | remainder
    return masks


def _parts_from_masks(
    masks: list[np.ndarray], support: np.ndarray, tokens: np.ndarray
) -> list[Part]:
    import cv2

    grid = tokens.shape[0]
    parts: list[Part] = []
    for mask in masks:
        cells = (
            cv2.resize(mask.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_AREA)
            > 0.5
        )
        embedding = (
            np.asarray(tokens[cells].mean(axis=0), dtype=np.float32)
            if cells.any()
            else np.zeros(tokens.shape[-1], dtype=np.float32)
        )
        descriptor = _descriptor(mask, support, embedding)
        if descriptor is None:
            continue
        descriptor.index = len(parts)
        parts.append(descriptor)
    return parts


def parse_structure(
    rgb: np.ndarray,
    support: np.ndarray,
    valid: np.ndarray,
    tokens: np.ndarray,
    cfg: PartConfig,
    *,
    category: str = "",
    image_id: str = "",
    sam: Any = None,
) -> StructureReport:
    support = np.ascontiguousarray(np.asarray(support, dtype=bool))
    grid_mask = _support_grid(support, cfg.grid)
    n_tokens = int(grid_mask.sum())
    texture = _texture_score(rgb, support)
    support_ratio = float(support.sum() / float(valid.sum() + 1e-6))

    if n_tokens < max(8, cfg.min_part_tokens):
        return StructureReport(
            category=category,
            image_id=image_id,
            k=0,
            texture_score=texture,
            separability=0.0,
            texture_mode=True,
            support_ratio=support_ratio,
        )

    features = _l2(tokens)[grid_mask]
    affinity = _affinity(features, cfg.knn)
    try:
        values, _vectors = _spectrum(affinity, cfg.max_parts)
    except Exception:
        values = np.zeros(2)
    if values.size >= 2:
        gaps = np.diff(values)
        k = int(np.argmax(gaps) + 1)
        separability = float(gaps.max() / max(values[1], 1e-6))
    else:
        k = 1
        separability = 0.0
    k = int(np.clip(k, 1, cfg.max_parts))

    from sklearn.cluster import SpectralClustering

    labels_grid = np.full((cfg.grid, cfg.grid), -1, dtype=np.int32)
    if k == 1:
        labels_grid[grid_mask] = 0
    else:
        model = SpectralClustering(
            n_clusters=k, affinity="precomputed", random_state=0, assign_labels="kmeans"
        )
        labels_grid[grid_mask] = model.fit_predict(affinity)

    masks: list[np.ndarray] = []
    for label in range(k):
        mask = _part_mask(labels_grid, label, support)
        if int(mask.sum()) < 256:
            continue
        masks.append(mask)

    # Texture decision is led by the texture score, with separability / support
    # ratio as guards. K is deliberately NOT used: single-part objects (capsule)
    # otherwise get misclassified as texture. Thresholds are global, not per class.
    texture_mode = bool(
        # branch 1: textured surface with weak part structure
        (
            texture >= cfg.laplacian_threshold
            and separability <= cfg.separability_threshold
        )
        # branch 2: strong texture covering most of the frame (grid-like)
        or (
            texture >= cfg.strong_texture_threshold
            and support_ratio >= cfg.support_ratio_threshold
        )
    )

    if cfg.part_source == "sam":
        masks = [] if (texture_mode or sam is None) else _sam_masks(rgb, support, sam, cfg)

    masks = _postprocess_masks(masks, support, cfg, tokens)
    parts = _parts_from_masks(masks, support, tokens)

    adjacency = np.zeros((len(parts), len(parts)), dtype=np.float32)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            import cv2

            d1 = cv2.distanceTransform((~parts[i].mask).astype(np.uint8), cv2.DIST_L2, 3)
            distance = float(d1[parts[j].mask].min()) if parts[j].mask.any() else float("nan")
            adjacency[i, j] = adjacency[j, i] = distance

    return StructureReport(
        category=category,
        image_id=image_id,
        k=len(parts),
        texture_score=float(texture),
        separability=float(separability),
        texture_mode=bool(texture_mode),
        support_ratio=support_ratio,
        parts=parts,
        adjacency=adjacency,
        support=support,
    )


__all__ = ["Part", "PartConfig", "StructureReport", "parse_structure"]

"""L2: learned placement scorer.

Scores candidate translations on a *normal* target image by how likely that
position is to host a real defect.

Supervision is built without leakage: few-shot defect masks are rasterised in
the object frame and re-projected onto normal images of the same category, so
the features always come from defect-free images (the inference domain).  The
position labels come from the few-shot masks only.

The proposal is a single connected blob on MVTec, so positions -- not
components -- are scored (see docs/l2_site_scorer.md).
"""

from __future__ import annotations

import json
import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import binary_dilation

from cadiff.v06_a.structure import physical_sdf_px_512, support_frame
from cadiff.v06_a.tokencut import tokencut_object_mask
from cadiff.v06_c.visa_pilot import (
    fewshot_identity_records,
    load_visa_identity,
    normal_identity_records,
)
from cadiff.v07.location import ObjectLocationPrior

DINO_DIM = 384
CANVAS = 512


@dataclass(frozen=True)
class ScorerConfig:
    normals_limit: int = 24
    positives_per_pair: int = 32
    negatives_per_pair: int = 320
    min_negative_distance_px: int = 6
    context_sigmas: tuple[int, ...] = (1, 2)
    pca_components: int = 48
    uv_bins: int = 128
    uv_extent: float = 1.5
    seed: int = 20261111


@dataclass
class PairSamples:
    normal_id: str
    defect_type: str
    support: np.ndarray
    positive_xy: np.ndarray
    negative_xy: np.ndarray
    positive_features: np.ndarray
    negative_features: np.ndarray


@dataclass
class ProjectedDataset:
    category: str
    defect_types: tuple[str, ...]
    pairs: list[PairSamples]
    masks: list[tuple[str, np.ndarray, np.ndarray]]  # (image_id, mask, support)

    def pair_ids(self) -> list[str]:
        return [f"{pair.normal_id}|{pair.defect_type}" for pair in self.pairs]


def _uv_grid(
    mask: np.ndarray, support: np.ndarray, bins: int, extent: float
) -> np.ndarray:
    frame = support_frame(support)
    rows, cols = np.nonzero(mask & support & np.isfinite(mask))
    if rows.size == 0:
        return np.zeros((bins, bins), dtype=bool)
    u = (cols - frame.center_x) / frame.scale_x
    v = (rows - frame.center_y) / frame.scale_y
    inside = (np.abs(u) <= extent) & (np.abs(v) <= extent)
    histogram, _, _ = np.histogram2d(
        u[inside],
        v[inside],
        bins=bins,
        range=[[-extent, extent], [-extent, extent]],
    )
    return histogram > 0


def _project_grid(
    grid: np.ndarray, support: np.ndarray, valid: np.ndarray, extent: float
) -> np.ndarray:
    bins = grid.shape[0]
    frame = support_frame(support)
    rows, cols = np.mgrid[0:CANVAS, 0:CANVAS]
    u = (cols - frame.center_x) / frame.scale_x
    v = (rows - frame.center_y) / frame.scale_y
    iu = np.clip(((u + extent) / (2 * extent) * bins).astype(np.int64), 0, bins - 1)
    iv = np.clip(((v + extent) / (2 * extent) * bins).astype(np.int64), 0, bins - 1)
    hit = grid[iv, iu]
    return np.ascontiguousarray(hit & support & valid, dtype=bool)


def _blur_grid(grid_values: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(grid_values, (k, k), sigmaX=sigma, sigmaY=sigma)


def _grid_coords(xy: np.ndarray, grid: int, canvas: int = CANVAS) -> np.ndarray:
    scale = float(grid) / float(canvas)
    return (np.asarray(xy, dtype=np.float32) + 0.5) * scale - 0.5


def _bilinear(grid_values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    size = grid_values.shape[0]
    x = np.clip(coords[:, 0], 0.0, size - 1.0)
    y = np.clip(coords[:, 1], 0.0, size - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, size - 1)
    y1 = np.minimum(y0 + 1, size - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = grid_values[y0, x0] * (1.0 - wx) + grid_values[y0, x1] * wx
    bottom = grid_values[y1, x0] * (1.0 - wx) + grid_values[y1, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _sample_positions(
    mask: np.ndarray, count: int, rng: np.random.RandomState
) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    take = rng.randint(0, rows.size, size=count)
    return np.stack([cols[take], rows[take]], axis=1).astype(np.float32)


def _position_features(
    xy: np.ndarray,
    *,
    tokens: np.ndarray,
    blurred: Sequence[np.ndarray],
    sdf: np.ndarray,
    support: np.ndarray,
    defect_type_index: int,
    type_count: int,
) -> np.ndarray:
    width = DINO_DIM * (1 + len(blurred)) + 5 + type_count
    if xy.shape[0] == 0:
        return np.zeros((0, width), dtype=np.float32)
    coords = _grid_coords(xy, tokens.shape[0])
    parts: list[np.ndarray] = [_bilinear(tokens, coords)]
    for blurred_grid in blurred:
        parts.append(_bilinear(blurred_grid, coords))
    frame = support_frame(support)
    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)
    u = (x - frame.center_x) / frame.scale_x
    v = (y - frame.center_y) / frame.scale_y
    radial = np.sqrt(u * u + v * v)
    scale = 0.5 * (frame.scale_x + frame.scale_y)
    sdf_value = sdf[np.clip(y.astype(int), 0, CANVAS - 1), np.clip(x.astype(int), 0, CANVAS - 1)]
    geometry = np.stack(
        [
            u,
            v,
            radial,
            np.clip(sdf_value / max(scale, 1e-3), -1.5, 1.5),
            np.sqrt((x - frame.center_x) ** 2 + (y - frame.center_y) ** 2)
            / max(scale, 1e-3),
        ],
        axis=1,
    ).astype(np.float32)
    onehot = np.zeros((xy.shape[0], type_count), dtype=np.float32)
    onehot[:, defect_type_index] = 1.0
    return np.concatenate([*parts, geometry, onehot], axis=1)


def _normal_records(inventory: Any, cfg: ScorerConfig) -> list[Any]:
    records = [record for record in normal_identity_records(inventory)]
    records.sort(key=lambda record: record.item_id)
    if len(records) > cfg.normals_limit:
        step = len(records) / float(cfg.normals_limit)
        records = [records[int(index * step)] for index in range(cfg.normals_limit)]
    return records


@dataclass
class NormalView:
    image_id: str
    tokens: np.ndarray
    blurred: list[np.ndarray]
    sdf: np.ndarray
    support: np.ndarray
    valid: np.ndarray


@dataclass
class MaskView:
    image_id: str
    defect_type: str
    mask: np.ndarray
    support: np.ndarray


def prepare_category(
    inventory: Any, category: str, encoder: Any, cfg: ScorerConfig
) -> tuple[list[MaskView], list[NormalView], dict[str, int]]:
    defect_types = tuple(sorted(item.name for item in inventory.types))
    type_index = {name: index for index, name in enumerate(defect_types)}
    masks: list[MaskView] = []
    for record in fewshot_identity_records(inventory):
        if record.category != category or record.defect_type not in type_index:
            continue
        identity = load_visa_identity(record, dataset_root=inventory.source_root)
        if identity.anomaly_gt is None or not identity.anomaly_gt.any():
            continue
        support = tokencut_object_mask(encoder.encode(identity.rgb).tokens, identity.valid)
        if not support.any():
            support = np.ascontiguousarray(identity.valid)
        masks.append(
            MaskView(
                image_id=record.item_id,
                defect_type=record.defect_type,
                mask=np.ascontiguousarray(identity.anomaly_gt, dtype=bool),
                support=support,
            )
        )
    if not masks:
        raise ValueError(f"no few-shot masks for category {category}")
    normals: list[NormalView] = []
    for record in _normal_records(inventory, cfg):
        identity = load_visa_identity(record, dataset_root=inventory.source_root)
        features = encoder.encode(identity.rgb)
        support = tokencut_object_mask(features.tokens, identity.valid)
        if not support.any():
            support = np.ascontiguousarray(identity.valid)
        normals.append(
            NormalView(
                image_id=record.item_id,
                tokens=features.tokens,
                blurred=[
                    _blur_grid(features.tokens, float(sigma))
                    for sigma in cfg.context_sigmas
                ],
                sdf=physical_sdf_px_512(support),
                support=support,
                valid=np.ascontiguousarray(identity.valid, dtype=bool),
            )
        )
    if not normals:
        raise ValueError(f"no normal images for category {category}")
    return masks, normals, type_index


def project_pairs(
    masks: Sequence[MaskView],
    normals: Sequence[NormalView],
    type_index: dict[str, int],
    cfg: ScorerConfig,
    *,
    exclude_mask: str | None = None,
) -> list[PairSamples]:
    usable = [item for item in masks if item.image_id != exclude_mask]
    if not usable:
        return []
    grids: dict[str, np.ndarray] = {}
    for defect_type in sorted({item.defect_type for item in usable}):
        grid = np.zeros((cfg.uv_bins, cfg.uv_bins), dtype=bool)
        for item in usable:
            if item.defect_type == defect_type:
                grid |= _uv_grid(item.mask, item.support, cfg.uv_bins, cfg.uv_extent)
        grids[defect_type] = grid

    pairs: list[PairSamples] = []
    for normal in normals:
        for defect_type, grid in grids.items():
            positive_mask = _project_grid(grid, normal.support, normal.valid, cfg.uv_extent)
            if not positive_mask.any():
                continue
            excluded = binary_dilation(
                positive_mask, iterations=cfg.min_negative_distance_px
            )
            negative_mask = normal.valid & normal.support & ~excluded
            if not negative_mask.any():
                continue
            seed = (
                cfg.seed
                + zlib.crc32(normal.image_id.encode("utf-8")) % 100000
                + zlib.crc32(defect_type.encode("utf-8")) % 1000
            )
            rng = np.random.RandomState(seed)
            positives = _sample_positions(positive_mask, cfg.positives_per_pair, rng)
            negatives = _sample_positions(negative_mask, cfg.negatives_per_pair, rng)
            index = type_index[defect_type]
            pairs.append(
                PairSamples(
                    normal_id=normal.image_id,
                    defect_type=defect_type,
                    support=normal.support,
                    positive_xy=positives,
                    negative_xy=negatives,
                    positive_features=_position_features(
                        positives,
                        tokens=normal.tokens,
                        blurred=normal.blurred,
                        sdf=normal.sdf,
                        support=normal.support,
                        defect_type_index=index,
                        type_count=len(type_index),
                    ),
                    negative_features=_position_features(
                        negatives,
                        tokens=normal.tokens,
                        blurred=normal.blurred,
                        sdf=normal.sdf,
                        support=normal.support,
                        defect_type_index=index,
                        type_count=len(type_index),
                    ),
                )
            )
    return pairs


def build_projected_dataset(
    inventory: Any,
    category: str,
    encoder: Any,
    cfg: ScorerConfig,
    *,
    exclude_mask: str | None = None,
) -> ProjectedDataset:
    masks, normals, type_index = prepare_category(inventory, category, encoder, cfg)
    pairs = project_pairs(masks, normals, type_index, cfg, exclude_mask=exclude_mask)
    if not pairs:
        raise ValueError(f"no projected samples for category {category}")
    return ProjectedDataset(
        category=category,
        defect_types=tuple(sorted(type_index)),
        pairs=pairs,
        masks=[(item.image_id, item.mask, item.support) for item in masks],
    )


class PlacementScorer:
    """Classifier over position features; PCA on the DINO block only."""

    def __init__(self, cfg: ScorerConfig) -> None:
        self.cfg = cfg
        self.pca: Any = None
        self.scaler: Any = None
        self.model: Any = None
        self.dino_dim = 0

    def _reduced(self, x: np.ndarray) -> np.ndarray:
        dino, rest = x[:, : self.dino_dim], x[:, self.dino_dim :]
        return np.concatenate([self.pca.transform(dino), rest], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PlacementScorer":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        dino_dim = DINO_DIM * (1 + len(self.cfg.context_sigmas))
        self.dino_dim = dino_dim
        components = min(self.cfg.pca_components, dino_dim, max(1, x.shape[0] - 1))
        self.pca = PCA(n_components=components, random_state=self.cfg.seed).fit(
            x[:, :dino_dim]
        )
        reduced = self._reduced(x)
        self.scaler = StandardScaler().fit(reduced)
        self.model = LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0
        ).fit(self.scaler.transform(reduced), y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        return self.model.predict_proba(self.scaler.transform(self._reduced(x)))[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "config": self.cfg,
                    "pca": self.pca,
                    "scaler": self.scaler,
                    "model": self.model,
                    "dino_dim": self.dino_dim,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "PlacementScorer":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        scorer = cls(payload["config"])
        scorer.pca = payload["pca"]
        scorer.scaler = payload["scaler"]
        scorer.model = payload["model"]
        scorer.dino_dim = payload["dino_dim"]
        return scorer


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if pos.size == 0 or neg.size == 0:
        return float("nan")
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    scores = np.concatenate([pos, neg])
    if np.all(scores == scores[0]):
        return 0.5
    return float(roc_auc_score(labels, scores))


def _topk_hit(pos: np.ndarray, neg: np.ndarray, fraction: float = 0.1) -> float:
    if pos.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    k = max(1, int(round(fraction * scores.size)))
    threshold = np.sort(scores)[::-1][k - 1]
    return float(np.mean(pos >= threshold))


def lomo_evaluate(
    inventory: Any,
    category: str,
    encoder: Any,
    cfg: ScorerConfig,
) -> dict[str, Any]:
    """Leave-one-mask-out over few-shot masks, on projected normal-image pairs."""

    masks, normals, type_index = prepare_category(inventory, category, encoder, cfg)
    held = sorted((item.image_id, item.defect_type) for item in masks)

    rows: list[dict[str, Any]] = []
    for image_id, defect_type in held:
        pairs = project_pairs(masks, normals, type_index, cfg, exclude_mask=image_id)
        if not pairs:
            continue
        xs = [pair.positive_features for pair in pairs] + [
            pair.negative_features for pair in pairs
        ]
        ys = [np.ones(len(pair.positive_features)) for pair in pairs] + [
            np.zeros(len(pair.negative_features)) for pair in pairs
        ]
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        if x.shape[0] == 0 or len(np.unique(y)) < 2:
            continue
        scorer = PlacementScorer(cfg).fit(x, y)

        selected = [pair for pair in pairs if pair.defect_type == defect_type]
        if not selected:
            continue
        pos = scorer.predict_proba(
            np.concatenate([pair.positive_features for pair in selected])
        )
        neg = scorer.predict_proba(
            np.concatenate([pair.negative_features for pair in selected])
        )

        train_masks = [item for item in masks if item.image_id != image_id]
        masks_by_type: dict[str, list[np.ndarray]] = {}
        supports_by_type: dict[str, list[np.ndarray]] = {}
        for item in train_masks:
            if item.defect_type != defect_type:
                continue
            masks_by_type.setdefault(defect_type, []).append(item.mask)
            supports_by_type.setdefault(defect_type, []).append(item.support)
        if not masks_by_type:
            continue
        prior = ObjectLocationPrior.from_masks(masks_by_type, supports_by_type)

        l1_pos: list[np.ndarray] = []
        l1_neg: list[np.ndarray] = []
        for pair in selected:
            heat = prior.heatmap(pair.support, defect_type)

            def score(xy: np.ndarray) -> np.ndarray:
                if xy.shape[0] == 0:
                    return np.zeros((0,), dtype=np.float64)
                xx = np.clip(xy[:, 0].astype(int), 0, CANVAS - 1)
                yy = np.clip(xy[:, 1].astype(int), 0, CANVAS - 1)
                return np.log(heat[yy, xx] + 1e-12)

            l1_pos.append(score(pair.positive_xy))
            l1_neg.append(score(pair.negative_xy))
        l1_pos_values = np.concatenate(l1_pos)
        l1_neg_values = np.concatenate(l1_neg)

        def _z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
            scale = float(reference.std()) + 1e-9
            return (values - float(reference.mean())) / scale

        ref_l2 = np.concatenate([pos, neg])
        ref_l1 = np.concatenate([l1_pos_values, l1_neg_values])
        z2_pos, z2_neg = _z(pos, ref_l2), _z(neg, ref_l2)
        z1_pos, z1_neg = _z(l1_pos_values, ref_l1), _z(l1_neg_values, ref_l1)
        ensemble = {}
        for weight in (0.3, 0.5, 0.7):
            ensemble[f"w{weight}"] = _auc(
                weight * z2_pos + (1 - weight) * z1_pos,
                weight * z2_neg + (1 - weight) * z1_neg,
            )
        rows.append(
            {
                "held_mask": image_id,
                "defect_type": defect_type,
                "learned_auc": _auc(pos, neg),
                "l1_auc": _auc(l1_pos_values, l1_neg_values),
                "ensemble_auc": ensemble["w0.5"],
                "learned_top10": _topk_hit(pos, neg, 0.1),
                "l1_top10": _topk_hit(l1_pos_values, l1_neg_values, 0.1),
                "ensemble_top10": _topk_hit(
                    0.5 * z2_pos + 0.5 * z1_pos, 0.5 * z2_neg + 0.5 * z1_neg, 0.1
                ),
            }
        )

    return {
        "schema": "v07_placement_scorer_lomo_v2",
        "category": category,
        "held_out": len(rows),
        "learned_auc_mean": float(np.nanmean([row["learned_auc"] for row in rows])),
        "l1_auc_mean": float(np.nanmean([row["l1_auc"] for row in rows])),
        "learned_top10_mean": float(np.nanmean([row["learned_top10"] for row in rows])),
        "l1_top10_mean": float(np.nanmean([row["l1_top10"] for row in rows])),
        "ensemble_auc_mean": float(np.nanmean([row["ensemble_auc"] for row in rows])),
        "ensemble_top10_mean": float(np.nanmean([row["ensemble_top10"] for row in rows])),
        "config": {
            "normals_limit": cfg.normals_limit,
            "positives_per_pair": cfg.positives_per_pair,
            "negatives_per_pair": cfg.negatives_per_pair,
            "pca_components": cfg.pca_components,
            "context_sigmas": list(cfg.context_sigmas),
            "uv_bins": cfg.uv_bins,
            "uv_extent": cfg.uv_extent,
        },
        "rows": rows,
    }


class _BoundTargetPrior:
    """Per-target view: dense learned/fused score map for one normal image."""

    def __init__(
        self,
        owner: "LearnedPlacementPrior",
        rgb: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        self.owner = owner
        self.rgb = np.ascontiguousarray(rgb)
        self.valid = np.ascontiguousarray(valid, dtype=bool)
        self._tokens: np.ndarray | None = None
        self._blurred: list[np.ndarray] | None = None
        self._cache: dict[str, np.ndarray] = {}

    def _encode(self) -> tuple[np.ndarray, list[np.ndarray]]:
        if self._tokens is None:
            features = self.owner.encoder.encode(self.rgb)
            self._tokens = features.tokens
            self._blurred = [
                _blur_grid(features.tokens, float(sigma))
                for sigma in self.owner.config.context_sigmas
            ]
        return self._tokens, self._blurred or []

    def heatmap(self, support: np.ndarray, defect_type: str | None = None) -> np.ndarray:
        key = str(defect_type)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        tokens, blurred = self._encode()
        grid = tokens.shape[0]
        rows, cols = np.mgrid[0:grid, 0:grid]
        xy = np.stack(
            [
                (cols.reshape(-1) + 0.5) * CANVAS / grid,
                (rows.reshape(-1) + 0.5) * CANVAS / grid,
            ],
            axis=1,
        ).astype(np.float32)
        sdf = physical_sdf_px_512(support)
        type_index = self.owner.type_index.get(str(defect_type), 0)
        features = _position_features(
            xy,
            tokens=tokens,
            blurred=blurred,
            sdf=sdf,
            support=support,
            defect_type_index=type_index,
            type_count=len(self.owner.type_index),
        )
        probability = self.owner.scorer.predict_proba(features).reshape(grid, grid)
        learned = (probability - probability.mean()) / (probability.std() + 1e-9)
        score = learned
        histogram = self.owner.histogram
        if histogram is not None and defect_type is not None:
            density = histogram.heatmap(support, defect_type)
            log_density = np.log(density + 1e-12)
            import cv2

            coarse = cv2.resize(
                log_density,
                (grid, grid),
                interpolation=cv2.INTER_AREA,
            )
            coarse = (coarse - coarse.mean()) / (coarse.std() + 1e-9)
            weight = float(self.owner.learned_weight)
            score = weight * learned + (1.0 - weight) * coarse
        import cv2

        dense = cv2.resize(score, (CANVAS, CANVAS), interpolation=cv2.INTER_CUBIC)
        weights = np.ascontiguousarray(np.exp(np.clip(dense, -12.0, 12.0)), dtype=np.float32)
        self._cache[key] = weights
        return weights

    def score_mask(
        self, mask: np.ndarray, support: np.ndarray, defect_type: str | None = None
    ) -> float:
        heat = self.heatmap(support, defect_type)
        binary = np.asarray(mask, dtype=bool) & np.asarray(support, dtype=bool)
        if not binary.any():
            return float("nan")
        return float(np.log(heat[binary] + 1e-12).mean())


class LearnedPlacementPrior:
    """L2 scorer wrapped as a location prior (fused with L1 when provided)."""

    def __init__(
        self,
        scorer: PlacementScorer,
        encoder: Any,
        defect_types: Sequence[str],
        *,
        histogram: Any = None,
        learned_weight: float = 0.5,
    ) -> None:
        self.scorer = scorer
        self.encoder = encoder
        self.config = scorer.cfg
        names = tuple(sorted(defect_types))
        self.type_index = {name: index for index, name in enumerate(names)}
        self.histogram = histogram
        self.learned_weight = float(learned_weight)

    def bind(self, rgb: np.ndarray, valid: np.ndarray) -> _BoundTargetPrior:
        return _BoundTargetPrior(self, rgb, valid)


__all__ = [
    "LearnedPlacementPrior",
    "MaskView",
    "NormalView",
    "PairSamples",
    "PlacementScorer",
    "ProjectedDataset",
    "ScorerConfig",
    "build_projected_dataset",
    "lomo_evaluate",
    "prepare_category",
    "project_pairs",
]

"""Diagnostic: official H vs log-uniform scale + clip-to-site.

One variable: how scale is sampled and what happens when T(g) does not fit.
Official arm = instantiate_on_component (shrink grid, extra-shrink to 0.01,
T(g)subseteq site). Diagnostic arm draws f from few-shot log-area range, then
clips T(g) cap site and never extra-shrinks. Same largest a2 CC. Uses current
TokenCut S_x^0 (relative-area union). Not generate(), not SAM, can deferred.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

from cadiff.provenance import write_json_atomic
from cadiff.v06_b.a2_component_place import (
    ANGLES_DEG,
    MIN_SHAPE_PIXELS,
    ComponentPlacement,
    EmptyFeasiblePlacement,
    instantiate_on_component,
    label_a2_components,
    pick_largest_site,
    rotate_binary_mask,
    sample_area_factor,
)
from cadiff.v06_b.a2_shape_instantiate import (
    _contained_output,
    _fail_rgb,
    _mask_u8,
    _overlay,
    _save_sheet,
    object_scale,
    scale_binary_mask,
)

SCHEMA = "v06_b_h_scale_clip_diag_v1"
FORBIDDEN_CATEGORIES = frozenset({"can"})
FLOW_COLUMNS = (
    "source",
    "source_gt",
    "site",
    "official",
    "unclipped",
    "clipped",
    "overlay",
)
COLUMN_MEANINGS = {
    "source": "few-shot RGB，红叠加为官方 GT",
    "source_gt": "few-shot 官方 GT",
    "site": "最大 8 连通 a2 部件（当前 TokenCut S_x^0）",
    "official": "正式 H：可行缩小网格均匀抽，T(g)subseteq site",
    "unclipped": "诊断臂放大后的 T(g)，可伸出 site",
    "clipped": "诊断臂 m_final = T(g) cap site；无 0.01 额外缩小",
    "overlay": "绿=site，黄=正式 H，品红=剪交，青=被剪掉的伸出",
}
MapsFn = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ClipPlacement:
    mask: NDArray[np.bool_]
    unclipped: NDArray[np.bool_]
    extra: NDArray[np.bool_]
    site: NDArray[np.bool_]
    site_index: int
    scale: float
    factor: float
    angle_deg: int
    top: int
    left: int
    score: float
    feasible_count: int
    source_gt_pixels: int
    placed_pixels: int
    used_proposal_floor: bool = False


def require_frozen_identities(
    frozen: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[str, ...]:
    identities = tuple(str(item["identity"]) for item in frozen)
    if not identities:
        raise ValueError("frozen identity list is empty")
    for item in frozen:
        category = str(item["category"])
        if category in FORBIDDEN_CATEGORIES:
            raise ValueError("can is deferred; scale-clip preview does not include MVTec AD 2")
    return identities


def place_shape_clip_site(
    shape: np.ndarray,
    site: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    min_pixels: int,
) -> ClipPlacement:
    template = np.asarray(shape, dtype=bool)
    allowed = np.asarray(site, dtype=bool)
    scores = np.asarray(similarity, dtype=np.float32)
    if template.shape != allowed.shape or allowed.shape != scores.shape:
        raise ValueError("shape, site, and similarity must share one grid")
    if type(min_pixels) is not int or min_pixels < 1:
        raise ValueError("min_pixels must be a positive integer")
    rows, cols = np.nonzero(template)
    if rows.size == 0:
        raise EmptyFeasiblePlacement("scaled GT is empty")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = np.ascontiguousarray(template[top:bottom, left:right])
    height, width = crop.shape
    if height > allowed.shape[0] or width > allowed.shape[1]:
        raise EmptyFeasiblePlacement("scaled GT is larger than the canvas")
    kernel = np.ascontiguousarray(crop, dtype=np.float64)
    inside = correlate(allowed.astype(np.float64), kernel, mode="valid", method="fft")
    feasible = inside >= float(min_pixels)
    if not feasible.any():
        raise EmptyFeasiblePlacement("clip remainder is below the area floor")
    score_map = correlate(scores.astype(np.float64), kernel, mode="valid", method="fft") / float(
        crop.sum()
    )
    weights = np.maximum(score_map[feasible], 0.0)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    ys, xs = np.nonzero(feasible)
    rng = np.random.RandomState(seed)
    index = int(rng.choice(ys.size, p=probs))
    paste_top = int(ys[index])
    paste_left = int(xs[index])
    unclipped = np.zeros_like(template, dtype=bool)
    unclipped[paste_top : paste_top + height, paste_left : paste_left + width] = crop
    clipped = np.ascontiguousarray(unclipped & allowed)
    extra = np.ascontiguousarray(unclipped & ~allowed)
    if int(clipped.sum()) < min_pixels:
        raise EmptyFeasiblePlacement("clip remainder is below the area floor")
    return ClipPlacement(
        mask=clipped,
        unclipped=np.ascontiguousarray(unclipped),
        extra=extra,
        site=np.ascontiguousarray(allowed),
        site_index=-1,
        scale=1.0,
        factor=1.0,
        angle_deg=0,
        top=paste_top,
        left=paste_left,
        score=float(score_map[paste_top, paste_left]),
        feasible_count=int(feasible.sum()),
        source_gt_pixels=int(template.sum()),
        placed_pixels=int(clipped.sum()),
    )


def instantiate_scale_clip(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    fewshot_areas: tuple[int, ...] | list[int],
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
) -> tuple[NDArray[np.bool_], ClipPlacement]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(min_shape_pixels) is not int or min_shape_pixels < 1:
        raise ValueError("min_shape_pixels must be a positive integer")
    _labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
    if not sites:
        raise EmptyFeasiblePlacement("a2 has no component")
    largest = pick_largest_site(sites)
    gt = np.asarray(source_gt, dtype=bool)
    rng = np.random.RandomState(seed)
    factor = sample_area_factor(int(gt.sum()), fewshot_areas, seed=seed, rng=rng)
    scale = object_scale(source_support, target_support) * factor
    canvas_size = int(gt.shape[0])
    try:
        scaled = scale_binary_mask(gt, scale, size=canvas_size)
    except EmptyFeasiblePlacement as exc:
        raise EmptyFeasiblePlacement("scaled GT does not fit the canvas") from exc
    hits: list[tuple[int, NDArray[np.bool_], ClipPlacement]] = []
    for angle in ANGLES_DEG:
        try:
            template = rotate_binary_mask(scaled, angle)
            placed = place_shape_clip_site(
                template,
                largest.mask,
                similarity,
                seed=seed,
                min_pixels=min_shape_pixels,
            )
        except EmptyFeasiblePlacement:
            continue
        hits.append((int(angle), np.ascontiguousarray(template), placed))
    if not hits:
        raise EmptyFeasiblePlacement("clip remainder is below the area floor")
    chosen_angle, template, placed = hits[int(rng.randint(0, len(hits)))]
    if (placed.mask & ~largest.mask).any() or (placed.mask & ~np.asarray(proposal, dtype=bool)).any():
        raise RuntimeError("scale-clip placement escaped the chosen a2 CC")
    return template, replace(
        placed,
        site=largest.mask,
        site_index=largest.index,
        scale=float(scale),
        factor=float(factor),
        angle_deg=chosen_angle,
        source_gt_pixels=int(gt.sum()),
        used_proposal_floor=False,
    )


def _row_arrays(
    *,
    source_rgb: np.ndarray,
    source_gt: np.ndarray,
    normal_rgb: np.ndarray,
    site: np.ndarray,
    official: np.ndarray,
    unclipped: np.ndarray,
    clipped: np.ndarray,
) -> dict[str, np.ndarray]:
    extra = np.ascontiguousarray(unclipped & ~clipped)
    overlay = _overlay(normal_rgb, site, (40, 200, 80), 0.35)
    overlay = _overlay(overlay, official, (240, 220, 40), 0.55)
    overlay = _overlay(overlay, clipped, (220, 70, 180), 0.55)
    overlay = _overlay(overlay, extra, (40, 190, 210), 0.70)
    return {
        "source": _overlay(source_rgb, source_gt, (255, 40, 40), 0.45),
        "source_gt": _mask_u8(source_gt),
        "site": _mask_u8(site),
        "official": _mask_u8(official),
        "unclipped": _mask_u8(unclipped),
        "clipped": _mask_u8(clipped),
        "overlay": overlay,
    }


def run_h_scale_clip_preview(
    *,
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    output: Path,
    maps_fn: MapsFn,
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
    tile: int = 160,
) -> dict[str, Any]:
    frozen = tuple(rows)
    require_frozen_identities(frozen)
    destination = _contained_output(output)
    destination.mkdir(parents=True)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "diagnostic_only": True,
            "not_official_generate": True,
            "frozen_before_images": True,
            "identities": [
                {
                    "category": str(row["category"]),
                    "item_id": str(row["item_id"]),
                    "identity": str(row["identity"]),
                }
                for row in frozen
            ],
        },
    )
    sheets: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    all_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    for row in frozen:
        identity = str(row["identity"])
        category = str(row["category"])
        maps = maps_fn(row)
        source_gt = np.asarray(maps["source_gt"], dtype=bool)
        proposal = np.asarray(maps["proposal"], dtype=bool)
        similarity = np.asarray(maps["similarity"], dtype=np.float32)
        seed = int(maps["seed"])
        empty = np.zeros(source_gt.shape, dtype=bool)
        official_mask = empty
        clipped = empty
        unclipped = empty
        site = empty
        official_info: dict[str, Any] = {"failed": True}
        diag_info: dict[str, Any] = {"failed": True}
        try:
            _official_t, official = instantiate_on_component(
                source_gt,
                maps["source_support"],
                maps["target_support"],
                proposal,
                similarity,
                seed=seed,
                min_shape_pixels=min_shape_pixels,
                allow_proposal_floor=True,
            )
            official_mask = np.asarray(official.mask, dtype=bool)
            site = np.asarray(official.site, dtype=bool)
            official_info = {
                "failed": False,
                "pixels": int(official.mask.sum()),
                "scale": float(official.scale),
                "angle_deg": int(official.angle_deg),
                "used_proposal_floor": bool(official.used_proposal_floor),
                "used_site_fallback": bool(official.used_site_fallback),
            }
        except EmptyFeasiblePlacement as exc:
            official_info["failure_reason"] = str(exc)
        try:
            _diag_t, diag = instantiate_scale_clip(
                source_gt,
                maps["source_support"],
                maps["target_support"],
                proposal,
                similarity,
                seed=seed,
                fewshot_areas=maps["fewshot_areas"],
                min_shape_pixels=min_shape_pixels,
            )
            clipped = np.asarray(diag.mask, dtype=bool)
            unclipped = np.asarray(diag.unclipped, dtype=bool)
            site = np.asarray(diag.site, dtype=bool)
            diag_info = {
                "failed": False,
                "pixels": int(diag.mask.sum()),
                "unclipped_pixels": int(diag.unclipped.sum()),
                "extra_pixels": int(diag.extra.sum()),
                "scale": float(diag.scale),
                "factor": float(diag.factor),
                "angle_deg": int(diag.angle_deg),
                "used_proposal_floor": False,
            }
        except EmptyFeasiblePlacement as exc:
            diag_info["failure_reason"] = str(exc)
        title = (
            f"{identity}  off={official_info.get('pixels', 0)} "
            f"clip={diag_info.get('pixels', 0)} f={diag_info.get('factor', None)}"
        )
        arrays = _row_arrays(
            source_rgb=np.asarray(maps["source_rgb"], dtype=np.uint8),
            source_gt=source_gt,
            normal_rgb=np.asarray(maps["normal_rgb"], dtype=np.uint8),
            site=site,
            official=official_mask,
            unclipped=unclipped,
            clipped=clipped,
        )
        if official_info["failed"] and diag_info["failed"]:
            arrays["overlay"] = _fail_rgb(np.asarray(maps["normal_rgb"]).shape)
        sheets[category].append((title, arrays))
        all_rows.append((title, arrays))
        records.append(
            {
                "identity": identity,
                "category": category,
                "official": official_info,
                "scale_clip": diag_info,
            }
        )
    for category, panels in sheets.items():
        _save_sheet(destination / f"sheet_{category}.png", panels, FLOW_COLUMNS, tile=tile)
    _save_sheet(destination / "all.png", all_rows, FLOW_COLUMNS, tile=tile)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "one_variable": "log_uniform_fewshot_area_then_clip_to_site",
        "official_arm": "instantiate_on_component",
        "column_meanings": COLUMN_MEANINGS,
        "n": len(records),
        "rows": records,
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def _fewshot_areas(sources: tuple[Any, ...]) -> tuple[int, ...]:
    areas: list[int] = []
    for item in sources:
        if item.anomaly_gt is None:
            raise ValueError("few-shot source is missing GT")
        areas.append(int(np.asarray(item.anomaly_gt, dtype=bool).sum()))
    return tuple(areas)


def run_h_scale_clip_preview_from_cli(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cable_experiment: Path,
    output: Path,
    repository_root: Path,
    device: str | None = None,
) -> dict[str, Any]:
    from cadiff.v06_a.a2 import A2ProposalGenerator, load_a2_generator
    from cadiff.v06_a.config import load_v06_a_experiment
    from cadiff.v06_a.data import load_identity_record
    from cadiff.v06_a.identities import load_v06_a_identity_manifest
    from cadiff.v06_a.structure import estimate_normal_structure
    from cadiff.v06_a.tokencut import load_official_dinov2_encoder
    from cadiff.v06_atlas_diag.cable_a import _lookup_p1_target, build_cable_a_slots
    from cadiff.v06_atlas_diag.extra_a import (
        ExtraAStructureConfig,
        build_extra_preview_slots,
        inventory_visa,
        load_extra_identity,
        select_visa_categories,
    )
    from cadiff.v06_c.visa_pilot import EXPECTED

    cwd = repository_root.resolve()
    experiment_path = cable_experiment if cable_experiment.is_absolute() else cwd / cable_experiment
    experiment = load_v06_a_experiment(experiment_path)
    manifest = load_v06_a_identity_manifest(cwd / experiment.identity_manifest, experiment=experiment)
    cable_slots = build_cable_a_slots(manifest, experiment=experiment)
    if tuple(select_visa_categories(visa_root)) != EXPECTED:
        raise ValueError("VisA extra-A selection drifted")
    visa_packs = tuple(inventory_visa(visa_root, name) for name in EXPECTED)
    visa_jobs = tuple((pack, slot) for pack in visa_packs for slot in build_extra_preview_slots(pack))
    identities: list[dict[str, Any]] = []
    for slot in cable_slots:
        identities.append(
            {
                "category": "cable",
                "item_id": f"{slot.defect_type}-{slot.normal_id}-{slot.seed}",
                "identity": f"cable/{slot.defect_type}-{slot.normal_id}-{slot.seed}",
                "dataset": "cable",
                "defect_type": slot.defect_type,
                "target_id": slot.normal_id,
                "source_id": slot.source_id,
                "seed": slot.seed,
            }
        )
    for pack, slot in visa_jobs:
        identities.append(
            {
                "category": pack.name,
                "item_id": f"{slot.defect_type}-{slot.normal_id}-{slot.seed}",
                "identity": f"visa/{pack.name}/{slot.defect_type}-{slot.normal_id}-{slot.seed}",
                "dataset": "visa",
                "defect_type": slot.defect_type,
                "target_id": slot.normal_id,
                "source_id": slot.source_id,
                "seed": slot.seed,
            }
        )
    require_frozen_identities(identities)

    encoder = load_official_dinov2_encoder(device=device)
    cable_generator = load_a2_generator(
        experiment_path=experiment_path,
        dataset_root=mvtec_root,
        device=device,
        encoder=encoder,
    )
    visa_generators: dict[str, A2ProposalGenerator] = {}
    visa_areas: dict[str, dict[str, tuple[int, ...]]] = {}
    visa_structures: dict[tuple[str, str], Any] = {}
    for pack in visa_packs:
        config = ExtraAStructureConfig(category=pack.name, defect_types=pack.defect_types)
        sources = {
            defect_type: tuple(
                load_extra_identity(record, dataset_root=pack.root) for record in rows
            )
            for defect_type, rows in pack.fewshot.items()
        }
        visa_generators[pack.name] = A2ProposalGenerator(
            sources,
            config=config,
            identity_manifest=None,
            encoder=encoder,
        )
        visa_areas[pack.name] = {
            defect_type: _fewshot_areas(items) for defect_type, items in sources.items()
        }
    cable_areas = {
        defect_type: _fewshot_areas(cable_generator._sources[defect_type])
        for defect_type in experiment.defect_types
    }
    cable_structures: dict[str, Any] = {}

    def maps_fn(row: Mapping[str, Any]) -> dict[str, Any]:
        dataset = str(row["dataset"])
        defect_type = str(row["defect_type"])
        seed = int(row["seed"])
        if dataset == "cable":
            target_id = str(row["target_id"])
            if target_id not in cable_structures:
                target_record = _lookup_p1_target(manifest, target_id)
                loaded_target = load_identity_record(
                    target_record,
                    dataset_root=mvtec_root,
                    config=experiment,
                )
                cable_structures[target_id] = estimate_normal_structure(
                    loaded_target, config=experiment
                )
            target = cable_structures[target_id]
            generator = cable_generator
            areas = cable_areas[defect_type]
            source = generator.select_source(defect_type, seed)
        else:
            pack = next(item for item in visa_packs if item.name == row["category"])
            config = ExtraAStructureConfig(category=pack.name, defect_types=pack.defect_types)
            key = (pack.name, str(row["target_id"]))
            if key not in visa_structures:
                target_record = next(
                    record for record in pack.normals if record.item_id == row["target_id"]
                )
                visa_structures[key] = estimate_normal_structure(
                    load_extra_identity(target_record, dataset_root=pack.root),
                    config=config,
                )
            target = visa_structures[key]
            generator = visa_generators[pack.name]
            areas = visa_areas[pack.name][defect_type]
            source = generator.select_source(defect_type, seed)
            if source.record.item_id != row["source_id"]:
                raise RuntimeError(
                    f"preview source drifted: {row['identity']} expected {row['source_id']}"
                )
        a2 = generator.generate(target, defect_type, seed)
        if a2.source_gt is None or a2.source_support is None or a2.target_support is None:
            raise ValueError("official a2 maps required")
        if a2.similarity is None:
            raise ValueError("official a2 similarity required")
        return {
            "source_rgb": np.asarray(source.rgb, dtype=np.uint8).copy(),
            "source_gt": np.asarray(a2.source_gt, dtype=bool),
            "normal_rgb": np.asarray(target.normal.rgb, dtype=np.uint8).copy(),
            "source_support": np.asarray(a2.source_support, dtype=bool),
            "target_support": np.ascontiguousarray(a2.target_support & target.valid, dtype=bool),
            "proposal": np.asarray(a2.proposal, dtype=bool),
            "similarity": np.asarray(a2.similarity, dtype=np.float32),
            "fewshot_areas": areas,
            "seed": seed,
        }

    return run_h_scale_clip_preview(
        rows=identities,
        output=output,
        maps_fn=maps_fn,
        min_shape_pixels=MIN_SHAPE_PIXELS,
    )

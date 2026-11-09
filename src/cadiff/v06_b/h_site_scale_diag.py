"""Diagnostic: TokenCut-CC site sample + fit-then-clip scale, 100-mask coverage.

Official arm = instantiate_on_component (largest a2 CC, shrink grid).
Diagnostic arm = uniform among proposal∩TokenCut-CC (≥64), log-uniform scale,
try T⊆site then max-overlap clip. Same frozen 46 identities. A maps once;
H drawn n_draws times. Masks/heatmaps only, not R, not generate(). can deferred.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cadiff.provenance import write_json_atomic
from cadiff.v06_b.a2_component_place import (
    MIN_SHAPE_PIXELS,
    ComponentPlacement,
    EmptyFeasiblePlacement,
    TokenCutSite,
    candidate_tokencut_sites,
    instantiate_on_component,
    instantiate_site_scale,
    pick_tokencut_site,
    place_shape_max_overlap_clip,
)
from cadiff.v06_b.a2_shape_instantiate import (
    _contained_output,
    _mask_u8,
    _overlay,
    _save_sheet,
)
from cadiff.v06_c.tokencut_normal_preview import FORBIDDEN_CATEGORIES, color_ranked_components, rank_components

SCHEMA = "v06_b_h_site_scale_coverage_v1"
N_DRAWS = 100
FLOW_COLUMNS = (
    "rgb",
    "tokencut_cc",
    "proposal",
    "official_heat",
    "diag_heat",
    "overlay",
)
COLUMN_MEANINGS = {
    "rgb": "目标正常图",
    "tokencut_cc": "当前 TokenCut S_x^0 连通块",
    "proposal": "当前 a2 proposal",
    "official_heat": "正式 H 100 次 m_final 叠加热图",
    "diag_heat": "诊断 H 100 次 m_final 叠加热图（抽 TokenCut 块 + 尺度）",
    "overlay": "黄=正式并, 品红=诊断并",
}
MapsFn = Callable[[Mapping[str, Any]], dict[str, Any]]
SiteScalePlacement = ComponentPlacement


def require_frozen_identities(
    frozen: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[str, ...]:
    identities = tuple(str(item["identity"]) for item in frozen)
    if not identities:
        raise ValueError("frozen identity list is empty")
    for item in frozen:
        if str(item["category"]) in FORBIDDEN_CATEGORIES:
            raise ValueError("can is deferred; site-scale coverage does not include MVTec AD 2")
    return identities


def _heat_u8(count: np.ndarray, n: int) -> np.ndarray:
    scale = float(max(n, 1))
    norm = np.clip(np.asarray(count, dtype=np.float32) / scale, 0.0, 1.0)
    image = np.zeros((*norm.shape, 3), dtype=np.uint8)
    image[..., 0] = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip(norm * 80.0, 0, 255).astype(np.uint8)
    return image


def _row_arrays(
    *,
    rgb: np.ndarray,
    object_mask: np.ndarray,
    proposal: np.ndarray,
    official_count: np.ndarray,
    diag_count: np.ndarray,
    n_draws: int,
) -> dict[str, np.ndarray]:
    official_union = official_count > 0
    diag_union = diag_count > 0
    overlay = _overlay(rgb, official_union, (240, 220, 40), 0.45)
    overlay = _overlay(overlay, diag_union, (220, 70, 180), 0.45)
    return {
        "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
        "tokencut_cc": color_ranked_components(rank_components(object_mask), object_mask.shape),
        "proposal": _mask_u8(proposal),
        "official_heat": _heat_u8(official_count, n_draws),
        "diag_heat": _heat_u8(diag_count, n_draws),
        "overlay": overlay,
    }


def run_h_site_scale_coverage(
    *,
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    output: Path,
    maps_fn: MapsFn,
    n_draws: int = N_DRAWS,
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
    tile: int = 160,
) -> dict[str, Any]:
    if type(n_draws) is not int or n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
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
            "not_multi_candidate_pick": True,
            "frozen_before_images": True,
            "n_draws": n_draws,
            "identities": [
                {"category": str(row["category"]), "identity": str(row["identity"])}
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
        obj = np.asarray(maps["target_support"], dtype=bool)
        proposal = np.asarray(maps["proposal"], dtype=bool)
        similarity = np.asarray(maps["similarity"], dtype=np.float32)
        base_seed = int(maps["seed"])
        official_count = np.zeros(proposal.shape, dtype=np.int32)
        diag_count = np.zeros(proposal.shape, dtype=np.int32)
        official_ok = 0
        diag_ok = 0
        diag_sites: dict[int, int] = defaultdict(int)
        diag_clip = 0
        for draw in range(n_draws):
            draw_seed = int(base_seed) + draw
            try:
                _t, official = instantiate_on_component(
                    source_gt,
                    maps["source_support"],
                    obj,
                    proposal,
                    similarity,
                    seed=draw_seed,
                    min_shape_pixels=min_shape_pixels,
                    allow_proposal_floor=True,
                )
                official_count += np.asarray(official.mask, dtype=np.int32)
                official_ok += 1
            except EmptyFeasiblePlacement:
                pass
            try:
                _t, diag = instantiate_site_scale(
                    source_gt,
                    maps["source_support"],
                    obj,
                    proposal,
                    similarity,
                    seed=draw_seed,
                    fewshot_areas=maps["fewshot_areas"],
                    min_shape_pixels=min_shape_pixels,
                )
                diag_count += np.asarray(diag.mask, dtype=np.int32)
                diag_ok += 1
                diag_sites[int(diag.site_index)] += 1
                diag_clip += int(diag.used_clip)
            except EmptyFeasiblePlacement:
                pass
        proposal_px = int(proposal.sum())
        title = (
            f"{identity}  off={official_ok}/{n_draws} "
            f"diag={diag_ok}/{n_draws} sites={dict(diag_sites)}"
        )
        arrays = _row_arrays(
            rgb=np.asarray(maps["rgb"], dtype=np.uint8),
            object_mask=obj,
            proposal=proposal,
            official_count=official_count,
            diag_count=diag_count,
            n_draws=n_draws,
        )
        sheets[category].append((title, arrays))
        all_rows.append((title, arrays))
        records.append(
            {
                "identity": identity,
                "category": category,
                "n_draws": n_draws,
                "official_ok": official_ok,
                "diag_ok": diag_ok,
                "diag_clip": diag_clip,
                "diag_site_counts": {str(key): int(value) for key, value in sorted(diag_sites.items())},
                "official_union_on_proposal": float((official_count > 0).sum() / proposal_px)
                if proposal_px
                else 0.0,
                "diag_union_on_proposal": float((diag_count > 0).sum() / proposal_px)
                if proposal_px
                else 0.0,
            }
        )
    for category, panels in sheets.items():
        _save_sheet(destination / f"sheet_{category}.png", panels, FLOW_COLUMNS, tile=tile)
    _save_sheet(destination / "all.png", all_rows, FLOW_COLUMNS, tile=tile)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_multi_candidate_pick": True,
        "one_variable": "tokencut_cc_site_plus_log_uniform_scale_fit_then_clip",
        "column_meanings": COLUMN_MEANINGS,
        "n": len(records),
        "n_draws": n_draws,
        "rows": records,
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def _fewshot_areas(sources: tuple[Any, ...]) -> tuple[int, ...]:
    from cadiff.v06_b.h_scale_clip_diag import _fewshot_areas as _areas

    return _areas(sources)


def run_h_site_scale_coverage_from_cli(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cable_experiment: Path,
    output: Path,
    repository_root: Path,
    device: str | None = None,
    n_draws: int = N_DRAWS,
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
        del source
        return {
            "rgb": np.asarray(target.normal.rgb, dtype=np.uint8).copy(),
            "source_gt": np.asarray(a2.source_gt, dtype=bool),
            "source_support": np.asarray(a2.source_support, dtype=bool),
            "target_support": np.ascontiguousarray(a2.target_support & target.valid, dtype=bool),
            "proposal": np.asarray(a2.proposal, dtype=bool),
            "similarity": np.asarray(a2.similarity, dtype=np.float32),
            "fewshot_areas": areas,
            "seed": seed,
        }

    return run_h_site_scale_coverage(
        rows=identities,
        output=output,
        maps_fn=maps_fn,
        n_draws=n_draws,
        min_shape_pixels=MIN_SHAPE_PIXELS,
    )

"""Diagnostic: current a2 proposal vs TokenCut object CCs.

Look-then-judge only. Same VisA extra identities as h_scale_clip (14 rows).
Shows whether q70 proposal lands on each relative-area TokenCut instance.
Not generate(), not official H, not SAM. can deferred.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from cadiff.provenance import write_json_atomic
from cadiff.v06_b.a2_shape_instantiate import _contained_output, _mask_u8, _overlay, _save_sheet
from cadiff.v06_c.tokencut_normal_preview import (
    FORBIDDEN_CATEGORIES,
    color_ranked_components,
    rank_components,
)

SCHEMA = "v06_c_proposal_on_tokencut_cc_v1"
MIN_HIT_PIXELS = 64
FLOW_COLUMNS = (
    "rgb",
    "tokencut_cc",
    "proposal",
    "hit",
    "miss",
    "overlay",
)
COLUMN_MEANINGS = {
    "rgb": "目标正常图",
    "tokencut_cc": "当前 TokenCut S_x^0 的 8 连通块（按面积着色）",
    "proposal": "当前 a2 proposal（q70 ∩ S ∩ V）",
    "hit": "proposal ∩ 该块，且面积 ≥64px 的块",
    "miss": "TokenCut 块上 proposal <64px（抽 site 会抽空）",
    "overlay": "青=miss 块，品红=proposal",
}
MapsFn = Callable[[Mapping[str, Any]], dict[str, Any]]


def proposal_hits_on_object_ccs(
    object_mask: np.ndarray,
    proposal: np.ndarray,
    *,
    min_pixels: int = MIN_HIT_PIXELS,
) -> dict[str, Any]:
    obj = np.asarray(object_mask, dtype=bool)
    prop = np.asarray(proposal, dtype=bool)
    if obj.shape != prop.shape:
        raise ValueError("object and proposal must share one grid")
    if type(min_pixels) is not int or min_pixels < 1:
        raise ValueError("min_pixels must be a positive integer")
    ranked = rank_components(obj)
    components: list[dict[str, Any]] = []
    hit_masks: list[np.ndarray] = []
    miss_masks: list[np.ndarray] = []
    for item in ranked:
        proposal_pixels = int((prop & item.mask).sum())
        hit = proposal_pixels >= min_pixels
        components.append(
            {
                "rank": item.rank,
                "object_pixels": item.pixels,
                "proposal_pixels": proposal_pixels,
                "hit": hit,
            }
        )
        if hit:
            hit_masks.append(np.ascontiguousarray(prop & item.mask))
        else:
            miss_masks.append(item.mask)
    hit_union = np.zeros(obj.shape, dtype=bool)
    miss_union = np.zeros(obj.shape, dtype=bool)
    for mask in hit_masks:
        hit_union |= mask
    for mask in miss_masks:
        miss_union |= mask
    return {
        "n_cc": len(ranked),
        "n_hit": sum(bool(item["hit"]) for item in components),
        "n_miss": sum(not bool(item["hit"]) for item in components),
        "components": components,
        "hit_mask": hit_union,
        "miss_mask": miss_union,
        "ranked": ranked,
    }


def require_frozen_identities(
    frozen: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[str, ...]:
    identities = tuple(str(item["identity"]) for item in frozen)
    if not identities:
        raise ValueError("frozen identity list is empty")
    for item in frozen:
        if str(item["category"]) in FORBIDDEN_CATEGORIES:
            raise ValueError("can is deferred; proposal-on-TokenCut preview does not include MVTec AD 2")
    return identities


def _row_arrays(
    *,
    rgb: np.ndarray,
    object_mask: np.ndarray,
    proposal: np.ndarray,
    report: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    ranked = report["ranked"]
    hit_ranked = tuple(
        item for item, rec in zip(ranked, report["components"]) if rec["hit"]
    )
    miss_ranked = tuple(
        item for item, rec in zip(ranked, report["components"]) if not rec["hit"]
    )
    overlay = _overlay(rgb, report["miss_mask"], (40, 190, 210), 0.45)
    overlay = _overlay(overlay, proposal, (220, 70, 180), 0.55)
    return {
        "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
        "tokencut_cc": color_ranked_components(ranked, object_mask.shape),
        "proposal": _mask_u8(proposal),
        "hit": color_ranked_components(hit_ranked, object_mask.shape),
        "miss": color_ranked_components(miss_ranked, object_mask.shape),
        "overlay": overlay,
    }


def run_proposal_on_tokencut_cc_preview(
    *,
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    output: Path,
    maps_fn: MapsFn,
    min_pixels: int = MIN_HIT_PIXELS,
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
        obj = np.asarray(maps["object"], dtype=bool)
        proposal = np.asarray(maps["proposal"], dtype=bool)
        report = proposal_hits_on_object_ccs(obj, proposal, min_pixels=min_pixels)
        title = f"{identity}  k={report['n_cc']} hit={report['n_hit']} miss={report['n_miss']}"
        arrays = _row_arrays(
            rgb=np.asarray(maps["rgb"], dtype=np.uint8),
            object_mask=obj,
            proposal=proposal,
            report=report,
        )
        sheets[category].append((title, arrays))
        all_rows.append((title, arrays))
        records.append(
            {
                "identity": identity,
                "category": category,
                "n_cc": report["n_cc"],
                "n_hit": report["n_hit"],
                "n_miss": report["n_miss"],
                "components": report["components"],
            }
        )
    for category, panels in sheets.items():
        _save_sheet(destination / f"sheet_{category}.png", panels, FLOW_COLUMNS, tile=tile)
    _save_sheet(destination / "all.png", all_rows, FLOW_COLUMNS, tile=tile)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_official_generate": True,
        "one_variable": "current_a2_proposal_on_tokencut_relative_ccs",
        "column_meanings": COLUMN_MEANINGS,
        "min_hit_pixels": min_pixels,
        "n": len(records),
        "rows": records,
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_proposal_on_tokencut_cc_preview_from_cli(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cable_experiment: Path,
    output: Path,
    repository_root: Path,
    device: str | None = None,
) -> dict[str, Any]:
    del mvtec_root
    from cadiff.v06_a.a2 import A2ProposalGenerator
    from cadiff.v06_a.structure import estimate_normal_structure
    from cadiff.v06_a.tokencut import load_official_dinov2_encoder
    from cadiff.v06_atlas_diag.extra_a import (
        ExtraAStructureConfig,
        build_extra_preview_slots,
        inventory_visa,
        load_extra_identity,
        select_visa_categories,
    )
    from cadiff.v06_c.visa_pilot import EXPECTED

    cwd = repository_root.resolve()
    del cable_experiment, cwd
    if tuple(select_visa_categories(visa_root)) != EXPECTED:
        raise ValueError("VisA extra-A selection drifted")
    visa_packs = tuple(inventory_visa(visa_root, name) for name in EXPECTED)
    identities: list[dict[str, Any]] = []
    for pack in visa_packs:
        for slot in build_extra_preview_slots(pack):
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
    visa_generators: dict[str, A2ProposalGenerator] = {}
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

    def maps_fn(row: Mapping[str, Any]) -> dict[str, Any]:
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
        source = generator.select_source(str(row["defect_type"]), int(row["seed"]))
        if source.record.item_id != row["source_id"]:
            raise RuntimeError(
                f"preview source drifted: {row['identity']} expected {row['source_id']}"
            )
        a2 = generator.generate(target, str(row["defect_type"]), int(row["seed"]))
        if a2.target_support is None:
            raise ValueError("official a2 TokenCut object required")
        return {
            "rgb": np.asarray(target.normal.rgb, dtype=np.uint8).copy(),
            "object": np.ascontiguousarray(a2.target_support & target.valid, dtype=bool),
            "proposal": np.asarray(a2.proposal, dtype=bool),
        }

    return run_proposal_on_tokencut_cc_preview(
        rows=identities,
        output=output,
        maps_fn=maps_fn,
        min_pixels=MIN_HIT_PIXELS,
    )

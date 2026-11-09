"""Extra-category diagnostic: same a2 largest-CC placement as Cable, no renderer.

Frozen 20260910 extra/cad3 preview identities. One variable vs Cable floor
diagnostic: category set. No new weights, no Cable/Hazelnut R, no generate().
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator, _material_proposal_maps
from cadiff.v06_a.structure import _estimate_support, estimate_normal_structure
from cadiff.v06_atlas_diag.cad3_a import inventory_cad3, select_cad3_categories
from cadiff.v06_atlas_diag.extra_a import (
    ExtraACategory,
    ExtraASlot,
    ExtraAStructureConfig,
    build_extra_preview_slots,
    inventory_mvtec,
    inventory_visa,
    load_extra_identity,
    select_mvtec_categories,
    select_visa_categories,
)
from cadiff.v06_atlas_diag.features import FrozenDinoEncoder
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source
from cadiff.v06_b.a2_component_place import (
    color_components,
    instantiate_on_component,
    label_a2_components,
    pick_largest_site,
)
from cadiff.v06_b.a2_shape_instantiate import (
    EmptyFeasiblePlacement,
    _contained_output,
    _fail_rgb,
    _mask_u8,
    _overlay,
    _save_sheet,
)
from cadiff.v06_b.inference import validate_realization

EXPECTED_PREVIEW_ROWS = 34
FLOW_COLUMNS = (
    "source",
    "source_gt",
    "inpaint",
    "shape",
    "normal",
    "a2_prop",
    "a2_cc",
    "site",
    "placed",
    "overlay",
)
COLUMN_MEANINGS = {
    "source": "few-shot RGB，红叠加为官方 GT",
    "source_gt": "few-shot 官方 GT",
    "inpaint": "最近邻填洞后的源图（a2 用）",
    "shape": "本行采用的缩放+旋转 GT 模板；保底时为未放下的源 GT",
    "normal": "目标正常图 x",
    "a2_prop": "正式 a2 整张材料区",
    "a2_cc": "a2 的 8 连通部件着色",
    "site": "最大 a2 连通部件",
    "placed": "T(g)；保底时等于 site",
    "overlay": "蓝=a2，绿=最大部件，黄=placed，品红=proposal 保底",
}


def load_extra_preview_packs(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cad3_root: Path,
) -> tuple[ExtraACategory, ...]:
    packs = [
        *[inventory_mvtec(mvtec_root, name) for name in select_mvtec_categories(mvtec_root)],
        *[inventory_visa(visa_root, name) for name in select_visa_categories(visa_root)],
        *[inventory_cad3(cad3_root, name) for name in select_cad3_categories(cad3_root)],
    ]
    return tuple(packs)


def preview_jobs(packs: tuple[ExtraACategory, ...]) -> tuple[tuple[ExtraACategory, ExtraASlot], ...]:
    jobs = tuple((pack, slot) for pack in packs for slot in build_extra_preview_slots(pack))
    if len(jobs) != EXPECTED_PREVIEW_ROWS:
        raise RuntimeError(
            f"frozen extra a2 preview must stay {EXPECTED_PREVIEW_ROWS} rows, got {len(jobs)}"
        )
    return jobs


def _identity_key(slot: ExtraASlot) -> str:
    return f"{slot.dataset}/{slot.category}/{slot.defect_type}-{slot.normal_id}-{slot.seed}"


def _status_label(record: dict[str, Any]) -> tuple[str, str]:
    identity = str(record["identity"])
    if record["failed"]:
        return identity + " FAIL", "FAIL " + str(record["failure_reason"])
    if record.get("used_proposal_floor"):
        return identity + " FLOOR", "OK FLOOR"
    if record.get("used_site_fallback"):
        return identity + " FALLBACK", "OK FALLBACK"
    return identity, "OK"


def run_a2_component_place_extra_diagnostic(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cad3_root: Path,
    output: Path,
    device: str | None = None,
    tile: int = 160,
) -> dict[str, Any]:
    destination = _contained_output(output)
    packs = load_extra_preview_packs(
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        cad3_root=cad3_root,
    )
    jobs = preview_jobs(packs)
    encoder = FrozenDinoEncoder(device=device)
    destination.mkdir(parents=True, exist_ok=False)
    identities = destination / "identities"
    identities.mkdir()
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    per_category: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    generators: dict[str, A2ProposalGenerator] = {}
    structures: dict[tuple[str, str, str], Any] = {}

    for pack, slot in jobs:
        pack_key = f"{pack.dataset}/{pack.name}"
        config = ExtraAStructureConfig(category=pack.name, defect_types=pack.defect_types)
        if pack_key not in generators:
            sources = {
                defect_type: tuple(
                    load_extra_identity(record, dataset_root=pack.root) for record in rows
                )
                for defect_type, rows in pack.fewshot.items()
            }
            generators[pack_key] = A2ProposalGenerator(
                sources,
                config=config,
                identity_manifest=None,
                encoder=encoder,
            )
        generator = generators[pack_key]
        identity = _identity_key(slot)
        sample_root = identities / pack.dataset / pack.name / f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        sample_root.mkdir(parents=True)
        structure_key = (pack.dataset, pack.name, slot.normal_id)
        if structure_key not in structures:
            target_record = next(
                record for record in pack.normals if record.item_id == slot.normal_id
            )
            structures[structure_key] = estimate_normal_structure(
                load_extra_identity(target_record, dataset_root=pack.root),
                config=config,
            )
        target = structures[structure_key]
        source = generator.select_source(slot.defect_type, slot.seed)
        if source.record.item_id != slot.source_id:
            raise RuntimeError(
                f"preview source drifted: {identity} expected {slot.source_id} got {source.record.item_id}"
            )
        assert source.anomaly_gt is not None
        inpainted = neighbor_fill_source(source.rgb, source.anomaly_gt, source.valid)
        inpainted_identity = replace(
            source,
            rgb=np.array(inpainted, copy=True),
            anomaly_gt=None,
            record=replace(source.record, role="normal_fit", defect_type="good"),
        )
        source_support = _estimate_support(inpainted_identity, config)
        source_feat = generator._encoder.encode(inpainted)
        target_feat = generator._encoder.encode(target.normal.rgb)
        proposal, tau, similarity = _material_proposal_maps(
            source.anomaly_gt,
            source.valid,
            target,
            source_feat=source_feat,
            target_feat=target_feat,
        )
        cc_labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
        preferred = pick_largest_site(sites) if sites else None
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        fail = _fail_rgb(normal.shape)
        empty_mask = np.zeros(proposal.shape, dtype=bool)
        preferred_mask = preferred.mask if preferred is not None else empty_mask
        arrays = {
            "source": _overlay(source.rgb, source.anomaly_gt, (255, 40, 40), 0.55),
            "source_gt": _mask_u8(source.anomaly_gt),
            "inpaint": np.ascontiguousarray(inpainted),
            "shape": _mask_u8(source.anomaly_gt),
            "normal": normal,
            "a2_prop": _mask_u8(proposal),
            "a2_cc": color_components(cc_labels),
            "site": _mask_u8(preferred_mask),
            "placed": _mask_u8(empty_mask),
            "overlay": _overlay(
                _overlay(normal, proposal, (80, 160, 255), 0.28),
                preferred_mask,
                (40, 200, 80),
                0.4,
            ),
        }
        record: dict[str, Any] = {
            "identity": identity,
            "dataset": slot.dataset,
            "category": slot.category,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": source.record.item_id,
            "seed": slot.seed,
            "a2_tau": tau,
            "proposal_pixels": int(proposal.sum()),
            "component_count": len(sites),
            "source_gt_pixels": int(source.anomaly_gt.sum()),
            "failed": False,
            "failure_reason": None,
            "site_index": None if preferred is None else preferred.index,
            "site_pixels": 0 if preferred is None else preferred.pixels,
            "used_site_fallback": False,
            "used_proposal_floor": False,
            "scale": None,
            "angle_deg": None,
            "placed_pixels": 0,
            "feasible_count": 0,
            "m_final_subset_of_site": None,
            "m_final_subset_of_proposal": None,
        }
        a2_illegal = (
            not proposal.any()
            or bool(np.all(proposal[target.valid]))
            or bool(np.any(proposal & ~target.valid))
        )
        try:
            if a2_illegal:
                raise EmptyFeasiblePlacement("a2 proposal is empty, full within V, or outside V_x")
            template, placement = instantiate_on_component(
                source.anomaly_gt,
                source_support,
                target.support & target.valid,
                proposal,
                similarity,
                seed=slot.seed,
            )
            validate_realization(placement.mask, placement.site)
            placed_color = (
                (220, 40, 180) if placement.used_proposal_floor else (255, 220, 0)
            )
            arrays["shape"] = _mask_u8(template)
            arrays["site"] = _mask_u8(placement.site)
            arrays["placed"] = _mask_u8(placement.mask)
            arrays["overlay"] = _overlay(
                _overlay(
                    _overlay(normal, proposal, (80, 160, 255), 0.28),
                    placement.site,
                    (40, 200, 80),
                    0.4,
                ),
                placement.mask,
                placed_color,
                0.75,
            )
            in_site = bool(not (placement.mask & ~placement.site).any())
            in_prop = bool(not (placement.mask & ~proposal).any())
            if not in_site or not in_prop:
                raise ValueError("placed shape violated the single-component contract")
            record.update(
                site_index=placement.site_index,
                site_pixels=int(placement.site.sum()),
                scale=placement.scale,
                angle_deg=placement.angle_deg,
                placed_pixels=placement.placed_pixels,
                feasible_count=placement.feasible_count,
                m_final_subset_of_site=in_site,
                m_final_subset_of_proposal=in_prop,
                used_site_fallback=placement.used_site_fallback,
                used_proposal_floor=placement.used_proposal_floor,
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
            arrays["placed"] = fail
            arrays["overlay"] = fail
        for name, array in arrays.items():
            Image.fromarray(array).save(sample_root / f"{name}.png")
        write_json_atomic(sample_root / "record.json", record)
        records.append(record)
        label_text, status = _status_label(record)
        flow = {name: arrays[name] for name in FLOW_COLUMNS}
        flow_rows.append((label_text, flow))
        per_category[pack_key].append((label_text, flow))
        print(identity, status, flush=True)

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    for pack_key, rows in per_category.items():
        slug = pack_key.replace("/", "_")
        _save_sheet(destination / f"flow_{slug}.png", rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": "v06_b_a2_component_place_extra_v1",
        "diagnostic_only": True,
        "not_learned": True,
        "not_official_pass": True,
        "no_renderer": True,
        "one_variable": "same_largest_cc_floor_placement_on_frozen_extra_preview",
        "identity_count": len(records),
        "expected_identity_count": EXPECTED_PREVIEW_ROWS,
        "failed_rows": sum(row["failed"] for row in records),
        "floor_rows": sum(bool(row.get("used_proposal_floor")) for row in records),
        "fallback_rows": sum(bool(row.get("used_site_fallback")) for row in records),
        "method": "fewshot_gt_shape_inside_largest_a2_cc_or_site_floor",
        "preview_rule": "first_2_types_x_first_2_frozen_normals_x_seed_2026082801",
        "categories": [f"{pack.dataset}/{pack.name}" for pack in packs],
        "empty_a2_still_fails": True,
        "proposal_floor_is_last_resort": True,
        "no_projection": True,
        "no_square_dilate": True,
        "did_not_call_module_b": True,
        "did_not_overwrite_f1_f4": True,
        "flow_columns": list(FLOW_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "destination": str(destination),
        "visible_defects_passed": None,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_a2_component_place_extra_from_cli(
    *,
    paths: Path,
    visa_root: Path,
    cad3_root: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_a2_component_place_extra_diagnostic(
        mvtec_root=project.dataset_root,
        visa_root=visa_root if visa_root.is_absolute() else cwd / visa_root,
        cad3_root=cad3_root if cad3_root.is_absolute() else cwd / cad3_root,
        output=output,
        device=device,
    )


__all__ = [
    "COLUMN_MEANINGS",
    "EXPECTED_PREVIEW_ROWS",
    "FLOW_COLUMNS",
    "load_extra_preview_packs",
    "preview_jobs",
    "run_a2_component_place_extra_diagnostic",
    "run_a2_component_place_extra_from_cli",
]

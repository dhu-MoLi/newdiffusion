"""Historical H pose-sample diagnostic (2026-09-16).

Yellow = greedy first-hit (historical). Magenta = then-unofficial sampled pose.
Official H is now instantiate_on_component (sampled pose + GT-in-site rescue).
Do not rerun landed pose-sample directories.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator, load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import load_v06_a_identity_manifest
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_atlas_diag.cable_a import _lookup_p1_target, build_cable_a_slots
from cadiff.v06_atlas_diag.extra_a import ExtraAStructureConfig, load_extra_identity
from cadiff.v06_b.a2_component_place_extra import (
    EXPECTED_PREVIEW_ROWS,
    _identity_key,
    load_extra_preview_packs,
    preview_jobs,
)
from cadiff.v06_b.a2_component_place import (
    MIN_SHAPE_PIXELS,
    ComponentPlacement,
    EmptyFeasiblePlacement,
    PoseCandidate,
    _placement_from_hit,
    _proposal_floor_placement,
    collect_feasible_poses,
    instantiate_on_component_greedy,
    label_a2_components,
    pick_largest_site,
)
from cadiff.v06_b.a2_shape_instantiate import (
    _contained_output,
    _mask_u8,
    _overlay,
    _save_sheet,
)

SCHEMA_VERSION = "v06_b_h_pose_sample_diag_v1"
FLOW_COLUMNS = (
    "source",
    "source_gt",
    "site",
    "greedy",
    "sampled",
    "overlay",
)
COLUMN_MEANINGS = {
    "source": "few-shot RGB，红叠加为官方 GT",
    "source_gt": "few-shot 官方 GT",
    "site": "最大 8 连通 a2 部件",
    "greedy": "历史 H：第一组能放下的 (尺度, 角度)",
    "sampled": "当时诊断臂：在同一网格的可行 (尺度, 角度) 里用 seed 抽一组",
    "overlay": "绿=site，黄=greedy，品红=sampled",
}


def instantiate_on_component_sampled(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
    allow_proposal_floor: bool = True,
) -> tuple[NDArray[np.bool_], ComponentPlacement, dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(min_shape_pixels) is not int or min_shape_pixels < 1:
        raise ValueError("min_shape_pixels must be a positive integer")
    if not isinstance(allow_proposal_floor, bool):
        raise ValueError("allow_proposal_floor must be boolean")
    _labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
    if not sites:
        raise EmptyFeasiblePlacement("a2 has no component")
    largest = pick_largest_site(sites)
    candidates, used_site_fallback = collect_feasible_poses(
        source_gt,
        source_support,
        target_support,
        largest.mask,
        min_shape_pixels=min_shape_pixels,
    )
    info: dict[str, Any] = {
        "pose_count": len(candidates),
        "used_site_fallback": used_site_fallback,
        "used_proposal_floor": False,
        "sampled_index": None,
        "sampled_factor": None,
    }
    if not candidates:
        if not allow_proposal_floor:
            raise EmptyFeasiblePlacement("shape does not fit the largest a2 component")
        placement = _proposal_floor_placement(
            site=largest,
            proposal=proposal,
            source_gt=source_gt,
            component_count=len(sites),
        )
        info["used_proposal_floor"] = True
        return np.ascontiguousarray(np.asarray(source_gt, dtype=bool)), placement, info
    rng = np.random.RandomState(seed)
    index = int(rng.randint(0, len(candidates)))
    chosen = candidates[index]
    info.update(
        sampled_index=index,
        sampled_factor=float(chosen.factor),
        used_site_fallback=bool(chosen.used_site_fallback),
    )
    placement = _placement_from_hit(
        template=chosen.template,
        site=largest,
        similarity=similarity,
        proposal=proposal,
        seed=seed,
        scale=chosen.scale,
        angle=chosen.angle_deg,
        source_gt=source_gt,
        component_count=len(sites),
        used_site_fallback=chosen.used_site_fallback,
    )
    return chosen.template, placement, info


def run_h_pose_sample_diagnostic(
    *,
    repository_root: Path,
    dataset_root: Path,
    output: Path,
    a_experiment: Path,
    device: str | None = None,
    tile: int = 160,
) -> dict[str, Any]:
    destination = _contained_output(output)
    experiment = load_v06_a_experiment(a_experiment)
    manifest = load_v06_a_identity_manifest(
        repository_root / experiment.identity_manifest,
        experiment=experiment,
    )
    slots = build_cable_a_slots(manifest, experiment=experiment)
    destination.mkdir(parents=True, exist_ok=False)
    identities = [
        {
            "identity": f"{slot.defect_type}-{slot.normal_id}-{slot.seed}",
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": slot.source_id,
            "seed": slot.seed,
        }
        for slot in slots
    ]
    write_json_atomic(
        destination / "identities.json",
        {
            "frozen_before_images": True,
            "schedule": "cable_p1_32",
            "normals": ["000", "071", "152", "223"],
            "seeds": [2026082801, 2026082802],
            "rows": identities,
        },
    )
    generator = load_a2_generator(
        experiment_path=a_experiment,
        dataset_root=dataset_root,
        device=device,
    )
    sample_root = destination / "identities"
    sample_root.mkdir()
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    per_type: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {
        name: [] for name in experiment.defect_types
    }
    records: list[dict[str, Any]] = []
    structures: dict[str, Any] = {}

    for slot in slots:
        identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        row_dir = sample_root / identity
        row_dir.mkdir()
        if slot.normal_id not in structures:
            target_record = _lookup_p1_target(manifest, slot.normal_id)
            loaded_target = load_identity_record(
                target_record,
                dataset_root=dataset_root,
                config=experiment,
            )
            structures[slot.normal_id] = estimate_normal_structure(
                loaded_target, config=experiment
            )
        target = structures[slot.normal_id]
        source = generator.select_source(slot.defect_type, slot.seed)
        assert source.anomaly_gt is not None
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        empty = np.zeros(source.anomaly_gt.shape, dtype=bool)
        arrays = {
            "source": _overlay(source.rgb, source.anomaly_gt, (255, 40, 40), 0.45),
            "source_gt": _mask_u8(source.anomaly_gt),
            "site": _mask_u8(empty),
            "greedy": _mask_u8(empty),
            "sampled": _mask_u8(empty),
            "overlay": _overlay(normal, empty, (80, 160, 255), 0.28),
        }
        record: dict[str, Any] = {
            "identity": identity,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": source.record.item_id,
            "seed": slot.seed,
            "failed": False,
            "failure_reason": None,
        }
        try:
            result = generator.generate(target, slot.defect_type, slot.seed)
            if result.target_support is None or result.source_support is None:
                raise ValueError("official a2 must return TokenCut supports")
            proposal = np.asarray(result.proposal, dtype=bool)
            similarity = np.asarray(result.similarity, dtype=np.float32)
            source_support = np.asarray(result.source_support, dtype=bool)
            target_support = np.asarray(result.target_support, dtype=bool)
            _greedy_template, greedy = instantiate_on_component_greedy(
                source.anomaly_gt,
                source_support,
                target_support,
                proposal,
                similarity,
                seed=slot.seed,
            )
            _sampled_template, sampled, info = instantiate_on_component_sampled(
                source.anomaly_gt,
                source_support,
                target_support,
                proposal,
                similarity,
                seed=slot.seed,
            )
            if (greedy.mask & ~greedy.site).any() or (sampled.mask & ~sampled.site).any():
                raise RuntimeError("placement escaped site")
            same_pose = (
                (not greedy.used_proposal_floor)
                and (not sampled.used_proposal_floor)
                and greedy.angle_deg == sampled.angle_deg
                and abs(float(greedy.scale) - float(sampled.scale)) < 1e-6
            )
            arrays.update(
                {
                    "site": _mask_u8(greedy.site),
                    "greedy": _mask_u8(greedy.mask),
                    "sampled": _mask_u8(sampled.mask),
                    "overlay": _overlay(
                        _overlay(
                            _overlay(normal, greedy.site, (40, 200, 80), 0.28),
                            greedy.mask,
                            (255, 220, 0),
                            0.7,
                        ),
                        sampled.mask,
                        (220, 40, 180),
                        0.55,
                    ),
                }
            )
            record.update(
                object_support_method=result.object_support_method,
                pose_count=int(info["pose_count"]),
                sampled_index=info.get("sampled_index"),
                sampled_factor=info.get("sampled_factor"),
                greedy_scale=float(greedy.scale),
                greedy_angle_deg=int(greedy.angle_deg),
                greedy_floor=bool(greedy.used_proposal_floor),
                greedy_fallback=bool(greedy.used_site_fallback),
                sampled_scale=float(sampled.scale),
                sampled_angle_deg=int(sampled.angle_deg),
                sampled_floor=bool(sampled.used_proposal_floor),
                sampled_fallback=bool(sampled.used_site_fallback),
                same_pose=same_pose,
                xor_pixels=int(np.logical_xor(greedy.mask, sampled.mask).sum()),
                greedy_pixels=int(greedy.mask.sum()),
                sampled_pixels=int(sampled.mask.sum()),
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        for name, array in arrays.items():
            Image.fromarray(array).save(row_dir / f"{name}.png")
        write_json_atomic(row_dir / "record.json", record)
        records.append(record)
        if record["failed"]:
            label_text = identity + " FAIL"
        elif record.get("greedy_floor") or record.get("sampled_floor"):
            label_text = identity + " FLOOR"
        elif record.get("same_pose"):
            label_text = (
                f"{identity} SAME a{record['greedy_angle_deg']} n{record['pose_count']}"
            )
        else:
            label_text = (
                f"{identity} DIFF g{record['greedy_angle_deg']}/"
                f"s{record['sampled_angle_deg']} n{record['pose_count']}"
            )
        flow = {name: arrays[name] for name in FLOW_COLUMNS}
        flow_rows.append((label_text, flow))
        per_type[slot.defect_type].append((label_text, flow))
        print(identity, label_text.split(" ", 1)[-1], flush=True)

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    for kind, rows in per_type.items():
        _save_sheet(destination / f"flow_{kind}.png", rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "not_official_h": True,
        "not_learned": True,
        "not_generate": True,
        "one_variable": "h_pose_greedy_first_hit_vs_sample_feasible_scale_angle",
        "identity_count": len(records),
        "failed_rows": sum(bool(row["failed"]) for row in records),
        "floor_rows": sum(bool(row.get("greedy_floor") or row.get("sampled_floor")) for row in records),
        "same_pose_rows": sum(bool(row.get("same_pose")) for row in records if not row["failed"]),
        "diff_pose_rows": sum(
            (not row["failed"]) and (not row.get("same_pose")) and (not row.get("greedy_floor"))
            for row in records
        ),
        "flow_columns": list(FLOW_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "sampling": "uniform among feasible (scale, angle) on the first non-empty grid; translation unchanged",
        "destination": str(destination),
        "visible_defects_passed": None,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def _pose_row_label(identity: str, record: dict[str, Any]) -> str:
    if record["failed"]:
        return identity + " FAIL"
    if record.get("greedy_floor") or record.get("sampled_floor"):
        return identity + " FLOOR"
    if record.get("same_pose"):
        return f"{identity} SAME a{record['greedy_angle_deg']} n{record['pose_count']}"
    return (
        f"{identity} DIFF g{record['greedy_angle_deg']}/"
        f"s{record['sampled_angle_deg']} n{record['pose_count']}"
    )


def run_h_pose_sample_extra_diagnostic(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cad3_root: Path,
    output: Path,
    device: str | None = None,
    tile: int = 160,
) -> dict[str, Any]:
    """Frozen extra/cad3 34-row preview: greedy first-hit vs sampled feasible pose.

    Informal only. Does not write generate(), official H, Cable pose sample, or extra corr3.
    """

    destination = _contained_output(Path(output))
    packs = load_extra_preview_packs(
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        cad3_root=cad3_root,
    )
    jobs = preview_jobs(packs)
    destination.mkdir(parents=True, exist_ok=False)
    identities = [
        {
            "identity": _identity_key(slot),
            "dataset": slot.dataset,
            "category": slot.category,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": slot.source_id,
            "seed": slot.seed,
        }
        for _pack, slot in jobs
    ]
    write_json_atomic(
        destination / "identities.json",
        {
            "frozen_before_images": True,
            "schedule": "extra_cad3_preview_34",
            "seed": 2026082801,
            "expected_rows": EXPECTED_PREVIEW_ROWS,
            "rows": identities,
        },
    )
    encoder = load_official_dinov2_encoder(device=device)
    generators: dict[str, A2ProposalGenerator] = {}
    structures: dict[tuple[str, str, str], Any] = {}
    sample_root = destination / "identities"
    sample_root.mkdir()
    records: list[dict[str, Any]] = []
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    per_category: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = defaultdict(list)

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
        row_dir = (
            sample_root
            / pack.dataset
            / pack.name
            / f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        )
        row_dir.mkdir(parents=True)
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
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        empty = np.zeros(source.anomaly_gt.shape, dtype=bool)
        arrays = {
            "source": _overlay(source.rgb, source.anomaly_gt, (255, 40, 40), 0.45),
            "source_gt": _mask_u8(source.anomaly_gt),
            "site": _mask_u8(empty),
            "greedy": _mask_u8(empty),
            "sampled": _mask_u8(empty),
            "overlay": _overlay(normal, empty, (80, 160, 255), 0.28),
        }
        record: dict[str, Any] = {
            "identity": identity,
            "dataset": slot.dataset,
            "category": slot.category,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": source.record.item_id,
            "seed": slot.seed,
            "failed": False,
            "failure_reason": None,
        }
        try:
            result = generator.generate(target, slot.defect_type, slot.seed)
            if result.target_support is None or result.source_support is None:
                raise ValueError("official a2 must return TokenCut supports")
            if result.object_support_method != "tokencut":
                raise ValueError(
                    f"{identity} expected tokencut support, got {result.object_support_method}"
                )
            proposal = np.asarray(result.proposal, dtype=bool)
            similarity = np.asarray(result.similarity, dtype=np.float32)
            source_support = np.asarray(result.source_support, dtype=bool)
            target_support = np.asarray(result.target_support, dtype=bool)
            _greedy_template, greedy = instantiate_on_component_greedy(
                source.anomaly_gt,
                source_support,
                target_support,
                proposal,
                similarity,
                seed=slot.seed,
            )
            _sampled_template, sampled, info = instantiate_on_component_sampled(
                source.anomaly_gt,
                source_support,
                target_support,
                proposal,
                similarity,
                seed=slot.seed,
            )
            if (greedy.mask & ~greedy.site).any() or (sampled.mask & ~sampled.site).any():
                raise RuntimeError("placement escaped site")
            same_pose = (
                (not greedy.used_proposal_floor)
                and (not sampled.used_proposal_floor)
                and greedy.angle_deg == sampled.angle_deg
                and abs(float(greedy.scale) - float(sampled.scale)) < 1e-6
            )
            arrays.update(
                {
                    "site": _mask_u8(greedy.site),
                    "greedy": _mask_u8(greedy.mask),
                    "sampled": _mask_u8(sampled.mask),
                    "overlay": _overlay(
                        _overlay(
                            _overlay(normal, greedy.site, (40, 200, 80), 0.28),
                            greedy.mask,
                            (255, 220, 0),
                            0.7,
                        ),
                        sampled.mask,
                        (220, 40, 180),
                        0.55,
                    ),
                }
            )
            record.update(
                object_support_method=result.object_support_method,
                pose_count=int(info["pose_count"]),
                sampled_index=info.get("sampled_index"),
                sampled_factor=info.get("sampled_factor"),
                greedy_scale=float(greedy.scale),
                greedy_angle_deg=int(greedy.angle_deg),
                greedy_floor=bool(greedy.used_proposal_floor),
                greedy_fallback=bool(greedy.used_site_fallback),
                sampled_scale=float(sampled.scale),
                sampled_angle_deg=int(sampled.angle_deg),
                sampled_floor=bool(sampled.used_proposal_floor),
                sampled_fallback=bool(sampled.used_site_fallback),
                same_pose=same_pose,
                xor_pixels=int(np.logical_xor(greedy.mask, sampled.mask).sum()),
                greedy_pixels=int(greedy.mask.sum()),
                sampled_pixels=int(sampled.mask.sum()),
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        for name, array in arrays.items():
            Image.fromarray(array).save(row_dir / f"{name}.png")
        write_json_atomic(row_dir / "record.json", record)
        records.append(record)
        label_text = _pose_row_label(identity, record)
        flow = {name: arrays[name] for name in FLOW_COLUMNS}
        flow_rows.append((label_text, flow))
        per_category[pack_key].append((label_text, flow))
        print(identity, "FAIL" if record["failed"] else label_text.rsplit(" ", 1)[-1], flush=True)

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    for pack_key, rows in per_category.items():
        slug = pack_key.replace("/", "_")
        _save_sheet(destination / f"flow_{slug}.png", rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "not_official_h": True,
        "not_learned": True,
        "not_generate": True,
        "one_variable": "h_pose_greedy_first_hit_vs_sample_feasible_scale_angle_extra_cad3",
        "identity_count": len(records),
        "expected_identity_count": EXPECTED_PREVIEW_ROWS,
        "failed_rows": sum(bool(row["failed"]) for row in records),
        "floor_rows": sum(bool(row.get("greedy_floor") or row.get("sampled_floor")) for row in records),
        "same_pose_rows": sum(bool(row.get("same_pose")) for row in records if not row["failed"]),
        "diff_pose_rows": sum(
            (not row["failed"]) and (not row.get("same_pose")) and (not row.get("greedy_floor"))
            for row in records
        ),
        "preview_rule": "first_2_types_x_first_2_frozen_normals_x_seed_2026082801",
        "categories": [f"{pack.dataset}/{pack.name}" for pack in packs],
        "flow_columns": list(FLOW_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "sampling": "uniform among feasible (scale, angle) on the first non-empty grid; translation unchanged",
        "destination": str(destination),
        "visible_defects_passed": None,
        "excluded": "can is informal third-class and not in this 34-row freeze",
        "did_not_overwrite_cable_pose_sample": True,
        "did_not_overwrite_extra_corr3": True,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_h_pose_sample_extra_diagnostic_from_cli(
    *,
    paths: Path,
    visa_root: Path,
    cad3_root: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_h_pose_sample_extra_diagnostic(
        mvtec_root=project.dataset_root,
        visa_root=visa_root if visa_root.is_absolute() else cwd / visa_root,
        cad3_root=cad3_root if cad3_root.is_absolute() else cwd / cad3_root,
        output=output if output.is_absolute() else cwd / output,
        device=device,
    )


def run_h_pose_sample_diagnostic_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_h_pose_sample_diagnostic(
        repository_root=cwd,
        dataset_root=project.dataset_root,
        output=output if output.is_absolute() else cwd / output,
        a_experiment=a_experiment if a_experiment.is_absolute() else cwd / a_experiment,
        device=device,
    )


__all__ = [
    "PoseCandidate",
    "collect_feasible_poses",
    "instantiate_on_component_sampled",
    "run_h_pose_sample_diagnostic",
    "run_h_pose_sample_diagnostic_from_cli",
    "run_h_pose_sample_extra_diagnostic",
    "run_h_pose_sample_extra_diagnostic_from_cli",
]

"""Diagnostic: same 46 identities, only support quantile, proposal vs TokenCut object.

This look froze then-official q=0.70 as the compare column. Current official A
is q=0.60. Re-thresholds the upsampled similarity on S_x^0. Not generate(),
not H, not SAM. can deferred. This file does not run R.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from cadiff.provenance import write_json_atomic
from cadiff.v06_b.a2_shape_instantiate import _contained_output, _mask_u8, _overlay, _save_sheet
from cadiff.v06_c.tokencut_normal_preview import FORBIDDEN_CATEGORIES

SCHEMA = "v06_a_quantile_cover_v1"
QUANTILES = (0.50, 0.70, 0.85)
OFFICIAL_Q = 0.70
FLOW_COLUMNS = (
    "rgb",
    "object",
    "q50",
    "q70",
    "q85",
    "extra50",
)
COLUMN_MEANINGS = {
    "rgb": "目标正常图",
    "object": "当前 TokenCut S_x^0",
    "q50": "物体上第 50 分位，约留 50%",
    "q70": "正式分位，约留 30%",
    "q85": "物体上第 85 分位，约留 15%",
    "extra50": "q50 有、q70 没有（q70 丢掉的物体区）",
}
MapsFn = Callable[[Mapping[str, Any]], dict[str, Any]]


def proposal_at_support_quantile(
    similarity: np.ndarray,
    object_mask: np.ndarray,
    *,
    q: float,
) -> tuple[NDArray[np.bool_], float, float]:
    scores = np.asarray(similarity, dtype=np.float32)
    obj = np.asarray(object_mask, dtype=bool)
    if scores.shape != obj.shape:
        raise ValueError("similarity and object must share one grid")
    if not np.isfinite(q) or q < 0.0 or q > 1.0:
        raise ValueError("q must be in [0, 1]")
    if not obj.any():
        raise ValueError("TokenCut object is empty")
    support_sims = scores[obj]
    tau = float(np.quantile(support_sims, q))
    proposal = np.ascontiguousarray((scores >= tau) & obj)
    coverage = float(proposal.sum() / obj.sum())
    return proposal, tau, coverage


def require_frozen_identities(
    frozen: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> tuple[str, ...]:
    identities = tuple(str(item["identity"]) for item in frozen)
    if not identities:
        raise ValueError("frozen identity list is empty")
    for item in frozen:
        if str(item["category"]) in FORBIDDEN_CATEGORIES:
            raise ValueError("can is deferred; quantile cover does not include MVTec AD 2")
    return identities


def _row_arrays(
    *,
    rgb: np.ndarray,
    object_mask: np.ndarray,
    by_q: Mapping[float, np.ndarray],
) -> dict[str, np.ndarray]:
    q50 = by_q[0.50]
    q70 = by_q[0.70]
    q85 = by_q[0.85]
    extra = np.ascontiguousarray(q50 & ~q70)
    return {
        "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
        "object": _mask_u8(object_mask),
        "q50": _overlay(rgb, q50, (40, 190, 210), 0.55),
        "q70": _overlay(rgb, q70, (220, 70, 180), 0.55),
        "q85": _overlay(rgb, q85, (240, 220, 40), 0.55),
        "extra50": _overlay(rgb, extra, (40, 200, 90), 0.60),
    }


def run_a2_quantile_cover_preview(
    *,
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    output: Path,
    maps_fn: MapsFn,
    quantiles: tuple[float, ...] = QUANTILES,
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
            "official_q_unchanged": OFFICIAL_Q,
            "frozen_before_images": True,
            "quantiles": list(quantiles),
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
        obj = np.asarray(maps["object"], dtype=bool)
        similarity = np.asarray(maps["similarity"], dtype=np.float32)
        by_q: dict[float, np.ndarray] = {}
        coverages: dict[str, float] = {}
        taus: dict[str, float] = {}
        for q in quantiles:
            mask, tau, coverage = proposal_at_support_quantile(similarity, obj, q=q)
            by_q[q] = mask
            key = str(q)
            coverages[key] = coverage
            taus[key] = tau
        title = (
            f"{identity}  "
            + " ".join(f"q{q:g}={coverages[str(q)]:.2f}" for q in quantiles)
        )
        arrays = _row_arrays(
            rgb=np.asarray(maps["rgb"], dtype=np.uint8),
            object_mask=obj,
            by_q=by_q,
        )
        sheets[category].append((title, arrays))
        all_rows.append((title, arrays))
        records.append(
            {
                "identity": identity,
                "category": category,
                "object_pixels": int(obj.sum()),
                "coverages": coverages,
                "taus": taus,
            }
        )
    for category, panels in sheets.items():
        _save_sheet(destination / f"sheet_{category}.png", panels, FLOW_COLUMNS, tile=tile)
    _save_sheet(destination / "all.png", all_rows, FLOW_COLUMNS, tile=tile)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_official_generate": True,
        "official_q_unchanged": OFFICIAL_Q,
        "future_named_experiment": "quantile_vs_downstream_generate",
        "one_variable": "support_quantile_on_tokencut_object",
        "column_meanings": COLUMN_MEANINGS,
        "quantiles": list(quantiles),
        "n": len(records),
        "rows": records,
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_a2_quantile_cover_preview_from_cli(
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
            source = generator.select_source(defect_type, seed)
            if source.record.item_id != row["source_id"]:
                raise RuntimeError(
                    f"preview source drifted: {row['identity']} expected {row['source_id']}"
                )
        a2 = generator.generate(target, defect_type, seed)
        if a2.similarity is None or a2.target_support is None:
            raise ValueError("official a2 similarity and TokenCut object required")
        obj = np.ascontiguousarray(a2.target_support & target.valid, dtype=bool)
        return {
            "rgb": np.asarray(target.normal.rgb, dtype=np.uint8).copy(),
            "object": obj,
            "similarity": np.asarray(a2.similarity, dtype=np.float32),
        }

    return run_a2_quantile_cover_preview(
        rows=identities,
        output=output,
        maps_fn=maps_fn,
        quantiles=QUANTILES,
    )

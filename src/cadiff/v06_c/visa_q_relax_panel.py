"""Per-image q relaxation when the scaled template does not fit q=0.60.

Official q stays 0.60 for any row that already fits. Otherwise q drops in
steps of 0.05 until a contained placement succeeds, and stops at the whole
TokenCut object. Does not place outside the object, retrain R, or call generate().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_b.a2_component_place import MIN_SHAPE_PIXELS, instantiate_site_scale
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement
from cadiff.v06_c.capsules_combined_panel import _contained_output, n_connected
from cadiff.v06_c.official_a2_place import placement_from_official_a2
from cadiff.v06_c.capsules_combined_panel import _find_image
from cadiff.v06_c.loc_pred_gt_overlay import _mask_path
from cadiff.v06_c.visa_flow_steps_panel import _cell, _join, _open_mask, _open_rgb, _paint, _stack
from cadiff.v06_c.visa_pilot import (
    SOURCE_ROOT,
    fewshot_identity_records,
    load_visa_identity,
    load_visa_magic_inventory,
    load_visa_seas_inventory,
    normal_identity_records,
)

SCHEMA = "v06_c_visa_q_relax_panel_v1"
Q_STEPS = tuple(round(0.60 - 0.05 * step, 2) for step in range(13))


def proposal_at_q(
    similarity: np.ndarray, support: np.ndarray, q: float
) -> tuple[np.ndarray, float]:
    obj = np.asarray(support, dtype=bool)
    sims = np.asarray(similarity, dtype=np.float32)[obj]
    if sims.size == 0:
        raise ValueError("proposal quantile needs a non-empty object")
    if q <= 0:
        return np.ascontiguousarray(obj), float(np.min(sims))
    tau = float(np.quantile(sims, q))
    proposal = (np.asarray(similarity, dtype=np.float32) >= tau) & obj
    return np.ascontiguousarray(proposal), tau


def relax_q(
    *,
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    similarity: np.ndarray,
    seed: int,
    fewshot_areas: tuple[int, ...],
) -> dict[str, Any]:
    chosen: dict[str, Any] | None = None
    for q in Q_STEPS:
        proposal, tau = proposal_at_q(similarity, target_support, q)
        try:
            template, placement = instantiate_site_scale(
                source_gt,
                source_support,
                target_support,
                proposal,
                similarity,
                seed=seed,
                fewshot_areas=fewshot_areas,
                min_shape_pixels=MIN_SHAPE_PIXELS,
            )
        except EmptyFeasiblePlacement:
            continue
        chosen = {
            "q": q,
            "tau": tau,
            "proposal": proposal,
            "template": np.ascontiguousarray(template, dtype=bool),
            "placement": placement,
        }
        if not placement.used_clip:
            break
    if chosen is None:
        raise EmptyFeasiblePlacement("q relaxation found no placement down to the full object")
    return chosen


def _spec_row(
    spec: dict[str, Any],
    generator: Any,
    normals: dict[str, Any],
    structures: dict[str, Any],
    structure_config: Any,
    fewshot_areas: tuple[int, ...],
    layout: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample = Path(spec["sample_dir"])
    saved = np.asarray(Image.open(sample / "m_final.png").convert("L")) > 127
    record = json.loads((sample / "record.json").read_text(encoding="utf-8"))
    target_id = spec["target_id"]
    if target_id not in structures:
        loaded = load_visa_identity(normals[target_id], dataset_root=SOURCE_ROOT)
        structures[target_id] = estimate_normal_structure(loaded, config=structure_config)
    target = structures[target_id]
    a2, official, _template = placement_from_official_a2(
        generator,
        target,
        spec["defect_type"],
        spec["seed"],
        min_shape_pixels=MIN_SHAPE_PIXELS,
        return_template=True,
    )
    if a2.source_id != spec["source_id"]:
        raise RuntimeError(
            f"{spec['job_id']} replay source {a2.source_id} != {spec['source_id']}"
        )
    mismatch = int(np.logical_xor(np.asarray(official.mask, dtype=bool), saved).sum())
    if mismatch:
        raise RuntimeError(f"{spec['job_id']} q=0.60 replay mismatches dump by {mismatch} px")
    support = np.ascontiguousarray(a2.target_support & target.valid, dtype=bool)
    relaxed = relax_q(
        source_gt=np.asarray(a2.source_gt, dtype=bool),
        source_support=np.asarray(a2.source_support, dtype=bool),
        target_support=support,
        similarity=np.asarray(a2.similarity, dtype=np.float32),
        seed=spec["seed"],
        fewshot_areas=fewshot_areas,
    )
    normal = np.ascontiguousarray(target.normal.rgb)
    official_mask = np.asarray(official.mask, dtype=bool)
    relaxed_mask = np.asarray(relaxed["placement"].mask, dtype=bool)
    test_dir = layout / "test" / spec["type_slug"]
    gt_root = layout / "ground_truth"
    test_rgb = _open_rgb(_find_image(test_dir, spec["test_stem"]))
    test_gt = _open_mask(_mask_path(gt_root, spec["type_slug"], spec["test_stem"]), test_rgb.shape[:2])
    cells = [
        _cell(_paint(test_rgb, test_gt, (40, 220, 80), 0.45), f"测试GT 绿\n{spec['test_stem']} unpaired"),
        _cell(normal, "参考正常图"),
        _cell(
            _paint(normal, np.asarray(a2.proposal, dtype=bool), (250, 210, 40), 0.45),
            f"q=0.60 proposal\n{int(np.asarray(a2.proposal).sum())} px",
        ),
        _cell(
            _paint(normal, official_mask, (240, 40, 180), 0.55),
            f"q=0.60 m_final\n{int(official_mask.sum())} cc={n_connected(official_mask)}",
        ),
        _cell(
            _paint(normal, relaxed["proposal"], (80, 180, 255), 0.4),
            f"放宽 q={relaxed['q']:.2f}\n{int(relaxed['proposal'].sum())} px",
        ),
        _cell(
            _paint(normal, relaxed_mask, (240, 40, 180), 0.55),
            f"放宽 m_final\n{int(relaxed_mask.sum())} cc={n_connected(relaxed_mask)} clip={int(bool(relaxed['placement'].used_clip))}",
        ),
    ]
    title = (
        f"{spec['category']}/{spec['type_slug']} src {spec['source_id']} "
        f"job {spec['job_id']} q {0.60:.2f}->{relaxed['q']:.2f}"
    )
    stats = {
        "job_id": spec["job_id"],
        "source_id": spec["source_id"],
        "q_official": 0.60,
        "q_relaxed": relaxed["q"],
        "changed": relaxed["q"] < 0.60,
        "official_px": int(official_mask.sum()),
        "official_cc": n_connected(official_mask),
        "official_clip": bool(official.used_clip),
        "relaxed_px": int(relaxed_mask.sum()),
        "relaxed_cc": n_connected(relaxed_mask),
        "relaxed_clip": bool(relaxed["placement"].used_clip),
        "relaxed_proposal_px": int(relaxed["proposal"].sum()),
        "support_px": int(support.sum()),
    }
    return _join(cells, title), stats


def _build(category: str, encoder: Any, *, split: str) -> tuple[Any, dict[str, Any], Any, dict[str, tuple[int, ...]]]:
    if split == "seas":
        inventory = load_visa_seas_inventory(SOURCE_ROOT, category)
    elif split == "magic":
        inventory = load_visa_magic_inventory(SOURCE_ROOT, category, generate_only=True)
    else:
        raise ValueError(f"unknown split {split}")
    fewshot = fewshot_identity_records(inventory)
    sources = {
        item.name: tuple(
            load_visa_identity(record, dataset_root=SOURCE_ROOT)
            for record in fewshot
            if record.defect_type == item.name
        )
        for item in inventory.types
    }
    generator = A2ProposalGenerator(
        sources,
        config=inventory.structure,
        identity_manifest=None,
        encoder=encoder,
    )
    normals = {record.item_id: record for record in normal_identity_records(inventory)}
    areas = {
        name: tuple(int(np.asarray(item.anomaly_gt, dtype=bool).sum()) for item in items)
        for name, items in sources.items()
    }
    return generator, normals, inventory.structure, areas


def run_visa_q_relax_panel(
    output: Path,
    specs: tuple[dict[str, Any], ...],
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": "frozen jobs; lower q only when q=0.60 placement clips; stop at contained or full TokenCut",
            "q_steps": list(Q_STEPS),
            "rows": list(specs),
            "not_generate": True,
        },
    )
    encoder = load_official_dinov2_encoder(device=device)
    built: dict[tuple[str, str], tuple[Any, dict[str, Any], Any, dict[str, tuple[int, ...]]]] = {}
    structures: dict[str, dict[str, Any]] = {}
    strips: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    for spec in specs:
        category = spec["category"]
        key = (category, spec["split"])
        if key not in built:
            built[key] = _build(category, encoder, split=spec["split"])
        generator, normals, config, areas = built[key]
        strip, row_stats = _spec_row(
            spec,
            generator,
            normals,
            structures.setdefault(category, {}),
            config,
            areas[spec["defect_type"]],
            Path(spec["layout"]),
        )
        Image.fromarray(strip).save(rows_dir / f"{spec['job_id']}.png")
        strips.append(strip)
        stats.append(row_stats)
        print(spec["job_id"], row_stats, flush=True)
    Image.fromarray(_stack(strips, "左 q=0.60  右 按张放宽到能整块放进；不超出 TokenCut")).save(
        destination / "sheet.png"
    )
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "n": len(stats),
        "n_changed": sum(1 for row in stats if row["changed"]),
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload

"""Diagnostic: keep the rigid template outside TokenCut, and stop shrinking it by a multi-instance source object.

Two arms, not stacked into generate():
- noclip: same official scale and pose, m_final is the unclipped template
- unit_scale: object_scale forced to 1, max-overlap with the object, overflow kept
Only for already-triggered stuck_together rows. Does not retrain R.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_b.a2_component_place import (
    ANGLES_DEG,
    MIN_SHAPE_PIXELS,
    place_shape_max_overlap_clip,
    rotate_binary_mask,
    sample_area_factor,
)
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement, scale_binary_mask
from cadiff.v06_c.capsules_combined_panel import n_connected
from cadiff.v06_c.official_a2_place import placement_from_official_a2
from cadiff.v06_c.visa_flow_steps_panel import _cell, _join, _open_mask, _open_rgb, _paint, _stack
from cadiff.v06_c.visa_pilot import SOURCE_ROOT, load_visa_identity
from cadiff.v06_c.visa_q_relax_panel import _build
from cadiff.v06_c.capsules_combined_panel import _find_image
from cadiff.v06_c.loc_pred_gt_overlay import _mask_path

SCHEMA = "v06_c_visa_overflow_panel_v1"


def _unit_scale_overflow(
    source_gt: np.ndarray,
    target_support: np.ndarray,
    similarity: np.ndarray,
    seed: int,
    fewshot_areas: tuple[int, ...],
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    factor = sample_area_factor(int(source_gt.sum()), fewshot_areas, seed=seed, rng=rng)
    scaled = scale_binary_mask(source_gt, factor, size=int(source_gt.shape[0]))
    hits = []
    for angle in ANGLES_DEG:
        template = rotate_binary_mask(scaled, angle)
        try:
            placed = place_shape_max_overlap_clip(
                template,
                target_support,
                seed=seed,
                min_pixels=MIN_SHAPE_PIXELS,
            )
        except EmptyFeasiblePlacement:
            continue
        if placed.unclipped is None:
            continue
        hits.append(placed.unclipped)
    if not hits:
        raise EmptyFeasiblePlacement("unit-scale overflow has no pose")
    return np.ascontiguousarray(hits[int(rng.randint(0, len(hits)))], dtype=bool)


def _outside(mask: np.ndarray, support: np.ndarray) -> int:
    return int((np.asarray(mask, dtype=bool) & ~np.asarray(support, dtype=bool)).sum())


def run_visa_overflow_panel(
    output: Path,
    specs: tuple[dict[str, Any], ...],
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    from cadiff.v06_c.capsules_combined_panel import _contained_output

    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rows": list(specs),
            "arms": ("official", "noclip_same_scale", "unit_object_scale_keep_overflow"),
            "not_generate": True,
        },
    )
    encoder = load_official_dinov2_encoder(device=device)
    built: dict[tuple[str, str], Any] = {}
    structures: dict[str, dict[str, Any]] = {}
    strips = []
    stats = []
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    for spec in specs:
        key = (spec["category"], spec["split"])
        if key not in built:
            built[key] = _build(spec["category"], encoder, split=spec["split"])
        generator, normals, config, areas = built[key]
        cache = structures.setdefault(spec["category"], {})
        if spec["target_id"] not in cache:
            loaded = load_visa_identity(normals[spec["target_id"]], dataset_root=SOURCE_ROOT)
            cache[spec["target_id"]] = estimate_normal_structure(loaded, config=config)
        target = cache[spec["target_id"]]
        a2, official, _template = placement_from_official_a2(
            generator,
            target,
            spec["defect_type"],
            spec["seed"],
            min_shape_pixels=MIN_SHAPE_PIXELS,
            return_template=True,
        )
        if a2.source_id != spec["source_id"]:
            raise RuntimeError(f"{spec['job_id']} source drifted")
        support = np.ascontiguousarray(a2.target_support & target.valid, dtype=bool)
        official_mask = np.asarray(official.mask, dtype=bool)
        noclip = np.asarray(
            official.unclipped if official.unclipped is not None else official.mask,
            dtype=bool,
        )
        unit = _unit_scale_overflow(
            np.asarray(a2.source_gt, dtype=bool),
            support,
            np.asarray(a2.similarity, dtype=np.float32),
            spec["seed"],
            areas[spec["defect_type"]],
        )
        normal = np.ascontiguousarray(target.normal.rgb)
        layout = Path(spec["layout"])
        test_rgb = _open_rgb(_find_image(layout / "test" / spec["type_slug"], spec["test_stem"]))
        test_gt = _open_mask(
            _mask_path(layout / "ground_truth", spec["type_slug"], spec["test_stem"]),
            test_rgb.shape[:2],
        )
        cells = [
            _cell(_paint(test_rgb, test_gt, (40, 220, 80), 0.45), f"测试GT\n{spec['test_stem']}"),
            _cell(_paint(normal, official_mask, (240, 40, 180), 0.55), f"正式\n{int(official_mask.sum())} 外{_outside(official_mask, support)}"),
            _cell(_paint(normal, noclip, (40, 200, 220), 0.5), f"同尺度不裁\n{int(noclip.sum())} 外{_outside(noclip, support)}"),
            _cell(_paint(normal, unit, (255, 180, 40), 0.5), f"不按双实例缩小\n{int(unit.sum())} 外{_outside(unit, support)}"),
        ]
        strip = _join(cells, f"{spec['category']}/{spec['type_slug']} {spec['job_id']}")
        Image.fromarray(strip).save(rows_dir / f"{spec['job_id']}.png")
        strips.append(strip)
        row = {
            "job_id": spec["job_id"],
            "official_px": int(official_mask.sum()),
            "official_out": _outside(official_mask, support),
            "noclip_px": int(noclip.sum()),
            "noclip_out": _outside(noclip, support),
            "noclip_cc": n_connected(noclip),
            "unit_px": int(unit.sum()),
            "unit_out": _outside(unit, support),
            "unit_cc": n_connected(unit),
        }
        stats.append(row)
        print(spec["job_id"], row, flush=True)
    Image.fromarray(_stack(strips, "正式 | 同尺度保留伸出 | 不按源物体面积缩小并保留伸出")).save(destination / "sheet.png")
    payload = {"schema_version": SCHEMA, "diagnostic_only": True, "rows": stats, "destination": str(destination)}
    write_json_atomic(destination / "summary.json", payload)
    return payload

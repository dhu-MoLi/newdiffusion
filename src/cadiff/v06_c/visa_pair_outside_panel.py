"""Look panel: pair-scale placement that may hang outside TokenCut.

Frozen identities from the 2026-10-05 q-relax sheet. One change on stuck_together:
scale = sqrt(2) * object_scale * factor, overlap the whole target TokenCut, keep the
unclipped mask. small_scratches stays official. R is the already trained checkpoint.
Does not write generate dumps and does not change official A/H.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.capsules_combined_panel import _contained_output, _find_image, n_connected
from cadiff.v06_c.loc_pred_gt_overlay import _mask_path
from cadiff.v06_c.official_a2_place import placement_from_official_a2
from cadiff.v06_c.pair_outside_place import place_pair_outside
from cadiff.v06_c.visa_exact_mask_r import load_visa_renderer, visa_p2_a1_config
from cadiff.v06_c.visa_flow_steps_panel import _cell, _join, _open_mask, _open_rgb, _paint, _stack
from cadiff.v06_c.visa_pilot import (
    SOURCE_ROOT,
    fewshot_identity_records,
    load_visa_identity,
    load_visa_magic_inventory,
    load_visa_seas_inventory,
    normal_identity_records,
)

SCHEMA = "v06_c_visa_pair_outside_panel_v1"
RUNS = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs")
PATHS = Path("configs/paths/omen-home.yaml")
COLUMNS = (
    "测试图+GT unpaired",
    "源图+源GT",
    "TokenCut 物体",
    "正式 m_final",
    "正式 final_y",
    "双实例 青=物体上 红=物体外",
    "同一 R 的 final_y",
)

# Written to disk before any replay. Same jobs as visa_q_relax_20261005.
PAIR_ROWS: tuple[dict[str, Any], ...] = (
    {
        "category": "cashew",
        "split": "seas",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "096",
        "job_id": "stuck_together-000-20340912",
        "seed": 20340912,
        "target_id": "000",
        "source_id": "094",
        "layout": RUNS / "v06_c/visa/cashew/mvtec_layout_seas_20260922/cashew",
        "sample_dir": RUNS / "v06_c/visa/cashew/generate/20260922-seas-500/samples/stuck_together-000-20340912",
        "checkpoint": RUNS / "v06_b/visa/cashew/exact-mask-r/seed-20260922-seas/checkpoint",
    },
    {
        "category": "cashew",
        "split": "seas",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "097",
        "job_id": "stuck_together-001-20340913",
        "seed": 20340913,
        "target_id": "001",
        "source_id": "095",
        "layout": RUNS / "v06_c/visa/cashew/mvtec_layout_seas_20260922/cashew",
        "sample_dir": RUNS / "v06_c/visa/cashew/generate/20260922-seas-500/samples/stuck_together-001-20340913",
        "checkpoint": RUNS / "v06_b/visa/cashew/exact-mask-r/seed-20260922-seas/checkpoint",
    },
    {
        "category": "cashew",
        "split": "seas",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "098",
        "job_id": "stuck_together-002-20340914",
        "seed": 20340914,
        "target_id": "002",
        "source_id": "094",
        "layout": RUNS / "v06_c/visa/cashew/mvtec_layout_seas_20260922/cashew",
        "sample_dir": RUNS / "v06_c/visa/cashew/generate/20260922-seas-500/samples/stuck_together-002-20340914",
        "checkpoint": RUNS / "v06_b/visa/cashew/exact-mask-r/seed-20260922-seas/checkpoint",
    },
    {
        "category": "cashew",
        "split": "seas",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "099",
        "job_id": "stuck_together-003-20340915",
        "seed": 20340915,
        "target_id": "003",
        "source_id": "095",
        "layout": RUNS / "v06_c/visa/cashew/mvtec_layout_seas_20260922/cashew",
        "sample_dir": RUNS / "v06_c/visa/cashew/generate/20260922-seas-500/samples/stuck_together-003-20340915",
        "checkpoint": RUNS / "v06_b/visa/cashew/exact-mask-r/seed-20260922-seas/checkpoint",
    },
    {
        "category": "cashew",
        "split": "seas",
        "apply_pair": False,
        "type_slug": "small_scratches",
        "defect_type": "small scratches",
        "test_stem": "083",
        "job_id": "small_scratches-000-20330912",
        "seed": 20330912,
        "target_id": "000",
        "source_id": "080",
        "layout": RUNS / "v06_c/visa/cashew/mvtec_layout_seas_20260922/cashew",
        "sample_dir": RUNS / "v06_c/visa/cashew/generate/20260922-seas-500/samples/small_scratches-000-20330912",
        "checkpoint": RUNS / "v06_b/visa/cashew/exact-mask-r/seed-20260922-seas/checkpoint",
    },
    {
        "category": "pipe_fryum",
        "split": "magic",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "093",
        "job_id": "stuck_together-000-20310912",
        "seed": 20310912,
        "target_id": "000",
        "source_id": "090",
        "layout": RUNS / "v06_c/visa/pipe_fryum/mvtec_layout_magic_20260927/pipe_fryum",
        "sample_dir": RUNS / "v06_c/visa/pipe_fryum/generate/20260927-magic-500/samples/stuck_together-000-20310912",
        "checkpoint": RUNS / "v06_b/visa/pipe_fryum/exact-mask-r/seed-20260927-magic/checkpoint",
    },
    {
        "category": "pipe_fryum",
        "split": "magic",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "094",
        "job_id": "stuck_together-001-20310913",
        "seed": 20310913,
        "target_id": "001",
        "source_id": "091",
        "layout": RUNS / "v06_c/visa/pipe_fryum/mvtec_layout_magic_20260927/pipe_fryum",
        "sample_dir": RUNS / "v06_c/visa/pipe_fryum/generate/20260927-magic-500/samples/stuck_together-001-20310913",
        "checkpoint": RUNS / "v06_b/visa/pipe_fryum/exact-mask-r/seed-20260927-magic/checkpoint",
    },
    {
        "category": "pipe_fryum",
        "split": "magic",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "095",
        "job_id": "stuck_together-002-20310914",
        "seed": 20310914,
        "target_id": "002",
        "source_id": "092",
        "layout": RUNS / "v06_c/visa/pipe_fryum/mvtec_layout_magic_20260927/pipe_fryum",
        "sample_dir": RUNS / "v06_c/visa/pipe_fryum/generate/20260927-magic-500/samples/stuck_together-002-20310914",
        "checkpoint": RUNS / "v06_b/visa/pipe_fryum/exact-mask-r/seed-20260927-magic/checkpoint",
    },
    {
        "category": "pipe_fryum",
        "split": "magic",
        "apply_pair": True,
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "096",
        "job_id": "stuck_together-003-20310915",
        "seed": 20310915,
        "target_id": "003",
        "source_id": "090",
        "layout": RUNS / "v06_c/visa/pipe_fryum/mvtec_layout_magic_20260927/pipe_fryum",
        "sample_dir": RUNS / "v06_c/visa/pipe_fryum/generate/20260927-magic-500/samples/stuck_together-003-20310915",
        "checkpoint": RUNS / "v06_b/visa/pipe_fryum/exact-mask-r/seed-20260927-magic/checkpoint",
    },
)


def _inventory(category: str, split: str) -> Any:
    if split == "seas":
        return load_visa_seas_inventory(SOURCE_ROOT, category)
    if split == "magic":
        return load_visa_magic_inventory(SOURCE_ROOT, category, generate_only=True)
    raise ValueError(f"unsupported split {split}")


def _build_generator(inventory: Any, encoder: Any) -> tuple[Any, dict[str, Any]]:
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
    return generator, normals


def _pair_paint(normal: np.ndarray, on_object: np.ndarray, outside: np.ndarray) -> NDArray[np.uint8]:
    painted = _paint(normal, on_object, (40, 200, 220), 0.55)
    return _paint(painted, outside, (240, 40, 40), 0.65)


def _render_row(
    spec: dict[str, Any],
    generator: Any,
    normals: dict[str, Any],
    structures: dict[str, Any],
    structure_config: Any,
    renderer: Any,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    layout = Path(spec["layout"])
    sample = Path(spec["sample_dir"])
    saved_y = np.asarray(Image.open(sample / "final_y.png").convert("RGB"))
    record = __import__("json").loads((sample / "record.json").read_text(encoding="utf-8"))
    if str(record["source_id"]) != spec["source_id"] or int(record["seed"]) != spec["seed"]:
        raise RuntimeError(f"frozen identity drifted: {spec['job_id']}")
    target_id = spec["target_id"]
    if target_id not in structures:
        loaded = load_visa_identity(normals[target_id], dataset_root=SOURCE_ROOT)
        structures[target_id] = estimate_normal_structure(loaded, config=structure_config)
    target = structures[target_id]
    a2, placement = placement_from_official_a2(
        generator,
        target,
        spec["defect_type"],
        spec["seed"],
    )
    if a2.source_id != spec["source_id"]:
        raise RuntimeError(f"{spec['job_id']} replay source {a2.source_id} != {spec['source_id']}")
    normal = np.ascontiguousarray(target.normal.rgb)
    support = np.ascontiguousarray(a2.target_support & target.valid, dtype=bool)
    official = np.ascontiguousarray(placement.mask, dtype=bool)
    test_dir = layout / "test" / spec["type_slug"]
    gt_root = layout / "ground_truth"
    test_rgb = _open_rgb(_find_image(test_dir, spec["test_stem"]))
    test_gt = _open_mask(_mask_path(gt_root, spec["type_slug"], spec["test_stem"]), test_rgb.shape[:2])
    source_rgb = _open_rgb(_find_image(test_dir, spec["source_id"]))
    source_gt = _open_mask(_mask_path(gt_root, spec["type_slug"], spec["source_id"]), source_rgb.shape[:2])
    stats: dict[str, Any] = {
        "job_id": spec["job_id"],
        "category": spec["category"],
        "apply_pair": bool(spec["apply_pair"]),
        "official_px": int(official.sum()),
        "official_cc": n_connected(official),
        "official_scale": float(placement.scale),
        "support_px": int(support.sum()),
        "source_id": a2.source_id,
    }
    if not spec["apply_pair"]:
        pair_view = np.zeros_like(normal)
        pair_view[:] = 32
        pair_caption = "未触发\n保持正式"
        rendered_view = saved_y
        rendered_caption = "正式 final_y"
        stats.update(pair_failed=False, pair_px=None, outside_px=None, pair_scale=None)
    else:
        fewshot = getattr(a2, "fewshot_gt_areas", None)
        areas = (
            (int(np.asarray(a2.source_gt, dtype=bool).sum()),)
            if fewshot is None
            else tuple(int(value) for value in fewshot)
        )
        try:
            pair = place_pair_outside(
                a2.source_gt,
                a2.source_support,
                support,
                seed=spec["seed"],
                fewshot_areas=areas,
            )
        except EmptyFeasiblePlacement as exc:
            pair_view = np.zeros_like(normal)
            pair_view[:] = (80, 20, 20)
            pair_caption = f"失败\n{exc}"
            rendered_view = np.zeros_like(normal)
            rendered_caption = "未画 R"
            stats.update(pair_failed=True, failure_reason=str(exc), pair_px=None, outside_px=None)
        else:
            raw = renderer(normal, spec["defect_type"], pair.mask, spec["seed"])
            final_y = compose_causal(normal, raw, pair.mask)
            pair_view = _pair_paint(normal, pair.on_object, pair.outside)
            pair_caption = (
                f"×{pair.scale:.2f} {pair.angle_deg}°\n"
                f"{int(pair.mask.sum())} px 外 {int(pair.outside.sum())}"
            )
            rendered_view = final_y
            rendered_caption = f"R y\ncc={n_connected(pair.mask)}"
            stats.update(
                pair_failed=False,
                pair_px=int(pair.mask.sum()),
                outside_px=int(pair.outside.sum()),
                on_object_px=int(pair.on_object.sum()),
                pair_cc=n_connected(pair.mask),
                outside_cc=n_connected(pair.outside),
                pair_scale=float(pair.scale),
                pair_factor=float(pair.factor),
                pair_angle=int(pair.angle_deg),
            )
    cells = [
        _cell(_paint(test_rgb, test_gt, (40, 220, 80), 0.45), f"测试+GT\n{spec['test_stem']} unpaired"),
        _cell(_paint(source_rgb, source_gt, (40, 200, 220), 0.45), f"源+GT\n{spec['source_id']}"),
        _cell(_paint(normal, support, (40, 200, 220), 0.35), f"TokenCut\n{int(support.sum())} px"),
        _cell(
            _paint(normal, official, (240, 40, 180), 0.55),
            f"正式 m\n{int(official.sum())} cc={n_connected(official)}",
        ),
        _cell(saved_y, "正式 y"),
        _cell(pair_view, pair_caption),
        _cell(rendered_view, rendered_caption),
    ]
    title = (
        f"{spec['category']}/{spec['type_slug']}  test {spec['test_stem']}  "
        f"src {spec['source_id']}  job {spec['job_id']}  "
        f"pair={int(bool(spec['apply_pair']))}"
    )
    return _join(cells, title), stats


def run_visa_pair_outside_panel(output: Path, *, device: str = "cuda:0", cwd: Path | None = None) -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    identity_rows = []
    for spec in PAIR_ROWS:
        row = {key: value for key, value in spec.items() if key != "checkpoint"}
        row["layout"] = str(spec["layout"])
        row["sample_dir"] = str(spec["sample_dir"])
        row["checkpoint"] = str(spec["checkpoint"])
        identity_rows.append(row)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": (
                "same jobs as visa_q_relax_20261005; pair arm only on stuck_together; "
                "scale = sqrt(2) * object_scale * factor; unclipped mask may leave TokenCut; "
                "small_scratches stays official"
            ),
            "columns": list(COLUMNS),
            "rows": identity_rows,
            "not_generate": True,
            "not_official": True,
        },
    )
    project = load_config(PATHS if cwd is None else cwd / PATHS)
    encoder = load_official_dinov2_encoder(device=device)
    bundles: dict[tuple[str, str], tuple[Any, dict[str, Any], Any, Any]] = {}
    structures: dict[str, dict[str, Any]] = {}
    strips: list[NDArray[np.uint8]] = []
    stats: list[dict[str, Any]] = []
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    for spec in PAIR_ROWS:
        key = (spec["category"], spec["split"])
        if key not in bundles:
            inventory = _inventory(spec["category"], spec["split"])
            generator, normals = _build_generator(inventory, encoder)
            renderer = load_visa_renderer(
                checkpoint=Path(spec["checkpoint"]),
                model_root=project.model_root,
                config=visa_p2_a1_config(
                    inventory.category,
                    tuple(item.name for item in inventory.types),
                ),
                device=device,
            )
            bundles[key] = (generator, normals, inventory.structure, renderer)
        generator, normals, structure_config, renderer = bundles[key]
        cache = structures.setdefault(spec["category"], {})
        strip, row_stats = _render_row(spec, generator, normals, cache, structure_config, renderer)
        Image.fromarray(strip).save(rows_dir / f"{spec['job_id']}.png")
        strips.append(strip)
        stats.append(row_stats)
        print(spec["job_id"], {k: row_stats[k] for k in row_stats if k != "job_id"}, flush=True)
    Image.fromarray(
        _stack(strips, "左正式  右双实例尺度并保留物体外像素  测试列 unpaired  非正式")
    ).save(destination / "sheet.png")
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "n": len(stats),
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


if __name__ == "__main__":
    run_visa_pair_outside_panel(
        RUNS / "v06_c/diagnostics/visa_pair_outside_20261005",
        device="cuda:0",
        cwd=Path("/home/dell/.codex/worktrees/370b/main-design-v0.6"),
    )

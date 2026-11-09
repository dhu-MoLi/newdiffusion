"""Look panel: copy the reference TokenCut object beside itself.

Same frozen jobs as the pair-scale panel. Stuck-together uses instance copy.
small_scratches stays the landed official sample. No R, no generate dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement
from cadiff.v06_c.capsules_combined_panel import _contained_output, _find_image, n_connected
from cadiff.v06_c.instance_copy import duplicate_instance
from cadiff.v06_c.loc_pred_gt_overlay import _mask_path
from cadiff.v06_c.visa_flow_steps_panel import _cell, _join, _open_mask, _open_rgb, _paint, _stack
from cadiff.v06_c.visa_pair_outside_panel import PAIR_ROWS
from cadiff.v06_c.visa_pilot import (
    SOURCE_ROOT,
    load_visa_identity,
    load_visa_magic_inventory,
    load_visa_seas_inventory,
    normal_identity_records,
)

SCHEMA = "v06_c_visa_instance_copy_panel_v1"
RUNS = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs")


def _inventory(category: str, split: str) -> Any:
    if split == "seas":
        return load_visa_seas_inventory(SOURCE_ROOT, category)
    if split == "magic":
        return load_visa_magic_inventory(SOURCE_ROOT, category, generate_only=True)
    raise ValueError(f"unsupported split {split}")


def _render_row(
    spec: dict[str, Any],
    loaded: Any,
    support: np.ndarray,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    layout = Path(spec["layout"])
    sample = Path(spec["sample_dir"])
    normal = np.ascontiguousarray(loaded.rgb)
    official_mask = np.asarray(Image.open(sample / "m_final.png").convert("L")) > 127
    official_y = np.asarray(Image.open(sample / "final_y.png").convert("RGB"))
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
        "support_px": int(support.sum()),
        "official_px": int(official_mask.sum()),
        "official_cc": n_connected(official_mask),
    }
    if not spec["apply_pair"]:
        copy_view = np.zeros_like(normal)
        copy_view[:] = 32
        copy_caption = "未触发\n保持正式"
        pasted = official_y
        pasted_caption = "正式 y"
        stats.update(copy_failed=False, copy_px=None, extra_px=None)
    else:
        try:
            copied = duplicate_instance(normal, support, seed=int(spec["seed"]))
        except EmptyFeasiblePlacement as exc:
            copy_view = np.zeros_like(normal)
            copy_view[:] = (80, 20, 20)
            copy_caption = f"失败\n{exc}"
            pasted = np.zeros_like(normal)
            pasted_caption = "未贴"
            stats.update(copy_failed=True, failure_reason=str(exc))
        else:
            painted = _paint(normal, copied.object_mask, (40, 200, 220), 0.45)
            copy_view = _paint(painted, copied.copy_mask, (240, 40, 40), 0.55)
            copy_caption = (
                f"并 {int(copied.mask.sum())} px\n"
                f"外 {int(copied.copy_mask.sum())} 移 {copied.shift_row},{copied.shift_col}"
            )
            pasted = copied.image
            pasted_caption = f"贴图 y\ncc={n_connected(copied.mask)}"
            stats.update(
                copy_failed=False,
                copy_px=int(copied.mask.sum()),
                extra_px=int(copied.copy_mask.sum()),
                copy_cc=n_connected(copied.mask),
                extra_cc=n_connected(copied.copy_mask),
                shift_row=int(copied.shift_row),
                shift_col=int(copied.shift_col),
            )
    cells = [
        _cell(_paint(test_rgb, test_gt, (40, 220, 80), 0.45), f"测试+GT\n{spec['test_stem']} unpaired"),
        _cell(_paint(source_rgb, source_gt, (40, 200, 220), 0.45), f"源+GT\n{spec['source_id']}"),
        _cell(_paint(normal, support, (40, 200, 220), 0.35), f"TokenCut\n{int(support.sum())} px"),
        _cell(
            _paint(normal, official_mask, (240, 40, 180), 0.55),
            f"正式 m\n{int(official_mask.sum())} cc={n_connected(official_mask)}",
        ),
        _cell(official_y, "正式 y"),
        _cell(copy_view, copy_caption),
        _cell(pasted, pasted_caption),
    ]
    title = (
        f"{spec['category']}/{spec['type_slug']}  test {spec['test_stem']}  "
        f"src {spec['source_id']}  job {spec['job_id']}"
    )
    return _join(cells, title), stats


def run_visa_instance_copy_panel(output: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for spec in PAIR_ROWS:
        row = {key: str(value) if isinstance(value, Path) else value for key, value in spec.items()}
        rows.append(row)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": (
                "same jobs as visa_pair_outside_20261005; stuck_together copies the reference "
                "TokenCut object along its long axis and pastes that object's RGB; "
                "mask is the union; small_scratches stays official; no R"
            ),
            "rows": rows,
            "not_generate": True,
            "not_official": True,
        },
    )
    encoder = load_official_dinov2_encoder(device=device)
    normals: dict[tuple[str, str], dict[str, Any]] = {}
    cache: dict[tuple[str, str], Any] = {}
    strips: list[NDArray[np.uint8]] = []
    stats: list[dict[str, Any]] = []
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    for spec in PAIR_ROWS:
        key = (spec["category"], spec["split"])
        if key not in normals:
            inventory = _inventory(spec["category"], spec["split"])
            normals[key] = {record.item_id: record for record in normal_identity_records(inventory)}
        target_id = spec["target_id"]
        cache_key = (spec["category"], target_id)
        if cache_key not in cache:
            cache[cache_key] = load_visa_identity(normals[key][target_id], dataset_root=SOURCE_ROOT)
        loaded = cache[cache_key]
        features = encoder.encode(np.ascontiguousarray(loaded.rgb))
        support = np.ascontiguousarray(
            tokencut_object_mask(features.tokens, loaded.valid) & loaded.valid,
            dtype=bool,
        )
        strip, row_stats = _render_row(spec, loaded, support)
        Image.fromarray(strip).save(rows_dir / f"{spec['job_id']}.png")
        strips.append(strip)
        stats.append(row_stats)
        print(spec["job_id"], row_stats, flush=True)
    Image.fromarray(
        _stack(strips, "左正式  右把参考物体沿长轴复制并贴上自身 RGB  测试列 unpaired  非正式")
    ).save(destination / "sheet.png")
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "not_train_r": True,
        "n": len(stats),
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


if __name__ == "__main__":
    run_visa_instance_copy_panel(RUNS / "v06_c/diagnostics/visa_instance_copy_fit_20261005", device="cuda:0")

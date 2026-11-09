"""Wave 2: synthetic vs GT mask density, frozen size bins, and problem IDs.

CPU only. Reads landed 500-pair dumps. Does not train, does not rerun generate,
does not load DualAnoDiff checkpoints. Size-stratified F1/AUPRO needs pixel
maps that eval_kit did not save; that step is not in this diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_c.wave1_label_geom import (
    _VISA_CATEGORIES,
    _cable_gt_masks,
    _can_gt_masks,
    _resolve_visa_dump,
    _visa_gt_masks,
    mask_area,
)

SIZE_EDGES = (64, 1024)
FLOOR_FRAC_MIN = 0.05
FLOOR_AREA_RATIO = 2.0
TINY_GT_MEDIAN = 16
SMALLER_RATIO = 0.6
FATTER_RATIO = 2.0
COLLAPSE_DENSITY_MAX = 0.40
MISMATCH_COSINE = 0.50
EMPTY_IOU_FRAC = 0.10
SCHEMA = "v06_c_wave2_density_size_v1"
_FONT = ImageFont.load_default()
_OPEN = (
    "appearance_unreadability",
    "quality_sample_size",
    "size_stratified_f1_aupro",
)


def size_bin(area: int) -> str:
    if type(area) is not int or area < 1:
        raise ValueError("area must be a positive integer")
    if area < SIZE_EDGES[0]:
        return "small"
    if area < SIZE_EDGES[1]:
        return "medium"
    return "large"


def density_map(masks: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("density needs at least one mask")
    stacked = np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0)
    if stacked.ndim != 3:
        raise ValueError("each mask must be 2D")
    return stacked.mean(axis=0).astype(np.float64)


def density_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("density maps must match")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("density maps must be non-empty")
    return float(np.dot(a, b) / denom)


def determine_problems(stats: Mapping[str, float]) -> list[dict[str, Any]]:
    fired: list[dict[str, Any]] = []
    gt_median = float(stats["gt_median_area"])
    placed_median = float(stats["placed_median_area"])
    floor_median = float(stats["floor_median_area"])
    floor_frac = float(stats["floor_frac"])
    if floor_frac >= FLOOR_FRAC_MIN and gt_median > 0 and floor_median / gt_median >= FLOOR_AREA_RATIO:
        fired.append(
            {
                "id": "floor_copies_site",
                "because": "FLOOR fraction and FLOOR area vs GT median",
            }
        )
    if gt_median < TINY_GT_MEDIAN:
        fired.append({"id": "tiny_heldout_gt", "because": "held-out GT median below 16px"})
    if gt_median > 0 and placed_median / gt_median < SMALLER_RATIO:
        fired.append({"id": "synth_smaller_than_gt", "because": "placed median / GT median"})
    if gt_median > 0 and placed_median / gt_median > FATTER_RATIO:
        fired.append({"id": "synth_fatter_than_gt", "because": "placed median / GT median"})
    if float(stats["synth_density_max"]) >= COLLAPSE_DENSITY_MAX:
        fired.append({"id": "spatial_collapse", "because": "max synthetic density"})
    if float(stats["density_cosine"]) < MISMATCH_COSINE:
        fired.append({"id": "location_mismatch", "because": "density cosine vs GT"})
    if float(stats["frac_iou_below_0_6"]) > EMPTY_IOU_FRAC:
        fired.append({"id": "empty_interior_thr1", "because": "Wave 1 IoU(R,m) at threshold 1"})
    return fired


def _bin_counts(areas: list[int]) -> dict[str, int]:
    counts = {"small": 0, "medium": 0, "large": 0}
    for area in areas:
        counts[size_bin(area)] += 1
    return counts


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array > 127


def read_dump_masks(dump: Path) -> tuple[dict[str, Any], ...]:
    samples = dump / "samples"
    if not samples.is_dir():
        raise FileNotFoundError(f"missing dump samples: {samples}")
    rows: list[dict[str, Any]] = []
    for record_path in sorted(samples.glob("*/record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if bool(record.get("failed")):
            continue
        mask = _read_mask(record_path.parent / "m_final.png")
        if not mask.any():
            continue
        rows.append(
            {
                "defect_type": str(record["defect_type"]),
                "used_proposal_floor": bool(record.get("used_proposal_floor")),
                "area": mask_area(mask),
                "mask": mask,
            }
        )
    if not rows:
        raise ValueError(f"dump has no usable masks: {dump}")
    return tuple(rows)


def _colorize(density: np.ndarray, vmax: float) -> np.ndarray:
    scale = float(vmax) if vmax > 0 else 1.0
    x = np.clip(np.asarray(density, dtype=np.float64) / scale, 0.0, 1.0)
    rgb = np.zeros(x.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * np.maximum(2 * x - 0.5, 0.0), 0, 255)
    rgb[..., 1] = np.clip(255 * np.minimum(2 * x, 2 - 2 * x), 0, 255)
    rgb[..., 2] = np.clip(255 * np.maximum(1.0 - 1.5 * x, 0.0), 0, 255)
    return rgb


def _view(rgb: np.ndarray, min_side: int = 256) -> np.ndarray:
    height, width = rgb.shape[:2]
    if min(height, width) >= min_side:
        return rgb
    scale = int(np.ceil(min_side / min(height, width)))
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def _save_pair(path: Path, synth: np.ndarray, gt: np.ndarray, title: str) -> None:
    vmax = float(max(synth.max(), gt.max(), 1e-9))
    left = _view(_colorize(synth, vmax))
    right = _view(_colorize(gt, vmax))
    gap = 12
    header = 28
    height = header + left.shape[0]
    width = left.shape[1] + gap + right.shape[1]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[header : header + left.shape[0], : left.shape[1]] = left
    canvas[header : header + right.shape[0], left.shape[1] + gap :] = right
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), f"{title}  synth|GT  vmax={vmax:.3f}", fill=(20, 20, 20), font=_FONT)
    image.save(path)


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def _wave1_iou_frac(wave1_summary: Mapping[str, Any], name: str) -> float:
    arms = wave1_summary.get("arms", {})
    arm = arms.get(name, {}) if isinstance(arms, Mapping) else {}
    synth = arm.get("synth", {}) if isinstance(arm, Mapping) else {}
    value = synth.get("frac_iou_below_0_6", 0.0) if isinstance(synth, Mapping) else 0.0
    return float(value)


def _arm_report(
    name: str,
    dump: Path,
    gt_masks: Mapping[str, tuple[NDArray[np.bool_], ...]],
    wave1_summary: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    rows = read_dump_masks(dump)
    synth_masks = tuple(row["mask"] for row in rows)
    gt_list = tuple(mask for masks in gt_masks.values() for mask in masks)
    synth_density = density_map(synth_masks)
    gt_density = density_map(gt_list)
    placed = [row for row in rows if not row["used_proposal_floor"]]
    floor = [row for row in rows if row["used_proposal_floor"]]
    placed_areas = [int(row["area"]) for row in placed] or [int(row["area"]) for row in rows]
    gt_areas = [mask_area(mask) for mask in gt_list]
    floor_areas = [int(row["area"]) for row in floor]
    stats = {
        "floor_frac": float(len(floor) / len(rows)),
        "floor_median_area": float(np.median(floor_areas)) if floor_areas else 0.0,
        "placed_median_area": float(np.median(placed_areas)),
        "gt_median_area": float(np.median(gt_areas)),
        "synth_density_max": float(synth_density.max()),
        "density_cosine": density_cosine(synth_density, gt_density),
        "frac_iou_below_0_6": _wave1_iou_frac(wave1_summary, name),
    }
    _save_pair(output / f"density_{name}.png", synth_density, gt_density, name)
    if floor:
        floor_density = density_map(tuple(row["mask"] for row in floor))
        _save_pair(
            output / f"density_{name}_floor.png",
            floor_density,
            gt_density,
            f"{name} FLOOR",
        )
    if placed:
        placed_density = density_map(tuple(row["mask"] for row in placed))
        _save_pair(
            output / f"density_{name}_placed.png",
            placed_density,
            gt_density,
            f"{name} placed",
        )
    problems = determine_problems(stats)
    return {
        "name": name,
        "dump": str(dump),
        "n_synth": len(rows),
        "n_gt": len(gt_list),
        "n_floor": len(floor),
        "size_bins": {
            "edges": list(SIZE_EDGES),
            "synth_placed": _bin_counts(placed_areas),
            "synth_floor": _bin_counts(floor_areas) if floor_areas else {"small": 0, "medium": 0, "large": 0},
            "gt": _bin_counts(gt_areas),
        },
        "stats": stats,
        "problems": problems,
    }


def run_wave2_density_size(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    output: Path,
    wave1_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = wave1_summary or {"arms": {}}
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Any] = {}
    all_problems: dict[str, list[dict[str, Any]]] = {}
    for name, spec in arms.items():
        print(name, "START", flush=True)
        reports[name] = _arm_report(
            name,
            Path(spec["dump"]),
            spec["gt_masks"],
            summary,
            destination,
        )
        all_problems[name] = reports[name]["problems"]
        print(name, "DONE", [item["id"] for item in all_problems[name]], flush=True)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "size_edges": list(SIZE_EDGES),
        "rules": {
            "floor_frac_min": FLOOR_FRAC_MIN,
            "floor_area_ratio": FLOOR_AREA_RATIO,
            "tiny_gt_median": TINY_GT_MEDIAN,
            "smaller_ratio": SMALLER_RATIO,
            "fatter_ratio": FATTER_RATIO,
            "collapse_density_max": COLLAPSE_DENSITY_MAX,
            "mismatch_cosine": MISMATCH_COSINE,
            "empty_iou_frac": EMPTY_IOU_FRAC,
        },
        "open_unresolved": list(_OPEN),
        "arm_names": list(reports),
        "arms": reports,
        "problems_by_arm": all_problems,
    }
    write_json_atomic(destination / "summary.json", payload)
    write_json_atomic(
        destination / "problems.json",
        {
            "schema_version": SCHEMA,
            "rules": payload["rules"],
            "open_unresolved": list(_OPEN),
            "by_arm": all_problems,
        },
    )
    return payload


def run_wave2_density_size_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    cable_dump: Path,
    can_dump: Path,
    can_layout: Path,
    visa_root: Path,
    wave1_summary: Path,
    output: Path,
    cwd: Path,
    visa_dump_root: Path | None = None,
    candle_dump: Path | None = None,
    capsules_dump: Path | None = None,
    cashew_dump: Path | None = None,
    chewinggum_dump: Path | None = None,
) -> dict[str, Any]:
    config = load_config(paths if paths.is_absolute() else cwd / paths)
    summary = json.loads(
        (wave1_summary if wave1_summary.is_absolute() else cwd / wave1_summary).read_text(
            encoding="utf-8"
        )
    )
    visa_dumps = {
        "candle": _resolve_visa_dump("candle", candle_dump, visa_dump_root),
        "capsules": _resolve_visa_dump("capsules", capsules_dump, visa_dump_root),
        "cashew": _resolve_visa_dump("cashew", cashew_dump, visa_dump_root),
        "chewinggum": _resolve_visa_dump("chewinggum", chewinggum_dump, visa_dump_root),
    }
    arms: dict[str, dict[str, Any]] = {
        "cable": {
            "dump": cable_dump,
            "gt_masks": _cable_gt_masks(a_experiment, config.dataset_root, cwd),
        },
        "can": {
            "dump": can_dump,
            "gt_masks": _can_gt_masks(can_layout if can_layout.is_absolute() else cwd / can_layout),
        },
    }
    for category in _VISA_CATEGORIES:
        arms[category] = {
            "dump": visa_dumps[category],
            "gt_masks": _visa_gt_masks(visa_root, category),
        }
    return run_wave2_density_size(arms=arms, wave1_summary=summary, output=output)


__all__ = [
    "SIZE_EDGES",
    "density_cosine",
    "density_map",
    "determine_problems",
    "read_dump_masks",
    "run_wave2_density_size",
    "run_wave2_density_size_from_cli",
    "size_bin",
]

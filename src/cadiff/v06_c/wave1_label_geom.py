"""Wave 1 diagnostic: synthetic m_final vs held-out GT geometry, plus IoU(R, m).

Reads landed 500-pair dumps. Does not train, does not call generate(), does not
overwrite those dumps. R is the thresholded mask-interior |y-x|.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import load_v06_a_identity_manifest

CHANGE_THRESHOLD = 1
IOU_REFERENCE = 0.6
SCHEMA = "v06_c_wave1_label_geom_v1"
_EIGHT = np.ones((3, 3), dtype=np.uint8)
_VISA_DUMP_REL = Path("generate") / "20260913-place-cc-500"
_VISA_CATEGORIES = ("candle", "capsules", "cashew", "chewinggum")
_FONT = ImageFont.load_default()


def mask_area(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be 2D")
    return int(binary.sum())


def component_count(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be 2D")
    if not binary.any():
        return 0
    _labels, count = label(binary, structure=_EIGHT)
    return int(count)


def changed_region(
    normal: np.ndarray,
    final_y: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: int = CHANGE_THRESHOLD,
) -> np.ndarray:
    if threshold < 1:
        raise ValueError("threshold must be a positive integer")
    rgb_n = np.asarray(normal)
    rgb_y = np.asarray(final_y)
    binary = np.asarray(mask, dtype=bool)
    if rgb_n.shape != rgb_y.shape or rgb_n.ndim != 3 or rgb_n.shape[2] != 3:
        raise ValueError("normal and final_y must be HxWx3")
    if binary.shape != rgb_n.shape[:2]:
        raise ValueError("mask shape must match the image spatial size")
    delta = np.max(np.abs(rgb_y.astype(np.int16) - rgb_n.astype(np.int16)), axis=2)
    return np.ascontiguousarray(binary & (delta >= threshold))


def realization_iou(changed: np.ndarray, mask: np.ndarray) -> float:
    region = np.asarray(changed, dtype=bool)
    binary = np.asarray(mask, dtype=bool)
    if region.shape != binary.shape:
        raise ValueError("changed and mask shapes must match")
    if not binary.any():
        raise ValueError("empty mask")
    if (region & ~binary).any():
        raise ValueError("changed region escaped the mask")
    union = int((region | binary).sum())
    return float(region.sum() / union)


def value_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    ordered = np.sort(array)
    return {
        "n": int(array.size),
        "min": float(ordered[0]),
        "p10": float(np.quantile(ordered, 0.10)),
        "p50": float(np.quantile(ordered, 0.50)),
        "p90": float(np.quantile(ordered, 0.90)),
        "max": float(ordered[-1]),
        "mean": float(array.mean()),
    }


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array > 127


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def read_dump_samples(dump: Path) -> tuple[dict[str, Any], ...]:
    samples = dump / "samples"
    if not samples.is_dir():
        raise FileNotFoundError(f"missing dump samples: {samples}")
    rows: list[dict[str, Any]] = []
    for record_path in sorted(samples.glob("*/record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if bool(record.get("failed")):
            continue
        sample = record_path.parent
        mask = _read_mask(sample / "m_final.png")
        if not mask.any():
            continue
        changed = changed_region(
            _read_rgb(sample / "normal.png"),
            _read_rgb(sample / "final_y.png"),
            mask,
        )
        rows.append(
            {
                "job_id": record.get("job_id", sample.name),
                "category": record.get("category"),
                "defect_type": str(record["defect_type"]),
                "used_proposal_floor": bool(record.get("used_proposal_floor")),
                "area": mask_area(mask),
                "n_cc": component_count(mask),
                "iou": realization_iou(changed, mask),
                "changed_pixels": int(changed.sum()),
            }
        )
    if not rows:
        raise ValueError(f"dump has no usable samples: {dump}")
    return tuple(rows)


def _summarize_rows(rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> dict[str, Any]:
    areas = np.asarray([row["area"] for row in rows], dtype=np.float64)
    ccs = np.asarray([row["n_cc"] for row in rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "n": len(rows),
        "area": value_summary(areas),
        "n_cc": value_summary(ccs),
        "frac_cc_eq_1": float(np.mean(ccs == 1.0)),
    }
    if rows and "iou" in rows[0]:
        ious = np.asarray([row["iou"] for row in rows], dtype=np.float64)
        summary["iou"] = value_summary(ious)
        summary["frac_iou_below_0_6"] = float(np.mean(ious < IOU_REFERENCE))
        summary["floor_rows"] = int(sum(bool(row.get("used_proposal_floor")) for row in rows))
    return summary


def _histogram_png(
    path: Path,
    series: Mapping[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    integer: bool = False,
) -> None:
    arrays = {name: np.asarray(values, dtype=np.float64).reshape(-1) for name, values in series.items()}
    if any(values.size == 0 for values in arrays.values()):
        raise ValueError("histogram series must be non-empty")
    lo = min(float(values.min()) for values in arrays.values())
    hi = max(float(values.max()) for values in arrays.values())
    if integer:
        left = int(np.floor(lo))
        right = int(np.ceil(hi))
        edges = np.arange(left, right + 2, dtype=np.float64) - 0.5
    else:
        span = hi - lo
        pad = 0.05 * span if span > 0 else 1.0
        edges = np.linspace(lo - pad, hi + pad, 25)
    colors = ((40, 110, 200), (200, 70, 40), (40, 150, 90))
    width, height = 960, 420
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 70, 36, width - 24, height - 48
    plot_w = right - left
    plot_h = bottom - top
    names = tuple(arrays)
    bar_group = max(plot_w / max(len(edges) - 1, 1), 1.0)
    peak = 0.0
    histograms: list[np.ndarray] = []
    for values in arrays.values():
        counts, _ = np.histogram(values, bins=edges)
        frac = counts.astype(np.float64) / max(values.size, 1)
        histograms.append(frac)
        peak = max(peak, float(frac.max()))
    peak = peak if peak > 0 else 1.0
    draw.rectangle((left, top, right, bottom), outline=(30, 30, 30), width=1)
    for index, (name, frac) in enumerate(zip(names, histograms, strict=True)):
        color = colors[index % len(colors)]
        offset = (index - (len(names) - 1) / 2) * (bar_group / max(len(names) + 1, 2))
        bar_w = max(bar_group / (len(names) + 1), 1.0)
        for bin_index, value in enumerate(frac):
            x = left + (bin_index + 0.5) * bar_group + offset
            h = (value / peak) * (plot_h - 4)
            draw.rectangle(
                (x - bar_w / 2, bottom - h, x + bar_w / 2, bottom),
                fill=color,
            )
        draw.rectangle((width - 210, 12 + index * 14, width - 196, 22 + index * 14), fill=color)
        draw.text((width - 190, 10 + index * 14), name, fill=(20, 20, 20), font=_FONT)
    draw.text((12, 8), title, fill=(20, 20, 20), font=_FONT)
    draw.text((left, height - 28), xlabel, fill=(20, 20, 20), font=_FONT)
    draw.text((8, top), "frac", fill=(20, 20, 20), font=_FONT)
    image.save(path)


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _arm_report(
    name: str,
    dump: Path,
    gt_masks: Mapping[str, tuple[np.ndarray, ...]],
    output: Path,
) -> dict[str, Any]:
    synth = read_dump_samples(dump)
    gt_rows: list[dict[str, Any]] = []
    for defect_type, masks in gt_masks.items():
        for index, mask in enumerate(masks):
            gt_rows.append(
                {
                    "defect_type": defect_type,
                    "index": index,
                    "area": mask_area(mask),
                    "n_cc": component_count(mask),
                }
            )
    if not gt_rows:
        raise ValueError(f"{name} has no held-out GT masks")
    placed = tuple(row for row in synth if not row["used_proposal_floor"])
    floor = tuple(row for row in synth if row["used_proposal_floor"])
    synth_areas = np.asarray([row["area"] for row in synth], dtype=np.float64)
    gt_areas = np.asarray([row["area"] for row in gt_rows], dtype=np.float64)
    synth_cc = np.asarray([row["n_cc"] for row in synth], dtype=np.float64)
    gt_cc = np.asarray([row["n_cc"] for row in gt_rows], dtype=np.float64)
    _histogram_png(
        output / f"hist_area_{name}.png",
        {"synth m_final": synth_areas, "held-out GT": gt_areas},
        title=f"{name} mask area (px)",
        xlabel="pixels",
    )
    _histogram_png(
        output / f"hist_cc_{name}.png",
        {"synth m_final": synth_cc, "held-out GT": gt_cc},
        title=f"{name} connected-component count",
        xlabel="8-connected components",
        integer=True,
    )
    _histogram_png(
        output / f"hist_iou_{name}.png",
        {"IoU(R, m)": np.asarray([row["iou"] for row in synth], dtype=np.float64)},
        title=f"{name} realization IoU  threshold={CHANGE_THRESHOLD}",
        xlabel="IoU(R, m)",
    )
    report = {
        "name": name,
        "dump": str(dump),
        "synth": _summarize_rows(synth),
        "synth_placed": _summarize_rows(placed) if placed else {"n": 0},
        "synth_floor": _summarize_rows(floor) if floor else {"n": 0},
        "gt": _summarize_rows(gt_rows),
        "median_area_synth_over_gt": float(np.median(synth_areas) / max(np.median(gt_areas), 1e-9)),
    }
    _write_jsonl(output / f"rows_synth_{name}.jsonl", synth)
    _write_jsonl(output / f"rows_gt_{name}.jsonl", gt_rows)
    return report


def run_wave1_label_geom(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Any] = {}
    for name, spec in arms.items():
        print(name, "START", flush=True)
        reports[name] = _arm_report(
            name,
            Path(spec["dump"]),
            spec["gt_masks"],
            destination,
        )
        print(name, "DONE", reports[name]["synth"]["n"], "synth", reports[name]["gt"]["n"], "gt", flush=True)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "change_threshold": CHANGE_THRESHOLD,
        "iou_reference": IOU_REFERENCE,
        "connectivity": 8,
        "arm_names": list(reports),
        "arms": reports,
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def _cable_gt_masks(a_experiment: Path, dataset_root: Path, cwd: Path) -> dict[str, tuple[NDArray[np.bool_], ...]]:
    experiment = load_v06_a_experiment(a_experiment if a_experiment.is_absolute() else cwd / a_experiment)
    manifest = load_v06_a_identity_manifest(
        cwd / experiment.identity_manifest,
        experiment=experiment,
    )
    grouped: dict[str, list[NDArray[np.bool_]]] = {}
    for record in manifest.open_report:
        loaded = load_identity_record(record, dataset_root=dataset_root, config=experiment)
        if loaded.anomaly_gt is None:
            raise ValueError(f"cable open-report missing GT: {record.sample_id}")
        grouped.setdefault(record.defect_type, []).append(np.asarray(loaded.anomaly_gt, dtype=bool))
    return {name: tuple(masks) for name, masks in grouped.items()}


def _can_gt_masks(can_layout: Path) -> dict[str, tuple[NDArray[np.bool_], ...]]:
    from cadiff.v06_c.can_mvtec2 import (
        SOURCE_ROOT,
        load_can_identity,
        mask_path_for,
        stage_can_mvtec_layout,
    )
    from cadiff.v06_c import can_mvtec2 as can_mod

    inventory = stage_can_mvtec_layout(source_root=SOURCE_ROOT, layout_root=can_layout)
    dataset_root = inventory.layout_root
    gt_root = dataset_root / "can" / "ground_truth" / "bad"
    masks: list[NDArray[np.bool_]] = []
    for path in inventory.heldout:
        record = can_mod._record(
            defect_type="bad",
            item_id=path.stem,
            image_relpath=can_mod._relpath(dataset_root, path),
            mask_relpath=can_mod._relpath(
                dataset_root,
                mask_path_for(path, ground_truth_root=gt_root),
            ),
            role="open_report",
        )
        loaded = load_can_identity(record, dataset_root=dataset_root)
        if loaded.anomaly_gt is None:
            raise ValueError(f"can held-out missing GT: {record.item_id}")
        masks.append(np.asarray(loaded.anomaly_gt, dtype=bool))
    return {"bad": tuple(masks)}


def _visa_gt_masks(visa_root: Path, category: str) -> dict[str, tuple[NDArray[np.bool_], ...]]:
    from cadiff.v06_c.visa_pilot import heldout_identity_records, load_visa_identity, load_visa_inventory

    inventory = load_visa_inventory(visa_root, category)
    grouped: dict[str, list[NDArray[np.bool_]]] = {}
    for record in heldout_identity_records(inventory):
        loaded = load_visa_identity(record, dataset_root=visa_root)
        if loaded.anomaly_gt is None:
            raise ValueError(f"VisA held-out missing GT: {record.sample_id}")
        grouped.setdefault(record.defect_type, []).append(np.asarray(loaded.anomaly_gt, dtype=bool))
    return {name: tuple(masks) for name, masks in grouped.items()}


def _resolve_visa_dump(category: str, explicit: Path | None, dump_root: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if dump_root is None:
        raise ValueError(f"missing {category} dump")
    return dump_root / category / _VISA_DUMP_REL


def run_wave1_label_geom_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    cable_dump: Path,
    can_dump: Path,
    can_layout: Path,
    visa_root: Path,
    output: Path,
    cwd: Path,
    visa_dump_root: Path | None = None,
    candle_dump: Path | None = None,
    capsules_dump: Path | None = None,
    cashew_dump: Path | None = None,
    chewinggum_dump: Path | None = None,
) -> dict[str, Any]:
    config = load_config(paths if paths.is_absolute() else cwd / paths)
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
    return run_wave1_label_geom(arms=arms, output=output)


__all__ = [
    "CHANGE_THRESHOLD",
    "IOU_REFERENCE",
    "changed_region",
    "component_count",
    "mask_area",
    "read_dump_samples",
    "realization_iou",
    "run_wave1_label_geom",
    "run_wave1_label_geom_from_cli",
    "value_summary",
]

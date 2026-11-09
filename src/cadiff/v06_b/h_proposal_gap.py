"""Diagnose H train/infer proposal mismatch. No new weights, no A rerun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label

from cadiff.config import ProjectConfig, load_config
from cadiff.p1_v1.dataset import load_training_index
from cadiff.p2_a0.fewshot import load_fewshot_split
from cadiff.p2_a1.config import load_p2_a1_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_b.config import V06BMaskFirstConfig, is_cable_scheme2, load_v06_b_config
from cadiff.v06_b.dataset import build_geometry_pair, build_unpaired_schedule, split_planner_targets
from cadiff.v06_b.mask_first_runner import (
    MaskFirstV06BContext,
    _fixed_panel_target_ids,
    load_mask_first_source_prompts,
    mask_first_data_authority_model,
    validate_mask_first_source_binding,
)

_EIGHT = np.ones((3, 3), dtype=np.uint8)
_GENERATE_REQUIRED = ("proposal.png", "m_final.png", "record.json")


def mask_geometry_stats(mask: np.ndarray) -> dict[str, float | int]:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    binary = np.asarray(mask, dtype=bool)
    pixels = int(binary.sum())
    if pixels == 0:
        return {
            "pixels": 0,
            "n_cc": 0,
            "max_cc_pixels": 0,
            "max_cc_fraction": 0.0,
            "bbox_area": 0,
            "bbox_fill": 0.0,
        }
    labels, count = label(binary, structure=_EIGHT)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    max_cc = int(sizes.max())
    rows, columns = np.where(binary)
    bbox_area = int((rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1))
    return {
        "pixels": pixels,
        "n_cc": int(count),
        "max_cc_pixels": max_cc,
        "max_cc_fraction": float(max_cc / pixels),
        "bbox_area": bbox_area,
        "bbox_fill": float(pixels / bbox_area),
    }


def _read_binary_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return array > 127


def _percentiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row["defect_type"]), []).append(row)

    def pack(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {name: _percentiles([float(item[name]) for item in group]) for name in fields}

    return {
        "all": pack(rows),
        "by_defect_type": {name: pack(group) for name, group in sorted(by_type.items())},
        "count": len(rows),
    }


def _scheme2_authority(context: MaskFirstV06BContext):
    config = context.experiment
    if not is_cable_scheme2(config):
        raise ValueError("H proposal-gap diagnostic is only authorized for Cable scheme 2")
    data_model = mask_first_data_authority_model(config, context.source_model)
    validate_mask_first_source_binding(config, context.source_model, context.source_prompts)
    split = load_fewshot_split(context.manifest, context.paths.dataset_root, data_model)
    training_index = load_training_index(
        context.manifest,
        context.paths.dataset_root,
        data_model,
        ignore_unknown_fewshot_types=True,
    )
    panel_ids = _fixed_panel_target_ids(
        training_index.normals,
        normal_count=config.panel_normal_count,
    )
    planner_targets = split_planner_targets(
        training_index.normals,
        panel_ids=panel_ids,
        calibration_count=config.planner_calibration_normal_count,
        seed=config.seed,
    )
    return split, training_index, planner_targets


def _unique_geometry_rows(context: MaskFirstV06BContext) -> list[dict[str, Any]]:
    config = context.experiment
    split, _training_index, planner_targets = _scheme2_authority(context)
    schedule = build_unpaired_schedule(
        sources=split.train,
        targets=planner_targets.training,
        defect_types=config.defect_types,
        steps=config.planner_training_steps,
        seed=config.seed,
    )
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for slot in schedule:
        key = (slot.source.sample_id, slot.target.sample_id, slot.defect_type)
        if key in seen:
            continue
        seen.add(key)
        pair = build_geometry_pair(
            slot.source,
            slot.target,
            image_size=config.image_size,
            proposal_radius=config.proposal_radius,
            support_corner_width=config.support_corner_width,
        )
        proposal = pair.proposal[0].numpy()
        ground_truth = pair.ground_truth[0].numpy()
        stats = mask_geometry_stats(proposal)
        gt_pixels = int(ground_truth.sum())
        rows.append(
            {
                "side": "train_square_dilate",
                "source_id": pair.source_id,
                "target_id": pair.target_id,
                "defect_type": pair.defect_type,
                "gt_pixels": gt_pixels,
                "gt_to_proposal": float(gt_pixels / stats["pixels"]) if stats["pixels"] else 0.0,
                "transfer_recentered": bool(pair.transfer_recentered),
                **stats,
            }
        )
    return rows


def _infer_atlas_rows(generate_root: Path) -> list[dict[str, Any]]:
    if generate_root.is_symlink() or not generate_root.is_dir():
        raise ValueError("generate root must be an existing regular directory")
    records_path = generate_root / "records.jsonl"
    if records_path.is_symlink() or not records_path.is_file():
        raise ValueError("generate records.jsonl is required")
    rows: list[dict[str, Any]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        identity = record["identity"]
        sample_root = generate_root / identity
        missing = [name for name in _GENERATE_REQUIRED if not (sample_root / name).is_file()]
        if missing:
            raise ValueError(f"generate sample is incomplete: {identity}")
        proposal = _read_binary_png(sample_root / "proposal.png")
        final_mask = _read_binary_png(sample_root / "m_final.png")
        stats = mask_geometry_stats(proposal)
        final_stats = mask_geometry_stats(final_mask)
        proposal_pixels = int(stats["pixels"])
        rows.append(
            {
                "side": "infer_atlas",
                "identity": identity,
                "source_id": record.get("source_id"),
                "target_id": record.get("target_id"),
                "defect_type": record["defect_type"],
                "used_geometry_fallback": record.get("used_geometry_fallback"),
                "collapsed": record.get("collapsed"),
                "failed": record.get("failed"),
                "m_final_pixels": final_stats["pixels"],
                "m_final_to_proposal": (
                    float(final_stats["pixels"] / proposal_pixels) if proposal_pixels else 0.0
                ),
                **stats,
            }
        )
    return rows


def run_h_proposal_gap_report(
    *,
    context: MaskFirstV06BContext,
    generate_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("H proposal-gap output must not already exist")
    train_rows = _unique_geometry_rows(context)
    infer_rows = _infer_atlas_rows(generate_root)
    geometry_fields = (
        "pixels",
        "n_cc",
        "max_cc_fraction",
        "bbox_fill",
    )
    report = {
        "schema_version": "v06_b_h_proposal_gap_v1",
        "diagnostic_only": True,
        "not_a_to_h": True,
        "no_new_weights": True,
        "category": context.experiment.category,
        "train_source": "unique SquareDilate_24 unpaired geometry from scheme-2 H schedule",
        "infer_source": str(generate_root),
        "train": {
            "rows": train_rows,
            "summary": _summarize_rows(
                train_rows,
                fields=(*geometry_fields, "gt_to_proposal"),
            ),
        },
        "infer": {
            "rows": infer_rows,
            "summary": _summarize_rows(
                infer_rows,
                fields=(*geometry_fields, "m_final_to_proposal"),
            ),
            "fallback_count": sum(row.get("used_geometry_fallback") is True for row in infer_rows),
            "collapsed_count": sum(row.get("collapsed") is True for row in infer_rows),
            "failed_count": sum(row.get("failed") is True for row in infer_rows),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "report.json", report)
    write_json_atomic(
        output / "summary.json",
        {
            "schema_version": report["schema_version"],
            "diagnostic_only": True,
            "not_a_to_h": True,
            "train_unique_pairs": len(train_rows),
            "infer_identities": len(infer_rows),
            "train": report["train"]["summary"],
            "infer": {
                key: value
                for key, value in report["infer"].items()
                if key != "rows"
            },
        },
    )
    return report


def run_h_proposal_gap_from_cli(
    *,
    paths: Path,
    experiment: Path,
    source_model: Path,
    source_prompts: Path,
    manifest: Path,
    generate_root: Path,
    output: Path,
    cwd: Path,
) -> dict[str, Any]:
    config = load_v06_b_config(experiment)
    if not isinstance(config, V06BMaskFirstConfig):
        raise ValueError("experiment must be a mask-first configuration")
    project = load_config(paths)
    if not isinstance(project, ProjectConfig):
        raise ValueError("paths must load a ProjectConfig")
    context = MaskFirstV06BContext(
        paths=project,
        experiment=config,
        source_model=load_p2_a1_config(source_model),
        source_prompts=load_mask_first_source_prompts(
            source_prompts,
            category=config.category,
        ),
        manifest=manifest,
        output=output,
        cwd=cwd,
    )
    return run_h_proposal_gap_report(
        context=context,
        generate_root=generate_root,
        output=output,
    )


_COMPARE_COLUMNS = ("train_square_dilate", "infer_atlas", "infer_m_final")
_TRAIN_PERCENTILES = (10, 50, 90)


def _pick_percentile_row(rows: list[dict[str, Any]], percent: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot pick a percentile from empty rows")
    ordered = sorted(
        rows,
        key=lambda row: (int(row["pixels"]), str(row["source_id"]), str(row["target_id"])),
    )
    target = float(np.percentile([int(row["pixels"]) for row in ordered], percent))
    return min(
        ordered,
        key=lambda row: (
            abs(int(row["pixels"]) - target),
            str(row["source_id"]),
            str(row["target_id"]),
        ),
    )


def _mask_to_rgb(
    proposal: np.ndarray,
    *,
    ground_truth: np.ndarray | None = None,
    m_final: np.ndarray | None = None,
) -> np.ndarray:
    rgb = np.zeros((*proposal.shape, 3), dtype=np.uint8)
    rgb[proposal] = (210, 210, 210)
    if ground_truth is not None:
        rgb[ground_truth] = (36, 170, 72)
    if m_final is not None:
        rgb[proposal & ~m_final] = (150, 160, 210)
        rgb[m_final] = (230, 170, 36)
    return rgb


def _resize_tile(array: np.ndarray, tile: int) -> Image.Image:
    image = Image.fromarray(array)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((tile, tile), Image.Resampling.NEAREST)


def _draw_panel(
    path: Path,
    rows: list[tuple[str, dict[str, np.ndarray]]],
    *,
    columns: tuple[str, ...],
    tile: int,
) -> None:
    from PIL import ImageDraw

    canvas = Image.new("RGB", (tile * len(columns), 28 + len(rows) * (tile + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(columns):
        draw.text((index * tile + 5, 7), name, fill="black")
    for row_index, (label, arrays) in enumerate(rows):
        top = 28 + row_index * (tile + 22)
        draw.text((5, top + 3), label, fill="black")
        for column_index, name in enumerate(columns):
            canvas.paste(_resize_tile(arrays[name], tile), (column_index * tile, top + 22))
    canvas.save(path)


def _rebuild_train_pair(
    context: MaskFirstV06BContext,
    row: dict[str, Any],
    *,
    sources: dict[str, Any],
    targets: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    config = context.experiment
    pair = build_geometry_pair(
        sources[str(row["source_id"])],
        targets[str(row["target_id"])],
        image_size=config.image_size,
        proposal_radius=config.proposal_radius,
        support_corner_width=config.support_corner_width,
    )
    return pair.proposal[0].numpy(), pair.ground_truth[0].numpy()


def render_h_proposal_gap_panels(
    *,
    context: MaskFirstV06BContext,
    generate_root: Path,
    output: Path,
    report: dict[str, Any] | None = None,
    tile: int = 192,
) -> dict[str, Any]:
    if report is None:
        report_path = output / "report.json"
        if report_path.is_symlink() or not report_path.is_file():
            raise FileNotFoundError("existing H proposal-gap report.json is required")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    train_rows = list(report["train"]["rows"])
    infer_rows = list(report["infer"]["rows"])
    by_train: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        by_train.setdefault(str(row["defect_type"]), []).append(row)
    by_infer: dict[str, list[dict[str, Any]]] = {}
    for row in infer_rows:
        by_infer.setdefault(str(row["defect_type"]), []).append(row)

    split, training_index, _planner_targets = _scheme2_authority(context)
    sources = {sample.sample_id: sample for sample in split.train}
    targets = {item.sample_id: item for item in training_index.normals}

    selected: dict[str, dict[str, dict[str, Any]]] = {}
    train_images: dict[tuple[str, int], np.ndarray] = {}
    for defect_type, rows in by_train.items():
        selected[defect_type] = {}
        for percent in _TRAIN_PERCENTILES:
            picked = _pick_percentile_row(rows, percent)
            selected[defect_type][str(percent)] = {
                "source_id": picked["source_id"],
                "target_id": picked["target_id"],
                "pixels": picked["pixels"],
                "gt_pixels": picked.get("gt_pixels"),
            }
            proposal, ground_truth = _rebuild_train_pair(
                context,
                picked,
                sources=sources,
                targets=targets,
            )
            train_images[(defect_type, percent)] = _mask_to_rgb(
                proposal,
                ground_truth=ground_truth,
            )

    overview_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    written: list[str] = []
    for defect_type in context.experiment.defect_types:
        infer_group = sorted(by_infer[defect_type], key=lambda row: str(row["identity"]))
        identity_rows: list[tuple[str, dict[str, np.ndarray]]] = []
        median_train = train_images[(defect_type, 50)]
        for row in infer_group:
            identity = str(row["identity"])
            atlas = _read_binary_png(generate_root / identity / "proposal.png")
            final_mask = _read_binary_png(generate_root / identity / "m_final.png")
            fallback = " FB" if row.get("used_geometry_fallback") is True else ""
            identity_rows.append(
                (
                    (
                        f"{identity}{fallback} atlas={row['pixels']} "
                        f"m_final={row['m_final_pixels']}"
                    ),
                    {
                        "train_square_dilate": median_train,
                        "infer_atlas": _mask_to_rgb(atlas),
                        "infer_m_final": _mask_to_rgb(atlas, m_final=final_mask),
                    },
                )
            )
            overview_rows.append(identity_rows[-1])
        type_path = output / f"compare_{defect_type}.png"
        _draw_panel(
            type_path,
            [
                (
                    (
                        f"{defect_type} TRAIN p10/p50/p90 px="
                        f"{selected[defect_type]['10']['pixels']}/"
                        f"{selected[defect_type]['50']['pixels']}/"
                        f"{selected[defect_type]['90']['pixels']}  "
                        "(green=transferred GT)"
                    ),
                    {
                        "train_square_dilate": train_images[(defect_type, 10)],
                        "infer_atlas": train_images[(defect_type, 50)],
                        "infer_m_final": train_images[(defect_type, 90)],
                    },
                ),
                *identity_rows,
            ],
            columns=_COMPARE_COLUMNS,
            tile=tile,
        )
        written.append(type_path.name)

    overview = output / "compare_overview.png"
    _draw_panel(overview, overview_rows, columns=_COMPARE_COLUMNS, tile=tile)
    manifest = {
        "schema_version": "v06_b_h_proposal_gap_panel_v1",
        "diagnostic_only": True,
        "not_a_to_h": True,
        "no_new_weights": True,
        "columns": list(_COMPARE_COLUMNS),
        "train_selection": selected,
        "files": ["compare_overview.png", *written],
        "legend": {
            "train_square_dilate": (
                "gray=SquareDilate_24, green=transferred GT; overview/type rows use type p50"
            ),
            "infer_atlas": "gray=atlas m_proposal from scheme-2 generate",
            "infer_m_final": "amber=m_final, blue-gray=proposal leftover",
            "first_row_on_type_panels": "train p10 / p50 / p90 in the three columns",
        },
    }
    write_json_atomic(output / "panel_manifest.json", manifest)
    return manifest


def run_h_proposal_gap_panels_from_cli(
    *,
    paths: Path,
    experiment: Path,
    source_model: Path,
    source_prompts: Path,
    manifest: Path,
    generate_root: Path,
    output: Path,
    cwd: Path,
) -> dict[str, Any]:
    config = load_v06_b_config(experiment)
    if not isinstance(config, V06BMaskFirstConfig):
        raise ValueError("experiment must be a mask-first configuration")
    project = load_config(paths)
    context = MaskFirstV06BContext(
        paths=project,
        experiment=config,
        source_model=load_p2_a1_config(source_model),
        source_prompts=load_mask_first_source_prompts(
            source_prompts,
            category=config.category,
        ),
        manifest=manifest,
        output=output,
        cwd=cwd,
    )
    return render_h_proposal_gap_panels(
        context=context,
        generate_root=generate_root,
        output=output,
    )


__all__ = [
    "mask_geometry_stats",
    "render_h_proposal_gap_panels",
    "run_h_proposal_gap_from_cli",
    "run_h_proposal_gap_panels_from_cli",
    "run_h_proposal_gap_report",
]

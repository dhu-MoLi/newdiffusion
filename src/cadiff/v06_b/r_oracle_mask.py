"""Cable R oracle-mask diagnostic. No new weights, no generate overwrite."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from cadiff.config import ProjectConfig, load_config
from cadiff.p1_v1.dataset import TrainingIndex, TrainingItem, load_training_index
from cadiff.p2_a0.fewshot import FewshotSample, FewshotSplit, load_fewshot_split
from cadiff.p2_a1.config import load_p2_a1_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_b.config import V06BMaskFirstConfig, is_cable_a_to_h, load_v06_b_config
from cadiff.v06_b.dataset import build_geometry_pair
from cadiff.v06_b.h_proposal_gap import mask_geometry_stats
from cadiff.v06_b.mask_first_runner import (
    MaskFirstV06BContext,
    load_mask_first_source_prompts,
    mask_first_data_authority_model,
    validate_mask_first_source_binding,
)
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.cable_generate import load_cable_generate_config, load_cable_runtime

_GENERATE_REQUIRED = (
    "normal.png",
    "proposal.png",
    "m_final.png",
    "raw_r.png",
    "final_y.png",
    "record.json",
)
_COMPARE_COLUMNS = (
    "normal",
    "proposal",
    "pred_m_final",
    "oracle_m_final",
    "pred_final_y",
    "oracle_final_y",
)
_FEWSHOT_COLUMNS = ("anomaly", "gt_mask", "raw_r", "recon_y")
_A_TO_H_B_EXPERIMENT = "configs/experiments/v06_b_cable_mask_first_scheme2_a_to_h.yaml"
_FEWSHOT_RECON_SEED = 20260910
_FROZEN_IDENTITY_COUNT = 32


RendererFn = Callable[[np.ndarray, str, np.ndarray, int], np.ndarray]


def build_oracle_mask(transferred_gt: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    if (
        not isinstance(transferred_gt, np.ndarray)
        or not isinstance(proposal, np.ndarray)
        or transferred_gt.shape != proposal.shape
        or transferred_gt.ndim != 2
    ):
        raise ValueError("transferred GT and atlas proposal must be matching HxW arrays")
    gt = np.asarray(transferred_gt, dtype=bool)
    atlas = np.asarray(proposal, dtype=bool)
    return np.ascontiguousarray(gt & atlas, dtype=bool)


def dice_coefficient(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("dice masks must share one grid")
    intersection = int((a & b).sum())
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def masked_mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return 0.0
    delta = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return float(delta[binary].mean())


def assert_fresh_diagnostic_output(output: Path, generate_root: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("R oracle-mask output must not already exist")
    if generate_root.is_symlink() or not generate_root.is_dir():
        raise ValueError("generate root must be an existing regular directory")
    destination = output.resolve()
    source = generate_root.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("diagnostic output must not overlap the frozen generate root")


def _read_binary_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return np.ascontiguousarray(array > 127)


def _read_rgb_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    if array.dtype != np.uint8:
        raise ValueError(f"RGB png must be uint8: {path}")
    return np.ascontiguousarray(array)


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255)


def _item_id(relpath: str) -> str:
    return PurePosixPath(relpath).stem


def fewshot_uint8_rgb(sample: FewshotSample) -> np.ndarray:
    pixels = sample.pixels.detach().cpu()
    rgb = (
        pixels.add(1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return np.ascontiguousarray(rgb)


def fewshot_binary_mask(sample: FewshotSample) -> np.ndarray:
    mask = sample.mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return np.ascontiguousarray(mask > 0.5)


def lookup_fewshot_source(
    split: FewshotSplit,
    *,
    defect_type: str,
    source_id: str,
) -> FewshotSample:
    matches = [
        sample
        for sample in split.train
        if sample.defect_type == defect_type and _item_id(sample.image_relpath) == source_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one few-shot source {defect_type}/{source_id}")
    return matches[0]


def lookup_training_target(
    training_index: TrainingIndex,
    *,
    target_id: str,
    category: str,
) -> TrainingItem:
    matches = [
        item
        for item in training_index.normals
        if item.category == category
        and item.defect_type == "good"
        and _item_id(item.image_relpath) == target_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one training normal {category}/{target_id}")
    return matches[0]


def _load_generate_records(generate_root: Path) -> list[dict[str, Any]]:
    records_path = generate_root / "records.jsonl"
    if records_path.is_symlink() or not records_path.is_file():
        raise ValueError("generate records.jsonl is required")
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != _FROZEN_IDENTITY_COUNT:
        raise ValueError("Cable R oracle-mask diagnostic requires the frozen 32 P1 identities")
    return rows


def _load_generate_arm(generate_root: Path, identity: str) -> dict[str, np.ndarray]:
    sample_root = generate_root / identity
    missing = [name for name in _GENERATE_REQUIRED if not (sample_root / name).is_file()]
    if missing:
        raise ValueError(f"generate sample is incomplete: {identity}")
    return {
        "normal": _read_rgb_png(sample_root / "normal.png"),
        "proposal": _read_binary_png(sample_root / "proposal.png"),
        "pred_m_final": _read_binary_png(sample_root / "m_final.png"),
        "pred_raw_r": _read_rgb_png(sample_root / "raw_r.png"),
        "pred_final_y": _read_rgb_png(sample_root / "final_y.png"),
    }


def _size_bins(pixels: Sequence[int]) -> list[str]:
    order = sorted(range(len(pixels)), key=lambda index: (int(pixels[index]), index))
    labels = [""] * len(pixels)
    count = len(order)
    for rank, index in enumerate(order):
        if rank < count / 3:
            labels[index] = "small"
        elif rank < 2 * count / 3:
            labels[index] = "medium"
        else:
            labels[index] = "large"
    return labels


def _percentiles(values: Sequence[float]) -> dict[str, float | int]:
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


def _resize_tile(array: np.ndarray, tile: int) -> Image.Image:
    image = Image.fromarray(array)
    if image.mode != "RGB":
        image = image.convert("RGB")
    resampling = (
        Image.Resampling.NEAREST if array.ndim == 2 or array.dtype != np.uint8 else Image.Resampling.LANCZOS
    )
    if array.ndim == 2:
        resampling = Image.Resampling.NEAREST
    return image.resize((tile, tile), resampling)


def _draw_panel(
    path: Path,
    rows: list[tuple[str, Mapping[str, np.ndarray]]],
    *,
    columns: tuple[str, ...],
    tile: int,
) -> None:
    canvas = Image.new("RGB", (tile * len(columns), 28 + len(rows) * (tile + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(columns):
        draw.text((index * tile + 5, 7), name, fill="black")
    for row_index, (label, arrays) in enumerate(rows):
        top = 28 + row_index * (tile + 22)
        draw.text((5, top + 3), label, fill="black")
        for column_index, name in enumerate(columns):
            value = arrays[name]
            if value.ndim == 2:
                tile_array = _mask_to_uint8(value)
            else:
                tile_array = value
            canvas.paste(_resize_tile(tile_array, tile), (column_index * tile, top + 22))
    canvas.save(path)


def _authority(
    context: MaskFirstV06BContext,
) -> tuple[FewshotSplit, TrainingIndex]:
    config = context.experiment
    if not is_cable_a_to_h(config):
        raise ValueError("R oracle-mask diagnostic is only authorized for Cable A→H")
    data_model = mask_first_data_authority_model(config, context.source_model)
    validate_mask_first_source_binding(config, context.source_model, context.source_prompts)
    split = load_fewshot_split(context.manifest, context.paths.dataset_root, data_model)
    training_index = load_training_index(
        context.manifest,
        context.paths.dataset_root,
        data_model,
        ignore_unknown_fewshot_types=True,
    )
    return split, training_index


def _validate_generate_root(generate_root: Path) -> None:
    summary_path = generate_root / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ValueError("generate summary.json is required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("b_experiment") != _A_TO_H_B_EXPERIMENT:
        raise ValueError("generate root must come from frozen A→H Cable B")
    if summary.get("identity_count") != _FROZEN_IDENTITY_COUNT:
        raise ValueError("generate root must contain the frozen 32 P1 identities")


def _renderer_from_runtime(runtime: object) -> RendererFn:
    module_b = getattr(runtime, "module_b", None)
    renderer = getattr(module_b, "renderer", None)
    if renderer is None or not callable(renderer):
        raise ValueError("runtime must expose ExactMaskRenderer on module_b.renderer")
    return renderer


def _failure_rgb(shape: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    image[..., 0] = 160
    return image


def run_r_oracle_mask_diagnostic(
    *,
    context: MaskFirstV06BContext,
    generate_root: Path,
    output: Path,
    runtime: object | None = None,
    render_fn: RendererFn | None = None,
    skip_render: bool = False,
    tile: int = 160,
) -> dict[str, Any]:
    assert_fresh_diagnostic_output(output, generate_root)
    _validate_generate_root(generate_root)
    records = _load_generate_records(generate_root)
    split, training_index = _authority(context)
    renderer = None if skip_render else (render_fn or _renderer_from_runtime(runtime))
    config = context.experiment
    output.mkdir(parents=True, exist_ok=False)
    identities_root = output / "identities"
    identities_root.mkdir()

    identity_rows: list[dict[str, Any]] = []
    panel_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    by_type: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {
        name: [] for name in config.defect_types
    }

    for record in records:
        identity = str(record["identity"])
        defect_type = str(record["defect_type"])
        source_id = str(record["source_id"])
        target_id = str(record["target_id"])
        seed = int(record["seed"])
        arm = _load_generate_arm(generate_root, identity)
        source = lookup_fewshot_source(split, defect_type=defect_type, source_id=source_id)
        target = lookup_training_target(
            training_index,
            target_id=target_id,
            category=config.category,
        )
        pair = build_geometry_pair(
            source,
            target,
            image_size=config.image_size,
            proposal_radius=config.proposal_radius,
            support_corner_width=config.support_corner_width,
        )
        transferred = np.ascontiguousarray(pair.ground_truth[0].cpu().numpy(), dtype=bool)
        oracle = build_oracle_mask(transferred, arm["proposal"])
        oracle_pixels = int(oracle.sum())
        pred_stats = mask_geometry_stats(arm["pred_m_final"])
        oracle_stats = mask_geometry_stats(oracle)
        transferred_stats = mask_geometry_stats(transferred)
        oracle_failed = oracle_pixels == 0
        oracle_reason = "empty_gt_intersection" if oracle_failed else None
        oracle_raw = _failure_rgb(arm["normal"].shape)
        oracle_y = _failure_rgb(arm["normal"].shape)
        if oracle_failed:
            pass
        elif renderer is not None:
            try:
                oracle_raw = renderer(
                    arm["normal"],
                    defect_type,
                    oracle.astype(np.uint8),
                    seed,
                )
                oracle_y = compose_causal(arm["normal"], oracle_raw, oracle)
            except (ValueError, RuntimeError) as error:
                oracle_failed = True
                oracle_reason = str(error)
                oracle_raw = _failure_rgb(arm["normal"].shape)
                oracle_y = _failure_rgb(arm["normal"].shape)
        elif skip_render:
            oracle_raw = np.zeros_like(arm["normal"])
            oracle_y = np.zeros_like(arm["normal"])

        sample_root = identities_root / identity
        sample_root.mkdir()
        Image.fromarray(_mask_to_uint8(oracle)).save(sample_root / "oracle_m_final.png")
        Image.fromarray(oracle_raw).save(sample_root / "oracle_raw_r.png")
        Image.fromarray(oracle_y).save(sample_root / "oracle_final_y.png")
        row = {
            "identity": identity,
            "defect_type": defect_type,
            "source_id": source_id,
            "target_id": target_id,
            "seed": seed,
            "generate_failed": bool(record.get("failed")),
            "used_geometry_fallback": bool(record.get("used_geometry_fallback")),
            "pred_m_final_pixels": int(pred_stats["pixels"]),
            "oracle_m_final_pixels": oracle_pixels,
            "transferred_gt_pixels": int(transferred_stats["pixels"]),
            "proposal_pixels": int(arm["proposal"].sum()),
            "oracle_to_proposal": (
                float(oracle_pixels / int(arm["proposal"].sum())) if arm["proposal"].any() else 0.0
            ),
            "pred_to_proposal": (
                float(int(pred_stats["pixels"]) / int(arm["proposal"].sum()))
                if arm["proposal"].any()
                else 0.0
            ),
            "dice_pred_vs_oracle": dice_coefficient(arm["pred_m_final"], oracle),
            "oracle_failed": oracle_failed,
            "oracle_failure_reason": oracle_reason,
            "oracle_subset_of_proposal": bool(not (oracle & ~arm["proposal"]).any()),
            "pred_n_cc": int(pred_stats["n_cc"]),
            "oracle_n_cc": int(oracle_stats["n_cc"]),
            "skip_render": skip_render,
        }
        write_json_atomic(sample_root / "record.json", row)
        identity_rows.append(row)
        fallback = " FB" if row["used_geometry_fallback"] else ""
        empty = " EMPTY" if oracle_failed else ""
        label = (
            f"{identity}{fallback}{empty} pred={row['pred_m_final_pixels']} "
            f"oracle={oracle_pixels}"
        )
        arrays = {
            "normal": arm["normal"],
            "proposal": arm["proposal"],
            "pred_m_final": arm["pred_m_final"],
            "oracle_m_final": oracle,
            "pred_final_y": arm["pred_final_y"],
            "oracle_final_y": oracle_y,
        }
        panel_rows.append((label, arrays))
        by_type[defect_type].append((label, arrays))
        print(
            identity,
            "ORACLE_EMPTY" if oracle_failed else "OK",
            flush=True,
        )

    bins = _size_bins([int(row["pred_m_final_pixels"]) for row in identity_rows])
    for row, size_bin in zip(identity_rows, bins, strict=True):
        row["pred_m_final_size_bin"] = size_bin

    fewshot_rows: list[dict[str, Any]] = []
    fewshot_panels: list[tuple[str, dict[str, np.ndarray]]] = []
    for sample in split.train:
        anomaly = fewshot_uint8_rgb(sample)
        gt_mask = fewshot_binary_mask(sample)
        raw = _failure_rgb(anomaly.shape)
        recon = _failure_rgb(anomaly.shape)
        failed = not gt_mask.any() or bool(gt_mask.all())
        reason = "empty_or_full_gt" if failed else None
        if failed:
            pass
        elif renderer is not None:
            try:
                raw = renderer(
                    anomaly,
                    sample.defect_type,
                    gt_mask.astype(np.uint8),
                    _FEWSHOT_RECON_SEED,
                )
                recon = compose_causal(anomaly, raw, gt_mask)
            except (ValueError, RuntimeError) as error:
                failed = True
                reason = str(error)
                raw = _failure_rgb(anomaly.shape)
                recon = _failure_rgb(anomaly.shape)
        elif skip_render:
            raw = np.zeros_like(anomaly)
            recon = np.zeros_like(anomaly)
        sample_id = _item_id(sample.image_relpath)
        fewshot_root = output / "fewshot" / f"{sample.defect_type}-{sample_id}"
        fewshot_root.mkdir(parents=True)
        Image.fromarray(anomaly).save(fewshot_root / "anomaly.png")
        Image.fromarray(_mask_to_uint8(gt_mask)).save(fewshot_root / "gt_mask.png")
        Image.fromarray(raw).save(fewshot_root / "raw_r.png")
        Image.fromarray(recon).save(fewshot_root / "recon_y.png")
        row = {
            "sample_id": sample_id,
            "image_relpath": sample.image_relpath,
            "defect_type": sample.defect_type,
            "gt_pixels": int(gt_mask.sum()),
            "recon_mae_inside": masked_mae(anomaly, recon, gt_mask),
            "recon_mae_outside": masked_mae(anomaly, recon, ~gt_mask),
            "failed": failed,
            "failure_reason": reason,
            "skip_render": skip_render,
            "seed": _FEWSHOT_RECON_SEED,
        }
        write_json_atomic(fewshot_root / "record.json", row)
        fewshot_rows.append(row)
        fewshot_panels.append(
            (
                f"{sample.defect_type} {sample_id} mae={row['recon_mae_inside']:.2f}",
                {
                    "anomaly": anomaly,
                    "gt_mask": gt_mask,
                    "raw_r": raw,
                    "recon_y": recon,
                },
            )
        )

    _draw_panel(output / "compare_overview.png", panel_rows, columns=_COMPARE_COLUMNS, tile=tile)
    written = ["compare_overview.png"]
    for defect_type, rows in by_type.items():
        path = output / f"compare_{defect_type}.png"
        _draw_panel(path, rows, columns=_COMPARE_COLUMNS, tile=tile)
        written.append(path.name)
    small_rows = [
        panel
        for panel, row in zip(panel_rows, identity_rows, strict=True)
        if row["pred_m_final_size_bin"] == "small"
    ]
    _draw_panel(
        output / "compare_small_pred_mask.png",
        small_rows,
        columns=_COMPARE_COLUMNS,
        tile=tile,
    )
    written.append("compare_small_pred_mask.png")
    _draw_panel(
        output / "fewshot_recon.png",
        fewshot_panels,
        columns=_FEWSHOT_COLUMNS,
        tile=tile,
    )
    written.append("fewshot_recon.png")

    def pack_identity(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "oracle_empty": sum(item["oracle_failed"] for item in group),
            "fallback": sum(item["used_geometry_fallback"] for item in group),
            "pred_m_final_pixels": _percentiles(
                [float(item["pred_m_final_pixels"]) for item in group]
            ),
            "oracle_m_final_pixels": _percentiles(
                [float(item["oracle_m_final_pixels"]) for item in group]
            ),
            "dice_pred_vs_oracle": _percentiles(
                [float(item["dice_pred_vs_oracle"]) for item in group]
            ),
            "oracle_to_proposal": _percentiles(
                [float(item["oracle_to_proposal"]) for item in group]
            ),
        }

    by_defect = {
        name: pack_identity([row for row in identity_rows if row["defect_type"] == name])
        for name in config.defect_types
    }
    by_size = {
        name: pack_identity([row for row, size in zip(identity_rows, bins, strict=True) if size == name])
        for name in ("small", "medium", "large")
    }
    report = {
        "schema_version": "v06_b_r_oracle_mask_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_proposal_as_label": True,
        "no_new_weights": True,
        "category": config.category,
        "generate_root": str(generate_root),
        "b_experiment": _A_TO_H_B_EXPERIMENT,
        "oracle_definition": "transferred_gt ∩ generate_atlas_proposal",
        "empty_oracle_policy": "record_failure_no_projection_no_square_dilate",
        "identities": identity_rows,
        "fewshot": fewshot_rows,
        "summary": {
            "identity_count": len(identity_rows),
            "oracle_empty": sum(row["oracle_failed"] for row in identity_rows),
            "generate_failed": sum(row["generate_failed"] for row in identity_rows),
            "fallback": sum(row["used_geometry_fallback"] for row in identity_rows),
            "all": pack_identity(identity_rows),
            "by_defect_type": by_defect,
            "by_pred_m_final_size": by_size,
            "fewshot_recon_mae_inside": _percentiles(
                [float(row["recon_mae_inside"]) for row in fewshot_rows]
            ),
        },
        "files": written,
    }
    write_json_atomic(output / "report.json", report)
    write_json_atomic(
        output / "summary.json",
        {
            "schema_version": report["schema_version"],
            "diagnostic_only": True,
            "not_official_generate": True,
            "not_proposal_as_label": True,
            "no_new_weights": True,
            "visible_defects_passed": None,
            **report["summary"],
            "files": written,
        },
    )
    return report


def run_r_oracle_mask_from_cli(
    *,
    paths: Path,
    experiment: Path,
    source_model: Path,
    source_prompts: Path,
    manifest: Path,
    generate_config: Path,
    generate_root: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
    skip_render: bool = False,
) -> dict[str, Any]:
    config = load_v06_b_config(experiment)
    if not isinstance(config, V06BMaskFirstConfig):
        raise ValueError("experiment must be a mask-first configuration")
    if not is_cable_a_to_h(config):
        raise ValueError("R oracle-mask diagnostic is only authorized for Cable A→H")
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
    runtime = None
    if not skip_render:
        loaded = load_cable_generate_config(generate_config, repository_root=cwd)
        runtime = load_cable_runtime(loaded, cwd, device)
    return run_r_oracle_mask_diagnostic(
        context=context,
        generate_root=generate_root,
        output=output,
        runtime=runtime,
        skip_render=skip_render,
    )


__all__ = [
    "assert_fresh_diagnostic_output",
    "build_oracle_mask",
    "dice_coefficient",
    "fewshot_binary_mask",
    "fewshot_uint8_rgb",
    "lookup_fewshot_source",
    "lookup_training_target",
    "masked_mae",
    "run_r_oracle_mask_diagnostic",
    "run_r_oracle_mask_from_cli",
]

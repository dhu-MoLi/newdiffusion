"""Cable diagnostic: instantiate few-shot GT shape inside official a2 proposal.

No new weights. Does not overwrite pixel a2→H generate. R is frozen scheme 2.
Placement requires T(g) ⊆ m_proposal; empty feasible set fails without clipping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy.signal import correlate

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import _material_proposal_maps, load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, _estimate_support, estimate_normal_structure
from cadiff.v06_atlas_diag.cable_a import _lookup_p1_target, build_cable_a_slots
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source
from cadiff.v06_b.config import (
    CABLE_SCHEME2_OUTPUT,
    is_cable_a2_to_h,
    is_cable_a_to_h,
    is_cable_scheme2,
    load_v06_b_config,
)
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.cable_generate import load_cable_generate_config, load_cable_runtime

FLOW_COLUMNS = (
    "source",
    "source_gt",
    "inpaint",
    "src_obj",
    "shape",
    "normal",
    "object",
    "a2_sim",
    "a2_prop",
    "placed",
    "overlay",
    "raw_r",
    "final_y",
)
COLUMN_MEANINGS = {
    "source": "few-shot RGB，红叠加为官方 GT",
    "source_gt": "few-shot 官方 GT",
    "inpaint": "最近邻填洞后的源图（a2 用）",
    "src_obj": "源图计算物体 support，用于面积比缩放",
    "shape": "按物体面积比缩放后的 GT 模板，尚未平移",
    "normal": "目标正常图 x",
    "object": "目标计算物体 support",
    "a2_sim": "a2 材质余弦图（仅 V_x 内着色）",
    "a2_prop": "正式 a2 m_proposal",
    "placed": "可行放置后的 T(g)，即本次 m_final",
    "overlay": "蓝=proposal，黄=placed，底为 normal",
    "raw_r": "冻结方案 2 R 的 raw 输出",
    "final_y": "因果合成 y",
    "pixel_H": "像素 a2→H 的 m_final（对照，非本诊断）",
    "pixel_y": "像素 a2→H 的 final_y（对照，非本诊断）",
}
COMPARE_COLUMNS = (
    "normal",
    "a2_prop",
    "pixel_H",
    "placed",
    "pixel_y",
    "final_y",
)


@dataclass(frozen=True)
class ShapePlacement:
    mask: NDArray[np.bool_]
    scale: float
    top: int
    left: int
    score: float
    feasible_count: int
    source_gt_pixels: int
    scaled_gt_pixels: int


class EmptyFeasiblePlacement(ValueError):
    """No rigid placement keeps the scaled GT inside the a2 proposal."""


def scale_binary_mask(mask: np.ndarray, scale: float, *, size: int = 512) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (size, size):
        raise ValueError("mask must be a square canvas")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        raise ValueError("cannot scale an empty mask")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = binary[top:bottom, left:right]
    new_h = max(1, int(round(crop.shape[0] * scale)))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    if new_h >= size or new_w >= size:
        raise EmptyFeasiblePlacement("scaled GT does not fit the canvas")
    resized = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255, mode="L").resize(
            (new_w, new_h),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    ) > 127
    if not resized.any():
        raise EmptyFeasiblePlacement("scaled GT vanished")
    canvas = np.zeros((size, size), dtype=bool)
    center_row = 0.5 * (top + bottom - 1)
    center_col = 0.5 * (left + right - 1)
    paste_top = int(round(center_row - 0.5 * (new_h - 1)))
    paste_left = int(round(center_col - 0.5 * (new_w - 1)))
    paste_top = min(max(paste_top, 0), size - new_h)
    paste_left = min(max(paste_left, 0), size - new_w)
    canvas[paste_top : paste_top + new_h, paste_left : paste_left + new_w] = resized
    return canvas


def object_scale(source_support: np.ndarray, target_support: np.ndarray) -> float:
    source_area = int(np.asarray(source_support, dtype=bool).sum())
    target_area = int(np.asarray(target_support, dtype=bool).sum())
    if source_area == 0 or target_area == 0:
        return 1.0
    return float(np.sqrt(target_area / source_area))


def place_shape_inside_proposal(
    shape: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
    weight_by_similarity: bool = True,
) -> ShapePlacement:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(weight_by_similarity, bool):
        raise ValueError("weight_by_similarity must be boolean")
    template = np.asarray(shape, dtype=bool)
    allowed = np.asarray(proposal, dtype=bool)
    scores = np.asarray(similarity, dtype=np.float32)
    if template.shape != allowed.shape or allowed.shape != scores.shape:
        raise ValueError("shape, proposal, and similarity must share one grid")
    rows, cols = np.nonzero(template)
    if rows.size == 0:
        raise EmptyFeasiblePlacement("scaled GT is empty")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    crop = np.ascontiguousarray(template[top:bottom, left:right])
    height, width = crop.shape
    if height > allowed.shape[0] or width > allowed.shape[1]:
        raise EmptyFeasiblePlacement("scaled GT is larger than the canvas")
    outside = np.ascontiguousarray(~allowed, dtype=np.float64)
    kernel = np.ascontiguousarray(crop, dtype=np.float64)
    overlap = correlate(outside, kernel, mode="valid", method="fft")
    score_map = correlate(scores.astype(np.float64), kernel, mode="valid", method="fft") / float(
        crop.sum()
    )
    feasible = overlap < 0.5
    if not feasible.any():
        raise EmptyFeasiblePlacement("no translation keeps T(g) inside a2 proposal")
    ys, xs = np.nonzero(feasible)
    rng = np.random.RandomState(seed)
    if weight_by_similarity:
        weights = np.maximum(score_map[feasible], 0.0)
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()
        index = int(rng.choice(ys.size, p=probs))
    else:
        index = int(rng.choice(ys.size))
    paste_top = int(ys[index])
    paste_left = int(xs[index])
    placed = np.zeros_like(template, dtype=bool)
    placed[paste_top : paste_top + height, paste_left : paste_left + width] = crop
    if (placed & ~allowed).any():
        raise RuntimeError("placement escaped the a2 proposal")
    return ShapePlacement(
        mask=np.ascontiguousarray(placed, dtype=bool),
        scale=1.0,
        top=paste_top,
        left=paste_left,
        score=float(score_map[paste_top, paste_left]),
        feasible_count=int(feasible.sum()),
        source_gt_pixels=int(template.sum()),
        scaled_gt_pixels=int(placed.sum()),
    )


def instantiate_shape(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, ShapePlacement]:
    scale = object_scale(source_support, target_support)
    scaled = scale_binary_mask(source_gt, scale)
    placement = place_shape_inside_proposal(scaled, proposal, similarity, seed=seed)
    return scaled, ShapePlacement(
        mask=placement.mask,
        scale=scale,
        top=placement.top,
        left=placement.left,
        score=placement.score,
        feasible_count=placement.feasible_count,
        source_gt_pixels=int(np.asarray(source_gt, dtype=bool).sum()),
        scaled_gt_pixels=int(scaled.sum()),
    )


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255)


def _fail_rgb(shape: tuple[int, ...]) -> np.ndarray:
    image = np.zeros(shape if len(shape) == 3 else (*shape, 3), dtype=np.uint8)
    image[..., 0] = 160
    return image


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    painted = np.array(rgb, dtype=np.uint8, copy=True)
    binary = np.asarray(mask, dtype=bool)
    painted[binary] = (
        (1.0 - alpha) * painted[binary] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return painted


def _heatmap(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float32)
    inside = np.asarray(valid, dtype=bool)
    image = np.zeros((*field.shape, 3), dtype=np.uint8)
    if not inside.any():
        return image
    lo = float(field[inside].min())
    hi = float(field[inside].max())
    norm = np.zeros_like(field, dtype=np.float32)
    if hi > lo:
        norm[inside] = (field[inside] - lo) / (hi - lo)
    image[..., 0] = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip((1.0 - np.abs(norm - 0.5) * 2.0) * 180.0, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
    image[~inside] = 0
    return image


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    parts = output.resolve().parts
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def _require_frozen_scheme2_renderer(generate_config: Path, repository_root: Path) -> None:
    data = load_cable_generate_config(generate_config, repository_root=repository_root)
    experiment = load_v06_b_config(repository_root / data["b_experiment"])
    if is_cable_a_to_h(experiment) or is_cable_a2_to_h(experiment):
        raise ValueError("a2 shape instantiate must load frozen scheme 2 R, not A→H / a2→H")
    if not is_cable_scheme2(experiment) or experiment.output_relpath != CABLE_SCHEME2_OUTPUT:
        raise ValueError("a2 shape instantiate R must be seed-20260909-scheme2")


def _save_sheet(
    path: Path,
    rows: list[tuple[str, dict[str, np.ndarray]]],
    columns: tuple[str, ...],
    *,
    tile: int,
) -> None:
    header = 28
    gutter = 280
    canvas = Image.new(
        "RGB",
        (gutter + tile * len(columns), header + len(rows) * (tile + 4)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(columns):
        draw.text((gutter + index * tile + 4, 6), name, fill="black")
    for row_index, (label, images) in enumerate(rows):
        top = header + row_index * (tile + 4)
        draw.text((6, top + max(tile // 2 - 6, 4)), label, fill="black")
        for column, name in enumerate(columns):
            array = images[name]
            resample = Image.Resampling.NEAREST if array.ndim == 2 else Image.Resampling.LANCZOS
            tile_image = Image.fromarray(array).convert("RGB").resize((tile, tile), resample)
            canvas.paste(tile_image, (gutter + column * tile, top))
    canvas.save(path)


def _read_optional_png(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB" if image.mode != "L" else "L"))
    return np.ascontiguousarray(array)


def run_a2_shape_instantiate_diagnostic(
    *,
    repository_root: Path,
    dataset_root: Path,
    output: Path,
    a_experiment: Path,
    generate_config: Path,
    pixel_h_root: Path | None,
    device: str | None = None,
    tile: int = 160,
    renderer: Callable[[np.ndarray, str, np.ndarray, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    destination = _contained_output(output)
    experiment = load_v06_a_experiment(a_experiment)
    manifest = load_v06_a_identity_manifest(
        repository_root / experiment.identity_manifest,
        experiment=experiment,
    )
    slots = build_cable_a_slots(manifest, experiment=experiment)
    generator = load_a2_generator(
        experiment_path=a_experiment,
        dataset_root=dataset_root,
        device=device,
    )
    if renderer is None:
        _require_frozen_scheme2_renderer(generate_config, repository_root)
        runtime = load_cable_runtime(
            load_cable_generate_config(generate_config, repository_root=repository_root),
            repository_root,
            device,
        )
        renderer = runtime.module_b.renderer
    destination.mkdir(parents=True, exist_ok=False)
    identities = destination / "identities"
    identities.mkdir()
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    compare_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    per_type: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {
        name: [] for name in experiment.defect_types
    }
    records: list[dict[str, Any]] = []
    structures: dict[str, NormalStructure] = {}

    for slot in slots:
        identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        sample_root = identities / identity
        sample_root.mkdir()
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
        inpainted = neighbor_fill_source(source.rgb, source.anomaly_gt, source.valid)
        inpainted_identity = replace(
            source,
            rgb=np.array(inpainted, copy=True),
            anomaly_gt=None,
            record=replace(source.record, role="normal_fit", defect_type="good"),
        )
        source_support = _estimate_support(inpainted_identity, experiment)
        source_feat = generator._encoder.encode(inpainted)
        target_feat = generator._encoder.encode(target.normal.rgb)
        proposal, tau, similarity = _material_proposal_maps(
            source.anomaly_gt,
            source.valid,
            target,
            source_feat=source_feat,
            target_feat=target_feat,
        )
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        fail = _fail_rgb(normal.shape)
        arrays = {
            "source": _overlay(source.rgb, source.anomaly_gt, (255, 40, 40), 0.55),
            "source_gt": _mask_u8(source.anomaly_gt),
            "inpaint": np.ascontiguousarray(inpainted),
            "src_obj": _mask_u8(source_support),
            "shape": _mask_u8(source.anomaly_gt),
            "normal": normal,
            "object": _mask_u8(target.support & target.valid),
            "a2_sim": _heatmap(similarity, target.valid),
            "a2_prop": _mask_u8(proposal),
            "placed": _mask_u8(np.zeros(proposal.shape, dtype=bool)),
            "overlay": _overlay(normal, proposal, (80, 160, 255), 0.45),
            "raw_r": fail.copy(),
            "final_y": fail.copy(),
            "pixel_H": fail.copy(),
            "pixel_y": fail.copy(),
        }
        if pixel_h_root is not None:
            pixel_h = _read_optional_png(pixel_h_root / identity / "m_final.png")
            pixel_y = _read_optional_png(pixel_h_root / identity / "final_y.png")
            if pixel_h is not None:
                arrays["pixel_H"] = pixel_h
            if pixel_y is not None:
                arrays["pixel_y"] = pixel_y
        record: dict[str, Any] = {
            "identity": identity,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": source.record.item_id,
            "seed": slot.seed,
            "a2_tau": tau,
            "proposal_pixels": int(proposal.sum()),
            "source_gt_pixels": int(source.anomaly_gt.sum()),
            "failed": False,
            "failure_reason": None,
            "feasible_count": 0,
            "scale": None,
            "placed_pixels": 0,
            "placement_score": None,
            "m_final_subset_of_proposal": None,
            "outside_rgb_unchanged": None,
        }
        try:
            scaled, placement = instantiate_shape(
                source.anomaly_gt,
                source_support,
                target.support & target.valid,
                proposal,
                similarity,
                seed=slot.seed,
            )
            validate_realization(placement.mask, proposal)
            arrays["shape"] = _mask_u8(scaled)
            arrays["placed"] = _mask_u8(placement.mask)
            arrays["overlay"] = _overlay(
                _overlay(normal, proposal, (80, 160, 255), 0.35),
                placement.mask,
                (255, 220, 0),
                0.7,
            )
            raw = renderer(normal, slot.defect_type, placement.mask.astype(np.uint8), slot.seed)
            final_y = compose_causal(normal, raw, placement.mask)
            subset = bool(not (placement.mask & ~proposal).any())
            unchanged = bool(np.array_equal(final_y[~placement.mask], normal[~placement.mask]))
            if not subset or not unchanged:
                raise ValueError("placed shape violated the causal mask contract")
            arrays["raw_r"] = np.ascontiguousarray(raw)
            arrays["final_y"] = np.ascontiguousarray(final_y)
            record.update(
                feasible_count=placement.feasible_count,
                scale=placement.scale,
                placed_pixels=int(placement.mask.sum()),
                placement_score=placement.score,
                m_final_subset_of_proposal=subset,
                outside_rgb_unchanged=unchanged,
                paste_top=placement.top,
                paste_left=placement.left,
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        for name, array in arrays.items():
            Image.fromarray(array).save(sample_root / f"{name}.png")
        write_json_atomic(sample_root / "record.json", record)
        records.append(record)
        label = identity + (" FAIL" if record["failed"] else "")
        flow = {name: arrays[name] for name in FLOW_COLUMNS}
        compare = {name: arrays[name] for name in COMPARE_COLUMNS}
        flow_rows.append((label, flow))
        compare_rows.append((label, compare))
        per_type[slot.defect_type].append((label, flow))
        print(identity, "FAIL " + str(record["failure_reason"]) if record["failed"] else "OK", flush=True)

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    _save_sheet(destination / "compare_pixel_h.png", compare_rows, COMPARE_COLUMNS, tile=tile)
    for kind, rows in per_type.items():
        _save_sheet(destination / f"flow_{kind}.png", rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": "v06_b_a2_shape_instantiate_v1",
        "diagnostic_only": True,
        "not_learned": True,
        "not_official_pass": True,
        "identity_count": len(records),
        "failed_rows": sum(row["failed"] for row in records),
        "method": "fewshot_gt_shape_inside_a2",
        "renderer": "frozen_scheme2",
        "no_projection": True,
        "no_square_dilate": True,
        "flow_columns": list(FLOW_COLUMNS),
        "compare_columns": list(COMPARE_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "pixel_h_root": None if pixel_h_root is None else str(pixel_h_root),
        "destination": str(destination),
        "visible_defects_passed": None,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_a2_shape_instantiate_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    generate_config: Path,
    output: Path,
    cwd: Path,
    pixel_h_root: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_a2_shape_instantiate_diagnostic(
        repository_root=cwd,
        dataset_root=project.dataset_root,
        output=output,
        a_experiment=a_experiment if a_experiment.is_absolute() else cwd / a_experiment,
        generate_config=(
            generate_config if generate_config.is_absolute() else cwd / generate_config
        ),
        pixel_h_root=pixel_h_root,
        device=device,
    )


__all__ = [
    "COLUMN_MEANINGS",
    "COMPARE_COLUMNS",
    "EmptyFeasiblePlacement",
    "FLOW_COLUMNS",
    "instantiate_shape",
    "object_scale",
    "place_shape_inside_proposal",
    "run_a2_shape_instantiate_diagnostic",
    "run_a2_shape_instantiate_from_cli",
    "scale_binary_mask",
]

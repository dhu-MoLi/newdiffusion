"""Diagnostic: DualAnoDiff loc checkpoint vs held-out GT overlays.

CPU-selectable identities are frozen before any forward pass: first two
DualAnoDiff held-out images per evaluated type, then the first good.
Green = real GT, red = localization heatmap / thresholded pred, yellow =
overlap. Not Module C. Does not overwrite eval dumps or train R.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from cadiff.provenance import write_json_atomic

SCHEMA = "v06_c_loc_pred_gt_overlay_v1"
IMG_DIM = 256
DISPLAY_SCALE = 2
PER_TYPE = 2
N_GOOD = 1
PRED_THR = 0.5
GT_COLOR = (40, 220, 80)
PRED_COLOR = (230, 50, 90)
OVERLAP_COLOR = (240, 220, 40)
EVAL_NAME = "20260917-official-200"
_FONT = ImageFont.load_default()
PredictFn = Callable[[NDArray[np.uint8], str], NDArray[np.float32]]


@dataclass(frozen=True)
class PreviewIdentity:
    arm: str
    class_name: str
    anomaly_name: str
    stem: str
    image_path: Path
    mask_path: Path | None


def sorted_image_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    files = [
        name
        for name in names
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    files.sort(
        key=lambda name: int(Path(name).stem) if Path(name).stem.isdigit() else name
    )
    return tuple(files)


def dualanodiff_heldout_names(names: tuple[str, ...] | list[str], anomaly_name: str) -> tuple[str, ...]:
    ordered = sorted_image_names(names)
    skip = 0 if anomaly_name == "good" else len(ordered) // 3
    return ordered[skip:]


def _mask_path(gt_root: Path, anomaly_name: str, stem: str) -> Path:
    preferred = gt_root / anomaly_name / f"{stem}_mask.png"
    fallback = gt_root / anomaly_name / f"{stem}.png"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(preferred)


def select_preview_identities(
    class_root: Path,
    *,
    anomaly_names: tuple[str, ...],
    arm: str | None = None,
    per_type: int = PER_TYPE,
    n_good: int = N_GOOD,
) -> tuple[PreviewIdentity, ...]:
    if per_type < 1 or n_good < 1:
        raise ValueError("preview counts must be positive")
    test_root = class_root / "test"
    gt_root = class_root / "ground_truth"
    class_name = class_root.name
    arm_name = arm or class_name
    rows: list[PreviewIdentity] = []
    for anomaly_name in anomaly_names:
        image_dir = test_root / anomaly_name
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        heldout = dualanodiff_heldout_names(tuple(p.name for p in image_dir.iterdir()), anomaly_name)
        picked = heldout[:per_type]
        if len(picked) < per_type:
            raise ValueError(f"{class_name}/{anomaly_name} has fewer than {per_type} held-out images")
        for name in picked:
            stem = Path(name).stem
            rows.append(
                PreviewIdentity(
                    arm=arm_name,
                    class_name=class_name,
                    anomaly_name=anomaly_name,
                    stem=stem,
                    image_path=image_dir / name,
                    mask_path=_mask_path(gt_root, anomaly_name, stem),
                )
            )
    good_dir = test_root / "good"
    if not good_dir.is_dir():
        raise FileNotFoundError(good_dir)
    goods = dualanodiff_heldout_names(tuple(p.name for p in good_dir.iterdir()), "good")[:n_good]
    if len(goods) < n_good:
        raise ValueError(f"{class_name}/good has fewer than {n_good} images")
    for name in goods:
        rows.append(
            PreviewIdentity(
                arm=arm_name,
                class_name=class_name,
                anomaly_name="good",
                stem=Path(name).stem,
                image_path=good_dir / name,
                mask_path=None,
            )
        )
    return tuple(rows)


def _blend(rgb: np.ndarray, color: tuple[int, int, int], weight: np.ndarray) -> NDArray[np.uint8]:
    base = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(weight, dtype=np.float32)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    tint = np.asarray(color, dtype=np.float32)
    out = base * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_pred_gt(
    rgb: np.ndarray,
    gt: np.ndarray,
    heat: np.ndarray,
    *,
    pred_thr: float = PRED_THR,
) -> dict[str, NDArray[np.uint8]]:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb must be HxWx3")
    mask = np.asarray(gt, dtype=bool)
    score = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    if mask.shape != image.shape[:2] or score.shape != image.shape[:2]:
        raise ValueError("gt and heat must match rgb spatial size")
    pred_bin = score >= pred_thr
    combined = _blend(image, PRED_COLOR, pred_bin.astype(np.float32) * 0.65)
    combined = _blend(combined, GT_COLOR, mask.astype(np.float32) * 0.65)
    combined[mask & pred_bin] = np.asarray(OVERLAP_COLOR, dtype=np.uint8)
    return {
        "rgb": image.copy(),
        "gt": _blend(image, GT_COLOR, mask.astype(np.float32) * 0.70),
        "heat": _blend(image, PRED_COLOR, score * 0.85),
        "combined": combined,
    }


def _read_rgb(path: Path, size: int) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.uint8)


def _read_mask(path: Path | None, size: int) -> NDArray[np.bool_]:
    if path is None:
        return np.zeros((size, size), dtype=bool)
    with Image.open(path) as image:
        array = np.asarray(image.convert("L").resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8)
    return array > 127


def _upscale(rgb: np.ndarray, scale: int) -> NDArray[np.uint8]:
    if scale == 1:
        return np.asarray(rgb, dtype=np.uint8)
    return np.repeat(np.repeat(np.asarray(rgb, dtype=np.uint8), scale, axis=0), scale, axis=1)


def _row_image(panels: Mapping[str, np.ndarray], title: str, *, scale: int = DISPLAY_SCALE) -> NDArray[np.uint8]:
    tiles = [_upscale(panels[name], scale) for name in ("rgb", "gt", "heat", "combined")]
    gap = 8
    header = 18
    height = header + tiles[0].shape[0]
    width = sum(tile.shape[1] for tile in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for tile in tiles:
        canvas[header : header + tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1] + gap
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((4, 2), f"{title}  rgb | GT-green | pred-red | overlay(yellow=overlap)", fill=(20, 20, 20), font=_FONT)
    return np.asarray(image, dtype=np.uint8)


def _stack_rows(rows: list[np.ndarray], heading: str) -> NDArray[np.uint8]:
    width = max(row.shape[1] for row in rows)
    header = 22
    height = header + sum(row.shape[0] for row in rows) + 6 * (len(rows) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), heading, fill=(20, 20, 20), font=_FONT)
    canvas = np.array(image, dtype=np.uint8, copy=True)
    y = header
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + 6
    return canvas


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def official200_overlay_arms(eval_root: Path, mvtec_path: Path) -> dict[str, dict[str, Any]]:
    root = eval_root
    visa = root / "visa"
    return {
        "cable": {
            "mvtec_path": mvtec_path,
            "class_name": "cable",
            "anomaly_names": ("bent_wire", "cut_inner_insulation", "cut_outer_insulation", "poke_insulation"),
            "ckpt": root / "cable/eval" / EVAL_NAME / "cable/ckpt_localization/cable.pckl",
        },
        "can": {
            "mvtec_path": root / "can/mvtec_layout",
            "class_name": "can",
            "anomaly_names": ("bad",),
            "ckpt": root / "can/eval" / EVAL_NAME / "can/ckpt_localization/can.pckl",
        },
        "candle": {
            "mvtec_path": visa / "candle/mvtec_layout",
            "class_name": "candle",
            "anomaly_names": (
                "chunk_of_wax_missing",
                "damaged_corner_of_packaging",
                "different_colour_spot",
                "extra_wax_in_candle",
            ),
            "ckpt": visa / "candle/eval" / EVAL_NAME / "candle/ckpt_localization/candle.pckl",
        },
        "capsules": {
            "mvtec_path": visa / "capsules/mvtec_layout",
            "class_name": "capsules",
            "anomaly_names": ("bubble",),
            "ckpt": visa / "capsules/eval" / EVAL_NAME / "capsules/ckpt_localization/capsules.pckl",
        },
        "cashew": {
            "mvtec_path": visa / "cashew/mvtec_layout",
            "class_name": "cashew",
            "anomaly_names": ("burnt", "corner_or_edge_breakage", "different_colour_spot", "middle_breakage"),
            "ckpt": visa / "cashew/eval" / EVAL_NAME / "cashew/ckpt_localization/cashew.pckl",
        },
        "chewinggum": {
            "mvtec_path": visa / "chewinggum/mvtec_layout",
            "class_name": "chewinggum",
            "anomaly_names": ("chunk_of_gum_missing", "corner_missing", "scratches", "similar_colour_spot"),
            "ckpt": visa / "chewinggum/eval" / EVAL_NAME / "chewinggum/ckpt_localization/chewinggum.pckl",
        },
    }


def _load_predict_fn(ckpt: Path, eval_kit: Path, device: str) -> PredictFn:
    kit = str(eval_kit.resolve())
    if kit not in sys.path:
        sys.path.insert(0, kit)
    import torch
    from evaluation.unet_utils.model_unet import DiscriminativeSubNetwork

    model = DiscriminativeSubNetwork(in_channels=3, out_channels=2)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    def predict(image: NDArray[np.uint8], identity: str) -> NDArray[np.float32]:
        del identity
        tensor = torch.from_numpy(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))[None]
        tensor = tensor.to(device)
        with torch.no_grad():
            heat = torch.softmax(model(tensor), dim=1)[0, 1].detach().cpu().numpy()
        return np.asarray(heat, dtype=np.float32)

    return predict


def _row_stats(gt: np.ndarray, heat: np.ndarray) -> dict[str, float]:
    mask = np.asarray(gt, dtype=bool)
    score = np.asarray(heat, dtype=np.float32)
    pred = score >= PRED_THR
    gt_area = int(mask.sum())
    pred_area = int(pred.sum())
    inside = float(score[mask].mean()) if gt_area else 0.0
    outside = float(score[~mask].mean()) if mask.size > gt_area else 0.0
    inter = int((mask & pred).sum())
    return {
        "gt_area": gt_area,
        "pred_area_thr": pred_area,
        "overlap_thr": inter,
        "mean_heat_in_gt": inside,
        "mean_heat_out_gt": outside,
    }


def run_loc_pred_gt_overlay(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    output: Path,
    predict_fn: PredictFn | None = None,
    eval_kit: Path | None = None,
    device: str = "cuda",
    img_dim: int = IMG_DIM,
) -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    selected: dict[str, tuple[PreviewIdentity, ...]] = {}
    identity_rows: list[dict[str, Any]] = []
    for name, spec in arms.items():
        class_root = Path(spec["mvtec_path"]) / str(spec["class_name"])
        rows = select_preview_identities(
            class_root,
            anomaly_names=tuple(spec["anomaly_names"]),
            arm=name,
        )
        selected[name] = rows
        for row in rows:
            identity_rows.append(
                {
                    "arm": row.arm,
                    "class_name": row.class_name,
                    "anomaly_name": row.anomaly_name,
                    "stem": row.stem,
                    "image_path": str(row.image_path),
                    "mask_path": None if row.mask_path is None else str(row.mask_path),
                }
            )
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": "first DualAnoDiff held-out PER_TYPE per anomaly, then first N_GOOD good; frozen before predict",
            "per_type": PER_TYPE,
            "n_good": N_GOOD,
            "img_dim": img_dim,
            "pred_thr_visual": PRED_THR,
            "gt_color": "green",
            "pred_color": "red",
            "overlap_color": "yellow",
            "identities": identity_rows,
        },
    )
    predictors: dict[str, PredictFn] = {}
    reports: dict[str, Any] = {}
    for name, spec in arms.items():
        if predict_fn is not None:
            predictors[name] = predict_fn
        else:
            if eval_kit is None:
                raise ValueError("eval_kit is required when predict_fn is omitted")
            predictors[name] = _load_predict_fn(Path(spec["ckpt"]), eval_kit, device)
        sample_root = destination / "samples" / name
        sample_root.mkdir(parents=True)
        sheet_rows: list[np.ndarray] = []
        sample_stats: list[dict[str, Any]] = []
        for row in selected[name]:
            rgb = _read_rgb(row.image_path, img_dim)
            gt = _read_mask(row.mask_path, img_dim)
            identity = f"{row.arm}-{row.anomaly_name}-{row.stem}"
            heat = np.asarray(predictors[name](rgb, identity), dtype=np.float32)
            if heat.shape != rgb.shape[:2]:
                raise ValueError(f"predict_fn returned {heat.shape}, expected {rgb.shape[:2]}")
            panels = overlay_pred_gt(rgb, gt, heat)
            title = f"{row.arm} {row.anomaly_name} {row.stem}"
            strip = _row_image(panels, title)
            Image.fromarray(strip).save(sample_root / f"{row.anomaly_name}-{row.stem}.png")
            sheet_rows.append(strip)
            stats = _row_stats(gt, heat)
            stats.update(anomaly_name=row.anomaly_name, stem=row.stem)
            sample_stats.append(stats)
        Image.fromarray(
            _stack_rows(sheet_rows, f"{name}  green=held-out GT  red=loc pred  yellow=overlap")
        ).save(destination / f"{name}_sheet.png")
        reports[name] = {
            "n": len(selected[name]),
            "ckpt": str(spec.get("ckpt")),
            "samples": sample_stats,
        }
        print(name, "DONE", len(selected[name]), flush=True)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "arm_names": list(reports),
        "arms": reports,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_loc_pred_gt_overlay_from_cli(
    *,
    eval_root: Path,
    mvtec_path: Path,
    eval_kit: Path,
    output: Path,
    device: str | None = None,
) -> dict[str, Any]:
    return run_loc_pred_gt_overlay(
        arms=official200_overlay_arms(eval_root, mvtec_path),
        output=output,
        eval_kit=eval_kit,
        device=device or "cuda",
    )


__all__ = [
    "N_GOOD",
    "PER_TYPE",
    "PreviewIdentity",
    "dualanodiff_heldout_names",
    "official200_overlay_arms",
    "overlay_pred_gt",
    "run_loc_pred_gt_overlay",
    "run_loc_pred_gt_overlay_from_cli",
    "select_preview_identities",
    "sorted_image_names",
]

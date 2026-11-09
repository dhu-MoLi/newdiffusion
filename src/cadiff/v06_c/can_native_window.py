"""Native-resolution window geometry for MVTec AD 2 can.

A stays on letterbox 512. H places native few-shot GT. R trains and infers
inside a deterministic 512 crop around m_final, then pastes back.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cadiff.data.geometry import (
    GeometryMeta,
    letterbox_geometry,
    resize_pad_image,
    resize_pad_mask,
    unletterbox_mask,
)
from cadiff.provenance import write_json_atomic
from cadiff.v06_c.can_mvtec2 import CATEGORY, DEFECT_TYPE, CanInventory, mask_path_for

WINDOW_SIZE = 512
WINDOW_MARGIN = 64
EVAL_SHORT_SIDE = 512
EVAL_SIZE_MULTIPLE = 32
A2_SIZE = 512


@dataclass(frozen=True)
class NativeWindow:
    top: int
    left: int
    height: int
    width: int
    scaled_height: int
    scaled_width: int
    scale: float
    pad_top: int
    pad_left: int
    size: int
    source_height: int
    source_width: int

    def as_record(self) -> dict[str, int | float]:
        return asdict(self)


def short_side_square_size(
    source_width: int,
    source_height: int,
    *,
    short_side: int = EVAL_SHORT_SIDE,
    multiple: int = EVAL_SIZE_MULTIPLE,
) -> int:
    if type(source_width) is not int or type(source_height) is not int:
        raise ValueError("source size must be integers")
    if source_width < 1 or source_height < 1:
        raise ValueError("source size must be positive")
    if type(short_side) is not int or short_side < 1:
        raise ValueError("short_side must be a positive integer")
    if type(multiple) is not int or multiple < 1:
        raise ValueError("multiple must be a positive integer")
    short = min(source_width, source_height)
    long = max(source_width, source_height)
    raw = int(round(long / short * short_side))
    size = int(math.ceil(raw / multiple) * multiple)
    if size < multiple:
        raise ValueError("letterbox square is smaller than the U-Net stride")
    return size


def load_native_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(np.array(image.convert("RGB"), copy=True))


def load_native_mask(path: Path, *, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if array.shape != shape:
        raise ValueError(f"native mask shape {array.shape} != image {shape}")
    return np.ascontiguousarray(array > 0)


def geometry_for_rgb(rgb: np.ndarray, *, size: int = A2_SIZE) -> GeometryMeta:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return letterbox_geometry(width, height, size)


def lift_binary_mask(mask: np.ndarray, meta: GeometryMeta) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (meta.target_size, meta.target_size):
        raise ValueError("letterboxed mask must match target size")
    image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    native = np.asarray(unletterbox_mask(image, meta), dtype=np.uint8)
    return np.ascontiguousarray(native >= 128)


def lift_float_map(values: np.ndarray, meta: GeometryMeta) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (meta.target_size, meta.target_size):
        raise ValueError("letterboxed map must match target size")
    crop = array[
        meta.pad_top : meta.pad_top + meta.resized_height,
        meta.pad_left : meta.pad_left + meta.resized_width,
    ]
    resized = Image.fromarray(np.ascontiguousarray(crop), mode="F").resize(
        (meta.source_width, meta.source_height),
        resample=Image.Resampling.BILINEAR,
    )
    return np.ascontiguousarray(np.array(resized, dtype=np.float32, copy=True))


def pad_to_square(array: np.ndarray, fill: Any = 0) -> tuple[np.ndarray, int, int]:
    if array.ndim not in (2, 3):
        raise ValueError("array must be HxW or HxWxC")
    height, width = int(array.shape[0]), int(array.shape[1])
    size = max(height, width)
    if array.ndim == 2:
        canvas = np.full((size, size), fill, dtype=array.dtype)
    else:
        canvas = np.full((size, size) + array.shape[2:], fill, dtype=array.dtype)
    canvas[:height, :width] = array
    return np.ascontiguousarray(canvas), height, width


def crop_to_hw(array: np.ndarray, height: int, width: int) -> np.ndarray:
    if array.shape[0] < height or array.shape[1] < width:
        raise ValueError("padded array is smaller than the native frame")
    return np.ascontiguousarray(array[:height, :width])


def window_from_mask(
    mask: np.ndarray,
    *,
    margin: int = WINDOW_MARGIN,
    size: int = WINDOW_SIZE,
) -> NativeWindow:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be HxW")
    if type(margin) is not int or margin < 0:
        raise ValueError("margin must be a non-negative integer")
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    height, width = int(binary.shape[0]), int(binary.shape[1])
    rows, cols = np.nonzero(binary)
    if rows.size == 0:
        raise ValueError("cannot window an empty mask")
    y0 = max(0, int(rows.min()) - margin)
    x0 = max(0, int(cols.min()) - margin)
    y1 = min(height, int(rows.max()) + 1 + margin)
    x1 = min(width, int(cols.max()) + 1 + margin)
    box_h = y1 - y0
    box_w = x1 - x0
    side = max(box_h, box_w)
    center_row = 0.5 * (int(rows.min()) + int(rows.max()))
    center_col = 0.5 * (int(cols.min()) + int(cols.max()))
    if side <= size and height >= size and width >= size:
        top = int(round(center_row - 0.5 * (size - 1)))
        left = int(round(center_col - 0.5 * (size - 1)))
        top = min(max(top, 0), height - size)
        left = min(max(left, 0), width - size)
        return NativeWindow(
            top=top,
            left=left,
            height=size,
            width=size,
            scaled_height=size,
            scaled_width=size,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            size=size,
            source_height=height,
            source_width=width,
        )
    scale = min(1.0, size / float(side))
    scaled_h = max(1, int(round(box_h * scale)))
    scaled_w = max(1, int(round(box_w * scale)))
    if scaled_h > size or scaled_w > size:
        scale = min(size / float(box_h), size / float(box_w))
        scaled_h = max(1, int(round(box_h * scale)))
        scaled_w = max(1, int(round(box_w * scale)))
    pad_y = size - scaled_h
    pad_x = size - scaled_w
    return NativeWindow(
        top=y0,
        left=x0,
        height=box_h,
        width=box_w,
        scaled_height=scaled_h,
        scaled_width=scaled_w,
        scale=float(scale),
        pad_top=pad_y // 2,
        pad_left=pad_x // 2,
        size=size,
        source_height=height,
        source_width=width,
    )


def extract_window(
    rgb: np.ndarray,
    mask: np.ndarray,
    window: NativeWindow,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] != mask.shape or rgb.shape[:2] != (window.source_height, window.source_width):
        raise ValueError("rgb/mask do not match the window source frame")
    crop_rgb = rgb[window.top : window.top + window.height, window.left : window.left + window.width]
    crop_mask = np.asarray(mask, dtype=bool)[
        window.top : window.top + window.height, window.left : window.left + window.width
    ]
    if window.scale != 1.0:
        crop_rgb = np.asarray(
            Image.fromarray(crop_rgb).resize(
                (window.scaled_width, window.scaled_height),
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
        crop_mask = (
            np.asarray(
                Image.fromarray(crop_mask.astype(np.uint8) * 255, mode="L").resize(
                    (window.scaled_width, window.scaled_height),
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            >= 128
        )
    canvas_rgb = np.zeros((window.size, window.size, 3), dtype=np.uint8)
    canvas_mask = np.zeros((window.size, window.size), dtype=bool)
    canvas_rgb[
        window.pad_top : window.pad_top + window.scaled_height,
        window.pad_left : window.pad_left + window.scaled_width,
    ] = crop_rgb
    canvas_mask[
        window.pad_top : window.pad_top + window.scaled_height,
        window.pad_left : window.pad_left + window.scaled_width,
    ] = crop_mask
    if not canvas_mask.any():
        raise ValueError("window emptied the mask")
    if canvas_mask.all():
        raise ValueError("window mask covers the full 512 crop")
    return np.ascontiguousarray(canvas_rgb), np.ascontiguousarray(canvas_mask)


def paste_window(native_rgb: np.ndarray, window_rgb: np.ndarray, window: NativeWindow) -> np.ndarray:
    if native_rgb.shape[:2] != (window.source_height, window.source_width):
        raise ValueError("native rgb does not match the window source frame")
    if window_rgb.shape != (window.size, window.size, 3) or window_rgb.dtype != np.uint8:
        raise ValueError("window rgb must be 512x512 uint8")
    patch = window_rgb[
        window.pad_top : window.pad_top + window.scaled_height,
        window.pad_left : window.pad_left + window.scaled_width,
    ]
    if window.scale != 1.0:
        patch = np.asarray(
            Image.fromarray(patch).resize(
                (window.width, window.height),
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
    out = np.array(native_rgb, copy=True)
    out[window.top : window.top + window.height, window.left : window.left + window.width] = patch
    return np.ascontiguousarray(out)


def letterbox_rgb(rgb: np.ndarray, size: int) -> tuple[np.ndarray, GeometryMeta]:
    image, _valid, meta = resize_pad_image(Image.fromarray(rgb), size)
    return np.ascontiguousarray(np.array(image, copy=True)), meta


def letterbox_binary(mask: np.ndarray, meta: GeometryMeta) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L")
    transformed = resize_pad_mask(image, meta)
    return np.ascontiguousarray(np.asarray(transformed, dtype=np.uint8) >= 128)


def mask_area_stats(areas: tuple[int, ...]) -> dict[str, float | int]:
    if not areas:
        raise ValueError("area stats require at least one mask")
    values = np.asarray(areas, dtype=np.int64)
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": int(values.max()),
    }


def heldout_letterbox_area_stats(inventory: CanInventory, *, size: int) -> dict[str, Any]:
    gt_root = inventory.layout_root / CATEGORY / "ground_truth" / DEFECT_TYPE
    native_areas: list[int] = []
    letterbox_areas: list[int] = []
    for path in inventory.heldout:
        rgb = load_native_rgb(path)
        mask = load_native_mask(mask_path_for(path, ground_truth_root=gt_root), shape=rgb.shape[:2])
        native_areas.append(int(mask.sum()))
        _canvas, meta = letterbox_rgb(rgb, size)
        letterbox_areas.append(int(letterbox_binary(mask, meta).sum()))
    native = tuple(native_areas)
    letterboxed = tuple(letterbox_areas)
    return {
        "eval_letterbox_size": size,
        "eval_short_side_target": EVAL_SHORT_SIDE,
        "native": mask_area_stats(native),
        "letterbox": mask_area_stats(letterboxed),
        "median_at_least_50": bool(np.median(np.asarray(letterboxed)) >= 50),
    }


def stage_letterbox_eval_layout(
    inventory: CanInventory,
    destination: Path,
    *,
    size: int,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite eval layout: {destination}")
    stats = heldout_letterbox_area_stats(inventory, size=size)
    can_root = destination / CATEGORY
    train_good = can_root / "train" / "good"
    test_bad = can_root / "test" / DEFECT_TYPE
    test_good = can_root / "test" / "good"
    gt_bad = can_root / "ground_truth" / DEFECT_TYPE
    for folder in (train_good, test_bad, test_good, gt_bad):
        folder.mkdir(parents=True, exist_ok=False)
    gt_root = inventory.layout_root / CATEGORY / "ground_truth" / DEFECT_TYPE
    for path in inventory.train_good:
        rgb = load_native_rgb(path)
        canvas, _meta = letterbox_rgb(rgb, size)
        Image.fromarray(canvas).save(train_good / path.name)
    for path in inventory.test_good:
        rgb = load_native_rgb(path)
        canvas, _meta = letterbox_rgb(rgb, size)
        Image.fromarray(canvas).save(test_good / path.name)
    for path in (*inventory.fewshot, *inventory.heldout):
        rgb = load_native_rgb(path)
        mask = load_native_mask(mask_path_for(path, ground_truth_root=gt_root), shape=rgb.shape[:2])
        canvas, meta = letterbox_rgb(rgb, size)
        Image.fromarray(canvas).save(test_bad / path.name)
        Image.fromarray(letterbox_binary(mask, meta).astype(np.uint8) * 255).save(
            gt_bad / f"{path.stem}_mask.png"
        )
    write_json_atomic(destination / "geometry_lock.json", stats)
    return stats


def pack_generated_letterbox(
    *,
    dump_root: Path,
    destination: Path,
    size: int,
) -> int:
    image_dir = destination / CATEGORY / DEFECT_TYPE / "composed"
    mask_dir = destination / CATEGORY / DEFECT_TYPE / "mask_defect"
    if image_dir.exists() or mask_dir.exists():
        raise FileExistsError(f"refuses to overwrite generated eval pack: {destination}")
    image_dir.mkdir(parents=True, exist_ok=False)
    mask_dir.mkdir(parents=True, exist_ok=False)
    manifest = dump_root / "synthetic_manifest.jsonl"
    rows = [
        line
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, line in enumerate(rows):
        row = json.loads(line)
        rgb = load_native_rgb(dump_root / row["image_relpath"])
        mask = load_native_mask(dump_root / row["mask_relpath"], shape=rgb.shape[:2])
        canvas, meta = letterbox_rgb(rgb, size)
        Image.fromarray(canvas).save(image_dir / f"{index}.png")
        Image.fromarray(letterbox_binary(mask, meta).astype(np.uint8) * 255).save(
            mask_dir / f"{index}.png"
        )
    return len(rows)


def write_mask_center_crops(
    dump_root: Path,
    output: Path,
    *,
    seed: int = 20260912,
    count: int = 8,
    crop: int = 64,
    zoom: int = 8,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite crop diagnostic: {output}")
    manifest = dump_root / "synthetic_manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < count:
        raise ValueError("not enough kept pairs for crop diagnostic")
    rng = np.random.RandomState(seed)
    chosen = [int(index) for index in rng.choice(len(rows), size=count, replace=False)]
    output.mkdir(parents=True, exist_ok=False)
    tiles: list[Image.Image] = []
    selection: list[dict[str, Any]] = []
    for order, index in enumerate(chosen):
        row = rows[index]
        y = load_native_rgb(dump_root / row["image_relpath"])
        mask = load_native_mask(dump_root / row["mask_relpath"], shape=y.shape[:2])
        normal_path = dump_root / "samples" / row["job_id"] / "normal.png"
        normal = load_native_rgb(normal_path) if normal_path.is_file() else y
        ys, xs = np.nonzero(mask)
        center_y = int(round(0.5 * (int(ys.min()) + int(ys.max()))))
        center_x = int(round(0.5 * (int(xs.min()) + int(xs.max()))))
        half = crop // 2
        top = min(max(center_y - half, 0), y.shape[0] - crop)
        left = min(max(center_x - half, 0), y.shape[1] - crop)
        patch_y = y[top : top + crop, left : left + crop]
        patch_x = normal[top : top + crop, left : left + crop]
        patch_m = mask[top : top + crop, left : left + crop]
        inside = patch_m
        l1_inside = float(np.abs(patch_y[inside].astype(np.int16) - patch_x[inside].astype(np.int16)).mean()) if inside.any() else 0.0
        l1_outside = float(np.abs(patch_y[~inside].astype(np.int16) - patch_x[~inside].astype(np.int16)).mean()) if (~inside).any() else 0.0
        zoomed = np.asarray(
            Image.fromarray(patch_y).resize((crop * zoom, crop * zoom), Image.Resampling.NEAREST)
        )
        Image.fromarray(patch_y).save(output / f"{order:02d}_{row['job_id']}_crop.png")
        Image.fromarray(zoomed).save(output / f"{order:02d}_{row['job_id']}_zoom.png")
        tiles.append(Image.fromarray(zoomed))
        selection.append(
            {
                "order": order,
                "job_id": row["job_id"],
                "placed_pixels": int(mask.sum()),
                "crop_top": top,
                "crop_left": left,
                "l1_inside": l1_inside,
                "l1_outside": l1_outside,
            }
        )
    sheet = Image.new("RGB", (tiles[0].width * count, tiles[0].height))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * tile.width, 0))
    sheet.save(output / "panel.png")
    write_json_atomic(output / "selection.json", {"seed": seed, "rows": selection})
    return {"count": count, "output": str(output)}


__all__ = [
    "A2_SIZE",
    "EVAL_SHORT_SIDE",
    "EVAL_SIZE_MULTIPLE",
    "NativeWindow",
    "WINDOW_MARGIN",
    "WINDOW_SIZE",
    "crop_to_hw",
    "extract_window",
    "geometry_for_rgb",
    "heldout_letterbox_area_stats",
    "letterbox_binary",
    "letterbox_rgb",
    "lift_binary_mask",
    "lift_float_map",
    "load_native_mask",
    "load_native_rgb",
    "pack_generated_letterbox",
    "pad_to_square",
    "paste_window",
    "short_side_square_size",
    "stage_letterbox_eval_layout",
    "window_from_mask",
    "write_mask_center_crops",
]

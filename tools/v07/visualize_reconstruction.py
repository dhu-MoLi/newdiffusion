#!/usr/bin/env python3
"""Random qualitative panels for reconstruction backends.

For each sampled normal image it places a real few-shot mask (or a B0 sample),
then renders neighbor / telea / learned completions and saves a contact sheet
with difference maps and per-method feature metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_c.visa_pilot import (
    fewshot_identity_records,
    load_visa_identity,
    load_visa_inventory_for_split,
    normal_identity_records,
)
from cadiff.v07.reconstructor import (
    LearnedReconstructor,
    _fast_neighbor_fill,
    _feature_cosine,
    place_shape_in_support,
    sample_hole,
)
from cadiff.v07.reconstruction import make_telea
from cadiff.v07.shape_prior import ParametricShapePrior

TILE = 256


def _label(image: np.ndarray, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + 22), (0, 0, 0))
    canvas.paste(Image.fromarray(image), (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), text, fill=(255, 255, 255))
    return canvas


def _resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (TILE, TILE), interpolation=cv2.INTER_AREA)


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    color = np.zeros_like(output)
    color[..., 0] = 255
    output[mask] = (0.55 * output[mask] + 0.45 * color[mask]).astype(np.uint8)
    return output


def _difference(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.abs(prediction.astype(np.float32) - target.astype(np.float32)).mean(axis=2)
    normalized = np.clip(difference / 40.0, 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _metrics(
    encoder: object,
    prediction: np.ndarray,
    target: np.ndarray,
    hole: np.ndarray,
) -> dict[str, float]:
    mask = hole.astype(bool)
    prediction_f = prediction.astype(np.float32) / 127.5 - 1.0
    target_f = target.astype(np.float32) / 127.5 - 1.0
    error = np.abs(prediction_f - target_f)
    masked_l1 = float(error[mask].mean()) if mask.any() else float("nan")
    mse = float((error[mask] ** 2).mean()) if mask.any() else float("nan")
    psnr = 10.0 * np.log10(4.0 / max(mse, 1e-12))
    prediction_t = (
        torch.from_numpy(prediction.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    )
    target_t = torch.from_numpy(target.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    hole_t = torch.from_numpy(hole.astype(np.float32)).unsqueeze(0)
    cosine = _feature_cosine(encoder, prediction_t, target_t, hole_t)
    return {"masked_l1": masked_l1, "psnr": psnr, "feature_cos": cosine}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--paths", default="configs/paths/asus-mvtec.yaml")
    parser.add_argument("--split", default="mvtec")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--real-mask-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    project = load_config(ROOT / arguments.paths)
    inventory = load_visa_inventory_for_split(
        project.dataset_root, arguments.category, arguments.split, None
    )
    normals = normal_identity_records(inventory)
    loaded_normals = [
        load_visa_identity(record, dataset_root=inventory.source_root) for record in normals
    ]
    masks_by_type: dict[str, list[np.ndarray]] = {}
    for record in fewshot_identity_records(inventory):
        loaded = load_visa_identity(record, dataset_root=inventory.source_root)
        if loaded.anomaly_gt is None:
            continue
        masks_by_type.setdefault(record.defect_type, []).append(
            np.ascontiguousarray(loaded.anomaly_gt, dtype=bool)
        )
    raw_masks = [mask for masks in masks_by_type.values() for mask in masks]
    prior = ParametricShapePrior(masks_by_type, rng_seed=arguments.seed)
    encoder = load_official_dinov2_encoder(device=arguments.device)

    reconstructor_root = Path(project.runs_root) / "v07" / "reconstructors" / arguments.category
    checkpoint = arguments.checkpoint
    if checkpoint is None:
        candidates = sorted(reconstructor_root.rglob("checkpoint.pt"))
        if not candidates:
            raise FileNotFoundError(f"no checkpoint under {reconstructor_root}")
        checkpoint = candidates[-1]
    learned = LearnedReconstructor(checkpoint, device=arguments.device)
    telea = make_telea()

    support_cache = None
    cache_candidates = sorted(reconstructor_root.rglob("supports.npz"))
    if cache_candidates:
        payload = np.load(cache_candidates[-1])
        supports = payload["supports"]
        if supports.shape[0] == len(loaded_normals):
            support_cache = supports

    output = (
        arguments.output
        if arguments.output is not None
        else Path(project.runs_root)
        / "v07"
        / "visualizations"
        / f"{arguments.category}_{checkpoint.parent.parent.name}"
    )
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(arguments.seed)
    indices = rng.permutation(len(loaded_normals))[: arguments.samples]
    rows: list[Image.Image] = []
    records: list[dict[str, object]] = []
    for order, index in enumerate(indices):
        normal = loaded_normals[int(index)]
        rgb = np.ascontiguousarray(normal.rgb)
        valid = np.ascontiguousarray(normal.valid)
        if support_cache is not None:
            support = np.ascontiguousarray(support_cache[int(index)], dtype=bool)
        else:
            features = encoder.encode(rgb)
            support = tokencut_object_mask(features.tokens, valid)
            if not support.any():
                support = valid
        hole = None
        if raw_masks and rng.rand() < arguments.real_mask_fraction:
            shape = raw_masks[int(rng.randint(0, len(raw_masks)))]
            hole = place_shape_in_support(shape, support, valid, rng)
        if hole is None or int(hole.sum()) < 64:
            hole = sample_hole(
                prior, tuple(masks_by_type), support, valid, rng
            )
        neighbor = _fast_neighbor_fill(rgb, hole)
        telea_fill = telea(rgb, hole, valid)
        learned_fill = learned(rgb, hole, valid)
        methods = {
            "neighbor": neighbor,
            "telea": telea_fill,
            "learned": learned_fill,
        }
        method_metrics = {
            name: _metrics(encoder, value, rgb, hole) for name, value in methods.items()
        }
        tiles = [
            _label(_resize(_overlay(rgb, hole)), f"{order} normal+hole"),
            _label(_resize(rgb), "target"),
        ]
        for name, value in methods.items():
            metrics = method_metrics[name]
            tiles.append(
                _label(
                    _resize(value),
                    f"{name} cos={metrics['feature_cos']:.3f}",
                )
            )
        tiles.append(_label(_resize(_difference(learned_fill, rgb)), "|learned-target|"))
        row = Image.new(
            "RGB",
            (TILE * len(tiles), tiles[0].height),
            (0, 0, 0),
        )
        for column, tile in enumerate(tiles):
            row.paste(tile, (column * TILE, 0))
        row.save(output / f"panel_{order:02d}.png")
        rows.append(row)
        records.append(
            {
                "index": int(index),
                "hole_pixels": int(hole.sum()),
                "methods": method_metrics,
            }
        )

    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), (0, 0, 0))
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height
    sheet.save(output / "contact_sheet.png")
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "category": arguments.category,
                "checkpoint": str(checkpoint),
                "samples": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

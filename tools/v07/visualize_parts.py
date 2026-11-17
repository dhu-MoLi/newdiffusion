#!/usr/bin/env python3
"""Visualise part decomposition (SAM vs spectral) with support contour."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v07.structure_parts import PartConfig, parse_structure

MVTEC_CLASSES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
PALETTE = [
    (255, 99, 71), (60, 179, 113), (30, 144, 255), (255, 215, 0), (186, 85, 211),
    (255, 140, 0), (0, 206, 209), (220, 20, 60), (154, 205, 50), (123, 104, 238),
    (255, 105, 180), (46, 139, 87), (70, 130, 180), (244, 164, 96),
]


def _load(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _tile(rgb: np.ndarray, parts, support: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for index, part in enumerate(parts):
        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        out[part.mask] = out[part.mask] * 0.45 + color * 0.55
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(
        support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    for part in parts:
        centre = (int(part.centroid[0]), int(part.centroid[1]))
        cv2.circle(out, centre, 3, (0, 0, 0), -1)
    viz = cv2.resize(out, (320, 320), interpolation=cv2.INTER_AREA)
    bar = np.zeros((52, 320, 3), dtype=np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 220, 170), 1, cv2.LINE_AA)
    return np.vstack([bar, viz])


def _sheet(tiles: list[np.ndarray], cols: int, path: Path) -> None:
    rows = int(np.ceil(len(tiles) / cols))
    height, width = tiles[0].shape[:2]
    canvas = np.full((rows * height, cols * width, 3), 30, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvtec-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    arguments = parser.parse_args(argv)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam_model = sam_model_registry["vit_h"](checkpoint=str(arguments.sam_checkpoint))
    sam_model.to(arguments.device)
    generator = SamAutomaticMaskGenerator(
        sam_model, points_per_side=32, pred_iou_thresh=0.85,
        stability_score_thresh=0.9, min_mask_region_area=100,
    )
    encoder = load_official_dinov2_encoder(device=arguments.device)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(arguments.seed)
    started = time.time()

    sam_cfg = PartConfig(part_source="sam")
    spec_cfg = PartConfig(part_source="spectral")

    def parse(rgb, features, cfg, sam):
        valid = np.ones((512, 512), dtype=bool)
        support = tokencut_object_mask(features.tokens, valid)
        if not support.any():
            support = valid.copy()
        return support, parse_structure(
            rgb, support, valid, features.tokens, cfg, sam=sam
        )

    def pick(root: Path, category: str, kind: str) -> Path:
        folder = (
            root / category / "train" / "good" if kind == "mvtec"
            else root / category / "Data" / "Images" / "Normal"
        )
        files = sorted(folder.glob("*"))
        return files[int(rng.randint(0, len(files)))]

    sam_tiles, spec_tiles = [], []
    for category in MVTEC_CLASSES:
        rgb = _load(pick(arguments.mvtec_root, category, "mvtec"))
        features = encoder.encode(rgb)
        support, sam_report = parse(rgb, features, sam_cfg, generator)
        _support, spec_report = parse(rgb, features, spec_cfg, None)
        mode = "TEXTURE (no parts)" if sam_report.texture_mode else f"{sam_report.k} parts"
        sam_tiles.append(_tile(rgb, sam_report.parts, support, f"MVTec {category} SAM", mode))
        spec_tiles.append(_tile(rgb, spec_report.parts, support, f"MVTec {category} spectral", f"{spec_report.k} parts"))
        print(category, "sam", sam_report.k, "texture", sam_report.texture_mode, "| spec", spec_report.k)
    _sheet(sam_tiles, 5, output / "mvtec_parts_sam.png")
    _sheet(spec_tiles, 5, output / "mvtec_parts_spectral.png")

    visa_tiles = []
    for category in VISA_CLASSES:
        rgb = _load(pick(arguments.visa_root, category, "visa"))
        features = encoder.encode(rgb)
        support, sam_report = parse(rgb, features, sam_cfg, generator)
        mode = "TEXTURE (no parts)" if sam_report.texture_mode else f"{sam_report.k} parts"
        visa_tiles.append(_tile(rgb, sam_report.parts, support, f"VisA {category} SAM", mode))
        print("VisA", category, "sam", sam_report.k, "texture", sam_report.texture_mode)
    _sheet(visa_tiles, 4, output / "visa_parts_sam.png")
    print(f"written {output} ({time.time()-started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

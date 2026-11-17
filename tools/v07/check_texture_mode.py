#!/usr/bin/env python3
"""Steps 1-3 visual check: one random image per class (MVTec + VisA).

Saves contact sheets with the object support overlay and the texture-mode
decision, plus a JSON with scores.
"""

from __future__ import annotations

import argparse
import json
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
MVTEC_TEXTURE = {"carpet", "grid", "leather", "tile", "wood"}
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def _load(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _pick_mvtec(root: Path, category: str, rng: np.random.RandomState) -> Path:
    files = sorted((root / category / "train" / "good").glob("*"))
    return files[int(rng.randint(0, len(files)))]


def _pick_visa(root: Path, category: str, rng: np.random.RandomState) -> Path:
    files = sorted((root / category / "Data" / "Images" / "Normal").glob("*"))
    return files[int(rng.randint(0, len(files)))]


def _overlay(rgb: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    tint = np.zeros_like(out)
    tint[..., 1] = 255
    blended = (out.astype(np.float32) * 0.72 + tint.astype(np.float32) * 0.28)
    out = np.where(support[..., None], blended, out.astype(np.float32)).astype(np.uint8)
    edge = support ^ cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out[edge] = (0, 255, 0)
    return out


def _tile(rgb: np.ndarray, support: np.ndarray, title: str, decision: str) -> np.ndarray:
    viz = _overlay(rgb, support)
    viz = cv2.resize(viz, (320, 320), interpolation=cv2.INTER_AREA)
    bar = np.zeros((56, 320, 3), dtype=np.uint8)
    color = (80, 200, 255) if decision == "TEXTURE" else (120, 255, 120)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, decision, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261116)
    arguments = parser.parse_args(argv)

    encoder = load_official_dinov2_encoder(device=arguments.device)
    cfg = PartConfig()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(arguments.seed)
    started = time.time()

    results: dict[str, dict] = {}
    tiles: list[np.ndarray] = []
    correct = 0
    total = 0
    for category in MVTEC_CLASSES:
        path = _pick_mvtec(arguments.mvtec_root, category, rng)
        rgb = _load(path)
        features = encoder.encode(rgb)
        valid = np.ones((512, 512), dtype=bool)
        support = tokencut_object_mask(features.tokens, valid)
        if not support.any():
            support = valid.copy()
        parsed = parse_structure(rgb, support, valid, features.tokens, cfg, category=category, image_id=str(path.name))
        expected = category in MVTEC_TEXTURE
        decision = "TEXTURE" if parsed.texture_mode else "OBJECT"
        ok = (parsed.texture_mode == expected)
        correct += int(ok)
        total += 1
        results[f"MVTec/{category}"] = {
            "image": str(path), "expected_texture": expected,
            "texture_mode": parsed.texture_mode, "ok": ok,
            "k": parsed.k, "texture_score": parsed.texture_score,
            "separability": parsed.separability, "support_ratio": parsed.support_ratio,
        }
        title = f"MVTec {category}  {'OK' if ok else 'MISS'}"
        tiles.append(_tile(rgb, support, title, f"{decision} k={parsed.k} tex={parsed.texture_score:.3f} sep={parsed.separability:.2f}"))
    _sheet(tiles, 5, output / "mvtec_texture_mode.png")

    visatiles: list[np.ndarray] = []
    for category in VISA_CLASSES:
        path = _pick_visa(arguments.visa_root, category, rng)
        rgb = _load(path)
        features = encoder.encode(rgb)
        valid = np.ones((512, 512), dtype=bool)
        support = tokencut_object_mask(features.tokens, valid)
        if not support.any():
            support = valid.copy()
        parsed = parse_structure(rgb, support, valid, features.tokens, cfg, category=category, image_id=str(path.name))
        decision = "TEXTURE" if parsed.texture_mode else "OBJECT"
        results[f"VisA/{category}"] = {
            "image": str(path), "expected_texture": None,
            "texture_mode": parsed.texture_mode,
            "k": parsed.k, "texture_score": parsed.texture_score,
            "separability": parsed.separability, "support_ratio": parsed.support_ratio,
        }
        title = f"VisA {category}"
        visatiles.append(_tile(rgb, support, title, f"{decision} k={parsed.k} tex={parsed.texture_score:.3f} sep={parsed.separability:.2f}"))
    _sheet(visatiles, 4, output / "visa_texture_mode.png")

    summary = {
        "schema": "v07_texture_mode_check_v1",
        "mvtec_accuracy": correct / max(1, total),
        "mvtec_total": total,
        "seconds": round(time.time() - started, 1),
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"MVTec texture-mode accuracy {correct}/{total} = {correct/total:.3f}")
    for key, value in results.items():
        if key.startswith("MVTec") and not value["ok"]:
            print(f"   MISS {key}: expected_texture={value['expected_texture']} got={value['texture_mode']} k={value['k']} tex={value['texture_score']:.3f} sep={value['separability']:.2f}")
    print(f"written {output} ({summary['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

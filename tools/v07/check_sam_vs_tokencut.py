#!/usr/bin/env python3
"""Compare SAM instance masks with TokenCut support on one image per class."""

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
    (255, 105, 180), (46, 139, 87),
]


def _load(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _pick_mvtec(root: Path, category: str, rng: np.random.RandomState, count: int = 1) -> list[Path]:
    files = sorted((root / category / "train" / "good").glob("*"))
    count = min(count, len(files))
    index = rng.choice(len(files), size=count, replace=False)
    return [files[int(i)] for i in sorted(index)]


def _pick_visa(root: Path, category: str, rng: np.random.RandomState, count: int = 1) -> list[Path]:
    files = sorted((root / category / "Data" / "Images" / "Normal").glob("*"))
    count = min(count, len(files))
    index = rng.choice(len(files), size=count, replace=False)
    return [files[int(i)] for i in sorted(index)]


def _border_fraction(segment: np.ndarray, width: int = 3) -> float:
    border = np.zeros_like(segment)
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True
    return float((segment & border).sum() / max(1, int(segment.sum())))


def _filter_instances(
    masks: list[dict],
    *,
    min_ratio: float = 0.005,
    max_ratio: float = 0.60,
    max_border: float = 0.15,
    max_instances: int = 12,
) -> list[np.ndarray]:
    """Keep object-like instances: no background sheet, no image border.

    SAM's automatic generator returns the table/background as its largest
    segment; without this filter the comparison is meaningless.
    """

    total = 512 * 512
    kept: list[np.ndarray] = []
    for item in sorted(masks, key=lambda m: m["area"], reverse=True):
        segment = item["segmentation"].astype(bool)
        area = float(segment.sum())
        if area < min_ratio * total or area > max_ratio * total:
            continue
        if _border_fraction(segment) > max_border:
            continue
        duplicate = False
        for existing in kept:
            inter = float((segment & existing).sum())
            if inter / max(1.0, area) > 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append(segment)
        if len(kept) >= max_instances:
            break
    return kept


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def _overlay(rgb: np.ndarray, instances: list[np.ndarray], support: np.ndarray, best: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for index, segment in enumerate(instances):
        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        out[segment] = out[segment] * 0.45 + color * 0.55
    out = out.astype(np.uint8)
    contour, _ = cv2.findContours(
        support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contour, -1, (0, 255, 0), 2)
    best_contour, _ = cv2.findContours(best.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, best_contour, -1, (255, 255, 255), 2)
    return out


def _tile(rgb, instances, support, title, line, best):
    viz = _overlay(rgb, instances, support, best)
    viz = cv2.resize(viz, (320, 320), interpolation=cv2.INTER_AREA)
    bar = np.zeros((56, 320, 3), dtype=np.uint8)
    cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, line, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 220, 160), 1, cv2.LINE_AA)
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
    parser.add_argument("--samples", type=int, default=2, help="images per class")
    parser.add_argument("--sam-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    arguments = parser.parse_args(argv)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[arguments.sam_type](checkpoint=str(arguments.sam_checkpoint))
    sam.to(arguments.device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.9,
        min_mask_region_area=100,
    )
    encoder = load_official_dinov2_encoder(device=arguments.device)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(arguments.seed)
    results: dict[str, dict] = {}
    started = time.time()

    def analyse(path: Path) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, dict, np.ndarray]:
        rgb = _load(path)
        features = encoder.encode(rgb)
        valid = np.ones((512, 512), dtype=bool)
        support = tokencut_object_mask(features.tokens, valid)
        if not support.any():
            support = valid.copy()
        raw = generator.generate(rgb)
        instances = _filter_instances(raw)
        union = np.zeros((512, 512), dtype=bool)
        for segment in instances:
            union |= segment
        best = max(instances, key=lambda seg: _iou(support, seg)) if instances else np.zeros((512, 512), dtype=bool)
        stats = {
            "n_instances": len(instances),
            "best_iou": _iou(support, best),
            "best_ratio": float(best.mean()),
            "union_ratio": float(union.mean()),
            "support_ratio": float(support.mean()),
            "support_covered_by_best": float((support & best).sum() / max(1, support.sum())),
            "best_covered_by_support": float((best & support).sum() / max(1, best.sum())),
            "support_covered_by_union": float((support & union).sum() / max(1, support.sum())),
            "raw_masks": len(raw),
        }
        return rgb, instances, support, stats, best

    def run_class(prefix: str, category: str, paths: list[Path], tiles: list) -> dict:
        rows = []
        for index, path in enumerate(paths):
            rgb, instances, support, stats, best = analyse(path)
            rows.append(stats)
            line = (f"SAM n={stats['n_instances']} IoU={stats['best_iou']:.2f} "
                    f"cover={stats['support_covered_by_best']:.2f}")
            tiles.append(_tile(rgb, instances, support, f"{prefix} {category} #{index+1}", line, best))
            print(prefix, category, index + 1, json.dumps(stats))
        mean = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            mean[key] = float(np.mean(values))
        mean["samples"] = len(rows)
        return mean

    mvtec_tiles = []
    for category in MVTEC_CLASSES:
        paths = _pick_mvtec(arguments.mvtec_root, category, rng, arguments.samples)
        results[f"MVTec/{category}"] = run_class("MVTec", category, paths, mvtec_tiles)
    _sheet(mvtec_tiles, 5, output / f"mvtec_sam_vs_tokencut_{arguments.samples}x.png")

    visa_tiles = []
    for category in VISA_CLASSES:
        paths = _pick_visa(arguments.visa_root, category, rng, arguments.samples)
        results[f"VisA/{category}"] = run_class("VisA", category, paths, visa_tiles)
    _sheet(visa_tiles, 4, output / f"visa_sam_vs_tokencut_{arguments.samples}x.png")

    for prefix in ("MVTec", "VisA"):
        rows = [value for key, value in results.items() if key.startswith(prefix)]
        if rows:
            print(prefix, "mean n_instances", round(float(np.mean([r["n_instances"] for r in rows])), 2),
                  "mean bestIoU", round(float(np.mean([r["best_iou"] for r in rows])), 3),
                  "mean coverage", round(float(np.mean([r["support_covered_by_best"] for r in rows])), 3))
    summary = {"schema": "v07_sam_vs_tokencut_v1", "seconds": round(time.time() - started, 1),
               "results": results}
    (output / "results_sam.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {output} ({summary['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

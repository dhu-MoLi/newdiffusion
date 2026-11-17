#!/usr/bin/env python3
"""Diagnose Omega reasonableness and VisA support-vs-mask mismatch."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.v06_a.structure import physical_sdf_px_512
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v07.allowed_region import OmegaConfig, build_allowed_region
from cadiff.v07.location import ObjectLocationPrior
from cadiff.v07.structure_parts import PartConfig, parse_structure

CANVAS = 512
SPLITS = {c: ("20260922-seas" if c in ("candle", "capsules", "cashew", "chewinggum") else "20260927-magic")
          for c in ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
                    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((CANVAS, CANVAS), Image.Resampling.BICUBIC), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((CANVAS, CANVAS), Image.Resampling.NEAREST)
    array = np.asarray(image)
    top = int(array.max())
    return array > ((top // 2) if top > 1 else 0)


def _tile(rgb, support=None, omega=None, gt=None, title="", subtitle="") -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if omega is not None:
        tint = np.array([0, 160, 255], np.float32)
        out[omega] = out[omega] * 0.35 + tint * 0.65
    out = out.astype(np.uint8)
    if omega is not None:
        contours, _ = cv2.findContours(omega.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 120, 255), 3)
    if support is not None:
        contours, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    if gt is not None:
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 0, 0), 2)
    viz = cv2.resize(out, (300, 300), interpolation=cv2.INTER_AREA)
    bar = np.zeros((46, 300, 3), dtype=np.uint8)
    cv2.putText(bar, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, subtitle, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 220, 170), 1, cv2.LINE_AA)
    return np.vstack([bar, viz])


def _sheet(tiles, cols, path):
    rows = int(np.ceil(len(tiles) / cols))
    h, w = tiles[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    encoder = load_official_dinov2_encoder(device=args.device)
    sam_model = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint)); sam_model.to(args.device)
    generator = SamAutomaticMaskGenerator(sam_model, points_per_side=32, pred_iou_thresh=0.85,
                                          stability_score_thresh=0.9, min_mask_region_area=100)
    args.output.mkdir(parents=True, exist_ok=True)
    part_cfg = PartConfig(part_source="sam")
    omega_cfg = OmegaConfig()
    rng = np.random.RandomState(20261117)

    def support_of(rgb):
        feats = encoder.encode(rgb)
        valid = np.ones((CANVAS, CANVAS), bool)
        support = tokencut_object_mask(feats.tokens, valid)
        if not support.any():
            support = valid.copy()
        return feats, support, valid

    # ---------- Sheet 1: VisA support vs GT mask ----------
    visa_metrics = {}
    tiles = []
    for category in ("candle", "capsules", "cashew", "chewinggum"):
        split = SPLITS[category]
        manifest = ROOT / "splits" / "v06_b" / "visa" / category / split / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        fs = [r for r in rows if "fewshot_train" in r.get("roles", []) and r.get("mask_relpath")]
        comp_counts, largest_fracs, in_support, distances = [], [], [], []
        for row in fs[:8]:
            rgb = _load_rgb(args.visa_root / row["image_relpath"])
            gt = _load_mask(args.visa_root / row["mask_relpath"])
            if not gt.any():
                continue
            _feats, support, _valid = support_of(rgb)
            gt_in_support = float((gt & support).sum() / max(1, int(gt.sum())))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
            areas = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)], reverse=True)
            comp = len(areas); largest = areas[0] / max(1, int(support.sum())) if areas else 0.0
            if (gt & support).any():
                dist = 0.0
            else:
                rows_g, cols_g = np.nonzero(gt)
                distance = distance_transform_edt(~support)
                dist = float(distance[rows_g, cols_g].min() / np.sqrt(max(1, int(support.sum()))))
            comp_counts.append(comp); largest_fracs.append(largest)
            in_support.append(gt_in_support); distances.append(dist)
            if len(tiles) < 8:
                tiles.append(_tile(rgb, support=support, gt=gt, title=f"VisA {category}",
                                   subtitle=f"comp={comp} GTinSup={gt_in_support:.2f} dist={dist:.2f}"))
        visa_metrics[category] = {
            "support_components_mean": float(np.mean(comp_counts)),
            "largest_component_fraction": float(np.mean(largest_fracs)),
            "gt_in_support_mean": float(np.mean(in_support)),
            "gt_distance_norm_mean": float(np.mean(distances)),
        }
        print(category, json.dumps(visa_metrics[category]))
    _sheet(tiles, 4, args.output / "visa_support_vs_gt.png")

    # ---------- Sheet 2/3: Omega on normal images ----------
    def omega_tiles(class_root: Path, categories, kind: str):
        out_tiles = []
        for category in categories:
            if kind == "mvtec":
                files = sorted((class_root / category / "train" / "good").glob("*"))
            else:
                files = sorted((class_root / category / "Data" / "Images" / "Normal").glob("*"))
            if not files:
                continue
            path = files[int(rng.randint(0, len(files)))]
            rgb = _load_rgb(path)
            feats, support, valid = support_of(rgb)
            report = parse_structure(rgb, support, valid, feats.tokens, part_cfg, sam=generator)
            frame = valid if report.texture_mode else support
            if kind == "mvtec":
                manifest = ROOT / "splits" / "mvtec" / category / "manifest.jsonl"
                data_root = args.mvtec_root
            else:
                manifest = ROOT / "splits" / "v06_b" / "visa" / category / SPLITS[category] / "manifest.jsonl"
                data_root = args.visa_root
            rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()] if manifest.is_file() else []
            few = [r for r in rows if "fewshot_train" in r.get("roles", []) and r.get("mask_relpath")]
            masks = [_load_mask(data_root / r["mask_relpath"]) for r in few]
            masks = [m for m in masks if m.any()]
            try:
                prior = ObjectLocationPrior.from_masks({"all": masks}, {"all": [frame] * len(masks)})
            except ValueError:
                prior = ObjectLocationPrior.from_masks({"all": masks}, {"all": [valid] * len(masks)})
                frame = valid
            ratios = [float(m.sum()) / max(1, int(frame.sum())) for m in masks]
            structural = bool(np.median(ratios) > 0.15)
            sdf = physical_sdf_px_512(frame)
            target = 0.85 if structural else 0.7
            omega, _score, stats, _domain = build_allowed_region(
                rgb, frame, valid, prior, "all", fewshot_area_ratios=ratios, sdf=sdf,
                cfg=omega_cfg, structural=structural, texture_mode=report.texture_mode,
                domain_dilate_px=64 if structural else 24, area_ratio_override=target,
                parts=report.parts,
            )
            out_tiles.append(_tile(rgb, support=support, omega=omega, title=f"{kind} {category}",
                                   subtitle=f"tex={report.texture_mode} parts={report.k} Ω={stats['area_ratio']:.2f}"))
        return out_tiles

    mv = omega_tiles(args.mvtec_root, ["bottle", "carpet", "grid", "transistor"], "mvtec")
    _sheet(mv, 4, args.output / "omega_mvtec.png")
    va = omega_tiles(args.visa_root, ["candle", "capsules", "cashew", "chewinggum"], "visa")
    _sheet(va, 4, args.output / "omega_visa.png")

    (args.output / "diag.json").write_text(json.dumps({"visa": visa_metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

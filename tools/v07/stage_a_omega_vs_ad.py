#!/usr/bin/env python3
"""Stage A: AD-generated masks, random pairing vs Omega-constrained placement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

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
SPLITS = {
    "candle": "20260922-seas", "capsules": "20260922-seas",
    "cashew": "20260922-seas", "chewinggum": "20260922-seas",
    "fryum": "20260927-magic", "macaroni1": "20260927-magic",
    "macaroni2": "20260927-magic", "pcb1": "20260927-magic",
    "pcb2": "20260927-magic", "pcb3": "20260927-magic",
    "pcb4": "20260927-magic", "pipe_fryum": "20260927-magic",
}


def _load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((CANVAS, CANVAS), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _load_mask(path: Path, size: int = CANVAS) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(image)
    top = int(array.max())
    # VisA masks use values 0..1, 0..6; MVTec/AD use 0/255
    return array > ((top // 2) if top > 1 else 0)


def _best_placement(omega: np.ndarray, mask: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Translation of `mask` that maximises overlap with omega (mask must fit)."""

    h, w = mask.shape
    correlation = fftconvolve(omega.astype(np.float32), mask[::-1, ::-1].astype(np.float32), mode="full")
    # placements fully inside the canvas: top-left in [0, CANVAS-h] x [0, CANVAS-w]
    valid = correlation[h - 1 : CANVAS, w - 1 : CANVAS]
    index = np.unravel_index(int(np.argmax(valid)), valid.shape)
    return float(valid[index]) / max(1.0, float(mask.sum())), (int(index[0]), int(index[1]))


def _place(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    out = np.zeros((CANVAS, CANVAS), dtype=bool)
    out[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return out


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--ad-mask-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("/home/asus/.cache/sam/sam_vit_h_4b8939.pth"))
    parser.add_argument("--categories", nargs="+", default=["candle", "capsules", "cashew", "chewinggum"])
    parser.add_argument("--images-per-class", type=int, default=6)
    parser.add_argument("--ad-masks", type=int, default=60)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    encoder = load_official_dinov2_encoder(device=arguments.device)
    sam_model = sam_model_registry["vit_h"](checkpoint=str(arguments.sam_checkpoint))
    sam_model.to(arguments.device)
    generator = SamAutomaticMaskGenerator(
        sam_model, points_per_side=32, pred_iou_thresh=0.85,
        stability_score_thresh=0.9, min_mask_region_area=100,
    )
    omega_cfg = OmegaConfig()
    part_cfg = PartConfig(part_source="sam")
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(20261117)
    started = time.time()
    results: dict[str, dict] = {}
    tiles: list[np.ndarray] = []

    for category in arguments.categories:
        split = SPLITS.get(category, "20260922-seas")
        manifest = ROOT / "splits" / "v06_b" / "visa" / category / split / "manifest.jsonl"
        fewshot = []
        if manifest.is_file():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "fewshot_train" in row.get("roles", []) and row.get("mask_relpath"):
                    fewshot.append(row)
        masks = [_load_mask(arguments.visa_root / row["mask_relpath"]) for row in fewshot]
        if not masks:
            print(category, "no few-shot masks"); continue

        normal_dir = arguments.visa_root / category / "Data" / "Images" / "Normal"
        normals = sorted(normal_dir.glob("*"))
        if len(normals) > arguments.images_per_class:
            index = rng.choice(len(normals), size=arguments.images_per_class, replace=False)
            normals = [normals[int(i)] for i in sorted(index)]

        views = []
        for path in normals:
            rgb = _load_rgb(path)
            features = encoder.encode(rgb)
            valid = np.ones((CANVAS, CANVAS), dtype=bool)
            support = tokencut_object_mask(features.tokens, valid)
            if not support.any():
                support = valid.copy()
            report = parse_structure(rgb, support, valid, features.tokens, part_cfg, sam=generator)
            views.append({
                "rgb": rgb, "support": support, "valid": valid,
                "texture_mode": report.texture_mode,
                "parts": [] if report.texture_mode else report.parts,
                "sdf": physical_sdf_px_512(support),
            })

        useful = [view["valid"] if view["texture_mode"] else view["support"] for view in views]
        area_ratios = [float(m.sum()) / max(1, int(u.sum())) for m, u in zip(masks, useful * len(masks))]
        outside = [1.0 - float((m & u).sum()) / max(1, int(m.sum())) for m, u in zip(masks, useful * len(masks))]
        structural = bool(
            np.median(area_ratios) > 0.15
            or np.median(outside) > 0.25
        )
        omegas = []
        for view in views:
            frame_support = view["valid"] if view["texture_mode"] else view["support"]
            try:
                prior_view = ObjectLocationPrior.from_masks(
                    {"all": masks}, {"all": [frame_support] * len(masks)}
                )
            except ValueError:
                # masks do not intersect the object support: fall back to the
                # image frame so the prior stays defined
                frame_support = view["valid"]
                prior_view = ObjectLocationPrior.from_masks(
                    {"all": masks}, {"all": [frame_support] * len(masks)}
                )
            sdf = view["sdf"] if not view["texture_mode"] else physical_sdf_px_512(frame_support)
            target = 0.85 if structural else 0.7
            omega, _score, _stats, _domain = build_allowed_region(
                view["rgb"], frame_support, view["valid"], prior_view, "all",
                fewshot_area_ratios=area_ratios, sdf=sdf, cfg=omega_cfg,
                structural=structural, texture_mode=view["texture_mode"],
                domain_dilate_px=64 if structural else 24,
                area_ratio_override=target, parts=view["parts"],
            )
            omegas.append(omega)

        ad_dir = arguments.ad_mask_root / category / "ko"
        ad_files = sorted(ad_dir.glob("*.jpg")) if ad_dir.is_dir() else []
        if not ad_files:
            print(category, "no AD masks"); continue
        pick = rng.choice(len(ad_files), size=min(arguments.ad_masks, len(ad_files)), replace=False)
        ad_masks = [_load_mask(ad_files[int(i)]) for i in sorted(pick)]

        native_align, constrained_align, rejected, displacements = [], [], 0, []
        for index, mask in enumerate(ad_masks):
            view_index = int(rng.randint(0, len(views)))
            omega = omegas[view_index]
            native_align.append(float((mask & omega).sum()) / max(1, int(mask.sum())))
            overlap, (top, left) = _best_placement(omega, mask)
            if overlap < 0.95:
                rejected += 1
                continue
            constrained = _place(mask, top, left)
            constrained_align.append(float((constrained & omega).sum()) / max(1, int(mask.sum())))
            displacements.append(float(np.hypot(top - CANVAS // 2, left - CANVAS // 2)))
            if index < 3:
                rgb = views[view_index]["rgb"]
                for title, overlay in (("native", mask), ("omega", constrained)):
                    canvas = rgb.astype(np.float32).copy()
                    canvas[overlay] = canvas[overlay] * 0.4 + np.array([255, 80, 80]) * 0.6
                    canvas = canvas.astype(np.uint8)
                    contours, _ = cv2.findContours(omega.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
                    tile = cv2.resize(canvas, (240, 240), interpolation=cv2.INTER_AREA)
                    bar = np.zeros((40, 240, 3), dtype=np.uint8)
                    cv2.putText(bar, f"{category} {title}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(bar, f"inside Omega={constrained_align[-1] if title=='omega' else native_align[-1]:.2f}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 220, 170), 1, cv2.LINE_AA)
                    tiles.append(np.vstack([bar, tile]))

        results[category] = {
            "n_ad_masks": len(ad_masks),
            "native_alignment_mean": float(np.mean(native_align)),
            "native_alignment_median": float(np.median(native_align)),
            "constrained_alignment_mean": float(np.mean(constrained_align)) if constrained_align else None,
            "constrained_alignment_median": float(np.median(constrained_align)) if constrained_align else None,
            "rejected": rejected,
            "rejected_rate": rejected / max(1, len(ad_masks)),
            "structural": structural,
            "omega_area_ratio": float(np.mean([float(o.mean()) for o in omegas])),
            "ad_area_ratio_mean": float(np.mean([float(m.mean()) for m in ad_masks])),
            "real_area_ratio_mean": float(np.mean(area_ratios)),
            "ad_vs_real_area_ks": _ks(np.array([float(m.mean()) for m in ad_masks]), np.array(area_ratios)),
        }
        print(category, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in results[category].items()}))

    sheet_rows = int(np.ceil(len(tiles) / 4))
    if tiles:
        height, width = tiles[0].shape[:2]
        canvas = np.full((sheet_rows * height, 4 * width, 3), 30, dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row, col = divmod(index, 4)
            canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
        cv2.imwrite(str(output / "stage_a_native_vs_omega.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    summary = {"schema": "v08_stage_a_v1", "categories": results, "seconds": round(time.time() - started, 1)}
    (output / "stage_a.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {output} ({summary['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

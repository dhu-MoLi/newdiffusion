#!/usr/bin/env python3
"""Stage B data generation: AD masks paired with normal images.

Group A: AD masks at their own coordinates (AD's random-pairing behaviour).
Group B: the same pairs, mask relocated into the target's allowed region (Omega)
         within the normal image; if it cannot fit, another AD mask is drawn
         (recorded) -- that mask does not meet our placement standard.
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
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.v06_a.structure import physical_sdf_px_512
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.visa_exact_mask_r import load_visa_renderer
from cadiff.v07.allowed_region import OmegaConfig, build_allowed_region
from cadiff.v07.location import ObjectLocationPrior
from cadiff.v07.structure_parts import PartConfig, parse_structure

CANVAS = 512
SPLITS = {c: ("20260922-seas" if c in ("candle", "capsules", "cashew", "chewinggum") else "20260927-magic")
          for c in ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
                    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((CANVAS, CANVAS), Image.Resampling.BICUBIC), dtype=np.uint8)


def _load_mask(path: Path, size: int = CANVAS) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(image)
    top = int(array.max())
    return array > ((top // 2) if top > 1 else 0)


def _place(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    out = np.zeros((CANVAS, CANVAS), dtype=bool)
    out[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return out


def _best_placement(omega: np.ndarray, mask: np.ndarray) -> tuple[float, tuple[int, int]]:
    h, w = mask.shape
    correlation = fftconvolve(omega.astype(np.float32), mask[::-1, ::-1].astype(np.float32), mode="full")
    valid = correlation[h - 1 : CANVAS, w - 1 : CANVAS]
    index = np.unravel_index(int(np.argmax(valid)), valid.shape)
    return float(valid[index]) / max(1.0, float(mask.sum())), (int(index[0]), int(index[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--ad-mask-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--group", choices=["A", "B"], required=True)
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--normals", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args(argv)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    encoder = load_official_dinov2_encoder(device=args.device)
    sam_model = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint))
    sam_model.to(args.device)
    generator = SamAutomaticMaskGenerator(sam_model, points_per_side=32, pred_iou_thresh=0.85,
                                          stability_score_thresh=0.9, min_mask_region_area=100)
    part_cfg = PartConfig(part_source="sam")
    omega_cfg = OmegaConfig()
    rng = np.random.RandomState(args.seed)
    started = time.time()

    for category in args.categories:
        split = SPLITS[category]
        manifest = ROOT / "splits" / "v06_b" / "visa" / category / split / "manifest.jsonl"
        rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        defect_types = tuple(sorted({r["defect_type"] for r in rows}))
        few = [r for r in rows if "fewshot_train" in r.get("roles", []) and r.get("mask_relpath")]
        few_masks = [_load_mask(args.visa_root / r["mask_relpath"]) for r in few]
        few_masks = [m for m in few_masks if m.any()]

        checkpoint_dir = args.checkpoint_root / category / "exact-mask-r"
        ckpts = sorted(checkpoint_dir.glob("seed-*"))
        if not ckpts:
            print(category, "no R checkpoint"); continue
        checkpoint = ckpts[0] / "checkpoint"
        from cadiff.v06_c.can_exact_mask_r import P2A1Config

        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        raw_config = dict(metadata["config"])
        raw_config["defect_types"] = tuple(raw_config["defect_types"])
        config = P2A1Config(**raw_config)
        defect_types = tuple(config.defect_types)
        renderer = load_visa_renderer(checkpoint=checkpoint, model_root=args.model_root,
                                      config=config, device=args.device)

        normals = sorted((args.visa_root / category / "Data" / "Images" / "Normal").glob("*"))
        pick = rng.choice(len(normals), size=min(args.normals, len(normals)), replace=False)
        normals = [normals[int(i)] for i in sorted(pick)]
        views = []
        for path in normals:
            rgb = _load_rgb(path)
            feats = encoder.encode(rgb)
            valid = np.ones((CANVAS, CANVAS), bool)
            support = tokencut_object_mask(feats.tokens, valid)
            if not support.any():
                support = valid.copy()
            report = parse_structure(rgb, support, valid, feats.tokens, part_cfg, sam=generator)
            frame = valid if report.texture_mode else support
            try:
                prior = ObjectLocationPrior.from_masks({"all": few_masks}, {"all": [frame] * len(few_masks)})
            except ValueError:
                frame = valid
                prior = ObjectLocationPrior.from_masks({"all": few_masks}, {"all": [frame] * len(few_masks)})
            ratios = [float(m.sum()) / max(1, int(frame.sum())) for m in few_masks]
            structural = bool(np.median(ratios) > 0.15)
            omega, _score, stats, _domain = build_allowed_region(
                rgb, frame, valid, prior, "all", fewshot_area_ratios=ratios,
                sdf=physical_sdf_px_512(frame), cfg=omega_cfg, structural=structural,
                texture_mode=report.texture_mode, domain_dilate_px=64 if structural else 24,
                area_ratio_override=0.85 if structural else 0.7, parts=report.parts,
            )
            views.append({"rgb": rgb, "omega": omega, "path": str(path), "omega_ratio": stats["area_ratio"]})

        ad_files = sorted((args.ad_mask_root / category / "ko").glob("*.jpg"))
        if not ad_files:
            print(category, "no AD masks"); continue
        # AD masks contain empty / full-black samples (their own report: 37%);
        # both groups must draw from the same valid pool
        valid_ad = []
        for path in ad_files:
            mask = _load_mask(path)
            if mask.any() and mask.mean() < 0.9:
                valid_ad.append(path)
        ad_files = valid_ad
        if not ad_files:
            print(category, "no valid AD masks"); continue
        print(category, "valid AD masks", len(ad_files), flush=True)
        out_dir = args.output / category
        composed_dir = out_dir / "ko" / "composed"
        mask_dir = out_dir / "ko" / "mask_defect"
        composed_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        records = []
        resampled = 0
        for index in range(args.pairs):
            view = views[index % len(views)]
            pair_rng = np.random.RandomState(args.seed + index)
            ad_index = int(pair_rng.randint(0, len(ad_files)))
            defect_type = defect_types[index % len(defect_types)]
            attempt = 0
            while True:
                ad_mask = _load_mask(ad_files[ad_index])
                if args.group == "A":
                    placed = ad_mask
                    ok = True
                    break
                overlap, (top, left) = _best_placement(view["omega"], ad_mask)
                if overlap >= 0.95:
                    placed = _place(ad_mask, top, left)
                    ok = True
                    break
                ad_index = int(pair_rng.randint(0, len(ad_files)))
                attempt += 1
                resampled += 1
                if attempt >= 20:
                    placed = ad_mask
                    ok = False
                    break
            normal = view["rgb"]
            seed = args.seed + index
            raw = renderer(normal, defect_type, placed.astype(np.uint8), seed)
            final = compose_causal(normal, raw, placed)
            name = f"{index:04d}.png"
            Image.fromarray(final).save(composed_dir / name)
            Image.fromarray((placed.astype(np.uint8) * 255)).save(mask_dir / name)
            records.append({
                "index": index, "normal": view["path"], "ad_mask": str(ad_files[ad_index]),
                "defect_type": defect_type, "placed": ok, "resampled": attempt,
                "omega_ratio": view["omega_ratio"],
                "mask_area": float(placed.mean()),
            })
        (out_dir / "pairs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(category, "done pairs", len(records), "resampled draws", resampled,
              f"{time.time()-started:.0f}s", flush=True)
    print("written", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

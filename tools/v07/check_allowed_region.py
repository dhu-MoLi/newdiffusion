#!/usr/bin/env python3
"""Evaluate the allowed region (Omega) with part-graph conditioning."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.v06_a.structure import physical_sdf_px_512
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v07.allowed_region import OmegaConfig, build_allowed_region
from cadiff.v07.location import ObjectLocationPrior
from cadiff.v07.structure_parts import PartConfig, parse_structure

AREA_TARGETS = (0.3, 0.5, 0.7, 0.9)
PART_TOUCH_FRACTION = 0.1
STRUCTURAL_AREA = 0.15
STRUCTURAL_PARTS = 2
STRUCTURAL_OUTSIDE = 0.25


def _load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def _entries(manifest: Path) -> tuple[list[dict], list[dict]]:
    fewshot, test = [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("mask_relpath"):
            continue
        (fewshot if "fewshot_train" in row.get("roles", []) else test).append(row)
    return fewshot, test


def _recall(mask: np.ndarray, omega: np.ndarray) -> float:
    total = float(mask.sum())
    return float("nan") if total <= 0 else float((mask & omega).sum() / total)


def _parts_touched(mask: np.ndarray, parts: list) -> int:
    area = max(1, int(mask.sum()))
    return sum(1 for part in parts if float((mask & part.mask).sum() / area) >= PART_TOUCH_FRACTION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-per-type", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("/home/asus/.cache/sam/sam_vit_h_4b8939.pth"))
    parser.add_argument("--sam-type", default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--struct-dilate", type=int, default=64)
    arguments = parser.parse_args(argv)

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    project = load_config(arguments.paths)
    dataset_root = Path(project.dataset_root)
    encoder = load_official_dinov2_encoder(device=arguments.device)
    sam_model = sam_model_registry[arguments.sam_type](checkpoint=str(arguments.sam_checkpoint))
    sam_model.to(arguments.device)
    generator = SamAutomaticMaskGenerator(
        sam_model, points_per_side=32, pred_iou_thresh=0.85,
        stability_score_thresh=0.9, min_mask_region_area=100,
    )
    cfg = OmegaConfig()
    part_cfg = PartConfig(part_source="sam")
    output = arguments.output or (project.runs_root / "v07" / "omega" / "check_parts.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, dict] = {}

    def view(rel: str) -> dict:
        if rel in cache:
            return cache[rel]
        rgb = _load_rgb(dataset_root / rel)
        features = encoder.encode(rgb)
        valid = np.ones((512, 512), dtype=bool)
        support = tokencut_object_mask(features.tokens, valid)
        if not support.any():
            support = valid.copy()
        report = parse_structure(rgb, support, valid, features.tokens, part_cfg, sam=generator)
        cache[rel] = {
            "rgb": rgb,
            "support": support,
            "valid": valid,
            "texture_mode": report.texture_mode,
            "sdf": physical_sdf_px_512(support),
            "parts": [] if report.texture_mode else report.parts,
        }
        return cache[rel]

    report_out: dict[str, dict] = {}
    started = time.time()
    for category in arguments.categories:
        manifest = ROOT / "splits" / "mvtec" / category / "manifest.jsonl"
        fewshot, test = _entries(manifest)
        by_type_fs: dict[str, list[dict]] = defaultdict(list)
        by_type_test: dict[str, list[dict]] = defaultdict(list)
        for row in fewshot:
            by_type_fs[row["defect_type"]].append(row)
        for row in test:
            by_type_test[row["defect_type"]].append(row)

        rows = []
        for defect_type, items in sorted(by_type_fs.items()):
            views = [view(row["image_relpath"]) for row in items]
            masks = [_load_mask(dataset_root / row["mask_relpath"]) for row in items]
            frames = [item["valid"] if item["texture_mode"] else item["support"] for item in views]
            area_ratios = [float(masks[i].sum()) / max(1, int(frames[i].sum())) for i in range(len(masks))]
            touched = [_parts_touched(masks[i], views[i]["parts"]) for i in range(len(masks))]
            outside = [
                1.0 - float((masks[i] & frames[i]).sum()) / max(1, int(masks[i].sum()))
                for i in range(len(masks))
            ]
            structural = bool(
                np.median(area_ratios) > STRUCTURAL_AREA
                or np.median(touched) >= STRUCTURAL_PARTS
                or np.median(outside) > STRUCTURAL_OUTSIDE
            )

            def omega_for(item: dict, prior, ratios: list[float]) -> dict:
                frame_support = item["valid"] if item["texture_mode"] else item["support"]
                sdf = item["sdf"] if not item["texture_mode"] else physical_sdf_px_512(frame_support)
                out = {}
                for target in AREA_TARGETS:
                    omega, _score, stats, _domain = build_allowed_region(
                        item["rgb"],
                        frame_support,
                        item["valid"],
                        prior,
                        defect_type,
                        fewshot_area_ratios=ratios,
                        sdf=sdf,
                        cfg=cfg,
                        structural=structural,
                        texture_mode=item["texture_mode"],
                        domain_dilate_px=arguments.struct_dilate if structural else 24,
                        area_ratio_override=target,
                        parts=item["parts"],
                    )
                    out[target] = (omega, stats)
                return out

            def prior_inputs(keep: list[int]):
                supports = [frames[j] for j in keep]
                return {defect_type: [masks[j] for j in keep]}, {defect_type: supports}

            for index in range(len(items)):
                keep = [j for j in range(len(items)) if j != index]
                if not keep:
                    continue
                try:
                    prior = ObjectLocationPrior.from_masks(*prior_inputs(keep))
                except ValueError:
                    continue
                ratios = [area_ratios[j] for j in keep]
                for target, (omega, stats) in omega_for(views[index], prior, ratios).items():
                    rows.append({"split": "fewshot", "defect_type": defect_type, "target": target,
                                 "structural": structural, "recall": _recall(masks[index], omega),
                                 "components": stats["components"], "area_ratio": stats["area_ratio"]})

            try:
                prior = ObjectLocationPrior.from_masks(*prior_inputs(list(range(len(items)))))
            except ValueError:
                prior = None
            if prior is not None:
                for row in by_type_test.get(defect_type, [])[: arguments.test_per_type]:
                    item = view(row["image_relpath"])
                    gt = _load_mask(dataset_root / row["mask_relpath"])
                    if not gt.any():
                        continue
                    for target, (omega, stats) in omega_for(item, prior, area_ratios).items():
                        rows.append({"split": "test", "defect_type": defect_type, "target": target,
                                     "structural": structural, "recall": _recall(gt, omega),
                                     "components": stats["components"], "area_ratio": stats["area_ratio"]})

        entry: dict = {"rows": rows}
        for split in ("fewshot", "test"):
            for target in AREA_TARGETS:
                values = [row["recall"] for row in rows if row["split"] == split and row["target"] == target]
                entry[f"{split}_recall@{target}"] = float(np.nanmean(values)) if values else None
        structural_types = sorted({row["defect_type"] for row in rows if row["structural"]})
        entry["structural_types"] = structural_types
        entry["frac_le3"] = float(np.mean([row["components"] <= 3 for row in rows])) if rows else None
        report_out[category] = entry
        line = f"{category:12s} struct={len(structural_types)} LOMO " + " ".join(
            f"{entry[f'fewshot_recall@{t}']:.2f}" if entry[f"fewshot_recall@{t}"] is not None else "--"
            for t in AREA_TARGETS)
        line += " | test " + " ".join(
            f"{entry[f'test_recall@{t}']:.2f}" if entry[f"test_recall@{t}"] is not None else "--"
            for t in AREA_TARGETS)
        print(line)

    summary = {"schema": "v07_omega_check_v3", "area_targets": list(AREA_TARGETS),
               "categories": report_out, "seconds": round(time.time() - started, 1)}
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {output} ({summary['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

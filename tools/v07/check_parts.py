#!/usr/bin/env python3
"""Step-0 check: part decomposition and texture degradation over MVTec classes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_c.visa_pilot import load_visa_identity, load_visa_inventory_for_split, normal_identity_records
from cadiff.v07.structure_parts import PartConfig, parse_structure

DEFAULT_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
TEXTURE_CLASSES = {"carpet", "grid", "leather", "tile", "wood"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--part-source", default="spectral", choices=["spectral", "sam"])
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("/home/asus/.cache/sam/sam_vit_h_4b8939.pth"))
    parser.add_argument("--sam-type", default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    arguments = parser.parse_args(argv)

    project = load_config(arguments.paths)
    encoder = load_official_dinov2_encoder(device=arguments.device)
    output = arguments.output or (project.runs_root / "v07" / "structure_parts" / f"g0_{arguments.part_source}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = PartConfig(part_source=arguments.part_source)
    sam = None
    if arguments.part_source == "sam":
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        model = sam_model_registry[arguments.sam_type](checkpoint=str(arguments.sam_checkpoint))
        model.to(arguments.device)
        sam = SamAutomaticMaskGenerator(
            model, points_per_side=32, pred_iou_thresh=0.85,
            stability_score_thresh=0.9, min_mask_region_area=100,
        )

    report: dict[str, dict] = {}
    started = time.time()
    print(f"{'class':12s} {'k':>4s} {'texture':>8s} {'separ':>6s} {'textmode':>8s} {'supR':>5s}")
    for category in arguments.categories:
        inventory = load_visa_inventory_for_split(
            project.dataset_root, category, "mvtec", None
        )
        records = sorted(normal_identity_records(inventory), key=lambda item: item.item_id)
        if len(records) > arguments.samples:
            step = len(records) / float(arguments.samples)
            records = [records[int(i * step)] for i in range(arguments.samples)]
        rows = []
        for record in records:
            identity = load_visa_identity(record, dataset_root=inventory.source_root)
            features = encoder.encode(identity.rgb)
            support = tokencut_object_mask(features.tokens, identity.valid)
            if not support.any():
                support = np.ascontiguousarray(identity.valid)
            parsed = parse_structure(
                identity.rgb,
                support,
                identity.valid,
                features.tokens,
                cfg,
                category=category,
                image_id=record.item_id,
                sam=sam,
            )
            rows.append(parsed)
        ks = [row.k for row in rows]
        textures = [row.texture_score for row in rows]
        seps = [row.separability for row in rows]
        modes = [row.texture_mode for row in rows]
        report[category] = {
            "k": ks,
            "texture_score": textures,
            "separability": seps,
            "texture_mode_rate": float(np.mean(modes)),
            "support_ratio": [row.support_ratio for row in rows],
            "parts_area_ratio": [
                [part.area_ratio for part in row.parts] for row in rows
            ],
            "covered_ratio": [
                float(
                    np.logical_or.reduce([part.mask for part in row.parts]).sum()
                    / max(1, int(row.support.sum()))
                )
                if row.parts
                else 0.0
                for row in rows
            ],
            "expected_texture": category in TEXTURE_CLASSES,
        }
        covered = report[category]["covered_ratio"]
        print(
            f"{category:12s} k={np.mean(ks):4.1f}±{np.std(ks):3.1f} tex={np.mean(textures):7.4f} "
            f"sep={np.mean(seps):5.2f} texmode={np.mean(modes):4.2f} cover={np.mean(covered):4.2f}"
        )

    texture_hits = [
        report[c]["texture_mode_rate"] for c in TEXTURE_CLASSES if c in report
    ]
    object_hits = [
        report[c]["texture_mode_rate"] for c in report if c not in TEXTURE_CLASSES
    ]
    summary = {
        "schema": "v07_structure_parts_g0",
        "config": {
            "grid": cfg.grid,
            "max_parts": cfg.max_parts,
            "knn": cfg.knn,
            "laplacian_threshold": cfg.laplacian_threshold,
            "separability_threshold": cfg.separability_threshold,
        },
        "texture_classes_mean_rate": float(np.mean(texture_hits)) if texture_hits else 0.0,
        "object_classes_mean_rate": float(np.mean(object_hits)) if object_hits else 0.0,
        "seconds": round(time.time() - started, 1),
        "classes": report,
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nG0: texture-mode rate  texture classes {summary['texture_classes_mean_rate']:.2f}"
        f" | object classes {summary['object_classes_mean_rate']:.2f}  ({summary['seconds']}s)"
    )
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

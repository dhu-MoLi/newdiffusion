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
    arguments = parser.parse_args(argv)

    project = load_config(arguments.paths)
    encoder = load_official_dinov2_encoder(device=arguments.device)
    output = arguments.output or (project.runs_root / "v07" / "structure_parts" / "g0.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = PartConfig()

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
            "expected_texture": category in TEXTURE_CLASSES,
        }
        print(
            f"{category:12s} {np.mean(ks):4.1f} {np.mean(textures):8.4f} "
            f"{np.mean(seps):6.2f} {np.mean(modes):8.2f} {np.mean([r.support_ratio for r in rows]):5.2f}"
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

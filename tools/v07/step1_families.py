#!/usr/bin/env python3
"""Step 1: family classification and analytic inversion over MVTec types."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.v06_a.structure import physical_sdf_px_512, support_frame
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_c.visa_pilot import (
    fewshot_identity_records,
    load_visa_identity,
    load_visa_inventory_for_split,
)
from cadiff.v07.psmg import (
    extract_evidence,
    family_scores,
    invert_and_rasterize,
    predict_family,
)
from cadiff.v07.structure_parts import PartConfig, parse_structure

TEXTURE_CLASSES = {"carpet", "grid", "leather", "tile", "wood"}

MANUAL = {
    ("bottle", "broken_large"): "material_loss",
    ("bottle", "broken_small"): "geometric_cut",
    ("bottle", "contamination"): "surface_texture",
    ("cable", "bent_wire"): "deformation",
    ("cable", "cable_swap"): "displacement",
    ("cable", "cut_inner_insulation"): "geometric_cut",
    ("cable", "cut_outer_insulation"): "geometric_cut",
    ("cable", "missing_cable"): "material_loss",
    ("cable", "missing_wire"): "material_loss",
    ("cable", "poke_insulation"): "deformation",
    ("capsule", "crack"): "geometric_cut",
    ("capsule", "faulty_imprint"): "surface_texture",
    ("capsule", "poke"): "deformation",
    ("capsule", "squeeze"): "deformation",
    ("carpet", "color"): "surface_texture",
    ("carpet", "cut"): "geometric_cut",
    ("carpet", "hole"): "material_loss",
    ("carpet", "metal_contamination"): "surface_texture",
    ("carpet", "thread"): "geometric_cut",
    ("grid", "bent"): "deformation",
    ("grid", "broken"): "geometric_cut",
    ("grid", "glue"): "surface_texture",
    ("grid", "metal_contamination"): "surface_texture",
    ("grid", "thread"): "geometric_cut",
    ("hazelnut", "crack"): "geometric_cut",
    ("hazelnut", "cut"): "geometric_cut",
    ("hazelnut", "hole"): "material_loss",
    ("hazelnut", "print"): "surface_texture",
    ("leather", "color"): "surface_texture",
    ("leather", "cut"): "geometric_cut",
    ("leather", "fold"): "deformation",
    ("leather", "glue"): "surface_texture",
    ("leather", "poke"): "deformation",
    ("metal_nut", "bent"): "deformation",
    ("metal_nut", "color"): "surface_texture",
    ("metal_nut", "flip"): "displacement",
    ("metal_nut", "scratch"): "geometric_cut",
    ("pill", "color"): "surface_texture",
    ("pill", "contamination"): "surface_texture",
    ("pill", "crack"): "geometric_cut",
    ("pill", "faulty_imprint"): "surface_texture",
    ("pill", "pill_type"): "displacement",
    ("pill", "scratch"): "geometric_cut",
    ("screw", "manipulated_front"): "deformation",
    ("screw", "scratch_head"): "geometric_cut",
    ("screw", "scratch_neck"): "geometric_cut",
    ("screw", "thread_side"): "deformation",
    ("screw", "thread_top"): "deformation",
    ("tile", "cracked"): "geometric_cut",
    ("tile", "glue_strip"): "material_gain",
    ("tile", "gray_stroke"): "surface_texture",
    ("tile", "oil"): "surface_texture",
    ("tile", "rough"): "surface_texture",
    ("transistor", "bent_lead"): "deformation",
    ("transistor", "cut_lead"): "geometric_cut",
    ("transistor", "damaged_case"): "material_loss",
    ("transistor", "misplaced"): "displacement",
    ("wood", "hole"): "material_loss",
    ("wood", "liquid"): "surface_texture",
    ("wood", "scratch"): "geometric_cut",
    ("zipper", "broken_teeth"): "geometric_cut",
    ("zipper", "fabric_border"): "surface_texture",
    ("zipper", "fabric_interior"): "surface_texture",
    ("zipper", "rough"): "surface_texture",
    ("zipper", "split_teeth"): "geometric_cut",
    ("zipper", "squeezed_teeth"): "deformation",
}

DEFAULT_CATEGORIES = sorted({key[0] for key in MANUAL})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    project = load_config(arguments.paths)
    encoder = load_official_dinov2_encoder(device=arguments.device)
    output = arguments.output or (project.runs_root / "v07" / "psmg" / "step1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = PartConfig()

    per_type: dict[str, dict] = {}
    feature_names = [
        "app_delta", "app_cos", "lap_contrast",
        "area_ratio", "aspect", "bbox_solidity", "hull_ratio", "log_elongation",
        "n_components", "max_part_iou", "crosses_parts", "sdf_depth", "radial",
        "boundary_touch", "two_blob_gap", "part_pair_gap",
    ]
    started = time.time()
    for category in arguments.categories:
        inventory = load_visa_inventory_for_split(
            project.dataset_root, category, "mvtec", None
        )
        by_type: dict[str, list[dict]] = defaultdict(list)
        for record in fewshot_identity_records(inventory):
            identity = load_visa_identity(record, dataset_root=inventory.source_root)
            if identity.anomaly_gt is None or not identity.anomaly_gt.any():
                continue
            features = encoder.encode(identity.rgb)
            support = tokencut_object_mask(features.tokens, identity.valid)
            if not support.any():
                support = np.ascontiguousarray(identity.valid)
            report = parse_structure(
                identity.rgb,
                support,
                identity.valid,
                features.tokens,
                cfg,
                category=category,
                image_id=record.item_id,
            )
            sdf = physical_sdf_px_512(support)
            frame = support_frame(support)
            import cv2
            from scipy.ndimage import binary_dilation as _dilate

            grid = features.tokens.shape[0]
            mask_grid = cv2.resize(
                identity.anomaly_gt.astype(np.uint8), (grid, grid),
                interpolation=cv2.INTER_AREA,
            ) > 0.5
            support_grid = cv2.resize(
                support.astype(np.uint8), (grid, grid), interpolation=cv2.INTER_AREA
            ) > 0.5
            ring = _dilate(mask_grid, iterations=2) & ~mask_grid & support_grid
            if mask_grid.any() and ring.any():
                v_in = features.tokens[mask_grid].mean(axis=0)
                v_out = features.tokens[ring].mean(axis=0)
                app_delta = float(np.linalg.norm(v_in - v_out) / max(np.linalg.norm(v_out), 1e-6))
                app_cos = float(
                    np.dot(v_in, v_out) / max(np.linalg.norm(v_in) * np.linalg.norm(v_out), 1e-6)
                )
            else:
                app_delta, app_cos = 0.0, 1.0
            gray = cv2.cvtColor(identity.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            lap_contrast = float(
                lap[identity.anomaly_gt].mean() / max(lap[support].mean(), 1e-6)
            ) if support.any() else 1.0
            evidence = extract_evidence(
                identity.anomaly_gt, support, identity.valid, report.parts, sdf, frame
            )
            family, confidence = predict_family(evidence, report.texture_mode)
            manual = MANUAL.get((category, record.defect_type))
            rebuilt = invert_and_rasterize(family, identity.anomaly_gt, support, report.parts)
            inter = float((rebuilt & identity.anomaly_gt).sum())
            union = float((rebuilt | identity.anomaly_gt).sum())
            iou_manual = None
            if manual is not None:
                manual_mask = invert_and_rasterize(
                    manual, identity.anomaly_gt, support, report.parts
                )
                m_inter = float((manual_mask & identity.anomaly_gt).sum())
                m_union = float((manual_mask | identity.anomaly_gt).sum())
                iou_manual = m_inter / max(m_union, 1.0)
            by_type[record.defect_type].append(
                {
                    "family": family,
                    "manual": manual,
                    "confidence": confidence,
                    "texture_mode": report.texture_mode,
                    "k": report.k,
                    "iou": inter / max(union, 1.0),
                    "iou_manual": iou_manual,
                    "area_ratio": evidence.area_ratio,
                    "features": {
                        "app_delta": app_delta,
                        "app_cos": app_cos,
                        "lap_contrast": lap_contrast,
                        "area_ratio": evidence.area_ratio,
                        "aspect": evidence.aspect,
                        "bbox_solidity": evidence.bbox_solidity,
                        "hull_ratio": evidence.hull_ratio,
                        "log_elongation": float(np.log(max(evidence.elongation, 1e-3))),
                        "n_components": evidence.n_components,
                        "max_part_iou": evidence.max_part_iou,
                        "crosses_parts": evidence.crosses_parts,
                        "sdf_depth": evidence.sdf_depth,
                        "radial": evidence.radial,
                        "boundary_touch": evidence.boundary_touch,
                        "two_blob_gap": evidence.two_blob_gap,
                        "part_pair_gap": evidence.part_pair_gap,
                    },
                }
            )
        for defect_type, rows in by_type.items():
            votes = Counter(row["family"] for row in rows)
            predicted = votes.most_common(1)[0][0]
            median = {
                name: float(np.median([row["features"][name] for row in rows]))
                for name in feature_names
            }
            iou_manual = [row["iou_manual"] for row in rows if row["iou_manual"] is not None]
            key = f"{category}/{defect_type}"
            per_type[key] = {
                "predicted": predicted,
                "vote_share": votes.most_common(1)[0][1] / len(rows),
                "manual": MANUAL.get((category, defect_type)),
                "texture_mode_rate": float(np.mean([row["texture_mode"] for row in rows])),
                "mean_iou": float(np.mean([row["iou"] for row in rows])),
                "mean_iou_manual": float(np.mean(iou_manual)) if iou_manual else None,
                "features": median,
                "n": len(rows),
            }

    labelled = {k: v for k, v in per_type.items() if v["manual"]}
    keys = sorted(labelled)
    confusion: dict[str, Counter] = defaultdict(Counter)
    correct: list[bool] = []
    if keys:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x_all = np.array([[labelled[key]["features"][name] for name in feature_names] for key in keys])
        y_all = np.array([labelled[key]["manual"] for key in keys])
        x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0)
        for index, key in enumerate(keys):
            mask = np.ones(len(keys), dtype=bool)
            mask[index] = False
            if len(set(y_all[mask])) < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
            )
            model.fit(x_all[mask], y_all[mask])
            predicted = str(model.predict(x_all[index : index + 1])[0])
            truth = str(y_all[index])
            labelled[key]["loto_prediction"] = predicted
            confusion[truth][predicted] += 1
            correct.append(predicted == truth)

    ious_by_family: dict[str, list[float]] = defaultdict(list)
    ious_pred: dict[str, list[float]] = defaultdict(list)
    for key, value in per_type.items():
        family = value["manual"] or value["predicted"]
        if value.get("mean_iou_manual") is not None:
            ious_by_family[family].append(value["mean_iou_manual"])
        ious_pred[value["predicted"]].append(value["mean_iou"])

    summary = {
        "schema": "v07_psmg_step1_v2",
        "types": len(per_type),
        "labelled_types": len(labelled),
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "mean_iou_manual_family": float(
            np.mean([v["mean_iou_manual"] for v in per_type.values() if v["mean_iou_manual"] is not None])
        ) if any(v["mean_iou_manual"] is not None for v in per_type.values()) else 0.0,
        "mean_iou_predicted_family": float(np.mean([v["mean_iou"] for v in per_type.values()])),
        "iou_by_manual_family": {k: float(np.mean(v)) for k, v in ious_by_family.items()},
        "iou_by_predicted_family": {k: float(np.mean(v)) for k, v in ious_pred.items()},
        "feature_names": feature_names,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "seconds": round(time.time() - started, 1),
        "per_type": per_type,
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"types {summary['types']}  labelled {summary['labelled_types']}")
    print(f"LOTO family accuracy {summary['accuracy']:.3f}")
    print(f"reconstruction IoU (manual family) {summary['mean_iou_manual_family']:.3f}"
          f" | (predicted family) {summary['mean_iou_predicted_family']:.3f}")
    for family, value in sorted(summary["iou_by_manual_family"].items()):
        print(f"   IoU {family:18s} {value:.3f}")
    print("confusion (manual -> loto predicted):")
    for family, row in summary["confusion"].items():
        print(f"   {family:18s} {row}")
    print(f"written {output}  ({summary['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

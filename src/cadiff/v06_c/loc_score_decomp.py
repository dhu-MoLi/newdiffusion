"""Read-only DualAnoDiff loc score decomposition on a landed eval checkpoint.

Writes identities before any forward pass. Scores every DualAnoDiff held-out
image (image score = 21x21 avg-pool max). Overlay subset is first two held-out
per type plus first good. No training, no can, does not overwrite eval dumps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from cadiff.provenance import write_json_atomic
from cadiff.v06_c.loc_pred_gt_overlay import (
    N_GOOD,
    PER_TYPE,
    PRED_THR,
    PreviewIdentity,
    _load_predict_fn,
    _mask_path,
    _read_mask,
    _read_rgb,
    _row_image,
    _stack_rows,
    dualanodiff_heldout_names,
    overlay_pred_gt,
    select_preview_identities,
)

SCHEMA = "v06_c_loc_score_decomp_v1"
IMG_DIM = 256
POOL = 21
PRED_THR_VISUAL = PRED_THR
SIZE_EDGES = (64, 1024)
EVAL_NAME = "20260920-official-1000"
PredictFn = Callable[[NDArray[np.uint8], str], NDArray[np.float32]]


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def size_bin(area: int) -> str:
    if area < SIZE_EDGES[0]:
        return "small"
    if area < SIZE_EDGES[1]:
        return "medium"
    return "large"


def image_score_max_avgpool(heat: np.ndarray, kernel: int = POOL) -> float:
    import torch

    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("pool kernel must be a positive odd integer")
    tensor = torch.from_numpy(np.asarray(heat, dtype=np.float32))[None, None]
    pad = kernel // 2
    smoothed = torch.nn.functional.avg_pool2d(tensor, kernel, stride=1, padding=pad)
    return float(smoothed.max())


def list_heldout_identities(
    class_root: Path,
    *,
    anomaly_names: Sequence[str],
    arm: str | None = None,
) -> tuple[PreviewIdentity, ...]:
    test_root = class_root / "test"
    gt_root = class_root / "ground_truth"
    class_name = class_root.name
    arm_name = arm or class_name
    rows: list[PreviewIdentity] = []
    for anomaly_name in anomaly_names:
        image_dir = test_root / anomaly_name
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        heldout = dualanodiff_heldout_names(tuple(p.name for p in image_dir.iterdir()), anomaly_name)
        if not heldout:
            raise ValueError(f"{class_name}/{anomaly_name} has no DualAnoDiff held-out images")
        for name in heldout:
            stem = Path(name).stem
            rows.append(
                PreviewIdentity(
                    arm=arm_name,
                    class_name=class_name,
                    anomaly_name=anomaly_name,
                    stem=stem,
                    image_path=image_dir / name,
                    mask_path=_mask_path(gt_root, anomaly_name, stem),
                )
            )
    good_dir = test_root / "good"
    if not good_dir.is_dir():
        raise FileNotFoundError(good_dir)
    goods = dualanodiff_heldout_names(tuple(p.name for p in good_dir.iterdir()), "good")
    if not goods:
        raise ValueError(f"{class_name}/good has no images")
    for name in goods:
        rows.append(
            PreviewIdentity(
                arm=arm_name,
                class_name=class_name,
                anomaly_name="good",
                stem=Path(name).stem,
                image_path=good_dir / name,
                mask_path=None,
            )
        )
    return tuple(rows)


def official1000_score_arms(eval_root: Path, mvtec_path: Path) -> dict[str, dict[str, Any]]:
    root = eval_root
    visa = root / "visa"
    stamp = EVAL_NAME
    return {
        "cable": {
            "mvtec_path": mvtec_path,
            "class_name": "cable",
            "anomaly_names": ("bent_wire", "cut_inner_insulation", "cut_outer_insulation", "poke_insulation"),
            "ckpt": root / "cable/eval" / stamp / "cable/ckpt_localization/cable.pckl",
        },
        "candle": {
            "mvtec_path": visa / "candle/mvtec_layout",
            "class_name": "candle",
            "anomaly_names": (
                "chunk_of_wax_missing",
                "damaged_corner_of_packaging",
                "different_colour_spot",
                "extra_wax_in_candle",
            ),
            "ckpt": visa / "candle/eval" / stamp / "candle/ckpt_localization/candle.pckl",
        },
        "capsules": {
            "mvtec_path": visa / "capsules/mvtec_layout",
            "class_name": "capsules",
            "anomaly_names": ("bubble",),
            "ckpt": visa / "capsules/eval" / stamp / "capsules/ckpt_localization/capsules.pckl",
        },
        "cashew": {
            "mvtec_path": visa / "cashew/mvtec_layout",
            "class_name": "cashew",
            "anomaly_names": ("burnt", "corner_or_edge_breakage", "different_colour_spot", "middle_breakage"),
            "ckpt": visa / "cashew/eval" / stamp / "cashew/ckpt_localization/cashew.pckl",
        },
        "chewinggum": {
            "mvtec_path": visa / "chewinggum/mvtec_layout",
            "class_name": "chewinggum",
            "anomaly_names": ("chunk_of_gum_missing", "corner_missing", "scratches", "similar_colour_spot"),
            "ckpt": visa / "chewinggum/eval" / stamp / "chewinggum/ckpt_localization/chewinggum.pckl",
        },
    }


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "p50": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p50": float(np.median(array)),
        "max": float(array.max()),
    }


def _max_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.min() == labels.max():
        return 0.0
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = (2 * prec * rec) / np.maximum(prec + rec, 1e-8)
    finite = f1[np.isfinite(f1)]
    return float(finite.max()) if finite.size else 0.0


def score_row(gt: np.ndarray, heat: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(gt, dtype=bool)
    score = np.asarray(heat, dtype=np.float32)
    pred = score >= PRED_THR_VISUAL
    gt_area = int(mask.sum())
    pred_area = int(pred.sum())
    inter = int((mask & pred).sum())
    bg = score[~mask]
    bg_median = float(np.median(bg)) if bg.size else 0.0
    if gt_area:
        inside = score[mask]
        frac_gt_above_bg_median = float((inside > bg_median).mean())
        mean_in = float(inside.mean())
    else:
        frac_gt_above_bg_median = 0.0
        mean_in = 0.0
    mean_out = float(bg.mean()) if bg.size else 0.0
    return {
        "image_score": image_score_max_avgpool(score),
        "gt_area": gt_area,
        "pred_area_thr": pred_area,
        "overlap_thr": inter,
        "pred_over_gt": (pred_area / gt_area) if gt_area else 0.0,
        "mean_heat_in_gt": mean_in,
        "mean_heat_out_gt": mean_out,
        "frac_gt_above_bg_median": frac_gt_above_bg_median,
        "size_bin": size_bin(gt_area) if gt_area else "good",
    }


def summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    goods = [row for row in rows if not row["has_anomaly"]]
    bads = [row for row in rows if row["has_anomaly"]]
    good_scores = [float(row["image_score"]) for row in goods]
    bad_scores = [float(row["image_score"]) for row in bads]
    max_good = max(good_scores) if good_scores else 0.0
    min_bad = min(bad_scores) if bad_scores else 0.0
    labels = np.array([1 if row["has_anomaly"] else 0 for row in rows], dtype=np.int32)
    scores = np.array([float(row["image_score"]) for row in rows], dtype=np.float64)
    image_metrics: dict[str, float] = {}
    if labels.min() != labels.max():
        image_metrics = {
            "auroc_image": float(roc_auc_score(labels, scores)),
            "ap_image": float(average_precision_score(labels, scores)),
            "f1_image": _max_f1(labels, scores),
        }
    by_type: dict[str, Any] = {}
    for name in sorted({str(row["anomaly_name"]) for row in rows}):
        subset = [row for row in rows if row["anomaly_name"] == name]
        by_type[name] = {
            **_percentiles([float(row["image_score"]) for row in subset]),
            "mean_gt_area": float(np.mean([row["gt_area"] for row in subset])),
            "mean_pred_over_gt": float(np.mean([row["pred_over_gt"] for row in subset])),
            "mean_frac_gt_above_bg_median": float(
                np.mean([row["frac_gt_above_bg_median"] for row in subset])
            ),
            "n_below_max_good": int(
                sum(1 for row in subset if row["has_anomaly"] and row["image_score"] <= max_good)
            ),
        }
    bins: dict[str, Any] = {}
    for name in ("small", "medium", "large"):
        members = [row for row in bads if row["size_bin"] == name]
        pixel_scores: list[np.ndarray] = []
        pixel_gt: list[np.ndarray] = []
        for row in members:
            heat = np.asarray(row["_heat"], dtype=np.float32).reshape(-1)
            gt = np.asarray(row["_gt"], dtype=np.uint8).reshape(-1)
            pixel_scores.append(heat)
            pixel_gt.append(gt)
        auroc = None
        if pixel_gt:
            y = np.concatenate(pixel_gt)
            s = np.concatenate(pixel_scores)
            if y.min() != y.max():
                auroc = float(roc_auc_score(y, s))
        bins[name] = {
            "n_images": len(members),
            "mean_gt_area": float(np.mean([row["gt_area"] for row in members])) if members else 0.0,
            "mean_pred_over_gt": float(np.mean([row["pred_over_gt"] for row in members])) if members else 0.0,
            "mean_frac_gt_above_bg_median": float(
                np.mean([row["frac_gt_above_bg_median"] for row in members])
            )
            if members
            else 0.0,
            "auroc_pixel": auroc,
        }
    def _brief(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "anomaly_name": row["anomaly_name"],
            "stem": row["stem"],
            "image_score": float(row["image_score"]),
            "gt_area": int(row["gt_area"]),
            "pred_over_gt": float(row["pred_over_gt"]),
            "frac_gt_above_bg_median": float(row["frac_gt_above_bg_median"]),
        }

    return {
        "n_good": len(goods),
        "n_bad": len(bads),
        "image_score_good": _percentiles(good_scores),
        "image_score_bad": _percentiles(bad_scores),
        "n_bad_at_or_below_max_good": int(sum(1 for row in bads if row["image_score"] <= max_good)),
        "n_good_at_or_above_min_bad": int(sum(1 for row in goods if row["image_score"] >= min_bad)),
        "worst_fn": [_brief(row) for row in sorted(bads, key=lambda item: item["image_score"])[:8]],
        "worst_fp": [_brief(row) for row in sorted(goods, key=lambda item: -item["image_score"])[:8]],
        "coverage": {
            "mean_gt_area": float(np.mean([row["gt_area"] for row in bads])) if bads else 0.0,
            "mean_pred_area": float(np.mean([row["pred_area_thr"] for row in bads])) if bads else 0.0,
            "mean_pred_over_gt": float(np.mean([row["pred_over_gt"] for row in bads])) if bads else 0.0,
            "mean_frac_gt_above_bg_median": float(
                np.mean([row["frac_gt_above_bg_median"] for row in bads])
            )
            if bads
            else 0.0,
        },
        "by_type": by_type,
        "size_bins": bins,
        **image_metrics,
    }


def run_loc_score_decomp(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    output: Path,
    predict_fn: PredictFn | None = None,
    eval_kit: Path | None = None,
    device: str = "cuda:0",
    img_dim: int = IMG_DIM,
) -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    identity_payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "rule_overlay": "first DualAnoDiff held-out PER_TYPE per anomaly, then first N_GOOD good; frozen before predict",
        "rule_scoring": "all DualAnoDiff held-out (skip first 1/3 of each anomaly type; keep every good)",
        "per_type": PER_TYPE,
        "n_good_overlay": N_GOOD,
        "img_dim": img_dim,
        "pool": POOL,
        "pred_thr_visual": PRED_THR_VISUAL,
        "size_edges": SIZE_EDGES,
        "no_can": True,
        "arms": {},
    }
    selected_overlay: dict[str, tuple[PreviewIdentity, ...]] = {}
    selected_all: dict[str, tuple[PreviewIdentity, ...]] = {}
    for name, spec in arms.items():
        if name == "can":
            raise ValueError("this diagnostic excludes MVTec AD 2 can")
        class_root = Path(spec["mvtec_path"]) / str(spec["class_name"])
        overlay_rows = select_preview_identities(
            class_root,
            anomaly_names=tuple(spec["anomaly_names"]),
            arm=name,
        )
        all_rows = list_heldout_identities(
            class_root,
            anomaly_names=tuple(spec["anomaly_names"]),
            arm=name,
        )
        selected_overlay[name] = overlay_rows
        selected_all[name] = all_rows
        identity_payload["arms"][name] = {
            "overlay": [
                {"anomaly_name": row.anomaly_name, "stem": row.stem, "image_path": str(row.image_path)}
                for row in overlay_rows
            ],
            "scoring_n": len(all_rows),
            "scoring_by_type": {
                key: int(sum(1 for row in all_rows if row.anomaly_name == key))
                for key in ("good", *spec["anomaly_names"])
            },
        }
    write_json_atomic(destination / "identities.json", identity_payload)

    reports: dict[str, Any] = {}
    for name, spec in arms.items():
        if predict_fn is not None:
            predict = predict_fn
        else:
            if eval_kit is None:
                raise ValueError("eval_kit is required when predict_fn is omitted")
            predict = _load_predict_fn(Path(spec["ckpt"]), eval_kit, device)
        overlay_keys = {(row.anomaly_name, row.stem) for row in selected_overlay[name]}
        sample_root = destination / "samples" / name
        sample_root.mkdir(parents=True)
        sheet_rows: list[np.ndarray] = []
        scored: list[dict[str, Any]] = []
        jsonl_path = destination / f"{name}_rows.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in selected_all[name]:
                rgb = _read_rgb(row.image_path, img_dim)
                gt = _read_mask(row.mask_path, img_dim)
                identity = f"{row.arm}-{row.anomaly_name}-{row.stem}"
                heat = np.asarray(predict(rgb, identity), dtype=np.float32)
                if heat.shape != rgb.shape[:2]:
                    raise ValueError(f"predict_fn returned {heat.shape}, expected {rgb.shape[:2]}")
                stats = score_row(gt, heat)
                record = {
                    "arm": name,
                    "anomaly_name": row.anomaly_name,
                    "stem": row.stem,
                    "has_anomaly": row.anomaly_name != "good",
                    **{key: value for key, value in stats.items()},
                }
                handle.write(json.dumps(record) + "\n")
                record["_heat"] = heat
                record["_gt"] = gt.astype(np.uint8)
                scored.append(record)
                if (row.anomaly_name, row.stem) in overlay_keys:
                    panels = overlay_pred_gt(rgb, gt, heat)
                    strip = _row_image(panels, f"{name} {row.anomaly_name} {row.stem}")
                    Image.fromarray(strip).save(sample_root / f"{row.anomaly_name}-{row.stem}.png")
                    sheet_rows.append(strip)
        Image.fromarray(
            _stack_rows(sheet_rows, f"{name}  green=held-out GT  red=loc pred  yellow=overlap")
        ).save(destination / f"{name}_sheet.png")
        summary = summarize_arm(scored)
        summary["ckpt"] = str(spec.get("ckpt"))
        summary["n_scored"] = len(scored)
        reports[name] = summary
        print(name, "DONE", len(scored), flush=True)
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_train_r": True,
        "no_can": True,
        "eval_name": EVAL_NAME,
        "arm_names": list(reports),
        "arms": reports,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_loc_score_decomp_from_cli(
    *,
    eval_root: Path,
    mvtec_path: Path,
    eval_kit: Path,
    output: Path,
    device: str | None = None,
) -> dict[str, Any]:
    return run_loc_score_decomp(
        arms=official1000_score_arms(eval_root, mvtec_path),
        output=output,
        eval_kit=eval_kit,
        device=device or "cuda:0",
    )


__all__ = [
    "EVAL_NAME",
    "SIZE_EDGES",
    "image_score_max_avgpool",
    "list_heldout_identities",
    "official1000_score_arms",
    "run_loc_score_decomp",
    "run_loc_score_decomp_from_cli",
    "score_row",
    "size_bin",
    "summarize_arm",
]

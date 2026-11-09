"""SeaS DualAnoDiff pixel-AP split: extra-A types vs new singles vs combined.

Read-only on the landed 20260922-seas-500 loc checkpoint. Writes identities
before any forward pass. Does not train, does not overwrite dumps, excludes can.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from cadiff.provenance import write_json_atomic
from cadiff.v06_c.loc_pred_gt_overlay import (
    PreviewIdentity,
    _load_predict_fn,
    _read_mask,
    _read_rgb,
)
from cadiff.v06_c.loc_score_decomp import list_heldout_identities
from cadiff.v06_c.visa_pilot import COMBINED_TYPE, EXPECTED, slug_type

SCHEMA = "v06_c_seas_pixel_ap_buckets_v1"
EVAL_NAME = "20260922-seas-500"
LAYOUT_NAME = "mvtec_layout_seas_20260922"
IMG_DIM = 256
BUCKETS = ("extra_a", "new_single", "combined")
PredictFn = Callable[[NDArray[np.uint8], str], NDArray[np.float32]]

# Frozen extra-A slugs (official-500/1000). Not the SeaS full type tree.
EXTRA_A_SLUGS: dict[str, tuple[str, ...]] = {
    "candle": (
        "chunk_of_wax_missing",
        "damaged_corner_of_packaging",
        "different_colour_spot",
        "extra_wax_in_candle",
    ),
    "capsules": ("bubble",),
    "cashew": (
        "burnt",
        "corner_or_edge_breakage",
        "different_colour_spot",
        "middle_breakage",
    ),
    "chewinggum": (
        "chunk_of_gum_missing",
        "corner_missing",
        "scratches",
        "similar_colour_spot",
    ),
}


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def assign_bucket(category: str, anomaly_name: str) -> str:
    if category not in EXTRA_A_SLUGS:
        raise ValueError(f"category must stay {EXPECTED}, got {category}")
    if anomaly_name == "good":
        return "good"
    if anomaly_name == COMBINED_TYPE or anomaly_name == slug_type(COMBINED_TYPE):
        return "combined"
    if anomaly_name in EXTRA_A_SLUGS[category]:
        return "extra_a"
    return "new_single"


def _list_anomaly_dirs(class_root: Path) -> tuple[str, ...]:
    test_root = class_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)
    names = tuple(
        sorted(
            path.name
            for path in test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )
    )
    if not names:
        raise ValueError(f"{class_root.name} has no anomaly test folders")
    return names


def seas_pixel_ap_arms(eval_root: Path) -> dict[str, dict[str, Any]]:
    root = eval_root
    stamp = EVAL_NAME
    arms: dict[str, dict[str, Any]] = {}
    for category in EXPECTED:
        layout = root / "visa" / category / LAYOUT_NAME
        class_root = layout / category
        anomaly_names = _list_anomaly_dirs(class_root)
        extra = EXTRA_A_SLUGS[category]
        missing = tuple(name for name in extra if name not in anomaly_names)
        if missing:
            raise ValueError(f"{category} SeaS layout is missing extra-A types {missing}")
        if COMBINED_TYPE not in anomaly_names:
            raise ValueError(f"{category} SeaS layout is missing {COMBINED_TYPE}")
        ckpt = root / "visa" / category / "eval" / stamp / category / "ckpt_localization" / f"{category}.pckl"
        arms[category] = {
            "mvtec_path": layout,
            "class_name": category,
            "anomaly_names": anomaly_names,
            "ckpt": ckpt,
        }
    return arms


def pixel_pool_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_images": 0,
            "n_anomaly_images": 0,
            "n_pixels": 0,
            "n_pos_pixels": 0,
            "ap_pixel": None,
            "auroc_pixel": None,
        }
    scores = [np.asarray(row["_heat"], dtype=np.float32).reshape(-1) for row in rows]
    labels = [(np.asarray(row["_gt"], dtype=np.uint8).reshape(-1) > 0).astype(np.uint8) for row in rows]
    pixel_scores = np.concatenate(scores)
    pixel_gt = np.concatenate(labels)
    n_pos = int(pixel_gt.sum())
    metrics: dict[str, Any] = {
        "n_images": len(rows),
        "n_anomaly_images": int(sum(1 for row in rows if row["has_anomaly"])),
        "n_pixels": int(pixel_gt.size),
        "n_pos_pixels": n_pos,
        "ap_pixel": None,
        "auroc_pixel": None,
    }
    if pixel_gt.min() != pixel_gt.max():
        metrics["ap_pixel"] = float(average_precision_score(pixel_gt, pixel_scores))
        metrics["auroc_pixel"] = float(roc_auc_score(pixel_gt, pixel_scores))
    return metrics


def _with_share(metrics: dict[str, Any], full_pos: int) -> dict[str, Any]:
    payload = dict(metrics)
    payload["pos_pixel_share"] = (
        float(metrics["n_pos_pixels"] / full_pos) if full_pos else 0.0
    )
    return payload


def summarize_buckets(
    *,
    category: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    goods = [row for row in rows if not row["has_anomaly"]]
    bads = [row for row in rows if row["has_anomaly"]]
    full = pixel_pool_metrics(rows)
    full_pos = int(full["n_pos_pixels"])
    by_type: dict[str, Any] = {}
    for name in sorted({str(row["anomaly_name"]) for row in bads}):
        subset = [row for row in bads if row["anomaly_name"] == name]
        by_type[name] = {
            "bucket": assign_bucket(category, name),
            "anomalies_only": _with_share(pixel_pool_metrics(subset), full_pos),
            "with_goods": _with_share(pixel_pool_metrics([*subset, *goods]), full_pos),
        }
    buckets: dict[str, Any] = {}
    for bucket in BUCKETS:
        subset = [row for row in bads if row["bucket"] == bucket]
        buckets[bucket] = {
            "types": tuple(sorted({str(row["anomaly_name"]) for row in subset})),
            "anomalies_only": _with_share(pixel_pool_metrics(subset), full_pos),
            "with_goods": _with_share(pixel_pool_metrics([*subset, *goods]), full_pos),
        }
    without_combined = [row for row in rows if row["bucket"] != "combined"]
    without_new = [row for row in rows if row["bucket"] != "new_single"]
    return {
        "n_good": len(goods),
        "n_bad": len(bads),
        "full": _with_share(full, full_pos),
        "without_combined": _with_share(pixel_pool_metrics(without_combined), full_pos),
        "without_new_single": _with_share(pixel_pool_metrics(without_new), full_pos),
        "buckets": buckets,
        "by_type": by_type,
    }


def run_seas_pixel_ap_buckets(
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
        "rule_scoring": "all DualAnoDiff held-out (skip first 1/3 of each anomaly type; keep every good)",
        "img_dim": img_dim,
        "eval_name": EVAL_NAME,
        "buckets": BUCKETS,
        "extra_a_slugs": {key: list(value) for key, value in EXTRA_A_SLUGS.items()},
        "no_can": True,
        "not_train_r": True,
        "not_overwrite_dump": True,
        "arms": {},
    }
    selected: dict[str, tuple[PreviewIdentity, ...]] = {}
    for name, spec in arms.items():
        if name == "can":
            raise ValueError("this diagnostic excludes MVTec AD 2 can")
        if name not in EXTRA_A_SLUGS:
            raise ValueError(f"category must stay {EXPECTED}, got {name}")
        class_root = Path(spec["mvtec_path"]) / str(spec["class_name"])
        rows = list_heldout_identities(
            class_root,
            anomaly_names=tuple(spec["anomaly_names"]),
            arm=name,
        )
        selected[name] = rows
        identity_payload["arms"][name] = {
            "scoring_n": len(rows),
            "scoring_by_type": {
                key: int(sum(1 for row in rows if row.anomaly_name == key))
                for key in ("good", *spec["anomaly_names"])
            },
            "buckets": {
                bucket: [
                    slug
                    for slug in spec["anomaly_names"]
                    if assign_bucket(name, slug) == bucket
                ]
                for bucket in BUCKETS
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
            ckpt = Path(spec["ckpt"])
            if not ckpt.is_file():
                raise FileNotFoundError(ckpt)
            predict = _load_predict_fn(ckpt, eval_kit, device)
        scored: list[dict[str, Any]] = []
        jsonl_path = destination / f"{name}_rows.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in selected[name]:
                rgb = _read_rgb(row.image_path, img_dim)
                gt = _read_mask(row.mask_path, img_dim)
                identity = f"{row.arm}-{row.anomaly_name}-{row.stem}"
                heat = np.asarray(predict(rgb, identity), dtype=np.float32)
                if heat.shape != rgb.shape[:2]:
                    raise ValueError(f"predict_fn returned {heat.shape}, expected {rgb.shape[:2]}")
                record = {
                    "arm": name,
                    "anomaly_name": row.anomaly_name,
                    "stem": row.stem,
                    "has_anomaly": row.anomaly_name != "good",
                    "bucket": assign_bucket(name, row.anomaly_name),
                    "gt_area": int(np.asarray(gt, dtype=bool).sum()),
                }
                handle.write(json.dumps(record) + "\n")
                record["_heat"] = heat
                record["_gt"] = np.asarray(gt, dtype=np.uint8)
                scored.append(record)
        summary = summarize_buckets(category=name, rows=scored)
        summary["ckpt"] = str(spec.get("ckpt"))
        summary["n_scored"] = len(scored)
        reports[name] = summary
        print(name, "DONE", len(scored), flush=True)
    table = {
        category: {
            "full_ap_pixel": reports[category]["full"]["ap_pixel"],
            "without_combined_ap_pixel": reports[category]["without_combined"]["ap_pixel"],
            "without_new_single_ap_pixel": reports[category]["without_new_single"]["ap_pixel"],
            "buckets": {
                bucket: {
                    "types": list(reports[category]["buckets"][bucket]["types"]),
                    "ap_pixel_with_goods": reports[category]["buckets"][bucket]["with_goods"]["ap_pixel"],
                    "ap_pixel_anomalies_only": reports[category]["buckets"][bucket]["anomalies_only"]["ap_pixel"],
                    "n_anomaly_images": reports[category]["buckets"][bucket]["anomalies_only"]["n_anomaly_images"],
                    "n_pos_pixels": reports[category]["buckets"][bucket]["anomalies_only"]["n_pos_pixels"],
                    "pos_pixel_share": reports[category]["buckets"][bucket]["anomalies_only"]["pos_pixel_share"],
                }
                for bucket in BUCKETS
            },
            "by_type": {
                slug: {
                    "bucket": reports[category]["by_type"][slug]["bucket"],
                    "ap_pixel_with_goods": reports[category]["by_type"][slug]["with_goods"]["ap_pixel"],
                    "n_images": reports[category]["by_type"][slug]["anomalies_only"]["n_images"],
                    "n_pos_pixels": reports[category]["by_type"][slug]["anomalies_only"]["n_pos_pixels"],
                    "pos_pixel_share": reports[category]["by_type"][slug]["anomalies_only"]["pos_pixel_share"],
                }
                for slug in reports[category]["by_type"]
            },
        }
        for category in reports
    }
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_train_r": True,
        "no_can": True,
        "eval_name": EVAL_NAME,
        "img_dim": img_dim,
        "arm_names": list(reports),
        "arms": reports,
        "table": table,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    write_json_atomic(destination / "table.json", table)
    return payload


def run_seas_pixel_ap_buckets_from_cli(
    *,
    eval_root: Path,
    eval_kit: Path,
    output: Path,
    device: str | None = None,
) -> dict[str, Any]:
    return run_seas_pixel_ap_buckets(
        arms=seas_pixel_ap_arms(eval_root),
        output=output,
        eval_kit=eval_kit,
        device=device or "cuda:0",
    )


__all__ = [
    "BUCKETS",
    "EVAL_NAME",
    "EXTRA_A_SLUGS",
    "assign_bucket",
    "pixel_pool_metrics",
    "run_seas_pixel_ap_buckets",
    "run_seas_pixel_ap_buckets_from_cli",
    "seas_pixel_ap_arms",
    "summarize_buckets",
]

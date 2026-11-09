"""Few-shot mask area over the source TokenCut area, per single-label type.

The fit test used later is |g| / |S_src| > 0.40. Both areas are on the 512 canvas.
S_src is TokenCut of the neighbor-filled source, the same object generate() uses.
Multi-instance supports are not split. Does not train, generate, or change q.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source
from cadiff.v06_c.capsules_combined_panel import _contained_output
from cadiff.v06_c.visa_pilot import (
    MAGIC_CATEGORIES,
    SOURCE_ROOT,
    fewshot_identity_records,
    load_visa_identity,
    load_visa_magic_inventory,
)

SCHEMA = "v06_c_visa_area_ratio_v1"
TRIGGER = 0.40


def source_tokencut_area(encoder: Any, rgb: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> int:
    filled = neighbor_fill_source(rgb, gt, valid)
    features = encoder.encode(filled)
    support = tokencut_object_mask(features.tokens, valid)
    if not support.any():
        raise ValueError("source TokenCut is empty")
    return int(support.sum())


def run_visa_area_ratio(output: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    encoder = load_official_dinov2_encoder(device=device)
    rows: list[dict[str, Any]] = []
    for category in MAGIC_CATEGORIES:
        inventory = load_visa_magic_inventory(SOURCE_ROOT, category, generate_only=False)
        fewshot = fewshot_identity_records(inventory)
        for item in inventory.types:
            shots = [record for record in fewshot if record.defect_type == item.name]
            if not shots:
                rows.append(
                    {
                        "category": category,
                        "type": item.name,
                        "slug": item.slug,
                        "fewshot_n": 0,
                        "mask_mean": None,
                        "source_tokencut_mean": None,
                        "ratio_mean": None,
                        "ratio_min": None,
                        "ratio_max": None,
                        "n_above_0_40": 0,
                        "triggers_q_relax": False,
                    }
                )
                print(f"{category}/{item.slug} no few-shot", flush=True)
                continue
            ratios: list[float] = []
            masks: list[int] = []
            objects: list[int] = []
            for record in shots:
                loaded = load_visa_identity(record, dataset_root=SOURCE_ROOT)
                mask_area = int(np.asarray(loaded.anomaly_gt, dtype=bool).sum())
                object_area = source_tokencut_area(
                    encoder, loaded.rgb, loaded.anomaly_gt, loaded.valid
                )
                masks.append(mask_area)
                objects.append(object_area)
                ratios.append(mask_area / object_area)
            ratio_mean = float(np.mean(ratios))
            row = {
                "category": category,
                "type": item.name,
                "slug": item.slug,
                "fewshot_n": len(shots),
                "mask_mean": float(np.mean(masks)),
                "source_tokencut_mean": float(np.mean(objects)),
                "ratio_mean": ratio_mean,
                "ratio_min": float(np.min(ratios)),
                "ratio_max": float(np.max(ratios)),
                "n_above_0_40": int(sum(value > TRIGGER for value in ratios)),
                "triggers_q_relax": bool(ratio_mean > TRIGGER),
            }
            rows.append(row)
            print(
                f"{category}/{item.slug} ratio={ratio_mean:.3f} "
                f"mask={row['mask_mean']:.0f} object={row['source_tokencut_mean']:.0f} "
                f"above={row['n_above_0_40']}/{len(shots)}",
                flush=True,
            )
    triggered = [row for row in rows if row["triggers_q_relax"]]
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "trigger": "|g|/|S_src| mean > 0.40",
        "support": "tokencut of neighbor-filled few-shot source; multi-instance not split",
        "n_types": len(rows),
        "n_triggered": len(triggered),
        "triggered": [f"{row['category']}/{row['slug']}" for row in triggered],
        "rows": rows,
    }
    write_json_atomic(destination / "table.json", payload)
    lines = [
        "category\ttype\tfewshot_n\tmask_mean\tsource_tokencut_mean\tratio_mean\tratio_min\tratio_max\tn_above_0_40\ttriggers",
    ]
    for row in rows:
        lines.append(
            "\t".join(
                str(row[key])
                for key in (
                    "category",
                    "slug",
                    "fewshot_n",
                    "mask_mean",
                    "source_tokencut_mean",
                    "ratio_mean",
                    "ratio_min",
                    "ratio_max",
                    "n_above_0_40",
                    "triggers_q_relax",
                )
            )
        )
    (destination / "table.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("triggered", payload["triggered"], flush=True)
    return payload

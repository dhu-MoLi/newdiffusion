"""Diagnostic cashew stuck_together pairs: copy the reference TokenCut object.

Reads the landed SeaS-500 job list and replaces only that type's image and mask.
Other types stay in the SeaS dump. Not official H, not generate(), no R.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement
from cadiff.v06_c.instance_copy import duplicate_instance, product_from_background
from cadiff.v06_c.visa_pilot import (
    SOURCE_ROOT,
    load_visa_identity,
    load_visa_magic_inventory,
    load_visa_seas_inventory,
    normal_identity_records,
)

SEAS_DUMP = Path(
    "/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c/visa/cashew/generate/20260922-seas-500"
)


def _stuck_jobs(dump: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (dump / "synthetic_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type_slug") == "stuck_together" and not row.get("failed"):
            rows.append(row)
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 cashew stuck_together jobs, found {len(rows)}")
    return rows


def run_cashew_stuck_instance_copy(
    output: Path,
    *,
    seas_dump: Path = SEAS_DUMP,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    jobs = _stuck_jobs(seas_dump)
    inventory = load_visa_seas_inventory(SOURCE_ROOT, "cashew")
    normals = {record.item_id: record for record in normal_identity_records(inventory)}
    encoder = load_official_dinov2_encoder(device=device)
    loaded_cache: dict[str, Any] = {}
    support_cache: dict[str, np.ndarray] = {}
    kept: list[dict[str, Any]] = []
    failed = 0
    for index, job in enumerate(jobs, start=1):
        target_id = str(job["target_id"])
        if target_id not in loaded_cache:
            loaded_cache[target_id] = load_visa_identity(normals[target_id], dataset_root=SOURCE_ROOT)
        loaded = loaded_cache[target_id]
        if target_id not in support_cache:
            features = encoder.encode(np.ascontiguousarray(loaded.rgb))
            support_cache[target_id] = np.ascontiguousarray(
                tokencut_object_mask(features.tokens, loaded.valid) & np.asarray(loaded.valid, dtype=bool),
                dtype=bool,
            )
        record = {
            "schema_version": "v06_c_visa_instance_copy_bulk_v1",
            "role": "synthetic_train",
            "diagnostic_only": True,
            "not_official_h": True,
            "not_renderer": True,
            "method": "tokencut_instance_copy",
            "job_id": job["job_id"],
            "category": "cashew",
            "defect_type": "stuck together",
            "type_slug": "stuck_together",
            "target_id": target_id,
            "seed": int(job["seed"]),
            "index": int(job["index"]),
            "failed": False,
            "failure_reason": None,
            "image_relpath": f"images/{job['job_id']}.png",
            "mask_relpath": f"masks/{job['job_id']}.png",
        }
        try:
            copied = duplicate_instance(np.ascontiguousarray(loaded.rgb), support_cache[target_id], seed=int(job["seed"]))
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as exc:
            record.update(failed=True, failure_reason=str(exc))
            failed += 1
        else:
            Image.fromarray(copied.image).save(output / record["image_relpath"])
            Image.fromarray(copied.mask.astype(np.uint8) * 255).save(output / record["mask_relpath"])
            record.update(
                placed_pixels=int(copied.mask.sum()),
                extra_pixels=int(copied.copy_mask.sum()),
                shift_row=int(copied.shift_row),
                shift_col=int(copied.shift_col),
                support_pixels=int(support_cache[target_id].sum()),
            )
            kept.append(record)
        if index % 25 == 0 or index == len(jobs):
            print(f"{index}/{len(jobs)} kept={len(kept)} failed={failed}", flush=True)
    (output / "synthetic_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in kept),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "v06_c_visa_instance_copy_bulk_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_module_c": True,
        "not_train_r": True,
        "method": "tokencut_instance_copy",
        "category": "cashew",
        "type_slug": "stuck_together",
        "source_jobs": str(seas_dump),
        "requested": len(jobs),
        "kept_rows": len(kept),
        "failed_rows": failed,
        "destination": str(output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def _slug_jobs(dump: Path, slug: str) -> list[dict[str, Any]]:
    rows = []
    for line in (dump / "synthetic_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type_slug") == slug and not row.get("failed"):
            rows.append(row)
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 {slug} jobs, found {len(rows)}")
    return rows


def run_product_stuck_copy(
    output: Path,
    *,
    category: str,
    dump: Path,
    slug: str,
) -> dict[str, Any]:
    """Copy the color-split product for one stuck-together type. No TokenCut, no R."""

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    jobs = _slug_jobs(dump, slug)
    inventory = load_visa_magic_inventory(SOURCE_ROOT, category, generate_only=True)
    normals = {record.item_id: record for record in normal_identity_records(inventory)}
    loaded_cache: dict[str, Any] = {}
    object_cache: dict[str, np.ndarray] = {}
    kept: list[dict[str, Any]] = []
    failed = 0
    for index, job in enumerate(jobs, start=1):
        target_id = str(job["target_id"])
        if target_id not in loaded_cache:
            loaded_cache[target_id] = load_visa_identity(normals[target_id], dataset_root=SOURCE_ROOT)
        loaded = loaded_cache[target_id]
        if target_id not in object_cache:
            object_cache[target_id] = product_from_background(
                np.ascontiguousarray(loaded.rgb),
                np.asarray(loaded.valid, dtype=bool),
            )
        record = {
            "schema_version": "v06_c_visa_product_copy_bulk_v1",
            "role": "synthetic_train",
            "diagnostic_only": True,
            "not_official_h": True,
            "not_renderer": True,
            "method": "background_product_instance_copy",
            "job_id": job["job_id"],
            "category": category,
            "defect_type": job["defect_type"],
            "type_slug": slug,
            "target_id": target_id,
            "seed": int(job["seed"]),
            "index": int(job["index"]),
            "failed": False,
            "failure_reason": None,
            "image_relpath": f"images/{job['job_id']}.png",
            "mask_relpath": f"masks/{job['job_id']}.png",
        }
        try:
            copied = duplicate_instance(
                np.ascontiguousarray(loaded.rgb),
                object_cache[target_id],
                seed=int(job["seed"]),
                keep_inside=np.asarray(loaded.valid, dtype=bool),
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as exc:
            record.update(failed=True, failure_reason=str(exc))
            failed += 1
        else:
            Image.fromarray(copied.image).save(output / record["image_relpath"])
            Image.fromarray(copied.mask.astype(np.uint8) * 255).save(output / record["mask_relpath"])
            record.update(
                placed_pixels=int(copied.mask.sum()),
                extra_pixels=int(copied.copy_mask.sum()),
                shift_row=int(copied.shift_row),
                shift_col=int(copied.shift_col),
                support_pixels=int(object_cache[target_id].sum()),
            )
            kept.append(record)
        if index % 50 == 0 or index == len(jobs):
            print(f"{category} {index}/{len(jobs)} kept={len(kept)} failed={failed}", flush=True)
    (output / "synthetic_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in kept),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "v06_c_visa_product_copy_bulk_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_module_c": True,
        "not_train_r": True,
        "method": "background_product_instance_copy",
        "category": category,
        "type_slug": slug,
        "source_jobs": str(dump),
        "requested": len(jobs),
        "kept_rows": len(kept),
        "failed_rows": failed,
        "destination": str(output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run_cashew_stuck_instance_copy(
        Path(
            "/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c/visa/cashew/generate/20261005-instance-copy-stuck"
        )
    )

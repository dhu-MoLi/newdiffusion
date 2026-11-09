"""Diagnostic pcb1 missing: replace one detected round sensor with a real footprint.

The held-out label is the empty disk where an ultrasonic sensor should be.
Hough circles find the two sensors on a normal board. One is overwritten by a
few-shot missing footprint. Not official H, not wired into generate().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_c.visa_pilot import (
    SOURCE_ROOT,
    fewshot_identity_records,
    load_visa_identity,
    load_visa_magic_inventory,
)

Circle = tuple[int, int, int]


def sensor_circles(rgb: np.ndarray) -> tuple[Circle, ...]:
    """Two largest round sensors. Tuned on the 512 letterboxed pcb1 boards."""

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be HxWx3 uint8")
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    found = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=140,
        param2=45,
        minRadius=40,
        maxRadius=90,
    )
    if found is None:
        return ()
    ranked = sorted((tuple(map(int, row)) for row in found[0]), key=lambda item: item[2], reverse=True)
    return tuple(ranked[:2])


def paste_footprint(
    normal: np.ndarray,
    circle: Circle,
    footprint: np.ndarray,
    footprint_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the footprint onto the sensor disk. The mask is the pasted footprint."""

    image = np.array(np.asarray(normal), copy=True)
    source = np.asarray(footprint)
    mask = np.asarray(footprint_mask, dtype=bool)
    if image.ndim != 3 or source.ndim != 3 or image.shape[2] != 3 or source.shape[:2] != mask.shape:
        raise ValueError("footprint and mask must share the source grid")
    x_center, y_center, radius = (int(circle[0]), int(circle[1]), int(circle[2]))
    if radius < 2:
        raise ValueError("sensor radius is too small")
    side = radius * 2
    patch = np.asarray(Image.fromarray(source).resize((side, side), Image.Resampling.BILINEAR))
    alpha = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((side, side), Image.Resampling.NEAREST)
    ) > 127
    out_mask = np.zeros(image.shape[:2], dtype=bool)
    y0 = y_center - radius
    x0 = x_center - radius
    src_top = max(0, -y0)
    src_left = max(0, -x0)
    dst_top = max(0, y0)
    dst_left = max(0, x0)
    height = min(side - src_top, image.shape[0] - dst_top)
    width = min(side - src_left, image.shape[1] - dst_left)
    if height <= 0 or width <= 0:
        raise ValueError("sensor circle is outside the image")
    window = alpha[src_top : src_top + height, src_left : src_left + width]
    image[dst_top : dst_top + height, dst_left : dst_left + width][window] = patch[
        src_top : src_top + height, src_left : src_left + width
    ][window]
    out_mask[dst_top : dst_top + height, dst_left : dst_left + width][window] = True
    if not out_mask.any():
        raise ValueError("pasted footprint is empty")
    return np.ascontiguousarray(image), np.ascontiguousarray(out_mask)


def _missing_jobs(dump: Path) -> list[dict[str, Any]]:
    rows = []
    for line in (dump / "synthetic_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type_slug") == "missing" and not row.get("failed"):
            rows.append(row)
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 pcb1 missing jobs, found {len(rows)}")
    return rows


def _footprints() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """White circular missing pads. Dark sensor disks and thin marks are dropped."""

    from scipy.ndimage import label

    inventory = load_visa_magic_inventory(SOURCE_ROOT, "pcb1", generate_only=True)
    missing = next(item for item in inventory.types if item.slug == "missing")
    found: list[tuple[np.ndarray, np.ndarray]] = []
    for record in fewshot_identity_records(inventory):
        if record.defect_type != missing.name:
            continue
        loaded = load_visa_identity(record, dataset_root=SOURCE_ROOT)
        rgb = np.ascontiguousarray(loaded.rgb)
        gt = np.asarray(loaded.anomaly_gt, dtype=bool)
        if gt.shape != rgb.shape[:2] or not gt.any():
            raise RuntimeError(f"pcb1 missing footprint is empty: {record.item_id}")
        labels, count = label(gt)
        for index in range(1, count + 1):
            component = labels == index
            area = int(component.sum())
            if area < 5000:
                continue
            rows, cols = np.nonzero(component)
            height = int(rows.max() - rows.min() + 1)
            width = int(cols.max() - cols.min() + 1)
            circularity = area / (np.pi * ((height + width) / 4.0) ** 2)
            if min(height, width) / max(height, width) < 0.85 or circularity < 0.9:
                continue
            if float(rgb[component].mean()) < 220.0:
                continue
            top, bottom = int(rows.min()), int(rows.max()) + 1
            left, right = int(cols.min()), int(cols.max()) + 1
            found.append(
                (
                    np.ascontiguousarray(rgb[top:bottom, left:right]),
                    np.ascontiguousarray(component[top:bottom, left:right]),
                )
            )
    if len(found) < 1:
        raise RuntimeError("pcb1 missing white footprints are empty")
    return tuple(found)


def run_pcb1_sensor_missing(output: Path, *, dump: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    jobs = _missing_jobs(dump)
    footprints = _footprints()
    kept: list[dict[str, Any]] = []
    failed = 0
    for index, job in enumerate(jobs, start=1):
        normal_path = dump / "samples" / job["job_id"] / "normal.png"
        normal = np.asarray(Image.open(normal_path).convert("RGB"))
        record = {
            "schema_version": "v06_c_pcb1_sensor_missing_v1",
            "role": "synthetic_train",
            "diagnostic_only": True,
            "not_official_h": True,
            "not_renderer": True,
            "method": "hough_sensor_footprint_paste",
            "job_id": job["job_id"],
            "category": "pcb1",
            "defect_type": job["defect_type"],
            "type_slug": "missing",
            "target_id": job["target_id"],
            "seed": int(job["seed"]),
            "index": int(job["index"]),
            "failed": False,
            "failure_reason": None,
            "image_relpath": f"images/{job['job_id']}.png",
            "mask_relpath": f"masks/{job['job_id']}.png",
        }
        try:
            circles = sensor_circles(normal)
            if len(circles) < 2:
                raise RuntimeError(f"found {len(circles)} sensor circles")
            chosen = circles[int(job["seed"]) % 2]
            footprint, footprint_mask = footprints[int(job["seed"]) % len(footprints)]
            image, mask = paste_footprint(normal, chosen, footprint, footprint_mask)
        except (RuntimeError, ValueError) as exc:
            record.update(failed=True, failure_reason=str(exc))
            failed += 1
        else:
            Image.fromarray(image).save(output / record["image_relpath"])
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / record["mask_relpath"])
            record.update(placed_pixels=int(mask.sum()), circle=[int(v) for v in chosen])
            kept.append(record)
        if index % 50 == 0 or index == len(jobs):
            print(f"pcb1 missing {index}/{len(jobs)} kept={len(kept)} failed={failed}", flush=True)
    (output / "synthetic_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in kept),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "v06_c_pcb1_sensor_missing_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_module_c": True,
        "not_train_r": True,
        "method": "hough_sensor_footprint_paste",
        "category": "pcb1",
        "type_slug": "missing",
        "source_jobs": str(dump),
        "requested": len(jobs),
        "kept_rows": len(kept),
        "failed_rows": failed,
        "footprints": len(footprints),
        "destination": str(output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


__all__ = ["paste_footprint", "run_pcb1_sensor_missing", "sensor_circles"]

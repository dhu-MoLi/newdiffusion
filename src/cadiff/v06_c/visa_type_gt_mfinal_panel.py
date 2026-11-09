"""Held-out GT vs generated m_final for named VisA types.

Read-only on a landed dump. Identities are the first N DualAnoDiff held-out
tests of each type and the first N kept generate jobs, frozen before the sheet
is rendered. Unpaired images. No train, no generate, no can, does not overwrite dumps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_c.capsules_combined_panel import (
    DISPLAY_SIZE,
    _contained_output,
    _find_image,
    render_row,
)
from cadiff.v06_c.loc_pred_gt_overlay import _mask_path, dualanodiff_heldout_names
from cadiff.v06_c.oneshot_appearance_panel import _stack_rows, list_kept_jobs

SCHEMA = "v06_c_visa_type_gt_mfinal_panel_v1"
EVAL_NAME = "20260922-seas-500"
LAYOUT_NAME = "mvtec_layout_seas_20260922"
N_ROWS_PER_TYPE = 4


def visa_type_roots(
    eval_root: Path,
    *,
    category: str,
    eval_name: str = EVAL_NAME,
    layout_name: str = LAYOUT_NAME,
) -> dict[str, Path]:
    if category == "can":
        raise ValueError("this diagnostic excludes MVTec AD 2 can")
    root = Path(eval_root)
    return {
        "layout": root / "visa" / category / layout_name / category,
        "dump": root / "visa" / category / "generate" / eval_name,
    }


def select_visa_type_identities(
    eval_root: Path,
    *,
    category: str,
    type_slug: str,
    n_rows: int = N_ROWS_PER_TYPE,
    eval_name: str = EVAL_NAME,
    layout_name: str = LAYOUT_NAME,
) -> tuple[dict[str, Any], ...]:
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if category == "can":
        raise ValueError("this diagnostic excludes MVTec AD 2 can")
    roots = visa_type_roots(
        eval_root,
        category=category,
        eval_name=eval_name,
        layout_name=layout_name,
    )
    class_root = roots["layout"]
    test_dir = class_root / "test" / type_slug
    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)
    heldout = dualanodiff_heldout_names(tuple(path.name for path in test_dir.iterdir()), type_slug)
    if len(heldout) < n_rows:
        raise ValueError(f"{category}/{type_slug} has fewer than {n_rows} DualAnoDiff held-out images")
    jobs = list_kept_jobs(roots["dump"], type_slug)
    if any(str(job.get("category") or category) == "can" for job in jobs):
        raise ValueError("this diagnostic excludes MVTec AD 2 can")
    if len(jobs) < n_rows:
        raise ValueError(f"{category}/{type_slug} dump has fewer than {n_rows} kept jobs")
    rows: list[dict[str, Any]] = []
    gt_root = class_root / "ground_truth"
    for test_name, job in zip(heldout[:n_rows], jobs[:n_rows], strict=True):
        stem = Path(test_name).stem
        source_id = str(job["source_id"])
        rows.append(
            {
                "category": category,
                "type_slug": type_slug,
                "test_stem": stem,
                "test_image": str(_find_image(test_dir, stem)),
                "test_mask": str(_mask_path(gt_root, type_slug, stem)),
                "job_id": str(job["job_id"]),
                "index": int(job["index"]),
                "seed": int(job["seed"]),
                "target_id": str(job["target_id"]),
                "source_id": source_id,
                "source_image": str(_find_image(test_dir, source_id)),
                "source_mask": str(_mask_path(gt_root, type_slug, source_id)),
                "sample_path": str(job["sample_path"]),
                "placed_pixels_record": int(job.get("placed_pixels") or 0),
                "unpaired": True,
            }
        )
    return tuple(rows)


def run_visa_type_gt_mfinal_panel(
    *,
    eval_root: Path,
    output: Path,
    category: str,
    type_slugs: Sequence[str],
    n_rows_per_type: int = N_ROWS_PER_TYPE,
    img_dim: int = DISPLAY_SIZE,
    eval_name: str = EVAL_NAME,
    layout_name: str = LAYOUT_NAME,
) -> dict[str, Any]:
    if category == "can":
        raise ValueError("this diagnostic excludes MVTec AD 2 can")
    slugs = tuple(str(item) for item in type_slugs)
    if not slugs:
        raise ValueError("type_slugs must be non-empty")
    destination = _contained_output(output)
    identities: list[dict[str, Any]] = []
    for type_slug in slugs:
        identities.extend(
            select_visa_type_identities(
                eval_root,
                category=category,
                type_slug=type_slug,
                n_rows=n_rows_per_type,
                eval_name=eval_name,
                layout_name=layout_name,
            )
        )
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": "first N DualAnoDiff held-out tests per type (skip first 1/3) paired by sheet order with first N kept generate jobs by index; unpaired images; frozen before render",
            "category": category,
            "type_slugs": list(slugs),
            "eval_name": eval_name,
            "layout_name": layout_name,
            "n_rows_per_type": n_rows_per_type,
            "img_dim": img_dim,
            "no_can": True,
            "not_train_r": True,
            "not_overwrite_dump": True,
            "unpaired": True,
            "columns": (
                "test RGB+GT",
                "few-shot source RGB+GT (copied)",
                "proposal site + m_final",
                "final_y",
                "zoom m_final",
                "zoom test GT",
            ),
            "rows": [{key: value for key, value in row.items()} for row in identities],
        },
    )
    samples = destination / "rows"
    samples.mkdir()
    strips: list[Any] = []
    stats: list[dict[str, Any]] = []
    for row in identities:
        rendered = render_row(row, img_dim=img_dim)
        strip = rendered.pop("strip")
        Image.fromarray(strip).save(samples / f"{row['type_slug']}-{row['test_stem']}-{row['job_id']}.png")
        strips.append(strip)
        stats.append(
            {
                "type_slug": row["type_slug"],
                "test_stem": row["test_stem"],
                "job_id": row["job_id"],
                "source_id": row["source_id"],
                **rendered,
            }
        )
    Image.fromarray(
        _stack_rows(
            strips,
            f"{category}  unpaired  {'+'.join(slugs)}  testGT-green | copied-source-cyan | site+m_final-magenta | y | zoom-m | zoom-GT",
        )
    ).save(destination / "sheet.png")
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_train_r": True,
        "no_can": True,
        "unpaired": True,
        "n": len(identities),
        "category": category,
        "type_slugs": list(slugs),
        "eval_name": eval_name,
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_visa_type_gt_mfinal_panel_from_cli(
    *,
    eval_root: Path,
    output: Path,
    category: str,
    type_slugs: Sequence[str],
    n_rows_per_type: int = N_ROWS_PER_TYPE,
    eval_name: str = EVAL_NAME,
    layout_name: str = LAYOUT_NAME,
) -> dict[str, Any]:
    return run_visa_type_gt_mfinal_panel(
        eval_root=eval_root,
        output=output,
        category=category,
        type_slugs=type_slugs,
        n_rows_per_type=n_rows_per_type,
        eval_name=eval_name,
        layout_name=layout_name,
    )


__all__ = [
    "EVAL_NAME",
    "LAYOUT_NAME",
    "N_ROWS_PER_TYPE",
    "run_visa_type_gt_mfinal_panel",
    "run_visa_type_gt_mfinal_panel_from_cli",
    "select_visa_type_identities",
]

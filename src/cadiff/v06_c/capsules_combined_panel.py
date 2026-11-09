"""Capsules combined: held-out GT vs one rigid generated m_final.

Read-only on the landed SeaS-500 dump. Identities are the first N DualAnoDiff
held-out combined tests and the first N kept generate jobs, frozen before the
sheet is rendered. No train, no generate, no can, does not overwrite dumps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

from cadiff.provenance import write_json_atomic
from cadiff.v06_c.loc_pred_gt_overlay import (
    _mask_path,
    _read_mask,
    _read_rgb,
    dualanodiff_heldout_names,
)
from cadiff.v06_c.oneshot_appearance_panel import (
    _blend,
    _resize_mask,
    _resize_rgb,
    _stack_rows,
    crop_around_mask,
    list_kept_jobs,
)
from cadiff.v06_c.visa_pilot import COMBINED_TYPE, slug_type

SCHEMA = "v06_c_capsules_combined_panel_v1"
CATEGORY = "capsules"
TYPE_SLUG = slug_type(COMBINED_TYPE)
EVAL_NAME = "20260922-seas-500"
LAYOUT_NAME = "mvtec_layout_seas_20260922"
N_ROWS = 8
DISPLAY_SIZE = 256
GT_COLOR = (40, 220, 80)
SOURCE_COLOR = (40, 200, 220)
MASK_COLOR = (240, 40, 180)
_FONT = ImageFont.load_default()


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def _find_image(folder: Path, stem: str) -> Path:
    matches = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.stem == stem and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if len(matches) != 1:
        raise FileNotFoundError(folder / stem)
    return matches[0]


def n_connected(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return 0
    _, count = label(binary)
    return int(count)


def capsules_combined_roots(eval_root: Path) -> dict[str, Path]:
    root = Path(eval_root)
    return {
        "layout": root / "visa" / CATEGORY / LAYOUT_NAME / CATEGORY,
        "dump": root / "visa" / CATEGORY / "generate" / EVAL_NAME,
    }


def select_capsules_combined_identities(
    eval_root: Path,
    *,
    n_rows: int = N_ROWS,
) -> tuple[dict[str, Any], ...]:
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    roots = capsules_combined_roots(eval_root)
    class_root = roots["layout"]
    test_dir = class_root / "test" / TYPE_SLUG
    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)
    heldout = dualanodiff_heldout_names(tuple(path.name for path in test_dir.iterdir()), TYPE_SLUG)
    if len(heldout) < n_rows:
        raise ValueError(f"{CATEGORY}/{TYPE_SLUG} has fewer than {n_rows} DualAnoDiff held-out images")
    jobs = list_kept_jobs(roots["dump"], TYPE_SLUG)
    if any(str(job.get("category") or CATEGORY) == "can" for job in jobs):
        raise ValueError("this diagnostic excludes MVTec AD 2 can")
    if len(jobs) < n_rows:
        raise ValueError(f"{CATEGORY}/{TYPE_SLUG} dump has fewer than {n_rows} kept jobs")
    rows: list[dict[str, Any]] = []
    gt_root = class_root / "ground_truth"
    for test_name, job in zip(heldout[:n_rows], jobs[:n_rows], strict=True):
        stem = Path(test_name).stem
        source_id = str(job["source_id"])
        rows.append(
            {
                "category": CATEGORY,
                "type_slug": TYPE_SLUG,
                "test_stem": stem,
                "test_image": str(_find_image(test_dir, stem)),
                "test_mask": str(_mask_path(gt_root, TYPE_SLUG, stem)),
                "job_id": str(job["job_id"]),
                "index": int(job["index"]),
                "seed": int(job["seed"]),
                "target_id": str(job["target_id"]),
                "source_id": source_id,
                "source_image": str(_find_image(test_dir, source_id)),
                "source_mask": str(_mask_path(gt_root, TYPE_SLUG, source_id)),
                "sample_path": str(job["sample_path"]),
                "placed_pixels_record": int(job.get("placed_pixels") or 0),
                "unpaired": True,
            }
        )
    return tuple(rows)


def _row_strip(tiles: Sequence[np.ndarray], title: str) -> NDArray[np.uint8]:
    gap = 8
    header = 18
    height = header + tiles[0].shape[0]
    width = sum(tile.shape[1] for tile in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for tile in tiles:
        canvas[header : header + tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1] + gap
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((4, 2), title, fill=(20, 20, 20), font=_FONT)
    return np.asarray(image, dtype=np.uint8)


def _overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    size: int,
) -> NDArray[np.uint8]:
    image = _resize_rgb(rgb, size)
    return _blend(image, color, _resize_mask(mask, size), 0.70)


def _read_dump_rgb(sample: Path, name: str) -> NDArray[np.uint8]:
    with Image.open(sample / name) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_dump_mask(sample: Path, name: str) -> NDArray[np.bool_]:
    with Image.open(sample / name) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def render_row(identity: Mapping[str, Any], *, img_dim: int = DISPLAY_SIZE) -> dict[str, Any]:
    test_rgb = _read_rgb(Path(identity["test_image"]), img_dim)
    test_gt = _read_mask(Path(identity["test_mask"]), img_dim)
    source_rgb = _read_rgb(Path(identity["source_image"]), img_dim)
    source_gt = _read_mask(Path(identity["source_mask"]), img_dim)
    sample = Path(identity["sample_path"])
    normal = _read_dump_rgb(sample, "normal.png")
    proposal = _read_dump_mask(sample, "proposal.png")
    m_final = _read_dump_mask(sample, "m_final.png")
    final_y = _read_dump_rgb(sample, "final_y.png")
    site = _blend(_resize_rgb(normal, img_dim), SOURCE_COLOR, _resize_mask(proposal, img_dim), 0.35)
    site = _blend(site, MASK_COLOR, _resize_mask(m_final, img_dim), 0.80)
    tiles = (
        _overlay(test_rgb, test_gt, GT_COLOR, size=img_dim),
        _overlay(source_rgb, source_gt, SOURCE_COLOR, size=img_dim),
        site,
        _resize_rgb(final_y, img_dim),
        crop_around_mask(final_y, m_final, out_size=img_dim),
        crop_around_mask(test_rgb, test_gt, out_size=img_dim),
    )
    title = (
        f"test {identity['test_stem']}  src {identity['source_id']}  "
        f"gen {identity['job_id']}  placed {int(m_final.sum())}"
    )
    strip = _row_strip(tiles, title)
    return {
        "strip": strip,
        "test_gt_area": int(np.asarray(test_gt, dtype=bool).sum()),
        "test_cc": n_connected(test_gt),
        "source_gt_area": int(np.asarray(source_gt, dtype=bool).sum()),
        "source_cc": n_connected(source_gt),
        "m_final_area": int(np.asarray(m_final, dtype=bool).sum()),
        "m_final_cc": n_connected(m_final),
        "proposal_area": int(np.asarray(proposal, dtype=bool).sum()),
    }


def run_capsules_combined_panel(
    *,
    eval_root: Path,
    output: Path,
    n_rows: int = N_ROWS,
    img_dim: int = DISPLAY_SIZE,
) -> dict[str, Any]:
    destination = _contained_output(output)
    identities = select_capsules_combined_identities(eval_root, n_rows=n_rows)
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": "first N DualAnoDiff held-out combined tests (skip first 1/3) paired by sheet order with first N kept generate jobs by index; unpaired images; frozen before render",
            "category": CATEGORY,
            "type_slug": TYPE_SLUG,
            "eval_name": EVAL_NAME,
            "n_rows": n_rows,
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
            "rows": [
                {key: value for key, value in row.items()}
                for row in identities
            ],
        },
    )
    samples = destination / "rows"
    samples.mkdir()
    strips: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    for row in identities:
        rendered = render_row(row, img_dim=img_dim)
        strip = rendered.pop("strip")
        Image.fromarray(strip).save(samples / f"{row['test_stem']}-{row['job_id']}.png")
        strips.append(strip)
        stats.append({"test_stem": row["test_stem"], "job_id": row["job_id"], "source_id": row["source_id"], **rendered})
    Image.fromarray(
        _stack_rows(
            strips,
            "capsules combined  unpaired  testGT-green | copied-source-cyan | site+m_final-magenta | y | zoom-m | zoom-GT",
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
        "eval_name": EVAL_NAME,
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_capsules_combined_panel_from_cli(*, eval_root: Path, output: Path) -> dict[str, Any]:
    return run_capsules_combined_panel(eval_root=eval_root, output=output)


__all__ = [
    "CATEGORY",
    "N_ROWS",
    "TYPE_SLUG",
    "run_capsules_combined_panel",
    "run_capsules_combined_panel_from_cli",
    "select_capsules_combined_identities",
]

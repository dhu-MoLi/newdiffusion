"""Capsules combined: large-cluster A* one stamp vs two disjoint stamps in one site.

One variable: placement count. Both arms use the large-cluster A* pool and the
same frozen 8 jobs as `capsules_combined_gt_vs_mfinal_20260924`. A maps once;
baseline is one `instantiate_site_scale`; diagnostic tries a second copy of that
same scaled template in `site \\ first`. No R, no generate dump, no source
filter, no can. Not official H. User judges the sheet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_b.a2_component_place import (
    MIN_SHAPE_PIXELS,
    EmptyFeasiblePlacement,
    instantiate_site_scale,
    place_shape_max_overlap_clip,
)
from cadiff.v06_b.a2_shape_instantiate import place_shape_inside_proposal
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_c.capsules_combined_panel import (
    CATEGORY,
    EVAL_NAME,
    N_ROWS,
    TYPE_SLUG,
    _contained_output,
    _overlay,
    _row_strip,
    n_connected,
    select_capsules_combined_identities,
)
from cadiff.v06_c.capsules_combined_upper_area_panel import MapsFn, build_combined_a2_maps_fn
from cadiff.v06_c.loc_pred_gt_overlay import _read_mask, _read_rgb
from cadiff.v06_c.official_a2_place import large_cluster_fewshot_areas
from cadiff.v06_c.oneshot_appearance_panel import (
    _blend,
    _resize_mask,
    _resize_rgb,
    _stack_rows,
    crop_around_mask,
)
from cadiff.v06_c.visa_pilot import SOURCE_ROOT

SCHEMA = "v06_c_capsules_combined_two_stamp_v1"
DISPLAY_SIZE = 256
GT_COLOR = (40, 220, 80)
SOURCE_COLOR = (40, 200, 220)
SITE_COLOR = (80, 160, 255)
ONE_COLOR = (255, 220, 0)
STAMP1_COLOR = (40, 210, 230)
STAMP2_COLOR = (240, 40, 180)
SECOND_SEED_STRIDE = 1_000_003
COLUMNS = (
    "unpaired test RGB+GT",
    "few-shot source RGB+GT",
    "TokenCut site",
    "largeA 1-stamp",
    "stamp1 cyan + stamp2 magenta",
    "overlay yellow=1-stamp magenta=union",
    "zoom 1-stamp",
    "zoom 2-stamp union",
)


def _empty_stamp(shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "failed": True,
        "failure_reason": None,
        "mask": np.zeros(shape, dtype=bool),
        "site": np.zeros(shape, dtype=bool),
        "pixels": 0,
        "scale": None,
        "factor": None,
        "used_clip": False,
        "used_site_fallback": False,
        "template": np.zeros(shape, dtype=bool),
    }


def place_second_stamp(
    template: np.ndarray,
    site: np.ndarray,
    first_mask: np.ndarray,
    similarity: np.ndarray,
    proposal: np.ndarray,
    *,
    seed: int,
    min_shape_pixels: int,
) -> dict[str, Any]:
    """Place one more copy of the same template in site minus the first stamp."""

    allowed = np.asarray(site, dtype=bool)
    first = np.asarray(first_mask, dtype=bool)
    remaining = np.ascontiguousarray(allowed & ~first)
    info = {
        "failed": True,
        "failure_reason": None,
        "mask": np.zeros(first.shape, dtype=bool),
        "pixels": 0,
        "used_clip": False,
    }
    if int(remaining.sum()) < min_shape_pixels:
        info["failure_reason"] = "remaining site below the area floor"
        return info
    second_seed = int(seed) + SECOND_SEED_STRIDE
    contained_reason = "no contained second stamp"
    try:
        placed = place_shape_inside_proposal(
            template,
            remaining,
            similarity,
            seed=second_seed,
            weight_by_similarity=False,
        )
        mask = np.ascontiguousarray(placed.mask, dtype=bool)
        if int(mask.sum()) < min_shape_pixels:
            raise EmptyFeasiblePlacement("second stamp below the area floor")
        if (mask & ~allowed).any() or (mask & ~np.asarray(proposal, dtype=bool)).any():
            raise ValueError("second stamp escaped site")
        if (mask & first).any():
            raise ValueError("second stamp overlapped the first stamp")
        info.update(failed=False, mask=mask, pixels=int(mask.sum()), used_clip=False)
        return info
    except EmptyFeasiblePlacement as exc:
        contained_reason = str(exc)
    try:
        placed = place_shape_max_overlap_clip(
            template,
            remaining,
            seed=second_seed,
            min_pixels=min_shape_pixels,
        )
        mask = np.ascontiguousarray(placed.mask, dtype=bool)
        if (mask & ~allowed).any() or (mask & ~np.asarray(proposal, dtype=bool)).any():
            raise ValueError("second stamp escaped site")
        if (mask & first).any():
            raise ValueError("second stamp overlapped the first stamp")
        info.update(failed=False, mask=mask, pixels=int(mask.sum()), used_clip=True)
        return info
    except EmptyFeasiblePlacement as exc:
        info["failure_reason"] = f"{contained_reason}; clip: {exc}"
        return info


def _place_large_cluster(
    maps: Mapping[str, Any],
    *,
    min_shape_pixels: int,
) -> dict[str, Any]:
    source_gt = np.asarray(maps["source_gt"], dtype=bool)
    info = _empty_stamp(source_gt.shape)
    fewshot = large_cluster_fewshot_areas(tuple(int(value) for value in maps["fewshot_areas"]))
    info["area_min"] = int(min(fewshot))
    info["area_max"] = int(max(fewshot))
    info["n_areas"] = len(fewshot)
    try:
        template, placement = instantiate_site_scale(
            source_gt,
            maps["source_support"],
            maps["target_support"],
            maps["proposal"],
            maps["similarity"],
            seed=int(maps["seed"]),
            fewshot_areas=fewshot,
            min_shape_pixels=min_shape_pixels,
        )
        validate_realization(placement.mask, placement.site)
        if (placement.mask & ~np.asarray(maps["proposal"], dtype=bool)).any():
            raise ValueError("placed shape escaped a2")
        info.update(
            failed=False,
            mask=np.ascontiguousarray(placement.mask, dtype=bool),
            site=np.ascontiguousarray(placement.site, dtype=bool),
            pixels=int(placement.mask.sum()),
            scale=float(placement.scale),
            factor=float(placement.factor),
            used_clip=bool(placement.used_clip),
            used_site_fallback=bool(placement.used_site_fallback),
            template=np.ascontiguousarray(template, dtype=bool),
        )
    except EmptyFeasiblePlacement as exc:
        info["failure_reason"] = str(exc)
    return info


def _json_arm(info: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "failed": bool(info["failed"]),
        "failure_reason": info.get("failure_reason"),
        "pixels": int(info["pixels"]),
        "scale": info.get("scale"),
        "factor": info.get("factor"),
        "used_clip": bool(info.get("used_clip")),
        "used_site_fallback": bool(info.get("used_site_fallback")),
        "area_min": info.get("area_min"),
        "area_max": info.get("area_max"),
        "n_areas": info.get("n_areas"),
    }
    if extra:
        payload.update(extra)
    return payload


def render_row(
    identity: Mapping[str, Any],
    maps: Mapping[str, Any],
    one: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    img_dim: int,
) -> dict[str, Any]:
    test_rgb = _read_rgb(Path(identity["test_image"]), img_dim)
    test_gt = _read_mask(Path(identity["test_mask"]), img_dim)
    source_rgb = np.asarray(maps["source_rgb"], dtype=np.uint8)
    source_gt = np.asarray(maps["source_gt"], dtype=bool)
    normal = np.asarray(maps["normal_rgb"], dtype=np.uint8)
    site = np.asarray(one["site"], dtype=bool)
    if not site.any():
        site = np.asarray(maps["proposal"], dtype=bool)
    one_mask = np.asarray(one["mask"], dtype=bool)
    two_mask = np.asarray(second["mask"], dtype=bool)
    union = np.ascontiguousarray(one_mask | two_mask)
    n_stamps = int(not one["failed"]) + int(not second["failed"])
    site_tile = _blend(_resize_rgb(normal, img_dim), SITE_COLOR, _resize_mask(site, img_dim), 0.45)
    one_tile = _overlay(normal, one_mask, ONE_COLOR, size=img_dim)
    stamps = _blend(_resize_rgb(normal, img_dim), STAMP1_COLOR, _resize_mask(one_mask, img_dim), 0.70)
    stamps = _blend(stamps, STAMP2_COLOR, _resize_mask(two_mask, img_dim), 0.70)
    both = _blend(_resize_rgb(normal, img_dim), SITE_COLOR, _resize_mask(site, img_dim), 0.30)
    both = _blend(both, ONE_COLOR, _resize_mask(one_mask, img_dim), 0.70)
    both = _blend(both, STAMP2_COLOR, _resize_mask(union, img_dim), 0.70)
    tiles = (
        _overlay(test_rgb, test_gt, GT_COLOR, size=img_dim),
        _overlay(source_rgb, source_gt, SOURCE_COLOR, size=img_dim),
        site_tile,
        one_tile,
        stamps,
        both,
        crop_around_mask(
            one_tile,
            _resize_mask(one_mask if one_mask.any() else site, img_dim),
            out_size=img_dim,
        ),
        crop_around_mask(
            stamps,
            _resize_mask(union if union.any() else site, img_dim),
            out_size=img_dim,
        ),
    )
    one_px = "FAIL" if one["failed"] else str(one["pixels"])
    two_px = "FAIL" if second["failed"] else str(second["pixels"])
    title = (
        f"test {identity['test_stem']}  src {identity['source_id']}  "
        f"{identity['job_id']}  1s={one_px}  2s={two_px}  y={n_stamps}"
    )
    return {
        "strip": _row_strip(tiles, title),
        "source_gt_pixels": int(source_gt.sum()),
        "source_cc": n_connected(source_gt),
        "proposal_pixels": int(np.asarray(maps["proposal"], dtype=bool).sum()),
        "site_pixels": int(site.sum()),
        "fewshot_areas": [int(value) for value in maps["fewshot_areas"]],
        "large_cluster_areas": [int(value) for value in maps["large_cluster_areas"]],
        "n_stamps": n_stamps,
        "one_stamp": _json_arm(one),
        "second_stamp": _json_arm(second),
        "union_pixels": int(union.sum()),
    }


def run_capsules_combined_two_stamp_panel(
    *,
    identities: Sequence[Mapping[str, Any]],
    output: Path,
    maps_fn: MapsFn,
    min_shape_pixels: int = MIN_SHAPE_PIXELS,
    img_dim: int = DISPLAY_SIZE,
) -> dict[str, Any]:
    frozen = tuple(identities)
    if not frozen:
        raise ValueError("identities must be non-empty")
    if any(str(row.get("category") or CATEGORY) == "can" for row in frozen):
        raise ValueError("this diagnostic excludes MVTec AD 2 can")
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "diagnostic_only": True,
            "not_official_h": True,
            "not_train_r": True,
            "no_can": True,
            "not_overwrite_dump": True,
            "not_source_filter": True,
            "one_variable": (
                "same large-cluster A* template placed once vs twice; "
                "second stamp is disjoint in site minus first"
            ),
            "frozen_before_images": True,
            "eval_name": EVAL_NAME,
            "category": CATEGORY,
            "type_slug": TYPE_SLUG,
            "n_rows": len(frozen),
            "img_dim": img_dim,
            "min_shape_pixels": min_shape_pixels,
            "columns": COLUMNS,
            "rule": (
                "same first N kept capsules combined generate jobs as "
                "capsules_combined_gt_vs_mfinal; unpaired test stems for context only"
            ),
            "rows": [dict(row) for row in frozen],
        },
    )
    samples = destination / "rows"
    samples.mkdir()
    strips: list[NDArray[np.uint8]] = []
    stats: list[dict[str, Any]] = []
    for row in frozen:
        maps = maps_fn(row)
        cluster = large_cluster_fewshot_areas(tuple(int(value) for value in maps["fewshot_areas"]))
        maps = {**maps, "large_cluster_areas": cluster}
        one = _place_large_cluster(maps, min_shape_pixels=min_shape_pixels)
        if one["failed"]:
            second = _empty_stamp(np.asarray(maps["source_gt"]).shape)
            second["failure_reason"] = "first stamp failed"
        else:
            second = place_second_stamp(
                one["template"],
                one["site"],
                one["mask"],
                maps["similarity"],
                maps["proposal"],
                seed=int(maps["seed"]),
                min_shape_pixels=min_shape_pixels,
            )
        rendered = render_row(row, maps, one, second, img_dim=img_dim)
        strip = rendered.pop("strip")
        Image.fromarray(strip).save(samples / f"{row['test_stem']}-{row['job_id']}.png")
        strips.append(strip)
        stats.append(
            {
                "test_stem": row["test_stem"],
                "job_id": row["job_id"],
                "source_id": row["source_id"],
                "seed": int(row["seed"]),
                **rendered,
            }
        )
    Image.fromarray(
        _stack_rows(
            strips,
            "capsules combined  largeA 1-stamp vs 2 disjoint stamps  cyan=1 magenta=2  no R",
        )
    ).save(destination / "sheet.png")
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_official_h": True,
        "not_module_c": True,
        "not_train_r": True,
        "no_can": True,
        "not_overwrite_dump": True,
        "not_source_filter": True,
        "n": len(frozen),
        "n_two_stamps": sum(1 for row in stats if row["n_stamps"] == 2),
        "eval_name": EVAL_NAME,
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_capsules_combined_two_stamp_from_cli(
    *,
    eval_root: Path,
    output: Path,
    source_root: Path = SOURCE_ROOT,
    device: str | None = None,
    n_rows: int = N_ROWS,
) -> dict[str, Any]:
    identities = select_capsules_combined_identities(eval_root, n_rows=n_rows)
    return run_capsules_combined_two_stamp_panel(
        identities=identities,
        output=output,
        maps_fn=build_combined_a2_maps_fn(source_root=source_root, device=device),
    )


__all__ = [
    "COLUMNS",
    "SCHEMA",
    "place_second_stamp",
    "run_capsules_combined_two_stamp_from_cli",
    "run_capsules_combined_two_stamp_panel",
]

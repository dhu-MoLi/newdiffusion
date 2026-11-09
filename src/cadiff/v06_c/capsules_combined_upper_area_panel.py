"""Capsules combined: official H vs upper-half few-shot area scale.

One variable: A* sampling range for `combined` only. Same frozen identities as
`capsules_combined_gt_vs_mfinal_20260924` (first N kept generate jobs). A maps
once per job; two `instantiate_site_scale` arms. No R, no generate dump, no can.
Not official H. User judges the sheet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_atlas_diag.extra_a import ExtraAStructureConfig
from cadiff.v06_b.a2_component_place import (
    MIN_SHAPE_PIXELS,
    EmptyFeasiblePlacement,
    instantiate_site_scale,
)
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_c.capsules_combined_panel import (
    CATEGORY,
    EVAL_NAME,
    N_ROWS,
    TYPE_SLUG,
    _contained_output,
    _overlay,
    _row_strip,
    select_capsules_combined_identities,
)
from cadiff.v06_c.loc_pred_gt_overlay import _read_mask, _read_rgb
from cadiff.v06_c.oneshot_appearance_panel import (
    _blend,
    _resize_mask,
    _resize_rgb,
    _stack_rows,
    crop_around_mask,
)
from cadiff.v06_c.visa_pilot import (
    COMBINED_TYPE,
    SOURCE_ROOT,
    fewshot_identity_records,
    load_visa_identity,
    load_visa_seas_inventory,
    normal_identity_records,
)

SCHEMA = "v06_c_capsules_combined_upper_area_v1"
DISPLAY_SIZE = 256
GT_COLOR = (40, 220, 80)
SOURCE_COLOR = (40, 200, 220)
SITE_COLOR = (80, 160, 255)
OFFICIAL_COLOR = (255, 220, 0)
DIAG_COLOR = (240, 40, 180)
MapsFn = Callable[[Mapping[str, Any]], dict[str, Any]]

COLUMNS = (
    "unpaired test RGB+GT",
    "few-shot source RGB+GT",
    "TokenCut site",
    "official H m_final",
    "upper-half A* m_final",
    "overlay yellow=official magenta=upper",
    "zoom official",
    "zoom upper-half",
)


def upper_half_fewshot_areas(areas: Sequence[int]) -> tuple[int, ...]:
    ranked = tuple(sorted(int(area) for area in areas if int(area) > 0))
    if not ranked:
        raise ValueError("few-shot areas are empty")
    kept = ranked[len(ranked) // 2 :]
    if not kept:
        raise ValueError("upper-half few-shot areas are empty")
    return kept


def _empty_place(shape: tuple[int, int]) -> dict[str, Any]:
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
    }


def _place_arm(
    maps: Mapping[str, Any],
    fewshot_areas: tuple[int, ...],
    *,
    min_shape_pixels: int,
) -> dict[str, Any]:
    source_gt = np.asarray(maps["source_gt"], dtype=bool)
    info = _empty_place(source_gt.shape)
    info["area_min"] = int(min(fewshot_areas))
    info["area_max"] = int(max(fewshot_areas))
    info["n_areas"] = len(fewshot_areas)
    try:
        _template, placement = instantiate_site_scale(
            source_gt,
            maps["source_support"],
            maps["target_support"],
            maps["proposal"],
            maps["similarity"],
            seed=int(maps["seed"]),
            fewshot_areas=fewshot_areas,
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
        )
    except EmptyFeasiblePlacement as exc:
        info["failure_reason"] = str(exc)
    return info


def _json_arm(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
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


def render_row(
    identity: Mapping[str, Any],
    maps: Mapping[str, Any],
    official: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    img_dim: int,
) -> dict[str, Any]:
    test_rgb = _read_rgb(Path(identity["test_image"]), img_dim)
    test_gt = _read_mask(Path(identity["test_mask"]), img_dim)
    source_rgb = np.asarray(maps["source_rgb"], dtype=np.uint8)
    source_gt = np.asarray(maps["source_gt"], dtype=bool)
    normal = np.asarray(maps["normal_rgb"], dtype=np.uint8)
    site = np.asarray(official["site"] if not official["failed"] else diagnostic["site"], dtype=bool)
    if not site.any():
        site = np.asarray(maps["proposal"], dtype=bool)
    official_mask = np.asarray(official["mask"], dtype=bool)
    diag_mask = np.asarray(diagnostic["mask"], dtype=bool)
    site_tile = _blend(_resize_rgb(normal, img_dim), SITE_COLOR, _resize_mask(site, img_dim), 0.45)
    official_tile = _overlay(normal, official_mask, OFFICIAL_COLOR, size=img_dim)
    diag_tile = _overlay(normal, diag_mask, DIAG_COLOR, size=img_dim)
    both = _blend(_resize_rgb(normal, img_dim), SITE_COLOR, _resize_mask(site, img_dim), 0.30)
    both = _blend(both, OFFICIAL_COLOR, _resize_mask(official_mask, img_dim), 0.70)
    both = _blend(both, DIAG_COLOR, _resize_mask(diag_mask, img_dim), 0.70)
    union = official_mask | diag_mask
    tiles = (
        _overlay(test_rgb, test_gt, GT_COLOR, size=img_dim),
        _overlay(source_rgb, source_gt, SOURCE_COLOR, size=img_dim),
        site_tile,
        official_tile,
        diag_tile,
        both,
        crop_around_mask(
            official_tile,
            _resize_mask(official_mask if official_mask.any() else site, img_dim),
            out_size=img_dim,
        ),
        crop_around_mask(
            diag_tile,
            _resize_mask(diag_mask if diag_mask.any() else site, img_dim),
            out_size=img_dim,
        ),
    )
    off_px = "FAIL" if official["failed"] else str(official["pixels"])
    diag_px = "FAIL" if diagnostic["failed"] else str(diagnostic["pixels"])
    title = (
        f"test {identity['test_stem']}  src {identity['source_id']}  "
        f"{identity['job_id']}  off={off_px}  up={diag_px}"
    )
    return {
        "strip": _row_strip(tiles, title),
        "source_gt_pixels": int(source_gt.sum()),
        "proposal_pixels": int(np.asarray(maps["proposal"], dtype=bool).sum()),
        "site_pixels": int(site.sum()),
        "fewshot_areas": [int(value) for value in maps["fewshot_areas"]],
        "upper_half_areas": [int(value) for value in maps["upper_half_areas"]],
        "official": _json_arm(official),
        "upper_half": _json_arm(diagnostic),
    }


def run_capsules_combined_upper_area_panel(
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
            "one_variable": "combined few-shot A* sampled from upper half of that type's areas",
            "frozen_before_images": True,
            "eval_name": EVAL_NAME,
            "category": CATEGORY,
            "type_slug": TYPE_SLUG,
            "n_rows": len(frozen),
            "img_dim": img_dim,
            "min_shape_pixels": min_shape_pixels,
            "columns": COLUMNS,
            "rule": "same first N kept capsules combined generate jobs as capsules_combined_gt_vs_mfinal; unpaired test stems for context only",
            "rows": [dict(row) for row in frozen],
        },
    )
    samples = destination / "rows"
    samples.mkdir()
    strips: list[NDArray[np.uint8]] = []
    stats: list[dict[str, Any]] = []
    for row in frozen:
        maps = maps_fn(row)
        fewshot = tuple(int(value) for value in maps["fewshot_areas"])
        upper = upper_half_fewshot_areas(fewshot)
        maps = {**maps, "upper_half_areas": upper}
        official = _place_arm(maps, fewshot, min_shape_pixels=min_shape_pixels)
        diagnostic = _place_arm(maps, upper, min_shape_pixels=min_shape_pixels)
        rendered = render_row(row, maps, official, diagnostic, img_dim=img_dim)
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
            "capsules combined  official H vs upper-half A*  yellow=official  magenta=upper  no R",
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
        "n": len(frozen),
        "eval_name": EVAL_NAME,
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def build_combined_a2_maps_fn(
    *,
    source_root: Path,
    device: str | None = None,
) -> MapsFn:
    inventory = load_visa_seas_inventory(source_root, CATEGORY)
    encoder = load_official_dinov2_encoder(device=device)
    fewshot_records = tuple(
        record
        for record in fewshot_identity_records(inventory)
        if record.defect_type == COMBINED_TYPE
    )
    if not fewshot_records:
        raise ValueError("capsules combined few-shot sources are empty")
    config = ExtraAStructureConfig(category=CATEGORY, defect_types=(COMBINED_TYPE,))
    generator = A2ProposalGenerator(
        {
            COMBINED_TYPE: tuple(
                load_visa_identity(record, dataset_root=inventory.source_root)
                for record in fewshot_records
            )
        },
        config=config,
        identity_manifest=None,
        encoder=encoder,
    )
    normals_by_id = {record.item_id: record for record in normal_identity_records(inventory)}
    structures: dict[str, Any] = {}
    sources_by_id = {
        item.record.item_id: item for item in generator._sources[COMBINED_TYPE]
    }

    def maps_fn(row: Mapping[str, Any]) -> dict[str, Any]:
        target_id = str(row["target_id"])
        if target_id not in structures:
            loaded = load_visa_identity(
                normals_by_id[target_id],
                dataset_root=inventory.source_root,
            )
            structures[target_id] = estimate_normal_structure(loaded, config=inventory.structure)
        target = structures[target_id]
        seed = int(row["seed"])
        a2 = generator.generate(target, COMBINED_TYPE, seed)
        expected = str(row["source_id"])
        if str(a2.source_id) != expected:
            raise ValueError(
                f"frozen source drifted: job {row['job_id']} expected {expected} got {a2.source_id}"
            )
        if a2.fewshot_gt_areas is None:
            raise ValueError("official a2 maps require fewshot_gt_areas")
        source = sources_by_id[str(a2.source_id)]
        return {
            "source_rgb": np.ascontiguousarray(source.rgb),
            "source_gt": np.ascontiguousarray(a2.source_gt, dtype=bool),
            "source_support": np.ascontiguousarray(a2.source_support, dtype=bool),
            "target_support": np.ascontiguousarray(a2.target_support & target.valid, dtype=bool),
            "proposal": np.ascontiguousarray(a2.proposal, dtype=bool),
            "similarity": np.ascontiguousarray(a2.similarity, dtype=np.float32),
            "normal_rgb": np.ascontiguousarray(target.normal.rgb),
            "fewshot_areas": tuple(int(value) for value in a2.fewshot_gt_areas),
            "seed": seed,
        }

    return maps_fn


def run_capsules_combined_upper_area_from_cli(
    *,
    eval_root: Path,
    output: Path,
    source_root: Path = SOURCE_ROOT,
    device: str | None = None,
    n_rows: int = N_ROWS,
) -> dict[str, Any]:
    identities = select_capsules_combined_identities(eval_root, n_rows=n_rows)
    return run_capsules_combined_upper_area_panel(
        identities=identities,
        output=output,
        maps_fn=build_combined_a2_maps_fn(source_root=source_root, device=device),
    )


__all__ = [
    "COLUMNS",
    "SCHEMA",
    "build_combined_a2_maps_fn",
    "run_capsules_combined_upper_area_from_cli",
    "run_capsules_combined_upper_area_panel",
    "upper_half_fewshot_areas",
]

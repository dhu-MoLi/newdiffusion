"""Diagnostic H: 3-point DINO correspondence + similarity warp of few-shot GT.

Not official H. Does not change generate(). On correspondence failure, fall back
to current rigid largest-CC placement. DINO is not m_proposal / m_final.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy.ndimage import label

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator, _l2_rows, _upsample_map, load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import load_v06_a_identity_manifest
from cadiff.v06_a.structure import (
    _fill_binary_holes,
    estimate_normal_structure,
)
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_atlas_diag.cable_a import _lookup_p1_target, build_cable_a_slots
from cadiff.v06_atlas_diag.extra_a import ExtraAStructureConfig, load_extra_identity
from cadiff.v06_atlas_diag.features import downsample_mask
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source
from cadiff.v06_b.a2_component_place import (
    EmptyFeasiblePlacement,
    instantiate_on_component,
    label_a2_components,
    pick_largest_site,
)
from cadiff.v06_b.a2_component_place_extra import (
    EXPECTED_PREVIEW_ROWS,
    _identity_key,
    load_extra_preview_packs,
    preview_jobs,
)
from cadiff.v06_b.a2_shape_instantiate import (
    _contained_output,
    _heatmap,
    _mask_u8,
    _overlay,
    _save_sheet,
)

SCHEMA_VERSION = "v06_b_h_corr3_diag_v1"
LINE_RATIO = 0.35
MIN_AXIS_PX = 4.0
CLIP_KEEP = 0.5
_EIGHT = np.ones((3, 3), dtype=np.uint8)
_POINT_COLORS = ((0, 220, 220), (255, 220, 0), (255, 40, 200))
FLOW_COLUMNS = (
    "source_pts",
    "sim_m",
    "sim_c",
    "sim_p",
    "target_pts",
    "a2_prop",
    "site",
    "rigid",
    "corr",
    "overlay",
)
COLUMN_MEANINGS = {
    "source_pts": "源图 + GT，青/黄/品红 = 主轴负端 / 质心 / 正端",
    "sim_m": "负端 DINO 余弦（只在 TokenCut∩site 上着色）",
    "sim_c": "质心 DINO 余弦",
    "sim_p": "正端 DINO 余弦",
    "target_pts": "目标正常图 + 三个对应点",
    "a2_prop": "正式 a2 proposal",
    "site": "最大 8 连通 a2 部件",
    "rigid": "当前正式 H：刚性放置（或 FLOOR）",
    "corr": "三点相似变换后 ∩ site；失败则等于 rigid",
    "overlay": "蓝=proposal，绿=site，黄=rigid，品红=corr",
}


class CorrespondenceFail(ValueError):
    """3-point correspondence cannot justify a similarity warp."""


def mask_axis_keypoints(mask: np.ndarray) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    binary = np.asarray(mask, dtype=bool)
    pixels = np.argwhere(binary).astype(np.float64)
    if pixels.shape[0] < 3:
        raise CorrespondenceFail("mask too small for an axis")
    center = pixels.mean(axis=0)
    centered = pixels - center
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projection = centered @ axis
    minus = pixels[int(np.argmin(projection))]
    plus = pixels[int(np.argmax(projection))]
    if float(np.linalg.norm(plus - minus)) < MIN_AXIS_PX:
        raise CorrespondenceFail("mask axis is too short")
    return minus, center, plus


def _rc_to_xy(points_rc: np.ndarray) -> NDArray[np.float64]:
    stacked = np.asarray(points_rc, dtype=np.float64)
    return np.stack((stacked[:, 1], stacked[:, 0]), axis=1)


def _xy_to_rc(points_xy: np.ndarray) -> NDArray[np.float64]:
    stacked = np.asarray(points_xy, dtype=np.float64)
    return np.stack((stacked[:, 1], stacked[:, 0]), axis=1)


def similarity_from_keypoints(
    source_xy: np.ndarray, target_xy: np.ndarray
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    src = np.asarray(source_xy, dtype=np.float64)
    dst = np.asarray(target_xy, dtype=np.float64)
    if src.shape != (3, 2) or dst.shape != (3, 2):
        raise CorrespondenceFail("need three source and three target points")
    src_axis = src[2] - src[0]
    dst_axis = dst[2] - dst[0]
    src_len = float(np.linalg.norm(src_axis))
    dst_len = float(np.linalg.norm(dst_axis))
    if src_len < MIN_AXIS_PX or dst_len < MIN_AXIS_PX:
        raise CorrespondenceFail("matched axis is too short")
    scale = dst_len / src_len
    src_ang = float(np.arctan2(src_axis[1], src_axis[0]))
    dst_ang = float(np.arctan2(dst_axis[1], dst_axis[0]))
    theta = dst_ang - src_ang
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = dst[1] - scale * (rotation @ src[1])
    return rotation, float(scale), translation


def apply_similarity(
    mask: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
    *,
    size: int | None = None,
) -> NDArray[np.bool_]:
    binary = np.asarray(mask, dtype=bool)
    if size is None:
        size = int(binary.shape[0])
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1] or binary.shape[0] != size:
        raise ValueError("mask must be a square canvas")
    if scale <= 1e-6:
        raise CorrespondenceFail("similarity scale vanished")
    rows, cols = np.indices((size, size))
    dest = np.stack((cols.reshape(-1), rows.reshape(-1)), axis=1).astype(np.float64)
    source_xy = ((dest - np.asarray(translation, dtype=np.float64)) / float(scale)) @ np.asarray(
        rotation, dtype=np.float64
    )
    src_x = np.rint(source_xy[:, 0]).astype(np.int32)
    src_y = np.rint(source_xy[:, 1]).astype(np.int32)
    inside = (src_x >= 0) & (src_x < size) & (src_y >= 0) & (src_y < size)
    warped = np.zeros((size, size), dtype=bool)
    hits = inside.copy()
    hits[inside] = binary[src_y[inside], src_x[inside]]
    warped[rows.reshape(-1)[hits], cols.reshape(-1)[hits]] = True
    if not warped.any():
        raise CorrespondenceFail("warped GT vanished")
    return warped


def assert_axis_consistent(minus: np.ndarray, center: np.ndarray, plus: np.ndarray) -> None:
    a = np.asarray(minus, dtype=np.float64).reshape(2)
    b = np.asarray(plus, dtype=np.float64).reshape(2)
    c = np.asarray(center, dtype=np.float64).reshape(2)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < MIN_AXIS_PX:
        raise CorrespondenceFail("axis too short")
    distance = float(np.abs(axis[0] * (a[1] - c[1]) - axis[1] * (a[0] - c[0])) / length)
    if distance > LINE_RATIO * length:
        raise CorrespondenceFail("line constraint: center is off the matched axis")


def _pixel_to_token(row: float, col: float, grid_h: int, grid_w: int, *, size: int) -> tuple[int, int]:
    token_row = min(max(int(row * grid_h / size), 0), grid_h - 1)
    token_col = min(max(int(col * grid_w / size), 0), grid_w - 1)
    return token_row, token_col


def _token_center(token_row: int, token_col: int, grid_h: int, grid_w: int, *, size: int) -> NDArray[np.float64]:
    return np.array(
        (
            (token_row + 0.5) * size / grid_h,
            (token_col + 0.5) * size / grid_w,
        ),
        dtype=np.float64,
    )


def match_keypoints(
    source_rc: np.ndarray,
    source_feat: Any,
    target_feat: Any,
    search: np.ndarray,
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float32], ...]]:
    src = np.asarray(source_rc, dtype=np.float64)
    if src.shape != (3, 2):
        raise CorrespondenceFail("need three source points")
    size = int(np.asarray(search).shape[0])
    source_desc = _l2_rows(source_feat.tokens)
    target_desc = _l2_rows(target_feat.tokens)
    grid_h = int(target_feat.grid_height)
    grid_w = int(target_feat.grid_width)
    search_grid = downsample_mask(np.asarray(search, dtype=bool), height=grid_h, width=grid_w)
    if not search_grid.any():
        raise CorrespondenceFail("search domain has no encoder tokens")
    target_flat = target_desc.reshape(-1, target_desc.shape[-1])
    active = search_grid.reshape(-1)
    matched = np.zeros((3, 2), dtype=np.float64)
    maps: list[NDArray[np.float32]] = []
    for index, point in enumerate(src):
        token_row, token_col = _pixel_to_token(
            float(point[0]), float(point[1]), int(source_feat.grid_height), int(source_feat.grid_width), size=size
        )
        query = source_desc[token_row, token_col]
        cosine = target_flat @ query
        cosine = np.where(active, cosine, np.float32(-np.inf))
        if not np.isfinite(cosine).any():
            raise CorrespondenceFail("no finite match")
        peak = int(np.argmax(cosine))
        peak_row, peak_col = divmod(peak, grid_w)
        matched[index] = _token_center(peak_row, peak_col, grid_h, grid_w, size=size)
        grid_map = cosine.reshape(grid_h, grid_w)
        grid_map = np.where(search_grid, grid_map, np.float32(-1.0)).astype(np.float32)
        maps.append(_upsample_map(grid_map, size=size))
    return matched, tuple(maps)


def _cc_count(mask: np.ndarray) -> int:
    _labels, count = label(np.asarray(mask, dtype=bool), structure=_EIGHT)
    return int(count)


def _has_holes(mask: np.ndarray) -> bool:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return False
    filled = _fill_binary_holes(binary)
    return bool((filled & ~binary).any())


def warp_gt_with_correspondence(
    source_gt: np.ndarray,
    *,
    source_feat: Any,
    target_feat: Any,
    search: np.ndarray,
    site: np.ndarray,
    size: int | None = None,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    binary = np.asarray(source_gt, dtype=bool)
    if size is None:
        size = int(binary.shape[0])
    minus, center, plus = mask_axis_keypoints(binary)
    source_rc = np.stack((minus, center, plus), axis=0)
    matched_rc, maps = match_keypoints(source_rc, source_feat, target_feat, search)
    assert_axis_consistent(matched_rc[0], matched_rc[1], matched_rc[2])
    rotation, scale, translation = similarity_from_keypoints(_rc_to_xy(source_rc), _rc_to_xy(matched_rc))
    warped = apply_similarity(binary, rotation, scale, translation, size=size)
    allowed = np.asarray(site, dtype=bool)
    clipped = warped & allowed
    if not clipped.any():
        raise CorrespondenceFail("empty after site clip")
    if int(clipped.sum()) < CLIP_KEEP * int(warped.sum()):
        raise CorrespondenceFail("site clip removed too much")
    if _cc_count(clipped) != _cc_count(binary) or _has_holes(clipped) != _has_holes(binary):
        raise CorrespondenceFail("topology changed")
    return clipped, {
        "source_rc": source_rc,
        "target_rc": matched_rc,
        "sim_maps": maps,
        "rotation": rotation,
        "corr_scale": scale,
        "translation": translation,
        "warped_pixels": int(warped.sum()),
        "clipped_pixels": int(clipped.sum()),
    }


def place_corr3_or_rigid(
    source_gt: np.ndarray,
    source_support: np.ndarray,
    target_support: np.ndarray,
    proposal: np.ndarray,
    similarity: np.ndarray,
    *,
    source_feat: Any,
    target_feat: Any,
    search: np.ndarray,
    seed: int,
    size: int | None = None,
    min_shape_pixels: int = 64,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]:
    _template, placement = instantiate_on_component(
        source_gt,
        source_support,
        target_support,
        proposal,
        similarity,
        seed=seed,
        min_shape_pixels=min_shape_pixels,
    )
    rigid = np.ascontiguousarray(placement.mask, dtype=bool)
    info: dict[str, Any] = {
        "used_corr_fallback": False,
        "corr_fail_reason": None,
        "used_proposal_floor": bool(placement.used_proposal_floor),
        "used_site_fallback": bool(placement.used_site_fallback),
        "rigid_pixels": int(rigid.sum()),
        "site_pixels": int(placement.site.sum()),
        "scale": float(placement.scale),
        "angle_deg": int(placement.angle_deg),
    }
    try:
        clipped, viz = warp_gt_with_correspondence(
            source_gt,
            source_feat=source_feat,
            target_feat=target_feat,
            search=search,
            site=placement.site,
            size=size if size is not None else int(source_gt.shape[0]),
        )
        if (clipped & ~placement.site).any() or (clipped & ~np.asarray(proposal, dtype=bool)).any():
            raise CorrespondenceFail("correspondence escaped site")
        info.update(viz)
        info["used_corr_fallback"] = False
        return rigid, np.ascontiguousarray(clipped, dtype=bool), info
    except CorrespondenceFail as error:
        info["used_corr_fallback"] = True
        info["corr_fail_reason"] = str(error)
        return rigid, rigid.copy(), info


def _draw_points(rgb: np.ndarray, points_rc: np.ndarray | None) -> np.ndarray:
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).convert("RGB")
    if points_rc is None:
        return np.asarray(image)
    draw = ImageDraw.Draw(image)
    for point, color in zip(np.asarray(points_rc, dtype=np.float64), _POINT_COLORS, strict=True):
        row, col = float(point[0]), float(point[1])
        box = (col - 5, row - 5, col + 5, row + 5)
        draw.ellipse(box, outline=(255, 255, 255), width=2)
        draw.ellipse((col - 3, row - 3, col + 3, row + 3), fill=color)
    return np.asarray(image)


def run_h_corr3_diagnostic(
    *,
    repository_root: Path,
    dataset_root: Path,
    output: Path,
    a_experiment: Path,
    device: str | None = None,
    tile: int = 160,
) -> dict[str, Any]:
    destination = _contained_output(output)
    experiment = load_v06_a_experiment(a_experiment)
    manifest = load_v06_a_identity_manifest(
        repository_root / experiment.identity_manifest,
        experiment=experiment,
    )
    slots = build_cable_a_slots(manifest, experiment=experiment)
    destination.mkdir(parents=True, exist_ok=False)
    identities = [
        {
            "identity": f"{slot.defect_type}-{slot.normal_id}-{slot.seed}",
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": slot.source_id,
            "seed": slot.seed,
        }
        for slot in slots
    ]
    write_json_atomic(
        destination / "identities.json",
        {
            "frozen_before_images": True,
            "schedule": "cable_p1_32",
            "normals": ["000", "071", "152", "223"],
            "seeds": [2026082801, 2026082802],
            "rows": identities,
        },
    )
    generator = load_a2_generator(
        experiment_path=a_experiment,
        dataset_root=dataset_root,
        device=device,
    )
    sample_root = destination / "identities"
    sample_root.mkdir()
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    per_type: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {
        name: [] for name in experiment.defect_types
    }
    records: list[dict[str, Any]] = []
    structures: dict[str, Any] = {}

    for slot in slots:
        identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        row_dir = sample_root / identity
        row_dir.mkdir()
        if slot.normal_id not in structures:
            target_record = _lookup_p1_target(manifest, slot.normal_id)
            loaded_target = load_identity_record(
                target_record,
                dataset_root=dataset_root,
                config=experiment,
            )
            structures[slot.normal_id] = estimate_normal_structure(
                loaded_target, config=experiment
            )
        target = structures[slot.normal_id]
        source = generator.select_source(slot.defect_type, slot.seed)
        assert source.anomaly_gt is not None
        inpainted = neighbor_fill_source(source.rgb, source.anomaly_gt, source.valid)
        result = generator.generate(target, slot.defect_type, slot.seed)
        proposal = np.asarray(result.proposal, dtype=bool)
        similarity = np.asarray(result.similarity, dtype=np.float32)
        source_support = np.asarray(result.source_support, dtype=bool)
        target_support = np.asarray(result.target_support, dtype=bool)
        _labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
        preferred = pick_largest_site(sites) if sites else None
        site_mask = preferred.mask if preferred is not None else np.zeros_like(proposal)
        search = target_support & site_mask & target.valid
        source_feat = generator._encoder.encode(inpainted)
        target_feat = generator._encoder.encode(target.normal.rgb)
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        empty = np.zeros(proposal.shape, dtype=bool)
        arrays = {
            "source_pts": _overlay(source.rgb, source.anomaly_gt, (255, 40, 40), 0.45),
            "sim_m": _heatmap(np.zeros(proposal.shape, dtype=np.float32), search),
            "sim_c": _heatmap(np.zeros(proposal.shape, dtype=np.float32), search),
            "sim_p": _heatmap(np.zeros(proposal.shape, dtype=np.float32), search),
            "target_pts": normal.copy(),
            "a2_prop": _mask_u8(proposal),
            "site": _mask_u8(site_mask),
            "rigid": _mask_u8(empty),
            "corr": _mask_u8(empty),
            "overlay": _overlay(normal, proposal, (80, 160, 255), 0.28),
        }
        record: dict[str, Any] = {
            "identity": identity,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": source.record.item_id,
            "seed": slot.seed,
            "failed": False,
            "failure_reason": None,
            "used_corr_fallback": None,
            "corr_fail_reason": None,
            "proposal_pixels": int(proposal.sum()),
            "site_pixels": int(site_mask.sum()),
        }
        try:
            if result.target_support is None or result.source_support is None:
                raise ValueError("official a2 must return TokenCut supports")
            rigid, corr, info = place_corr3_or_rigid(
                source.anomaly_gt,
                source_support,
                target_support,
                proposal,
                similarity,
                source_feat=source_feat,
                target_feat=target_feat,
                search=search,
                seed=slot.seed,
            )
            record.update(
                used_corr_fallback=info["used_corr_fallback"],
                corr_fail_reason=info.get("corr_fail_reason"),
                used_proposal_floor=info.get("used_proposal_floor"),
                used_site_fallback=info.get("used_site_fallback"),
                rigid_pixels=int(rigid.sum()),
                corr_pixels=int(corr.sum()),
                corr_scale=info.get("corr_scale"),
                rigid_scale=info.get("scale"),
                m_final_subset_of_site=bool(not (corr & ~site_mask).any()) if site_mask.any() else False,
            )
            source_rc = info.get("source_rc")
            target_rc = info.get("target_rc")
            if source_rc is None:
                try:
                    minus, center, plus = mask_axis_keypoints(source.anomaly_gt)
                    source_rc = np.stack((minus, center, plus), axis=0)
                except CorrespondenceFail:
                    source_rc = None
            arrays["source_pts"] = _draw_points(arrays["source_pts"], source_rc)
            arrays["target_pts"] = _draw_points(normal, target_rc if not info["used_corr_fallback"] else None)
            maps = info.get("sim_maps")
            if maps is not None:
                arrays["sim_m"] = _heatmap(maps[0], search)
                arrays["sim_c"] = _heatmap(maps[1], search)
                arrays["sim_p"] = _heatmap(maps[2], search)
            arrays["site"] = _mask_u8(site_mask)
            arrays["rigid"] = _mask_u8(rigid)
            arrays["corr"] = _mask_u8(corr)
            arrays["overlay"] = _overlay(
                _overlay(
                    _overlay(
                        _overlay(normal, proposal, (80, 160, 255), 0.28),
                        site_mask,
                        (40, 200, 80),
                        0.35,
                    ),
                    rigid,
                    (255, 220, 0),
                    0.55,
                ),
                corr,
                (220, 40, 180) if not info["used_corr_fallback"] else (255, 140, 0),
                0.55,
            )
        except (EmptyFeasiblePlacement, CorrespondenceFail, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        for name, array in arrays.items():
            Image.fromarray(array).save(row_dir / f"{name}.png")
        write_json_atomic(row_dir / "record.json", record)
        records.append(record)
        if record["failed"]:
            label_text = identity + " FAIL"
        elif record.get("used_corr_fallback"):
            label_text = identity + " RIGID"
        else:
            label_text = identity + " CORR"
        flow = {name: arrays[name] for name in FLOW_COLUMNS}
        flow_rows.append((label_text, flow))
        per_type[slot.defect_type].append((label_text, flow))
        print(identity, label_text.split(" ", 1)[-1], flush=True)

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    for kind, rows in per_type.items():
        _save_sheet(destination / f"flow_{kind}.png", rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "not_official_h": True,
        "not_learned": True,
        "not_generate": True,
        "one_variable": "h_shape_rigid_vs_3point_dino_similarity",
        "identity_count": len(records),
        "failed_rows": sum(bool(row["failed"]) for row in records),
        "corr_rows": sum(not row["failed"] and not row.get("used_corr_fallback") for row in records),
        "rigid_fallback_rows": sum(bool(row.get("used_corr_fallback")) for row in records),
        "flow_columns": list(FLOW_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "line_ratio": LINE_RATIO,
        "search": "tokencut_object ∩ largest_a2_cc ∩ V",
        "encoder": "same frozen dinov2 as official a2",
        "destination": str(destination),
        "visible_defects_passed": None,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_h_corr3_extra_diagnostic(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cad3_root: Path,
    output: Path,
    device: str | None = None,
    tile: int = 160,
) -> dict[str, Any]:
    """Frozen extra/cad3 34-row preview: same 3-point DINO similarity vs rigid.

    Informal only. Does not write generate(), official H, Cable corr3, or extra place 20260911.
    """

    destination = _contained_output(Path(output))
    packs = load_extra_preview_packs(
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        cad3_root=cad3_root,
    )
    jobs = preview_jobs(packs)
    destination.mkdir(parents=True, exist_ok=False)
    identities = [
        {
            "identity": _identity_key(slot),
            "dataset": slot.dataset,
            "category": slot.category,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": slot.source_id,
            "seed": slot.seed,
        }
        for _pack, slot in jobs
    ]
    write_json_atomic(
        destination / "identities.json",
        {
            "frozen_before_images": True,
            "schedule": "extra_cad3_preview_34",
            "seed": 2026082801,
            "expected_rows": EXPECTED_PREVIEW_ROWS,
            "rows": identities,
        },
    )
    encoder = load_official_dinov2_encoder(device=device)
    generators: dict[str, A2ProposalGenerator] = {}
    structures: dict[tuple[str, str, str], Any] = {}
    sample_root = destination / "identities"
    sample_root.mkdir()
    records: list[dict[str, Any]] = []
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    per_category: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = defaultdict(list)

    for pack, slot in jobs:
        pack_key = f"{pack.dataset}/{pack.name}"
        config = ExtraAStructureConfig(category=pack.name, defect_types=pack.defect_types)
        if pack_key not in generators:
            sources = {
                defect_type: tuple(
                    load_extra_identity(record, dataset_root=pack.root) for record in rows
                )
                for defect_type, rows in pack.fewshot.items()
            }
            generators[pack_key] = A2ProposalGenerator(
                sources,
                config=config,
                identity_manifest=None,
                encoder=encoder,
            )
        generator = generators[pack_key]
        identity = _identity_key(slot)
        row_dir = sample_root / pack.dataset / pack.name / f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        row_dir.mkdir(parents=True)
        structure_key = (pack.dataset, pack.name, slot.normal_id)
        if structure_key not in structures:
            target_record = next(
                record for record in pack.normals if record.item_id == slot.normal_id
            )
            structures[structure_key] = estimate_normal_structure(
                load_extra_identity(target_record, dataset_root=pack.root),
                config=config,
            )
        target = structures[structure_key]
        source = generator.select_source(slot.defect_type, slot.seed)
        if source.record.item_id != slot.source_id:
            raise RuntimeError(
                f"preview source drifted: {identity} expected {slot.source_id} got {source.record.item_id}"
            )
        assert source.anomaly_gt is not None
        inpainted = neighbor_fill_source(source.rgb, source.anomaly_gt, source.valid)
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        empty = np.zeros(source.rgb.shape[:2], dtype=bool)
        arrays = {
            "source_pts": _overlay(source.rgb, source.anomaly_gt, (255, 40, 40), 0.45),
            "sim_m": _heatmap(np.zeros(empty.shape, dtype=np.float32), empty),
            "sim_c": _heatmap(np.zeros(empty.shape, dtype=np.float32), empty),
            "sim_p": _heatmap(np.zeros(empty.shape, dtype=np.float32), empty),
            "target_pts": normal.copy(),
            "a2_prop": _mask_u8(empty),
            "site": _mask_u8(empty),
            "rigid": _mask_u8(empty),
            "corr": _mask_u8(empty),
            "overlay": _overlay(normal, empty, (80, 160, 255), 0.28),
        }
        record: dict[str, Any] = {
            "identity": identity,
            "dataset": slot.dataset,
            "category": slot.category,
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "source_id": source.record.item_id,
            "seed": slot.seed,
            "failed": False,
            "failure_reason": None,
            "used_corr_fallback": None,
            "corr_fail_reason": None,
            "object_support_method": None,
            "proposal_pixels": 0,
            "site_pixels": 0,
        }
        try:
            result = generator.generate(target, slot.defect_type, slot.seed)
            proposal = np.asarray(result.proposal, dtype=bool)
            similarity = np.asarray(result.similarity, dtype=np.float32)
            source_support = np.asarray(result.source_support, dtype=bool)
            target_support = np.asarray(result.target_support, dtype=bool)
            _labels, sites = label_a2_components(proposal, similarity, min_pixels=1)
            preferred = pick_largest_site(sites) if sites else None
            site_mask = preferred.mask if preferred is not None else np.zeros_like(proposal)
            search = target_support & site_mask & target.valid
            source_feat = generator._encoder.encode(inpainted)
            target_feat = generator._encoder.encode(target.normal.rgb)
            arrays.update(
                {
                    "sim_m": _heatmap(np.zeros(proposal.shape, dtype=np.float32), search),
                    "sim_c": _heatmap(np.zeros(proposal.shape, dtype=np.float32), search),
                    "sim_p": _heatmap(np.zeros(proposal.shape, dtype=np.float32), search),
                    "a2_prop": _mask_u8(proposal),
                    "site": _mask_u8(site_mask),
                    "overlay": _overlay(normal, proposal, (80, 160, 255), 0.28),
                }
            )
            record.update(
                object_support_method=result.object_support_method,
                proposal_pixels=int(proposal.sum()),
                site_pixels=int(site_mask.sum()),
            )
            if result.target_support is None or result.source_support is None:
                raise ValueError("official a2 must return TokenCut supports")
            if result.object_support_method != "tokencut":
                raise ValueError(
                    f"{identity} expected tokencut support, got {result.object_support_method}"
                )
            rigid, corr, info = place_corr3_or_rigid(
                source.anomaly_gt,
                source_support,
                target_support,
                proposal,
                similarity,
                source_feat=source_feat,
                target_feat=target_feat,
                search=search,
                seed=slot.seed,
            )
            record.update(
                used_corr_fallback=info["used_corr_fallback"],
                corr_fail_reason=info.get("corr_fail_reason"),
                used_proposal_floor=info.get("used_proposal_floor"),
                used_site_fallback=info.get("used_site_fallback"),
                rigid_pixels=int(rigid.sum()),
                corr_pixels=int(corr.sum()),
                corr_scale=info.get("corr_scale"),
                rigid_scale=info.get("scale"),
                m_final_subset_of_site=bool(not (corr & ~site_mask).any()) if site_mask.any() else False,
            )
            source_rc = info.get("source_rc")
            target_rc = info.get("target_rc")
            if source_rc is None:
                try:
                    minus, center, plus = mask_axis_keypoints(source.anomaly_gt)
                    source_rc = np.stack((minus, center, plus), axis=0)
                except CorrespondenceFail:
                    source_rc = None
            arrays["source_pts"] = _draw_points(arrays["source_pts"], source_rc)
            arrays["target_pts"] = _draw_points(
                normal, target_rc if not info["used_corr_fallback"] else None
            )
            maps = info.get("sim_maps")
            if maps is not None:
                arrays["sim_m"] = _heatmap(maps[0], search)
                arrays["sim_c"] = _heatmap(maps[1], search)
                arrays["sim_p"] = _heatmap(maps[2], search)
            arrays["site"] = _mask_u8(site_mask)
            arrays["rigid"] = _mask_u8(rigid)
            arrays["corr"] = _mask_u8(corr)
            arrays["overlay"] = _overlay(
                _overlay(
                    _overlay(
                        _overlay(normal, proposal, (80, 160, 255), 0.28),
                        site_mask,
                        (40, 200, 80),
                        0.35,
                    ),
                    rigid,
                    (255, 220, 0),
                    0.55,
                ),
                corr,
                (220, 40, 180) if not info["used_corr_fallback"] else (255, 140, 0),
                0.55,
            )
        except (EmptyFeasiblePlacement, CorrespondenceFail, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        for name, array in arrays.items():
            Image.fromarray(array).save(row_dir / f"{name}.png")
        write_json_atomic(row_dir / "record.json", record)
        records.append(record)
        if record["failed"]:
            label_text = identity + " FAIL"
        elif record.get("used_corr_fallback"):
            label_text = identity + " RIGID"
        else:
            label_text = identity + " CORR"
        flow = {name: arrays[name] for name in FLOW_COLUMNS}
        flow_rows.append((label_text, flow))
        per_category[pack_key].append((label_text, flow))
        print(identity, label_text.split(" ", 1)[-1], flush=True)

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    for pack_key, rows in per_category.items():
        slug = pack_key.replace("/", "_")
        _save_sheet(destination / f"flow_{slug}.png", rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "not_official_h": True,
        "not_learned": True,
        "not_generate": True,
        "one_variable": "h_shape_rigid_vs_3point_dino_similarity_extra_cad3",
        "identity_count": len(records),
        "expected_identity_count": EXPECTED_PREVIEW_ROWS,
        "failed_rows": sum(bool(row["failed"]) for row in records),
        "corr_rows": sum(not row["failed"] and not row.get("used_corr_fallback") for row in records),
        "rigid_fallback_rows": sum(bool(row.get("used_corr_fallback")) for row in records),
        "preview_rule": "first_2_types_x_first_2_frozen_normals_x_seed_2026082801",
        "categories": [f"{pack.dataset}/{pack.name}" for pack in packs],
        "flow_columns": list(FLOW_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "line_ratio": LINE_RATIO,
        "search": "tokencut_object ∩ largest_a2_cc ∩ V",
        "encoder": "same frozen dinov2 as official a2",
        "destination": str(destination),
        "visible_defects_passed": None,
        "excluded": "can is informal third-class and not in this 34-row freeze",
        "did_not_overwrite_cable_corr3": True,
        "did_not_overwrite_extra_place_20260911": True,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_h_corr3_extra_diagnostic_from_cli(
    *,
    paths: Path,
    visa_root: Path,
    cad3_root: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_h_corr3_extra_diagnostic(
        mvtec_root=project.dataset_root,
        visa_root=visa_root if visa_root.is_absolute() else cwd / visa_root,
        cad3_root=cad3_root if cad3_root.is_absolute() else cwd / cad3_root,
        output=output if output.is_absolute() else cwd / output,
        device=device,
    )


def run_h_corr3_diagnostic_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_h_corr3_diagnostic(
        repository_root=cwd,
        dataset_root=project.dataset_root,
        output=output if output.is_absolute() else cwd / output,
        a_experiment=a_experiment if a_experiment.is_absolute() else cwd / a_experiment,
        device=device,
    )


__all__ = [
    "CorrespondenceFail",
    "apply_similarity",
    "assert_axis_consistent",
    "mask_axis_keypoints",
    "match_keypoints",
    "place_corr3_or_rigid",
    "run_h_corr3_diagnostic",
    "run_h_corr3_diagnostic_from_cli",
    "run_h_corr3_extra_diagnostic",
    "run_h_corr3_extra_diagnostic_from_cli",
    "similarity_from_keypoints",
    "warp_gt_with_correspondence",
]

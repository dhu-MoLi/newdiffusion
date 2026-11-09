"""Step-by-step official A/H panels for already frozen VisA failure rows.

Replays TokenCut, a2 q=0.60, and site-scale placement on the SeaS-500 seeds.
final_y is the landed PNG. Does not train, does not generate, does not overwrite dumps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_b.a2_component_place import MIN_SHAPE_PIXELS, candidate_tokencut_sites
from cadiff.v06_c.capsules_combined_panel import _contained_output, _find_image, n_connected
from cadiff.v06_c.loc_pred_gt_overlay import _mask_path
from cadiff.v06_c.official_a2_place import placement_from_official_a2
from cadiff.v06_c.oneshot_appearance_panel import _blend, mask_edge
from cadiff.v06_c.visa_pilot import (
    SOURCE_ROOT,
    fewshot_identity_records,
    load_visa_identity,
    load_visa_seas_inventory,
    normal_identity_records,
)

SCHEMA = "v06_c_visa_flow_steps_panel_v1"
EVAL_ROOT = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_c")
EVAL_NAME = "20260922-seas-500"
LAYOUT_NAME = "mvtec_layout_seas_20260922"
CELL = 220
CAP_H = 44
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT = ImageFont.truetype(FONT_PATH, 15)
SITE_COLORS = (
    (80, 180, 255),
    (255, 180, 40),
    (160, 80, 255),
    (80, 220, 140),
    (255, 80, 80),
    (255, 120, 200),
)

# Identities already used in the 2026-09-27 look. Written to disk before replay.
FLOW_ROWS: tuple[dict[str, Any], ...] = (
    {
        "category": "cashew",
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "096",
        "job_id": "stuck_together-000-20340912",
        "seed": 20340912,
        "target_id": "000",
        "source_id": "094",
    },
    {
        "category": "cashew",
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "097",
        "job_id": "stuck_together-001-20340913",
        "seed": 20340913,
        "target_id": "001",
        "source_id": "095",
    },
    {
        "category": "cashew",
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "098",
        "job_id": "stuck_together-002-20340914",
        "seed": 20340914,
        "target_id": "002",
        "source_id": "094",
    },
    {
        "category": "cashew",
        "type_slug": "stuck_together",
        "defect_type": "stuck together",
        "test_stem": "099",
        "job_id": "stuck_together-003-20340915",
        "seed": 20340915,
        "target_id": "003",
        "source_id": "095",
    },
    {
        "category": "candle",
        "type_slug": "extra_wax_in_candle",
        "defect_type": "extra wax in candle",
        "test_stem": "063",
        "job_id": "extra_wax_in_candle-0000-20300912",
        "seed": 20300912,
        "target_id": "0000",
        "source_id": "062",
    },
    {
        "category": "candle",
        "type_slug": "damaged_corner_of_packaging",
        "defect_type": "damaged corner of packaging",
        "test_stem": "018",
        "job_id": "damaged_corner_of_packaging-0000-20280912",
        "seed": 20280912,
        "target_id": "0000",
        "source_id": "010",
    },
    {
        "category": "candle",
        "type_slug": "weird_candle_wick",
        "defect_type": "weird candle wick",
        "test_stem": "033",
        "job_id": "weird_candle_wick-0000-20330912",
        "seed": 20330912,
        "target_id": "0000",
        "source_id": "032",
    },
)

COLUMNS = (
    "测试图+GT unpaired",
    "源图+源GT",
    "正常图 x",
    "TokenCut 物体",
    "材料余弦",
    "proposal q=0.60",
    "site 块",
    "模板 T(g)",
    "未裁放置",
    "裁掉的部分",
    "site + m_final",
    "落盘 final_y",
)


def _open_rgb(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _open_mask(path: Path, size: tuple[int, int]) -> NDArray[np.bool_]:
    height, width = size
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((width, height), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return array > 127


def _layout(category: str) -> Path:
    return EVAL_ROOT / "visa" / category / LAYOUT_NAME / category


def _sample(category: str, job_id: str) -> Path:
    return EVAL_ROOT / "visa" / category / "generate" / EVAL_NAME / "samples" / job_id


def _cell(rgb: np.ndarray, caption: str) -> NDArray[np.uint8]:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("cell rgb must be HxWx3")
    tile = np.asarray(
        Image.fromarray(image).resize((CELL, CELL), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.full((CELL + CAP_H, CELL, 3), 255, dtype=np.uint8)
    canvas[CAP_H:] = tile
    drawn = Image.fromarray(canvas)
    pen = ImageDraw.Draw(drawn)
    pen.multiline_text((4, 2), caption, fill=(20, 20, 20), font=_FONT, spacing=1)
    return np.asarray(drawn, dtype=np.uint8)


def _paint(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], weight: float) -> NDArray[np.uint8]:
    out = _blend(rgb, color, mask, weight)
    edge = mask_edge(mask)
    out[edge] = np.asarray(color, dtype=np.uint8)
    return out


def _heat(similarity: np.ndarray, support: np.ndarray) -> NDArray[np.uint8]:
    sim = np.asarray(similarity, dtype=np.float32)
    obj = np.asarray(support, dtype=bool)
    out = np.zeros((*sim.shape, 3), dtype=np.uint8)
    if not obj.any():
        return out
    vals = sim[obj]
    lo = float(vals.min())
    hi = float(vals.max())
    norm = np.zeros_like(sim)
    if hi > lo:
        norm = (sim - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    out[..., 0] = (norm * 255).astype(np.uint8)
    out[..., 1] = ((0.25 + 0.55 * norm) * 255).astype(np.uint8)
    out[..., 2] = ((1.0 - norm) * 255).astype(np.uint8)
    out[~obj] = 16
    return out


def _sites_overlay(rgb: np.ndarray, sites: tuple[Any, ...], chosen_index: int) -> NDArray[np.uint8]:
    out = np.asarray(rgb, dtype=np.uint8).copy()
    for site in sites:
        color = SITE_COLORS[(int(site.index) - 1) % len(SITE_COLORS)]
        out = _blend(out, color, site.mask, 0.45)
        out[mask_edge(site.mask)] = np.asarray(color, dtype=np.uint8)
    chosen = next(site for site in sites if int(site.index) == int(chosen_index))
    out[mask_edge(chosen.mask)] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def _join(cells: list[NDArray[np.uint8]], title: str) -> NDArray[np.uint8]:
    gap = 4
    header = 28
    width = sum(cell.shape[1] for cell in cells) + gap * (len(cells) - 1)
    canvas = np.full((header + cells[0].shape[0], width, 3), 245, dtype=np.uint8)
    image = Image.fromarray(canvas)
    ImageDraw.Draw(image).text((4, 4), title, fill=(20, 20, 20), font=_FONT)
    canvas = np.array(image, dtype=np.uint8, copy=True)
    x = 0
    y = header
    for cell in cells:
        canvas[y : y + cell.shape[0], x : x + cell.shape[1]] = cell
        x += cell.shape[1] + gap
    return canvas


def _stack(rows: list[NDArray[np.uint8]], heading: str) -> NDArray[np.uint8]:
    gap = 8
    header = 32
    width = max(row.shape[1] for row in rows)
    height = header + sum(row.shape[0] for row in rows) + gap * (len(rows) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    image = Image.fromarray(canvas)
    ImageDraw.Draw(image).text((4, 4), heading, fill=(20, 20, 20), font=_FONT)
    canvas = np.array(image, dtype=np.uint8, copy=True)
    y = header
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + gap
    return canvas


def _build_generator(category: str, encoder: Any) -> tuple[Any, dict[str, Any], Any]:
    inventory = load_visa_seas_inventory(SOURCE_ROOT, category)
    fewshot = fewshot_identity_records(inventory)
    sources = {
        item.name: tuple(
            load_visa_identity(record, dataset_root=SOURCE_ROOT)
            for record in fewshot
            if record.defect_type == item.name
        )
        for item in inventory.types
    }
    generator = A2ProposalGenerator(
        sources,
        config=inventory.structure,
        identity_manifest=None,
        encoder=encoder,
    )
    normals = {
        record.item_id: record for record in normal_identity_records(inventory)
    }
    return generator, normals, inventory.structure


def _render_row(
    spec: dict[str, Any],
    generator: Any,
    normals: dict[str, Any],
    structures: dict[str, Any],
    structure_config: Any,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    category = spec["category"]
    layout = _layout(category)
    test_dir = layout / "test" / spec["type_slug"]
    gt_root = layout / "ground_truth"
    sample = _sample(category, spec["job_id"])
    saved_mask = np.asarray(Image.open(sample / "m_final.png").convert("L")) > 127
    saved_y = np.asarray(Image.open(sample / "final_y.png").convert("RGB"))
    record = __import__("json").loads((sample / "record.json").read_text(encoding="utf-8"))
    if str(record["source_id"]) != spec["source_id"] or int(record["seed"]) != spec["seed"]:
        raise RuntimeError(f"frozen identity drifted from dump record: {spec['job_id']}")
    target_id = spec["target_id"]
    if target_id not in structures:
        loaded = load_visa_identity(normals[target_id], dataset_root=SOURCE_ROOT)
        structures[target_id] = estimate_normal_structure(loaded, config=structure_config)
    target = structures[target_id]
    a2, placement, template = placement_from_official_a2(
        generator,
        target,
        spec["defect_type"],
        spec["seed"],
        min_shape_pixels=MIN_SHAPE_PIXELS,
        return_template=True,
    )
    if a2.source_id != spec["source_id"]:
        raise RuntimeError(
            f"{spec['job_id']} replay source {a2.source_id} != frozen {spec['source_id']}"
        )
    replay = np.ascontiguousarray(placement.mask, dtype=bool)
    mismatch = int(np.logical_xor(replay, saved_mask).sum())
    if mismatch:
        raise RuntimeError(f"{spec['job_id']} replay m_final mismatches the dump by {mismatch} px")
    normal = np.ascontiguousarray(target.normal.rgb)
    support = np.ascontiguousarray(a2.target_support & target.valid, dtype=bool)
    proposal = np.ascontiguousarray(a2.proposal, dtype=bool)
    sites = candidate_tokencut_sites(support, proposal, min_pixels=MIN_SHAPE_PIXELS)
    unclipped = np.ascontiguousarray(
        placement.unclipped if placement.unclipped is not None else placement.mask,
        dtype=bool,
    )
    extra = np.ascontiguousarray(
        placement.extra if placement.extra is not None else np.zeros_like(replay),
        dtype=bool,
    )
    test_rgb = _open_rgb(_find_image(test_dir, spec["test_stem"]))
    test_gt = _open_mask(_mask_path(gt_root, spec["type_slug"], spec["test_stem"]), test_rgb.shape[:2])
    source_rgb = _open_rgb(_find_image(test_dir, spec["source_id"]))
    source_gt = _open_mask(_mask_path(gt_root, spec["type_slug"], spec["source_id"]), source_rgb.shape[:2])
    template_mask = np.ascontiguousarray(template, dtype=bool)
    blank = np.zeros((*template_mask.shape, 3), dtype=np.uint8)
    blank[template_mask] = 255
    cells = [
        _cell(_paint(test_rgb, test_gt, (40, 220, 80), 0.45), f"测试+GT 绿\n{spec['test_stem']} unpaired"),
        _cell(_paint(source_rgb, source_gt, (40, 200, 220), 0.45), f"源图+源GT 青\n{spec['source_id']}"),
        _cell(normal, "正常图 x"),
        _cell(_paint(normal, support, (40, 200, 220), 0.4), f"TokenCut 物体\n{int(support.sum())} px"),
        _cell(_heat(np.asarray(a2.similarity), support), f"材料余弦\ntau={a2.tau:.3f}"),
        _cell(_paint(normal, proposal, (250, 210, 40), 0.45), f"proposal\n{int(proposal.sum())} px"),
        _cell(
            _sites_overlay(normal, sites, placement.site_index),
            f"site 块 白边=抽中\n#{placement.site_index}/{len(sites)}",
        ),
        _cell(blank, f"模板 T(g)\n×{placement.scale:.2f} {placement.angle_deg}°"),
        _cell(_paint(normal, unclipped, (40, 200, 220), 0.5), f"未裁放置\n{int(unclipped.sum())} px"),
        _cell(_paint(normal, extra, (240, 40, 40), 0.65), f"裁掉 红\n{int(extra.sum())} px"),
        _cell(
            _paint(_paint(normal, placement.site, (60, 120, 255), 0.28), replay, (240, 40, 180), 0.55),
            f"site+m_final 品红\n{int(replay.sum())} px cc={n_connected(replay)}",
        ),
        _cell(saved_y, f"落盘 final_y\nclip={int(bool(placement.used_clip))}"),
    ]
    title = (
        f"{category}/{spec['type_slug']}  test {spec['test_stem']}  "
        f"src {spec['source_id']}  job {spec['job_id']}  "
        f"clip={int(bool(placement.used_clip))}"
    )
    stats = {
        "job_id": spec["job_id"],
        "source_id": a2.source_id,
        "tau": None if a2.tau is None else float(a2.tau),
        "support_px": int(support.sum()),
        "proposal_px": int(proposal.sum()),
        "n_sites": len(sites),
        "site_index": int(placement.site_index),
        "site_px": int(placement.site.sum()),
        "template_px": int(template_mask.sum()),
        "unclipped_px": int(unclipped.sum()),
        "extra_px": int(extra.sum()),
        "m_final_px": int(replay.sum()),
        "m_final_cc": n_connected(replay),
        "used_clip": bool(placement.used_clip),
        "scale": float(placement.scale),
        "angle_deg": int(placement.angle_deg),
        "mismatch_px": mismatch,
    }
    return _join(cells, title), stats


def run_visa_flow_steps_panel(output: Path, *, device: str = "cpu") -> dict[str, Any]:
    destination = _contained_output(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": "rows frozen from the 2026-09-27 look before replay; SeaS-500 seeds; test images unpaired",
            "columns": list(COLUMNS),
            "rows": list(FLOW_ROWS),
            "device": device,
            "not_train_r": True,
            "not_overwrite_dump": True,
            "no_can": True,
        },
    )
    encoder = load_official_dinov2_encoder(device=device)
    by_category: dict[str, tuple[Any, dict[str, Any]]] = {}
    structures: dict[tuple[str, str], Any] = {}
    strips: list[NDArray[np.uint8]] = []
    stats: list[dict[str, Any]] = []
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    grouped: dict[str, list[NDArray[np.uint8]]] = {}
    for spec in FLOW_ROWS:
        category = spec["category"]
        if category not in by_category:
            by_category[category] = _build_generator(category, encoder)
        generator, normals, structure_config = by_category[category]
        cache = structures.setdefault(category, {})
        strip, row_stats = _render_row(spec, generator, normals, cache, structure_config)
        Image.fromarray(strip).save(rows_dir / f"{spec['job_id']}.png")
        strips.append(strip)
        grouped.setdefault(category, []).append(strip)
        stats.append(row_stats)
        print(spec["job_id"], row_stats, flush=True)
    for category, rows in grouped.items():
        Image.fromarray(_stack(rows, f"{category}  左到右是正式 A/H 的中间步  测试列 unpaired")).save(
            destination / f"sheet_{category}.png"
        )
    Image.fromarray(_stack(strips, "正式 A/H 分步  测试列与生成列 unpaired")).save(destination / "sheet.png")
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_train_r": True,
        "no_can": True,
        "unpaired": True,
        "n": len(stats),
        "rows": stats,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload

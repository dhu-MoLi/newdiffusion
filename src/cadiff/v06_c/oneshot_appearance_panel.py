"""Read-only 1-shot appearance + placement panel from landed official dumps.

Columns: normal | TokenCut CC | proposal | m_final | final_y | zoom.
Identities are the first PER_TYPE kept jobs per type, sorted by dump index,
written before any DINO/TokenCut call. No train, no generate, no can, does
not overwrite dumps. Not Module C.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.tokencut import tokencut_object_mask
from cadiff.v06_c.tokencut_normal_preview import (
    color_ranked_components,
    rank_components,
)

SCHEMA = "v06_c_oneshot_appearance_v1"
STAMP_500 = "20260918-official-500"
STAMP_PLUS = "20260920-official-plus500"
PER_TYPE = 2
DISPLAY_SIZE = 256
ZOOM_PAD = 16
ZOOM_MIN = 64
PROPOSAL_COLOR = (40, 200, 220)
MASK_COLOR = (240, 40, 180)
EDGE_COLOR = (250, 230, 40)
_FONT = ImageFont.load_default()
ObjectFn = Callable[[NDArray[np.uint8], str], NDArray[np.bool_]]

PANEL_TYPES: tuple[dict[str, str], ...] = (
    {"category": "candle", "type_slug": "extra_wax_in_candle", "role": "fail_location"},
    {"category": "candle", "type_slug": "damaged_corner_of_packaging", "role": "fail_scope"},
    {"category": "candle", "type_slug": "different_colour_spot", "role": "fail_tiny"},
    {"category": "cable", "type_slug": "bent_wire", "role": "win"},
    {"category": "chewinggum", "type_slug": "chunk_of_gum_missing", "role": "win"},
    {"category": "cashew", "type_slug": "burnt", "role": "image_lag"},
    {"category": "cable", "type_slug": "cut_outer_insulation", "role": "thin_cover"},
)


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def record_type_slug(record: Mapping[str, Any]) -> str:
    if record.get("type_slug"):
        return str(record["type_slug"])
    return str(record["defect_type"]).replace(" ", "_")


def official_dump_roots(runs_v06c: Path) -> dict[str, tuple[Path, ...]]:
    root = Path(runs_v06c)
    visa = root / "visa"
    return {
        "cable": (
            root / "cable/generate" / STAMP_500,
            root / "cable/generate" / STAMP_PLUS,
        ),
        "candle": (
            visa / "candle/generate" / STAMP_500,
            visa / "candle/generate" / STAMP_PLUS,
        ),
        "cashew": (
            visa / "cashew/generate" / STAMP_500,
            visa / "cashew/generate" / STAMP_PLUS,
        ),
        "chewinggum": (
            visa / "chewinggum/generate" / STAMP_500,
            visa / "chewinggum/generate" / STAMP_PLUS,
        ),
    }


def _sample_complete(sample: Path) -> bool:
    return all(
        (sample / name).is_file()
        for name in ("normal.png", "proposal.png", "m_final.png", "final_y.png", "record.json")
    )


def list_kept_jobs(dump: Path, type_slug: str) -> tuple[dict[str, Any], ...]:
    samples = dump / "samples"
    if not samples.is_dir():
        raise FileNotFoundError(samples)
    rows: list[dict[str, Any]] = []
    for record_path in samples.glob("*/record.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if bool(record.get("failed")):
            continue
        if record_type_slug(record) != type_slug:
            continue
        category = str(record.get("category") or dump.parent.parent.name)
        if category == "can":
            raise ValueError("this diagnostic excludes MVTec AD 2 can")
        sample = record_path.parent
        if not _sample_complete(sample):
            continue
        rows.append(
            {
                "category": category,
                "type_slug": type_slug,
                "defect_type": str(record["defect_type"]),
                "job_id": str(record.get("job_id", sample.name)),
                "index": int(record["index"]),
                "seed": int(record["seed"]),
                "target_id": str(record.get("target_id")),
                "source_id": str(record.get("source_id")),
                "site_index": record.get("site_index"),
                "placed_pixels": int(record.get("placed_pixels") or 0),
                "proposal_pixels": int(record.get("proposal_pixels") or 0),
                "dump": dump.name,
                "dump_path": str(dump),
                "sample_path": str(sample),
            }
        )
    rows.sort(key=lambda item: (item["index"], item["job_id"]))
    return tuple(rows)


def select_panel_identities(
    runs_v06c: Path,
    *,
    per_type: int = PER_TYPE,
    types: Sequence[Mapping[str, str]] = PANEL_TYPES,
) -> tuple[dict[str, Any], ...]:
    if per_type < 1:
        raise ValueError("per_type must be positive")
    dumps = official_dump_roots(runs_v06c)
    selected: list[dict[str, Any]] = []
    for spec in types:
        category = str(spec["category"])
        if category == "can":
            raise ValueError("this diagnostic excludes MVTec AD 2 can")
        type_slug = str(spec["type_slug"])
        picked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for dump in dumps[category]:
            for row in list_kept_jobs(dump, type_slug):
                if row["job_id"] in seen:
                    continue
                seen.add(row["job_id"])
                picked.append({**row, "role": str(spec["role"])})
                if len(picked) >= per_type:
                    break
            if len(picked) >= per_type:
                break
        if len(picked) < per_type:
            raise ValueError(f"{category}/{type_slug} has fewer than {per_type} kept dump jobs")
        selected.extend(picked)
    return tuple(selected)


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _read_mask(path: Path) -> NDArray[np.bool_]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array > 127


def _resize_rgb(rgb: np.ndarray, size: int) -> NDArray[np.uint8]:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.shape[0] == size and array.shape[1] == size:
        return array
    return np.asarray(
        Image.fromarray(array).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _resize_mask(mask: np.ndarray, size: int) -> NDArray[np.bool_]:
    array = np.asarray(mask, dtype=bool)
    if array.shape[0] == size and array.shape[1] == size:
        return array
    return (
        np.asarray(
            Image.fromarray(array.astype(np.uint8) * 255).resize(
                (size, size), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        > 127
    )


def _blend(rgb: np.ndarray, color: tuple[int, int, int], mask: np.ndarray, weight: float) -> NDArray[np.uint8]:
    base = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(mask, dtype=np.float32) * float(weight)
    tint = np.asarray(color, dtype=np.float32)
    out = base * (1.0 - alpha[..., None]) + tint * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_edge(mask: np.ndarray) -> NDArray[np.bool_]:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return binary
    return np.ascontiguousarray(binary & ~binary_erosion(binary, iterations=1))


def crop_around_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    pad: int = ZOOM_PAD,
    min_size: int = ZOOM_MIN,
    out_size: int = DISPLAY_SIZE,
) -> NDArray[np.uint8]:
    image = np.asarray(rgb, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if image.shape[:2] != binary.shape:
        raise ValueError("rgb and mask spatial size must match")
    height, width = binary.shape
    if not binary.any():
        return _resize_rgb(image, out_size)
    ys, xs = np.where(binary)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    side = max(y1 - y0, x1 - x0, min_size) + 2 * pad
    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    y1 = min(height, y0 + side)
    x1 = min(width, x0 + side)
    y0 = max(0, y1 - (x1 - x0))
    x0 = max(0, x1 - (y1 - y0))
    crop = image[y0:y1, x0:x1]
    return np.asarray(
        Image.fromarray(crop).resize((out_size, out_size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


def letterbox_valid(rgb: np.ndarray) -> NDArray[np.bool_]:
    occupied = np.any(np.asarray(rgb, dtype=np.uint8) > 8, axis=2)
    if float(occupied.mean()) < 0.20:
        return np.ones(rgb.shape[:2], dtype=bool)
    return occupied


def default_object_fn(encode_fn: Callable[[np.ndarray, str], np.ndarray]) -> ObjectFn:
    def object_mask(rgb: np.ndarray, identity: str) -> NDArray[np.bool_]:
        tokens = encode_fn(rgb, identity)
        return np.asarray(tokencut_object_mask(tokens, letterbox_valid(rgb)), dtype=bool)

    return object_mask


def _row_image(tiles: Sequence[np.ndarray], title: str) -> NDArray[np.uint8]:
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
    draw.text(
        (4, 2),
        f"{title}  rgb | TokenCut-CC | proposal | m_final | final_y | zoom-y",
        fill=(20, 20, 20),
        font=_FONT,
    )
    return np.asarray(image, dtype=np.uint8)


def _stack_rows(rows: Sequence[np.ndarray], heading: str) -> NDArray[np.uint8]:
    width = max(row.shape[1] for row in rows)
    header = 36
    height = header + sum(row.shape[0] for row in rows) + 6 * (len(rows) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), heading, fill=(20, 20, 20), font=_FONT)
    canvas = np.array(image, dtype=np.uint8, copy=True)
    y = header
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + 6
    return canvas


def run_oneshot_appearance_panel(
    *,
    runs_v06c: Path,
    output: Path,
    object_fn: ObjectFn | None = None,
    encode_fn: Callable[[np.ndarray, str], np.ndarray] | None = None,
    per_type: int = PER_TYPE,
) -> dict[str, Any]:
    destination = _contained_output(output)
    identities = select_panel_identities(runs_v06c, per_type=per_type)
    destination.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "rule": "first PER_TYPE kept dump jobs per type, sorted by index; official-500 then plus500; frozen before TokenCut",
            "per_type": per_type,
            "stamp_500": STAMP_500,
            "stamp_plus": STAMP_PLUS,
            "no_can": True,
            "not_train_r": True,
            "not_generate": True,
            "look_for": [
                "extra_wax: mask on wax body vs wick / other candle",
                "packaging: mask on box vs on candle object",
                "colour_spot: large blob vs needle GT; is final_y readable",
                "win types: readable local rewrite on the object",
            ],
            "identities": identities,
        },
    )
    if object_fn is None:
        if encode_fn is None:
            raise ValueError("encode_fn is required when object_fn is omitted")
        object_fn = default_object_fn(encode_fn)

    sample_root = destination / "samples"
    sample_root.mkdir()
    sheets: dict[str, list[np.ndarray]] = {}
    all_rows: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for row in identities:
        sample = Path(row["sample_path"])
        rgb = _read_rgb(sample / "normal.png")
        proposal = _read_mask(sample / "proposal.png")
        m_final = _read_mask(sample / "m_final.png")
        final_y = _read_rgb(sample / "final_y.png")
        identity = f"{row['category']}-{row['type_slug']}-{row['job_id']}"
        object_mask = np.asarray(object_fn(rgb, identity), dtype=bool)
        if object_mask.shape != rgb.shape[:2]:
            raise ValueError(f"object_fn returned {object_mask.shape}, expected {rgb.shape[:2]}")
        ranked = rank_components(object_mask)
        tokencut_color = color_ranked_components(ranked, object_mask.shape)
        display = DISPLAY_SIZE
        base = _resize_rgb(rgb, display)
        y_disp = _resize_rgb(final_y, display)
        occupied = _resize_rgb(tokencut_color, display)
        tokencut_on_rgb = base.copy()
        lit = occupied.any(axis=2)
        tokencut_on_rgb[lit] = (
            0.35 * base[lit].astype(np.float32) + 0.65 * occupied[lit].astype(np.float32)
        ).astype(np.uint8)
        proposal_d = _resize_mask(proposal, display)
        mask_d = _resize_mask(m_final, display)
        zoom_y = crop_around_mask(final_y, m_final)
        zoom_mask = crop_around_mask(
            np.repeat(m_final.astype(np.uint8)[..., None] * 255, 3, axis=2),
            m_final,
        )[..., 0] > 127
        zoom_y = _blend(zoom_y, EDGE_COLOR, mask_edge(zoom_mask), 1.0)
        tiles = (
            base,
            tokencut_on_rgb,
            _blend(base, PROPOSAL_COLOR, proposal_d, 0.55),
            _blend(base, MASK_COLOR, mask_d, 0.70),
            y_disp,
            zoom_y,
        )
        title = (
            f"{row['role']} {row['category']}/{row['type_slug']} "
            f"idx={row['index']} site={row['site_index']} px={row['placed_pixels']}"
        )
        strip = _row_image(tiles, title)
        Image.fromarray(strip).save(sample_root / f"{row['role']}-{row['job_id']}.png")
        sheets.setdefault(row["type_slug"], []).append(strip)
        all_rows.append(strip)
        reports.append(
            {
                **{key: row[key] for key in row},
                "n_tokencut_cc": len(ranked),
                "object_pixels": int(object_mask.sum()),
                "mask_on_object": float((m_final & object_mask).sum() / m_final.sum()) if m_final.any() else 0.0,
            }
        )
        print(row["job_id"], "DONE", flush=True)

    heading = (
        "1-shot appearance+place  official-1000 dump  "
        "rgb | TokenCut-CC | proposal-cyan | m_final-magenta | final_y | zoom-y (yellow=mask edge)  "
        "not Module C, not train R"
    )
    Image.fromarray(_stack_rows(all_rows, heading)).save(destination / "sheet_all.png")
    for type_slug, rows in sheets.items():
        Image.fromarray(_stack_rows(rows, f"{type_slug}  {heading}")).save(
            destination / f"sheet_{type_slug}.png"
        )
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_train_r": True,
        "not_generate": True,
        "no_can": True,
        "n": len(reports),
        "per_type": per_type,
        "types": [spec["type_slug"] for spec in PANEL_TYPES],
        "rows": reports,
        "destination": str(destination),
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_oneshot_appearance_panel_from_cli(
    *,
    runs_v06c: Path,
    output: Path,
    device: str | None = None,
) -> dict[str, Any]:
    from cadiff.v06_a.tokencut import load_official_dinov2_encoder

    encoder = load_official_dinov2_encoder(device=device or "cuda:0")

    def encode(image: np.ndarray, identity: str) -> np.ndarray:
        del identity
        return encoder.encode(image).tokens

    return run_oneshot_appearance_panel(runs_v06c=runs_v06c, output=output, encode_fn=encode)


__all__ = [
    "PANEL_TYPES",
    "PER_TYPE",
    "crop_around_mask",
    "list_kept_jobs",
    "official_dump_roots",
    "record_type_slug",
    "run_oneshot_appearance_panel",
    "run_oneshot_appearance_panel_from_cli",
    "select_panel_identities",
]

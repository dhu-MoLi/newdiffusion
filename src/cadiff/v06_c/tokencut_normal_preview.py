"""Diagnostic: TokenCut on frozen train/good normals, before vs after finalize.

Look-then-judge only. Identities are written before DINO. Not m_proposal /
m_final, not SAM, not generate(), not Module C. Cable + VisA pilots; can is
deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.tokencut import _finalize_largest_component, tokencut_bipartition_upsampled

SCHEMA = "v06_c_tokencut_normal_preview_v1"
N_PER_CATEGORY = 4
COVERAGE_KS = (1, 2, 3, 5)
DISPLAY_SIZE = 256
PILOT_CATEGORIES = ("cable", "candle", "capsules", "cashew", "chewinggum")
FORBIDDEN_CATEGORIES = frozenset({"can"})
_EIGHT = np.ones((3, 3), dtype=np.uint8)
_CC_COLORS = (
    (220, 70, 70),
    (50, 170, 80),
    (50, 110, 220),
    (220, 180, 40),
    (180, 70, 180),
)
_REST_COLOR = (120, 120, 120)
_RAW_COLOR = (40, 190, 210)
_OFFICIAL_COLOR = (40, 200, 90)
_DROPPED_COLOR = (220, 40, 180)
_FONT = ImageFont.load_default()
EncodeFn = Callable[[NDArray[np.uint8], str], NDArray[np.floating]]


@dataclass(frozen=True)
class RankedComponent:
    rank: int
    pixels: int
    mask: NDArray[np.bool_]


def even_normal_ids(ids: tuple[str, ...] | list[str], count: int) -> tuple[str, ...]:
    ordered = tuple(ids)
    if count < 1:
        raise ValueError("count must be positive")
    if len(ordered) < count:
        raise ValueError("not enough normal images for the TokenCut preview")
    if count == 1:
        return (ordered[0],)
    return tuple(ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count))


def select_preview_records(arms: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        category = str(arm["category"])
        if category in FORBIDDEN_CATEGORIES:
            raise ValueError("can is deferred; TokenCut preview does not include MVTec AD 2")
        item_ids = tuple(arm["item_ids"])
        relpaths = tuple(arm["relpaths"])
        if len(item_ids) != len(relpaths):
            raise ValueError("item_ids and relpaths must align")
        for item_id, relpath in zip(item_ids, relpaths, strict=True):
            rows.append(
                {
                    "category": category,
                    "item_id": str(item_id),
                    "image_relpath": str(relpath),
                }
            )
    return tuple(rows)


def rank_components(raw: np.ndarray) -> tuple[RankedComponent, ...]:
    binary = np.asarray(raw, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("raw mask must be 2D")
    if not binary.any():
        return ()
    labels, count = label(binary, structure=_EIGHT)
    ranked: list[RankedComponent] = []
    for raw_id in range(1, count + 1):
        mask = labels == raw_id
        ranked.append(RankedComponent(rank=0, pixels=int(mask.sum()), mask=np.ascontiguousarray(mask)))
    ranked.sort(key=lambda item: (-item.pixels, int(np.argmax(item.mask))))
    return tuple(
        RankedComponent(rank=index + 1, pixels=item.pixels, mask=item.mask)
        for index, item in enumerate(ranked)
    )


def coverage_at_k(ranked: tuple[RankedComponent, ...] | list[RankedComponent], k: int) -> float:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    total = int(sum(item.pixels for item in ranked))
    if total == 0:
        return 0.0
    kept = int(sum(item.pixels for item in tuple(ranked)[:k]))
    return float(kept / total)


def dropped_by_finalize(raw: np.ndarray, official: np.ndarray) -> NDArray[np.bool_]:
    source = np.asarray(raw, dtype=bool)
    kept = np.asarray(official, dtype=bool)
    if source.shape != kept.shape:
        raise ValueError("raw and official must share one grid")
    return np.ascontiguousarray(source & ~kept)


def _blend(rgb: np.ndarray, color: tuple[int, int, int], weight: np.ndarray) -> NDArray[np.uint8]:
    base = np.asarray(rgb, dtype=np.float32)
    alpha = np.asarray(weight, dtype=np.float32)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    tint = np.asarray(color, dtype=np.float32)
    out = base * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def color_ranked_components(ranked: tuple[RankedComponent, ...], shape: tuple[int, int]) -> NDArray[np.uint8]:
    image = np.zeros((*shape, 3), dtype=np.uint8)
    for item in ranked:
        color = _CC_COLORS[item.rank - 1] if item.rank <= len(_CC_COLORS) else _REST_COLOR
        image[item.mask] = color
    return image


def _resize_rgb(rgb: np.ndarray, size: int) -> NDArray[np.uint8]:
    if rgb.shape[0] == size:
        return np.asarray(rgb, dtype=np.uint8)
    return np.asarray(
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def _resize_mask(mask: np.ndarray, size: int) -> NDArray[np.bool_]:
    if mask.shape[0] == size:
        return np.asarray(mask, dtype=bool)
    return (
        np.asarray(
            Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(
                (size, size), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        > 127
    )


def _row_image(
    rgb: np.ndarray,
    raw: np.ndarray,
    ranked: tuple[RankedComponent, ...],
    official: np.ndarray,
    dropped: np.ndarray,
    title: str,
) -> NDArray[np.uint8]:
    display = DISPLAY_SIZE
    base = _resize_rgb(rgb, display)
    raw_d = _resize_mask(raw, display)
    official_d = _resize_mask(official, display)
    dropped_d = _resize_mask(dropped, display)
    ranked_color = color_ranked_components(
        tuple(
            RankedComponent(rank=item.rank, pixels=item.pixels, mask=_resize_mask(item.mask, display))
            for item in ranked
        ),
        (display, display),
    )
    ranked_on_rgb = base.copy()
    occupied = ranked_color.any(axis=2)
    ranked_on_rgb[occupied] = (
        0.35 * base[occupied].astype(np.float32) + 0.65 * ranked_color[occupied].astype(np.float32)
    ).astype(np.uint8)
    tiles = (
        base,
        _blend(base, _RAW_COLOR, raw_d.astype(np.float32) * 0.70),
        ranked_on_rgb,
        _blend(base, _OFFICIAL_COLOR, official_d.astype(np.float32) * 0.70),
        _blend(base, _DROPPED_COLOR, dropped_d.astype(np.float32) * 0.80),
    )
    gap = 8
    header = 18
    height = header + display
    width = display * len(tiles) + gap * (len(tiles) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for tile in tiles:
        canvas[header : header + display, x : x + display] = tile
        x += display + gap
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text(
        (4, 2),
        f"{title}  rgb | raw | ranked-CC | official-finalize | dropped",
        fill=(20, 20, 20),
        font=_FONT,
    )
    return np.asarray(image, dtype=np.uint8)


def _stack_rows(rows: list[np.ndarray], heading: str) -> NDArray[np.uint8]:
    width = max(row.shape[1] for row in rows)
    header = 22
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


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(("f1_", "f2_", "f3_", "f4_")) for part in output.resolve().parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def _analyze_row(
    rgb: np.ndarray,
    valid: np.ndarray,
    tokens: np.ndarray,
) -> dict[str, Any]:
    raw = np.asarray(tokencut_bipartition_upsampled(tokens), dtype=bool) & np.asarray(valid, dtype=bool)
    official = _finalize_largest_component(raw, valid)
    ranked = rank_components(raw)
    dropped = dropped_by_finalize(raw, official)
    coverages = {f"top{k}": coverage_at_k(ranked, k) for k in COVERAGE_KS}
    return {
        "raw": raw,
        "official": official,
        "ranked": ranked,
        "dropped": dropped,
        "n_cc": len(ranked),
        "raw_pixels": int(raw.sum()),
        "official_pixels": int(official.sum()),
        "dropped_pixels": int(dropped.sum()),
        "coverages": coverages,
        "cc_areas": [item.pixels for item in ranked[:10]],
    }


def run_tokencut_normal_preview(
    *,
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    output: Path,
    encode_fn: EncodeFn,
) -> dict[str, Any]:
    destination = _contained_output(output)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        category = str(row["category"])
        if category in FORBIDDEN_CATEGORIES:
            raise ValueError("can is deferred; TokenCut preview does not include MVTec AD 2")
        item_id = str(row["item_id"])
        rgb = np.asarray(row["rgb"], dtype=np.uint8)
        valid = np.asarray(row["valid"], dtype=bool)
        if rgb.shape != (512, 512, 3) or valid.shape != (512, 512):
            raise ValueError("preview rows must be 512x512")
        prepared.append(
            {
                "category": category,
                "item_id": item_id,
                "identity": f"{category}/{item_id}",
                "rgb": rgb,
                "valid": valid,
            }
        )
    destination.mkdir(parents=True)
    write_json_atomic(
        destination / "identities.json",
        {
            "schema_version": SCHEMA,
            "diagnostic_only": True,
            "not_m_proposal": True,
            "not_m_final": True,
            "n_per_category": N_PER_CATEGORY,
            "pilot_categories": list(PILOT_CATEGORIES),
            "identities": [
                {"category": row["category"], "item_id": row["item_id"], "identity": row["identity"]}
                for row in prepared
            ],
        },
    )
    reports: dict[str, Any] = {}
    sheets: dict[str, list[np.ndarray]] = {}
    all_rows: list[np.ndarray] = []
    for row in prepared:
        tokens = encode_fn(row["rgb"], row["identity"])
        analyzed = _analyze_row(row["rgb"], row["valid"], np.asarray(tokens))
        coverages = analyzed["coverages"]
        title = (
            f"{row['identity']}  n_cc={analyzed['n_cc']}  "
            f"top1={coverages['top1']:.2f} top2={coverages['top2']:.2f} "
            f"top3={coverages['top3']:.2f} top5={coverages['top5']:.2f}"
        )
        panel = _row_image(
            row["rgb"],
            analyzed["raw"],
            analyzed["ranked"],
            analyzed["official"],
            analyzed["dropped"],
            title,
        )
        arm = reports.setdefault(
            row["category"],
            {"n": 0, "rows": []},
        )
        arm["n"] += 1
        arm["rows"].append(
            {
                "identity": row["identity"],
                "n_cc": analyzed["n_cc"],
                "raw_pixels": analyzed["raw_pixels"],
                "official_pixels": analyzed["official_pixels"],
                "dropped_pixels": analyzed["dropped_pixels"],
                "coverages": coverages,
                "cc_areas": analyzed["cc_areas"],
            }
        )
        sheets.setdefault(row["category"], []).append(panel)
        all_rows.append(panel)
    for category, panels in sheets.items():
        Image.fromarray(_stack_rows(panels, f"TokenCut normals {category}")).save(
            destination / f"sheet_{category}.png"
        )
    Image.fromarray(_stack_rows(all_rows, "TokenCut normals Cable+VisA (no can)")).save(
        destination / "all.png"
    )
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "not_m_proposal": True,
        "not_m_final": True,
        "arm_names": list(reports),
        "arms": reports,
        "column_meanings": {
            "rgb": "train/good normal, letterbox 512",
            "raw": "TokenCut bipartition before finalize",
            "ranked-CC": "8-connected CCs of raw, colored by area rank",
            "official-finalize": "historical S_x^0 = largest CC + fill holes",
            "dropped": "raw pixels discarded by finalize",
        },
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def collect_pilot_rows(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cable_experiment: Path,
    repository_root: Path,
) -> tuple[dict[str, Any], ...]:
    from cadiff.v06_a.config import load_v06_a_experiment
    from cadiff.v06_a.data import load_identity_record
    from cadiff.v06_a.identities import load_v06_a_identity_manifest
    from cadiff.v06_atlas_diag.extra_a import _item_id, _record, load_extra_identity
    from cadiff.v06_c.visa_pilot import EXPECTED, load_visa_inventory

    cwd = repository_root.resolve()
    experiment = load_v06_a_experiment(
        cable_experiment if cable_experiment.is_absolute() else cwd / cable_experiment
    )
    manifest = load_v06_a_identity_manifest(cwd / experiment.identity_manifest, experiment=experiment)
    fit_ids = even_normal_ids(tuple(record.item_id for record in manifest.normal_fit), N_PER_CATEGORY)
    fit_map = {record.item_id: record for record in manifest.normal_fit}
    rows: list[dict[str, Any]] = []
    for item_id in fit_ids:
        loaded = load_identity_record(fit_map[item_id], dataset_root=mvtec_root, config=experiment)
        rows.append(
            {
                "category": "cable",
                "item_id": item_id,
                "rgb": np.asarray(loaded.rgb, dtype=np.uint8).copy(),
                "valid": np.asarray(loaded.valid, dtype=bool).copy(),
            }
        )
    if tuple(EXPECTED) != ("candle", "capsules", "cashew", "chewinggum"):
        raise ValueError("VisA pilot list drifted")
    for category in EXPECTED:
        inventory = load_visa_inventory(visa_root, category)
        chosen = even_normal_ids(tuple(_item_id(path) for path in inventory.normals), N_PER_CATEGORY)
        suffix = inventory.normals[0].suffix
        for item_id in chosen:
            record = _record(
                category=category,
                defect_type="good",
                item_id=item_id,
                image_relpath=str(
                    PurePosixPath(category) / "Data" / "Images" / "Normal" / f"{item_id}{suffix}"
                ),
                mask_relpath=None,
                role="normal_dev",
            )
            loaded = load_extra_identity(record, dataset_root=visa_root)
            rows.append(
                {
                    "category": category,
                    "item_id": item_id,
                    "rgb": np.asarray(loaded.rgb, dtype=np.uint8).copy(),
                    "valid": np.asarray(loaded.valid, dtype=bool).copy(),
                }
            )
    return tuple(rows)


def run_tokencut_normal_preview_from_cli(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cable_experiment: Path,
    output: Path,
    repository_root: Path,
    device: str | None = None,
) -> dict[str, Any]:
    from cadiff.v06_a.tokencut import load_official_dinov2_encoder

    rows = collect_pilot_rows(
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        cable_experiment=cable_experiment,
        repository_root=repository_root,
    )
    encoder = load_official_dinov2_encoder(device=device)

    def encode(image: np.ndarray, identity: str) -> np.ndarray:
        del identity
        return encoder.encode(image).tokens

    return run_tokencut_normal_preview(rows=rows, output=output, encode_fn=encode)

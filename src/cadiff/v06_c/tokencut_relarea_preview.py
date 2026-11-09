"""Diagnostic: TokenCut largest-CC finalize vs relative-area union.

One variable: keep CCs with |C| >= tau * |C_max|. Same frozen train/good
identities as tokencut_normals_20260918. Not m_proposal / m_final, not SAM,
not generate(). can is deferred.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from cadiff.provenance import write_json_atomic
from cadiff.v06_a.tokencut import (
    RELATIVE_AREA_TAU,
    _finalize_largest_component,
    _finalize_object,
    tokencut_bipartition_upsampled,
)
from cadiff.v06_c.tokencut_normal_preview import (
    DISPLAY_SIZE,
    FORBIDDEN_CATEGORIES,
    EncodeFn,
    _blend,
    _contained_output,
    _resize_mask,
    _resize_rgb,
    _stack_rows,
    collect_pilot_rows,
    color_ranked_components,
    rank_components,
)

SCHEMA = "v06_c_tokencut_relarea_preview_v1"
RELATIVE_TAU = RELATIVE_AREA_TAU
_OFFICIAL_COLOR = (40, 200, 90)
_TAU_COLOR = (40, 190, 210)
_EXTRA_COLOR = (240, 220, 40)
_SPECK_COLOR = (220, 40, 180)
_FONT = ImageFont.load_default()


def union_relative_components(
    raw: np.ndarray,
    valid: np.ndarray,
    *,
    tau: float = RELATIVE_TAU,
) -> NDArray[np.bool_]:
    return _finalize_object(raw, valid, tau=tau)


def require_frozen_identities(
    frozen: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None = None,
) -> tuple[tuple[str, str], ...]:
    identities = tuple((str(item["category"]), str(item["item_id"])) for item in frozen)
    if not identities:
        raise ValueError("frozen identity list is empty")
    for category, _item_id in identities:
        if category in FORBIDDEN_CATEGORIES:
            raise ValueError("can is deferred; TokenCut preview does not include MVTec AD 2")
    if rows is not None:
        got = tuple((str(row["category"]), str(row["item_id"])) for row in rows)
        if got != identities:
            raise ValueError("identity list drifted from the frozen TokenCut preview")
    return identities


def load_frozen_identities(path: Path) -> tuple[dict[str, str], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(payload["identities"])
    require_frozen_identities(rows)
    return tuple(
        {"category": str(item["category"]), "item_id": str(item["item_id"])} for item in rows
    )


def _row_image(
    rgb: np.ndarray,
    ranked_color: np.ndarray,
    official: np.ndarray,
    tau_mask: np.ndarray,
    extra: np.ndarray,
    specks: np.ndarray,
    title: str,
) -> NDArray[np.uint8]:
    display = DISPLAY_SIZE
    base = _resize_rgb(rgb, display)
    ranked_d = _resize_rgb(ranked_color, display)
    official_d = _resize_mask(official, display)
    tau_d = _resize_mask(tau_mask, display)
    extra_d = _resize_mask(extra, display)
    speck_d = _resize_mask(specks, display)
    ranked_on_rgb = base.copy()
    occupied = ranked_d.any(axis=2)
    ranked_on_rgb[occupied] = (
        0.35 * base[occupied].astype(np.float32) + 0.65 * ranked_d[occupied].astype(np.float32)
    ).astype(np.uint8)
    tiles = (
        base,
        ranked_on_rgb,
        _blend(base, _OFFICIAL_COLOR, official_d.astype(np.float32) * 0.70),
        _blend(base, _TAU_COLOR, tau_d.astype(np.float32) * 0.70),
        _blend(base, _EXTRA_COLOR, extra_d.astype(np.float32) * 0.80),
        _blend(base, _SPECK_COLOR, speck_d.astype(np.float32) * 0.80),
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
        f"{title}  rgb | ranked-CC | official | tau-union | extra-kept | still-dropped",
        fill=(20, 20, 20),
        font=_FONT,
    )
    return np.asarray(image, dtype=np.uint8)


def run_tokencut_relarea_preview(
    *,
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    output: Path,
    encode_fn: EncodeFn,
    identities_path: Path,
    tau: float = RELATIVE_TAU,
) -> dict[str, Any]:
    frozen = load_frozen_identities(identities_path)
    require_frozen_identities(frozen, rows)
    destination = _contained_output(output)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        category = str(row["category"])
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
            "source_identities": str(identities_path),
            "tau": tau,
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
        raw = np.asarray(tokencut_bipartition_upsampled(tokens), dtype=bool) & row["valid"]
        official = _finalize_largest_component(raw, row["valid"])
        tau_mask = union_relative_components(raw, row["valid"], tau=tau)
        extra = np.ascontiguousarray(tau_mask & ~official)
        specks = np.ascontiguousarray(raw & ~tau_mask)
        ranked = rank_components(raw)
        n_tau = int(sum(float(item.pixels) >= tau * float(ranked[0].pixels) for item in ranked)) if ranked else 0
        title = (
            f"{row['identity']}  n_cc={len(ranked)} n_tau={n_tau}  "
            f"official={int(official.sum())} tau={int(tau_mask.sum())} "
            f"extra={int(extra.sum())} drop={int(specks.sum())}"
        )
        ranked_color = color_ranked_components(ranked, raw.shape)
        panel = _row_image(row["rgb"], ranked_color, official, tau_mask, extra, specks, title)
        arm = reports.setdefault(row["category"], {"n": 0, "rows": []})
        arm["n"] += 1
        arm["rows"].append(
            {
                "identity": row["identity"],
                "n_cc": len(ranked),
                "n_tau": n_tau,
                "official_pixels": int(official.sum()),
                "tau_pixels": int(tau_mask.sum()),
                "extra_pixels": int(extra.sum()),
                "dropped_pixels": int(specks.sum()),
                "cc_areas": [item.pixels for item in ranked[:10]],
            }
        )
        sheets.setdefault(row["category"], []).append(panel)
        all_rows.append(panel)
    for category, panels in sheets.items():
        Image.fromarray(_stack_rows(panels, f"TokenCut relarea tau={tau} {category}")).save(
            destination / f"sheet_{category}.png"
        )
    Image.fromarray(_stack_rows(all_rows, f"TokenCut official vs tau={tau} union")).save(
        destination / "all.png"
    )
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "not_module_c": True,
        "not_official_generate": True,
        "not_m_proposal": True,
        "not_m_final": True,
        "tau": tau,
        "arm_names": list(reports),
        "arms": reports,
        "column_meanings": {
            "rgb": "frozen train/good normal",
            "ranked-CC": "raw TokenCut CCs by area",
            "official": "historical S_x^0 = largest CC + fill holes",
            "tau-union": "current official S_x^0 = CCs with |C| >= tau * |C_max| + fill holes",
            "extra-kept": "in tau-union but not official (recovered instances)",
            "still-dropped": "raw CCs below tau (specks)",
        },
    }
    write_json_atomic(destination / "summary.json", payload)
    return payload


def run_tokencut_relarea_preview_from_cli(
    *,
    identities_path: Path,
    mvtec_root: Path,
    visa_root: Path,
    cable_experiment: Path,
    output: Path,
    repository_root: Path,
    device: str | None = None,
    tau: float = RELATIVE_TAU,
) -> dict[str, Any]:
    from cadiff.v06_a.tokencut import load_official_dinov2_encoder

    frozen = load_frozen_identities(identities_path)
    rows = collect_pilot_rows(
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        cable_experiment=cable_experiment,
        repository_root=repository_root,
    )
    require_frozen_identities(frozen, rows)
    encoder = load_official_dinov2_encoder(device=device)

    def encode(image: np.ndarray, identity: str) -> np.ndarray:
        del identity
        return encoder.encode(image).tokens

    return run_tokencut_relarea_preview(
        rows=rows,
        output=output,
        encode_fn=encode,
        identities_path=identities_path,
        tau=tau,
    )

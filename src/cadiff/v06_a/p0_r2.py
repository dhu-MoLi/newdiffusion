"""CPU-only P0-R2 context teacher diagnostic.  No formal targets, no GPU."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .g0 import (
    G0Config,
    G0ScheduleRow,
    PreparedG0Row,
    SourceGeometry,
    _flatten_prepared,
    _heatmap,
    _overlay_mask,
    compute_g0_row,
    estimate_source_geometry,
    load_g0_config,
    prepare_g0_categories,
)
from .targets import local_to_world, rasterize_pixel_cell_overlap, square_dilate_24
from .teacher import (
    affine_from_candidate_table,
    build_context_ring_stencil,
    context_feature_maps,
    expected_candidate_table_occupancy,
    inverse_cdf_sample,
    score_context_affine,
    score_context_candidate_table,
    shared_raw_quantile,
    stable_geometry_softmax,
)

_EXPECTED = {
    "schema_version": "v06_a_p0_r2_f1",
    "status": "awaiting_human_review",
    "g0_config": "configs/diagnostics/v06_a_g0_rc2.yaml",
    "context_mode": "spatial_ring_stencil",
    "context_version": "v06_a_ctx_r2_f1",
    "lambda_geo": 1.0,
    "lambda_ctx": 2.0,
    "context_temperature": 0.35,
    "context_outer_px": 72,
    "context_gaps": [32, 48],
    "canonical_gap_px": 32,
    "context_stencil_spacing_px": 8,
    "context_min_observable_fraction": 0.75,
    "formal_targets_materialized": False,
    "training_authorized": False,
}


@dataclass(frozen=True)
class P0R2Config:
    path: Path
    g0: G0Config
    rows: tuple[G0ScheduleRow, ...]
    values: Mapping[str, object]


@dataclass(frozen=True)
class P0R2Artifact:
    destination: Path
    json_path: Path
    combined_panel_path: Path


def load_p0_r2_config(path: Path) -> P0R2Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P0-R2 config must be a mapping")
    if set(raw) != set(_EXPECTED):
        raise ValueError(
            "P0-R2 schema mismatch: "
            f"missing={sorted(set(_EXPECTED) - set(raw))}, extra={sorted(set(raw) - set(_EXPECTED))}"
        )
    for key, expected in _EXPECTED.items():
        if raw[key] != expected:
            raise ValueError(f"P0-R2 {key} must equal the frozen value {expected!r}")
    root = path.resolve().parents[2]
    g0 = load_g0_config(root / str(raw["g0_config"]))
    return P0R2Config(path=path.resolve(strict=True), g0=g0, rows=g0.rows, values=raw)


def _output_destination(*, output_root: Path, output_relpath: str) -> Path:
    if not output_root.is_dir():
        raise FileNotFoundError("P0-R2 output root does not exist")
    relative = PurePosixPath(output_relpath)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != output_relpath
        or tuple(relative.parts[:2]) != ("v06_a", "p0_r2")
    ):
        raise ValueError("P0-R2 output must be contained under v06_a/p0_r2")
    destination = output_root.resolve(strict=True).joinpath(*relative.parts)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"P0-R2 refuses to overwrite an existing output: {destination}")
    return destination


def _masked_ring(rgb: np.ndarray, source_geometry: SourceGeometry, gap: int) -> np.ndarray:
    stencil = build_context_ring_stencil(
        source_geometry.envelopes.score.local_bounds, gap=gap
    )
    points = np.rint(local_to_world(stencil, source_geometry.frame)).astype(np.int32)
    mask = np.zeros((512, 512), dtype=bool)
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < 512)
        & (points[:, 1] >= 0)
        & (points[:, 1] < 512)
    )
    mask[points[inside, 1], points[inside, 0]] = True
    dilated = square_dilate_24(mask)
    return _overlay_mask(rgb, dilated, (0, 255, 255), 0.45)


def _transfer(source_gt: np.ndarray, affine, target_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exact = rasterize_pixel_cell_overlap(source_gt, affine, target_valid)
    proposal = square_dilate_24(exact.mask) & target_valid
    return exact.mask, proposal


def compute_p0_r2_row(
    index: int,
    prepared: PreparedG0Row,
    source_geometry: SourceGeometry,
    *,
    raw_q: float,
    config: P0R2Config,
) -> dict[str, Any]:
    geo = compute_g0_row(
        index, prepared, source_geometry, raw_q=raw_q, config=config.g0
    )
    if geo.candidates is None or geo.scores is None or prepared.source.anomaly_gt is None:
        raise ValueError("P0-R2 requires a valid G0 geometry row before scoring context")
    source_features = context_feature_maps(prepared.source.rgb, prepared.source.valid)
    target_features = context_feature_maps(prepared.target.rgb, prepared.target.valid)
    shuffled = target_features.copy()
    rng = np.random.RandomState(int(prepared.schedule.seed) % (2**31 - 1))
    perm = rng.permutation(512 * 512)
    shuffled = shuffled.reshape(512 * 512, -1)[perm].reshape(512, 512, -1)
    gray = prepared.target.rgb.mean(axis=2, keepdims=True).astype(np.uint8)
    removed = context_feature_maps(np.repeat(gray, 3, axis=2), prepared.target.valid)
    gap_payload: dict[int, dict[str, Any]] = {}
    for gap in (32, 48):
        stencil = build_context_ring_stencil(
            source_geometry.envelopes.score.local_bounds,
            gap=gap,
            outer_px=int(config.values["context_outer_px"]),
            spacing=int(config.values["context_stencil_spacing_px"]),
        )
        context_distance = score_context_candidate_table(
            geo.candidates,
            stencil,
            source_geometry.frame,
            source_features,
            target_features,
            prepared.target.valid,
            candidate_chunk_size=config.g0.candidate_chunk_size,
            stencil_chunk_size=config.g0.cell_chunk_size,
        )
        joint = (
            float(config.values["lambda_geo"]) * geo.scores.distance
            + float(config.values["lambda_ctx"]) * context_distance
        )
        probabilities = stable_geometry_softmax(
            joint,
            geo.scores.valid,
            temperature=float(config.values["context_temperature"]),
        )
        sample = inverse_cdf_sample(probabilities, geo.scores.valid, raw_q)
        affine = affine_from_candidate_table(geo.candidates, sample.index)
        transferred, proposal = _transfer(
            prepared.source.anomaly_gt, affine, prepared.target.valid
        )
        occupancy = expected_candidate_table_occupancy(
            geo.candidates,
            source_geometry.envelopes.score,
            source_geometry.frame,
            probabilities,
            prepared.target.valid,
            candidate_chunk_size=config.g0.candidate_chunk_size,
            cell_chunk_size=config.g0.cell_chunk_size,
        )
        sampled_ctx = float(context_distance[sample.index])
        sampled_geo = float(geo.scores.distance[sample.index])
        shuffle_distance = score_context_affine(
            affine, stencil, source_geometry.frame, source_features, shuffled, prepared.target.valid
        )
        remove_distance = score_context_affine(
            affine, stencil, source_geometry.frame, source_features, removed, prepared.target.valid
        )
        gap_payload[gap] = {
            "probabilities": probabilities,
            "anchor_marginal": probabilities.reshape(1024, 72).sum(axis=1).reshape(32, 32),
            "occupancy": occupancy,
            "sample": sample,
            "affine": affine,
            "transferred": transferred,
            "proposal": proposal,
            "sampled_d_ctx": sampled_ctx,
            "sampled_d_geo": sampled_geo,
            "shuffle_d_ctx": float(shuffle_distance),
            "remove_d_ctx": float(remove_distance),
            "normalized_entropy": float(
                -(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum()
                / np.log(max(int(geo.scores.valid.sum()), 2))
            ),
            "sampled_valid_rank": int(sample.valid_rank),
            "valid_candidate_count": int(geo.scores.valid.sum()),
        }
    return {"geo": geo, "gaps": gap_payload}


def _cell(rgb: np.ndarray, title: str) -> Image.Image:
    image = Image.fromarray(rgb).resize((128, 128), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (128, 146), (0, 0, 0))
    canvas.paste(image, (0, 18))
    ImageDraw.Draw(canvas).text((2, 2), title, fill=(255, 255, 255))
    return canvas


def _render_row(payload: dict[str, Any]) -> Image.Image:
    geo = payload["geo"]
    gap32 = payload["gaps"][32]
    gap48 = payload["gaps"][48]
    source = geo.source.rgb.copy()
    if geo.source.anomaly_gt is not None:
        source = _overlay_mask(source, geo.source.anomaly_gt, (255, 0, 0), 0.7)
    target = geo.target.rgb.copy()
    geo_transfer = _overlay_mask(target.copy(), geo.transferred_gt, (255, 0, 0), 0.7)
    ctx32_transfer = _overlay_mask(target.copy(), gap32["transferred"], (255, 0, 0), 0.7)
    ctx48_transfer = _overlay_mask(target.copy(), gap48["transferred"], (255, 0, 0), 0.7)
    tiles = (
        _cell(source, "source+GT"),
        _cell(_masked_ring(geo.source.rgb, geo.source_geometry, 32), "gap32 ring"),
        _cell(_masked_ring(geo.source.rgb, geo.source_geometry, 48), "gap48 ring"),
        _cell(target, "target"),
        _cell(_heatmap(geo.anchor_marginal), "geo π_u"),
        _cell(_heatmap(gap32["anchor_marginal"]), "ctx32 π_u"),
        _cell(_heatmap(geo.expected_occupancy), "geo Ω"),
        _cell(_heatmap(gap32["occupancy"]), "ctx32 Ω"),
        _cell(geo_transfer, "geo GT"),
        _cell(ctx32_transfer, "ctx32 GT"),
        _cell(ctx48_transfer, "ctx48 GT"),
        _cell(_overlay_mask(target.copy(), gap32["proposal"], (0, 255, 0), 0.55), "ctx32 p*"),
    )
    width = sum(tile.width for tile in tiles)
    row = Image.new("RGB", (width, tiles[0].height + 20), (16, 16, 16))
    draw = ImageDraw.Draw(row)
    title = (
        f"{geo.index:02d} {geo.schedule.category}/{geo.schedule.defect_type} "
        f"src={geo.schedule.source_id} tgt={geo.schedule.target_id} "
        f"{'VALID' if geo.valid else 'INVALID'}"
    )
    draw.text((4, 2), title, fill=(255, 255, 255))
    offset = 0
    for tile in tiles:
        row.paste(tile, (offset, 20))
        offset += tile.width
    return row


def run_p0_r2(
    config: P0R2Config,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
) -> P0R2Artifact:
    destination = _output_destination(output_root=output_root, output_relpath=output_relpath)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    categories = prepare_g0_categories(
        config.g0, repository_root=repository_root.resolve(strict=True), dataset_root=dataset_root
    )
    ordered = _flatten_prepared(categories, config.g0)
    raw_quantiles = {
        seed: shared_raw_quantile(seed) for seed in dict.fromkeys(row.seed for row in config.rows)
    }
    source_cache: dict[tuple[str, str], SourceGeometry] = {}
    rows_out: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for index, (prepared, _references) in enumerate(ordered):
        key = (prepared.schedule.category, prepared.schedule.defect_type)
        if key not in source_cache:
            source_cache[key] = estimate_source_geometry(
                prepared.source,
                _references,
                min_stencil_observable_fraction=float(
                    config.g0.values["score_stencil_min_observable_fraction"]
                ),
            )
        payload = compute_p0_r2_row(
            index,
            prepared,
            source_cache[key],
            raw_q=raw_quantiles[prepared.schedule.seed],
            config=config,
        )
        panels.append(_render_row(payload))
        geo = payload["geo"]
        gap32 = payload["gaps"][32]
        gap48 = payload["gaps"][48]
        rows_out.append(
            {
                "row_index": index,
                "category": geo.schedule.category,
                "defect_type": geo.schedule.defect_type,
                "source_id": geo.schedule.source_id,
                "target_id": geo.schedule.target_id,
                "seed": geo.schedule.seed,
                "geo_valid": geo.valid,
                "geo_normalized_entropy": None,
                "ctx32_normalized_entropy": gap32["normalized_entropy"],
                "ctx48_normalized_entropy": gap48["normalized_entropy"],
                "ctx32_rank": f"{gap32['sampled_valid_rank']}/{gap32['valid_candidate_count']}",
                "ctx48_rank": f"{gap48['sampled_valid_rank']}/{gap48['valid_candidate_count']}",
                "ctx32_d_geo": gap32["sampled_d_geo"],
                "ctx32_d_ctx": gap32["sampled_d_ctx"],
                "ctx32_shuffle_d_ctx": gap32["shuffle_d_ctx"],
                "ctx32_remove_d_ctx": gap32["remove_d_ctx"],
                "component_material_boundary_review": "awaiting_human_review",
            }
        )

    combined = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "black")
    offset = 0
    for panel in panels:
        combined.paste(panel, (0, offset))
        offset += panel.height
    combined_path = destination / "p0_r2_combined_16x12.png"
    combined.save(combined_path)
    sidecar = {
        "schema_version": "v06_a_p0_r2_artifact_f1",
        "status": "awaiting_human_review",
        "scientific_status": "awaiting human review",
        "valid_count_is_not_visual_success": True,
        "formal_targets_materialized": False,
        "training_authorized": False,
        "formal_gpu_training": "NO-GO",
        "gpu_used": False,
        "context_mode": "spatial_ring_stencil",
        "context_version": "v06_a_ctx_r2_f1",
        "g0_evidence": "runs/v06_a/g0_rc2/f2c_support_sdfprofile_20260831",
        "wall_seconds": time.perf_counter() - started,
        "rows": rows_out,
    }
    json_path = destination / "p0_r2_diagnostic.json"
    json_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return P0R2Artifact(destination, json_path, combined_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Module-A RC2 CPU-only P0-R2 diagnostic")
    parser.add_argument("mode", choices=("run",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    arguments = parser.parse_args(argv)
    config = load_p0_r2_config(arguments.config)
    artifact = run_p0_r2(
        config,
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        output_relpath=arguments.output_relpath,
    )
    print(
        json.dumps(
            {
                "destination": str(artifact.destination),
                "json": str(artifact.json_path),
                "combined_panel": str(artifact.combined_panel_path),
                "status": "awaiting_human_review",
                "gpu_used": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Isolation runner for atlas label propagation. Does not write G0/P0-R2 outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import IdentityManifest, IdentityRecord, load_v06_a_identity_manifest

from .correspondence import (
    CollapseLimits,
    CorrespondenceResult,
    geometry_fallback_mask,
    match_feature_maps,
    select_used_mask,
    source_match_support,
    splat_mask,
    square_dilate,
)
from .features import FrozenDinoEncoder, downsample_mask
from .inpaint import inpaint_source
from .panel import render_row, write_combined_panel
from .support import rgb_support, target_support

_OUTPUT_PREFIX = ("v06_a", "diag_atlas_prop")
_SCHEMAS = {
    "v06_a_diag_atlas_prop_f1": {
        "source_match_dilate_px": 0,
        "wipe_on_collapse": True,
        "use_kept_splat": False,
        "splat_neighbor_radius": 0,
    },
    "v06_a_diag_atlas_prop_f2": {
        "source_match_dilate_px": 16,
        "wipe_on_collapse": True,
        "use_kept_splat": False,
        "splat_neighbor_radius": 0,
    },
    "v06_a_diag_atlas_prop_f3": {
        "source_match_dilate_px": 16,
        "wipe_on_collapse": False,
        "use_kept_splat": False,
        "splat_neighbor_radius": 0,
    },
    "v06_a_diag_atlas_prop_f4": {
        "source_match_dilate_px": 16,
        "wipe_on_collapse": False,
        "use_kept_splat": True,
        "splat_neighbor_radius": 1,
    },
    "v06_a_diag_atlas_prop_a0v": {
        "source_match_dilate_px": 16,
        "wipe_on_collapse": False,
        "use_kept_splat": True,
        "splat_neighbor_radius": 1,
    },
}


@dataclass(frozen=True)
class AtlasRow:
    category: str
    defect_type: str
    source_id: str
    source_role: str
    target_id: str
    target_role: str
    seed: int
    review_role: str


@dataclass(frozen=True)
class AtlasDiagConfig:
    schema_version: str
    status: str
    square_dilate_radius_px: int
    inpaint_radius: int
    dino_model: str
    source_match_dilate_px: int
    wipe_on_collapse: bool
    use_kept_splat: bool
    splat_neighbor_radius: int
    limits: CollapseLimits
    smooth_radius: int
    hazelnut_experiment: str
    cable_experiment: str
    rows: tuple[AtlasRow, ...]


def load_atlas_diag_config(path: Path) -> AtlasDiagConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = str(loaded["schema_version"])
    if schema not in _SCHEMAS:
        raise ValueError(f"schema_version must be one of {sorted(_SCHEMAS)}")
    frozen = _SCHEMAS[schema]
    source_match_dilate_px = int(loaded.get("source_match_dilate_px", frozen["source_match_dilate_px"]))
    if source_match_dilate_px != frozen["source_match_dilate_px"]:
        raise ValueError(f"{schema} freezes source_match_dilate_px={frozen['source_match_dilate_px']}")
    wipe_on_collapse = bool(loaded.get("wipe_on_collapse", frozen["wipe_on_collapse"]))
    if wipe_on_collapse != frozen["wipe_on_collapse"]:
        raise ValueError(f"{schema} freezes wipe_on_collapse={frozen['wipe_on_collapse']}")
    use_kept_splat = bool(loaded.get("use_kept_splat", frozen["use_kept_splat"]))
    if use_kept_splat != frozen["use_kept_splat"]:
        raise ValueError(f"{schema} freezes use_kept_splat={frozen['use_kept_splat']}")
    splat_neighbor_radius = int(loaded.get("splat_neighbor_radius", frozen["splat_neighbor_radius"]))
    if splat_neighbor_radius != frozen["splat_neighbor_radius"]:
        raise ValueError(f"{schema} freezes splat_neighbor_radius={frozen['splat_neighbor_radius']}")
    limits = CollapseLimits(
        max_cycle_fail_ratio=float(loaded["max_cycle_fail_ratio"]),
        min_valid_matches=int(loaded["min_valid_matches"]),
        max_median_displacement_px=float(loaded["max_median_displacement_px"]),
        wipe_on_collapse=wipe_on_collapse,
    )
    if (
        limits.max_cycle_fail_ratio != 0.45
        or limits.min_valid_matches != 16
        or limits.max_median_displacement_px != 256
    ):
        raise ValueError("atlas diagnostic freezes cycle limits 0.45 / 16 / 256")
    rows = tuple(
        AtlasRow(
            category=str(row["category"]),
            defect_type=str(row["defect_type"]),
            source_id=str(row["source_id"]),
            source_role=str(row["source_role"]),
            target_id=str(row["target_id"]),
            target_role=str(row["target_role"]),
            seed=int(row["seed"]),
            review_role=str(row["review_role"]),
        )
        for row in loaded["rows"]
    )
    if len(rows) != 16:
        raise ValueError("atlas diagnostic freezes exactly 16 rows")
    return AtlasDiagConfig(
        schema_version=schema,
        status=str(loaded["status"]),
        square_dilate_radius_px=int(loaded["square_dilate_radius_px"]),
        inpaint_radius=int(loaded["inpaint_radius"]),
        dino_model=str(loaded["dino_model"]),
        source_match_dilate_px=source_match_dilate_px,
        wipe_on_collapse=wipe_on_collapse,
        use_kept_splat=use_kept_splat,
        splat_neighbor_radius=splat_neighbor_radius,
        limits=limits,
        smooth_radius=int(loaded["smooth_radius"]),
        hazelnut_experiment=str(loaded["hazelnut_experiment"]),
        cable_experiment=str(loaded["cable_experiment"]),
        rows=rows,
    )


def _contained_output(output_root: Path, output_relpath: str) -> Path:
    root = output_root.resolve(strict=True)
    destination = root.joinpath(*PurePosixPath(output_relpath).parts)
    if tuple(PurePosixPath(output_relpath).parts[:2]) != _OUTPUT_PREFIX:
        raise ValueError("atlas diagnostic output must live under v06_a/diag_atlas_prop")
    resolved = destination.resolve(strict=False)
    resolved.relative_to(root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {destination}")
    return destination


def _lookup_source(manifest: IdentityManifest, row: AtlasRow) -> IdentityRecord:
    matches = tuple(
        record
        for record in manifest.fewshot
        if record.defect_type == row.defect_type and record.item_id == row.source_id
    )
    if len(matches) != 1:
        raise ValueError("atlas source must be the exact frozen fewshot record")
    return matches[0]


def _lookup_target(manifest: IdentityManifest, row: AtlasRow) -> IdentityRecord:
    matches = tuple(
        record
        for record in manifest.normal_dev
        if record.item_id == row.target_id and record.defect_type == "good"
    )
    if len(matches) != 1:
        raise ValueError("atlas target must be the exact frozen normal_dev record")
    return matches[0]


def run_atlas_diag(
    config: AtlasDiagConfig,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
    device: str | None = None,
) -> dict[str, Any]:
    destination = _contained_output(output_root, output_relpath)
    destination.mkdir(parents=True)
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    encoder = FrozenDinoEncoder(model_name=config.dino_model, device=device)
    manifests: dict[str, tuple[Any, IdentityManifest]] = {}
    for category, relative in (
        ("hazelnut", config.hazelnut_experiment),
        ("cable", config.cable_experiment),
    ):
        experiment_path = repository_root.joinpath(*PurePosixPath(relative).parts)
        experiment = load_v06_a_experiment(experiment_path)
        manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
        manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
        manifests[category] = (experiment, manifest)

    sidecar_rows: list[dict[str, Any]] = []
    panel_images = []
    for index, row in enumerate(config.rows):
        experiment, manifest = manifests[row.category]
        source = load_identity_record(
            _lookup_source(manifest, row), dataset_root=dataset_root, config=experiment
        )
        target = load_identity_record(
            _lookup_target(manifest, row), dataset_root=dataset_root, config=experiment
        )
        if source.anomaly_gt is None:
            raise ValueError("atlas source requires ground-truth mask")
        inpainted = inpaint_source(
            source.rgb, source.anomaly_gt, source.valid, radius=config.inpaint_radius
        )
        source_object = rgb_support(inpainted, source.valid)
        target_object = target_support(target, experiment)
        source_feat = encoder.encode(inpainted)
        target_feat = encoder.encode(target.rgb)
        source_match = source_match_support(
            source.anomaly_gt,
            source_object,
            source.valid,
            dilate_px=config.source_match_dilate_px,
        )
        source_grid = downsample_mask(
            source_match,
            height=source_feat.grid_height,
            width=source_feat.grid_width,
        )
        target_grid = downsample_mask(
            target_object & target.valid,
            height=target_feat.grid_height,
            width=target_feat.grid_width,
        )
        matched = match_feature_maps(
            source_feat.tokens,
            target_feat.tokens,
            source_grid,
            target_grid,
            limits=config.limits,
            smooth_radius=config.smooth_radius,
        )
        propagated = splat_mask(
            source.anomaly_gt,
            matched,
            neighbor_radius=config.splat_neighbor_radius,
        )
        geometry = geometry_fallback_mask(source.anomaly_gt, source_object, target_object)
        used, used_fallback = select_used_mask(
            propagated,
            geometry,
            collapsed=matched.collapsed,
            use_kept_splat=config.use_kept_splat,
        )
        proposal = square_dilate(used, config.square_dilate_radius_px) & target.valid
        if not used_fallback:
            status = "PROP" if not matched.collapsed else "KEEP"
        else:
            status = "FALLBACK"
        title = (
            f"{index:02d} {row.category}/{row.defect_type} "
            f"src={row.source_id} tgt={row.target_id} "
            f"{status} "
            f"cycle_fail={matched.cycle_fail_ratio:.2f} "
            f"n={matched.valid_match_count} {row.review_role}"
        )
        image = render_row(
            source_rgb=source.rgb,
            source_gt=source.anomaly_gt,
            inpaint_rgb=inpainted,
            target_rgb=target.rgb,
            flow_xy=matched.flow_xy,
            match_valid=matched.match_valid,
            propagated=propagated,
            proposal=proposal,
            geometry_mask=geometry,
            collapsed=matched.collapsed,
            title=title,
        )
        image.save(rows_dir / f"{index:02d}_{row.category}_{row.defect_type}.png")
        panel_images.append(image)
        sidecar_rows.append(
            {
                "index": index,
                "category": row.category,
                "defect_type": row.defect_type,
                "source_id": row.source_id,
                "target_id": row.target_id,
                "seed": row.seed,
                "review_role": row.review_role,
                "collapsed": matched.collapsed,
                "collapse_reason": matched.collapse_reason,
                "cycle_fail_ratio": matched.cycle_fail_ratio,
                "valid_match_count": matched.valid_match_count,
                "median_displacement_px": (
                    None
                    if not np.isfinite(matched.median_displacement_px)
                    else float(matched.median_displacement_px)
                ),
                "source_match_dilate_px": config.source_match_dilate_px,
                "wipe_on_collapse": config.wipe_on_collapse,
                "use_kept_splat": config.use_kept_splat,
                "splat_neighbor_radius": config.splat_neighbor_radius,
                "source_match_cells": int(source_grid.sum()),
                "propagated_pixels": int(propagated.sum()),
                "proposal_pixels": int(proposal.sum()),
                "used_geometry_fallback": used_fallback,
            }
        )
        print(title, flush=True)

    combined = destination / "atlas_prop_combined_16x8.png"
    write_combined_panel(panel_images, combined)
    summary = {
        "schema_version": config.schema_version,
        "status": "awaiting_human_review",
        "source_match_dilate_px": config.source_match_dilate_px,
        "wipe_on_collapse": config.wipe_on_collapse,
        "use_kept_splat": config.use_kept_splat,
        "splat_neighbor_radius": config.splat_neighbor_radius,
        "encoder": encoder.model_name,
        "destination": str(destination),
        "combined_panel": str(combined),
        "collapsed_rows": sum(1 for item in sidecar_rows if item["collapsed"]),
        "rows": sidecar_rows,
        "not_module_a": True,
        "not_g0_or_p0_r2": True,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolation atlas label-propagation diagnostic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_atlas_diag_config(arguments.config)
    summary = run_atlas_diag(
        config,
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        output_relpath=arguments.output_relpath,
        device=arguments.device,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    return 0

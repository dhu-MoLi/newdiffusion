"""P1-identity recheck of the atlas f4 candidate. Not Module A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import IdentityManifest, load_v06_a_identity_manifest

from .correspondence import (
    CollapseLimits,
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
from .runner import AtlasDiagConfig, AtlasRow, _lookup_source, _lookup_target
from .support import rgb_support, target_support

_OUTPUT_PREFIX = ("v06_a", "diag_atlas_prop")
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")
_PRESSURE_TYPES = frozenset({"cut_inner_insulation", "cut_outer_insulation"})
_FROZEN_F4 = {
    "source_match_dilate_px": 16,
    "wipe_on_collapse": False,
    "use_kept_splat": True,
    "splat_neighbor_radius": 1,
}


def build_p1_atlas_rows(
    *,
    hazelnut: tuple[Any, IdentityManifest],
    cable: tuple[Any, IdentityManifest],
) -> tuple[AtlasRow, ...]:
    rows: list[AtlasRow] = []
    for category, (_experiment, manifest) in (("hazelnut", hazelnut), ("cable", cable)):
        if len(manifest.p1_identities) != 32:
            raise ValueError("P1 recheck requires the frozen 32-identity schedule")
        for identity in manifest.p1_identities:
            sources = tuple(
                sorted(
                    (record for record in manifest.fewshot if record.defect_type == identity.defect_type),
                    key=lambda record: record.sample_id,
                )
            )
            if not sources:
                raise ValueError(f"missing frozen fewshot sources for {identity.defect_type}")
            source = sources[identity.seed % len(sources)]
            rows.append(
                AtlasRow(
                    category=category,
                    defect_type=identity.defect_type,
                    source_id=source.item_id,
                    source_role="fewshot",
                    target_id=identity.normal_id,
                    target_role="normal_dev",
                    seed=identity.seed,
                    review_role="pressure" if identity.defect_type in _PRESSURE_TYPES else "primary",
                )
            )
    if len(rows) != 64:
        raise ValueError("P1 recheck freezes exactly 64 rows")
    return tuple(rows)


def load_p1_recheck_config(path: Path, *, repository_root: Path) -> AtlasDiagConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if str(loaded["schema_version"]) != "v06_a_diag_atlas_prop_p1":
        raise ValueError("P1 recheck requires schema v06_a_diag_atlas_prop_p1")
    if "rows" in loaded:
        raise ValueError("P1 recheck rows must be built from frozen manifests, not listed in yaml")
    source_match_dilate_px = int(loaded.get("source_match_dilate_px", _FROZEN_F4["source_match_dilate_px"]))
    wipe_on_collapse = bool(loaded.get("wipe_on_collapse", _FROZEN_F4["wipe_on_collapse"]))
    use_kept_splat = bool(loaded.get("use_kept_splat", _FROZEN_F4["use_kept_splat"]))
    splat_neighbor_radius = int(loaded.get("splat_neighbor_radius", _FROZEN_F4["splat_neighbor_radius"]))
    if (
        source_match_dilate_px != _FROZEN_F4["source_match_dilate_px"]
        or wipe_on_collapse != _FROZEN_F4["wipe_on_collapse"]
        or use_kept_splat != _FROZEN_F4["use_kept_splat"]
        or splat_neighbor_radius != _FROZEN_F4["splat_neighbor_radius"]
    ):
        raise ValueError("P1 recheck freezes the f4 match rules")
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
    bundles = {}
    for category, relative in (
        ("hazelnut", str(loaded["hazelnut_experiment"])),
        ("cable", str(loaded["cable_experiment"])),
    ):
        experiment_path = repository_root.joinpath(*PurePosixPath(relative).parts)
        experiment = load_v06_a_experiment(experiment_path)
        manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
        manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
        bundles[category] = (experiment, manifest)
    rows = build_p1_atlas_rows(hazelnut=bundles["hazelnut"], cable=bundles["cable"])
    return AtlasDiagConfig(
        schema_version="v06_a_diag_atlas_prop_p1",
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


def p1_recheck_output_path(output_root: Path, output_relpath: str) -> Path:
    parts = PurePosixPath(output_relpath).parts
    if parts[:2] != _OUTPUT_PREFIX or len(parts) != 3:
        raise ValueError("P1 recheck output must live under v06_a/diag_atlas_prop/<name>")
    name = parts[2]
    if name.startswith(_FORBIDDEN_OUTPUT_PREFIXES):
        raise ValueError(f"refuses to write f1-f4 outputs: {name}")
    root = output_root.resolve(strict=True)
    destination = root.joinpath(*parts)
    resolved = destination.resolve(strict=False)
    resolved.relative_to(root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {destination}")
    return destination


def run_p1_recheck(
    config: AtlasDiagConfig,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
    device: str | None = None,
) -> dict[str, Any]:
    destination = p1_recheck_output_path(output_root, output_relpath)
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
            f"src={row.source_id} tgt={row.target_id} seed={row.seed} "
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
        image.save(rows_dir / f"{index:02d}_{row.category}_{row.defect_type}_{row.target_id}_{row.seed}.png")
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
                "used_geometry_fallback": used_fallback,
                "propagated_pixels": int(propagated.sum()),
                "proposal_pixels": int(proposal.sum()),
                "source_selected_by_seed_mod_n": True,
            }
        )
        print(title, flush=True)

    hazelnut_panel = destination / "atlas_prop_p1_hazelnut_32x8.png"
    cable_panel = destination / "atlas_prop_p1_cable_32x8.png"
    combined = destination / "atlas_prop_p1_combined_64x8.png"
    write_combined_panel(panel_images[:32], hazelnut_panel)
    write_combined_panel(panel_images[32:], cable_panel)
    write_combined_panel(panel_images, combined)
    summary = {
        "schema_version": config.schema_version,
        "status": "awaiting_human_review",
        "encoder": encoder.model_name,
        "destination": str(destination),
        "hazelnut_panel": str(hazelnut_panel),
        "cable_panel": str(cable_panel),
        "combined_panel": str(combined),
        "look_at": "blue proposal column; red is geometry fallback control",
        "collapsed_rows": sum(1 for item in sidecar_rows if item["collapsed"]),
        "fallback_rows": sum(1 for item in sidecar_rows if item["used_geometry_fallback"]),
        "rows": sidecar_rows,
        "not_module_a": True,
        "not_formal_completion": True,
        "did_not_overwrite_f1_f4": True,
        "used_open_report_or_test": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1-identity atlas blue-column recheck")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_p1_recheck(
        load_p1_recheck_config(arguments.config, repository_root=arguments.repository_root),
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        output_relpath=arguments.output_relpath,
        device=arguments.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Same-identity A0-V contrast for the atlas f4 candidate. Not Module A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import LoadedIdentity, load_identity_record
from cadiff.v06_a.identities import IdentityManifest, load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalAtlas, build_normal_atlas, estimate_normal_structure
from cadiff.v06_a.targets import construct_plain_target

from .correspondence import (
    CorrespondenceResult,
    match_feature_maps,
    select_used_mask,
    source_match_support,
    splat_mask,
    square_dilate,
)
from .features import FrozenDinoEncoder, downsample_mask
from .inpaint import inpaint_source
from .panel import _CELL, _overlay_mask, _to_image
from .runner import (
    AtlasDiagConfig,
    _lookup_source,
    _lookup_target,
    load_atlas_diag_config,
)
from .support import rgb_support, target_support

_OUTPUT_PREFIX = ("v06_a", "diag_atlas_prop")
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")
COMPARE_COLUMNS = (
    "source_gt",
    "target",
    "atlas_blue",
    "a0v_amber",
    "overlay",
)
_ATLAS_COLOR = (80, 160, 255)
_A0V_COLOR = (255, 170, 40)


def load_atlas_a0v_compare_config(path: Path) -> AtlasDiagConfig:
    config = load_atlas_diag_config(path)
    if config.schema_version != "v06_a_diag_atlas_prop_a0v":
        raise ValueError("A0-V compare requires schema v06_a_diag_atlas_prop_a0v")
    return config


def compare_output_path(output_root: Path, output_relpath: str) -> Path:
    parts = PurePosixPath(output_relpath).parts
    if parts[:2] != _OUTPUT_PREFIX or len(parts) != 3:
        raise ValueError("A0-V compare output must live under v06_a/diag_atlas_prop/<name>")
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


def mask_centroid(mask: NDArray[np.bool_]) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return (float("nan"), float("nan"))
    return (float(cols.mean()), float(rows.mean()))


def mask_iou(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    if left.shape != right.shape:
        raise ValueError("IoU masks must share one shape")
    union = int((left | right).sum())
    if union == 0:
        return 0.0
    return float((left & right).sum() / union)


def a0v_proposal_for_pair(
    source: LoadedIdentity,
    target: LoadedIdentity,
    *,
    source_atlas: NormalAtlas,
    target_structure,
    config,
    manifest: IdentityManifest,
) -> NDArray[np.bool_]:
    outcome = construct_plain_target(
        source,
        source_atlas,
        target,
        target_structure,
        config,
        manifest,
    )
    return outcome.proposal_512.copy()


def atlas_proposal_for_pair(
    source: LoadedIdentity,
    target: LoadedIdentity,
    *,
    encoder: FrozenDinoEncoder,
    config: AtlasDiagConfig,
    experiment,
) -> tuple[NDArray[np.bool_], CorrespondenceResult, bool]:
    if source.anomaly_gt is None:
        raise ValueError("atlas source requires ground-truth mask")
    inpainted = inpaint_source(source.rgb, source.anomaly_gt, source.valid, radius=config.inpaint_radius)
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
    geometry = np.zeros_like(source.anomaly_gt)
    used, used_fallback = select_used_mask(
        propagated,
        geometry,
        collapsed=matched.collapsed,
        use_kept_splat=config.use_kept_splat,
    )
    proposal = square_dilate(used, config.square_dilate_radius_px) & target.valid
    return proposal, matched, used_fallback


def render_compare_row(
    *,
    source_rgb: NDArray[np.uint8],
    source_gt: NDArray[np.bool_],
    target_rgb: NDArray[np.uint8],
    atlas_proposal: NDArray[np.bool_],
    a0v_proposal: NDArray[np.bool_],
    title: str,
) -> Image.Image:
    overlay = _overlay_mask(target_rgb, atlas_proposal, _ATLAS_COLOR)
    overlay = _overlay_mask(overlay, a0v_proposal, _A0V_COLOR)
    cells = [
        _to_image(_overlay_mask(source_rgb, source_gt, (255, 220, 40))),
        _to_image(target_rgb),
        _to_image(_overlay_mask(target_rgb, atlas_proposal, _ATLAS_COLOR)),
        _to_image(_overlay_mask(target_rgb, a0v_proposal, _A0V_COLOR)),
        _to_image(overlay),
    ]
    canvas = Image.new("RGB", (_CELL * len(COMPARE_COLUMNS), _CELL + 28), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 4), title, fill=(240, 240, 240), font=font)
    for index, cell in enumerate(cells):
        canvas.paste(cell, (index * _CELL, 28))
    return canvas


def _write_compare_panel(row_images: list[Image.Image], destination: Path) -> None:
    if not row_images:
        raise ValueError("compare panel requires at least one row")
    width = row_images[0].width
    height = sum(image.height for image in row_images) + 22
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 4), "  ".join(COMPARE_COLUMNS), fill=(200, 200, 200), font=font)
    top = 22
    for image in row_images:
        canvas.paste(image, (0, top))
        top += image.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _load_category_bundle(
    *,
    category: str,
    relative: str,
    repository_root: Path,
    dataset_root: Path,
) -> tuple[Any, IdentityManifest, NormalAtlas]:
    experiment_path = repository_root.joinpath(*PurePosixPath(relative).parts)
    experiment = load_v06_a_experiment(experiment_path)
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    references = tuple(
        load_identity_record(
            next(record for record in manifest.normal_fit if record.item_id == item_id),
            dataset_root=dataset_root,
            config=experiment,
        )
        for item_id in manifest.atlas_normal_ids
    )
    atlas = build_normal_atlas(references, config=experiment, identity_manifest=manifest)
    return experiment, manifest, atlas


def run_atlas_a0v_compare(
    config: AtlasDiagConfig,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
    device: str | None = None,
) -> dict[str, Any]:
    destination = compare_output_path(output_root, output_relpath)
    destination.mkdir(parents=True)
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    encoder = FrozenDinoEncoder(model_name=config.dino_model, device=device)
    bundles: dict[str, tuple[Any, IdentityManifest, NormalAtlas]] = {}
    for category, relative in (
        ("hazelnut", config.hazelnut_experiment),
        ("cable", config.cable_experiment),
    ):
        bundles[category] = _load_category_bundle(
            category=category,
            relative=relative,
            repository_root=repository_root,
            dataset_root=dataset_root,
        )

    sidecar_rows: list[dict[str, Any]] = []
    panel_images = []
    for index, row in enumerate(config.rows):
        experiment, manifest, source_atlas = bundles[row.category]
        source = load_identity_record(
            _lookup_source(manifest, row), dataset_root=dataset_root, config=experiment
        )
        target = load_identity_record(
            _lookup_target(manifest, row), dataset_root=dataset_root, config=experiment
        )
        atlas_proposal, matched, used_fallback = atlas_proposal_for_pair(
            source,
            target,
            encoder=encoder,
            config=config,
            experiment=experiment,
        )
        a0v_error = None
        a0v_proposal = np.zeros((512, 512), dtype=bool)
        try:
            target_structure = estimate_normal_structure(target, config=experiment)
            a0v_proposal = a0v_proposal_for_pair(
                source,
                target,
                source_atlas=source_atlas,
                target_structure=target_structure,
                config=experiment,
                manifest=manifest,
            )
        except ValueError as error:
            a0v_error = str(error)
        atlas_centroid = mask_centroid(atlas_proposal)
        a0v_centroid = mask_centroid(a0v_proposal)
        shift = (
            None
            if not (np.isfinite(atlas_centroid).all() and np.isfinite(a0v_centroid).all())
            else float(np.hypot(atlas_centroid[0] - a0v_centroid[0], atlas_centroid[1] - a0v_centroid[1]))
        )
        iou = mask_iou(atlas_proposal, a0v_proposal)
        title = (
            f"{index:02d} {row.category}/{row.defect_type} "
            f"src={row.source_id} tgt={row.target_id} "
            f"iou={iou:.2f} shift={0.0 if shift is None else shift:.1f} "
            f"{row.review_role}"
        )
        image = render_compare_row(
            source_rgb=source.rgb,
            source_gt=source.anomaly_gt,
            target_rgb=target.rgb,
            atlas_proposal=atlas_proposal,
            a0v_proposal=a0v_proposal,
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
                "used_geometry_fallback": used_fallback,
                "atlas_pixels": int(atlas_proposal.sum()),
                "a0v_pixels": int(a0v_proposal.sum()),
                "atlas_centroid_xy": [atlas_centroid[0], atlas_centroid[1]],
                "a0v_centroid_xy": [a0v_centroid[0], a0v_centroid[1]],
                "centroid_shift_px": shift,
                "proposal_iou": iou,
                "a0v_error": a0v_error,
                "same_source_as_f4": True,
                "a0v_used_seed_source_index": False,
            }
        )
        print(title, flush=True)

    combined = destination / "atlas_vs_a0v_combined_16x5.png"
    _write_compare_panel(panel_images, combined)
    summary = {
        "schema_version": config.schema_version,
        "status": "awaiting_human_review",
        "encoder": encoder.model_name,
        "destination": str(destination),
        "combined_panel": str(combined),
        "look_at": "column 3 atlas blue vs column 4 A0-V amber; 000->000 is control",
        "rows": sidecar_rows,
        "not_module_a": True,
        "not_formal_completion": True,
        "same_identities_as_f4": True,
        "did_not_overwrite_f1_f4": True,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-identity atlas vs A0-V compare")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_atlas_a0v_compare(
        load_atlas_a0v_compare_config(arguments.config),
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

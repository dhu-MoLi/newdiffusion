"""Fixed, diagnostic-only P0 gate and panel writer for Module A RC1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .config import V06AExperimentConfig, load_v06_a_experiment
from .data import LoadedIdentity, load_identity_record
from .identities import IdentityManifest, P0Pair, load_v06_a_identity_manifest
from .structure import (
    NormalAtlas,
    NormalStructure,
    build_normal_atlas,
    estimate_normal_structure,
)
from .teacher import TeacherPairDiagnostic, diagnose_teacher_pair

_CATEGORY_ORDER = ("hazelnut", "cable")
_COLUMNS = (
    "source anomaly",
    "GT",
    "masked context",
    "target normal",
    "pi heatmap",
    "sampled position",
    "transferred GT",
    "p*",
)
_BLIND_GAPS = (32, 48)
_CANONICAL_GAP = 32
_CELL_SIZE = 128
_HEATMAP_CELL_SIZE = 16
_EXPECTED_P0 = {
    "hazelnut": (
        ("crack", "000", "000", 202608280100, "fewshot"),
        ("crack", "001", "260", 202608280101, "fewshot"),
        ("cut", "000", "000", 202608280110, "fewshot"),
        ("cut", "001", "260", 202608280111, "fewshot"),
        ("hole", "000", "000", 202608280120, "fewshot"),
        ("hole", "001", "260", 202608280121, "fewshot"),
        ("print", "000", "000", 202608280130, "fewshot"),
        ("print", "001", "260", 202608280131, "fewshot"),
    ),
    "cable": (
        ("bent_wire", "000", "000", 202608280200, "fewshot"),
        ("bent_wire", "001", "152", 202608280201, "fewshot"),
        ("cut_inner_insulation", "000", "000", 202608280210, "fewshot"),
        ("cut_inner_insulation", "001", "152", 202608280211, "fewshot"),
        ("cut_outer_insulation", "000", "000", 202608280220, "fewshot"),
        ("cut_outer_insulation", "001", "152", 202608280221, "fewshot"),
        ("poke_insulation", "000", "000", 202608280230, "fewshot"),
        ("poke_insulation", "001", "152", 202608280231, "fewshot"),
    ),
}


@dataclass(frozen=True)
class PreparedP0Pair:
    pair: P0Pair
    source: LoadedIdentity
    target: LoadedIdentity
    target_structure: NormalStructure


@dataclass(frozen=True)
class P0CategoryBundle:
    config: V06AExperimentConfig
    manifest: IdentityManifest
    atlas: NormalAtlas
    pairs: tuple[PreparedP0Pair, ...]


@dataclass(frozen=True)
class P0Artifact:
    destination: Path
    json_path: Path
    combined_panel_path: Path
    pair_panel_paths: tuple[Path, ...]
    diagnostics: tuple[TeacherPairDiagnostic, ...]


def _contained_file(path: Path, *, root: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {name}: {path}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must remain within the repository root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    return resolved


def prepare_p0_category(
    *,
    experiment_path: Path,
    repository_root: Path,
    dataset_root: Path,
) -> P0CategoryBundle:
    """Load exactly the frozen P0 and atlas identities for one category."""

    if not repository_root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {repository_root}")
    root = repository_root.resolve(strict=True)
    experiment_file = _contained_file(
        experiment_path, root=root, name="P0 experiment configuration"
    )
    config = load_v06_a_experiment(experiment_file)
    manifest_file = _contained_file(
        root.joinpath(*PurePosixPath(config.identity_manifest).parts),
        root=root,
        name="P0 identity manifest",
    )
    manifest = load_v06_a_identity_manifest(manifest_file, experiment=config)
    atlas_records = tuple(
        next(record for record in manifest.normal_fit if record.item_id == item_id)
        for item_id in manifest.atlas_normal_ids
    )
    atlas_references = tuple(
        load_identity_record(record, dataset_root=dataset_root, config=config)
        for record in atlas_records
    )
    atlas = build_normal_atlas(atlas_references, config=config, identity_manifest=manifest)

    target_cache: dict[str, tuple[LoadedIdentity, NormalStructure]] = {}
    prepared: list[PreparedP0Pair] = []
    for pair in manifest.p0_pairs:
        source_record = next(
            record
            for record in manifest.fewshot
            if record.defect_type == pair.defect_type and record.item_id == pair.source_id
        )
        source = load_identity_record(source_record, dataset_root=dataset_root, config=config)
        if pair.target_normal_id not in target_cache:
            target_record = next(
                record
                for record in (*manifest.normal_fit, *manifest.normal_dev)
                if record.item_id == pair.target_normal_id
            )
            target = load_identity_record(target_record, dataset_root=dataset_root, config=config)
            target_cache[pair.target_normal_id] = (
                target,
                estimate_normal_structure(target, config=config),
            )
        target, structure = target_cache[pair.target_normal_id]
        prepared.append(PreparedP0Pair(pair, source, target, structure))
    bundle = P0CategoryBundle(config, manifest, atlas, tuple(prepared))
    _validate_bundle(bundle, expected_category=config.category)
    return bundle


def _validate_bundle(bundle: P0CategoryBundle, *, expected_category: str) -> None:
    config, manifest = bundle.config, bundle.manifest
    if (
        config.category != expected_category
        or manifest.category != expected_category
        or manifest.defect_types != config.defect_types
    ):
        raise ValueError("P0 bundle category and vocabulary must match the fixed category order")
    teacher_config = (
        config.teacher_outer_radius,
        tuple(config.teacher_blind_gaps),
        config.teacher_candidate_stride,
        config.teacher_temperature,
        config.teacher_relation_weight,
        config.teacher_min_context_fraction,
        config.p0_pairs_per_category,
    )
    if teacher_config != (72, (32, 48), 16, 0.25, 2.0, 0.75, 8):
        raise ValueError("P0 requires the frozen teacher configuration")
    observed_schedule = tuple(
        (
            pair.defect_type,
            pair.source_id,
            pair.target_normal_id,
            pair.seed,
            pair.source_role,
        )
        for pair in manifest.p0_pairs
    )
    if observed_schedule != _EXPECTED_P0[expected_category]:
        raise ValueError("P0 manifest must preserve the exact frozen P0 schedule")
    if len(bundle.pairs) != 8 or tuple(item.pair for item in bundle.pairs) != manifest.p0_pairs:
        raise ValueError("P0 bundle pairs must preserve the exact frozen P0 order")
    if not bundle.atlas.matches_manifest(manifest):
        raise ValueError("P0 bundle atlas must match its frozen identity manifest")
    for prepared, expected_pair in zip(bundle.pairs, manifest.p0_pairs, strict=True):
        expected_source = next(
            record
            for record in manifest.fewshot
            if record.defect_type == expected_pair.defect_type
            and record.item_id == expected_pair.source_id
        )
        expected_target = next(
            record
            for record in (*manifest.normal_fit, *manifest.normal_dev)
            if record.item_id == expected_pair.target_normal_id
        )
        if prepared.source.record != expected_source:
            raise ValueError("P0 bundle must contain the exact frozen P0 source identities")
        if prepared.target.record != expected_target:
            raise ValueError("P0 bundle must contain the exact frozen train-normal targets")
        if prepared.source.record.sample_id == prepared.target.record.sample_id:
            raise ValueError("P0 source and target identities must differ")
        if prepared.target_structure.normal.record != prepared.target.record or not np.array_equal(
            prepared.target_structure.valid, prepared.target.valid
        ):
            raise ValueError("P0 target structure must belong to its frozen target identity")


def _output_destination(*, output_root: Path, output_relpath: str) -> Path:
    if not output_root.is_dir():
        raise FileNotFoundError(f"P0 output root does not exist: {output_root}")
    root = output_root.resolve(strict=True)
    if not isinstance(output_relpath, str) or not output_relpath:
        raise ValueError("P0 output must be a caller-specified contained relative path")
    relative = PurePosixPath(output_relpath)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != output_relpath
    ):
        raise ValueError("P0 output must be a caller-specified contained relative path")
    destination = root.joinpath(*relative.parts)
    resolved = destination.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("P0 output must be a caller-specified contained relative path") from error
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"P0 refuses to overwrite an existing output: {destination}")
    return destination


def _blank() -> np.ndarray:
    return np.zeros((512, 512, 3), dtype=np.uint8)


def _mask_rgb(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return _blank()
    values = mask.astype(np.uint8) * 255
    return np.repeat(values[..., None], 3, axis=2)


def _canonical_distribution(diagnostic: TeacherPairDiagnostic):
    return next(
        (item.distribution for item in diagnostic.gaps if item.gap == diagnostic.canonical_gap),
        None,
    )


def _masked_context(diagnostic: TeacherPairDiagnostic) -> np.ndarray:
    distribution = _canonical_distribution(diagnostic)
    if distribution is None:
        return _blank()
    result = _blank()
    used = distribution.source_descriptor.used_mask
    result[used] = diagnostic.source.rgb[used]
    return result


def _heatmap(diagnostic: TeacherPairDiagnostic) -> np.ndarray:
    distribution = _canonical_distribution(diagnostic)
    if distribution is None:
        return _blank()
    values = np.zeros((512, 512), dtype=np.float64)
    maximum = float(distribution.probabilities.max())
    for (x, y), probability in zip(
        distribution.candidates, distribution.probabilities, strict=True
    ):
        half_cell = _HEATMAP_CELL_SIZE // 2
        left, top = max(int(x) - half_cell, 0), max(int(y) - half_cell, 0)
        right = min(int(x) + half_cell, 512)
        bottom = min(int(y) + half_cell, 512)
        values[top:bottom, left:right] = float(probability / maximum)
    result = np.zeros((512, 512, 3), dtype=np.uint8)
    result[..., 0] = np.rint(values * 255).astype(np.uint8)
    result[..., 2] = np.rint((1.0 - values) * 96).astype(np.uint8)
    result[values == 0] = 0
    return result


def _sampled_position(diagnostic: TeacherPairDiagnostic) -> np.ndarray:
    result = diagnostic.target.rgb.copy()
    if diagnostic.sampled_position is None:
        return result
    x, y = diagnostic.sampled_position
    left, right = max(x - 7, 0), min(x + 8, 512)
    top, bottom = max(y - 7, 0), min(y + 8, 512)
    stroke_left, stroke_right = max(x - 2, 0), min(x + 3, 512)
    stroke_top, stroke_bottom = max(y - 2, 0), min(y + 3, 512)
    result[top:bottom, stroke_left:stroke_right] = (255, 255, 0)
    result[stroke_top:stroke_bottom, left:right] = (255, 255, 0)
    return result


def _cell(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).resize((_CELL_SIZE, _CELL_SIZE), resample=Image.Resampling.NEAREST)


def _pair_panel(diagnostic: TeacherPairDiagnostic) -> Image.Image:
    columns = (
        diagnostic.source.rgb,
        _mask_rgb(diagnostic.source.anomaly_gt),
        _masked_context(diagnostic),
        diagnostic.target.rgb,
        _heatmap(diagnostic),
        _sampled_position(diagnostic),
        _mask_rgb(diagnostic.transferred_gt),
        _mask_rgb(diagnostic.proposal),
    )
    panel = Image.new("RGB", (_CELL_SIZE * len(_COLUMNS), _CELL_SIZE))
    for index, rgb in enumerate(columns):
        panel.paste(_cell(rgb), (index * _CELL_SIZE, 0))
    return panel


def _failed_diagnostic(
    prepared: PreparedP0Pair, *, category: str, error: ValueError
) -> TeacherPairDiagnostic:
    return TeacherPairDiagnostic(
        category=category,
        pair=prepared.pair,
        source=prepared.source,
        target=prepared.target,
        canonical_gap=_CANONICAL_GAP,
        gaps=(),
        gap_js_divergence=None,
        sampled_position=None,
        sampled_object_center_override=None,
        transferred_bbox_center=None,
        sampled_retained_fraction=None,
        transferred_gt=None,
        proposal=None,
        valid=False,
        error=f"teacher diagnostic failed: {error}",
    )


def _validate_diagnostic(
    diagnostic: TeacherPairDiagnostic,
    prepared: PreparedP0Pair,
    *,
    category: str,
) -> None:
    if (
        diagnostic.category != category
        or diagnostic.pair != prepared.pair
        or diagnostic.source.record != prepared.source.record
        or diagnostic.target.record != prepared.target.record
        or diagnostic.canonical_gap != _CANONICAL_GAP
        or tuple(item.gap for item in diagnostic.gaps) not in {_BLIND_GAPS, ()}
    ):
        raise ValueError("P0 diagnostic identity or frozen gap schedule mismatch")
    if diagnostic.valid and (
        diagnostic.sampled_position is None
        or diagnostic.sampled_object_center_override is None
        or diagnostic.transferred_bbox_center is None
        or abs(diagnostic.sampled_position[0] - diagnostic.transferred_bbox_center[0]) > 1.0
        or abs(diagnostic.sampled_position[1] - diagnostic.transferred_bbox_center[1]) > 1.0
    ):
        raise ValueError("P0 valid diagnostic must align sampled u and transferred GT centre")


def _gap_json(diagnostic: TeacherPairDiagnostic) -> list[dict[str, object]]:
    indexed = {item.gap: item for item in diagnostic.gaps}
    return [
        {
            "auxiliary_diagnostic_only": True,
            "entropy": None if gap not in indexed else indexed[gap].entropy,
            "gap": gap,
            "valid_candidate_count": (
                0 if gap not in indexed else indexed[gap].valid_candidate_count
            ),
        }
        for gap in _BLIND_GAPS
    ]


def _pair_json(
    diagnostic: TeacherPairDiagnostic, *, index: int, panel_relpath: str
) -> dict[str, object]:
    position = (
        None
        if diagnostic.sampled_position is None
        else {"x": diagnostic.sampled_position[0], "y": diagnostic.sampled_position[1]}
    )
    override = (
        None
        if diagnostic.sampled_object_center_override is None
        else {
            "x": diagnostic.sampled_object_center_override[0],
            "y": diagnostic.sampled_object_center_override[1],
        }
    )
    transferred_center = (
        None
        if diagnostic.transferred_bbox_center is None
        else {
            "x": diagnostic.transferred_bbox_center[0],
            "y": diagnostic.transferred_bbox_center[1],
        }
    )
    return {
        "auxiliary_diagnostics_only": True,
        "category": diagnostic.category,
        "defect_type": diagnostic.pair.defect_type,
        "error": diagnostic.error,
        "gap_distribution_stability_js": diagnostic.gap_js_divergence,
        "gaps": _gap_json(diagnostic),
        "index": index,
        "pair_panel_relpath": panel_relpath,
        "retained_fraction": diagnostic.sampled_retained_fraction,
        "sampled_object_center_override": override,
        "sampled_position": position,
        "seed": diagnostic.pair.seed,
        "source_id": diagnostic.pair.source_id,
        "source_role": diagnostic.source.record.role,
        "source_sample_id": diagnostic.source.record.sample_id,
        "target_normal_id": diagnostic.pair.target_normal_id,
        "target_role": diagnostic.target.record.role,
        "target_sample_id": diagnostic.target.record.sample_id,
        "transferred_gt_bbox_center": transferred_center,
        "valid": diagnostic.valid,
    }


def run_fixed_p0(
    bundles: tuple[P0CategoryBundle, ...],
    *,
    output_root: Path,
    output_relpath: str,
) -> P0Artifact:
    """Write a fixed 16-row diagnostic gate; never authorise or materialise A1-CTX."""

    if len(bundles) != 2:
        raise ValueError("P0 requires exactly Hazelnut and Cable category bundles")
    for bundle, category in zip(bundles, _CATEGORY_ORDER, strict=True):
        _validate_bundle(bundle, expected_category=category)
    destination = _output_destination(output_root=output_root, output_relpath=output_relpath)

    diagnostics: list[TeacherPairDiagnostic] = []
    for bundle in bundles:
        for prepared in bundle.pairs:
            try:
                diagnostic = diagnose_teacher_pair(
                    prepared.pair,
                    prepared.source,
                    bundle.atlas,
                    prepared.target,
                    prepared.target_structure,
                    config=bundle.config,
                    identity_manifest=bundle.manifest,
                )
            except ValueError as error:
                diagnostic = _failed_diagnostic(
                    prepared, category=bundle.config.category, error=error
                )
            _validate_diagnostic(diagnostic, prepared, category=bundle.config.category)
            diagnostics.append(diagnostic)
    if len(diagnostics) != 16:
        raise AssertionError("frozen P0 schedule must produce exactly 16 rows")

    rows = tuple(_pair_panel(diagnostic) for diagnostic in diagnostics)
    combined = Image.new("RGB", (_CELL_SIZE * len(_COLUMNS), _CELL_SIZE * 16))
    for index, row in enumerate(rows):
        combined.paste(row, (0, index * _CELL_SIZE))

    destination.mkdir(parents=True, exist_ok=False)
    pairs_directory = destination / "pairs"
    pairs_directory.mkdir()
    pair_paths: list[Path] = []
    pair_json: list[dict[str, object]] = []
    for index, (diagnostic, row) in enumerate(zip(diagnostics, rows, strict=True)):
        filename = (
            f"{index:02d}_{diagnostic.category}_{diagnostic.pair.defect_type}_"
            f"{diagnostic.pair.source_id}_{diagnostic.pair.target_normal_id}.png"
        )
        path = pairs_directory / filename
        row.save(path, format="PNG", optimize=False, compress_level=9)
        pair_paths.append(path)
        pair_json.append(_pair_json(diagnostic, index=index, panel_relpath=f"pairs/{filename}"))
    combined_path = destination / "p0_combined_16x8.png"
    combined.save(combined_path, format="PNG", optimize=False, compress_level=9)
    gate = {
        "a1_ctx_targets_materialized": False,
        "a1_ctx_training_authorized": False,
        "artifacts_are_diagnostic_only": True,
        "auxiliary_metrics_are_not_acceptance_criteria": True,
        "blind_gaps": list(_BLIND_GAPS),
        "canonical_gap": _CANONICAL_GAP,
        "category_order": list(_CATEGORY_ORDER),
        "columns": list(_COLUMNS),
        "combined_panel_relpath": combined_path.name,
        "pair_count": 16,
        "pairs": pair_json,
        "panel_cell_size": _CELL_SIZE,
        "review_required": "total_commander_manual_visual_review",
        "schema_version": "v06_a_p0_diagnostic_rc1",
        "status": "pending_review",
        "visual_judgment_required": True,
    }
    json_path = destination / "p0_gate.json"
    json_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return P0Artifact(
        destination,
        json_path,
        combined_path,
        tuple(pair_paths),
        tuple(diagnostics),
    )

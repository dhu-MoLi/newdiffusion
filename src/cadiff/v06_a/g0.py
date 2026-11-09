"""CPU-only RC2 geometry pre-diagnostic for Module A.

This module is intentionally independent from the frozen RC1 P0 writer.  It
never constructs a formal training target and has no CUDA or training import.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from yaml.constructor import ConstructorError

from .config import V06AExperimentConfig, load_v06_a_experiment
from .data import LoadedIdentity, load_identity_record
from .identities import IdentityManifest, IdentityRecord, load_v06_a_identity_manifest
from .structure import (
    BlindReferenceMasks,
    GeometryField,
    LocalFrame,
    NormalStructure,
    ReferenceGeometryInvalid,
    ReferenceGeometryMatch,
    ReferenceProxy,
    aggregate_reference_proxy,
    build_blind_reference_masks,
    estimate_geometry_field,
    estimate_local_frame_from_observable_ring,
    estimate_normal_structure,
    geometry_grid_centers_512,
    geometry_observable_128,
    geometry_sample_observability,
    match_reference_geometry,
    physical_sdf_px_512,
    require_stencil_observable_coverage,
    sample_continuous_field,
)
from .targets import (
    CoarseEnvelopes,
    GeometryAffine,
    build_coarse_envelopes,
    downsample_any_4x4,
    local_box_corners,
    local_to_world,
    rasterize_pixel_cell_overlap,
    square_dilate_24,
    transform_points,
)
from .teacher import (
    GeometryCandidateTable,
    GeometryScores,
    QuantileSample,
    _candidate_slot_order,
    _sample_field,
    affine_from_candidate_table,
    enumerate_geometry_candidates,
    expected_candidate_table_occupancy,
    inverse_cdf_sample,
    orientation_evidence_mass,
    score_geometry_candidate_table,
    shared_raw_quantile,
    stable_geometry_softmax,
)


@dataclass(frozen=True)
class G0ScheduleRow:
    category: str
    defect_type: str
    source_id: str
    source_role: str
    target_id: str
    target_role: str
    seed: int


@dataclass(frozen=True)
class G0Config:
    path: Path
    status: str
    rows: tuple[G0ScheduleRow, ...]
    candidate_chunk_size: int
    cell_chunk_size: int
    nominal_candidates_per_row: int
    values: Mapping[str, object]


@dataclass(frozen=True)
class PreparedReferenceSet:
    normals: tuple[LoadedIdentity, ...]
    structures: tuple[NormalStructure, ...]
    rgb_fields: tuple[GeometryField, ...]
    geometry_fields: tuple[GeometryField, ...]
    supports: np.ndarray
    physical_sdf_128: np.ndarray
    observable_128: np.ndarray
    identities: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.normals) != 16 or len(self.structures) != 16:
            raise ValueError("G0 reference set must contain exactly 16 normals")
        if len(self.rgb_fields) != 16 or len(self.geometry_fields) != 16:
            raise ValueError("G0 reference set must contain exactly 16 geometry fields")
        if self.supports.shape != (16, 512, 512) or self.supports.dtype != bool:
            raise ValueError("G0 reference supports must be bool [16,512,512]")
        if self.physical_sdf_128.shape != (16, 128, 128):
            raise ValueError("G0 reference physical SDF must be [16,128,128]")
        if self.observable_128.shape != (128, 128) or self.observable_128.dtype != bool:
            raise ValueError("G0 common reference observability must be bool [128,128]")
        self.supports.setflags(write=False)
        self.physical_sdf_128.setflags(write=False)
        self.observable_128.setflags(write=False)


@dataclass(frozen=True)
class PreparedG0Row:
    schedule: G0ScheduleRow
    source: LoadedIdentity
    target: LoadedIdentity
    target_structure: NormalStructure


@dataclass(frozen=True)
class PreparedG0Category:
    experiment: V06AExperimentConfig
    manifest: IdentityManifest
    references: PreparedReferenceSet
    rows: tuple[PreparedG0Row, ...]


@dataclass(frozen=True)
class SourceGeometry:
    source: LoadedIdentity
    masks: BlindReferenceMasks
    match: ReferenceGeometryMatch
    proxy: ReferenceProxy
    frame: LocalFrame
    envelopes: CoarseEnvelopes
    source_rgb_field: GeometryField
    stencil_observable: np.ndarray

    def __post_init__(self) -> None:
        mask = np.array(self.stencil_observable, dtype=bool, copy=True)
        if mask.ndim != 1 or mask.size == 0:
            raise ValueError("source stencil observability must be a non-empty bool vector")
        if mask.shape[0] != self.envelopes.score.stencil_local.shape[0]:
            raise ValueError("source stencil observability must match E_score")
        mask.setflags(write=False)
        object.__setattr__(self, "stencil_observable", mask)


@dataclass(frozen=True)
class SourceGeometryFailure:
    """Target-independent source failure evidence retained for both paired rows."""

    stage: str
    reason: str
    masks: BlindReferenceMasks | None
    source_rgb_field: GeometryField | None
    match: ReferenceGeometryMatch | None = None
    proxy: ReferenceProxy | None = None
    frame: LocalFrame | None = None
    envelopes: CoarseEnvelopes | None = None
    source_ring_mean_energy: float | None = None

    def __post_init__(self) -> None:
        if self.source_ring_mean_energy is not None and not math.isfinite(
            self.source_ring_mean_energy
        ):
            raise ValueError("source failure energy evidence must be finite")


class _SourceGeometryInvalid(ValueError):
    def __init__(self, evidence: SourceGeometryFailure) -> None:
        super().__init__(evidence.reason)
        self.evidence = evidence


@dataclass(frozen=True)
class G0RowResult:
    index: int
    schedule: G0ScheduleRow
    source: LoadedIdentity
    target: LoadedIdentity
    target_structure: NormalStructure
    source_geometry: SourceGeometry | None
    source_failure: SourceGeometryFailure | None
    target_field: GeometryField | None
    candidates: GeometryCandidateTable | None
    scores: GeometryScores | None
    probabilities: np.ndarray
    anchor_marginal: np.ndarray
    expected_occupancy: np.ndarray
    sample: QuantileSample | None
    sampled_affine: GeometryAffine | None
    transferred_gt: np.ndarray | None
    proposal_512: np.ndarray | None
    proposal_128: np.ndarray | None
    transformed_envelope_corners: np.ndarray | None
    target_physical_sdf_128: np.ndarray | None
    raw_q: float
    timings: Mapping[str, float]
    valid: bool
    error: str | None
    failure_stage: str | None = None
    completed_stage: str = "not_started"

    def __post_init__(self) -> None:
        if not math.isfinite(self.raw_q) or not 0.0 <= self.raw_q < 1.0:
            raise ValueError("G0 row raw quantile must be finite in [0,1)")
        if (self.source_geometry is None) == (self.source_failure is None):
            raise ValueError("G0 row requires exactly one source geometry or failure evidence")
        if self.valid and (self.failure_stage is not None or self.error is not None):
            raise ValueError("valid G0 rows cannot carry a failure stage or error")
        if not self.valid and (not self.failure_stage or self.error is None):
            raise ValueError("invalid G0 rows require a failure stage and error")
        if (self.target_field is None) != (self.target_physical_sdf_128 is None):
            raise ValueError("target field and physical SDF must be defined together")
        for value in (self.probabilities, self.anchor_marginal, self.expected_occupancy):
            if not np.isfinite(value).all():
                raise ValueError("G0 row numeric output must be finite")
            value.setflags(write=False)
        for value in (
            self.transferred_gt,
            self.proposal_512,
            self.proposal_128,
            self.transformed_envelope_corners,
            self.target_physical_sdf_128,
        ):
            if value is not None:
                value.setflags(write=False)


@dataclass(frozen=True)
class G0Artifact:
    destination: Path
    json_path: Path
    combined_panel_path: Path
    legend_path: Path
    row_panel_paths: tuple[Path, ...]
    numeric_paths: tuple[Path, ...]
    results: tuple[G0RowResult, ...]


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key ({key!r})",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_EXPECTED_KEYS = {
    "schema_version",
    "status",
    "canonical_gap_px",
    "blind_bbox_quantum_px",
    "blind_safety_px",
    "filter_guard_px",
    "reference_outer_ring_px",
    "reference_temperature",
    "reference_min_coverage",
    "reference_min_cells_128",
    "reference_count",
    "source_ring_min_mean_energy",
    "reference_distance_weights",
    "field_grid",
    "field_grid_phase_px",
    "tensor_radii_128",
    "tensor_weights",
    "tensor_lengths_px",
    "numeric_epsilon",
    "boundary_epsilon",
    "energy_gate",
    "evidence_floor",
    "orientation_floor",
    "scale_fallback_px",
    "scale_min_px",
    "scale_max_px",
    "sdf_channel_weight",
    "sdf_normalization_px",
    "support_distance_mad_k",
    "support_close_px",
    "sdf_relation_reduction",
    "cover_quantum_px",
    "score_seed_quantum_px",
    "score_quantum_px",
    "score_dilation_cells",
    "score_stencil_spacing_px",
    "score_stencil_min_observable_fraction",
    "anchor_stride_px",
    "angle_residuals",
    "scale_multipliers",
    "source_target_scale_ratio",
    "nominal_candidates_per_row",
    "candidate_order",
    "candidate_chunk_size",
    "cell_chunk_size",
    "geo_weights",
    "geo_temperature",
    "relation_band",
    "exact_gt_required_retained",
    "allowed_lost_raster_pixels",
    "square_dilate_radius_px",
    "sampling_rng",
    "sampling_method",
    "geometry_science_lock_version",
    "context_mode",
    "context_version",
    "recovery_mode",
    "recovery_version",
    "hazelnut_experiment",
    "cable_experiment",
    "rows",
}

_EXPECTED_CONSTANTS: dict[str, object] = {
    "schema_version": "v06_a_g0_rc2_f3",
    "status": "awaiting_human_review",
    "canonical_gap_px": 32,
    "blind_bbox_quantum_px": 8,
    "blind_safety_px": 8,
    "filter_guard_px": 20,
    "reference_outer_ring_px": 96,
    "reference_temperature": 0.25,
    "reference_min_coverage": 0.75,
    "reference_min_cells_128": 64,
    "reference_count": 16,
    "source_ring_min_mean_energy": 0.10,
    "reference_distance_weights": [0.5, 0.25, 0.125],
    "field_grid": 128,
    "field_grid_phase_px": 2,
    "tensor_radii_128": [1, 2, 4],
    "tensor_weights": [0.5, 0.3, 0.2],
    "tensor_lengths_px": [12, 20, 36],
    "numeric_epsilon": 1e-12,
    "boundary_epsilon": 1e-9,
    "energy_gate": 2.0**-12,
    "evidence_floor": 0.10,
    "orientation_floor": 0.10,
    "scale_fallback_px": 20.0,
    "scale_min_px": 4.0,
    "scale_max_px": 64.0,
    "sdf_channel_weight": 0.5,
    "sdf_normalization_px": 64,
    "support_distance_mad_k": 3.5,
    "support_close_px": 4,
    "sdf_relation_reduction": "occupied_bin_mean",
    "cover_quantum_px": 4,
    "score_seed_quantum_px": 16,
    "score_quantum_px": 16,
    "score_dilation_cells": 1,
    "score_stencil_spacing_px": 4,
    "score_stencil_min_observable_fraction": 0.75,
    "anchor_stride_px": 16,
    "angle_residuals": 8,
    "scale_multipliers": [2.0**-0.5, 1.0, 2.0**0.5],
    "source_target_scale_ratio": [0.5, 2.0],
    "nominal_candidates_per_row": 73_728,
    "candidate_order": "anchor_angle_parallel_perpendicular",
    "candidate_chunk_size": 256,
    "cell_chunk_size": 256,
    "geo_weights": [4.0, 2.0, 1.0, 1.0, 2.0],
    "geo_temperature": 0.35,
    "relation_band": 0.125,
    "exact_gt_required_retained": 1.0,
    "allowed_lost_raster_pixels": 0,
    "square_dilate_radius_px": 24,
    "sampling_rng": "PCG64",
    "sampling_method": "shared_raw_quantile_inverse_cdf",
    "geometry_science_lock_version": "v06_a_geometry_lock_f3",
    "context_mode": "no_context",
    "context_version": None,
    "recovery_mode": "none",
    "recovery_version": None,
    "hazelnut_experiment": "configs/experiments/v06_a_hazelnut_rc1.yaml",
    "cable_experiment": "configs/experiments/v06_a_cable_rc1.yaml",
}

_EXPECTED_ROWS = (
    ("hazelnut", "crack", "000", "000", 202608280100),
    ("hazelnut", "crack", "000", "260", 202608280100),
    ("hazelnut", "cut", "000", "000", 202608280110),
    ("hazelnut", "cut", "000", "260", 202608280110),
    ("hazelnut", "hole", "000", "000", 202608280120),
    ("hazelnut", "hole", "000", "260", 202608280120),
    ("hazelnut", "print", "000", "000", 202608280130),
    ("hazelnut", "print", "000", "260", 202608280130),
    ("cable", "bent_wire", "000", "000", 202608280200),
    ("cable", "bent_wire", "000", "152", 202608280200),
    ("cable", "cut_inner_insulation", "000", "000", 202608280210),
    ("cable", "cut_inner_insulation", "000", "152", 202608280210),
    ("cable", "cut_outer_insulation", "000", "000", 202608280220),
    ("cable", "cut_outer_insulation", "000", "152", 202608280220),
    ("cable", "poke_insulation", "000", "000", 202608280230),
    ("cable", "poke_insulation", "000", "152", 202608280230),
)


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid G0 configuration: {error}") from error
    if not isinstance(loaded, Mapping) or not all(isinstance(key, str) for key in loaded):
        raise ValueError("G0 configuration must be a string-keyed YAML mapping")
    return dict(loaded)


def load_g0_config(path: Path) -> G0Config:
    """Load the one frozen first-panel G0 schedule and numeric contract."""

    loaded = _load_mapping(path)
    if set(loaded) != _EXPECTED_KEYS:
        raise ValueError(
            "G0 schema mismatch: "
            f"missing={sorted(_EXPECTED_KEYS - set(loaded))}, "
            f"extra={sorted(set(loaded) - _EXPECTED_KEYS)}"
        )
    for key, expected in _EXPECTED_CONSTANTS.items():
        if loaded[key] != expected:
            raise ValueError(f"G0 {key} must equal the frozen first-panel value {expected!r}")
    raw_rows = loaded["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != 16:
        raise ValueError("G0 rows must contain exactly 16 fixed identities")
    rows: list[G0ScheduleRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "category",
            "defect_type",
            "source_id",
            "source_role",
            "target_id",
            "target_role",
            "seed",
        }:
            raise ValueError("each G0 row must use the exact identity schema")
        rows.append(G0ScheduleRow(**raw))
    observed = tuple(
        (row.category, row.defect_type, row.source_id, row.target_id, row.seed) for row in rows
    )
    if observed != _EXPECTED_ROWS:
        raise ValueError("G0 rows must preserve the fixed 16-row identity order")
    if any(row.source_role != "fewshot" or row.target_role != "normal_dev" for row in rows):
        raise ValueError("G0 roles must be frozen fewshot-to-normal_dev")
    return G0Config(
        path=path.resolve(strict=True),
        status=str(loaded["status"]),
        rows=tuple(rows),
        candidate_chunk_size=int(loaded["candidate_chunk_size"]),
        cell_chunk_size=int(loaded["cell_chunk_size"]),
        nominal_candidates_per_row=int(loaded["nominal_candidates_per_row"]),
        values=loaded,
    )


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


def _output_destination(*, output_root: Path, output_relpath: str) -> Path:
    if not output_root.is_dir():
        raise FileNotFoundError("G0 output root does not exist")
    if not isinstance(output_relpath, str) or not output_relpath:
        raise ValueError("G0 output must be a caller-specified contained relative path")
    relative = PurePosixPath(output_relpath)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != output_relpath
    ):
        raise ValueError("G0 output must be a caller-specified contained relative path")
    if tuple(relative.parts[:2]) != ("v06_a", "g0_rc2"):
        raise ValueError("G0 output must be contained under v06_a/g0_rc2")
    root = output_root.resolve(strict=True)
    destination = root.joinpath(*relative.parts)
    resolved = destination.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("G0 output must be a caller-specified contained relative path") from error
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"G0 refuses to overwrite an existing output: {destination}")
    return destination


def _physical_sdf(structure: NormalStructure) -> np.ndarray:
    return sample_continuous_field(
        physical_sdf_px_512(structure.support), geometry_grid_centers_512()
    )


def _record_for_source(manifest: IdentityManifest, row: G0ScheduleRow) -> IdentityRecord:
    matches = tuple(
        record
        for record in manifest.fewshot
        if record.defect_type == row.defect_type and record.item_id == row.source_id
    )
    if len(matches) != 1 or matches[0].role != "fewshot":
        raise ValueError("G0 source identity must be the exact frozen fewshot record")
    return matches[0]


def _record_for_target(manifest: IdentityManifest, row: G0ScheduleRow) -> IdentityRecord:
    matches = tuple(
        record
        for record in manifest.normal_dev
        if record.item_id == row.target_id and record.defect_type == "good"
    )
    if len(matches) != 1 or matches[0].role != "normal_dev":
        raise ValueError("G0 target identity must be the exact frozen normal_dev record")
    return matches[0]


def _prepare_reference_set(
    manifest: IdentityManifest,
    experiment: V06AExperimentConfig,
    dataset_root: Path,
) -> PreparedReferenceSet:
    records = tuple(
        next(record for record in manifest.normal_fit if record.item_id == item_id)
        for item_id in manifest.atlas_normal_ids
    )
    normals = tuple(
        load_identity_record(record, dataset_root=dataset_root, config=experiment)
        for record in records
    )
    structures = tuple(estimate_normal_structure(normal, config=experiment) for normal in normals)
    physical_sdf_512 = tuple(physical_sdf_px_512(structure.support) for structure in structures)
    grid_points = geometry_grid_centers_512()
    physical_sdf = np.stack(
        tuple(sample_continuous_field(value, grid_points) for value in physical_sdf_512)
    )
    rgb_fields = tuple(
        estimate_geometry_field(normal.rgb, valid=normal.valid, sdf_128=None) for normal in normals
    )
    geometry_fields = tuple(
        estimate_geometry_field(
            normal.rgb,
            valid=normal.valid,
            sdf_px_512=physical,
        )
        for normal, physical in zip(normals, physical_sdf_512, strict=True)
    )
    observable_128 = np.logical_and.reduce(
        tuple(geometry_observable_128(normal.valid) for normal in normals)
    )
    identities = tuple(normal.record.sample_id for normal in normals)
    if identities != tuple(record.sample_id for record in records):
        raise AssertionError("G0 reference identity order changed while loading")
    return PreparedReferenceSet(
        normals,
        structures,
        rgb_fields,
        geometry_fields,
        np.stack(tuple(structure.support for structure in structures)),
        physical_sdf,
        observable_128,
        identities,
    )


def prepare_g0_categories(
    config: G0Config,
    *,
    repository_root: Path,
    dataset_root: Path,
) -> tuple[PreparedG0Category, ...]:
    """Load only fixed fit references, source-000, and paired dev normals."""

    if not repository_root.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError("G0 repository_root and dataset_root must exist")
    root = repository_root.resolve(strict=True)
    dataset = dataset_root.resolve(strict=True)
    prepared_categories: list[PreparedG0Category] = []
    for category in ("hazelnut", "cable"):
        relative_experiment = str(config.values[f"{category}_experiment"])
        experiment_path = _contained_file(
            root.joinpath(*PurePosixPath(relative_experiment).parts),
            root=root,
            name=f"{category} experiment",
        )
        experiment = load_v06_a_experiment(experiment_path)
        if experiment.category != category:
            raise ValueError("G0 experiment category does not match its fixed schedule")
        manifest_path = _contained_file(
            root.joinpath(*PurePosixPath(experiment.identity_manifest).parts),
            root=root,
            name=f"{category} identity manifest",
        )
        manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
        if not manifest.is_authoritative():
            raise ValueError("G0 requires loader-authoritative identity manifests")
        references = _prepare_reference_set(manifest, experiment, dataset)
        target_cache: dict[str, tuple[LoadedIdentity, NormalStructure]] = {}
        rows: list[PreparedG0Row] = []
        for schedule in (row for row in config.rows if row.category == category):
            source_record = _record_for_source(manifest, schedule)
            target_record = _record_for_target(manifest, schedule)
            source = load_identity_record(source_record, dataset_root=dataset, config=experiment)
            if schedule.target_id not in target_cache:
                target = load_identity_record(
                    target_record, dataset_root=dataset, config=experiment
                )
                target_cache[schedule.target_id] = (
                    target,
                    estimate_normal_structure(target, config=experiment),
                )
            target, target_structure = target_cache[schedule.target_id]
            if source.record.sample_id == target.record.sample_id:
                raise ValueError("G0 source and target sample identities must differ")
            rows.append(PreparedG0Row(schedule, source, target, target_structure))
        if len(rows) != 8:
            raise ValueError("each G0 category must prepare exactly eight fixed rows")
        prepared_categories.append(
            PreparedG0Category(experiment, manifest, references, tuple(rows))
        )
    return tuple(prepared_categories)


def estimate_source_geometry(
    source: LoadedIdentity,
    references: PreparedReferenceSet,
    *,
    min_stencil_observable_fraction: float = 0.75,
) -> SourceGeometry:
    """Compute one target-independent source object exactly once."""

    if source.anomaly_gt is None:
        raise ValueError("G0 source requires exact GT")
    reference_valid = np.stack(tuple(normal.valid for normal in references.normals))
    try:
        masks = build_blind_reference_masks(source.anomaly_gt, source.valid, reference_valid)
    except ValueError as error:
        raise _SourceGeometryInvalid(
            SourceGeometryFailure("blind_and_ring", str(error), None, None)
        ) from error
    blinded_valid = source.valid & ~masks.blind_512
    source_rgb_field = estimate_geometry_field(source.rgb, valid=blinded_valid, sdf_128=None)
    try:
        match = match_reference_geometry(
            source_rgb_field,
            references.rgb_fields,
            masks,
            references.identities,
        )
    except ReferenceGeometryInvalid as error:
        raise _SourceGeometryInvalid(
            SourceGeometryFailure(
                "reference_matching",
                str(error),
                masks,
                source_rgb_field,
                source_ring_mean_energy=error.source_ring_mean_energy,
            )
        ) from error
    except ValueError as error:
        raise _SourceGeometryInvalid(
            SourceGeometryFailure("reference_matching", str(error), masks, source_rgb_field)
        ) from error
    try:
        proxy = aggregate_reference_proxy(
            references.geometry_fields,
            references.supports,
            references.physical_sdf_128,
            match.weights,
            observable_128=references.observable_128,
        )
    except ValueError as error:
        raise _SourceGeometryInvalid(
            SourceGeometryFailure(
                "reference_proxy",
                str(error),
                masks,
                source_rgb_field,
                match=match,
            )
        ) from error
    try:
        frame = estimate_local_frame_from_observable_ring(
            proxy.field,
            masks,
            observable_128=proxy.observable_128,
        )
    except ValueError as error:
        raise _SourceGeometryInvalid(
            SourceGeometryFailure(
                "source_frame",
                str(error),
                masks,
                source_rgb_field,
                match=match,
                proxy=proxy,
            )
        ) from error
    try:
        envelopes = build_coarse_envelopes(source.anomaly_gt, frame)
    except ValueError as error:
        raise _SourceGeometryInvalid(
            SourceGeometryFailure(
                "coarse_envelopes",
                str(error),
                masks,
                source_rgb_field,
                match=match,
                proxy=proxy,
                frame=frame,
            )
        ) from error
    try:
        stencil_observable = geometry_sample_observability(
            proxy.observable_128,
            local_to_world(envelopes.score.stencil_local, frame),
        )
        require_stencil_observable_coverage(
            stencil_observable,
            min_fraction=min_stencil_observable_fraction,
        )
    except ValueError as error:
        raise _SourceGeometryInvalid(
            SourceGeometryFailure(
                "proxy_observability",
                str(error),
                masks,
                source_rgb_field,
                match=match,
                proxy=proxy,
                frame=frame,
                envelopes=envelopes,
            )
        ) from error
    return SourceGeometry(
        source,
        masks,
        match,
        proxy,
        frame,
        envelopes,
        source_rgb_field,
        stencil_observable,
    )


def _target_geometry(prepared: PreparedG0Row) -> tuple[np.ndarray, GeometryField]:
    sdf_512 = physical_sdf_px_512(prepared.target_structure.support)
    sdf_128 = sample_continuous_field(sdf_512, geometry_grid_centers_512())
    field = estimate_geometry_field(
        prepared.target.rgb,
        valid=prepared.target.valid,
        sdf_px_512=sdf_512,
    )
    return sdf_128, field


def _empty_row(
    index: int,
    prepared: PreparedG0Row,
    *,
    source_geometry: SourceGeometry | None,
    source_failure: SourceGeometryFailure | None,
    raw_q: float,
    timings: Mapping[str, float],
    error: str,
    failure_stage: str | None = None,
    completed_stage: str = "source_geometry_partial",
    target_field: GeometryField | None = None,
    target_physical_sdf_128: np.ndarray | None = None,
    populate_target_geometry: bool = True,
) -> G0RowResult:
    if (target_field is None) != (target_physical_sdf_128 is None):
        raise ValueError("target field and physical SDF must be defined together")
    if target_field is None and populate_target_geometry:
        try:
            target_physical_sdf_128, target_field = _target_geometry(prepared)
        except ValueError as target_error:
            error = f"{error}; target_geometry unavailable: {target_error}"
    empty_probability = np.zeros(73_728, dtype=np.float64)
    empty_valid = np.zeros(73_728, dtype=bool)
    sample = inverse_cdf_sample(empty_probability, empty_valid, raw_q)
    return G0RowResult(
        index=index,
        schedule=prepared.schedule,
        source=prepared.source,
        target=prepared.target,
        target_structure=prepared.target_structure,
        source_geometry=source_geometry,
        source_failure=source_failure,
        target_field=target_field,
        candidates=None,
        scores=None,
        probabilities=empty_probability,
        anchor_marginal=np.zeros((32, 32), dtype=np.float64),
        expected_occupancy=np.zeros((128, 128), dtype=np.float64),
        sample=sample,
        sampled_affine=None,
        transferred_gt=None,
        proposal_512=None,
        proposal_128=None,
        transformed_envelope_corners=None,
        target_physical_sdf_128=target_physical_sdf_128,
        raw_q=raw_q,
        timings=dict(timings),
        valid=False,
        error=error,
        failure_stage=(
            failure_stage
            if failure_stage is not None
            else (source_failure.stage if source_failure is not None else "row_initialization")
        ),
        completed_stage=completed_stage,
    )


def compute_g0_row(
    index: int,
    prepared: PreparedG0Row,
    source_geometry: SourceGeometry,
    *,
    raw_q: float,
    config: G0Config,
) -> G0RowResult:
    """Run one fixed CPU row; scientific invalidity stays in this slot."""

    timings: dict[str, float] = {}
    started = time.perf_counter()
    active_stage = "target_geometry"
    completed_stage = "source_geometry"
    target_field: GeometryField | None = None
    target_physical_sdf: np.ndarray | None = None
    candidates: GeometryCandidateTable | None = None
    scores: GeometryScores | None = None
    probabilities = np.zeros(73_728, dtype=np.float64)
    anchor_marginal = np.zeros((32, 32), dtype=np.float64)
    occupancy = np.zeros((128, 128), dtype=np.float64)
    sample: QuantileSample | None = None
    sampled_affine: GeometryAffine | None = None
    transferred_gt: np.ndarray | None = None
    proposal_512: np.ndarray | None = None
    proposal_128: np.ndarray | None = None
    target_envelope_corners: np.ndarray | None = None
    try:
        phase = time.perf_counter()
        target_physical_sdf, target_field = _target_geometry(prepared)
        timings["target_geometry_seconds"] = time.perf_counter() - phase
        completed_stage = "target_geometry"

        active_stage = "candidate_enumeration"
        phase = time.perf_counter()
        candidates = enumerate_geometry_candidates(
            target_field,
            source_geometry.frame,
            source_geometry.envelopes.cover,
            prepared.target.valid,
        )
        timings["candidate_seconds"] = time.perf_counter() - phase
        completed_stage = "candidate_enumeration"

        active_stage = "geometry_scoring"
        phase = time.perf_counter()
        scores = score_geometry_candidate_table(
            candidates,
            source_geometry.envelopes.score,
            source_geometry.frame,
            source_geometry.proxy.field,
            source_geometry.proxy.sdf_128,
            target_field,
            target_physical_sdf,
            prepared.target.valid,
            candidate_chunk_size=config.candidate_chunk_size,
            stencil_chunk_size=config.cell_chunk_size,
            source_stencil_observable=source_geometry.stencil_observable,
        )
        if not np.array_equal(scores.valid, candidates.valid):
            raise AssertionError("G0 scorer changed the pre-sampling valid-slot mask")
        probabilities = stable_geometry_softmax(scores.distance, scores.valid)
        if not scores.valid.any():
            raise ValueError("G0 row has zero valid candidate slots")
        timings["score_seconds"] = time.perf_counter() - phase
        completed_stage = "geometry_scoring"

        active_stage = "occupancy"
        phase = time.perf_counter()
        anchor_marginal = probabilities.reshape(1024, 72).sum(axis=1).reshape(32, 32)
        occupancy = expected_candidate_table_occupancy(
            candidates,
            source_geometry.envelopes.score,
            source_geometry.frame,
            probabilities,
            prepared.target.valid,
            candidate_chunk_size=config.candidate_chunk_size,
            cell_chunk_size=config.cell_chunk_size,
        )
        timings["occupancy_seconds"] = time.perf_counter() - phase
        completed_stage = "occupancy"

        active_stage = "sampling"
        phase = time.perf_counter()
        sample = inverse_cdf_sample(probabilities, scores.valid, raw_q)
        if sample.index < 0:
            raise ValueError("G0 row has no sample despite valid probability mass")
        sampled_affine = affine_from_candidate_table(candidates, sample.index)
        if not sampled_affine.valid:
            raise AssertionError("G0 inverse CDF selected an invalid affine")
        completed_stage = "sampling"

        active_stage = "rasterization"
        assert prepared.source.anomaly_gt is not None
        exact = rasterize_pixel_cell_overlap(
            prepared.source.anomaly_gt,
            sampled_affine,
            prepared.target.valid,
        )
        transferred_gt = exact.mask
        completed_stage = "rasterization"

        active_stage = "proposal_512"
        proposal_512 = square_dilate_24(exact.mask) & prepared.target.valid
        completed_stage = "proposal_512"

        active_stage = "proposal_128"
        proposal_128 = downsample_any_4x4(proposal_512)
        completed_stage = "proposal_128"

        active_stage = "envelope_transform"
        source_envelope_corners = local_to_world(
            local_box_corners(source_geometry.envelopes.score.local_bounds),
            source_geometry.frame,
        )
        target_envelope_corners = transform_points(source_envelope_corners, sampled_affine)
        completed_stage = "envelope_transform"
        timings["sample_and_raster_seconds"] = time.perf_counter() - phase
        timings["total_seconds"] = time.perf_counter() - started
        return G0RowResult(
            index=index,
            schedule=prepared.schedule,
            source=prepared.source,
            target=prepared.target,
            target_structure=prepared.target_structure,
            source_geometry=source_geometry,
            source_failure=None,
            target_field=target_field,
            candidates=candidates,
            scores=scores,
            probabilities=probabilities,
            anchor_marginal=anchor_marginal,
            expected_occupancy=occupancy,
            sample=sample,
            sampled_affine=sampled_affine,
            transferred_gt=transferred_gt,
            proposal_512=proposal_512,
            proposal_128=proposal_128,
            transformed_envelope_corners=target_envelope_corners,
            target_physical_sdf_128=target_physical_sdf,
            raw_q=raw_q,
            timings=timings,
            valid=True,
            error=None,
            failure_stage=None,
            completed_stage="complete",
        )
    except (AssertionError, ValueError) as error:
        timings["total_seconds"] = time.perf_counter() - started
        if target_field is None or target_physical_sdf is None:
            return _empty_row(
                index,
                prepared,
                source_geometry=source_geometry,
                source_failure=None,
                raw_q=raw_q,
                timings=timings,
                error=str(error),
                failure_stage=active_stage,
                completed_stage=completed_stage,
                target_field=target_field,
                target_physical_sdf_128=target_physical_sdf,
                populate_target_geometry=False,
            )
        return G0RowResult(
            index=index,
            schedule=prepared.schedule,
            source=prepared.source,
            target=prepared.target,
            target_structure=prepared.target_structure,
            source_geometry=source_geometry,
            source_failure=None,
            target_field=target_field,
            candidates=candidates,
            scores=scores,
            probabilities=probabilities,
            anchor_marginal=anchor_marginal,
            expected_occupancy=occupancy,
            sample=sample,
            sampled_affine=sampled_affine,
            transferred_gt=transferred_gt,
            proposal_512=proposal_512,
            proposal_128=proposal_128,
            transformed_envelope_corners=target_envelope_corners,
            target_physical_sdf_128=target_physical_sdf,
            raw_q=raw_q,
            timings=timings,
            valid=False,
            error=str(error),
            failure_stage=active_stage,
            completed_stage=completed_stage,
        )


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(repr(contiguous.shape).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_hash(value: object) -> str:
    """Hash one science-identity payload with a single fail-closed encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_reference_sets(reference_sets: object) -> dict[str, str]:
    if not isinstance(reference_sets, Mapping) or set(reference_sets) != {
        "hazelnut",
        "cable",
    }:
        raise ValueError("science lock requires hazelnut and cable reference sets")
    digests: dict[str, str] = {}
    required = {
        "ordinal",
        "item_id",
        "sample_identity",
        "image_relpath",
        "input_sha256",
    }
    for category in ("hazelnut", "cable"):
        value = reference_sets[category]
        if not isinstance(value, Mapping) or "references" not in value:
            raise ValueError("science lock reference set is malformed")
        references = value["references"]
        if not isinstance(references, list) or len(references) != 16:
            raise ValueError("science lock requires exactly 16 ordered references per category")
        if any(not isinstance(item, Mapping) or set(item) != required for item in references):
            raise ValueError("science lock reference input identity is incomplete")
        if [item["ordinal"] for item in references] != list(range(16)):
            raise ValueError("science lock reference ordinals are not ordered")
        if len({item["sample_identity"] for item in references}) != 16:
            raise ValueError("science lock reference identities must be unique")
        digests[category] = _canonical_json_hash(references)
    return digests


def _validate_row_science_lock_shape(lock: object) -> None:
    if not isinstance(lock, Mapping):
        raise ValueError("row science lock must be a mapping")
    required = {
        "geometry_contract_lock_digest",
        "reference_set_digest",
        "source_geometry",
        "target_geometry",
        "sampling",
        "frame_affine_scale_orientation",
        "orientation_evidence_mass",
        "pipeline",
    }
    if not required.issubset(lock):
        raise ValueError("row science lock is incomplete")
    source = lock["source_geometry"]
    target = lock["target_geometry"]
    sampling = lock["sampling"]
    frame_affine = lock["frame_affine_scale_orientation"]
    orientation_evidence = lock["orientation_evidence_mass"]
    pipeline = lock["pipeline"]
    if not isinstance(source, Mapping) or not {
        "status",
        "blind_bounds_512",
        "ring_inner_bounds_512",
        "theoretical_ring_bounds_512",
        "observable_denominator",
        "valid_count",
        "coverage",
        "c_cover_sha256",
        "e_seed_sha256",
        "e_score_sha256",
        "source_frame_sha256",
        "reference_weights_sha256",
        "reference_distances_sha256",
        "source_ring_mean_energy",
    }.issubset(source):
        raise ValueError("row source-geometry science lock is incomplete")
    if not isinstance(target, Mapping) or not {
        "status",
        "identity",
        "input_sha256",
        "valid_512_sha256",
        "support_512_sha256",
        "physical_sdf_128_sha256",
        "g_128_sha256",
        "target_geometry_defined_sha256",
    }.issubset(target):
        raise ValueError("row target-geometry science lock is incomplete")
    if not isinstance(sampling, Mapping) or not {
        "raw_q_hex",
        "cdf_sha256",
        "sampled_index",
    }.issubset(sampling):
        raise ValueError("row sampling science lock is incomplete")
    if not isinstance(frame_affine, Mapping) or not {"status", "digest"}.issubset(frame_affine):
        raise ValueError("row frame/affine science lock is incomplete")
    if not isinstance(orientation_evidence, Mapping) or not {
        "columns",
        "sha256",
    }.issubset(orientation_evidence):
        raise ValueError("row orientation-evidence science lock is incomplete")
    if not isinstance(pipeline, Mapping) or not {
        "completed_stage",
        "failure_stage",
        "valid",
    }.issubset(pipeline):
        raise ValueError("row pipeline science lock is incomplete")


def _finalize_geometry_science_lock(sidecar: dict[str, object]) -> None:
    """Seal every row and the complete geometry-only science identity."""

    if (
        sidecar.get("context_mode") != "no_context"
        or sidecar.get("context_version") is not None
        or sidecar.get("recovery_mode") != "none"
        or sidecar.get("recovery_version") is not None
    ):
        raise ValueError("G0 F1 must remain explicit null/no-context")
    reference_digests = _validate_reference_sets(sidecar.get("reference_sets"))
    rows = sidecar.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("science lock requires exactly 16 fixed rows")
    row_digests: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or "row_science_lock" not in row:
            raise ValueError("science lock row payload is missing")
        lock = row["row_science_lock"]
        _validate_row_science_lock_shape(lock)
        digest = _canonical_json_hash(lock)
        row["row_science_lock_digest"] = digest
        row_digests.append(digest)
    lock = sidecar.get("geometry_science_lock")
    if not isinstance(lock, dict):
        raise ValueError("geometry science lock must be a mapping")
    lock["context_contract"] = {
        "context_mode": "no_context",
        "context_version": None,
        "recovery_mode": "none",
        "recovery_version": None,
    }
    lock["reference_set_digests"] = reference_digests
    lock["row_science_lock_digests"] = row_digests
    sidecar["geometry_science_lock_digest"] = _canonical_json_hash(lock)


def _validate_geometry_science_lock(sidecar: Mapping[str, object]) -> None:
    """Independently recompute the F1 row and top-level lock digests."""

    if (
        sidecar.get("context_mode") != "no_context"
        or sidecar.get("context_version") is not None
        or sidecar.get("recovery_mode") != "none"
        or sidecar.get("recovery_version") is not None
    ):
        raise ValueError("G0 F1 null/no-context contract is missing")
    reference_digests = _validate_reference_sets(sidecar.get("reference_sets"))
    rows = sidecar.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("science lock requires exactly 16 fixed rows")
    observed_row_digests: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("science lock row must be a mapping")
        lock = row.get("row_science_lock")
        _validate_row_science_lock_shape(lock)
        observed = _canonical_json_hash(lock)
        if row.get("row_science_lock_digest") != observed:
            raise ValueError("row science lock digest mismatch")
        observed_row_digests.append(observed)
    lock = sidecar.get("geometry_science_lock")
    if not isinstance(lock, Mapping):
        raise ValueError("geometry science lock must be a mapping")
    if lock.get("context_contract") != {
        "context_mode": "no_context",
        "context_version": None,
        "recovery_mode": "none",
        "recovery_version": None,
    }:
        raise ValueError("geometry science lock context contract mismatch")
    if lock.get("reference_set_digests") != reference_digests:
        raise ValueError("geometry science lock reference-set digest mismatch")
    if lock.get("row_science_lock_digests") != observed_row_digests:
        raise ValueError("geometry science lock row-digest list mismatch")
    if sidecar.get("geometry_science_lock_digest") != _canonical_json_hash(lock):
        raise ValueError("geometry science lock digest mismatch")


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_rss_kib() -> int:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _frame_array(frame: LocalFrame) -> np.ndarray:
    return np.asarray(
        (
            frame.center[0],
            frame.center[1],
            frame.alpha,
            frame.rho,
            frame.s_parallel,
            frame.s_perp,
            float(frame.orientation_unknown),
        ),
        dtype=np.float64,
    )


def _source_evidence(
    result: G0RowResult,
) -> tuple[
    BlindReferenceMasks | None,
    ReferenceGeometryMatch | None,
    ReferenceProxy | None,
    LocalFrame | None,
    CoarseEnvelopes | None,
]:
    """Expose every completed source stage, including INVALID rows."""

    if result.source_geometry is not None:
        source = result.source_geometry
        return source.masks, source.match, source.proxy, source.frame, source.envelopes
    failure = result.source_failure
    if failure is None:
        raise AssertionError("G0 row has neither source geometry nor failure evidence")
    return failure.masks, failure.match, failure.proxy, failure.frame, failure.envelopes


def _bbox(mask: np.ndarray | None) -> list[int] | None:
    if mask is None or not mask.any():
        return None
    rows, columns = np.nonzero(mask)
    return [int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())]


def _mass_coverage_counts(probabilities: np.ndarray) -> dict[str, int]:
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return {"50": 0, "80": 0, "95": 0}
    descending = np.sort(positive)[::-1]
    cumulative = np.cumsum(descending, dtype=np.float64)
    return {
        str(level): int(np.searchsorted(cumulative, level / 100.0, side="left") + 1)
        for level in (50, 80, 95)
    }


def _relation_summary(values: np.ndarray) -> dict[str, float | bool]:
    inside = float(np.mean(values > 0.125))
    outside = float(np.mean(values < -0.125))
    boundary = float(1.0 - inside - outside)
    return {
        "inside_fraction": inside,
        "boundary_fraction": boundary,
        "outside_fraction": outside,
        "crossing": bool(inside > 0.0 and outside > 0.0),
    }


def _identity_input_lock(identity: LoadedIdentity, *, dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve(strict=True)
    image_path = _contained_file(
        root.joinpath(*PurePosixPath(identity.record.image_relpath).parts),
        root=root,
        name="identity RGB input",
    )
    mask_sha256: str | None = None
    if identity.record.mask_relpath is not None:
        mask_path = _contained_file(
            root.joinpath(*PurePosixPath(identity.record.mask_relpath).parts),
            root=root,
            name="identity mask input",
        )
        mask_sha256 = _file_hash(mask_path)
    payload = {
        "item_id": identity.record.item_id,
        "sample_identity": identity.record.sample_id,
        "role": identity.record.role,
        "defect_type": identity.record.defect_type,
        "image_relpath": identity.record.image_relpath,
        "image_sha256": _file_hash(image_path),
        "mask_relpath": identity.record.mask_relpath,
        "mask_sha256": mask_sha256,
    }
    payload["input_sha256"] = _canonical_json_hash(payload)
    return payload


def _reference_sets_lock(
    categories: tuple[PreparedG0Category, ...], *, dataset_root: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in categories:
        references: list[dict[str, Any]] = []
        for ordinal, normal in enumerate(category.references.normals):
            identity = _identity_input_lock(normal, dataset_root=dataset_root)
            references.append(
                {
                    "ordinal": ordinal,
                    "item_id": identity["item_id"],
                    "sample_identity": identity["sample_identity"],
                    "image_relpath": identity["image_relpath"],
                    "input_sha256": identity["input_sha256"],
                }
            )
        result[category.experiment.category] = {"references": references}
    _validate_reference_sets(result)
    return result


def _row_numeric_arrays(result: G0RowResult) -> dict[str, np.ndarray]:
    masks, match, proxy, frame, envelopes = _source_evidence(result)
    candidates = result.candidates
    scores = result.scores
    affine = result.sampled_affine
    sample = result.sample
    target_field = result.target_field
    target_sdf = result.target_physical_sdf_128
    if (target_field is None) != (target_sdf is None):
        raise AssertionError("target field and physical SDF must be defined together")
    target_geometry_defined = target_field is not None
    evidence_mass = np.zeros(4, dtype=np.float64)
    if proxy is not None and frame is not None and envelopes is not None:
        if affine is None:
            evidence_mass = orientation_evidence_mass(
                envelopes.score,
                frame,
                proxy.field,
            )
        else:
            evidence_mass = orientation_evidence_mass(
                envelopes.score,
                frame,
                proxy.field,
                affine=affine,
                target_field=target_field,
                target_valid=result.target.valid,
            )
    sampled_probability = (
        0.0 if sample is None or sample.index < 0 else float(result.probabilities[sample.index])
    )
    arrays: dict[str, np.ndarray] = {
        "probabilities": result.probabilities.astype(np.float64, copy=False),
        "anchor_marginal": result.anchor_marginal.astype(np.float64, copy=False),
        "expected_occupancy": result.expected_occupancy.astype(np.float64, copy=False),
        "raw_q": np.asarray((result.raw_q,), dtype=np.float64),
        "target_geometry_defined": np.asarray((target_geometry_defined,), dtype=np.uint8),
        "affine_defined": np.asarray((affine is not None,), dtype=np.uint8),
        "sampled_probability": np.asarray((sampled_probability,), dtype=np.float64),
        "orientation_evidence_mass": evidence_mass,
        "candidate_slots": _candidate_slot_order(),
        "target_valid_512": result.target.valid.astype(np.uint8),
        "target_support_512": result.target_structure.support.astype(np.uint8),
        "target_physical_sdf_128": (
            np.empty((0, 0), dtype=np.float64)
            if target_sdf is None
            else target_sdf.astype(np.float64, copy=False)
        ),
        "target_g": (
            np.empty((0, 0, 5), dtype=np.float64) if target_field is None else target_field.g
        ),
        "target_orientation_unknown_128": (
            np.empty((0, 0), dtype=np.uint8)
            if target_field is None
            else target_field.orientation_unknown.astype(np.uint8)
        ),
        "target_scale_fallback_128": (
            np.empty((0, 0), dtype=np.uint8)
            if target_field is None
            else target_field.scale_fallback.astype(np.uint8)
        ),
        "transferred_gt": (
            np.empty((0, 0), dtype=np.uint8)
            if result.transferred_gt is None
            else result.transferred_gt.astype(np.uint8)
        ),
        "proposal_512": (
            np.empty((0, 0), dtype=np.uint8)
            if result.proposal_512 is None
            else result.proposal_512.astype(np.uint8)
        ),
        "proposal_128": (
            np.empty((0, 0), dtype=np.uint8)
            if result.proposal_128 is None
            else result.proposal_128.astype(np.uint8)
        ),
        "candidate_parameters": np.empty((0, 6), dtype=np.float64),
        "candidate_valid": np.zeros((73_728,), dtype=np.uint8),
        "candidate_invalid_reason": np.full((73_728,), 255, dtype=np.uint8),
        "affine_matrices": np.empty((0, 3, 3), dtype=np.float64),
        "affine_inverse_linear": np.empty((0, 2, 2), dtype=np.float64),
        "affine_relative_rotation": np.empty((0,), dtype=np.float64),
        "affine_target_axis_lift": np.empty((0,), dtype=np.float64),
        "d_geo_components": np.empty((0, 5), dtype=np.float64),
        "d_geo": np.empty((0,), dtype=np.float64),
        "sampled_affine": np.empty((0, 3, 3), dtype=np.float64),
        "transformed_envelope_corners": np.empty((0, 2), dtype=np.float64),
    }
    if masks is not None:
        arrays.update(
            {
                "blind_bounds_512": np.asarray(masks.blind_bounds, dtype=np.int64),
                "ring_inner_bounds_512": np.asarray(masks.ring_inner_bounds, dtype=np.int64),
                "theoretical_ring_bounds_512": np.asarray(
                    (*masks.outer_bounds, *masks.ring_inner_bounds), dtype=np.int64
                ),
                "blind_512": masks.blind_512.astype(np.uint8),
                "theoretical_ring_128": masks.theoretical_ring_128.astype(np.uint8),
                "observable_ring_128": masks.observable_ring_128.astype(np.uint8),
                "dependency_valid_128": masks.dependency_valid_128.astype(np.uint8),
                "common_ring_128": masks.common_ring_128.astype(np.uint8),
                "ring_counts_coverage": np.asarray(
                    (
                        masks.theoretical_cell_count,
                        masks.ring_denominator,
                        masks.valid_cell_count,
                        masks.coverage,
                    ),
                    dtype=np.float64,
                ),
            }
        )
    else:
        arrays.update(
            {
                "blind_bounds_512": np.empty((0,), dtype=np.int64),
                "ring_inner_bounds_512": np.empty((0,), dtype=np.int64),
                "theoretical_ring_bounds_512": np.empty((0,), dtype=np.int64),
                "blind_512": np.empty((0, 0), dtype=np.uint8),
                "theoretical_ring_128": np.empty((0, 0), dtype=np.uint8),
                "observable_ring_128": np.empty((0, 0), dtype=np.uint8),
                "dependency_valid_128": np.empty((0, 0), dtype=np.uint8),
                "common_ring_128": np.empty((0, 0), dtype=np.uint8),
                "ring_counts_coverage": np.empty((0,), dtype=np.float64),
            }
        )
    source_energy = (
        match.source_ring_mean_energy
        if match is not None
        else (
            None if result.source_failure is None else result.source_failure.source_ring_mean_energy
        )
    )
    if (
        match is not None
        or proxy is not None
        or frame is not None
        or envelopes is not None
        or source_energy is not None
    ):
        arrays.update(
            {
                "source_frame": (
                    np.empty((0,), dtype=np.float64) if frame is None else _frame_array(frame)
                ),
                "reference_weights": (
                    np.empty((0,), dtype=np.float64) if match is None else match.weights
                ),
                "reference_distances": (
                    np.empty((0,), dtype=np.float64) if match is None else match.distances
                ),
                "reference_source_ring_mean_energy": np.asarray(
                    () if source_energy is None else (source_energy,), dtype=np.float64
                ),
                "c_cover_local_bounds": (
                    np.empty((0,), dtype=np.float64)
                    if envelopes is None
                    else envelopes.cover.local_bounds
                ),
                "e_seed_local_bounds": (
                    np.empty((0,), dtype=np.float64)
                    if envelopes is None
                    else envelopes.seed.local_bounds
                ),
                "e_score_local_bounds": (
                    np.empty((0,), dtype=np.float64)
                    if envelopes is None
                    else envelopes.score.local_bounds
                ),
                "e_score_stencil_local": (
                    np.empty((0, 2), dtype=np.float64)
                    if envelopes is None
                    else envelopes.score.stencil_local
                ),
                "source_proxy_g": (
                    np.empty((0, 0, 5), dtype=np.float64) if proxy is None else proxy.field.g
                ),
                "source_proxy_sdf_128": (
                    np.empty((0, 0), dtype=np.float64) if proxy is None else proxy.sdf_128
                ),
                "source_proxy_support": (
                    np.empty((0, 0), dtype=np.float64)
                    if proxy is None
                    else proxy.support_probability
                ),
                "source_proxy_observable_128": (
                    np.empty((0, 0), dtype=np.uint8)
                    if proxy is None
                    else proxy.observable_128.astype(np.uint8)
                ),
            }
        )
    else:
        arrays.update(
            {
                "source_frame": np.empty((0,), dtype=np.float64),
                "reference_weights": np.empty((0,), dtype=np.float64),
                "reference_distances": np.empty((0,), dtype=np.float64),
                "reference_source_ring_mean_energy": np.empty((0,), dtype=np.float64),
                "c_cover_local_bounds": np.empty((0,), dtype=np.float64),
                "e_seed_local_bounds": np.empty((0,), dtype=np.float64),
                "e_score_local_bounds": np.empty((0,), dtype=np.float64),
                "e_score_stencil_local": np.empty((0, 2), dtype=np.float64),
                "source_proxy_g": np.empty((0, 0, 5), dtype=np.float64),
                "source_proxy_sdf_128": np.empty((0, 0), dtype=np.float64),
                "source_proxy_support": np.empty((0, 0), dtype=np.float64),
                "source_proxy_observable_128": np.empty((0, 0), dtype=np.uint8),
            }
        )
    if candidates is not None:
        if not np.array_equal(candidates.slots, arrays["candidate_slots"]):
            raise AssertionError("G0 candidate slot order differs from frozen template")
        arrays.update(
            {
                "candidate_parameters": candidates.parameters,
                "candidate_valid": candidates.valid.astype(np.uint8),
                "candidate_invalid_reason": candidates.invalid_reason,
                "affine_matrices": candidates.matrices,
                "affine_inverse_linear": candidates.inverse_linear,
                "affine_relative_rotation": candidates.relative_rotation,
                "affine_target_axis_lift": candidates.target_axis_lift,
            }
        )
    if scores is not None:
        arrays["d_geo_components"] = scores.components
        arrays["d_geo"] = scores.distance
    if sample is not None:
        if sample.raw_q.hex() != result.raw_q.hex():
            raise AssertionError("G0 row sample consumed a different raw quantile")
        arrays["cdf"] = sample.cdf
        arrays["sample_index_rank"] = np.asarray((sample.index, sample.valid_rank), dtype=np.int64)
    else:
        arrays["cdf"] = np.empty((0,), dtype=np.float64)
        arrays["sample_index_rank"] = np.asarray((-1, -1), dtype=np.int64)
    if affine is not None:
        arrays["sampled_affine"] = affine.matrix
    if result.transformed_envelope_corners is not None:
        arrays["transformed_envelope_corners"] = result.transformed_envelope_corners
    if any(value.dtype == object for value in arrays.values()):
        raise AssertionError("G0 numeric NPZ cannot contain object arrays")
    if any(
        not np.isfinite(value).all()
        for value in arrays.values()
        if np.issubdtype(value.dtype, np.floating)
    ):
        raise AssertionError("G0 numeric NPZ must use finite floating-point sentinels")
    return arrays


def _row_sidecar(
    result: G0RowResult,
    numeric_path: Path,
    *,
    geometry_contract_lock_digest: str,
    reference_set_digest: str,
    source_input_lock: Mapping[str, Any],
    target_input_lock: Mapping[str, Any],
) -> dict[str, Any]:
    schedule = result.schedule
    source = result.source_geometry
    source_failure = result.source_failure
    masks, match, proxy, frame, envelopes = _source_evidence(result)
    candidates = result.candidates
    scores = result.scores
    sample = result.sample
    affine = result.sampled_affine
    arrays = _row_numeric_arrays(result)
    array_hashes = {name: _array_hash(value) for name, value in arrays.items()}
    support_pixels = int(result.target_structure.support.sum())
    valid_pixels = int(result.target.valid.sum())
    sampled_probability = (
        0.0 if sample is None or sample.index < 0 else float(result.probabilities[sample.index])
    )
    maximum_probability = float(result.probabilities.max(initial=0.0))
    positive = result.probabilities > 0.0
    entropy = float(
        -(result.probabilities[positive] * np.log(result.probabilities[positive])).sum()
    )
    valid_count = int(positive.sum())
    normalized_entropy = entropy / math.log(valid_count) if valid_count > 1 else 0.0
    source_energy = (
        match.source_ring_mean_energy
        if match is not None
        else (None if source_failure is None else source_failure.source_ring_mean_energy)
    )
    hard_deleted = (
        0.0
        if result.proposal_512 is None or not result.proposal_512.any()
        else float(
            (result.proposal_512 & ~result.target_structure.support).sum()
            / result.proposal_512.sum()
        )
    )
    sampled_components = (
        None
        if scores is None or sample is None or sample.index < 0
        else [float(value) for value in scores.components[sample.index]]
    )
    source_relation: dict[str, float | bool] | None = None
    target_relation: dict[str, float | bool] | None = None
    if proxy is not None and frame is not None and envelopes is not None:
        source_points = local_to_world(envelopes.score.stencil_local, frame)
        source_values = np.clip(
            _sample_field(proxy.sdf_128, source_points) / frame.s_perp,
            -1.0,
            1.0,
        )
        source_relation = _relation_summary(source_values)
        if affine is not None and result.target_physical_sdf_128 is not None:
            target_points = transform_points(source_points, affine)
            target_values = np.clip(
                _sample_field(result.target_physical_sdf_128, target_points)
                / affine.candidate.s_perp,
                -1.0,
                1.0,
            )
            target_relation = _relation_summary(target_values)
    science_lock: dict[str, str] = {}
    if frame is not None:
        science_lock["source_frame"] = _array_hash(_frame_array(frame))
    if match is not None:
        science_lock["reference_weights"] = _array_hash(match.weights)
    if envelopes is not None:
        science_lock.update(
            {
                "c_cover": array_hashes["c_cover_local_bounds"],
                "e_seed": array_hashes["e_seed_local_bounds"],
                "e_score": _canonical_json_hash(
                    {
                        "bounds": array_hashes["e_score_local_bounds"],
                        "stencil": array_hashes["e_score_stencil_local"],
                    }
                ),
            }
        )
    if candidates is not None:
        science_lock.update(
            {
                "candidate_slot_order": _array_hash(candidates.slots),
                "valid_mask": _array_hash(candidates.valid),
                "affine_parameters": _array_hash(
                    np.column_stack(
                        (
                            candidates.parameters,
                            candidates.relative_rotation,
                            candidates.target_axis_lift,
                        )
                    )
                ),
                "affine_matrices": _array_hash(candidates.matrices),
            }
        )
    if scores is not None:
        science_lock.update(
            {
                "d_geo_components": _array_hash(scores.components),
                "d_geo": _array_hash(scores.distance),
                "pi_geo": _array_hash(result.probabilities),
                "cdf": array_hashes["cdf"],
            }
        )
    source_identity_hash = _array_hash(
        np.concatenate(
            (
                result.source.rgb.ravel(),
                result.source.valid.astype(np.uint8).ravel(),
                result.source.anomaly_gt.astype(np.uint8).ravel(),
            )
        )
    )
    target_identity_hash = _array_hash(
        np.concatenate((result.target.rgb.ravel(), result.target.valid.astype(np.uint8).ravel()))
    )
    source_geometry_lock: dict[str, Any] = {
        "status": "valid" if source is not None else "invalid",
        "failure_stage": None if source_failure is None else source_failure.stage,
        "failure_reason": None if source_failure is None else source_failure.reason,
        "blind_bounds_512": None if masks is None else list(masks.blind_bounds),
        "ring_inner_bounds_512": None if masks is None else list(masks.ring_inner_bounds),
        "theoretical_ring_bounds_512": (
            None
            if masks is None
            else {"outer": list(masks.outer_bounds), "inner": list(masks.ring_inner_bounds)}
        ),
        "theoretical_cell_count": (0 if masks is None else masks.theoretical_cell_count),
        "observable_denominator": 0 if masks is None else masks.ring_denominator,
        "valid_count": 0 if masks is None else masks.valid_cell_count,
        "coverage": 0.0 if masks is None else masks.coverage,
        "minimum_cells": 64,
        "minimum_coverage": 0.75,
        "source_ring_mean_energy": source_energy,
        "source_input_sha256": source_input_lock["input_sha256"],
        "source_frame_sha256": None if frame is None else array_hashes["source_frame"],
        "reference_weights_sha256": (None if match is None else array_hashes["reference_weights"]),
        "reference_distances_sha256": (
            None if match is None else array_hashes["reference_distances"]
        ),
        "c_cover_sha256": (None if envelopes is None else array_hashes["c_cover_local_bounds"]),
        "e_seed_sha256": (None if envelopes is None else array_hashes["e_seed_local_bounds"]),
        "e_score_sha256": (
            None
            if envelopes is None
            else _canonical_json_hash(
                {
                    "bounds": array_hashes["e_score_local_bounds"],
                    "stencil": array_hashes["e_score_stencil_local"],
                }
            )
        ),
    }
    target_geometry_lock = {
        "status": (
            "computed"
            if bool(arrays["target_geometry_defined"][0])
            else "not_computed_finite_sentinel"
        ),
        "identity": result.target.record.sample_id,
        "input_sha256": target_input_lock["input_sha256"],
        "valid_512_sha256": array_hashes["target_valid_512"],
        "support_512_sha256": array_hashes["target_support_512"],
        "physical_sdf_128_sha256": array_hashes["target_physical_sdf_128"],
        "g_128_sha256": array_hashes["target_g"],
        "orientation_unknown_128_sha256": array_hashes["target_orientation_unknown_128"],
        "scale_fallback_128_sha256": array_hashes["target_scale_fallback_128"],
        "target_geometry_defined_sha256": array_hashes["target_geometry_defined"],
    }
    sampling_lock = {
        "seed": schedule.seed,
        "raw_q_hex": result.raw_q.hex(),
        "raw_q_sha256": array_hashes["raw_q"],
        "cdf_sha256": array_hashes["cdf"],
        "sampled_index": -1 if sample is None else sample.index,
        "sampled_valid_rank": -1 if sample is None else sample.valid_rank,
        "status": "sampled" if sample is not None and sample.index >= 0 else "not_computed",
    }
    frame_affine_payload = {
        "status": (
            "sampled_affine"
            if affine is not None
            else ("candidates_computed" if candidates is not None else "not_computed")
        ),
        "source_frame_sha256": (None if frame is None else array_hashes["source_frame"]),
        "candidate_slot_order_sha256": array_hashes["candidate_slots"],
        "candidate_parameters_sha256": array_hashes["candidate_parameters"],
        "candidate_valid_sha256": array_hashes["candidate_valid"],
        "candidate_invalid_reason_sha256": array_hashes["candidate_invalid_reason"],
        "affine_matrices_sha256": array_hashes["affine_matrices"],
        "affine_relative_rotation_sha256": array_hashes["affine_relative_rotation"],
        "affine_target_axis_lift_sha256": array_hashes["affine_target_axis_lift"],
        "sampled_affine_sha256": array_hashes["sampled_affine"],
        "affine_defined_sha256": array_hashes["affine_defined"],
        "orientation_evidence_mass_sha256": array_hashes["orientation_evidence_mass"],
        "target_orientation_unknown_sha256": array_hashes["target_orientation_unknown_128"],
        "target_scale_fallback_sha256": array_hashes["target_scale_fallback_128"],
        "d_geo_components_sha256": array_hashes["d_geo_components"],
        "d_geo_sha256": array_hashes["d_geo"],
        "pi_geo_sha256": array_hashes["probabilities"],
        "parameter_columns": [
            "center_x_512_px",
            "center_y_512_px",
            "alpha_axis_rad",
            "s_parallel_512_px",
            "s_perp_512_px",
            "orientation_unknown",
        ],
        "invalid_reason_sentinel_not_computed": 255,
    }
    frame_affine_lock = {
        "status": frame_affine_payload["status"],
        "digest": _canonical_json_hash(frame_affine_payload),
        "payload": frame_affine_payload,
    }
    row_science_lock = {
        "geometry_contract_lock_digest": geometry_contract_lock_digest,
        "reference_set_digest": reference_set_digest,
        "source_geometry_lock_digest": _canonical_json_hash(source_geometry_lock),
        "source_geometry": source_geometry_lock,
        "target_geometry": target_geometry_lock,
        "sampling": sampling_lock,
        "frame_affine_scale_orientation": frame_affine_lock,
        "orientation_evidence_mass": {
            "columns": ["source", "target", "joint", "valid_stencil_fraction"],
            "sha256": array_hashes["orientation_evidence_mass"],
        },
        "pipeline": {
            "completed_stage": result.completed_stage,
            "failure_stage": result.failure_stage,
            "valid": result.valid,
        },
    }
    return {
        "row_index": result.index,
        "category": schedule.category,
        "defect_type": schedule.defect_type,
        "source_id": schedule.source_id,
        "source_role": schedule.source_role,
        "target_id": schedule.target_id,
        "target_role": schedule.target_role,
        "seed": schedule.seed,
        "raw_q": result.raw_q,
        "raw_q_hex": result.raw_q.hex(),
        "source_sample_identity": result.source.record.sample_id,
        "target_sample_identity": result.target.record.sample_id,
        "source_input": dict(source_input_lock),
        "target_input": dict(target_input_lock),
        "source_input_hash": source_identity_hash,
        "target_input_hash": target_identity_hash,
        "failure_stage": result.failure_stage,
        "completed_stage": result.completed_stage,
        "source_frame": (
            None
            if frame is None
            else {
                "center": list(frame.center),
                "alpha": frame.alpha,
                "rho": frame.rho,
                "s_parallel": frame.s_parallel,
                "s_perp": frame.s_perp,
                "orientation_unknown": frame.orientation_unknown,
                "gauge": frame.gauge,
                "axis_semantics": (
                    "unknown_no_measured_axes"
                    if frame.orientation_unknown
                    else "measured_double_angle_axis"
                ),
            }
        ),
        "reference": (
            {
                "status": "invalid",
                "failure_stage": None if source_failure is None else source_failure.stage,
                "failure_reason": None if source_failure is None else source_failure.reason,
                "theoretical_ring_bounds": source_geometry_lock["theoretical_ring_bounds_512"],
                "theoretical_cell_count": source_geometry_lock["theoretical_cell_count"],
                "observable_denominator": source_geometry_lock["observable_denominator"],
                "valid_cell_count": source_geometry_lock["valid_count"],
                "coverage": source_geometry_lock["coverage"],
                "source_ring_mean_energy": source_energy,
            }
            if match is None
            else {
                "status": "valid" if source is not None else "computed_before_invalid",
                "weights": match.weights.tolist(),
                "distances": match.distances.tolist(),
                "entropy_nats": match.entropy,
                "normalized_entropy": match.entropy / math.log(match.weights.size),
                "source_ring_mean_energy": match.source_ring_mean_energy,
                "identity_order_hash": match.identity_order_hash,
                "common_ring_hash": match.common_ring_hash,
                "theoretical_ring_bounds": source_geometry_lock["theoretical_ring_bounds_512"],
                "theoretical_cell_count": 0 if masks is None else masks.theoretical_cell_count,
                "ring_denominator": 0 if masks is None else masks.ring_denominator,
                "valid_cell_count": 0 if masks is None else masks.valid_cell_count,
                "coverage": 0.0 if masks is None else masks.coverage,
            }
        ),
        "coarse_envelopes": (
            None
            if envelopes is None
            else {
                "c_cover_local_bounds": envelopes.cover.local_bounds.tolist(),
                "c_cover_quantum_px": envelopes.cover.quantum_px,
                "e_seed_local_bounds": envelopes.seed.local_bounds.tolist(),
                "e_seed_quantum_px": envelopes.seed.quantum_px,
                "e_score_local_bounds": envelopes.score.local_bounds.tolist(),
                "e_score_quantum_px": envelopes.score.quantum_px,
                "e_score_dilation_cells": envelopes.score.dilation_cells,
                "e_score_stencil_spacing_px": envelopes.score.stencil_spacing_px,
            }
        ),
        "sampled_target_frame": (
            None
            if affine is None
            else {
                "center": [affine.candidate.center_x, affine.candidate.center_y],
                "alpha_axis": affine.candidate.alpha,
                "target_axis_lift": affine.target_axis_lift,
                "relative_rotation": affine.relative_rotation,
                "s_parallel": affine.candidate.s_parallel,
                "s_perp": affine.candidate.s_perp,
                "orientation_unknown": affine.candidate.orientation_unknown,
                "axis_semantics": "candidate_hypothesis",
            }
        ),
        "sampled_affine": None if affine is None else affine.matrix.tolist(),
        "candidate_count": 73_728,
        "valid_candidate_count": 0 if candidates is None else int(candidates.valid.sum()),
        "sampled_index": -1 if sample is None else sample.index,
        "sampled_valid_rank": -1 if sample is None else sample.valid_rank,
        "sampled_probability_rank": (
            -1
            if sample is None or sample.index < 0
            else int(np.sum(result.probabilities > sampled_probability) + 1)
        ),
        "cdf_before": None if sample is None else sample.cdf_before,
        "cdf_at": None if sample is None else sample.cdf_at,
        "p_sample": sampled_probability,
        "p_max": maximum_probability,
        "p_sample_over_p_max": (
            0.0 if maximum_probability == 0.0 else sampled_probability / maximum_probability
        ),
        "entropy_nats": entropy,
        "normalized_entropy": normalized_entropy,
        "orientation_evidence_mass": {
            "status": (
                "sampled"
                if affine is not None
                else (
                    "source_only"
                    if proxy is not None and frame is not None and envelopes is not None
                    else "not_computed"
                )
            ),
            "source": float(arrays["orientation_evidence_mass"][0]),
            "target": float(arrays["orientation_evidence_mass"][1]),
            "joint": float(arrays["orientation_evidence_mass"][2]),
            "valid_stencil_fraction": float(arrays["orientation_evidence_mass"][3]),
        },
        "mass_coverage_count": _mass_coverage_counts(result.probabilities),
        "d_geo_sampled_components": sampled_components,
        "exact_gt_retained_fraction": 1.0 if result.transferred_gt is not None else 0.0,
        "envelope_retained_fraction": (
            None if sampled_components is None else 1.0 - sampled_components[0]
        ),
        "transferred_gt_bbox": _bbox(result.transferred_gt),
        "transferred_envelope_bbox": (
            None
            if result.transformed_envelope_corners is None
            else [
                float(result.transformed_envelope_corners[:, 0].min()),
                float(result.transformed_envelope_corners[:, 1].min()),
                float(result.transformed_envelope_corners[:, 0].max()),
                float(result.transformed_envelope_corners[:, 1].max()),
            ]
        ),
        "relation_before": source_relation,
        "relation_after": target_relation,
        "hard_support_deleted_fraction": hard_deleted,
        "support_area_ratio": support_pixels / valid_pixels,
        "full_canvas_fallback": bool(
            np.array_equal(result.target_structure.support, result.target.valid)
        ),
        "target_sdf_minmax_px": (
            None
            if result.target_physical_sdf_128 is None
            else [
                float(result.target_physical_sdf_128.min()),
                float(result.target_physical_sdf_128.max()),
            ]
        ),
        "science_lock": science_lock,
        "row_science_lock": row_science_lock,
        "numeric_npz": numeric_path.name,
        "numeric_npz_sha256": _file_hash(numeric_path),
        "numeric_array_hashes": array_hashes,
        "candidate_chunk_size": 256,
        "cell_chunk_size": 256,
        "stencil_count": (0 if envelopes is None else int(envelopes.score.stencil_local.shape[0])),
        "peak_array_shapes": (
            None
            if envelopes is None
            else {
                "transformed_stencil": [
                    256,
                    min(int(envelopes.score.stencil_local.shape[0]), 256),
                    2,
                ],
                "distance_component": [
                    256,
                    min(int(envelopes.score.stencil_local.shape[0]), 256),
                ],
                "occupancy_cells": [256, 2],
            }
        ),
        "timings_seconds": dict(result.timings),
        "valid": result.valid,
        "error": result.error,
    }


_PANEL_COLUMNS = (
    "source RGB",
    "exact GT + blind/score",
    "source local frame",
    "target normal",
    "support",
    "SDF",
    "G_x / local field",
    "anchor marginal",
    "expected occupancy",
    "sampled affine hypothesis + envelope",
    "transferred exact GT",
    "p* (outside support kept)",
)


def _heatmap(values: np.ndarray) -> np.ndarray:
    data = values.astype(np.float64, copy=False)
    # Fixed global [0,1] log display; never row-wise min-max normalization.
    normalized = np.log1p(999.0 * np.clip(data, 0.0, 1.0)) / math.log(1000.0)
    red = np.clip(1.5 * normalized, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(3.0 * normalized - 1.5), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - normalized), 0.0, 1.0)
    return np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


def _sdf_rgb(values: np.ndarray) -> np.ndarray:
    maximum = max(float(np.max(np.abs(values))), 1e-12)
    normalized = np.clip(values / maximum, -1.0, 1.0)
    red = np.where(normalized < 0.0, 255.0 * -normalized, 0.0)
    blue = np.where(normalized > 0.0, 255.0 * normalized, 0.0)
    green = 255.0 * (1.0 - np.abs(normalized))
    return np.rint(np.stack((red, green, blue), axis=-1)).astype(np.uint8)


def _geometry_rgb(field: GeometryField) -> np.ndarray:
    rho = np.clip(field.g[..., 2], 0.0, 1.0)
    red = (field.q[..., 0] * rho + 1.0) / 2.0
    green = (field.q[..., 1] * rho + 1.0) / 2.0
    anisotropy = np.clip((field.g[..., 3] - field.g[..., 4]) / math.log(4.0), 0.0, 1.0)
    return np.rint(np.stack((red, green, anisotropy), axis=-1) * 255.0).astype(np.uint8)


def _overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> np.ndarray:
    output = rgb.astype(np.float64, copy=True)
    output[mask] = (1.0 - alpha) * output[mask] + alpha * np.asarray(color)
    return np.rint(output).astype(np.uint8)


def _draw_polyline(
    rgb: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    width: int = 3,
    closed: bool = True,
) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    coordinates = [tuple(map(float, point)) for point in points]
    if closed and coordinates:
        coordinates.append(coordinates[0])
    if len(coordinates) >= 2:
        draw.line(coordinates, fill=color, width=width)
    return np.asarray(image, dtype=np.uint8).copy()


def _draw_frame(
    rgb: np.ndarray,
    frame: LocalFrame,
    *,
    axis_semantics: str,
) -> np.ndarray:
    if axis_semantics not in {"measured", "candidate_hypothesis"}:
        raise ValueError("frame axis semantics must be measured or candidate_hypothesis")
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    center = np.asarray(frame.center, dtype=np.float64)
    axes = frame.axes
    colors = (
        ((0, 255, 255), (255, 0, 255))
        if axis_semantics == "measured"
        else ((255, 128, 0), (80, 160, 255))
    )
    scales = (frame.s_parallel, frame.s_perp)
    if axis_semantics == "candidate_hypothesis" or not frame.orientation_unknown:
        for axis, scale, color in zip(axes.T, scales, colors, strict=True):
            start = center - axis * scale
            stop = center + axis * scale
            draw.line((tuple(start), tuple(stop)), fill=color, width=4)
    radius = 5
    draw.ellipse(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        outline=(255, 255, 0),
        width=3,
    )
    return np.asarray(image, dtype=np.uint8).copy()


def _resize_tile(rgb: np.ndarray, cell_size: int) -> Image.Image:
    return Image.fromarray(rgb).resize((cell_size, cell_size), Image.Resampling.BILINEAR)


def _row_tiles(result: G0RowResult) -> tuple[np.ndarray, ...]:
    source_rgb = result.source.rgb.copy()
    target_rgb = result.target.rgb.copy()
    gt = result.source.anomaly_gt
    gt_view = source_rgb.copy()
    source_frame_view = source_rgb.copy()
    support_view = np.repeat(
        (result.target_structure.support.astype(np.uint8) * 255)[..., None], 3, axis=2
    )
    sdf_values = (
        result.target_structure.sdf_128
        if result.target_physical_sdf_128 is None
        else result.target_physical_sdf_128
    )
    sdf_view = _sdf_rgb(sdf_values)
    sdf_view = np.repeat(np.repeat(sdf_view, 4, axis=0), 4, axis=1)
    diagnostic_masks, _, _, source_frame, source_envelopes = _source_evidence(result)
    if diagnostic_masks is not None and gt is not None:
        gt_view = _overlay_mask(gt_view, diagnostic_masks.blind_512, (255, 180, 0), 0.18)
        gt_view = _overlay_mask(gt_view, gt, (255, 0, 0), 0.75)
    if source_frame is not None and source_envelopes is not None and gt is not None:
        score_corners = local_to_world(
            local_box_corners(source_envelopes.score.local_bounds),
            source_frame,
        )
        gt_view = _draw_polyline(gt_view, score_corners, (0, 255, 255), width=4)
        source_frame_view = _draw_frame(
            source_frame_view,
            source_frame,
            axis_semantics="measured",
        )
    if result.target_field is None:
        geometry_view = np.zeros((512, 512, 3), dtype=np.uint8)
    else:
        geometry_view = np.repeat(
            np.repeat(_geometry_rgb(result.target_field), 4, axis=0), 4, axis=1
        )
    anchor_view = np.repeat(np.repeat(_heatmap(result.anchor_marginal), 16, axis=0), 16, axis=1)
    occupancy_view = np.repeat(np.repeat(_heatmap(result.expected_occupancy), 4, axis=0), 4, axis=1)
    affine_view = target_rgb.copy()
    if result.transformed_envelope_corners is not None:
        affine_view = _draw_polyline(
            affine_view, result.transformed_envelope_corners, (0, 255, 255), width=4
        )
    if result.sampled_affine is not None:
        target_frame = LocalFrame(
            center=(
                result.sampled_affine.candidate.center_x,
                result.sampled_affine.candidate.center_y,
            ),
            alpha=result.sampled_affine.target_axis_lift,
            rho=0.0 if result.sampled_affine.candidate.orientation_unknown else 1.0,
            s_parallel=result.sampled_affine.candidate.s_parallel,
            s_perp=result.sampled_affine.candidate.s_perp,
            orientation_unknown=result.sampled_affine.candidate.orientation_unknown,
            gauge=(
                "world_axis"
                if result.sampled_affine.candidate.orientation_unknown
                else "measured_axis"
            ),
        )
        affine_view = _draw_frame(
            affine_view,
            target_frame,
            axis_semantics="candidate_hypothesis",
        )
    transferred_view = (
        target_rgb.copy()
        if result.transferred_gt is None
        else _overlay_mask(target_rgb, result.transferred_gt, (255, 0, 0), 0.75)
    )
    proposal_view = target_rgb.copy()
    if result.proposal_512 is not None:
        outside = result.proposal_512 & ~result.target_structure.support
        inside = result.proposal_512 & result.target_structure.support
        proposal_view = _overlay_mask(proposal_view, inside, (0, 255, 0), 0.55)
        proposal_view = _overlay_mask(proposal_view, outside, (255, 0, 255), 0.75)
    return (
        source_rgb,
        gt_view,
        source_frame_view,
        target_rgb,
        support_view,
        sdf_view,
        geometry_view,
        anchor_view,
        occupancy_view,
        affine_view,
        transferred_view,
        proposal_view,
    )


def _render_row_panel(result: G0RowResult, *, cell_size: int) -> Image.Image:
    title_height = 42
    image = Image.new("RGB", (cell_size * len(_PANEL_COLUMNS), cell_size + title_height), "black")
    draw = ImageDraw.Draw(image)
    title = (
        f"{result.index:02d} {result.schedule.category}/{result.schedule.defect_type} "
        f"src={result.schedule.source_id} tgt={result.schedule.target_id} "
        f"seed={result.schedule.seed} {'VALID' if result.valid else 'INVALID'}"
    )
    draw.text((4, 2), title, fill=(255, 255, 255))
    if result.error:
        draw.text((4, 18), result.error[:180], fill=(255, 80, 80))
    for index, (name, tile) in enumerate(zip(_PANEL_COLUMNS, _row_tiles(result), strict=True)):
        left = index * cell_size
        image.paste(_resize_tile(tile, cell_size), (left, title_height))
        draw.rectangle(
            (left, title_height, left + cell_size - 1, title_height + 17), fill=(0, 0, 0)
        )
        draw.text((left + 2, title_height + 2), name, fill=(255, 255, 255))
    return image


def _render_legend() -> Image.Image:
    image = Image.new("RGB", (1400, 220), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    lines = (
        "G0 RC2 geometry-only legend " "(one global color rule; heatmaps use fixed log [0,1])",
        "red: exact/transferred GT; cyan: filled quantized E_score; " "yellow/orange: blind region",
        "source measured axes: cyan parallel / magenta perpendicular; unknown source: center only",
        "candidate hypothesis axes: orange parallel / blue perpendicular; yellow point: center",
        "support: white; SDF: red outside / green boundary / blue inside",
        "G_x: red=rho*cos(2alpha), green=rho*sin(2alpha), blue=log anisotropy",
        "p*: green inside soft support, magenta outside " "(outside is preserved, never clipped)",
        "Panels are diagnostic evidence only: no row or 16/16 validity count is a G0 PASS.",
    )
    for index, line in enumerate(lines):
        draw.text((16, 16 + index * 27), line, fill=(240, 240, 240))
    return image


def _science_file_hashes(repository_root: Path, config: G0Config) -> dict[str, str]:
    relative_paths = (
        config.path.relative_to(repository_root).as_posix(),
        str(config.values["hazelnut_experiment"]),
        str(config.values["cable_experiment"]),
        "configs/manifests/v06_a_hazelnut_rc1.yaml",
        "configs/manifests/v06_a_cable_rc1.yaml",
        "configs/checkpoints/v06_b_hazelnut_mask_first_causal_renderer.yaml",
        "src/cadiff/v06_a/structure.py",
        "src/cadiff/v06_a/targets.py",
        "src/cadiff/v06_a/teacher.py",
        "src/cadiff/v06_a/g0.py",
    )
    return {
        relative: _file_hash(
            _contained_file(repository_root / relative, root=repository_root, name=relative)
        )
        for relative in relative_paths
    }


def _rc1_evidence_hashes(repository_root: Path) -> dict[str, str]:
    relative_paths = (
        "runs/v06_a/p0_rc1_cpu_20260828_round4/p0_gate.json",
        "runs/v06_a/p0_rc1_cpu_20260828_round4/p0_combined_16x8.png",
        "runs/v06_a/p0_rc1_cpu_20260828_round4_overlay/p0_supplemental_overlay.json",
        "runs/v06_a/p0_rc1_cpu_20260828_round4_overlay/p0_supplemental_overlay_16x8.png",
        "runs/v06_a/p0_rc1_cpu_20260828_round4_overlay/p0_supplemental_overlay_legend.png",
    )
    return {
        relative: _file_hash(
            _contained_file(repository_root / relative, root=repository_root, name=relative)
        )
        for relative in relative_paths
    }


def _parent_g0_fail_hashes(repository_root: Path) -> dict[str, str]:
    parent = repository_root / "runs/v06_a/g0_rc2/first_complete_20260830"
    if not parent.is_dir():
        raise FileNotFoundError("immutable parent G0 FAIL evidence is missing")
    paths = tuple(sorted(path for path in parent.rglob("*") if path.is_file()))
    if not paths:
        raise FileNotFoundError("immutable parent G0 FAIL evidence is empty")
    return {path.relative_to(repository_root).as_posix(): _file_hash(path) for path in paths}


def _flatten_prepared(
    categories: tuple[PreparedG0Category, ...], config: G0Config
) -> tuple[tuple[PreparedG0Row, PreparedReferenceSet], ...]:
    lookup: dict[tuple[str, str, str], tuple[PreparedG0Row, PreparedReferenceSet]] = {}
    for category in categories:
        for row in category.rows:
            key = (row.schedule.category, row.schedule.defect_type, row.schedule.target_id)
            lookup[key] = (row, category.references)
    ordered = tuple(lookup[(row.category, row.defect_type, row.target_id)] for row in config.rows)
    if tuple(item[0].schedule for item in ordered) != config.rows:
        raise AssertionError("prepared G0 row order differs from frozen config")
    return ordered


def run_g0_preflight(
    config: G0Config,
    *,
    repository_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    """Run exactly fixed row zero in memory and report CPU time/RAM only."""

    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current_rss_before = _current_rss_kib()
    categories = prepare_g0_categories(
        config, repository_root=repository_root, dataset_root=dataset_root
    )
    prepared, references = _flatten_prepared(categories, config)[0]
    source_started = time.perf_counter()
    source_failure: SourceGeometryFailure | None = None
    try:
        source_geometry = estimate_source_geometry(
            prepared.source,
            references,
            min_stencil_observable_fraction=float(
                config.values["score_stencil_min_observable_fraction"]
            ),
        )
    except _SourceGeometryInvalid as error:
        source_geometry = None
        source_failure = error.evidence
    source_seconds = time.perf_counter() - source_started
    raw_q = shared_raw_quantile(prepared.schedule.seed)
    if source_geometry is None:
        assert source_failure is not None
        row = _empty_row(
            0,
            prepared,
            source_geometry=None,
            source_failure=source_failure,
            raw_q=raw_q,
            timings={"total_seconds": source_seconds},
            error=source_failure.reason,
        )
    else:
        row = compute_g0_row(
            0,
            prepared,
            source_geometry,
            raw_q=raw_q,
            config=config,
        )
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current_rss_after = _current_rss_kib()
    return {
        "mode": "one_fixed_row_cpu_preflight",
        "row_index": 0,
        "category": prepared.schedule.category,
        "defect_type": prepared.schedule.defect_type,
        "source_id": prepared.schedule.source_id,
        "target_id": prepared.schedule.target_id,
        "seed": prepared.schedule.seed,
        "valid": row.valid,
        "error": row.error,
        "source_geometry_seconds": source_seconds,
        "row_timings_seconds": dict(row.timings),
        "total_seconds": time.perf_counter() - started,
        "current_rss_before_kib": current_rss_before,
        "current_rss_after_kib": current_rss_after,
        "ru_maxrss_before_kib": rss_before,
        "ru_maxrss_after_kib": rss_after,
        "candidate_count": 73_728,
        "valid_candidate_count": (0 if row.candidates is None else int(row.candidates.valid.sum())),
        "stencil_count": (
            0
            if source_geometry is None
            else int(source_geometry.envelopes.score.stencil_local.shape[0])
        ),
        "candidate_chunk_size": config.candidate_chunk_size,
        "cell_chunk_size": config.cell_chunk_size,
        "gpu_used": False,
        "artifacts_created": False,
    }


def run_g0(
    config: G0Config,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
) -> G0Artifact:
    """Generate the one immutable 16-row CPU G0 artifact and nothing beyond it."""

    root = repository_root.resolve(strict=True)
    destination = _output_destination(output_root=output_root, output_relpath=output_relpath)
    destination.mkdir(parents=True, exist_ok=False)
    row_directory = destination / "rows"
    numeric_directory = destination / "numeric"
    row_directory.mkdir()
    numeric_directory.mkdir()
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current_rss_before = _current_rss_kib()
    categories = prepare_g0_categories(config, repository_root=root, dataset_root=dataset_root)
    ordered = _flatten_prepared(categories, config)
    reference_sets = _reference_sets_lock(categories, dataset_root=dataset_root)
    reference_set_digests = _validate_reference_sets(reference_sets)
    science_file_hashes = _science_file_hashes(root, config)
    code_hashes = {
        name: digest for name, digest in science_file_hashes.items() if name.startswith("src/")
    }
    configuration_hashes = {
        name: digest for name, digest in science_file_hashes.items() if name.startswith("configs/")
    }
    candidate_slot_template_hash = _array_hash(_candidate_slot_order())
    geometry_contract_lock = {
        "lock_schema": str(config.values["geometry_science_lock_version"]),
        "configuration_hashes": configuration_hashes,
        "code_hashes": code_hashes,
        "fixed_schedule": [
            {
                "category": row.category,
                "defect_type": row.defect_type,
                "source_id": row.source_id,
                "source_role": row.source_role,
                "target_id": row.target_id,
                "target_role": row.target_role,
                "seed": row.seed,
            }
            for row in config.rows
        ],
        "candidate_slot_template_sha256": candidate_slot_template_hash,
        "candidate_order": str(config.values["candidate_order"]),
        "geo_weights": list(config.values["geo_weights"]),
        "geo_temperature": config.values["geo_temperature"],
        "context_mode": "no_context",
        "context_version": None,
        "recovery_mode": "none",
        "recovery_version": None,
    }
    geometry_contract_lock_digest = _canonical_json_hash(geometry_contract_lock)
    raw_quantiles = {
        seed: shared_raw_quantile(seed) for seed in dict.fromkeys(row.seed for row in config.rows)
    }
    source_cache: dict[tuple[str, str], SourceGeometry] = {}
    source_errors: dict[tuple[str, str], SourceGeometryFailure] = {}
    results: list[G0RowResult] = []
    row_paths: list[Path] = []
    numeric_paths: list[Path] = []
    row_sidecars: list[dict[str, Any]] = []
    for index, (prepared, references) in enumerate(ordered):
        source_key = (prepared.schedule.category, prepared.schedule.defect_type)
        if source_key not in source_cache and source_key not in source_errors:
            try:
                source_cache[source_key] = estimate_source_geometry(
                    prepared.source,
                    references,
                    min_stencil_observable_fraction=float(
                        config.values["score_stencil_min_observable_fraction"]
                    ),
                )
            except _SourceGeometryInvalid as error:
                source_errors[source_key] = error.evidence
            except ValueError as error:
                source_errors[source_key] = SourceGeometryFailure(
                    "source_geometry", str(error), None, None
                )
        source_geometry = source_cache.get(source_key)
        if source_geometry is None:
            result = _empty_row(
                index,
                prepared,
                source_geometry=None,
                source_failure=source_errors[source_key],
                raw_q=raw_quantiles[prepared.schedule.seed],
                timings={"total_seconds": 0.0},
                error=source_errors[source_key].reason,
            )
        else:
            result = compute_g0_row(
                index,
                prepared,
                source_geometry,
                raw_q=raw_quantiles[prepared.schedule.seed],
                config=config,
            )
        results.append(result)
        stem = (
            f"{index:02d}_{prepared.schedule.category}_{prepared.schedule.defect_type}_"
            f"{prepared.schedule.source_id}_{prepared.schedule.target_id}"
        )
        numeric_path = numeric_directory / f"{stem}.npz"
        np.savez_compressed(numeric_path, **_row_numeric_arrays(result))
        with np.load(numeric_path, allow_pickle=False) as loaded:
            if any(loaded[name].dtype == object for name in loaded.files):
                raise AssertionError("written G0 NPZ contains an object array")
        numeric_paths.append(numeric_path)
        row_path = row_directory / f"{stem}.png"
        _render_row_panel(result, cell_size=256).save(row_path)
        row_paths.append(row_path)
        source_input_lock = _identity_input_lock(prepared.source, dataset_root=dataset_root)
        target_input_lock = _identity_input_lock(prepared.target, dataset_root=dataset_root)
        row_sidecars.append(
            _row_sidecar(
                result,
                numeric_path,
                geometry_contract_lock_digest=geometry_contract_lock_digest,
                reference_set_digest=reference_set_digests[prepared.schedule.category],
                source_input_lock=source_input_lock,
                target_input_lock=target_input_lock,
            )
        )

    combined_rows = tuple(_render_row_panel(result, cell_size=128) for result in results)
    combined_width = combined_rows[0].width
    combined_height = sum(image.height for image in combined_rows)
    combined = Image.new("RGB", (combined_width, combined_height), "black")
    offset = 0
    for image in combined_rows:
        combined.paste(image, (0, offset))
        offset += image.height
    combined_path = destination / "g0_combined_16x12.png"
    combined.save(combined_path)
    legend_path = destination / "g0_legend.png"
    _render_legend().save(legend_path)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current_rss_after = _current_rss_kib()
    artifact_hashes = {
        path.relative_to(destination).as_posix(): _file_hash(path)
        for path in (*row_paths, *numeric_paths, combined_path, legend_path)
    }
    parent_fail_hashes = _parent_g0_fail_hashes(root)
    sidecar: dict[str, object] = {
        "schema_version": "v06_a_g0_rc2_artifact_f3",
        "status": "awaiting_human_review",
        "scientific_status": "awaiting human review",
        "p0_r2": "not_started",
        "context_mode": "no_context",
        "context_version": None,
        "recovery_mode": "none",
        "recovery_version": None,
        "formal_targets_materialized": False,
        "training_authorized": False,
        "formal_gpu_training": "NO-GO",
        "gpu_used": False,
        "actual_device": "cpu",
        "requested_cuda_policy": "not_applicable",
        "candidate_count_per_row": 73_728,
        "nominal_candidate_count_total": 1_179_648,
        "candidate_order": (
            "anchor -> angle residual -> parallel multiplier -> perpendicular multiplier"
        ),
        "candidate_chunk_size": config.candidate_chunk_size,
        "cell_chunk_size": config.cell_chunk_size,
        "row_count": 16,
        "valid_computation_row_count": sum(result.valid for result in results),
        "valid_count_is_not_visual_success": True,
        "shared_quantiles": {str(seed): value for seed, value in raw_quantiles.items()},
        "science_file_hashes": science_file_hashes,
        "rc1_evidence_hashes": _rc1_evidence_hashes(root),
        "parent_g0_fail_evidence": {
            "path": "runs/v06_a/g0_rc2/first_complete_20260830",
            "immutable_file_hashes": parent_fail_hashes,
            "set_digest": _canonical_json_hash(parent_fail_hashes),
        },
        "reference_sets": reference_sets,
        "geometry_contract_lock": geometry_contract_lock,
        "geometry_contract_lock_digest": geometry_contract_lock_digest,
        "geometry_science_lock": {
            "lock_schema": str(config.values["geometry_science_lock_version"]),
            "config_sha256": _file_hash(config.path),
            "code_sha256": _canonical_json_hash(code_hashes),
            "candidate_slot_template_sha256": candidate_slot_template_hash,
            "geometry_contract_lock_digest": geometry_contract_lock_digest,
            "parent_fail_evidence_set_digest": _canonical_json_hash(parent_fail_hashes),
        },
        "artifact_hashes": artifact_hashes,
        "runtime_provenance_not_science_identity": {
            "wall_seconds": time.perf_counter() - started,
            "current_rss_before_kib": current_rss_before,
            "current_rss_after_kib": current_rss_after,
            "ru_maxrss_before_kib": rss_before,
            "ru_maxrss_after_kib": rss_after,
        },
        "rows": row_sidecars,
    }
    _finalize_geometry_science_lock(sidecar)
    _validate_geometry_science_lock(sidecar)
    json_path = destination / "g0_diagnostic.json"
    json_path.write_text(
        json.dumps(
            sidecar,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return G0Artifact(
        destination,
        json_path,
        combined_path,
        legend_path,
        tuple(row_paths),
        tuple(numeric_paths),
        tuple(results),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module-A RC2 CPU-only G0 diagnostic")
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-relpath")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = load_g0_config(arguments.config)
    if arguments.mode == "preflight":
        if arguments.output_root is not None or arguments.output_relpath is not None:
            raise ValueError("G0 preflight does not accept or create an output artifact")
        result = run_g0_preflight(
            config,
            repository_root=arguments.repository_root,
            dataset_root=arguments.dataset_root,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["valid"] else 2
    if arguments.output_root is None or arguments.output_relpath is None:
        raise ValueError("G0 run requires --output-root and --output-relpath")
    artifact = run_g0(
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
                "valid_computation_rows": sum(result.valid for result in artifact.results),
                "status": "awaiting_human_review",
                "p0_r2": "not_started",
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

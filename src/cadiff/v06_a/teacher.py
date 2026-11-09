"""Frozen, diagnostic-only context teacher for the Module-A P0 gate.

This module does not create training targets.  It computes only the fixed P0
diagnostics needed before any A1-CTX work may be authorised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import LoadedIdentity
from .identities import IdentityManifest, P0Pair
from .structure import (
    GeometryField,
    LocalFrame,
    NormalAtlas,
    NormalStructure,
    ObjectFrame,
    sample_continuous_field,
    support_frame,
)
from .targets import (
    CandidateParameters,
    CoverageCertificate,
    GeometryAffine,
    ProposalTarget,
    ScoreEnvelope,
    construct_plain_target,
    local_box_corners,
    local_to_world,
    rectangular_valid_bounds,
    transform_points,
)

_OUTER_RADIUS = 72
_BLIND_GAPS = (32, 48)
_CANONICAL_GAP = 32
_CANDIDATE_STRIDE = 16
_TEMPERATURE = 0.25
_RELATION_WEIGHT = 2.0
_MIN_CONTEXT_FRACTION = 0.75
_GRADIENT_GUARD = 1
_ORIENTATION_BINS = 8

_G0_ANCHOR_STRIDE = 16
_G0_ANGLE_COUNT = 8
_G0_SCALE_MULTIPLIERS = np.array((2.0**-0.5, 1.0, 2.0**0.5), dtype=np.float64)
_G0_TEMPERATURE = 0.35
_G0_CANDIDATE_CHUNK = 256
_G0_CELL_CHUNK = 256
_G0_RELATION_BAND = 0.125
_G0_INVALID_DISTANCE = 10.0
_G0_ORIENTATION_FLOOR = 0.10
_G0_SCALE_EVIDENCE_FLOOR = 0.10
_G0_BOUNDARY_EPSILON = 1e-9


@dataclass(frozen=True)
class GeometryCandidateTable:
    """All 73,728 fixed G0 slots in canonical order."""

    slots: np.ndarray
    parameters: np.ndarray
    matrices: np.ndarray
    inverse_linear: np.ndarray
    relative_rotation: np.ndarray
    target_axis_lift: np.ndarray
    valid: np.ndarray
    invalid_reason: np.ndarray

    def __post_init__(self) -> None:
        count = 73_728
        expected = {
            "slots": (self.slots, (count, 4)),
            "parameters": (self.parameters, (count, 6)),
            "matrices": (self.matrices, (count, 3, 3)),
            "inverse_linear": (self.inverse_linear, (count, 2, 2)),
            "relative_rotation": (self.relative_rotation, (count,)),
            "target_axis_lift": (self.target_axis_lift, (count,)),
            "valid": (self.valid, (count,)),
            "invalid_reason": (self.invalid_reason, (count,)),
        }
        for name, (value, shape) in expected.items():
            if value.shape != shape:
                raise ValueError(f"candidate {name} must have shape {shape}")
            value.setflags(write=False)


@dataclass(frozen=True)
class GeometryScores:
    components: np.ndarray
    distance: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.components.ndim != 2 or self.components.shape[1] != 5:
            raise ValueError("geometry components must have shape [N,5]")
        if self.distance.shape != (self.components.shape[0],):
            raise ValueError("geometry distance must have shape [N]")
        if self.valid.shape != self.distance.shape or self.valid.dtype != bool:
            raise ValueError("geometry score valid mask must be boolean [N]")
        if not np.isfinite(self.components).all() or not np.isfinite(self.distance).all():
            raise ValueError("geometry scores must use finite sentinels")
        for value in (self.components, self.distance, self.valid):
            value.setflags(write=False)


@dataclass(frozen=True)
class QuantileSample:
    raw_q: float
    index: int
    valid_rank: int
    cdf: np.ndarray
    cdf_before: float
    cdf_at: float

    def __post_init__(self) -> None:
        if self.cdf.ndim != 1 or self.cdf.dtype != np.float64:
            raise ValueError("sample CDF must be one-dimensional float64")
        self.cdf.setflags(write=False)


class _TeacherConfig(Protocol):
    category: str
    defect_types: tuple[str, ...]
    image_size: int
    proposal_size: int
    square_dilate_radius: int
    teacher_outer_radius: int
    teacher_blind_gaps: tuple[int, ...]
    teacher_candidate_stride: int
    teacher_temperature: float
    teacher_relation_weight: float
    teacher_min_context_fraction: float


@dataclass(frozen=True)
class ContextDescriptor:
    """One fixed descriptor and the exact pixels from which it was computed."""

    feature: np.ndarray
    used_mask: np.ndarray
    context_fraction: float
    gap: int
    coarse_bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        self.feature.setflags(write=False)
        self.used_mask.setflags(write=False)


@dataclass(frozen=True)
class TeacherDistribution:
    """A finite soft distribution over legal full-canvas stride candidates."""

    gap: int
    candidates: np.ndarray
    probabilities: np.ndarray
    heatmap: np.ndarray
    retained_fractions: np.ndarray
    object_center_overrides: np.ndarray
    expected_bbox_centers: np.ndarray
    source_descriptor: ContextDescriptor

    def __post_init__(self) -> None:
        for array in (
            self.candidates,
            self.probabilities,
            self.heatmap,
            self.retained_fractions,
            self.object_center_overrides,
            self.expected_bbox_centers,
        ):
            array.setflags(write=False)


@dataclass(frozen=True)
class GapDiagnostic:
    gap: int
    entropy: float
    valid_candidate_count: int
    distribution: TeacherDistribution


@dataclass(frozen=True)
class TeacherGapEvidence:
    """Per-gap visual evidence for a diagnostic-only P0 sidecar.

    ``used_mask`` and ``context_fraction`` are retained even when the gap is
    invalid, so a panel can show why the frozen teacher rejected it.
    """

    gap: int
    coarse_bbox: tuple[int, int, int, int]
    used_mask: np.ndarray
    context_fraction: float
    distribution: TeacherDistribution | None
    valid: bool
    error: str | None

    def __post_init__(self) -> None:
        self.used_mask.setflags(write=False)


@dataclass(frozen=True)
class TeacherPairDiagnostic:
    """In-memory P0 evidence for one frozen source-target pair."""

    category: str
    pair: P0Pair
    source: LoadedIdentity
    target: LoadedIdentity
    canonical_gap: int
    gaps: tuple[GapDiagnostic, ...]
    gap_js_divergence: float | None
    sampled_position: tuple[int, int] | None
    sampled_object_center_override: tuple[float, float] | None
    transferred_bbox_center: tuple[float, float] | None
    sampled_retained_fraction: float | None
    transferred_gt: np.ndarray | None
    proposal: np.ndarray | None
    valid: bool
    error: str | None

    def __post_init__(self) -> None:
        if self.transferred_gt is not None:
            self.transferred_gt.setflags(write=False)
        if self.proposal is not None:
            self.proposal.setflags(write=False)


@dataclass(frozen=True)
class _IntegralFeatures:
    values: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class _Footprint:
    x: np.ndarray
    y: np.ndarray
    complete_count: int
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class _CandidateTransform:
    """One candidate u and its exact constructor-frame translation."""

    candidate_position: tuple[int, int]
    object_center_override: tuple[float, float]
    transferred_x: np.ndarray
    transferred_y: np.ndarray
    expected_bbox: tuple[int, int, int, int]
    expected_bbox_center: tuple[float, float]


def _validate_frozen_config(config: _TeacherConfig, *, gap: int | None = None) -> None:
    observed = (
        config.image_size,
        config.proposal_size,
        config.square_dilate_radius,
        config.teacher_outer_radius,
        tuple(config.teacher_blind_gaps),
        config.teacher_candidate_stride,
        config.teacher_temperature,
        config.teacher_relation_weight,
        config.teacher_min_context_fraction,
    )
    expected = (512, 128, 24, 72, (32, 48), 16, 0.25, 2.0, 0.75)
    if observed != expected:
        raise ValueError("teacher configuration must equal the frozen Module-A RC1 contract")
    if gap is not None and gap not in _BLIND_GAPS:
        raise ValueError("teacher gap must be one of the frozen blind gaps (32, 48)")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    if mask.shape != (512, 512) or mask.dtype != bool or not mask.any():
        raise ValueError("source ground truth must be a non-empty 512x512 bool mask")
    y, x = np.nonzero(mask)
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def _bounds_area(bounds: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bounds
    return max(right - left + 1, 0) * max(bottom - top + 1, 0)


def _expanded(bounds: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    return left - margin, top - margin, right + margin, bottom + margin


def _contracted(bounds: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    return left + margin, top + margin, right - margin, bottom - margin


def _clipped(bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = bounds
    left, top, right, bottom = max(left, 0), max(top, 0), min(right, 511), min(bottom, 511)
    return None if left > right or top > bottom else (left, top, right, bottom)


def _rectangle_mask(bounds: tuple[int, int, int, int]) -> np.ndarray:
    result = np.zeros((512, 512), dtype=bool)
    clipped = _clipped(bounds)
    if clipped is not None:
        left, top, right, bottom = clipped
        result[top : bottom + 1, left : right + 1] = True
    return result


def _feature_maps(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("descriptor RGB must be a 512x512 uint8 image")
    values = rgb.astype(np.float64) / 255.0
    channel_sum = values.sum(axis=2, keepdims=True)
    chromaticity = values / np.maximum(channel_sum, 1.0 / 255.0)
    gray = values @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    bins = np.minimum((angle * (_ORIENTATION_BINS / np.pi)).astype(np.intp), 7)
    orientation = np.zeros((512, 512, _ORIENTATION_BINS), dtype=np.float64)
    for index in range(_ORIENTATION_BINS):
        orientation[..., index] = np.where(bins == index, magnitude, 0.0)
    return np.concatenate((chromaticity, chromaticity**2, orientation), axis=2)


def _feature_from_sums(sums: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("context descriptor has no visible pixels")
    means = sums[:3] / count
    variances = np.maximum(sums[3:6] / count - means**2, 0.0)
    histogram = sums[6:]
    total_gradient = float(histogram.sum())
    if total_gradient > 0:
        histogram = histogram / total_gradient
    else:
        histogram = np.full(_ORIENTATION_BINS, 1.0 / _ORIENTATION_BINS)
    feature = np.concatenate((means, np.sqrt(variances), histogram)).astype(np.float32)
    if not np.isfinite(feature).all():
        raise ValueError("context descriptor must be finite")
    return feature


def _safe_gradient_pixels(valid: np.ndarray) -> np.ndarray:
    padded = np.pad(valid, 1, constant_values=False)
    return valid & padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, 1:-1] & padded[2:, 1:-1]


def _ring_masks(
    valid: np.ndarray,
    coarse_bbox: tuple[int, int, int, int],
    *,
    gap: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    outer_bounds = _expanded(coarse_bbox, _OUTER_RADIUS)
    # One extra pixel is excluded so centered finite differences outside the
    # blind region cannot read a pixel from inside that region.
    inner_bounds = _expanded(coarse_bbox, gap + _GRADIENT_GUARD)
    outer = _rectangle_mask(outer_bounds)
    inner = _rectangle_mask(inner_bounds)
    used = valid & outer & ~inner
    # Gradient samples are another pixel away from either context boundary, so
    # every neighbour read by np.gradient is itself in ``used``.  Consequently
    # no RGB value from the blind interior can affect the histogram.
    orientation_used = (
        _safe_gradient_pixels(valid)
        & _rectangle_mask(_contracted(outer_bounds, 1))
        & ~_rectangle_mask(_expanded(inner_bounds, 1))
    )
    nominal = _bounds_area(outer_bounds) - _bounds_area(inner_bounds)
    if nominal <= 0:
        raise ValueError("blind context frame has non-positive nominal area")
    return used, orientation_used, float(used.sum() / nominal)


def describe_source_context(
    source: LoadedIdentity, *, config: _TeacherConfig, gap: int
) -> ContextDescriptor:
    """Describe only the fixed coarse frame outside a GT-bbox blind region."""

    _validate_frozen_config(config, gap=gap)
    if source.record.role != "fewshot" or source.anomaly_gt is None:
        raise ValueError("teacher source must be a fewshot identity with ground truth")
    coarse_bbox = _bbox(source.anomaly_gt)
    used, orientation_used, fraction = _ring_masks(source.valid, coarse_bbox, gap=gap)
    if fraction < _MIN_CONTEXT_FRACTION:
        raise ValueError(f"source context fraction {fraction:.3f} is below frozen minimum 0.750")
    maps = _feature_maps(source.rgb)
    sums = np.concatenate(
        (maps[..., :6][used].sum(axis=0), maps[..., 6:][orientation_used].sum(axis=0))
    )
    feature = _feature_from_sums(sums, int(used.sum()))
    return ContextDescriptor(feature, used, fraction, gap, coarse_bbox)


def _integral_features(rgb: np.ndarray, valid: np.ndarray) -> _IntegralFeatures:
    maps = _feature_maps(rgb)
    maps[..., :6] *= valid[..., None]
    maps[..., 6:] *= _safe_gradient_pixels(valid)[..., None]
    values = np.pad(maps.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    counts = np.pad(valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return _IntegralFeatures(values, counts)


def _rect_sum(integral: np.ndarray, bounds: tuple[int, int, int, int]):
    clipped = _clipped(bounds)
    if clipped is None:
        return np.zeros(integral.shape[2:], dtype=integral.dtype)
    left, top, right, bottom = clipped
    return (
        integral[bottom + 1, right + 1]
        - integral[top, right + 1]
        - integral[bottom + 1, left]
        + integral[top, left]
    )


def _target_descriptor(
    fields: _IntegralFeatures,
    coarse_bbox: tuple[int, int, int, int],
    *,
    gap: int,
) -> tuple[np.ndarray, float]:
    outer = _expanded(coarse_bbox, _OUTER_RADIUS)
    inner = _expanded(coarse_bbox, gap + _GRADIENT_GUARD)
    count = int(_rect_sum(fields.counts[..., None], outer)[0]) - int(
        _rect_sum(fields.counts[..., None], inner)[0]
    )
    nominal = _bounds_area(outer) - _bounds_area(inner)
    fraction = float(count / nominal) if nominal > 0 else 0.0
    if count <= 0:
        raise ValueError("target context descriptor has no visible pixels")
    chromaticity_sums = _rect_sum(fields.values, outer)[:6] - _rect_sum(fields.values, inner)[:6]
    orientation_outer = _contracted(outer, 1)
    orientation_inner = _expanded(inner, 1)
    orientation_sums = (
        _rect_sum(fields.values, orientation_outer)[6:]
        - _rect_sum(fields.values, orientation_inner)[6:]
    )
    sums = np.concatenate((chromaticity_sums, orientation_sums))
    return _feature_from_sums(sums, count), fraction


def _relative_footprint(
    source_gt: np.ndarray,
    source_frame: ObjectFrame,
    target_frame: ObjectFrame,
    *,
    maximum_legal_count: int,
) -> _Footprint:
    """Return an exact, bounded representation of the relative transfer raster.

    The theoretical raster may span roughly 131k pixels per axis at the real
    frame limits.  Axis inverse maps stay one-dimensional; no extended 2-D
    raster is ever allocated.  Coordinates are retained only when they could
    enter a 512 canvas for one of the frozen stride candidates.
    """

    y, x = np.nonzero(source_gt)
    ratio_x = target_frame.scale_x / source_frame.scale_x
    ratio_y = target_frame.scale_y / source_frame.scale_y
    mapped_x = (x - source_frame.center_x) * ratio_x
    mapped_y = (y - source_frame.center_y) * ratio_y
    min_x = int(np.floor(mapped_x.min() - abs(ratio_x) / 2.0 - 1.0))
    max_x = int(np.ceil(mapped_x.max() + abs(ratio_x) / 2.0 + 1.0))
    min_y = int(np.floor(mapped_y.min() - abs(ratio_y) / 2.0 - 1.0))
    max_y = int(np.ceil(mapped_y.max() + abs(ratio_y) / 2.0 + 1.0))
    target_x_full = np.arange(min_x, max_x + 1, dtype=np.float64)
    target_y_full = np.arange(min_y, max_y + 1, dtype=np.float64)
    source_x_full = np.rint(
        target_x_full * source_frame.scale_x / target_frame.scale_x + source_frame.center_x
    ).astype(int)
    source_y_full = np.rint(
        target_y_full * source_frame.scale_y / target_frame.scale_y + source_frame.center_y
    ).astype(int)
    in_x_full = (source_x_full >= 0) & (source_x_full < 512)
    in_y_full = (source_y_full >= 0) & (source_y_full < 512)
    x_counts = np.bincount(source_x_full[in_x_full], minlength=512)
    y_counts = np.bincount(source_y_full[in_y_full], minlength=512)
    complete_count = int((source_gt.astype(np.int64) * y_counts[:, None] * x_counts[None, :]).sum())
    if complete_count <= 0:
        raise ValueError("teacher transfer footprint is empty")
    if complete_count > maximum_legal_count:
        raise ValueError("teacher transfer footprint cannot retain 0.950 on any valid 512 canvas")

    reachable_columns = x_counts > 0
    reachable_rows = y_counts > 0
    source_columns = source_gt[reachable_rows].any(axis=0) & reachable_columns
    source_rows = source_gt[:, reachable_columns].any(axis=1) & reachable_rows
    x_hits_full = in_x_full & source_columns[np.clip(source_x_full, 0, 511)]
    y_hits_full = in_y_full & source_rows[np.clip(source_y_full, 0, 511)]
    bounds = (
        int(target_x_full[x_hits_full].min()),
        int(target_y_full[y_hits_full].min()),
        int(target_x_full[x_hits_full].max()),
        int(target_y_full[y_hits_full].max()),
    )

    maximum_center = 512 - _CANDIDATE_STRIDE
    relative_center_x = (bounds[0] + bounds[2]) / 2.0
    relative_center_y = (bounds[1] + bounds[3]) / 2.0
    minimum_override_x = int(np.rint(-relative_center_x))
    maximum_override_x = int(np.rint(maximum_center - relative_center_x))
    minimum_override_y = int(np.rint(-relative_center_y))
    maximum_override_y = int(np.rint(maximum_center - relative_center_y))
    visible_minimum_x, visible_maximum_x = -maximum_override_x, 511 - minimum_override_x
    visible_minimum_y, visible_maximum_y = -maximum_override_y, 511 - minimum_override_y
    visible_x = np.arange(
        max(min_x, visible_minimum_x), min(max_x, visible_maximum_x) + 1, dtype=np.int32
    )
    visible_y = np.arange(
        max(min_y, visible_minimum_y), min(max_y, visible_maximum_y) + 1, dtype=np.int32
    )
    source_x = np.rint(
        visible_x * source_frame.scale_x / target_frame.scale_x + source_frame.center_x
    ).astype(int)
    source_y = np.rint(
        visible_y * source_frame.scale_y / target_frame.scale_y + source_frame.center_y
    ).astype(int)
    valid_x = (source_x >= 0) & (source_x < 512)
    collected_x: list[np.ndarray] = []
    collected_y: list[np.ndarray] = []
    for offset_y, source_row in zip(visible_y, source_y, strict=True):
        if not 0 <= source_row < 512:
            continue
        hits = valid_x & source_gt[source_row, np.clip(source_x, 0, 511)]
        if hits.any():
            row_x = visible_x[hits]
            collected_x.append(row_x)
            collected_y.append(np.full(row_x.size, offset_y, dtype=np.int32))
    if not collected_x:
        raise ValueError("teacher transfer footprint cannot enter a frozen candidate canvas")
    return _Footprint(
        np.concatenate(collected_x),
        np.concatenate(collected_y),
        complete_count,
        bounds,
    )


def _candidate_transform(
    footprint: _Footprint, candidate_position: tuple[int, int]
) -> _CandidateTransform:
    """Map anomaly-centre candidate ``u`` to the constructor's object-frame centre.

    The base footprint is defined at object-frame centre zero.  Integer frame
    translations preserve the exact nearest-neighbour raster phase.  We choose
    the unique deterministic integer translation whose transferred bbox centre
    is nearest to ``u``; the remaining quantisation error is at most 0.5 px.
    """

    candidate_x, candidate_y = candidate_position
    if (
        isinstance(candidate_x, bool)
        or isinstance(candidate_y, bool)
        or not isinstance(candidate_x, int)
        or not isinstance(candidate_y, int)
    ):
        raise ValueError("teacher candidate position must use integer canvas coordinates")
    left, top, right, bottom = footprint.bounds
    relative_center_x = (left + right) / 2.0
    relative_center_y = (top + bottom) / 2.0
    override_x = int(np.rint(candidate_x - relative_center_x))
    override_y = int(np.rint(candidate_y - relative_center_y))
    expected_bbox = (
        left + override_x,
        top + override_y,
        right + override_x,
        bottom + override_y,
    )
    actual_center = (
        (expected_bbox[0] + expected_bbox[2]) / 2.0,
        (expected_bbox[1] + expected_bbox[3]) / 2.0,
    )
    if abs(actual_center[0] - candidate_x) > 0.5 or abs(actual_center[1] - candidate_y) > 0.5:
        raise AssertionError("nearest-raster candidate alignment exceeded 0.5 pixel")
    return _CandidateTransform(
        (candidate_x, candidate_y),
        (float(override_x), float(override_y)),
        footprint.x.astype(np.int64) + override_x,
        footprint.y.astype(np.int64) + override_y,
        expected_bbox,
        actual_center,
    )


def _relation_from_support(support: np.ndarray, x: float, y: float) -> float:
    column, row = int(np.rint(x)), int(np.rint(y))
    if not (0 <= column < 512 and 0 <= row < 512):
        return -1.0
    inside = bool(support[row, column])
    top, bottom = max(row - 1, 0), min(row + 2, 512)
    left, right = max(column - 1, 0), min(column + 2, 512)
    neighbourhood = support[top:bottom, left:right]
    if neighbourhood.any() and not neighbourhood.all():
        return 0.0
    return 1.0 if inside else -1.0


def _relation_from_sdf(structure: NormalStructure, x: float, y: float) -> float:
    column = min(max(int(np.floor(x / 4.0)), 0), 127)
    row = min(max(int(np.floor(y / 4.0)), 0), 127)
    value = float(structure.sdf_128[row, column])
    boundary_width = 4.0 / max(structure.frame.scale_x, structure.frame.scale_y, 1.0)
    if abs(value) <= boundary_width:
        return 0.0
    return 1.0 if value > 0 else -1.0


def _validate_distribution_inputs(
    source: LoadedIdentity,
    source_atlas: NormalAtlas,
    target: LoadedIdentity,
    target_structure: NormalStructure,
    *,
    config: _TeacherConfig,
    identity_manifest: IdentityManifest,
) -> None:
    if (
        identity_manifest.category != config.category
        or identity_manifest.defect_types != config.defect_types
    ):
        raise ValueError("teacher manifest must agree with the experiment")
    if not source_atlas.matches_manifest(identity_manifest):
        raise ValueError("teacher atlas must match the frozen identity manifest")
    if source.record not in identity_manifest.fewshot:
        raise ValueError("teacher source must exactly match a frozen fewshot identity")
    expected_targets = (*identity_manifest.normal_fit, *identity_manifest.normal_dev)
    if target.record not in expected_targets:
        raise ValueError("teacher target must exactly match a frozen train-normal identity")
    if source.record.sample_id == target.record.sample_id:
        raise ValueError("teacher source and target identities must differ")
    if target_structure.normal.record != target.record or not np.array_equal(
        target_structure.valid, target.valid
    ):
        raise ValueError("teacher target structure must belong to the target identity")
    if target_structure.frame != support_frame(target_structure.support):
        raise ValueError("teacher target frame must be derived from its supplied soft support")


def build_teacher_distribution(
    source: LoadedIdentity,
    source_atlas: NormalAtlas,
    target: LoadedIdentity,
    target_structure: NormalStructure,
    *,
    config: _TeacherConfig,
    identity_manifest: IdentityManifest,
    gap: int,
) -> TeacherDistribution:
    """Compute the diagnostic soft distribution without selecting a best candidate."""

    _validate_frozen_config(config, gap=gap)
    _validate_distribution_inputs(
        source,
        source_atlas,
        target,
        target_structure,
        config=config,
        identity_manifest=identity_manifest,
    )
    source_descriptor = describe_source_context(source, config=config, gap=gap)
    assert source.anomaly_gt is not None
    source_frame = support_frame(source_atlas.support)
    footprint = _relative_footprint(
        source.anomaly_gt,
        source_frame,
        target_structure.frame,
        maximum_legal_count=int(np.floor(target.valid.sum() / 0.95)),
    )
    fields = _integral_features(target.rgb, target.valid)
    source_bbox = source_descriptor.coarse_bbox
    source_center_x = (source_bbox[0] + source_bbox[2]) / 2.0
    source_center_y = (source_bbox[1] + source_bbox[3]) / 2.0
    source_relation = _relation_from_support(source_atlas.support, source_center_x, source_center_y)

    candidates: list[tuple[int, int]] = []
    features: list[np.ndarray] = []
    relations: list[float] = []
    retained_fractions: list[float] = []
    object_center_overrides: list[tuple[float, float]] = []
    expected_bbox_centers: list[tuple[float, float]] = []
    for center_y in range(0, 512, _CANDIDATE_STRIDE):
        for center_x in range(0, 512, _CANDIDATE_STRIDE):
            if not target.valid[center_y, center_x]:
                continue
            transform = _candidate_transform(footprint, (center_x, center_y))
            transferred_x = transform.transferred_x
            transferred_y = transform.transferred_y
            on_canvas = (
                (transferred_x >= 0)
                & (transferred_x < 512)
                & (transferred_y >= 0)
                & (transferred_y < 512)
            )
            retained_pixels = np.zeros(on_canvas.shape, dtype=bool)
            if on_canvas.any():
                retained_pixels[on_canvas] = target.valid[
                    transferred_y[on_canvas].astype(np.intp),
                    transferred_x[on_canvas].astype(np.intp),
                ]
            retained = int(retained_pixels.sum())
            retained_fraction = float(retained / footprint.complete_count)
            if retained_fraction < 0.95:
                continue
            retained_x = transferred_x[retained_pixels]
            retained_y = transferred_y[retained_pixels]
            coarse_bbox = (
                int(retained_x.min()),
                int(retained_y.min()),
                int(retained_x.max()),
                int(retained_y.max()),
            )
            retained_bbox_center = (
                (coarse_bbox[0] + coarse_bbox[2]) / 2.0,
                (coarse_bbox[1] + coarse_bbox[3]) / 2.0,
            )
            if (
                abs(retained_bbox_center[0] - center_x) > 1.0
                or abs(retained_bbox_center[1] - center_y) > 1.0
            ):
                continue
            try:
                feature, context_fraction = _target_descriptor(fields, coarse_bbox, gap=gap)
            except ValueError:
                continue
            if context_fraction < _MIN_CONTEXT_FRACTION:
                continue
            candidates.append((center_x, center_y))
            features.append(feature)
            relations.append(
                abs(source_relation - _relation_from_sdf(target_structure, center_x, center_y))
                / 2.0
            )
            retained_fractions.append(retained_fraction)
            object_center_overrides.append(transform.object_center_override)
            expected_bbox_centers.append(retained_bbox_center)
    if len(candidates) < 2:
        raise ValueError("teacher requires at least two legal soft-distribution candidates")

    feature_matrix = np.stack(features).astype(np.float64)
    source_feature = source_descriptor.feature.astype(np.float64)
    denominator = np.linalg.norm(feature_matrix, axis=1) * np.linalg.norm(source_feature)
    similarities = (feature_matrix @ source_feature) / np.maximum(denominator, 1e-12)
    logits = (similarities - _RELATION_WEIGHT * np.asarray(relations)) / _TEMPERATURE
    logits -= logits.max()
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("teacher probabilities must be finite and normalizable")
    candidate_array = np.asarray(candidates, dtype=np.int16)
    heatmap = np.zeros((512, 512), dtype=np.float64)
    heatmap[candidate_array[:, 1], candidate_array[:, 0]] = probabilities
    return TeacherDistribution(
        gap,
        candidate_array,
        probabilities,
        heatmap,
        np.asarray(retained_fractions, dtype=np.float64),
        np.asarray(object_center_overrides, dtype=np.float64),
        np.asarray(expected_bbox_centers, dtype=np.float64),
        source_descriptor,
    )


def inspect_teacher_gap(
    source: LoadedIdentity,
    source_atlas: NormalAtlas,
    target: LoadedIdentity,
    target_structure: NormalStructure,
    *,
    config: _TeacherConfig,
    identity_manifest: IdentityManifest,
    gap: int,
) -> TeacherGapEvidence:
    """Return independent visual evidence for one frozen blind gap.

    This is diagnostic-only: it neither samples a position nor constructs a
    target.  Unlike the historical pair diagnostic, a failure at one gap does
    not discard the other gap's observable ring evidence.
    """

    _validate_frozen_config(config, gap=gap)
    _validate_distribution_inputs(
        source,
        source_atlas,
        target,
        target_structure,
        config=config,
        identity_manifest=identity_manifest,
    )
    if source.anomaly_gt is None:
        raise ValueError("teacher source must have a ground truth for gap evidence")
    coarse_bbox = _bbox(source.anomaly_gt)
    used_mask, _orientation_used, context_fraction = _ring_masks(source.valid, coarse_bbox, gap=gap)
    try:
        distribution = build_teacher_distribution(
            source,
            source_atlas,
            target,
            target_structure,
            config=config,
            identity_manifest=identity_manifest,
            gap=gap,
        )
    except ValueError as error:
        return TeacherGapEvidence(
            gap,
            coarse_bbox,
            used_mask,
            context_fraction,
            None,
            False,
            str(error),
        )
    return TeacherGapEvidence(
        gap,
        coarse_bbox,
        used_mask,
        context_fraction,
        distribution,
        True,
        None,
    )


def sample_teacher_position(distribution: TeacherDistribution, *, seed: int) -> tuple[int, int]:
    """Draw exactly one categorical sample; this function has no retry path."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("teacher seed must be an integer")
    index = int(
        np.random.default_rng(seed).choice(
            distribution.candidates.shape[0], p=distribution.probabilities
        )
    )
    x, y = distribution.candidates[index]
    return int(x), int(y)


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities > 0
    return float(-(probabilities[positive] * np.log(probabilities[positive])).sum())


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left_values, right_values = left.ravel(), right.ravel()
    middle = (left_values + right_values) / 2.0

    def divergence(values: np.ndarray) -> float:
        positive = values > 0
        return float((values[positive] * np.log(values[positive] / middle[positive])).sum())

    return (divergence(left_values) + divergence(right_values)) / 2.0


def _validate_p0_pair(
    pair: P0Pair,
    source: LoadedIdentity,
    target: LoadedIdentity,
    *,
    identity_manifest: IdentityManifest,
) -> None:
    if pair not in identity_manifest.p0_pairs:
        raise ValueError("teacher pair must exactly match the frozen P0 schedule")
    expected_source = next(
        record
        for record in identity_manifest.fewshot
        if record.defect_type == pair.defect_type and record.item_id == pair.source_id
    )
    if source.record != expected_source:
        raise ValueError("source must exactly match the frozen P0 source identity")
    expected_target = next(
        record
        for record in (*identity_manifest.normal_fit, *identity_manifest.normal_dev)
        if record.item_id == pair.target_normal_id
    )
    if target.record != expected_target:
        raise ValueError("target must exactly match the frozen train-normal target identity")
    if source.record.sample_id == target.record.sample_id:
        raise ValueError("P0 source and target identities must differ")


def diagnose_teacher_pair(
    pair: P0Pair,
    source: LoadedIdentity,
    source_atlas: NormalAtlas,
    target: LoadedIdentity,
    target_structure: NormalStructure,
    *,
    config: _TeacherConfig,
    identity_manifest: IdentityManifest,
) -> TeacherPairDiagnostic:
    """Run both blind gaps and one canonical sample for one fixed P0 pair."""

    _validate_frozen_config(config)
    _validate_p0_pair(pair, source, target, identity_manifest=identity_manifest)
    _validate_distribution_inputs(
        source,
        source_atlas,
        target,
        target_structure,
        config=config,
        identity_manifest=identity_manifest,
    )
    distributions = tuple(
        build_teacher_distribution(
            source,
            source_atlas,
            target,
            target_structure,
            config=config,
            identity_manifest=identity_manifest,
            gap=gap,
        )
        for gap in _BLIND_GAPS
    )
    gaps = tuple(
        GapDiagnostic(
            distribution.gap,
            _entropy(distribution.probabilities),
            int(distribution.candidates.shape[0]),
            distribution,
        )
        for distribution in distributions
    )
    stability = _js_divergence(distributions[0].heatmap, distributions[1].heatmap)
    canonical = distributions[0]
    sampled_position = sample_teacher_position(canonical, seed=pair.seed)
    matches = np.flatnonzero(np.all(canonical.candidates == np.asarray(sampled_position), axis=1))
    sampled_index = int(matches[0])
    sampled_retained = float(canonical.retained_fractions[sampled_index])
    sampled_override = tuple(map(float, canonical.object_center_overrides[sampled_index]))
    proposal_target: ProposalTarget | None = None
    transferred_bbox_center: tuple[float, float] | None = None
    error: str | None = None
    try:
        constructed = construct_plain_target(
            source,
            source_atlas,
            target,
            target_structure,
            config,
            identity_manifest,
            target_center_override=sampled_override,
        )
        transferred_bbox = _bbox(constructed.transferred_gt_512)
        transferred_bbox_center = (
            (transferred_bbox[0] + transferred_bbox[2]) / 2.0,
            (transferred_bbox[1] + transferred_bbox[3]) / 2.0,
        )
        if (
            abs(transferred_bbox_center[0] - sampled_position[0]) > 1.0
            or abs(transferred_bbox_center[1] - sampled_position[1]) > 1.0
        ):
            raise ValueError("transferred GT bbox centre does not align with sampled u")
        proposal_target = constructed
    except ValueError as caught:
        error = str(caught)
    if proposal_target is not None:
        sampled_retained = proposal_target.retained_fraction
    return TeacherPairDiagnostic(
        category=config.category,
        pair=pair,
        source=source,
        target=target,
        canonical_gap=_CANONICAL_GAP,
        gaps=gaps,
        gap_js_divergence=stability,
        sampled_position=sampled_position,
        sampled_object_center_override=sampled_override,
        transferred_bbox_center=transferred_bbox_center,
        sampled_retained_fraction=sampled_retained,
        transferred_gt=(
            None if proposal_target is None else proposal_target.transferred_gt_512.copy()
        ),
        proposal=None if proposal_target is None else proposal_target.proposal_512.copy(),
        valid=proposal_target is not None,
        error=error,
    )


# ---------------------------------------------------------------------------
# RC2 G0 geometry-only path.  The RC1 context teacher above remains frozen.


_sample_field = sample_continuous_field


def _candidate_slot_order() -> np.ndarray:
    anchor, residual, parallel, perpendicular = np.indices((1024, 8, 3, 3))
    return np.column_stack(
        (
            anchor.ravel(),
            residual.ravel(),
            parallel.ravel(),
            perpendicular.ravel(),
        )
    ).astype(np.int16)


def enumerate_geometry_candidates(
    target_field: GeometryField,
    source_frame: LocalFrame,
    coverage: CoverageCertificate,
    target_valid: np.ndarray,
) -> GeometryCandidateTable:
    """Enumerate the exact 1024x8x3x3 G0 family without deduplication."""

    rectangular_valid_bounds(target_valid)
    slots = _candidate_slot_order()
    anchor_index = slots[:, 0].astype(np.intp)
    anchor_row, anchor_column = np.divmod(np.arange(1024), 32)
    anchor_centers = np.column_stack(
        (anchor_column * _G0_ANCHOR_STRIDE + 8.0, anchor_row * _G0_ANCHOR_STRIDE + 8.0)
    ).astype(np.float64)
    anchor_geometry = _sample_field(target_field.g, anchor_centers)
    anchor_rho = np.clip(anchor_geometry[:, 2], 0.0, 1.0)
    anchor_unknown = anchor_rho < 0.10
    anchor_alpha = 0.5 * np.arctan2(anchor_geometry[:, 1], anchor_geometry[:, 0])
    anchor_alpha[anchor_unknown] = 0.0
    base_parallel = np.exp(anchor_geometry[:, 3])
    base_perpendicular = np.exp(anchor_geometry[:, 4])

    residual = slots[:, 1].astype(np.float64) * (math.pi / _G0_ANGLE_COUNT)
    parameters = np.empty((slots.shape[0], 6), dtype=np.float64)
    parameters[:, :2] = anchor_centers[anchor_index]
    parameters[:, 2] = anchor_alpha[anchor_index] + residual
    parameters[:, 3] = (
        base_parallel[anchor_index] * _G0_SCALE_MULTIPLIERS[slots[:, 2].astype(np.intp)]
    )
    parameters[:, 4] = (
        base_perpendicular[anchor_index] * _G0_SCALE_MULTIPLIERS[slots[:, 3].astype(np.intp)]
    )
    parameters[:, 5] = anchor_unknown[anchor_index].astype(np.float64)

    valid = np.ones(slots.shape[0], dtype=bool)
    invalid_reason = np.zeros(slots.shape[0], dtype=np.uint8)
    raw_invalid = (
        (parameters[:, 3] < 4.0)
        | (parameters[:, 3] > 64.0)
        | (parameters[:, 4] < 4.0)
        | (parameters[:, 4] > 64.0)
    )
    valid[raw_invalid] = False
    invalid_reason[raw_invalid] = 1
    ratio_parallel = parameters[:, 3] / source_frame.s_parallel
    ratio_perpendicular = parameters[:, 4] / source_frame.s_perp
    ratio_invalid = (
        (ratio_parallel < 0.5)
        | (ratio_parallel > 2.0)
        | (ratio_perpendicular < 0.5)
        | (ratio_perpendicular > 2.0)
    ) & valid
    valid[ratio_invalid] = False
    invalid_reason[ratio_invalid] = 2

    matrices = np.zeros((slots.shape[0], 3, 3), dtype=np.float64)
    inverse_linear = np.zeros((slots.shape[0], 2, 2), dtype=np.float64)
    relative = np.zeros(slots.shape[0], dtype=np.float64)
    target_lift = np.zeros(slots.shape[0], dtype=np.float64)
    coverage_indices = np.flatnonzero(valid)
    if source_frame.orientation_unknown and not math.isclose(
        source_frame.s_parallel, source_frame.s_perp, rel_tol=0.0, abs_tol=1e-12
    ):
        invalid_reason[coverage_indices] = 2
        valid[coverage_indices] = False
        coverage_indices = np.empty(0, dtype=np.intp)
    if coverage_indices.size:
        source_axis = 0.0 if source_frame.orientation_unknown else source_frame.alpha
        selected_alpha = parameters[coverage_indices, 2]
        selected_relative = (selected_alpha - source_axis + math.pi / 2.0) % math.pi - math.pi / 2.0
        selected_lift = source_axis + selected_relative
        cosine, sine = np.cos(selected_lift), np.sin(selected_lift)
        target_rotation = np.empty((coverage_indices.size, 2, 2), dtype=np.float64)
        target_rotation[:, 0, 0] = cosine
        target_rotation[:, 0, 1] = -sine
        target_rotation[:, 1, 0] = sine
        target_rotation[:, 1, 1] = cosine
        source_rotation = _candidate_rotation(source_axis)
        selected_ratio = np.column_stack(
            (ratio_parallel[coverage_indices], ratio_perpendicular[coverage_indices])
        )
        scaled_target = target_rotation * selected_ratio[:, None, :]
        selected_linear = scaled_target @ source_rotation.T
        scaled_source = source_rotation[None, :, :] / selected_ratio[:, None, :]
        selected_inverse = scaled_source @ np.swapaxes(target_rotation, 1, 2)
        selected_matrix = np.zeros((coverage_indices.size, 3, 3), dtype=np.float64)
        selected_matrix[:, 2, 2] = 1.0
        selected_matrix[:, :2, :2] = selected_linear
        source_center = np.asarray(source_frame.center, dtype=np.float64)
        selected_center = parameters[coverage_indices, :2]
        selected_matrix[:, :2, 2] = selected_center - np.einsum(
            "nij,j->ni", selected_linear, source_center
        )

        source_corners = local_to_world(local_box_corners(coverage.local_bounds), source_frame)
        target_corners = (
            np.einsum("nij,pj->npi", selected_linear, source_corners)
            + selected_matrix[:, None, :2, 2]
        )
        left, top, right, bottom = rectangular_valid_bounds(target_valid)
        retained = (
            np.all(target_corners[..., 0] >= left - _G0_BOUNDARY_EPSILON, axis=1)
            & np.all(target_corners[..., 0] <= right + _G0_BOUNDARY_EPSILON, axis=1)
            & np.all(target_corners[..., 1] >= top - _G0_BOUNDARY_EPSILON, axis=1)
            & np.all(target_corners[..., 1] <= bottom + _G0_BOUNDARY_EPSILON, axis=1)
        )
        rejected = coverage_indices[~retained]
        valid[rejected] = False
        invalid_reason[rejected] = 3
        accepted = coverage_indices[retained]
        matrices[accepted] = selected_matrix[retained]
        inverse_linear[accepted] = selected_inverse[retained]
        relative[accepted] = selected_relative[retained]
        target_lift[accepted] = selected_lift[retained]
    return GeometryCandidateTable(
        slots=slots,
        parameters=parameters,
        matrices=matrices,
        inverse_linear=inverse_linear,
        relative_rotation=relative,
        target_axis_lift=target_lift,
        valid=valid,
        invalid_reason=invalid_reason,
    )


def orientation_mismatch(
    source_alpha: float,
    source_rho: float,
    target_alpha: float,
    target_rho: float,
) -> float:
    """Confidence mismatch plus confidence-gated double-angle mismatch."""

    source_confidence = float(np.clip(source_rho, 0.0, 1.0))
    target_confidence = float(np.clip(target_rho, 0.0, 1.0))
    if source_confidence < _G0_ORIENTATION_FLOOR:
        source_confidence = 0.0
    if target_confidence < _G0_ORIENTATION_FLOOR:
        target_confidence = 0.0
    confidence_loss = abs(source_confidence - target_confidence)
    angular_loss = (1.0 - math.cos(2.0 * (target_alpha - source_alpha))) / 2.0
    return float(
        np.clip(
            confidence_loss + min(source_confidence, target_confidence) * angular_loss,
            0.0,
            1.0,
        )
    )


def stable_geometry_softmax(
    distance: np.ndarray, valid: np.ndarray, *, temperature: float = _G0_TEMPERATURE
) -> np.ndarray:
    values = np.asarray(distance, dtype=np.float64)
    legal = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or legal.shape != values.shape:
        raise ValueError("geometry distance and valid mask must be one-dimensional peers")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("geometry temperature must be positive and finite")
    if not np.isfinite(values).all():
        raise ValueError("geometry distance must use finite invalid sentinels")
    probability = np.zeros(values.shape, dtype=np.float64)
    if not legal.any():
        return probability
    legal_distance = values[legal]
    logits = -(legal_distance - legal_distance.min()) / temperature
    weights = np.exp(logits, dtype=np.float64)
    denominator = weights.sum(dtype=np.float64)
    if not math.isfinite(float(denominator)) or denominator <= 0.0:
        raise ValueError("geometry probabilities are not normalizable")
    probability[legal] = weights / denominator
    return probability


def shared_raw_quantile(seed: int) -> float:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("G0 sampling seed must be a non-negative signed 64-bit integer")
    return float(np.random.Generator(np.random.PCG64(seed)).random())


def inverse_cdf_sample(
    probabilities: np.ndarray, valid: np.ndarray, raw_q: float
) -> QuantileSample:
    probability = np.asarray(probabilities, dtype=np.float64)
    legal = np.asarray(valid, dtype=bool)
    if probability.ndim != 1 or legal.shape != probability.shape:
        raise ValueError("probabilities and valid must be one-dimensional peers")
    if not math.isfinite(raw_q) or not 0.0 <= raw_q < 1.0:
        raise ValueError("raw quantile must be finite in [0,1)")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if np.any(probability[~legal] != 0.0):
        raise ValueError("invalid candidate probability must be exactly zero")
    cdf = np.zeros(probability.shape, dtype=np.float64)
    if not legal.any():
        return QuantileSample(raw_q, -1, -1, cdf, 0.0, 0.0)
    total = probability.sum(dtype=np.float64)
    if not math.isfinite(float(total)) or total <= 0.0:
        raise ValueError("valid candidates require positive probability mass")
    normalized = probability / total
    cdf = np.cumsum(normalized, dtype=np.float64)
    last_valid = int(np.flatnonzero(legal)[-1])
    cdf[last_valid:] = 1.0
    index = int(np.searchsorted(cdf, raw_q, side="right"))
    if index > last_valid or not legal[index] or normalized[index] <= 0.0:
        # Numerical plateaux (including internal invalid slots) are skipped in
        # canonical slot order without consuming another random number.
        candidates = np.flatnonzero(legal & (normalized > 0.0) & (cdf > raw_q))
        if candidates.size == 0:
            raise ValueError("inverse CDF has no positive-mass valid slot")
        index = int(candidates[0])
    legal_indices = np.flatnonzero(legal)
    valid_rank = int(np.searchsorted(legal_indices, index))
    before = float(cdf[index - 1]) if index > 0 else 0.0
    return QuantileSample(raw_q, index, valid_rank, cdf, before, float(cdf[index]))


def _angles_from_geometry(samples: np.ndarray) -> np.ndarray:
    return 0.5 * np.arctan2(samples[..., 1], samples[..., 0])


def _sigma_from_geometry(samples: np.ndarray) -> np.ndarray:
    alpha = _angles_from_geometry(samples)
    parallel = np.exp(samples[..., 3]) ** 2
    perpendicular = np.exp(samples[..., 4]) ** 2
    cosine, sine = np.cos(alpha), np.sin(alpha)
    sigma = np.empty((*samples.shape[:-1], 2, 2), dtype=np.float64)
    sigma[..., 0, 0] = cosine**2 * parallel + sine**2 * perpendicular
    sigma[..., 1, 1] = sine**2 * parallel + cosine**2 * perpendicular
    sigma[..., 0, 1] = sigma[..., 1, 0] = cosine * sine * (parallel - perpendicular)
    return sigma


def _spd_log2(matrix: np.ndarray) -> np.ndarray:
    """Symmetric matrix logarithm for positive-definite 2x2 batches."""

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return np.einsum("...ik,...k,...jk->...ij", eigenvectors, np.log(eigenvalues), eigenvectors)


def _relation_simplex(normalized_sdf: np.ndarray) -> np.ndarray:
    inside = np.clip(normalized_sdf / _G0_RELATION_BAND, 0.0, 1.0)
    boundary = 1.0 - np.clip(np.abs(normalized_sdf) / _G0_RELATION_BAND, 0.0, 1.0)
    outside = np.clip(-normalized_sdf / _G0_RELATION_BAND, 0.0, 1.0)
    return np.stack((inside, boundary, outside), axis=-1)


def _relation_bins(normalized_sdf: np.ndarray) -> np.ndarray:
    bins = np.ones(normalized_sdf.shape, dtype=np.int8)
    bins[normalized_sdf > _G0_RELATION_BAND] = 0
    bins[normalized_sdf < -_G0_RELATION_BAND] = 2
    return bins


def _occupied_bin_mean(bin_sums: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    occupied = bin_counts > 0.0
    if not occupied.any():
        raise ValueError("SDF relation profile requires at least one occupied source bin")
    return (bin_sums[:, occupied] / bin_counts[occupied]).mean(axis=1)


def _candidate_rotation(alpha: float) -> np.ndarray:
    cosine, sine = math.cos(alpha), math.sin(alpha)
    return np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _full_stencil_element_validity(
    target_points: np.ndarray,
    linear: np.ndarray,
    source_frame: LocalFrame,
    valid_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Check complete transformed 4-pixel stencil elements against V.

    ``target_points`` contains transformed element centres.  Each source-local
    element is a 4x4 square, so its half-extent is two pixels on both axes.
    For rectangular ``V`` the axis-aligned extent of the transformed
    parallelogram is exact and avoids materialising a [...,4,2] corner array.
    """

    if target_points.shape[-1] != 2:
        raise ValueError("target stencil points must have trailing shape 2")
    if linear.shape[-2:] != (2, 2):
        raise ValueError("affine linear part must have trailing shape 2x2")
    source_axis = 0.0 if source_frame.orientation_unknown else source_frame.alpha
    source_rotation = _candidate_rotation(source_axis)
    local_to_target = np.matmul(linear, source_rotation)
    half_extent = 2.0 * np.abs(local_to_target).sum(axis=-1)
    left, top, right, bottom = valid_bounds
    return (
        (target_points[..., 0] - half_extent[..., 0, None] >= left - _G0_BOUNDARY_EPSILON)
        & (target_points[..., 0] + half_extent[..., 0, None] <= right + _G0_BOUNDARY_EPSILON)
        & (target_points[..., 1] - half_extent[..., 1, None] >= top - _G0_BOUNDARY_EPSILON)
        & (target_points[..., 1] + half_extent[..., 1, None] <= bottom + _G0_BOUNDARY_EPSILON)
    )


def score_geometry_affines(
    affines: tuple[GeometryAffine, ...],
    envelope: ScoreEnvelope,
    source_frame: LocalFrame,
    source_field: GeometryField,
    source_sdf_128: np.ndarray,
    target_field: GeometryField,
    target_sdf_128: np.ndarray,
    target_valid: np.ndarray,
    *,
    candidate_chunk_size: int = _G0_CANDIDATE_CHUNK,
    stencil_chunk_size: int = _G0_CELL_CHUNK,
    source_stencil_observable: np.ndarray | None = None,
) -> GeometryScores:
    """Score full filled-envelope layouts for a bounded affine batch."""

    count = len(affines)
    parameters = np.zeros((count, 6), dtype=np.float64)
    matrices = np.zeros((count, 3, 3), dtype=np.float64)
    target_axis_lift = np.zeros(count, dtype=np.float64)
    legal = np.zeros(count, dtype=bool)
    for index, affine in enumerate(affines):
        candidate = affine.candidate
        parameters[index] = (
            candidate.center_x,
            candidate.center_y,
            candidate.alpha,
            candidate.s_parallel,
            candidate.s_perp,
            float(candidate.orientation_unknown),
        )
        if affine.valid:
            matrices[index] = affine.matrix
            target_axis_lift[index] = affine.target_axis_lift
            legal[index] = True
    return _score_geometry_arrays(
        parameters,
        matrices,
        target_axis_lift,
        legal,
        envelope,
        source_frame,
        source_field,
        source_sdf_128,
        target_field,
        target_sdf_128,
        target_valid,
        candidate_chunk_size=candidate_chunk_size,
        stencil_chunk_size=stencil_chunk_size,
        source_stencil_observable=source_stencil_observable,
    )


def _score_geometry_arrays(
    parameters: np.ndarray,
    matrices: np.ndarray,
    target_axis_lift: np.ndarray,
    legal: np.ndarray,
    envelope: ScoreEnvelope,
    source_frame: LocalFrame,
    source_field: GeometryField,
    source_sdf_128: np.ndarray,
    target_field: GeometryField,
    target_sdf_128: np.ndarray,
    target_valid: np.ndarray,
    *,
    candidate_chunk_size: int,
    stencil_chunk_size: int,
    source_stencil_observable: np.ndarray | None = None,
) -> GeometryScores:
    """One float64 implementation shared by scalar and table scoring."""

    if not 1 <= candidate_chunk_size <= _G0_CANDIDATE_CHUNK:
        raise ValueError("candidate chunk must be positive and at most 256")
    if not 1 <= stencil_chunk_size <= _G0_CELL_CHUNK:
        raise ValueError("stencil chunk must be positive and at most 256")
    if source_sdf_128.shape != (128, 128) or target_sdf_128.shape != (128, 128):
        raise ValueError("G0 SDF inputs must have shape 128x128")
    count = legal.shape[0]
    if (
        parameters.shape != (count, 6)
        or matrices.shape != (count, 3, 3)
        or target_axis_lift.shape != (count,)
        or legal.dtype != bool
    ):
        raise ValueError("geometry score arrays have inconsistent candidate shapes")
    components = np.ones((count, 5), dtype=np.float64)
    distance = np.full(count, _G0_INVALID_DISTANCE, dtype=np.float64)
    scored = np.zeros(count, dtype=bool)
    left, top, right, bottom = rectangular_valid_bounds(target_valid)
    if not legal.any():
        return GeometryScores(components, distance, scored)

    source_points = local_to_world(envelope.stencil_local, source_frame)
    if source_stencil_observable is None:
        source_observable = np.ones(source_points.shape[0], dtype=bool)
    else:
        source_observable = np.asarray(source_stencil_observable)
        if source_observable.shape != (source_points.shape[0],) or source_observable.dtype != bool:
            raise ValueError("source stencil observability must be a bool vector matching E_score")
    point_count = float(source_observable.sum(dtype=np.float64))
    if point_count <= 0.0:
        raise ValueError("source stencil observability is empty")
    source_geometry = _sample_field(source_field.g, source_points)
    source_eta = np.clip(_sample_field(source_field.eta, source_points), 0.0, 1.0)
    source_sdf = _sample_field(source_sdf_128, source_points)
    source_normalized_sdf = np.clip(source_sdf / source_frame.s_perp, -1.0, 1.0)
    source_alpha = _angles_from_geometry(source_geometry)
    source_rho = np.clip(source_geometry[:, 2], 0.0, 1.0)
    source_rho = np.where(source_rho >= _G0_ORIENTATION_FLOOR, source_rho, 0.0)
    source_tangent = np.column_stack((np.cos(source_alpha), np.sin(source_alpha)))
    source_sigma = _sigma_from_geometry(source_geometry)
    source_relation = _relation_simplex(source_normalized_sdf)
    observed_sdf = source_normalized_sdf[source_observable]
    source_crossing = bool(
        np.any(observed_sdf > _G0_RELATION_BAND) and np.any(observed_sdf < -_G0_RELATION_BAND)
    )
    source_bins = _relation_bins(source_normalized_sdf)
    source_bin_counts = np.array(
        [
            float(np.count_nonzero(source_observable & (source_bins == bin_id)))
            for bin_id in range(3)
        ],
        dtype=np.float64,
    )

    for candidate_start in range(0, count, candidate_chunk_size):
        candidate_stop = min(candidate_start + candidate_chunk_size, count)
        active = np.flatnonzero(legal[candidate_start:candidate_stop]) + candidate_start
        if active.size == 0:
            continue
        linear = matrices[active, :2, :2]
        translation = matrices[active, :2, 2]
        candidate_angle = target_axis_lift[active]
        cosine, sine = np.cos(candidate_angle), np.sin(candidate_angle)
        rotation = np.empty((active.size, 2, 2), dtype=np.float64)
        rotation[:, 0, 0] = cosine
        rotation[:, 0, 1] = -sine
        rotation[:, 1, 0] = sine
        rotation[:, 1, 1] = cosine
        inverse_scale = np.zeros_like(rotation)
        inverse_scale[:, 0, 0] = 1.0 / parameters[active, 3]
        inverse_scale[:, 1, 1] = 1.0 / parameters[active, 4]
        common = inverse_scale @ np.swapaxes(rotation, 1, 2)

        retained_sum = np.zeros(active.size, dtype=np.float64)
        sdf_bin_sum = np.zeros((active.size, 3), dtype=np.float64)
        orientation_sum = np.zeros(active.size, dtype=np.float64)
        scale_sum = np.zeros(active.size, dtype=np.float64)
        layout_bin_sum = np.zeros((active.size, 3), dtype=np.float64)
        incomplete = np.zeros(active.size, dtype=bool)
        target_positive = np.zeros(active.size, dtype=bool)
        target_negative = np.zeros(active.size, dtype=bool)

        for stencil_start in range(0, source_points.shape[0], stencil_chunk_size):
            stencil_stop = min(stencil_start + stencil_chunk_size, source_points.shape[0])
            stencil = slice(stencil_start, stencil_stop)
            points = source_points[stencil]
            counted = source_observable[stencil]
            target_points = np.einsum("cij,sj->csi", linear, points) + translation[:, None, :]
            point_valid_target = _full_stencil_element_validity(
                target_points,
                linear,
                source_frame,
                (left, top, right, bottom),
            )
            counted_2d = counted[None, :]
            point_valid = counted_2d & point_valid_target
            invalid = counted_2d & ~point_valid_target
            retained_sum += invalid.sum(axis=1, dtype=np.float64)
            incomplete |= invalid.any(axis=1)

            target_sdf = _sample_field(target_sdf_128, target_points)
            target_normalized_sdf = np.clip(target_sdf / parameters[active, 4, None], -1.0, 1.0)
            source_sdf_chunk = source_normalized_sdf[stencil]
            sdf_loss = np.abs(source_sdf_chunk[None, :] - target_normalized_sdf) / 2.0
            for bin_id in range(3):
                member = counted & (source_bins[stencil] == bin_id)
                if not member.any():
                    continue
                member_2d = member[None, :]
                sdf_bin_sum[:, bin_id] += np.where(
                    member_2d, np.where(point_valid_target, sdf_loss, 1.0), 0.0
                ).sum(axis=1, dtype=np.float64)

            target_geometry = _sample_field(target_field.g, target_points)
            target_rho = np.clip(target_geometry[..., 2], 0.0, 1.0)
            target_rho = np.where(target_rho >= _G0_ORIENTATION_FLOOR, target_rho, 0.0)
            target_alpha = _angles_from_geometry(target_geometry)
            transported_tangent = np.einsum("cij,sj->csi", linear, source_tangent[stencil])
            transported_tangent /= np.maximum(
                np.linalg.norm(transported_tangent, axis=2, keepdims=True), 1e-12
            )
            transported_alpha = np.arctan2(transported_tangent[..., 1], transported_tangent[..., 0])
            source_rho_chunk = source_rho[stencil]
            orientation_loss = (
                np.abs(source_rho_chunk[None, :] - target_rho)
                + np.minimum(source_rho_chunk[None, :], target_rho)
                * (1.0 - np.cos(2.0 * (target_alpha - transported_alpha)))
                / 2.0
            )
            orientation_sum += np.where(
                counted_2d,
                np.where(point_valid_target, np.clip(orientation_loss, 0.0, 1.0), 1.0),
                0.0,
            ).sum(axis=1, dtype=np.float64)

            target_eta = np.clip(_sample_field(target_field.eta, target_points), 0.0, 1.0)
            source_eta_chunk = source_eta[stencil]
            scale_loss = np.full(target_eta.shape, 0.5, dtype=np.float64)
            reliable_scale = (
                point_valid
                & (source_eta_chunk[None, :] >= _G0_SCALE_EVIDENCE_FLOOR)
                & (target_eta >= _G0_SCALE_EVIDENCE_FLOOR)
            )
            reliable_candidate, reliable_point = np.nonzero(reliable_scale)
            if reliable_candidate.size:
                reliable_linear = linear[reliable_candidate]
                transported_sigma = np.einsum(
                    "nij,njl,nml->nim",
                    reliable_linear,
                    source_sigma[stencil][reliable_point],
                    reliable_linear,
                )
                target_sigma = _sigma_from_geometry(
                    target_geometry[reliable_candidate, reliable_point]
                )
                reliable_common = common[reliable_candidate]
                normalized_source = np.einsum(
                    "nij,njl,nml->nim", reliable_common, transported_sigma, reliable_common
                )
                normalized_target = np.einsum(
                    "nij,njl,nml->nim", reliable_common, target_sigma, reliable_common
                )
                log_difference = _spd_log2(normalized_source) - _spd_log2(normalized_target)
                delta_scale = np.clip(
                    np.linalg.norm(log_difference, axis=(1, 2))
                    / (2.0 * math.sqrt(2.0) * math.log(4.0)),
                    0.0,
                    1.0,
                )
                evidence = np.minimum(
                    source_eta_chunk[reliable_point],
                    target_eta[reliable_candidate, reliable_point],
                )
                scale_loss[reliable_candidate, reliable_point] = (
                    evidence * delta_scale + (1.0 - evidence) * 0.5
                )
            scale_sum += np.where(
                counted_2d, np.where(point_valid_target, scale_loss, 1.0), 0.0
            ).sum(axis=1, dtype=np.float64)

            target_relation = _relation_simplex(target_normalized_sdf)
            layout_loss = (
                np.abs(source_relation[stencil][None, :, :] - target_relation).sum(axis=2) / 2.0
            )
            for bin_id in range(3):
                member = counted & (source_bins[stencil] == bin_id)
                if not member.any():
                    continue
                member_2d = member[None, :]
                layout_bin_sum[:, bin_id] += np.where(
                    member_2d, np.where(point_valid_target, layout_loss, 1.0), 0.0
                ).sum(axis=1, dtype=np.float64)
            target_positive |= np.any(
                point_valid & (target_normalized_sdf > _G0_RELATION_BAND), axis=1
            )
            target_negative |= np.any(
                point_valid & (target_normalized_sdf < -_G0_RELATION_BAND), axis=1
            )

        d_ret = retained_sum / point_count
        d_sdf = _occupied_bin_mean(sdf_bin_sum, source_bin_counts)
        d_ori = orientation_sum / point_count
        d_scale = scale_sum / point_count
        layout = _occupied_bin_mean(layout_bin_sum, source_bin_counts)
        target_crossing = target_positive & target_negative
        crossing_mismatch = np.where(
            incomplete, 1.0, (target_crossing != source_crossing).astype(np.float64)
        )
        d_relation = 0.75 * layout + 0.25 * crossing_mismatch
        chunk_components = np.column_stack((d_ret, d_sdf, d_ori, d_scale, d_relation))
        chunk_distance = 4.0 * d_ret + 2.0 * d_sdf + d_ori + d_scale + 2.0 * d_relation
        components[active] = chunk_components
        distance[active] = chunk_distance
        scored[active] = True
    return GeometryScores(components, distance, scored)


def _score_membership(
    target_points: np.ndarray,
    affine: GeometryAffine,
    envelope: ScoreEnvelope,
    source_frame: LocalFrame,
) -> np.ndarray:
    source_center = np.asarray(source_frame.center, dtype=np.float64)
    target_center = np.array(
        (affine.candidate.center_x, affine.candidate.center_y), dtype=np.float64
    )
    source_world = (target_points - target_center) @ affine.inverse_linear.T + source_center
    source_axis = 0.0 if source_frame.orientation_unknown else source_frame.alpha
    source_local = (source_world - source_center) @ _candidate_rotation(source_axis)
    left, top, right, bottom = envelope.local_bounds
    return (
        (source_local[:, 0] >= left)
        & (source_local[:, 0] < right)
        & (source_local[:, 1] >= top)
        & (source_local[:, 1] < bottom)
    )


def _cell_index_chunks(
    minimum_row: int,
    maximum_row: int,
    minimum_column: int,
    maximum_column: int,
    *,
    chunk_size: int,
):
    """Yield bounded row-major cell-index chunks without a bbox-sized mesh."""

    width = maximum_column - minimum_column + 1
    total = (maximum_row - minimum_row + 1) * width
    for start in range(0, total, chunk_size):
        flat = np.arange(start, min(start + chunk_size, total), dtype=np.intp)
        yield minimum_row + flat // width, minimum_column + flat % width


def _centers_in_valid_rectangle(
    points: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    left, top, right, bottom = bounds
    return (
        (points[:, 0] >= left)
        & (points[:, 0] < right)
        & (points[:, 1] >= top)
        & (points[:, 1] < bottom)
    )


def expected_score_occupancy(
    envelope: ScoreEnvelope,
    source_frame: LocalFrame,
    affines: tuple[GeometryAffine, ...],
    probabilities: np.ndarray,
    target_valid: np.ndarray,
    *,
    candidate_chunk_size: int = _G0_CANDIDATE_CHUNK,
    cell_chunk_size: int = _G0_CELL_CHUNK,
) -> np.ndarray:
    """Accumulate inverse-affine target-cell-centre membership on 128."""

    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(affines),) or not np.isfinite(probability).all():
        raise ValueError("occupancy probabilities must be finite [candidate]")
    if np.any(probability < 0.0):
        raise ValueError("occupancy probabilities must be non-negative")
    if not 1 <= candidate_chunk_size <= _G0_CANDIDATE_CHUNK:
        raise ValueError("occupancy candidate chunk must be positive and at most 256")
    if not 1 <= cell_chunk_size <= _G0_CELL_CHUNK:
        raise ValueError("occupancy cell chunk must be positive and at most 256")
    left, top, right, bottom = rectangular_valid_bounds(target_valid)
    valid_bounds = (left, top, right, bottom)
    output = np.zeros((128, 128), dtype=np.float64)
    score_corners_world = local_to_world(local_box_corners(envelope.local_bounds), source_frame)
    for chunk_start in range(0, len(affines), candidate_chunk_size):
        chunk_stop = min(chunk_start + candidate_chunk_size, len(affines))
        for index in range(chunk_start, chunk_stop):
            affine = affines[index]
            mass = float(probability[index])
            # This branch precedes transformed bounds and every inverse use.
            if not affine.valid or mass == 0.0:
                continue
            target_corners = transform_points(score_corners_world, affine)
            minimum_column = max(
                int(math.ceil((max(float(target_corners[:, 0].min()), left) - 2.0) / 4.0)),
                0,
            )
            maximum_column = min(
                int(math.floor((min(float(target_corners[:, 0].max()), right) - 2.0) / 4.0)),
                127,
            )
            minimum_row = max(
                int(math.ceil((max(float(target_corners[:, 1].min()), top) - 2.0) / 4.0)),
                0,
            )
            maximum_row = min(
                int(math.floor((min(float(target_corners[:, 1].max()), bottom) - 2.0) / 4.0)),
                127,
            )
            if minimum_column > maximum_column or minimum_row > maximum_row:
                continue
            for row_chunk, column_chunk in _cell_index_chunks(
                minimum_row,
                maximum_row,
                minimum_column,
                maximum_column,
                chunk_size=cell_chunk_size,
            ):
                target_points = np.column_stack(
                    (
                        column_chunk.astype(np.float64) * 4.0 + 2.0,
                        row_chunk.astype(np.float64) * 4.0 + 2.0,
                    )
                )
                member = _centers_in_valid_rectangle(
                    target_points, valid_bounds
                ) & _score_membership(target_points, affine, envelope, source_frame)
                output[row_chunk[member], column_chunk[member]] += mass
    return output


def affine_from_candidate_table(table: GeometryCandidateTable, index: int) -> GeometryAffine:
    if not 0 <= index < table.slots.shape[0]:
        raise IndexError("candidate table index out of range")
    parameters = table.parameters[index]
    candidate = CandidateParameters(*map(float, parameters[:5]), bool(parameters[5]))
    if not table.valid[index]:
        reasons = {
            1: "raw_scale_out_of_range",
            2: "source_target_scale_ratio_out_of_range",
            3: "coverage_certificate_outside_valid_canvas",
        }
        return GeometryAffine(
            candidate,
            np.zeros((3, 3), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            0.0,
            0.0,
            False,
            reasons.get(int(table.invalid_reason[index]), "invalid_candidate"),
        )
    return GeometryAffine(
        candidate,
        table.matrices[index].copy(),
        table.matrices[index, :2, :2].copy(),
        table.inverse_linear[index].copy(),
        float(table.relative_rotation[index]),
        float(table.target_axis_lift[index]),
        True,
        "",
    )


def score_geometry_candidate_table(
    table: GeometryCandidateTable,
    envelope: ScoreEnvelope,
    source_frame: LocalFrame,
    source_field: GeometryField,
    source_sdf_128: np.ndarray,
    target_field: GeometryField,
    target_sdf_128: np.ndarray,
    target_valid: np.ndarray,
    *,
    candidate_chunk_size: int = _G0_CANDIDATE_CHUNK,
    stencil_chunk_size: int = _G0_CELL_CHUNK,
    source_stencil_observable: np.ndarray | None = None,
) -> GeometryScores:
    return _score_geometry_arrays(
        table.parameters,
        table.matrices,
        table.target_axis_lift,
        table.valid,
        envelope,
        source_frame,
        source_field,
        source_sdf_128,
        target_field,
        target_sdf_128,
        target_valid,
        candidate_chunk_size=candidate_chunk_size,
        stencil_chunk_size=stencil_chunk_size,
        source_stencil_observable=source_stencil_observable,
    )


def orientation_evidence_mass(
    envelope: ScoreEnvelope,
    source_frame: LocalFrame,
    source_field: GeometryField,
    *,
    affine: GeometryAffine | None = None,
    target_field: GeometryField | None = None,
    target_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Return source, target, joint and valid orientation evidence mass.

    The denominator is always the complete frozen E_score stencil.  Missing
    target evidence and out-of-V stencil elements therefore contribute zero
    evidence instead of disappearing from the diagnostic.
    """

    source_points = local_to_world(envelope.stencil_local, source_frame)
    if source_points.shape[0] == 0:
        raise ValueError("orientation evidence requires a non-empty E_score stencil")
    source_rho = np.clip(_sample_field(source_field.g, source_points)[..., 2], 0.0, 1.0)
    source_rho = np.where(source_rho >= _G0_ORIENTATION_FLOOR, source_rho, 0.0)
    source_mass = float(source_rho.mean(dtype=np.float64))
    result = np.asarray((source_mass, 0.0, 0.0, 0.0), dtype=np.float64)
    if affine is None and target_field is None and target_valid is None:
        return result
    if affine is None or target_field is None or target_valid is None:
        raise ValueError("sampled orientation evidence requires affine, target field and V")
    if not affine.valid:
        return result

    left, top, right, bottom = rectangular_valid_bounds(target_valid)
    target_points = transform_points(source_points, affine)
    point_valid = _full_stencil_element_validity(
        target_points,
        affine.linear,
        source_frame,
        (left, top, right, bottom),
    )
    target_rho = np.clip(_sample_field(target_field.g, target_points)[..., 2], 0.0, 1.0)
    target_rho = np.where(target_rho >= _G0_ORIENTATION_FLOOR, target_rho, 0.0)
    visible_target = np.where(point_valid, target_rho, 0.0)
    joint = np.where(point_valid, np.minimum(source_rho, target_rho), 0.0)
    result[:] = (
        source_mass,
        float(visible_target.mean(dtype=np.float64)),
        float(joint.mean(dtype=np.float64)),
        float(point_valid.mean(dtype=np.float64)),
    )
    return result


def expected_candidate_table_occupancy(
    table: GeometryCandidateTable,
    envelope: ScoreEnvelope,
    source_frame: LocalFrame,
    probabilities: np.ndarray,
    target_valid: np.ndarray,
    *,
    candidate_chunk_size: int = _G0_CANDIDATE_CHUNK,
    cell_chunk_size: int = _G0_CELL_CHUNK,
) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != table.valid.shape or not np.isfinite(probability).all():
        raise ValueError("occupancy probabilities must be finite [candidate]")
    if np.any(probability < 0.0) or np.any(probability[~table.valid] != 0.0):
        raise ValueError("invalid candidates must have exactly zero occupancy mass")
    if not 1 <= candidate_chunk_size <= _G0_CANDIDATE_CHUNK:
        raise ValueError("occupancy candidate chunk must be positive and at most 256")
    if not 1 <= cell_chunk_size <= _G0_CELL_CHUNK:
        raise ValueError("occupancy cell chunk must be positive and at most 256")
    left, top, right, bottom = rectangular_valid_bounds(target_valid)
    valid_bounds = (left, top, right, bottom)
    output = np.zeros((128, 128), dtype=np.float64)
    source_corners = local_to_world(local_box_corners(envelope.local_bounds), source_frame)
    source_center = np.asarray(source_frame.center, dtype=np.float64)
    source_axis = 0.0 if source_frame.orientation_unknown else source_frame.alpha
    source_rotation = _candidate_rotation(source_axis)
    bounds = envelope.local_bounds
    count = table.valid.shape[0]
    for candidate_start in range(0, count, candidate_chunk_size):
        candidate_stop = min(candidate_start + candidate_chunk_size, count)
        active = (
            np.flatnonzero(
                table.valid[candidate_start:candidate_stop]
                & (probability[candidate_start:candidate_stop] > 0.0)
            )
            + candidate_start
        )
        for index in active:
            linear = table.matrices[index, :2, :2]
            translation = table.matrices[index, :2, 2]
            target_corners = source_corners @ linear.T + translation
            minimum_column = max(
                int(math.ceil((max(float(target_corners[:, 0].min()), left) - 2.0) / 4.0)),
                0,
            )
            maximum_column = min(
                int(math.floor((min(float(target_corners[:, 0].max()), right) - 2.0) / 4.0)),
                127,
            )
            minimum_row = max(
                int(math.ceil((max(float(target_corners[:, 1].min()), top) - 2.0) / 4.0)),
                0,
            )
            maximum_row = min(
                int(math.floor((min(float(target_corners[:, 1].max()), bottom) - 2.0) / 4.0)),
                127,
            )
            if minimum_column > maximum_column or minimum_row > maximum_row:
                continue
            for row_chunk, column_chunk in _cell_index_chunks(
                minimum_row,
                maximum_row,
                minimum_column,
                maximum_column,
                chunk_size=cell_chunk_size,
            ):
                target_points = np.column_stack(
                    (
                        column_chunk.astype(np.float64) * 4.0 + 2.0,
                        row_chunk.astype(np.float64) * 4.0 + 2.0,
                    )
                )
                target_center = table.parameters[index, :2]
                source_world = (target_points - target_center) @ table.inverse_linear[
                    index
                ].T + source_center
                source_local = (source_world - source_center) @ source_rotation
                member = (
                    (source_local[:, 0] >= bounds[0])
                    & (source_local[:, 0] < bounds[2])
                    & (source_local[:, 1] >= bounds[1])
                    & (source_local[:, 1] < bounds[3])
                    & _centers_in_valid_rectangle(target_points, valid_bounds)
                )
                output[row_chunk[member], column_chunk[member]] += probability[index]
    return output


# ---------------------------------------------------------------------------
# RC2 P0-R2 spatial context.  The RC1 orderless ring descriptor stays unused.


_CTX_OUTER_PX = 72
_CTX_SPACING_PX = 8
_CTX_FEATURE_CHANNELS = 6


def build_context_ring_stencil(
    score_bounds: np.ndarray,
    gap: int,
    outer_px: int = _CTX_OUTER_PX,
    spacing: int = _CTX_SPACING_PX,
) -> np.ndarray:
    """Return a row-major local stencil on the coarse envelope ring.

    The ring is the axis-aligned expansion of the score bbox.  Interior and
    the blind gap are excluded, so exact GT contours never enter the points.
    """

    if score_bounds.shape != (4,) or not np.isfinite(score_bounds).all():
        raise ValueError("context score bounds must be a finite length-4 array")
    if gap <= 0 or outer_px <= gap or spacing <= 0:
        raise ValueError("context ring requires 0 < gap < outer_px and positive spacing")
    left, top, right, bottom = (float(value) for value in score_bounds)
    columns = np.arange(left - outer_px + spacing / 2.0, right + outer_px, spacing, dtype=np.float64)
    rows = np.arange(top - outer_px + spacing / 2.0, bottom + outer_px, spacing, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(columns, rows)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distance = _chebyshev_to_bounds(points, score_bounds)
    kept = (distance >= float(gap)) & (distance < float(outer_px) + 1e-9)
    stencil = points[kept]
    if stencil.shape[0] == 0:
        raise ValueError("context ring stencil is empty")
    return stencil


def _chebyshev_to_bounds(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    delta_x = np.maximum(bounds[0] - points[:, 0], points[:, 0] - bounds[2])
    delta_y = np.maximum(bounds[1] - points[:, 1], points[:, 1] - bounds[3])
    return np.maximum(np.maximum(delta_x, 0.0), np.maximum(delta_y, 0.0))


def context_feature_maps(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Illumination-normalized color, brightness and oriented edges on 512."""

    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("context RGB must be a 512x512 uint8 image")
    if valid.shape != (512, 512) or valid.dtype != bool:
        raise ValueError("context valid mask must be a 512x512 bool array")
    values = rgb.astype(np.float64) / 255.0
    channel_sum = values.sum(axis=2, keepdims=True)
    chromaticity = values / np.maximum(channel_sum, 1.0 / 255.0)
    gray = values @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.clip(np.hypot(gradient_x, gradient_y), 0.0, 1.0)
    angle = np.arctan2(gradient_y, gradient_x)
    features = np.empty((512, 512, _CTX_FEATURE_CHANNELS), dtype=np.float64)
    features[..., 0] = chromaticity[..., 0]
    features[..., 1] = chromaticity[..., 1]
    features[..., 2] = gray
    features[..., 3] = magnitude
    features[..., 4] = 0.5 + 0.5 * np.cos(2.0 * angle)
    features[..., 5] = 0.5 + 0.5 * np.sin(2.0 * angle)
    features[~valid] = 0.0
    return features


def score_context_affine(
    affine: GeometryAffine,
    stencil_local: np.ndarray,
    source_frame: LocalFrame,
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_valid: np.ndarray,
) -> float:
    """Compare one affine's transported ring against the target feature field."""

    if not affine.valid:
        return 1.0
    return float(
        _score_context_arrays(
            affine.matrix[None, ...],
            np.array([True], dtype=bool),
            stencil_local,
            source_frame,
            source_features,
            target_features,
            target_valid,
        )[0]
    )


def score_context_candidate_table(
    table: GeometryCandidateTable,
    stencil_local: np.ndarray,
    source_frame: LocalFrame,
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_valid: np.ndarray,
    *,
    candidate_chunk_size: int = _G0_CANDIDATE_CHUNK,
    stencil_chunk_size: int = _G0_CELL_CHUNK,
) -> np.ndarray:
    """Return one [N] context distance aligned with the G0 candidate table."""

    return _score_context_arrays(
        table.matrices,
        table.valid,
        stencil_local,
        source_frame,
        source_features,
        target_features,
        target_valid,
        candidate_chunk_size=candidate_chunk_size,
        stencil_chunk_size=stencil_chunk_size,
    )


def _score_context_arrays(
    matrices: np.ndarray,
    legal: np.ndarray,
    stencil_local: np.ndarray,
    source_frame: LocalFrame,
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_valid: np.ndarray,
    *,
    candidate_chunk_size: int = _G0_CANDIDATE_CHUNK,
    stencil_chunk_size: int = _G0_CELL_CHUNK,
) -> np.ndarray:
    if stencil_local.ndim != 2 or stencil_local.shape[1] != 2:
        raise ValueError("context stencil must be [K,2]")
    source_points = local_to_world(stencil_local, source_frame)
    source_in_canvas = np.all((source_points >= 0.0) & (source_points < 512.0), axis=1)
    if not source_in_canvas.any():
        raise ValueError("context ring has no source-visible points")
    source_samples = _sample_field(source_features, source_points)
    left, top, right, bottom = rectangular_valid_bounds(target_valid)
    distances = np.ones(legal.shape[0], dtype=np.float64)
    observable_count = float(source_in_canvas.sum(dtype=np.float64))
    for candidate_start in range(0, legal.shape[0], candidate_chunk_size):
        candidate_stop = min(candidate_start + candidate_chunk_size, legal.shape[0])
        active = np.flatnonzero(legal[candidate_start:candidate_stop]) + candidate_start
        if active.size == 0:
            continue
        linear = matrices[active, :2, :2]
        translation = matrices[active, :2, 2]
        loss_sum = np.zeros(active.size, dtype=np.float64)
        for stencil_start in range(0, source_points.shape[0], stencil_chunk_size):
            stencil_stop = min(stencil_start + stencil_chunk_size, source_points.shape[0])
            counted = source_in_canvas[stencil_start:stencil_stop]
            if not counted.any():
                continue
            points = source_points[stencil_start:stencil_stop]
            source_chunk = source_samples[stencil_start:stencil_stop]
            target_points = np.einsum("cij,sj->csi", linear, points) + translation[:, None, :]
            in_target = (
                (target_points[..., 0] >= left)
                & (target_points[..., 0] < right)
                & (target_points[..., 1] >= top)
                & (target_points[..., 1] < bottom)
            )
            point_valid = counted[None, :] & in_target
            target_chunk = _sample_field(target_features, target_points)
            delta = np.abs(source_chunk[None, :, :] - target_chunk)
            feature_loss = 0.6 * delta[..., :3].mean(axis=-1) + 0.4 * delta[..., 3:].mean(axis=-1)
            loss_sum += np.where(
                counted[None, :], np.where(point_valid, feature_loss, 1.0), 0.0
            ).sum(axis=1, dtype=np.float64)
        distances[active] = loss_sum / observable_count
    distances[~legal] = 1.0
    return distances

"""Independent full-valid-canvas geometry targets for Module A RC1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import LoadedIdentity
from .identities import IdentityManifest
from .structure import LocalFrame, NormalAtlas, NormalStructure, ObjectFrame, support_frame


class _TargetConfig(Protocol):
    category: str
    defect_types: tuple[str, ...]
    image_size: int
    proposal_size: int
    square_dilate_radius: int


_TRUNCATION_LIMIT = 0.95

# RC2 geometry constants.  This path is diagnostic-only and deliberately lives
# beside, rather than inside, the frozen RC1 constructor below.  Coverage and
# scoring have deliberately independent quantization: exact-GT retention uses
# the tighter 4-pixel certificate, while geometry scoring starts from a filled
# 16-pixel seed bbox and expands it by one 16-pixel cell.
_COVER_QUANTUM_PX = 4.0
_SCORE_SEED_QUANTUM_PX = 16.0
_SCORE_QUANTUM_PX = 16.0
_SCORE_DILATION_CELLS = 1
_STENCIL_SPACING_PX = 4.0
_SCALE_MIN_PX = 4.0
_SCALE_MAX_PX = 64.0
_SCALE_RATIO_MIN = 0.5
_SCALE_RATIO_MAX = 2.0
_GEOMETRY_EPSILON = 1e-12
_BOUNDARY_EPSILON = 1e-9


@dataclass(frozen=True)
class CoverageCertificate:
    """A filled quantized source-local bbox used only for exact retention."""

    local_bounds: np.ndarray
    quantum_px: float = _COVER_QUANTUM_PX

    def __post_init__(self) -> None:
        if self.local_bounds.shape != (4,) or self.local_bounds.dtype != np.float64:
            raise ValueError("coverage local_bounds must be float64 [left, top, right, bottom]")
        left, top, right, bottom = self.local_bounds
        if not np.isfinite(self.local_bounds).all() or left >= right or top >= bottom:
            raise ValueError("coverage local_bounds must be finite and non-empty")
        self.local_bounds.setflags(write=False)


@dataclass(frozen=True)
class SeedEnvelope:
    """The filled topology-free 16-pixel bbox from which scoring expands."""

    local_bounds: np.ndarray
    quantum_px: float = _SCORE_SEED_QUANTUM_PX

    def __post_init__(self) -> None:
        if self.local_bounds.shape != (4,) or self.local_bounds.dtype != np.float64:
            raise ValueError("seed local_bounds must be float64 [left, top, right, bottom]")
        left, top, right, bottom = self.local_bounds
        if not np.isfinite(self.local_bounds).all() or left >= right or top >= bottom:
            raise ValueError("seed local_bounds must be finite and non-empty")
        self.local_bounds.setflags(write=False)


@dataclass(frozen=True)
class ScoreEnvelope:
    """The filled, quantized bbox used by every RC2 geometry score.

    ``stencil_local`` is a dense Cartesian grid of 4-pixel cell centres.  It
    carries no exact-GT contour, connectivity, or topology bits.
    """

    local_bounds: np.ndarray
    stencil_local: np.ndarray
    quantum_px: float = _SCORE_QUANTUM_PX
    dilation_cells: int = _SCORE_DILATION_CELLS
    stencil_spacing_px: float = _STENCIL_SPACING_PX

    def __post_init__(self) -> None:
        if self.local_bounds.shape != (4,) or self.local_bounds.dtype != np.float64:
            raise ValueError("score local_bounds must be float64 [left, top, right, bottom]")
        if (
            self.stencil_local.ndim != 2
            or self.stencil_local.shape[1:] != (2,)
            or self.stencil_local.dtype != np.float64
            or self.stencil_local.shape[0] == 0
        ):
            raise ValueError("score stencil must be a non-empty float64 [K,2] array")
        left, top, right, bottom = self.local_bounds
        if not np.isfinite(self.local_bounds).all() or left >= right or top >= bottom:
            raise ValueError("score local_bounds must be finite and non-empty")
        if not np.isfinite(self.stencil_local).all():
            raise ValueError("score stencil must be finite")
        self.local_bounds.setflags(write=False)
        self.stencil_local.setflags(write=False)


@dataclass(frozen=True)
class CoarseEnvelopes:
    cover: CoverageCertificate
    seed: SeedEnvelope
    score: ScoreEnvelope


@dataclass(frozen=True)
class CandidateParameters:
    center_x: float
    center_y: float
    alpha: float
    s_parallel: float
    s_perp: float
    orientation_unknown: bool


@dataclass(frozen=True)
class GeometryAffine:
    """One immutable affine, including a finite sentinel for invalid slots."""

    candidate: CandidateParameters
    matrix: np.ndarray
    linear: np.ndarray
    inverse_linear: np.ndarray
    relative_rotation: float
    target_axis_lift: float
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3) or self.matrix.dtype != np.float64:
            raise ValueError("geometry affine matrix must be float64 [3,3]")
        if self.linear.shape != (2, 2) or self.linear.dtype != np.float64:
            raise ValueError("geometry affine linear part must be float64 [2,2]")
        if self.inverse_linear.shape != (2, 2) or self.inverse_linear.dtype != np.float64:
            raise ValueError("geometry affine inverse must be float64 [2,2]")
        if not all(
            np.isfinite(value).all() for value in (self.matrix, self.linear, self.inverse_linear)
        ):
            raise ValueError("geometry affine arrays must be finite")
        self.matrix.setflags(write=False)
        self.linear.setflags(write=False)
        self.inverse_linear.setflags(write=False)


@dataclass(frozen=True)
class OverlapRaster:
    mask: np.ndarray
    retained_fraction: float
    source_pixel_count: int

    def __post_init__(self) -> None:
        _mask_512(self.mask, name="overlap raster")
        if self.source_pixel_count <= 0 or self.retained_fraction != 1.0:
            raise ValueError("exact overlap raster must represent a fully retained non-empty GT")
        self.mask.setflags(write=False)


@dataclass(frozen=True)
class ProposalTarget:
    transferred_gt_512: np.ndarray
    proposal_512: np.ndarray
    proposal_128: np.ndarray
    valid_128: np.ndarray
    z0: np.ndarray
    retained_fraction: float
    target_support: np.ndarray
    target_sdf_128: np.ndarray
    target_u_x: np.ndarray
    target_u_y: np.ndarray
    target_frame: ObjectFrame

    def __post_init__(self) -> None:
        for array in (
            self.transferred_gt_512,
            self.proposal_512,
            self.proposal_128,
            self.valid_128,
            self.z0,
            self.target_support,
            self.target_sdf_128,
            self.target_u_x,
            self.target_u_y,
        ):
            array.setflags(write=False)


def _mask_512(value: np.ndarray, *, name: str) -> np.ndarray:
    if value.shape != (512, 512) or value.dtype != bool:
        raise ValueError(f"{name} must be a 512x512 bool array")
    return value


def _rotation(alpha: float) -> np.ndarray:
    cosine, sine = math.cos(alpha), math.sin(alpha)
    return np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _local_gt_extents(source_gt: np.ndarray, frame: LocalFrame) -> np.ndarray:
    _mask_512(source_gt, name="source_gt")
    if not source_gt.any():
        raise ValueError("source_gt must be non-empty")
    rows, columns = np.nonzero(source_gt)
    # Only the filled world AABB may influence RC2 envelopes.  Projecting every
    # true pixel would allow two masks with the same bbox but different sparse
    # topology to produce different local extents in a rotated frame.  Pixel
    # cells use half-open bounds, hence the +1 on the right and bottom edges.
    left, right = float(columns.min()), float(columns.max() + 1)
    top, bottom = float(rows.min()), float(rows.max() + 1)
    world = np.array(
        ((left, top), (right, top), (right, bottom), (left, bottom)),
        dtype=np.float64,
    )
    local = (world - np.asarray(frame.center, dtype=np.float64)) @ _rotation(frame.alpha)
    return np.array(
        (local[:, 0].min(), local[:, 1].min(), local[:, 0].max(), local[:, 1].max()),
        dtype=np.float64,
    )


def _quantize_outward(bounds: np.ndarray, quantum: float) -> np.ndarray:
    left, top, right, bottom = bounds
    quantized = np.array(
        (
            math.floor(left / quantum) * quantum,
            math.floor(top / quantum) * quantum,
            math.ceil(right / quantum) * quantum,
            math.ceil(bottom / quantum) * quantum,
        ),
        dtype=np.float64,
    )
    if quantized[0] == quantized[2]:
        quantized[2] += quantum
    if quantized[1] == quantized[3]:
        quantized[3] += quantum
    return quantized


def build_coarse_envelopes(source_gt: np.ndarray, frame: LocalFrame) -> CoarseEnvelopes:
    """Build topology-free RC2 coverage and scoring bboxes.

    The coverage bbox is an independent conservative full-retention
    certificate quantized outward to 4 pixels.  The scoring seed is the same
    topology-free bbox independently quantized outward to 16 pixels, and the
    score bbox expands that seed by exactly one 16-pixel cell.  It is filled by
    a fixed 4-pixel diagnostic stencil.  Consequently masks with the same
    world bbox have identical geometry inputs even when their exact final
    rasters differ.
    """

    raw_bounds = _local_gt_extents(source_gt, frame)
    cover_bounds = _quantize_outward(raw_bounds, _COVER_QUANTUM_PX)
    seed_bounds = _quantize_outward(raw_bounds, _SCORE_SEED_QUANTUM_PX)
    score_bounds = seed_bounds + np.array(
        (-_SCORE_QUANTUM_PX, -_SCORE_QUANTUM_PX, _SCORE_QUANTUM_PX, _SCORE_QUANTUM_PX),
        dtype=np.float64,
    )
    columns = np.arange(
        score_bounds[0] + _STENCIL_SPACING_PX / 2.0,
        score_bounds[2],
        _STENCIL_SPACING_PX,
        dtype=np.float64,
    )
    rows = np.arange(
        score_bounds[1] + _STENCIL_SPACING_PX / 2.0,
        score_bounds[3],
        _STENCIL_SPACING_PX,
        dtype=np.float64,
    )
    grid_x, grid_y = np.meshgrid(columns, rows)
    stencil = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)
    return CoarseEnvelopes(
        CoverageCertificate(cover_bounds),
        SeedEnvelope(seed_bounds),
        ScoreEnvelope(score_bounds, stencil),
    )


def _invalid_affine(candidate: CandidateParameters, reason: str) -> GeometryAffine:
    return GeometryAffine(
        candidate=candidate,
        matrix=np.zeros((3, 3), dtype=np.float64),
        linear=np.zeros((2, 2), dtype=np.float64),
        inverse_linear=np.zeros((2, 2), dtype=np.float64),
        relative_rotation=0.0,
        target_axis_lift=0.0,
        valid=False,
        reason=reason,
    )


def _minimal_unoriented_rotation(source_alpha: float, target_alpha: float) -> float:
    """Return the half-open [-pi/2, pi/2) relative rotation modulo pi."""

    return float((target_alpha - source_alpha + math.pi / 2.0) % math.pi - math.pi / 2.0)


def build_geometry_affine(
    source_frame: LocalFrame, candidate: CandidateParameters
) -> GeometryAffine:
    """Create the sole RC2 affine or a finite invalid-slot sentinel.

    Raw scales are validated before any reciprocal/inverse operation.  They
    are never clipped, so an out-of-range slot cannot duplicate probability
    mass at a boundary affine.
    """

    numeric = (
        candidate.center_x,
        candidate.center_y,
        candidate.alpha,
        candidate.s_parallel,
        candidate.s_perp,
        source_frame.center[0],
        source_frame.center[1],
        source_frame.alpha,
        source_frame.s_parallel,
        source_frame.s_perp,
    )
    if not all(math.isfinite(value) for value in numeric):
        return _invalid_affine(candidate, "non_finite_parameter")
    if source_frame.s_parallel <= 0.0 or source_frame.s_perp <= 0.0:
        return _invalid_affine(candidate, "invalid_source_scale")
    if not (
        _SCALE_MIN_PX <= candidate.s_parallel <= _SCALE_MAX_PX
        and _SCALE_MIN_PX <= candidate.s_perp <= _SCALE_MAX_PX
    ):
        return _invalid_affine(candidate, "raw_scale_out_of_range")
    ratios = (
        candidate.s_parallel / source_frame.s_parallel,
        candidate.s_perp / source_frame.s_perp,
    )
    if not all(_SCALE_RATIO_MIN <= value <= _SCALE_RATIO_MAX for value in ratios):
        return _invalid_affine(candidate, "source_target_scale_ratio_out_of_range")

    if source_frame.orientation_unknown:
        if not math.isclose(
            source_frame.s_parallel,
            source_frame.s_perp,
            rel_tol=0.0,
            abs_tol=_GEOMETRY_EPSILON,
        ):
            return _invalid_affine(candidate, "anisotropic_unknown_source_frame")
        source_axis = 0.0
        target_lift = _minimal_unoriented_rotation(0.0, candidate.alpha)
        relative = target_lift
    else:
        source_axis = source_frame.alpha
        relative = _minimal_unoriented_rotation(source_axis, candidate.alpha)
        target_lift = source_axis + relative

    source_rotation = _rotation(source_axis)
    target_rotation = _rotation(target_lift)
    ratio_matrix = np.diag(np.asarray(ratios, dtype=np.float64))
    linear = target_rotation @ ratio_matrix @ source_rotation.T
    # Reciprocal work happens only after every invalidity test above.
    inverse_linear = source_rotation @ np.diag(1.0 / np.asarray(ratios)) @ target_rotation.T
    source_center = np.asarray(source_frame.center, dtype=np.float64)
    target_center = np.array((candidate.center_x, candidate.center_y), dtype=np.float64)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = target_center - linear @ source_center
    return GeometryAffine(
        candidate=candidate,
        matrix=matrix,
        linear=linear,
        inverse_linear=inverse_linear,
        relative_rotation=relative,
        target_axis_lift=target_lift,
        valid=True,
        reason="",
    )


def local_box_corners(bounds: np.ndarray) -> np.ndarray:
    if bounds.shape != (4,):
        raise ValueError("local bounds must have four coordinates")
    left, top, right, bottom = map(float, bounds)
    return np.array(((left, top), (right, top), (right, bottom), (left, bottom)), dtype=np.float64)


def local_to_world(points: np.ndarray, frame: LocalFrame) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("local points must have shape [N,2]")
    source_axis = 0.0 if frame.orientation_unknown else frame.alpha
    return points @ _rotation(source_axis).T + np.asarray(frame.center, dtype=np.float64)


def transform_points(points: np.ndarray, affine: GeometryAffine) -> np.ndarray:
    if not affine.valid:
        raise ValueError(f"cannot transform points with invalid affine: {affine.reason}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("world points must have shape [N,2]")
    return points @ affine.linear.T + affine.matrix[:2, 2]


def rectangular_valid_bounds(valid: np.ndarray) -> tuple[float, float, float, float]:
    _mask_512(valid, name="valid")
    rows, columns = np.nonzero(valid)
    if columns.size == 0:
        raise ValueError("valid canvas must be non-empty")
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    rectangle = np.zeros_like(valid)
    rectangle[top:bottom, left:right] = True
    if not np.array_equal(rectangle, valid):
        raise ValueError("G0 requires V to be one filled axis-aligned rectangle")
    return float(left), float(top), float(right), float(bottom)


def coverage_certificate_is_retained(
    certificate: CoverageCertificate,
    source_frame: LocalFrame,
    affine: GeometryAffine,
    target_valid: np.ndarray,
) -> bool:
    if not affine.valid:
        return False
    left, top, right, bottom = rectangular_valid_bounds(target_valid)
    source_corners = local_to_world(local_box_corners(certificate.local_bounds), source_frame)
    target_corners = transform_points(source_corners, affine)
    return bool(
        np.all(target_corners[:, 0] >= left - _BOUNDARY_EPSILON)
        and np.all(target_corners[:, 0] <= right + _BOUNDARY_EPSILON)
        and np.all(target_corners[:, 1] >= top - _BOUNDARY_EPSILON)
        and np.all(target_corners[:, 1] <= bottom + _BOUNDARY_EPSILON)
    )


def _clip_polygon_axis(
    polygon: list[tuple[float, float]], *, axis: int, boundary: float, keep_greater: bool
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    def inside(point: tuple[float, float]) -> bool:
        return point[axis] >= boundary if keep_greater else point[axis] <= boundary

    result: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if abs(denominator) > _GEOMETRY_EPSILON:
                fraction = (boundary - previous[axis]) / denominator
                point = (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
                result.append(point)
        if current_inside:
            result.append(current)
        previous, previous_inside = current, current_inside
    return result


def _positive_overlap_with_pixel(polygon: np.ndarray, column: int, row: int) -> bool:
    clipped = [tuple(point) for point in polygon]
    clipped = _clip_polygon_axis(clipped, axis=0, boundary=float(column), keep_greater=True)
    clipped = _clip_polygon_axis(clipped, axis=0, boundary=float(column + 1), keep_greater=False)
    clipped = _clip_polygon_axis(clipped, axis=1, boundary=float(row), keep_greater=True)
    clipped = _clip_polygon_axis(clipped, axis=1, boundary=float(row + 1), keep_greater=False)
    if len(clipped) < 3:
        return False
    x = np.fromiter((point[0] for point in clipped), dtype=np.float64)
    y = np.fromiter((point[1] for point in clipped), dtype=np.float64)
    area = 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
    return area > _GEOMETRY_EPSILON


def rasterize_pixel_cell_overlap(
    source_gt: np.ndarray, affine: GeometryAffine, target_valid: np.ndarray
) -> OverlapRaster:
    """Rasterize transformed GT as positive pixel-cell overlap, never nearest.

    The input is the union of source unit pixel cells.  A target pixel is set
    exactly when its unit cell has positive-area overlap with that transformed
    union.  Rectangular-V and full-retention checks happen before any raster is
    materialized; partial clipping is an error rather than a repaired result.
    """

    _mask_512(source_gt, name="source_gt")
    if not source_gt.any():
        raise ValueError("source_gt must be non-empty")
    if not affine.valid:
        raise ValueError(f"cannot rasterize invalid affine: {affine.reason}")
    left, top, right, bottom = rectangular_valid_bounds(target_valid)
    rows, columns = np.nonzero(source_gt)
    output = np.zeros((512, 512), dtype=bool)
    base = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    for row, column in zip(rows, columns, strict=True):
        polygon = transform_points(base + (float(column), float(row)), affine)
        if (
            np.any(polygon[:, 0] < left - _BOUNDARY_EPSILON)
            or np.any(polygon[:, 0] > right + _BOUNDARY_EPSILON)
            or np.any(polygon[:, 1] < top - _BOUNDARY_EPSILON)
            or np.any(polygon[:, 1] > bottom + _BOUNDARY_EPSILON)
        ):
            raise ValueError("transformed exact GT is not fully retained by rectangular V")
        minimum_x = max(int(math.floor(float(polygon[:, 0].min()))), int(left))
        maximum_x = min(int(math.ceil(float(polygon[:, 0].max()))) - 1, int(right) - 1)
        minimum_y = max(int(math.floor(float(polygon[:, 1].min()))), int(top))
        maximum_y = min(int(math.ceil(float(polygon[:, 1].max()))) - 1, int(bottom) - 1)
        for target_row in range(minimum_y, maximum_y + 1):
            for target_column in range(minimum_x, maximum_x + 1):
                if _positive_overlap_with_pixel(polygon, target_column, target_row):
                    output[target_row, target_column] = True
    if not output.any():
        raise ValueError("transformed exact GT overlap raster is empty")
    return OverlapRaster(output, 1.0, int(source_gt.sum()))


def square_dilate_24(mask: np.ndarray) -> np.ndarray:
    """Apply the fixed 49x49 square footprint on the 512 grid."""

    return square_dilate(mask, radius=24)


def square_dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:
    _mask_512(mask, name="mask")
    if radius != 24:
        raise ValueError("v0.6-A SquareDilate radius must be exactly 24")
    padded = np.pad(mask.astype(np.uint8), radius)
    integral = np.pad(padded.cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64), ((1, 0), (1, 0)))
    width = 2 * radius + 1
    sums = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return sums > 0


def downsample_any_4x4(mask: np.ndarray) -> np.ndarray:
    _mask_512(mask, name="mask")
    return mask.reshape(128, 4, 128, 4).any(axis=(1, 3))


def downsample_all_4x4(valid: np.ndarray) -> np.ndarray:
    _mask_512(valid, name="valid")
    return valid.reshape(128, 4, 128, 4).all(axis=(1, 3))


def nearest_upsample_4x4(mask: np.ndarray) -> np.ndarray:
    if mask.shape != (128, 128) or mask.dtype != bool:
        raise ValueError("mask must be a 128x128 bool array")
    return np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)


def _transfer(
    source_gt: np.ndarray,
    source_frame: ObjectFrame,
    target_valid: np.ndarray,
    target_frame: ObjectFrame,
) -> tuple[np.ndarray, float]:
    ys, xs = np.nonzero(source_gt)
    ratio_x = target_frame.scale_x / source_frame.scale_x
    ratio_y = target_frame.scale_y / source_frame.scale_y
    mapped_x = (xs - source_frame.center_x) * ratio_x + target_frame.center_x
    mapped_y = (ys - source_frame.center_y) * ratio_y + target_frame.center_y
    min_x = int(np.floor(mapped_x.min() - abs(ratio_x) / 2.0 - 1.0))
    max_x = int(np.ceil(mapped_x.max() + abs(ratio_x) / 2.0 + 1.0))
    min_y = int(np.floor(mapped_y.min() - abs(ratio_y) / 2.0 - 1.0))
    max_y = int(np.ceil(mapped_y.max() + abs(ratio_y) / 2.0 + 1.0))
    target_x = np.arange(min_x, max_x + 1, dtype=np.float64)
    target_y = np.arange(min_y, max_y + 1, dtype=np.float64)
    source_x = np.rint(
        (target_x - target_frame.center_x) * source_frame.scale_x / target_frame.scale_x
        + source_frame.center_x
    ).astype(int)
    source_y = np.rint(
        (target_y - target_frame.center_y) * source_frame.scale_y / target_frame.scale_y
        + source_frame.center_y
    ).astype(int)
    x_count = np.bincount(source_x[(source_x >= 0) & (source_x < 512)], minlength=512)
    y_count = np.bincount(source_y[(source_y >= 0) & (source_y < 512)], minlength=512)
    complete_count = int((source_gt * y_count[:, None] * x_count[None, :]).sum())
    if complete_count == 0:
        raise ValueError("transformed ground truth raster is empty")
    grid_y, grid_x = np.indices((512, 512), dtype=np.float64)
    source_x = np.rint(
        (grid_x - target_frame.center_x) * source_frame.scale_x / target_frame.scale_x
        + source_frame.center_x
    ).astype(int)
    source_y = np.rint(
        (grid_y - target_frame.center_y) * source_frame.scale_y / target_frame.scale_y
        + source_frame.center_y
    ).astype(int)
    in_source = (source_x >= 0) & (source_x < 512) & (source_y >= 0) & (source_y < 512)
    transferred = np.zeros((512, 512), dtype=bool)
    transferred[in_source] = source_gt[source_y[in_source], source_x[in_source]]
    transferred &= target_valid
    retained_fraction = float(transferred.sum() / complete_count)
    return transferred, retained_fraction


def construct_plain_target(
    source: LoadedIdentity,
    source_atlas: NormalAtlas,
    target: LoadedIdentity,
    target_structure: NormalStructure,
    config: _TargetConfig,
    identity_manifest: IdentityManifest,
    target_center_override: tuple[float, float] | None = None,
) -> ProposalTarget:
    """Construct one ordinary A0/A1-plain target without support clipping or repair."""

    if config.image_size != 512 or config.proposal_size != 128 or config.square_dilate_radius != 24:
        raise ValueError("v0.6-A target grids and SquareDilate radius are frozen")
    if not isinstance(source_atlas, NormalAtlas):
        raise ValueError("source atlas must be a validated NormalAtlas")
    if source_atlas.category != config.category:
        raise ValueError("atlas category must match the configured category")
    if (
        identity_manifest.category != config.category
        or identity_manifest.defect_types != config.defect_types
    ):
        raise ValueError("target manifest must agree with the experiment config")
    if not source_atlas.matches_manifest(identity_manifest):
        raise ValueError("atlas provenance must match the supplied identity manifest")
    if source.record.role != "fewshot":
        raise ValueError("source must be a fewshot identity")
    if source.record not in identity_manifest.fewshot:
        raise ValueError("source must exactly match a frozen manifest fewshot record")
    if source.record.category != config.category:
        raise ValueError("source category must match the configured category")
    if source.record.defect_type not in config.defect_types:
        raise ValueError("source defect type must be in the configured vocabulary")
    if target.record.category != config.category or target.record.defect_type != "good":
        raise ValueError("target category must match the configured normal category")
    if target.record.role not in {"normal_fit", "normal_dev", "open_normal"}:
        raise ValueError("target must be a prepared normal identity")
    if target_structure.normal.record.sample_id != target.record.sample_id or not np.array_equal(
        target_structure.valid, target.valid
    ):
        raise ValueError("target structure must belong to the supplied target normal")
    if source.anomaly_gt is None or not source.anomaly_gt.any():
        raise ValueError("source requires a non-empty anomaly ground truth")
    source_frame = support_frame(source_atlas.support)
    target_frame = target_structure.frame
    if target_center_override is not None:
        center_x, center_y = target_center_override
        if not all(
            isinstance(value, int | float) and np.isfinite(value) for value in (center_x, center_y)
        ):
            raise ValueError("target center override must be finite")
        target_frame = ObjectFrame(
            float(center_x), float(center_y), target_frame.scale_x, target_frame.scale_y
        )
    transferred, retained_fraction = _transfer(
        source.anomaly_gt, source_frame, target.valid, target_frame
    )
    if retained_fraction < _TRUNCATION_LIMIT:
        raise ValueError(f"valid canvas truncation retained only {retained_fraction:.3f}")
    dilated = square_dilate_24(transferred) & target.valid
    valid_128 = downsample_all_4x4(target.valid)
    proposal_128 = downsample_any_4x4(dilated) & valid_128
    proposal_512 = nearest_upsample_4x4(proposal_128)
    if (
        not proposal_128.any()
        or np.all(proposal_128[valid_128])
        or not np.all(proposal_512 <= target.valid)
    ):
        raise ValueError("proposal must be non-empty, non-full, and subset of valid canvas")
    z0 = np.where(proposal_128, 1, -1).astype(np.int8)
    z0[~valid_128] = -1
    return ProposalTarget(
        transferred,
        proposal_512,
        proposal_128,
        valid_128,
        z0,
        retained_fraction,
        target_structure.support.copy(),
        target_structure.sdf_128.copy(),
        target_structure.u_x.copy(),
        target_structure.u_y.copy(),
        target_frame,
    )

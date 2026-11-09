"""Deterministic soft normal-structure references for Module A."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter

from .data import LoadedIdentity
from .identities import IdentityManifest, IdentityRecord

_ATLAS_TOKEN = object()

_GEOMETRY_RADII_128 = (1, 2, 4)
_GEOMETRY_WEIGHTS = np.asarray((0.5, 0.3, 0.2), dtype=np.float64)
_GEOMETRY_LENGTHS_512 = np.asarray((12.0, 20.0, 36.0), dtype=np.float64)
_GEOMETRY_EPSILON = 1.0e-12
_GEOMETRY_ENERGY_FLOOR = 2.0**-12
_GEOMETRY_EVIDENCE_THRESHOLD = 0.10
_GEOMETRY_SCALE_FALLBACK = 20.0
_GEOMETRY_SCALE_MIN = 4.0
_GEOMETRY_SCALE_MAX = 64.0
# 3.5 keeps a filled cable disk on the 16-atlas set; 6.0 collapses jacket to cores.
_SUPPORT_DISTANCE_MAD_K = 3.5
_SUPPORT_CLOSE_PX = 4


def geometry_grid_centers_512() -> np.ndarray:
    """Return the 128-grid node locations in continuous 512-pixel coordinates."""

    coordinates = np.arange(128, dtype=np.float64) * 4.0 + 2.0
    x, y = np.meshgrid(coordinates, coordinates)
    points = np.stack((x, y), axis=-1)
    points.setflags(write=False)
    return points


def _continuous_bilinear_stencil(
    points_512: np.ndarray, *, grid_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row/column/weight stencils under the frozen edge-clamp rule."""

    points = np.asarray(points_512, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("continuous sample points must be finite [...,2]")
    if grid_size == 512:
        origin, spacing = 0.5, 1.0
    elif grid_size == 128:
        origin, spacing = 2.0, 4.0
    else:
        raise ValueError("continuous sampling supports only 512 or 128 grids")
    x = np.clip((points[..., 0] - origin) / spacing, 0.0, float(grid_size - 1))
    y = np.clip((points[..., 1] - origin) / spacing, 0.0, float(grid_size - 1))
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, grid_size - 1), np.minimum(y0 + 1, grid_size - 1)
    wx, wy = x - x0, y - y0
    rows = np.stack((y0, y0, y1, y1), axis=-1)
    columns = np.stack((x0, x1, x0, x1), axis=-1)
    weights = np.stack(
        (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ),
        axis=-1,
    )
    return rows, columns, weights


def sample_continuous_field(values: np.ndarray, points_512: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 512 pixel-centre or 128 field-centre array.

    A 512 array is defined at pixel centres ``(x+0.5,y+0.5)``; a 128 array is
    defined at ``(4c+2,4r+2)``.  Finite coordinates outside the first/last
    node use one explicit edge-clamp rule.  Callers that score validity keep
    their own canvas mask, so clamped values can never make an invalid point
    legal.
    """

    array = np.asarray(values)
    if array.ndim < 2 or array.shape[0] != array.shape[1] or array.shape[0] not in (128, 512):
        raise ValueError("continuous sampled field must have leading shape 128x128 or 512x512")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("continuous sampled field must be finite numeric data")
    rows, columns, weights = _continuous_bilinear_stencil(points_512, grid_size=array.shape[0])
    samples = array[rows, columns].astype(np.float64, copy=False)
    trailing_dimensions = array.ndim - 2
    expanded_weights = weights.reshape(weights.shape + (1,) * trailing_dimensions)
    neighbor_axis = weights.ndim - 1
    return np.sum(samples * expanded_weights, axis=neighbor_axis, dtype=np.float64)


class _StructureConfig(Protocol):
    category: str
    image_size: int
    proposal_size: int
    support_border_width: int
    support_background_margin: float
    atlas_reference_count: int
    atlas_consensus: float


@dataclass(frozen=True)
class ObjectFrame:
    center_x: float
    center_y: float
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class GeometryField:
    """Frozen five-channel geometry field plus its diagnostic sufficient data.

    ``g`` is the only conditioning representation.  The remaining arrays make
    the energy/coherence calculation auditable without adding hidden feature
    channels to ``g``.
    """

    g: np.ndarray
    q: np.ndarray
    h: np.ndarray
    omega: np.ndarray
    eta: np.ndarray
    orientation_unknown: np.ndarray
    scale_fallback: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "g": ((128, 128, 5), np.floating),
            "q": ((128, 128, 2), np.floating),
            "h": ((128, 128, 3), np.floating),
            "omega": ((128, 128, 3), np.floating),
            "eta": ((128, 128), np.floating),
            "orientation_unknown": ((128, 128), np.bool_),
            "scale_fallback": ((128, 128), np.bool_),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"geometry field {name} must have shape {shape}")
            if dtype is np.bool_:
                if value.dtype != bool:
                    raise ValueError(f"geometry field {name} must be boolean")
            elif not np.issubdtype(value.dtype, dtype):
                raise ValueError(f"geometry field {name} must be floating point")
            if value.dtype != bool and not np.isfinite(value).all():
                raise ValueError(f"geometry field {name} must be finite")
            value.setflags(write=False)


@dataclass(frozen=True)
class LocalFrame:
    """One 512-continuous-pixel local frame with an explicit unknown gauge."""

    center: tuple[float, float]
    alpha: float
    rho: float
    s_parallel: float
    s_perp: float
    orientation_unknown: bool
    gauge: str

    def __post_init__(self) -> None:
        values = (*self.center, self.alpha, self.rho, self.s_parallel, self.s_perp)
        if len(self.center) != 2 or not all(np.isfinite(value) for value in values):
            raise ValueError("local frame values must be finite")
        if self.s_parallel <= 0.0 or self.s_perp <= 0.0:
            raise ValueError("local frame scales must be positive")
        expected_gauge = "world_axis" if self.orientation_unknown else "measured_axis"
        if self.gauge != expected_gauge:
            raise ValueError(f"local frame gauge must be {expected_gauge!r}")

    @property
    def axes(self) -> np.ndarray:
        cosine, sine = np.cos(self.alpha), np.sin(self.alpha)
        return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)

    @property
    def matrix(self) -> np.ndarray:
        return self.axes @ np.diag((self.s_parallel, self.s_perp))


@dataclass(frozen=True)
class BlindReferenceMasks:
    """Frozen blind region and one common dependency-valid reference ring."""

    b8_bounds: tuple[int, int, int, int]
    safety_bounds: tuple[int, int, int, int]
    blind_bounds: tuple[int, int, int, int]
    ring_inner_bounds: tuple[int, int, int, int]
    outer_bounds: tuple[int, int, int, int]
    coarse_center: tuple[float, float]
    blind_512: np.ndarray
    theoretical_ring_128: np.ndarray
    observable_ring_128: np.ndarray
    dependency_valid_128: np.ndarray
    common_ring_128: np.ndarray
    ring_denominator: int
    theoretical_cell_count: int
    valid_cell_count: int
    coverage: float
    common_ring_hash: str

    def __post_init__(self) -> None:
        for name in (
            "blind_512",
            "theoretical_ring_128",
            "observable_ring_128",
            "dependency_valid_128",
            "common_ring_128",
        ):
            value = getattr(self, name)
            expected_shape = (512, 512) if name == "blind_512" else (128, 128)
            if value.shape != expected_shape or value.dtype != bool:
                raise ValueError(f"{name} must be a {expected_shape} bool array")
            value.setflags(write=False)
        if self.ring_denominator < 0 or self.valid_cell_count < 0:
            raise ValueError("reference ring counts cannot be negative")
        if self.valid_cell_count > self.ring_denominator:
            raise ValueError("reference valid count cannot exceed its observable denominator")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("reference ring coverage must lie in [0, 1]")


@dataclass(frozen=True)
class ReferenceGeometryMatch:
    """Deterministic sixteen-reference geometry match evidence."""

    distances: np.ndarray
    weights: np.ndarray
    per_reference_denominator: np.ndarray
    entropy: float
    identity_order_hash: str
    common_ring_hash: str
    source_ring_mean_energy: float

    def __post_init__(self) -> None:
        for name in ("distances", "weights", "per_reference_denominator"):
            value = getattr(self, name)
            if value.shape != (16,):
                raise ValueError(f"reference match {name} must have shape (16,)")
            if name == "per_reference_denominator":
                if not np.issubdtype(value.dtype, np.integer):
                    raise ValueError("reference denominators must be integers")
            elif not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError(f"reference match {name} must be finite floating point")
            value.setflags(write=False)
        if not np.isfinite(self.entropy) or not np.isfinite(self.source_ring_mean_energy):
            raise ValueError("reference entropy must be finite")


class ReferenceGeometryInvalid(ValueError):
    """Fail-closed reference match carrying already computed energy evidence."""

    def __init__(self, reason: str, *, source_ring_mean_energy: float) -> None:
        if not np.isfinite(source_ring_mean_energy):
            raise ValueError("invalid reference energy evidence must be finite")
        super().__init__(reason)
        self.source_ring_mean_energy = float(source_ring_mean_energy)


@dataclass(frozen=True)
class ReferenceProxy:
    """Weighted normal-reference support, physical SDF, and geometry field."""

    field: GeometryField
    support_probability: np.ndarray
    sdf_128: np.ndarray
    observable_128: np.ndarray

    def __post_init__(self) -> None:
        if self.support_probability.shape != (512, 512) or not np.issubdtype(
            self.support_probability.dtype, np.floating
        ):
            raise ValueError("reference support probability must be floating 512x512")
        if self.sdf_128.shape != (128, 128) or not np.issubdtype(self.sdf_128.dtype, np.floating):
            raise ValueError("reference SDF must be floating 128x128")
        if self.observable_128.shape != (128, 128) or self.observable_128.dtype != bool:
            raise ValueError("reference observable mask must be a bool 128x128 array")
        if not np.isfinite(self.support_probability).all() or not np.isfinite(self.sdf_128).all():
            raise ValueError("reference proxy arrays must be finite")
        self.support_probability.setflags(write=False)
        self.sdf_128.setflags(write=False)
        self.observable_128.setflags(write=False)


def _half_open_rectangle(bounds: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bounds
    result = np.zeros((512, 512), dtype=bool)
    left, top = max(left, 0), max(top, 0)
    right, bottom = min(right, 512), min(bottom, 512)
    if left < right and top < bottom:
        result[top:bottom, left:right] = True
    return result


def _expand_half_open(bounds: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    return left - margin, top - margin, right + margin, bottom + margin


def _lattice_rectangle_count(bounds: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bounds
    first_column = int(np.ceil((left - 2.0) / 4.0))
    final_column = int(np.ceil((right - 2.0) / 4.0))
    first_row = int(np.ceil((top - 2.0) / 4.0))
    final_row = int(np.ceil((bottom - 2.0) / 4.0))
    return max(final_column - first_column, 0) * max(final_row - first_row, 0)


def _all_within_guard(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask.astype(np.int32), radius, mode="constant")
    integral = np.pad(
        padded.cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64),
        ((1, 0), (1, 0)),
    )
    width = 2 * radius + 1
    count = (
        integral[width:, width:]
        - integral[:-width, width:]
        - integral[width:, :-width]
        + integral[:-width, :-width]
    )
    return count == width * width


def geometry_observable_128(valid_512: np.ndarray) -> np.ndarray:
    """Return cells whose frozen 20-pixel geometry RF is wholly observable."""

    if valid_512.shape != (512, 512) or valid_512.dtype != bool:
        raise ValueError("geometry observability requires a bool 512x512 canvas")
    return _all_within_guard(valid_512, 20)[2::4, 2::4]


def build_blind_reference_masks(
    source_gt: np.ndarray,
    source_valid: np.ndarray,
    reference_valid: np.ndarray,
) -> BlindReferenceMasks:
    """Build B8/safety/blind masks and the fixed common 16-reference ring."""

    if source_gt.shape != (512, 512) or source_gt.dtype != bool or not source_gt.any():
        raise ValueError("source GT must be a non-empty 512x512 bool array")
    if source_valid.shape != (512, 512) or source_valid.dtype != bool:
        raise ValueError("source valid canvas must be a 512x512 bool array")
    if reference_valid.shape != (16, 512, 512) or reference_valid.dtype != bool:
        raise ValueError("reference valid canvases must be a 16x512x512 bool array")

    rows, columns = np.nonzero(source_gt)
    b8_bounds = (
        int(columns.min() // 8 * 8),
        int(rows.min() // 8 * 8),
        int((columns.max() + 8) // 8 * 8),
        int((rows.max() + 8) // 8 * 8),
    )
    safety_bounds = _expand_half_open(b8_bounds, 8)
    # The observation blind only protects the source context.  The additional
    # 20-pixel guard belongs to dependency validity, so the reference ring
    # starts outside it without enlarging the blind itself.
    blind_bounds = _expand_half_open(safety_bounds, 32)
    ring_inner_bounds = _expand_half_open(safety_bounds, 32 + 20)
    outer_bounds = _expand_half_open(safety_bounds, 96)
    blind_512 = _half_open_rectangle(blind_bounds)

    coordinates = np.arange(128, dtype=np.float64) * 4.0 + 2.0
    x, y = np.meshgrid(coordinates, coordinates)
    outer = (
        (x >= outer_bounds[0])
        & (x < outer_bounds[2])
        & (y >= outer_bounds[1])
        & (y < outer_bounds[3])
    )
    inner = (
        (x >= ring_inner_bounds[0])
        & (x < ring_inner_bounds[2])
        & (y >= ring_inner_bounds[1])
        & (y < ring_inner_bounds[3])
    )
    theoretical_ring_128 = outer & ~inner
    theoretical_cell_count = _lattice_rectangle_count(outer_bounds) - _lattice_rectangle_count(
        ring_inner_bounds
    )

    common_valid_512 = source_valid & np.all(reference_valid, axis=0)
    observable_128 = common_valid_512[2::4, 2::4]
    observable_ring_128 = theoretical_ring_128 & observable_128
    # The denominator is only the theoretical ring intersected with the
    # jointly observable canvas.  The numerator additionally proves that the
    # complete frozen RF of source and every reference does not touch either
    # the observation blind or an unobservable pixel.
    source_dependency = geometry_observable_128(source_valid & ~blind_512)
    reference_dependency = np.logical_and.reduce(
        tuple(geometry_observable_128(value) for value in reference_valid)
    )
    dependency_valid_128 = source_dependency & reference_dependency
    common_ring_128 = observable_ring_128 & dependency_valid_128
    ring_denominator = int(observable_ring_128.sum())
    valid_cell_count = int(common_ring_128.sum())
    coverage = float(valid_cell_count / ring_denominator) if ring_denominator > 0 else 0.0
    common_ring_hash = sha256(common_ring_128.tobytes(order="C")).hexdigest()
    coarse_center = (
        (b8_bounds[0] + b8_bounds[2]) / 2.0,
        (b8_bounds[1] + b8_bounds[3]) / 2.0,
    )
    return BlindReferenceMasks(
        b8_bounds,
        safety_bounds,
        blind_bounds,
        ring_inner_bounds,
        outer_bounds,
        coarse_center,
        blind_512,
        theoretical_ring_128,
        observable_ring_128,
        dependency_valid_128,
        common_ring_128,
        ring_denominator,
        theoretical_cell_count,
        valid_cell_count,
        coverage,
        common_ring_hash,
    )


def _geometry_inputs(
    rgb: np.ndarray,
    valid: np.ndarray,
    sdf_128: np.ndarray | None,
    sdf_px_512: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("geometry RGB must be a 512x512 uint8 image")
    if valid.shape != (512, 512) or valid.dtype != bool:
        raise ValueError("geometry valid canvas must be a 512x512 bool array")
    if not valid.any():
        raise ValueError("geometry valid canvas must be non-empty")
    if sdf_128 is not None and sdf_px_512 is not None:
        raise ValueError("provide either legacy sdf_128 or sdf_px_512, not both")
    if sdf_128 is not None:
        if sdf_128.shape != (128, 128) or not np.issubdtype(sdf_128.dtype, np.floating):
            raise ValueError("geometry SDF must be a floating 128x128 array")
        if not np.isfinite(sdf_128).all():
            raise ValueError("geometry SDF must be finite")
        # Compatibility path for existing callers.  New G0 paths supply the
        # physical 512-pixel SDF and therefore never lose sub-cell gradients.
        sdf_px_512 = np.repeat(np.repeat(sdf_128.astype(np.float64), 4, axis=0), 4, axis=1)
    if sdf_px_512 is not None:
        if sdf_px_512.shape != (512, 512) or not np.issubdtype(sdf_px_512.dtype, np.floating):
            raise ValueError("geometry physical SDF must be a floating 512x512 array")
        if not np.isfinite(sdf_px_512).all():
            raise ValueError("geometry physical SDF must be finite")

    rgb_values = rgb.astype(np.float64)
    red, green, blue = np.moveaxis(rgb_values, -1, 0)
    luminance = (red + 2.0 * green + blue) / (4.0 * 255.0)
    chroma_rg = (red - green) / (2.0 * 255.0)
    chroma_yb = (red + green - 2.0 * blue) / (4.0 * 255.0)
    d = (
        np.zeros((512, 512), dtype=np.float64)
        if sdf_px_512 is None
        else 0.5 * np.clip(sdf_px_512.astype(np.float64, copy=False) / 64.0, -1.0, 1.0)
    )
    # All four channels live in the same 512-pixel coordinate system.  The
    # tensor sufficient statistics are formed before their sole 4x4 pooling.
    features = np.stack((luminance, chroma_rg, chroma_yb, d), axis=-1)
    valid_128 = valid.reshape(128, 4, 128, 4).all(axis=(1, 3))
    return features, valid_128


def _gradient_dependency(valid: np.ndarray) -> np.ndarray:
    dependency = np.zeros_like(valid)
    dependency[1:-1, 1:-1] = (
        valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:] & valid[:-2, 1:-1] & valid[2:, 1:-1]
    )
    return dependency


def _masked_box_mean(values: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    padded_values = np.pad(values * mask, radius, mode="constant")
    padded_mask = np.pad(mask.astype(np.float64), radius, mode="constant")
    value_integral = np.pad(
        padded_values.cumsum(0, dtype=np.float64).cumsum(1, dtype=np.float64),
        ((1, 0), (1, 0)),
    )
    mask_integral = np.pad(
        padded_mask.cumsum(0, dtype=np.float64).cumsum(1, dtype=np.float64),
        ((1, 0), (1, 0)),
    )
    width = 2 * radius + 1
    value_sum = (
        value_integral[width:, width:]
        - value_integral[:-width, width:]
        - value_integral[width:, :-width]
        + value_integral[:-width, :-width]
    )
    count = (
        mask_integral[width:, width:]
        - mask_integral[:-width, width:]
        - mask_integral[width:, :-width]
        + mask_integral[:-width, :-width]
    )
    return np.divide(value_sum, count, out=np.zeros_like(value_sum), where=count > 0.0)


def _aggregate_tensor_4x4(values: np.ndarray, dependency: np.ndarray) -> np.ndarray:
    sums = (values * dependency).reshape(128, 4, 128, 4).sum(axis=(1, 3))
    counts = dependency.reshape(128, 4, 128, 4).sum(axis=(1, 3))
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def estimate_geometry_field(
    rgb: np.ndarray,
    *,
    valid: np.ndarray,
    sdf_128: np.ndarray | None = None,
    sdf_px_512: np.ndarray | None = None,
) -> GeometryField:
    """Estimate the frozen category-free geometry field on the 128 grid.

    Tensor radii are 128-grid cells.  All reported scales are 512-grid
    continuous pixels.  RGB and optional 512-physical-pixel SDF contribute
    only deterministic geometry gradients; the returned conditioning tensor
    remains exactly the authoritative five channels.
    """

    features, valid_128 = _geometry_inputs(rgb, valid, sdf_128, sdf_px_512)
    # Apply the dependency mask before any derivative or multi-scale filter.
    # A caller can therefore remove a blind region (including its fixed guard)
    # from ``valid`` without any RGB inside that region entering the tensor.
    features = np.where(valid[..., None], features, 0.0)
    gradient_y, gradient_x = np.gradient(features, axis=(0, 1), edge_order=1)
    dependency_512 = _gradient_dependency(valid)
    gradient_x = np.where(dependency_512[..., None], gradient_x, 0.0)
    gradient_y = np.where(dependency_512[..., None], gradient_y, 0.0)

    raw_xx = _aggregate_tensor_4x4(np.sum(gradient_x * gradient_x, axis=2), dependency_512)
    raw_xy = _aggregate_tensor_4x4(np.sum(gradient_x * gradient_y, axis=2), dependency_512)
    raw_yy = _aggregate_tensor_4x4(np.sum(gradient_y * gradient_y, axis=2), dependency_512)
    dependency = _gradient_dependency(valid_128)

    h = np.zeros((128, 128, 3), dtype=np.float64)
    scale_evidence = np.zeros_like(h)
    coherence = np.zeros_like(h)
    tangent = np.zeros((128, 128, 3, 2), dtype=np.float64)
    for index, (radius, weight) in enumerate(
        zip(_GEOMETRY_RADII_128, _GEOMETRY_WEIGHTS, strict=True)
    ):
        tensor_xx = _masked_box_mean(raw_xx, dependency, radius)
        tensor_xy = _masked_box_mean(raw_xy, dependency, radius)
        tensor_yy = _masked_box_mean(raw_yy, dependency, radius)
        energy = tensor_xx + tensor_yy
        delta = np.sqrt(
            np.maximum((tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy**2, 0.0)
            + _GEOMETRY_EPSILON**2
        )
        coherence[..., index] = np.clip(delta / (energy + _GEOMETRY_EPSILON), 0.0, 1.0)
        gate = energy / (energy + _GEOMETRY_ENERGY_FLOOR)
        h[..., index] = weight * coherence[..., index] * gate
        scale_evidence[..., index] = weight * gate
        tangent[..., index, 0] = (tensor_yy - tensor_xx) / delta
        tangent[..., index, 1] = (-2.0 * tensor_xy) / delta

    h *= valid_128[..., None]
    scale_evidence *= valid_128[..., None]
    z = np.sum(h[..., None] * tangent, axis=2)
    rho = np.clip(np.linalg.norm(z, axis=2), 0.0, 1.0)
    orientation_unknown = (rho < _GEOMETRY_EVIDENCE_THRESHOLD) | ~valid_128
    rho[orientation_unknown] = 0.0
    q = np.divide(
        z,
        rho[..., None],
        out=np.zeros_like(z),
        where=(~orientation_unknown)[..., None],
    )

    eta = np.sum(scale_evidence, axis=2)
    omega = scale_evidence / (eta[..., None] + _GEOMETRY_EPSILON)
    scale_fallback = (eta < _GEOMETRY_EVIDENCE_THRESHOLD) | ~valid_128
    log_length = np.sum(omega * np.log(_GEOMETRY_LENGTHS_512), axis=2)
    local_length = np.exp(log_length)
    local_length[scale_fallback] = _GEOMETRY_SCALE_FALLBACK

    mean_coherence = np.sum(omega * coherence, axis=2)
    anisotropy = np.sqrt((1.0 + mean_coherence) / (1.0 - mean_coherence + _GEOMETRY_EPSILON))
    anisotropy = np.clip(anisotropy, 1.0, 4.0)
    anisotropy[orientation_unknown] = 1.0
    root_anisotropy = np.sqrt(anisotropy)
    s_parallel = np.clip(local_length * root_anisotropy, _GEOMETRY_SCALE_MIN, _GEOMETRY_SCALE_MAX)
    s_perp = np.clip(local_length / root_anisotropy, _GEOMETRY_SCALE_MIN, _GEOMETRY_SCALE_MAX)

    g = np.empty((128, 128, 5), dtype=np.float64)
    g[..., :2] = rho[..., None] * q
    g[..., 2] = rho
    g[..., 3] = np.log(s_parallel)
    g[..., 4] = np.log(s_perp)
    return GeometryField(g, q, h, omega, eta, orientation_unknown, scale_fallback)


def _bilinear_128(values: np.ndarray, *, center: tuple[float, float]) -> np.ndarray:
    """Sample a 128 field whose cell centres are at ``4c+2`` in 512 pixels."""

    return sample_continuous_field(values, np.asarray(center, dtype=np.float64))


def geometry_sample_observability(
    observable_128: np.ndarray, sample_points_512: np.ndarray
) -> np.ndarray:
    """Return a per-point mask for bilinear 128-proxy observability."""

    if observable_128.shape != (128, 128) or observable_128.dtype != bool:
        raise ValueError("proxy geometry observability must be a bool 128x128 array")
    if (
        sample_points_512.ndim != 2
        or sample_points_512.shape[1] != 2
        or not np.issubdtype(sample_points_512.dtype, np.floating)
        or not np.isfinite(sample_points_512).all()
    ):
        raise ValueError("proxy geometry samples must be finite Nx2 floats")
    in_canvas = np.all((sample_points_512 >= 0.0) & (sample_points_512 < 512.0), axis=1)
    rows, columns, weights = _continuous_bilinear_stencil(sample_points_512, grid_size=128)
    unavailable_contributor = (weights > 0.0) & ~observable_128[rows, columns]
    return in_canvas & ~unavailable_contributor.any(axis=-1)


def require_geometry_samples_observable(
    observable_128: np.ndarray, sample_points_512: np.ndarray
) -> None:
    """Reject scoring samples whose 4x4 source proxy cell is unavailable."""

    if not geometry_sample_observability(observable_128, sample_points_512).all():
        raise ValueError("proxy geometry is not observable at every scoring sample")


def require_stencil_observable_coverage(
    observable_mask: np.ndarray, *, min_fraction: float = 0.75
) -> None:
    """Reject a stencil whose observable fraction falls below the frozen floor."""

    mask = np.asarray(observable_mask)
    if mask.ndim != 1 or mask.dtype != bool or mask.size == 0:
        raise ValueError("stencil observability must be a non-empty bool vector")
    if not np.isfinite(min_fraction) or not (0.0 < float(min_fraction) <= 1.0):
        raise ValueError("min observable fraction must lie in (0, 1]")
    coverage = float(mask.mean(dtype=np.float64))
    if coverage < float(min_fraction):
        raise ValueError(
            "proxy geometry observable coverage "
            f"{coverage:.6f} is below {float(min_fraction):.6f}"
        )


def estimate_local_frame(field: GeometryField, *, center: tuple[float, float]) -> LocalFrame:
    """Read one bilinearly interpolated local frame in 512-pixel coordinates."""

    if not isinstance(field, GeometryField):
        raise ValueError("local frame requires a GeometryField")
    if len(center) != 2 or not all(np.isfinite(value) for value in center):
        raise ValueError("local frame center must contain two finite coordinates")
    center_x, center_y = map(float, center)
    if not (0.0 <= center_x < 512.0 and 0.0 <= center_y < 512.0):
        raise ValueError("local frame center must lie in the 512 canvas")
    point = np.asarray((center_x, center_y), dtype=np.float64)
    rho = float(sample_continuous_field(field.g[..., 2], point))
    log_scales = sample_continuous_field(field.g[..., 3:5], point)
    q = sample_continuous_field(field.q, point)
    unknown = rho < _GEOMETRY_EVIDENCE_THRESHOLD
    if unknown:
        rho = 0.0
    else:
        q_norm = float(np.linalg.norm(q))
        unknown = q_norm <= _GEOMETRY_EPSILON
        if not unknown:
            q = q / q_norm
    s_parallel, s_perp = np.exp(log_scales)
    if unknown:
        alpha = 0.0
        isotropic = float(np.sqrt(s_parallel * s_perp))
        s_parallel = s_perp = isotropic
        gauge = "world_axis"
    else:
        alpha = 0.5 * float(np.arctan2(q[1], q[0]))
        gauge = "measured_axis"
    return LocalFrame(
        (center_x, center_y),
        alpha,
        rho,
        float(s_parallel),
        float(s_perp),
        unknown,
        gauge,
    )


def estimate_local_frame_from_observable_ring(
    field: GeometryField,
    masks: BlindReferenceMasks,
    *,
    observable_128: np.ndarray | None = None,
) -> LocalFrame:
    """Aggregate one source frame from the blind-gap ring, keeping GT center."""

    if not isinstance(field, GeometryField):
        raise ValueError("ring frame requires a GeometryField")
    if not isinstance(masks, BlindReferenceMasks):
        raise ValueError("ring frame requires frozen blind/reference masks")
    ring = np.asarray(masks.common_ring_128, dtype=bool)
    if observable_128 is not None:
        if observable_128.shape != (128, 128) or observable_128.dtype != bool:
            raise ValueError("ring-frame observability must be a bool 128x128 array")
        ring = ring & observable_128
    rows, columns = np.nonzero(ring)
    if rows.size == 0:
        raise ValueError("observable ring is empty")
    center_x, center_y = (float(masks.coarse_center[0]), float(masks.coarse_center[1]))
    if not (0.0 <= center_x < 512.0 and 0.0 <= center_y < 512.0):
        raise ValueError("local frame center must lie in the 512 canvas")

    rho = field.g[rows, columns, 2]
    oriented = (rho >= _GEOMETRY_EVIDENCE_THRESHOLD) & ~field.orientation_unknown[rows, columns]
    eta = field.eta[rows, columns]
    scale_ok = (eta >= _GEOMETRY_EVIDENCE_THRESHOLD) & ~field.scale_fallback[rows, columns]
    if scale_ok.any():
        log_scales = np.average(
            field.g[rows, columns][scale_ok, 3:5],
            axis=0,
            weights=eta[scale_ok],
        )
        s_parallel, s_perp = np.exp(log_scales)
    else:
        s_parallel = s_perp = _GEOMETRY_SCALE_FALLBACK
    s_parallel = float(np.clip(s_parallel, _GEOMETRY_SCALE_MIN, _GEOMETRY_SCALE_MAX))
    s_perp = float(np.clip(s_perp, _GEOMETRY_SCALE_MIN, _GEOMETRY_SCALE_MAX))

    if not oriented.any():
        isotropic = float(np.sqrt(s_parallel * s_perp))
        return LocalFrame(
            (center_x, center_y),
            0.0,
            0.0,
            isotropic,
            isotropic,
            True,
            "world_axis",
        )
    z = np.sum(field.g[rows, columns][oriented, :2], axis=0, dtype=np.float64)
    z_norm = float(np.linalg.norm(z))
    if z_norm <= _GEOMETRY_EPSILON:
        isotropic = float(np.sqrt(s_parallel * s_perp))
        return LocalFrame(
            (center_x, center_y),
            0.0,
            0.0,
            isotropic,
            isotropic,
            True,
            "world_axis",
        )
    q = z / z_norm
    return LocalFrame(
        (center_x, center_y),
        0.5 * float(np.arctan2(q[1], q[0])),
        float(np.clip(np.mean(rho[oriented], dtype=np.float64), 0.0, 1.0)),
        s_parallel,
        s_perp,
        False,
        "measured_axis",
    )


def match_reference_geometry(
    source_field: GeometryField,
    reference_fields: tuple[GeometryField, ...],
    masks: BlindReferenceMasks,
    reference_identities: tuple[str, ...],
) -> ReferenceGeometryMatch:
    """Match exactly 16 RGB-derived fields on one common dependency ring."""

    if not isinstance(source_field, GeometryField):
        raise ValueError("source reference matching requires a GeometryField")
    if len(reference_fields) != 16 or any(
        not isinstance(field, GeometryField) for field in reference_fields
    ):
        raise ValueError("reference matching requires exactly 16 GeometryFields")
    if (
        len(reference_identities) != 16
        or len(set(reference_identities)) != 16
        or any(not isinstance(identity, str) or not identity for identity in reference_identities)
    ):
        raise ValueError("reference matching requires 16 unique non-empty identities")
    if not isinstance(masks, BlindReferenceMasks):
        raise ValueError("reference matching requires frozen blind/reference masks")
    ring = masks.common_ring_128
    denominator = int(ring.sum())
    if denominator != masks.valid_cell_count:
        raise ValueError("reference matching ring is inconsistent or degenerate")
    if denominator < 64 or masks.coverage < 0.75:
        raise ValueError(
            "reference dependency ring invalid: "
            f"observable_denominator={masks.ring_denominator}, "
            f"valid_count={masks.valid_cell_count}, coverage={masks.coverage:.6f}; "
            "requires min_cells=64 and min_coverage=0.750000"
        )

    ring_rows, ring_columns = np.nonzero(ring)
    ring_points = np.column_stack(
        (
            ring_columns.astype(np.float64) * 4.0 + 2.0,
            ring_rows.astype(np.float64) * 4.0 + 2.0,
        )
    )
    source_energy = float(
        np.mean(sample_continuous_field(source_field.eta, ring_points), dtype=np.float64)
    )
    if source_energy < _GEOMETRY_EVIDENCE_THRESHOLD:
        raise ReferenceGeometryInvalid(
            "source ring mean energy is below 0.10",
            source_ring_mean_energy=source_energy,
        )

    source_q = sample_continuous_field(source_field.q, ring_points)
    source_g = sample_continuous_field(source_field.g, ring_points)
    source_rho = source_g[:, 2].copy()
    source_q[source_rho < _GEOMETRY_EVIDENCE_THRESHOLD] = 0.0
    source_rho[source_rho < _GEOMETRY_EVIDENCE_THRESHOLD] = 0.0
    source_log_scales = source_g[:, 3:5]
    distances = np.empty(16, dtype=np.float64)
    for index, reference in enumerate(reference_fields):
        reference_q = sample_continuous_field(reference.q, ring_points)
        reference_g = sample_continuous_field(reference.g, ring_points)
        reference_rho = reference_g[:, 2].copy()
        reference_q[reference_rho < _GEOMETRY_EVIDENCE_THRESHOLD] = 0.0
        reference_rho[reference_rho < _GEOMETRY_EVIDENCE_THRESHOLD] = 0.0
        reference_log_scales = reference_g[:, 3:5]
        orientation = 0.5 * np.linalg.norm(reference_q - source_q, axis=1) / 2.0
        confidence = 0.25 * np.abs(reference_rho - source_rho)
        scale = (
            0.125 * np.sum(np.abs(reference_log_scales - source_log_scales), axis=1) / np.log(16.0)
        )
        distances[index] = np.mean(orientation + confidence + scale)
    logits = -distances / 0.25
    logits -= logits.max()
    unnormalized = np.exp(logits)
    weights = unnormalized / unnormalized.sum(dtype=np.float64)
    entropy = float(-(weights * np.log(weights)).sum(dtype=np.float64))
    identity_order_hash = sha256("\0".join(reference_identities).encode()).hexdigest()
    per_reference_denominator = np.full(16, denominator, dtype=np.int64)
    return ReferenceGeometryMatch(
        distances,
        weights,
        per_reference_denominator,
        entropy,
        identity_order_hash,
        masks.common_ring_hash,
        source_energy,
    )


def aggregate_reference_proxy(
    reference_fields: tuple[GeometryField, ...],
    supports: np.ndarray,
    sdf_128: np.ndarray,
    weights: np.ndarray,
    *,
    observable_128: np.ndarray | None = None,
) -> ReferenceProxy:
    """Aggregate normal-reference geometry/support/SDF without resampling."""

    if len(reference_fields) != 16 or any(
        not isinstance(field, GeometryField) for field in reference_fields
    ):
        raise ValueError("reference proxy requires exactly 16 GeometryFields")
    if supports.shape != (16, 512, 512) or supports.dtype != bool:
        raise ValueError("reference supports must be a 16x512x512 bool array")
    if sdf_128.shape != (16, 128, 128) or not np.issubdtype(sdf_128.dtype, np.floating):
        raise ValueError("reference SDFs must be a floating 16x128x128 array")
    if not np.isfinite(sdf_128).all():
        raise ValueError("reference SDFs must be finite")
    if (
        weights.shape != (16,)
        or not np.issubdtype(weights.dtype, np.floating)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
    ):
        raise ValueError("reference weights must be 16 finite non-negative floats")
    weight_sum = float(weights.sum(dtype=np.float64))
    if weight_sum <= 0.0:
        raise ValueError("reference weights must have positive mass")
    normalized_weights = weights.astype(np.float64, copy=True) / weight_sum
    if observable_128 is None:
        observable_128 = np.ones((128, 128), dtype=bool)
    if observable_128.shape != (128, 128) or observable_128.dtype != bool:
        raise ValueError("reference proxy observable mask must be a bool 128x128 array")

    first_three = np.stack([field.g[..., :3] for field in reference_fields])
    weighted_first_three = np.tensordot(normalized_weights, first_three, axes=(0, 0))
    z = weighted_first_three[..., :2]
    rho = np.linalg.norm(z, axis=2)
    orientation_unknown = rho < _GEOMETRY_EVIDENCE_THRESHOLD
    rho[orientation_unknown] = 0.0
    q = np.divide(
        z,
        rho[..., None],
        out=np.zeros_like(z),
        where=(~orientation_unknown)[..., None],
    )
    rho = np.clip(rho, 0.0, 1.0)

    log_scales = np.tensordot(
        normalized_weights,
        np.stack([field.g[..., 3:5] for field in reference_fields]),
        axes=(0, 0),
    )
    eta = np.tensordot(
        normalized_weights,
        np.stack([field.eta for field in reference_fields]),
        axes=(0, 0),
    )
    scale_fallback = eta < _GEOMETRY_EVIDENCE_THRESHOLD
    log_scales[scale_fallback] = np.log(_GEOMETRY_SCALE_FALLBACK)
    isotropic_log_scale = np.mean(log_scales, axis=2)
    log_scales[orientation_unknown, 0] = isotropic_log_scale[orientation_unknown]
    log_scales[orientation_unknown, 1] = isotropic_log_scale[orientation_unknown]

    h = np.tensordot(
        normalized_weights,
        np.stack([field.h for field in reference_fields]),
        axes=(0, 0),
    )
    omega = np.tensordot(
        normalized_weights,
        np.stack([field.omega for field in reference_fields]),
        axes=(0, 0),
    )
    omega[scale_fallback] = 0.0
    g = np.empty((128, 128, 5), dtype=np.float64)
    g[..., :2] = rho[..., None] * q
    g[..., 2] = rho
    g[..., 3:5] = log_scales
    field = GeometryField(g, q, h, omega, eta, orientation_unknown, scale_fallback)

    support_probability = np.tensordot(normalized_weights, supports.astype(np.float64), axes=(0, 0))
    proxy_sdf = np.tensordot(
        normalized_weights, sdf_128.astype(np.float64, copy=False), axes=(0, 0)
    )
    return ReferenceProxy(field, support_probability, proxy_sdf, observable_128.copy())


@dataclass(frozen=True)
class NormalStructure:
    normal: LoadedIdentity
    valid: np.ndarray
    support: np.ndarray
    frame: ObjectFrame
    sdf_128: np.ndarray
    u_x: np.ndarray
    u_y: np.ndarray

    def __post_init__(self) -> None:
        for array in (self.valid, self.support, self.sdf_128, self.u_x, self.u_y):
            array.setflags(write=False)


class NormalAtlas:
    """An immutable support atlas whose constructor is private to this module."""

    __slots__ = ("_category", "_support_bytes", "_item_ids", "_sample_ids", "_authority", "_sealed")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("NormalAtlas is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        token: object,
        category: str,
        support: np.ndarray,
        item_ids: tuple[str, ...],
        sample_ids: tuple[str, ...],
        authority: str,
    ) -> None:
        if token is not _ATLAS_TOKEN:
            raise ValueError("NormalAtlas instances must be built by build_normal_atlas")
        if support.shape != (512, 512) or support.dtype != bool:
            raise ValueError("atlas support must be a 512x512 bool array")
        if len(item_ids) != 16 or len(sample_ids) != 16:
            raise ValueError("normal atlas requires exactly 16 reference IDs")
        if len(set(item_ids)) != 16 or len(set(sample_ids)) != 16:
            raise ValueError("normal atlas reference IDs must be unique")
        self._category, self._support_bytes = category, support.tobytes()
        self._item_ids, self._sample_ids, self._authority = item_ids, sample_ids, authority
        self._sealed = True

    @property
    def category(self) -> str:
        return self._category

    @property
    def support(self) -> np.ndarray:
        return np.frombuffer(self._support_bytes, dtype=bool).reshape(512, 512)

    @property
    def reference_item_ids(self) -> tuple[str, ...]:
        return self._item_ids

    @property
    def reference_sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def matches_manifest(self, manifest: IdentityManifest) -> bool:
        return manifest.is_authoritative() and self._authority == manifest.authority_digest


def _record_identity(record: IdentityRecord) -> tuple[str, ... | None]:
    return (
        record.category,
        record.defect_type,
        record.item_id,
        record.image_relpath,
        record.mask_relpath,
        record.role,
        record.sample_id,
    )


def _authority(manifest: IdentityManifest) -> str:
    payload = repr(
        (
            manifest.schema_version,
            manifest.category,
            manifest.defect_types,
            manifest.atlas_normal_ids,
            tuple(map(_record_identity, manifest.normal_fit)),
            tuple(map(_record_identity, manifest.fewshot)),
        )
    )
    return sha256(payload.encode()).hexdigest()


def _frame(mask: np.ndarray) -> ObjectFrame:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("support frame requires non-empty support")
    left, right, top, bottom = xs.min(), xs.max(), ys.min(), ys.max()
    return ObjectFrame(
        center_x=(float(left) + float(right)) / 2.0,
        center_y=(float(top) + float(bottom)) / 2.0,
        scale_x=max((float(right) - float(left) + 1.0) / 2.0, 1.0),
        scale_y=max((float(bottom) - float(top) + 1.0) / 2.0, 1.0),
    )


def support_frame(support: np.ndarray) -> ObjectFrame:
    """Return an object coordinate frame; support remains a soft reference."""

    if support.shape != (512, 512) or support.dtype != bool:
        raise ValueError("support must be a 512x512 bool array")
    return _frame(support)


def _valid_border(valid: np.ndarray, width: int) -> np.ndarray:
    ys, xs = np.nonzero(valid)
    if not len(xs):
        raise ValueError("normal valid canvas must be non-empty")
    left, right, top, bottom = xs.min(), xs.max(), ys.min(), ys.max()
    border = np.zeros_like(valid)
    border[top : min(top + width, bottom + 1), left : right + 1] = True
    border[max(bottom - width + 1, top) : bottom + 1, left : right + 1] = True
    border[top : bottom + 1, left : min(left + width, right + 1)] = True
    border[top : bottom + 1, max(right - width + 1, left) : right + 1] = True
    return border & valid


def _squared_edt_1d(features: np.ndarray) -> np.ndarray:
    """Exact squared Euclidean distances to true positions on one finite line."""

    length = features.size
    source = np.flatnonzero(features)
    if source.size == 0:
        return np.full(length, np.inf, dtype=np.float64)
    vertices = np.empty(source.size, dtype=np.intp)
    intersections = np.empty(source.size + 1, dtype=np.float64)
    vertices[0] = source[0]
    intersections[0] = -np.inf
    intersections[1] = np.inf
    count = 0
    for position in source[1:]:
        intersection = float(position * position - vertices[count] * vertices[count]) / float(
            2 * (position - vertices[count])
        )
        while intersection <= intersections[count]:
            count -= 1
            intersection = float(position * position - vertices[count] * vertices[count]) / float(
                2 * (position - vertices[count])
            )
        count += 1
        vertices[count] = position
        intersections[count] = intersection
        intersections[count + 1] = np.inf
    distance = np.empty(length, dtype=np.float64)
    count = 0
    for position in range(length):
        while intersections[count + 1] < position:
            count += 1
        distance[position] = float((position - vertices[count]) ** 2)
    return distance


def _distance_to(features: np.ndarray) -> np.ndarray:
    """Exact two-pass 2-D Euclidean distance transform using NumPy only."""

    if features.ndim != 2 or features.dtype != bool:
        raise ValueError("distance-transform features must be a two-dimensional bool array")
    height, width = features.shape
    vertical = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        vertical[row] = _squared_edt_1d(features[row])
    squared = np.empty((height, width), dtype=np.float64)
    for column in range(width):
        costs = vertical[:, column]
        source = np.flatnonzero(np.isfinite(costs))
        if source.size == 0:
            squared[:, column] = np.inf
            continue
        vertices = np.empty(source.size, dtype=np.intp)
        intersections = np.empty(source.size + 1, dtype=np.float64)
        vertices[0] = source[0]
        intersections[0] = -np.inf
        intersections[1] = np.inf
        count = 0
        for position in source[1:]:
            intersection = float(
                costs[position]
                + position * position
                - costs[vertices[count]]
                - vertices[count] * vertices[count]
            ) / float(2 * (position - vertices[count]))
            while intersection <= intersections[count]:
                count -= 1
                intersection = float(
                    costs[position]
                    + position * position
                    - costs[vertices[count]]
                    - vertices[count] * vertices[count]
                ) / float(2 * (position - vertices[count]))
            count += 1
            vertices[count] = position
            intersections[count] = intersection
            intersections[count + 1] = np.inf
        count = 0
        for position in range(height):
            while intersections[count + 1] < position:
                count += 1
            squared[position, column] = (position - vertices[count]) ** 2 + costs[vertices[count]]
    return np.sqrt(squared)


def _boundary(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            if y_offset != 1 or x_offset != 1:
                interior &= padded[y_offset : y_offset + height, x_offset : x_offset + width]
    return mask & ~interior


def physical_sdf_px_512(support: np.ndarray) -> np.ndarray:
    """Return the signed physical-pixel SDF used by the four-channel field."""

    if support.shape != (512, 512) or support.dtype != bool:
        raise ValueError("physical SDF support must be a 512x512 bool array")
    padded_support = np.pad(support, 1, constant_values=False)
    inner = _distance_to(~padded_support)[1:-1, 1:-1]
    outer = _distance_to(support)
    signed = inner - outer
    signed[_boundary(support)] = 0.0
    return signed


def _sdf_and_coordinates(
    support: np.ndarray, frame: ObjectFrame, proposal_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if proposal_size != 128:
        raise ValueError("v0.6-A proposal_size must be exactly 128")
    signed = physical_sdf_px_512(support)
    # Keep the RC1 model-conditioning structure unchanged.  G0 obtains its
    # physical 4c+2 diagnostic SDF directly through ``physical_sdf_px_512``.
    normalization = max(frame.scale_x, frame.scale_y, 1.0)
    grid = np.arange(128, dtype=np.float32) * 4.0 + 1.5
    x, y = np.meshgrid(grid, grid)
    sdf = signed[1::4, 1::4].astype(np.float32) / normalization
    return (
        sdf,
        ((x - frame.center_x) / frame.scale_x).astype(np.float32),
        ((y - frame.center_y) / frame.scale_y).astype(np.float32),
    )


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array(mask, copy=True)
    size = 2 * radius + 1
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    closed = image.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.asarray(closed, dtype=bool)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    best = np.zeros(mask.shape, dtype=bool)
    best_count = 0
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for start_row, start_column in zip(*np.nonzero(mask), strict=True):
        if visited[start_row, start_column]:
            continue
        queue: deque[tuple[int, int]] = deque(((start_row, start_column),))
        visited[start_row, start_column] = True
        cells = [(start_row, start_column)]
        while queue:
            row, column = queue.popleft()
            for delta_row, delta_column in neighbors:
                next_row, next_column = row + delta_row, column + delta_column
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
                    cells.append((next_row, next_column))
        if len(cells) > best_count:
            best_count = len(cells)
            best = np.zeros(mask.shape, dtype=bool)
            rows, columns = zip(*cells, strict=True)
            best[rows, columns] = True
    return best


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    inverse = ~mask
    height, width = mask.shape
    reachable = np.zeros(mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def seed(row: int, column: int) -> None:
        if inverse[row, column] and not reachable[row, column]:
            reachable[row, column] = True
            queue.append((row, column))

    for column in range(width):
        seed(0, column)
        seed(height - 1, column)
    for row in range(height):
        seed(row, 0)
        seed(row, width - 1)
    while queue:
        row, column = queue.popleft()
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_column = row + delta_row, column + delta_column
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and inverse[next_row, next_column]
                and not reachable[next_row, next_column]
            ):
                reachable[next_row, next_column] = True
                queue.append((next_row, next_column))
    return mask | ~reachable


def _estimate_support(normal: LoadedIdentity, config: _StructureConfig) -> np.ndarray:
    border = _valid_border(normal.valid, config.support_border_width)
    background = normal.rgb[border].astype(np.float32)
    median = np.median(background, axis=0)
    distance = np.max(np.abs(normal.rgb.astype(np.float32) - median), axis=2)
    border_distance = distance[border]
    location = float(np.median(border_distance))
    scatter = float(np.median(np.abs(border_distance - location)))
    channel_mad = np.median(np.abs(background - median), axis=0)
    legacy_threshold = float(np.max(channel_mad) + config.support_background_margin * 255.0)
    threshold = max(location + _SUPPORT_DISTANCE_MAD_K * scatter, legacy_threshold)
    raw = normal.valid & (distance > threshold)
    if not raw.any():
        return normal.valid.copy()
    component = _largest_connected_component(raw)
    closed = _binary_close(component, _SUPPORT_CLOSE_PX) & normal.valid
    if not closed.any():
        closed = component
    filled = _fill_binary_holes(closed) & normal.valid
    return filled if filled.any() else component


def estimate_normal_structure(
    normal: LoadedIdentity, *, config: _StructureConfig
) -> NormalStructure:
    """Estimate a deterministic, non-legal coarse support from normal RGB and V only."""

    if normal.record.role not in {"normal_fit", "normal_dev", "open_normal"}:
        raise ValueError("structure estimation accepts normal identities only")
    if normal.anomaly_gt is not None:
        raise ValueError("structure estimation cannot read an anomaly ground truth")
    if config.image_size != 512 or config.proposal_size != 128:
        raise ValueError("v0.6-A requires 512 image and 128 proposal grids")
    if config.support_border_width <= 0 or config.support_background_margin < 0:
        raise ValueError("invalid frozen support estimator configuration")
    support = _estimate_support(normal, config)
    frame = _frame(support)
    sdf, u_x, u_y = _sdf_and_coordinates(support, frame, config.proposal_size)
    return NormalStructure(normal, normal.valid.copy(), support, frame, sdf, u_x, u_y)


def build_normal_atlas(
    references: tuple[LoadedIdentity, ...],
    *,
    config: _StructureConfig,
    identity_manifest: IdentityManifest,
) -> NormalAtlas:
    """Build the fixed 16-reference category atlas using a 0.5 support consensus."""

    if len(references) != config.atlas_reference_count or len(references) != 16:
        raise ValueError("normal atlas requires exactly 16 loaded A-fit references")
    if (
        identity_manifest.category != config.category
        or identity_manifest.defect_types != config.defect_types
    ):
        raise ValueError("atlas manifest must agree with the experiment config")
    if not identity_manifest.is_authoritative():
        raise ValueError("atlas requires a loader-authoritative identity manifest")
    if config.atlas_consensus != 0.5:
        raise ValueError("normal atlas consensus must equal frozen value 0.5")
    categories = {item.record.category for item in references}
    item_ids = tuple(item.record.item_id for item in references)
    sample_ids = tuple(item.record.sample_id for item in references)
    if (
        len(categories) != 1
        or categories != {config.category}
        or any(item.record.role != "normal_fit" for item in references)
    ):
        raise ValueError("atlas references must be same-category normal_fit identities")
    expected = tuple(
        record
        for item_id in identity_manifest.atlas_normal_ids
        for record in identity_manifest.normal_fit
        if record.item_id == item_id
    )
    if (
        tuple(item.record for item in references) != expected
        or item_ids != identity_manifest.atlas_normal_ids
        or len(set(sample_ids)) != 16
    ):
        raise ValueError("atlas references must exactly match frozen manifest records")
    supports = np.stack([_estimate_support(item, config) for item in references])
    return NormalAtlas(
        _ATLAS_TOKEN,
        config.category,
        np.mean(supports, axis=0) >= 0.5,
        item_ids,
        sample_ids,
        identity_manifest.authority_digest,
    )

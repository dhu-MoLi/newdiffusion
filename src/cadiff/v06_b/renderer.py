from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class BackendOutput:
    rendered: np.ndarray
    q: np.ndarray


def _rgb(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 numpy array")
    if value.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8")
    return np.ascontiguousarray(value)


def _binary_mask(value: object, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must be an HxW numpy array matching RGB")
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return value.astype(bool, copy=True)


class CausalRenderer:
    """Return raw candidate ``r`` and ``m_final``; compose_causal produces final ``y``."""

    def __init__(
        self,
        *,
        backend: Callable[[np.ndarray, str, np.ndarray, int], BackendOutput],
        defect_types: tuple[str, ...],
        realization_threshold: float = 0.5,
    ) -> None:
        if not callable(backend):
            raise ValueError("backend must be callable")
        if (
            not defect_types
            or any(not isinstance(value, str) or not value for value in defect_types)
            or len(set(defect_types)) != len(defect_types)
        ):
            raise ValueError("defect_types must contain unique non-empty strings")
        if (
            isinstance(realization_threshold, bool)
            or not isinstance(realization_threshold, int | float)
            or not math.isfinite(float(realization_threshold))
            or not 0.0 < float(realization_threshold) < 1.0
        ):
            raise ValueError("realization_threshold must lie strictly between zero and one")
        self.backend = backend
        self.defect_types = defect_types
        self.realization_threshold = float(realization_threshold)

    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        proposal: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        normal_array = _rgb(normal, name="normal")
        proposal_array = _binary_mask(
            proposal, name="proposal", shape=normal_array.shape[:2]
        )
        if not proposal_array.any() or proposal_array.all():
            raise ValueError("proposal must contain both allowed and disallowed pixels")
        if defect_type not in self.defect_types:
            raise ValueError("defect_type is outside the configured vocabulary")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        output = self.backend(normal_array.copy(), defect_type, proposal_array.copy(), seed)
        if not isinstance(output, BackendOutput):
            raise ValueError("backend must return BackendOutput")
        rendered = _rgb(output.rendered, name="backend rendered").copy()
        if rendered.shape != normal_array.shape:
            raise ValueError("backend rendered shape must match normal")
        q = np.asarray(output.q)
        if (
            q.shape != proposal_array.shape
            or not np.issubdtype(q.dtype, np.floating)
            or not np.isfinite(q).all()
            or np.any((q < 0.0) | (q > 1.0))
        ):
            raise ValueError("backend q must be finite floating HxW in [0,1]")

        # Raw renderer output is proposal-confined; the final mask remains a head decision.
        rendered[~proposal_array] = normal_array[~proposal_array]
        final = proposal_array & (q >= self.realization_threshold)
        if (final & ~proposal_array).any():
            raise RuntimeError("realization mask escaped proposal")
        return np.ascontiguousarray(rendered), np.ascontiguousarray(final)


def compose_causal(
    normal: np.ndarray,
    rendered: np.ndarray,
    realization_mask: np.ndarray,
) -> np.ndarray:
    normal_array = _rgb(normal, name="normal")
    rendered_array = _rgb(rendered, name="rendered")
    if rendered_array.shape != normal_array.shape:
        raise ValueError("rendered shape must match normal")
    mask = _binary_mask(
        realization_mask,
        name="realization_mask",
        shape=normal_array.shape[:2],
    )
    composite = normal_array.copy()
    composite[mask] = rendered_array[mask]
    if not np.array_equal(composite[~mask], normal_array[~mask]):
        raise RuntimeError("causal composition changed RGB outside realization mask")
    if not np.array_equal(composite[mask], rendered_array[mask]):
        raise RuntimeError("causal composition did not copy raw RGB inside realization mask")
    return np.ascontiguousarray(composite)


__all__ = ["BackendOutput", "CausalRenderer", "compose_causal"]

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .exact_renderer import ExactMaskRenderer
from .planner import RealizationPlanner


def _binary_mask(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty HxW numpy array")
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return np.ascontiguousarray(value, dtype=bool)


def validate_realization(
    realization_mask: np.ndarray,
    proposal: np.ndarray,
) -> np.ndarray:
    proposal_array = _binary_mask(proposal, name="proposal")
    realization = _binary_mask(realization_mask, name="realization_mask")
    if realization.shape != proposal_array.shape:
        raise ValueError("realization_mask must match proposal geometry")
    if not realization.any():
        raise ValueError("realization mask is empty")
    if realization.all():
        raise ValueError("realization mask covers the full image")
    if (realization & ~proposal_array).any():
        raise ValueError("realization mask escaped proposal")
    return realization.copy()


class MaskFirstCausalRenderer:
    def __init__(
        self,
        *,
        planner: RealizationPlanner,
        renderer: ExactMaskRenderer,
        defect_types: tuple[str, ...],
    ) -> None:
        if not callable(planner):
            raise ValueError("planner must be callable")
        if not callable(renderer):
            raise ValueError("renderer must be callable")
        if (
            not isinstance(defect_types, tuple)
            or not defect_types
            or any(not isinstance(value, str) or not value for value in defect_types)
            or len(set(defect_types)) != len(defect_types)
        ):
            raise ValueError("defect_types must contain unique non-empty strings")
        self.planner: Callable[
            [np.ndarray, str, np.ndarray, int], tuple[np.ndarray, np.ndarray]
        ] = planner
        self.renderer: Callable[[np.ndarray, str, np.ndarray, int], np.ndarray] = renderer
        self.defect_types = defect_types

    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        proposal: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if defect_type not in self.defect_types:
            raise ValueError("defect_type is outside the configured vocabulary")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        _q, m_final = self.planner(normal, defect_type, proposal, seed)
        validate_realization(m_final, proposal)
        raw_r = self.renderer(normal, defect_type, m_final, seed)
        return raw_r, m_final


__all__ = ["MaskFirstCausalRenderer", "validate_realization"]

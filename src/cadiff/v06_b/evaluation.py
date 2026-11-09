from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .renderer import compose_causal


@dataclass(frozen=True)
class RealizationComparison:
    final_dice: float
    final_iou: float
    proposal_dice: float
    proposal_iou: float
    without_generation_dice: float
    without_generation_iou: float
    dice_gain_over_proposal: float
    dice_gain_over_without_generation: float
    change_alignment: float
    without_generation_change_alignment: float
    alignment_gain_over_without_generation: float
    q_ablation_l1: float
    final_area_fraction_of_proposal: float
    change_concentration_in_final: float
    outside_final_rgb_exact: bool
    generation_dependent: bool


@dataclass(frozen=True)
class SeedResponse:
    appearance_l1: float
    q_l1: float
    final_mask_iou: float


@dataclass(frozen=True)
class MaskFirstDiagnostic:
    diagnostic_only: bool
    m_final_dice: float
    m_final_iou: float
    proposal_dice: float
    proposal_iou: float
    dice_gain_over_proposal: float
    m_final_subset_of_proposal: bool
    outside_rgb_byte_exact: bool


def _rgb_pair(normal: object, rendered: object) -> tuple[np.ndarray, np.ndarray]:
    if (
        not isinstance(normal, np.ndarray)
        or not isinstance(rendered, np.ndarray)
        or normal.ndim != 3
        or normal.shape[2] != 3
        or rendered.shape != normal.shape
        or normal.dtype != np.uint8
        or rendered.dtype != np.uint8
    ):
        raise ValueError("normal and rendered must be matching uint8 HxWx3 arrays")
    return normal, rendered


def _mask(value: object, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or not np.isin(value, (0, 1)).all()
    ):
        raise ValueError(f"{name} must be a matching binary HxW array")
    return value.astype(bool, copy=True)


def _q(value: object, *, name: str, proposal: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != proposal.shape:
        raise ValueError(f"{name} must be a matching HxW array")
    if (
        not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or np.any((value < 0.0) | (value > 1.0))
        or np.any(value[~proposal] != 0.0)
    ):
        raise ValueError(f"{name} must be finite, in [0,1], and proposal-gated")
    return value.astype(np.float64, copy=False)


def _overlap(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(predicted & target))
    predicted_count = int(np.count_nonzero(predicted))
    target_count = int(np.count_nonzero(target))
    union = predicted_count + target_count - intersection
    dice_denominator = predicted_count + target_count
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def evaluate_mask_first_diagnostic(
    *,
    normal: np.ndarray,
    final_y: np.ndarray,
    transferred_ground_truth: np.ndarray,
    proposal: np.ndarray,
    m_final: np.ndarray,
) -> MaskFirstDiagnostic:
    normal, final_y = _rgb_pair(normal, final_y)
    shape = normal.shape[:2]
    transferred_ground_truth = _mask(
        transferred_ground_truth,
        name="transferred_ground_truth",
        shape=shape,
    )
    proposal = _mask(proposal, name="proposal", shape=shape)
    m_final = _mask(m_final, name="m_final", shape=shape)
    if not transferred_ground_truth.any() or not proposal.any() or proposal.all():
        raise ValueError("transferred_ground_truth and proposal must be non-trivial")
    if (transferred_ground_truth & ~proposal).any():
        raise ValueError("transferred_ground_truth must be a subset of proposal")
    final_dice, final_iou = _overlap(m_final, transferred_ground_truth)
    proposal_dice, proposal_iou = _overlap(proposal, transferred_ground_truth)
    return MaskFirstDiagnostic(
        diagnostic_only=True,
        m_final_dice=final_dice,
        m_final_iou=final_iou,
        proposal_dice=proposal_dice,
        proposal_iou=proposal_iou,
        dice_gain_over_proposal=final_dice - proposal_dice,
        m_final_subset_of_proposal=bool(np.all(~m_final | proposal)),
        outside_rgb_byte_exact=bool(np.array_equal(final_y[~m_final], normal[~m_final])),
    )


def evaluate_realization(
    *,
    normal: np.ndarray,
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    proposal: np.ndarray,
    q: np.ndarray,
    q_without_generation: np.ndarray,
    threshold: float,
) -> RealizationComparison:
    normal, rendered = _rgb_pair(normal, rendered)
    ground_truth = _mask(ground_truth, name="ground_truth", shape=normal.shape[:2])
    proposal = _mask(proposal, name="proposal", shape=normal.shape[:2])
    if not ground_truth.any() or not proposal.any() or proposal.all():
        raise ValueError("ground_truth and proposal must be non-trivial")
    if (ground_truth & ~proposal).any():
        raise ValueError("ground_truth must be a subset of proposal")
    if not np.array_equal(rendered[~proposal], normal[~proposal]):
        raise ValueError("rendered RGB must be proposal-confined before evaluation")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(float(threshold))
        or not 0.0 < float(threshold) < 1.0
    ):
        raise ValueError("threshold must lie strictly between zero and one")
    q = _q(q, name="q", proposal=proposal)
    q_without_generation = _q(
        q_without_generation,
        name="q_without_generation",
        proposal=proposal,
    )

    final = proposal & (q >= float(threshold))
    without_generation = proposal & (q_without_generation >= float(threshold))
    final_dice, final_iou = _overlap(final, ground_truth)
    proposal_dice, proposal_iou = _overlap(proposal, ground_truth)
    without_dice, without_iou = _overlap(without_generation, ground_truth)
    energy = np.abs(rendered.astype(np.float64) - normal.astype(np.float64)).mean(axis=2)
    energy /= 255.0
    active = proposal
    alignment = _cosine(q[active], energy[active])
    without_alignment = _cosine(q_without_generation[active], energy[active])
    q_ablation_l1 = float(np.mean(np.abs(q[active] - q_without_generation[active])))
    total_change = float(energy[active].sum())
    concentration = 0.0 if total_change <= 1e-12 else float(energy[final].sum() / total_change)
    final_fraction = float(final.sum() / proposal.sum())
    composite = compose_causal(normal, rendered, final)
    outside_exact = bool(np.array_equal(composite[~final], normal[~final]))
    dice_gain_without = final_dice - without_dice
    alignment_gain = alignment - without_alignment
    generation_dependent = bool(
        q_ablation_l1 >= 0.01 and (dice_gain_without >= 0.02 or alignment_gain >= 0.02)
    )
    return RealizationComparison(
        final_dice=final_dice,
        final_iou=final_iou,
        proposal_dice=proposal_dice,
        proposal_iou=proposal_iou,
        without_generation_dice=without_dice,
        without_generation_iou=without_iou,
        dice_gain_over_proposal=final_dice - proposal_dice,
        dice_gain_over_without_generation=dice_gain_without,
        change_alignment=alignment,
        without_generation_change_alignment=without_alignment,
        alignment_gain_over_without_generation=alignment_gain,
        q_ablation_l1=q_ablation_l1,
        final_area_fraction_of_proposal=final_fraction,
        change_concentration_in_final=concentration,
        outside_final_rgb_exact=outside_exact,
        generation_dependent=generation_dependent,
    )


def evaluate_seed_response(
    *,
    first_rendered: np.ndarray,
    second_rendered: np.ndarray,
    first_q: np.ndarray,
    second_q: np.ndarray,
    proposal: np.ndarray,
    threshold: float,
) -> SeedResponse:
    first_rendered, second_rendered = _rgb_pair(first_rendered, second_rendered)
    proposal = _mask(proposal, name="proposal", shape=first_rendered.shape[:2])
    if not proposal.any() or proposal.all():
        raise ValueError("proposal must be non-trivial")
    first_q = _q(first_q, name="first_q", proposal=proposal)
    second_q = _q(second_q, name="second_q", proposal=proposal)
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    appearance = (
        np.abs(first_rendered.astype(np.float64) - second_rendered.astype(np.float64)).mean(axis=2)
        / 255.0
    )
    first_final = proposal & (first_q >= float(threshold))
    second_final = proposal & (second_q >= float(threshold))
    _, final_iou = _overlap(first_final, second_final)
    return SeedResponse(
        appearance_l1=float(appearance[proposal].mean()),
        q_l1=float(np.abs(first_q[proposal] - second_q[proposal]).mean()),
        final_mask_iou=final_iou,
    )


__all__ = [
    "MaskFirstDiagnostic",
    "RealizationComparison",
    "SeedResponse",
    "evaluate_mask_first_diagnostic",
    "evaluate_realization",
    "evaluate_seed_response",
]

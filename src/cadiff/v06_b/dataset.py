from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from cadiff.data.geometry import resize_pad_image
from cadiff.p1_v1.dataset import TrainingItem, load_training_image
from cadiff.p2_a0.fewshot import FewshotSample

from .proposal import (
    coarsen_ground_truth,
    extract_training_support,
    transfer_training_proposal,
)


@dataclass(frozen=True, eq=False)
class GeometryPair:
    source_id: str
    target_id: str
    defect_type: str
    relationship: str
    target_pixels: Tensor
    target_valid: Tensor
    ground_truth: Tensor
    proposal: Tensor
    target_support: Tensor
    transfer_retained_fraction: float
    transfer_initial_retained_fraction: float
    transfer_recentered: bool
    transfer_row_shift: int
    transfer_column_shift: int


@dataclass(frozen=True, eq=False)
class SourceTrainingProposal:
    source_id: str
    defect_type: str
    relationship: str
    ground_truth: Tensor
    proposal: Tensor
    source_support: Tensor


@dataclass(frozen=True)
class PlannerTargetSplit:
    training: tuple[TrainingItem, ...]
    calibration: tuple[TrainingItem, ...]
    panel_ids: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class UnpairedSlot:
    step: int
    source: FewshotSample
    target: TrainingItem
    defect_type: str
    seed: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnpairedSlot):
            return NotImplemented
        return (
            self.step,
            self.source.sample_id,
            self.target.sample_id,
            self.defect_type,
            self.seed,
        ) == (
            other.step,
            other.source.sample_id,
            other.target.sample_id,
            other.defect_type,
            other.seed,
        )


def build_source_training_proposal(
    source: FewshotSample,
    *,
    image_size: int = 512,
    proposal_radius: int = 24,
    support_corner_width: int = 16,
) -> SourceTrainingProposal:
    """Coarsen authoritative GT on its own real-anomaly image for renderer training."""

    if not isinstance(source, FewshotSample) or source.role != "fewshot_train":
        raise ValueError("source must be a fewshot_train sample")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size <= 0
    ):
        raise ValueError("image_size must be a positive integer")
    if (
        not isinstance(source.mask, Tensor)
        or source.mask.shape != (1, image_size, image_size)
        or not torch.isfinite(source.mask).all().item()
        or not torch.all((source.mask == 0) | (source.mask == 1)).item()
    ):
        raise ValueError("source mask must be binary [1,H,W]")
    source_rgb = _normalized_rgb(
        source.pixels,
        name="source pixels",
        image_size=image_size,
    )
    with Image.open(source.image_path) as image:
        _, valid_image, _ = resize_pad_image(image, image_size)
    valid = np.asarray(valid_image, dtype=np.uint8).copy() != 0
    ground_truth = source.mask[0].detach().cpu().numpy().astype(bool, copy=True)
    if not ground_truth.any():
        raise ValueError("source ground truth must be non-empty")
    if (ground_truth & ~valid).any():
        raise ValueError("source ground truth must be inside valid image geometry")
    support = extract_training_support(
        source_rgb,
        valid,
        category="hazelnut",
        corner_width=support_corner_width,
    )
    support |= ground_truth
    proposal = coarsen_ground_truth(
        ground_truth,
        support,
        radius=proposal_radius,
    )

    def as_mask(value: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).bool().unsqueeze(0)

    return SourceTrainingProposal(
        source_id=source.sample_id,
        defect_type=source.defect_type,
        relationship="same_image_proposal_conditioning",
        ground_truth=as_mask(ground_truth),
        proposal=as_mask(proposal),
        source_support=as_mask(support),
    )


def _normalized_rgb(pixels: object, *, name: str, image_size: int) -> np.ndarray:
    if (
        not isinstance(pixels, Tensor)
        or pixels.shape != (3, image_size, image_size)
        or pixels.dtype is not torch.float32
        or not torch.isfinite(pixels).all().item()
        or torch.any((pixels < -1.0) | (pixels > 1.0)).item()
    ):
        raise ValueError(f"{name} must be finite float32 [3,{image_size},{image_size}] in [-1,1]")
    return (
        pixels.detach()
        .cpu()
        .permute(1, 2, 0)
        .add(1.0)
        .mul(0.5)
        .numpy()
        .copy()
    )


def build_geometry_pair(
    source: FewshotSample,
    target: TrainingItem,
    *,
    image_size: int = 512,
    proposal_radius: int = 24,
    support_corner_width: int = 16,
) -> GeometryPair:
    """Build one training-only, unpaired mask-geometry transfer."""

    if not isinstance(source, FewshotSample) or source.role != "fewshot_train":
        raise ValueError("source must be a fewshot_train sample")
    if (
        not isinstance(target, TrainingItem)
        or target.role != "normal_train"
        or target.defect_type != "good"
    ):
        raise ValueError("target must be a normal_train/good item")
    if source.image_path == target.image_path:
        raise ValueError("source and target images must be distinct")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if (
        not isinstance(source.mask, Tensor)
        or source.mask.shape != (1, image_size, image_size)
        or not torch.isfinite(source.mask).all().item()
        or not torch.all((source.mask == 0) | (source.mask == 1)).item()
    ):
        raise ValueError("source mask must be binary [1,H,W]")

    target_loaded = load_training_image(target, image_size)
    source_rgb = _normalized_rgb(source.pixels, name="source pixels", image_size=image_size)
    target_rgb = _normalized_rgb(
        target_loaded.pixels, name="target pixels", image_size=image_size
    )
    with Image.open(source.image_path) as image:
        _, source_valid_image, _ = resize_pad_image(image, image_size)
    source_valid = np.asarray(source_valid_image, dtype=np.uint8).copy() != 0
    target_valid = target_loaded.valid[0].cpu().numpy().copy()
    source_ground_truth = (
        source.mask[0].detach().cpu().numpy().astype(bool, copy=True)
    )
    if not source_ground_truth.any():
        raise ValueError("source ground truth must be non-empty")
    if (source_ground_truth & ~source_valid).any():
        raise ValueError("source ground truth must be inside valid source image geometry")

    category = target.category
    source_support = extract_training_support(
        source_rgb,
        source_valid,
        category=category,
        corner_width=support_corner_width,
    )
    # Few-shot GT is the source-side geometry authority. Dark cracks and holes can
    # disappear from the color-derived foreground, so support must include GT rather
    # than silently clipping the label used for transfer.
    source_support |= source_ground_truth
    target_support = extract_training_support(
        target_rgb,
        target_valid,
        category=category,
        corner_width=support_corner_width,
    )
    transferred = transfer_training_proposal(
        source_ground_truth,
        source_support,
        target_support,
        proposal_radius=proposal_radius,
    )
    if (
        transferred.proposal.all()
        or not np.all(transferred.ground_truth <= transferred.proposal)
        or not np.any(transferred.proposal & ~transferred.ground_truth)
    ):
        raise ValueError(
            "planner proposal must be a non-full strict superset of transferred GT"
        )

    def as_mask(value: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).bool().unsqueeze(0)

    return GeometryPair(
        source_id=source.sample_id,
        target_id=target.sample_id,
        defect_type=source.defect_type,
        relationship="unpaired_geometry_transfer",
        target_pixels=target_loaded.pixels.clone(),
        target_valid=target_loaded.valid.clone(),
        ground_truth=as_mask(transferred.ground_truth),
        proposal=as_mask(transferred.proposal),
        target_support=as_mask(target_support),
        transfer_retained_fraction=transferred.retained_fraction,
        transfer_initial_retained_fraction=(
            transferred.initial_retained_fraction
        ),
        transfer_recentered=transferred.recentered,
        transfer_row_shift=transferred.row_shift,
        transfer_column_shift=transferred.column_shift,
    )


def build_unpaired_schedule(
    *,
    sources: tuple[FewshotSample, ...],
    targets: tuple[TrainingItem, ...],
    defect_types: tuple[str, ...],
    steps: int,
    seed: int,
) -> tuple[UnpairedSlot, ...]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not defect_types or len(set(defect_types)) != len(defect_types):
        raise ValueError("defect_types must be non-empty and unique")
    if not targets or any(
        not isinstance(target, TrainingItem)
        or target.role != "normal_train"
        or target.defect_type != "good"
        for target in targets
    ):
        raise ValueError("targets must contain only normal_train/good items")
    groups = {
        defect_type: tuple(
            sorted(
                (
                    source
                    for source in sources
                    if source.defect_type == defect_type
                    and source.role == "fewshot_train"
                ),
                key=lambda source: source.image_relpath,
            )
        )
        for defect_type in defect_types
    }
    if any(not group for group in groups.values()):
        raise ValueError("every defect type must have a fewshot_train source")

    shuffled_targets = list(sorted(targets, key=lambda target: target.image_relpath))
    random.Random(seed).shuffle(shuffled_targets)
    offsets = {defect_type: 0 for defect_type in defect_types}
    slots: list[UnpairedSlot] = []
    for index in range(steps):
        defect_type = defect_types[index % len(defect_types)]
        group = groups[defect_type]
        source = group[offsets[defect_type] % len(group)]
        offsets[defect_type] += 1
        target = shuffled_targets[index % len(shuffled_targets)]
        if source.image_path == target.image_path:
            raise ValueError("unpaired schedule contains an identical source and target")
        slots.append(
            UnpairedSlot(
                step=index + 1,
                source=source,
                target=target,
                defect_type=defect_type,
                seed=seed + index,
            )
        )
    return tuple(slots)


def split_planner_targets(
    targets: tuple[TrainingItem, ...],
    *,
    panel_ids: tuple[str, ...],
    calibration_count: int,
    seed: int,
) -> PlannerTargetSplit:
    """Reserve deterministic calibration normals outside the fixed panel cohort."""

    if not isinstance(targets, tuple) or not targets or any(
        not isinstance(target, TrainingItem)
        or target.role != "normal_train"
        or target.defect_type != "good"
        for target in targets
    ):
        raise ValueError("targets must contain only normal_train/good items")
    if (
        not isinstance(panel_ids, tuple)
        or any(not isinstance(panel_id, str) or not panel_id for panel_id in panel_ids)
        or len(set(panel_ids)) != len(panel_ids)
    ):
        raise ValueError("panel_ids must be unique non-empty target identities")
    if (
        isinstance(calibration_count, bool)
        or not isinstance(calibration_count, int)
        or calibration_count <= 0
    ):
        raise ValueError("calibration_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    target_by_id = {target.sample_id: target for target in targets}
    if len(target_by_id) != len(targets):
        raise ValueError("targets must have unique sample identities")
    if any(panel_id not in target_by_id for panel_id in panel_ids):
        raise ValueError("panel_ids must identify supplied targets")
    panel_id_set = set(panel_ids)
    calibration_pool = sorted(
        (target for target in targets if target.sample_id not in panel_id_set),
        key=lambda target: target.image_relpath,
    )
    if calibration_count > len(calibration_pool):
        raise ValueError("calibration_count exceeds normals outside the fixed panel")
    random.Random(seed).shuffle(calibration_pool)
    calibration = tuple(calibration_pool[:calibration_count])
    calibration_ids = {target.sample_id for target in calibration}
    return PlannerTargetSplit(
        training=tuple(
            target for target in targets if target.sample_id not in calibration_ids
        ),
        calibration=calibration,
        panel_ids=panel_ids,
    )


__all__ = [
    "GeometryPair",
    "PlannerTargetSplit",
    "SourceTrainingProposal",
    "UnpairedSlot",
    "build_geometry_pair",
    "build_source_training_proposal",
    "build_unpaired_schedule",
    "split_planner_targets",
]

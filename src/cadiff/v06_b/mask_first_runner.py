from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from cadiff.config import ProjectConfig
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p1_v1.dataset import TrainingIndex, load_training_index
from cadiff.p2_a0.fewshot import (
    FewshotSample,
    FewshotSplit,
    PreparedSample,
    _prepare_samples,
    balanced_training_order,
    load_fewshot_split,
)
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_b2.config import P2B2Config, load_p2_b2_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_b.config import V06BMaskFirstConfig, is_cable_scheme2
from cadiff.v06_b.dataset import (
    GeometryPair,
    PlannerTargetSplit,
    UnpairedSlot,
    build_geometry_pair,
    build_unpaired_schedule,
    split_planner_targets,
)
from cadiff.v06_b.exact_renderer_training import (
    ExactMaskRendererBatch,
    build_exact_renderer_optimizer,
    exact_renderer_optimizer_step,
)
from cadiff.v06_b.mask_first_checkpoint import (
    load_mask_first_checkpoint,
    renderer_initialization_label,
    save_mask_first_checkpoint,
    type_condition_initialization_label,
    validate_mask_first_renderer_continue_configs,
)
from cadiff.v06_b.planner import PlannerNetwork
from cadiff.v06_b.planner_threshold import (
    ThresholdCalibration,
    ThresholdRecord,
    calibrate_realization_threshold,
)
from cadiff.v06_b.planner_training import (
    PlannerBatch,
    build_planner_optimizer,
    planner_optimizer_step,
)
from cadiff.v06_b.prompts import V06BTypePrompts, load_v06_b_type_prompts
from cadiff.v06_b.source import (
    FreshP2B2Renderer,
    SourceRendererManifest,
    bind_fresh_type_vocabulary,
    load_fresh_p2_b2_renderer,
    load_source_renderer_manifest,
)

_P2_B2_RENDERER_ID = "p2_b2_hazelnut_type_prompt_attempt1"
_P2_B2_CHECKPOINT_RELPATH = Path(
    "p2_b2/hazelnut/type-prompt-training/seed-20260817-attempt1/checkpoint"
)
_CABLE_P2_B2_RENDERER_ID = "p2_b2_cable_type_prompt_scheme2"
_CABLE_P2_B2_CHECKPOINT_RELPATH = Path(
    "p2_b2/cable/type-prompt-training/seed-20260817/checkpoint"
)
_CALIBRATION_POLICY = "fewshot_train_geometry_support_macro_dice_v1"
_PLANNER_LOSS_FIELDS = ("total", "bce", "dice", "grad_l2")
_RENDERER_LOSS_FIELDS = (
    "total",
    "epsilon_gt",
    "x0_gt",
    "rgb_gt",
    "edge_gt",
    "type_margin",
    "wrong_semantic",
    "grad_l2",
)
_PLANNER_METRIC_FIELDS = frozenset(
    {
        "diagnostic_only",
        "step",
        "source_id",
        "target_id",
        "defect_type",
        "seed",
        "relationship",
        *_PLANNER_LOSS_FIELDS,
        "finite",
    }
)
_RENDERER_METRIC_FIELDS = frozenset(
    {
        "diagnostic_only",
        "exact_ground_truth",
        "relationship",
        "step",
        "sample_id",
        "defect_type",
        "wrong_defect_type",
        "seed",
        *_RENDERER_LOSS_FIELDS,
        "finite",
    }
)
_ARCHITECTURE_CATEGORY = "hazelnut"
_HAZELNUT_ARCHITECTURE_TYPES = ("crack", "cut", "hole", "print")


def load_mask_first_source_prompts(path: Path, *, category: str) -> P2B2Config | V06BTypePrompts:
    if category == "cable":
        return load_v06_b_type_prompts(path)
    return load_p2_b2_config(path)


def mask_first_data_authority_model(
    config: V06BMaskFirstConfig, source_model: P2A1Config
) -> P2A1Config:
    if config.category == "hazelnut":
        return source_model
    if config.category != "cable":
        raise ValueError("mask-first category must be hazelnut or cable")
    if is_cable_scheme2(config):
        if source_model.category != "cable" or source_model.defect_types != config.defect_types:
            raise ValueError("Cable scheme 2 data authority must be Cable P2-A1")
        return source_model
    return replace(
        source_model,
        category="cable",
        defect_types=config.defect_types,
        normal_prompt="a photo of a normal cable",
        capacity_defect_type=config.defect_types[0],
    )


def validate_mask_first_source_binding(
    config: V06BMaskFirstConfig,
    source_model: P2A1Config,
    source_prompts: P2B2Config | V06BTypePrompts,
) -> None:
    if source_model.type_rank != source_prompts.type_rank:
        raise ValueError("source model and exact P2-B2 prompt rank must match")
    if config.category == "hazelnut":
        if not isinstance(source_prompts, P2B2Config):
            raise ValueError("source_prompts must be a P2B2Config")
        if (
            source_model.category != "hazelnut"
            or source_prompts.category != "hazelnut"
            or source_model.defect_types != config.defect_types
            or source_prompts.defect_types != config.defect_types
        ):
            raise ValueError("source category and ordered vocabulary must match Module B")
        return
    if config.category != "cable":
        raise ValueError("mask-first category must be hazelnut or cable")
    if not isinstance(source_prompts, V06BTypePrompts):
        raise ValueError("Cable source_prompts must be V06BTypePrompts")
    if source_prompts.category != "cable" or source_prompts.defect_types != config.defect_types:
        raise ValueError("Cable B prompts must match the Cable vocabulary")
    if is_cable_scheme2(config):
        if source_model.category != "cable" or source_model.defect_types != config.defect_types:
            raise ValueError("Cable scheme 2 architecture init must be Cable P2-A1")
        return
    if (
        source_model.category != "hazelnut"
        or source_model.defect_types != _HAZELNUT_ARCHITECTURE_TYPES
    ):
        raise ValueError("Cable B architecture init must be frozen Hazelnut P2-A1")


def bind_mask_first_renderer(
    loaded: FreshP2B2Renderer,
    *,
    config: V06BMaskFirstConfig,
    unet: torch.nn.Module,
    device: torch.device,
) -> FreshP2B2Renderer:
    if config.category != "cable":
        return loaded
    if is_cable_scheme2(config):
        if loaded.vocabulary.category != "cable" or loaded.vocabulary.labels != config.defect_types:
            raise ValueError("Cable scheme 2 renderer must already use the Cable vocabulary")
        return loaded
    return bind_fresh_type_vocabulary(
        loaded,
        category="cable",
        defect_types=config.defect_types,
        unet=unet,
        device=device,
    )


@dataclass(frozen=True)
class MaskFirstV06BContext:
    paths: ProjectConfig
    experiment: V06BMaskFirstConfig
    source_model: P2A1Config
    source_prompts: P2B2Config | V06BTypePrompts
    manifest: Path
    output: Path
    cwd: Path


@dataclass(frozen=True)
class PlannerGeometryCertification:
    step: int
    source_id: str
    target_id: str
    defect_type: str
    seed: int
    relationship: str
    pair_sha256: str


@dataclass(frozen=True)
class MaskFirstPreflight:
    context: MaskFirstV06BContext
    source_renderer: SourceRendererManifest
    split: FewshotSplit
    training_index: TrainingIndex
    planner_targets: PlannerTargetSplit
    planner_schedule: tuple[UnpairedSlot, ...]
    planner_geometry_certifications: tuple[PlannerGeometryCertification, ...]
    calibration_pairs: tuple[GeometryPair, ...]
    renderer_schedule: tuple[FewshotSample, ...]


def validate_mask_first_preflight(
    context: MaskFirstV06BContext,
    *,
    skip_planner_geometry: bool = False,
) -> MaskFirstPreflight:
    if not isinstance(context, MaskFirstV06BContext):
        raise ValueError("context must be a MaskFirstV06BContext")
    if not isinstance(context.paths, ProjectConfig):
        raise ValueError("paths must be a ProjectConfig")
    if (
        not isinstance(context.experiment, V06BMaskFirstConfig)
        or context.experiment.schema_version != "v06_b_mask_first_v1"
    ):
        raise ValueError("experiment must be the mask-first Module-B schema")
    if not isinstance(context.source_model, P2A1Config):
        raise ValueError("source_model must be a P2A1Config")
    if not isinstance(context.source_prompts, P2B2Config | V06BTypePrompts):
        raise ValueError("source_prompts must be P2B2Config or V06BTypePrompts")
    if not isinstance(context.cwd, Path) or not context.cwd.is_dir():
        raise ValueError("cwd must be an existing directory")
    if not isinstance(context.manifest, Path) or not isinstance(context.output, Path):
        raise ValueError("manifest and output must be pathlib.Path values")

    config = context.experiment
    root = context.paths.runs_root.resolve(strict=True)
    expected_output = (root / config.output_relpath).resolve(strict=False)
    try:
        expected_output.relative_to(root)
    except ValueError as error:
        raise ValueError("output_relpath escapes runs_root") from error
    if context.output.resolve(strict=False) != expected_output:
        raise ValueError("output must exactly match runs_root/output_relpath")
    if context.output.is_symlink() or context.output.exists():
        raise FileExistsError("mask-first output must not already exist")

    model_config_path = context.cwd / config.source_model_config
    prompt_config_path = context.cwd / config.source_experiment_config
    for path, name in (
        (model_config_path, "canonical source model config"),
        (prompt_config_path, "canonical source prompt config"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} must be a regular file")
    if load_p2_a1_config(model_config_path) != context.source_model:
        raise ValueError("supplied source model differs from canonical source model config")
    if (
        load_mask_first_source_prompts(prompt_config_path, category=config.category)
        != context.source_prompts
    ):
        raise ValueError("supplied source prompts differ from canonical source prompt config")

    if (
        context.paths.image_size != config.image_size
        or context.source_model.image_size != config.image_size
    ):
        raise ValueError("project, source model, and Module-B image_size must match")
    validate_mask_first_source_binding(config, context.source_model, context.source_prompts)

    architecture_category = "cable" if is_cable_scheme2(config) else _ARCHITECTURE_CATEGORY
    source_renderer = load_source_renderer_manifest(
        context.cwd / config.source_renderer_manifest,
        runs_root=root,
        category=architecture_category,
    )
    if is_cable_scheme2(config):
        expected_checkpoint = (root / _CABLE_P2_B2_CHECKPOINT_RELPATH).resolve(strict=False)
        if (
            source_renderer.renderer_id != _CABLE_P2_B2_RENDERER_ID
            or source_renderer.category != "cable"
            or source_renderer.checkpoint.resolve(strict=False) != expected_checkpoint
        ):
            raise ValueError("source renderer must be the Cable P2-B2 scheme-2 checkpoint")
    else:
        expected_checkpoint = (root / _P2_B2_CHECKPOINT_RELPATH).resolve(strict=False)
        if (
            source_renderer.renderer_id != _P2_B2_RENDERER_ID
            or source_renderer.category != _ARCHITECTURE_CATEGORY
            or source_renderer.checkpoint.resolve(strict=False) != expected_checkpoint
        ):
            raise ValueError("source renderer must be the frozen P2-B2 attempt1")

    data_model = mask_first_data_authority_model(config, context.source_model)
    split = load_fewshot_split(
        context.manifest,
        context.paths.dataset_root,
        data_model,
    )
    training_index = load_training_index(
        context.manifest,
        context.paths.dataset_root,
        data_model,
        ignore_unknown_fewshot_types=config.category == "cable",
    )
    train_ids = {sample.sample_id for sample in split.train}
    evaluation_ids = {sample.sample_id for sample in split.evaluation}
    if (
        not split.train
        or train_ids & evaluation_ids
        or len(train_ids) != len(split.train)
        or any(
            sample.role != "fewshot_train" or sample.defect_type not in config.defect_types
            for sample in split.train
        )
    ):
        raise ValueError("few-shot training authority must be unique and train-only")
    if training_index.category != config.category:
        raise ValueError("training index category differs from Module B")

    panel_ids = _fixed_panel_target_ids(
        training_index.normals,
        normal_count=config.panel_normal_count,
    )
    planner_targets = split_planner_targets(
        training_index.normals,
        panel_ids=panel_ids,
        calibration_count=config.planner_calibration_normal_count,
        seed=config.seed,
    )
    planner_schedule = build_unpaired_schedule(
        sources=split.train,
        targets=planner_targets.training,
        defect_types=config.defect_types,
        steps=config.planner_training_steps,
        seed=config.seed,
    )
    _validate_planner_schedule(planner_schedule, train_ids=train_ids, config=config)
    if skip_planner_geometry:
        planner_geometry_certifications = ()
        calibration_pairs = ()
    else:
        planner_geometry_certifications = tuple(
            _certify_planner_geometry(
                slot,
                build_geometry_pair(
                    slot.source,
                    slot.target,
                    image_size=config.image_size,
                    proposal_radius=config.proposal_radius,
                    support_corner_width=config.support_corner_width,
                ),
                image_size=config.image_size,
            )
            for slot in planner_schedule
        )
        _validate_planner_geometry_certifications(
            planner_schedule,
            planner_geometry_certifications,
        )
        calibration_schedule = build_unpaired_schedule(
            sources=split.train,
            targets=planner_targets.calibration,
            defect_types=config.defect_types,
            steps=max(len(planner_targets.calibration), len(config.defect_types)),
            seed=config.seed + 100_000_000,
        )
        calibration_pairs = tuple(
            build_geometry_pair(
                slot.source,
                slot.target,
                image_size=config.image_size,
                proposal_radius=config.proposal_radius,
                support_corner_width=config.support_corner_width,
            )
            for slot in calibration_schedule
        )
        calibration_ids = {target.sample_id for target in planner_targets.calibration}
        if (
            {pair.target_id for pair in calibration_pairs} != calibration_ids
            or {pair.defect_type for pair in calibration_pairs} != set(config.defect_types)
            or any(pair.source_id not in train_ids for pair in calibration_pairs)
        ):
            raise RuntimeError("calibration pairs do not cover exact support authority")
    renderer_schedule = balanced_training_order(
        split.train,
        config.renderer_training_steps,
        config.defect_types,
    )
    if any(
        sample.role != "fewshot_train"
        or sample.sample_id not in train_ids
        or sample.defect_type != config.defect_types[index % len(config.defect_types)]
        for index, sample in enumerate(renderer_schedule)
    ):
        raise RuntimeError("renderer schedule must be balanced cyclic fewshot_train only")
    return MaskFirstPreflight(
        context=context,
        source_renderer=source_renderer,
        split=split,
        training_index=training_index,
        planner_targets=planner_targets,
        planner_schedule=planner_schedule,
        planner_geometry_certifications=planner_geometry_certifications,
        calibration_pairs=calibration_pairs,
        renderer_schedule=renderer_schedule,
    )


def _fixed_panel_target_ids(
    targets: tuple[Any, ...],
    *,
    normal_count: int,
) -> tuple[str, ...]:
    ordered = tuple(sorted(targets, key=lambda target: target.image_relpath))
    if normal_count <= 0 or normal_count > len(ordered):
        raise ValueError("panel normal_count must fit the normal_train pool")
    indices = (
        (0,)
        if normal_count == 1
        else tuple(
            round(index * (len(ordered) - 1) / (normal_count - 1)) for index in range(normal_count)
        )
    )
    return tuple(ordered[index].sample_id for index in indices)


def _validate_planner_schedule(
    schedule: tuple[UnpairedSlot, ...],
    *,
    train_ids: set[str],
    config: V06BMaskFirstConfig,
) -> None:
    if len(schedule) != config.planner_training_steps or any(
        slot.source.sample_id not in train_ids
        or slot.source.role != "fewshot_train"
        or slot.defect_type != config.defect_types[index % len(config.defect_types)]
        for index, slot in enumerate(schedule)
    ):
        raise RuntimeError("planner schedule must be balanced cyclic fewshot_train only")


def _validate_planner_geometry_pair(
    slot: UnpairedSlot,
    pair: GeometryPair,
    *,
    image_size: int,
    proposal_must_subset_support: bool = True,
) -> None:
    if not isinstance(slot, UnpairedSlot) or not isinstance(pair, GeometryPair):
        raise ValueError("planner geometry certification requires a slot and pair")
    if (
        pair.source_id != slot.source.sample_id
        or pair.target_id != slot.target.sample_id
        or pair.defect_type != slot.defect_type
        or slot.source.defect_type != slot.defect_type
        or pair.relationship != "unpaired_geometry_transfer"
    ):
        raise RuntimeError("planner geometry pair differs from its scheduled authority")
    if (
        not isinstance(pair.target_pixels, Tensor)
        or pair.target_pixels.shape != (3, image_size, image_size)
        or pair.target_pixels.dtype is not torch.float32
        or pair.target_pixels.requires_grad
        or not torch.isfinite(pair.target_pixels).all().item()
        or torch.any((pair.target_pixels < -1.0) | (pair.target_pixels > 1.0)).item()
    ):
        raise ValueError("planner target pixels must be finite float32 [3,H,W] in [-1,1]")
    masks = (
        pair.target_valid,
        pair.ground_truth,
        pair.proposal,
        pair.target_support,
    )
    if any(
        not isinstance(mask, Tensor)
        or mask.shape != (1, image_size, image_size)
        or mask.dtype is not torch.bool
        or mask.requires_grad
        for mask in masks
    ):
        raise ValueError("planner geometry masks must be detached bool [1,H,W]")
    if (
        not pair.ground_truth.any().item()
        or pair.proposal.all().item()
        or not torch.all(~pair.ground_truth | pair.proposal).item()
        or not torch.any(pair.proposal & ~pair.ground_truth).item()
    ):
        raise ValueError("planner proposal must be a non-full strict superset of transferred GT")
    if proposal_must_subset_support:
        proposal_domain = pair.target_support
        domain_error = "planner proposal/support must stay inside target validity"
    else:
        proposal_domain = pair.target_valid
        domain_error = "planner proposal must stay inside target validity"
    if (
        torch.any(pair.proposal & ~proposal_domain).item()
        or torch.any(pair.target_support & ~pair.target_valid).item()
    ):
        raise ValueError(domain_error)
    fractions = (
        pair.transfer_retained_fraction,
        pair.transfer_initial_retained_fraction,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in fractions
    ) or not (
        0.0 < float(pair.transfer_retained_fraction) <= 1.0
        and 0.0 <= float(pair.transfer_initial_retained_fraction) <= 1.0
    ):
        raise ValueError("planner transfer retention metadata must be finite fractions")
    if (
        type(pair.transfer_recentered) is not bool
        or type(pair.transfer_row_shift) is not int
        or type(pair.transfer_column_shift) is not int
        or pair.transfer_recentered != bool(pair.transfer_row_shift or pair.transfer_column_shift)
    ):
        raise ValueError("planner transfer recenter metadata is inconsistent")


def _update_geometry_digest(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(4, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _certify_planner_geometry(
    slot: UnpairedSlot,
    pair: GeometryPair,
    *,
    image_size: int,
    proposal_must_subset_support: bool = True,
) -> PlannerGeometryCertification:
    _validate_planner_geometry_pair(
        slot,
        pair,
        image_size=image_size,
        proposal_must_subset_support=proposal_must_subset_support,
    )
    digest = hashlib.sha256()
    scalar_payload = json.dumps(
        {
            "schema_version": "v06_b_planner_geometry_certification_v1",
            "step": slot.step,
            "source_id": slot.source.sample_id,
            "target_id": slot.target.sample_id,
            "defect_type": slot.defect_type,
            "seed": slot.seed,
            "relationship": pair.relationship,
            "transfer_retained_fraction": pair.transfer_retained_fraction,
            "transfer_initial_retained_fraction": (pair.transfer_initial_retained_fraction),
            "transfer_recentered": pair.transfer_recentered,
            "transfer_row_shift": pair.transfer_row_shift,
            "transfer_column_shift": pair.transfer_column_shift,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    _update_geometry_digest(digest, "scalar_metadata", scalar_payload)
    for name, tensor in (
        ("target_pixels", pair.target_pixels),
        ("target_valid", pair.target_valid),
        ("ground_truth", pair.ground_truth),
        ("proposal", pair.proposal),
        ("target_support", pair.target_support),
    ):
        array = tensor.detach().cpu().contiguous().numpy()
        canonical = array.astype(array.dtype.newbyteorder("<"), copy=False)
        tensor_header = json.dumps(
            {
                "dtype": canonical.dtype.str,
                "shape": list(canonical.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        _update_geometry_digest(digest, f"{name}:header", tensor_header)
        _update_geometry_digest(
            digest,
            f"{name}:bytes",
            canonical.tobytes(order="C"),
        )
    return PlannerGeometryCertification(
        step=slot.step,
        source_id=slot.source.sample_id,
        target_id=slot.target.sample_id,
        defect_type=slot.defect_type,
        seed=slot.seed,
        relationship=pair.relationship,
        pair_sha256=digest.hexdigest(),
    )


def _validate_planner_geometry_certifications(
    schedule: tuple[UnpairedSlot, ...],
    certifications: tuple[PlannerGeometryCertification, ...],
) -> None:
    if len(certifications) != len(schedule):
        raise RuntimeError("planner geometry certification count differs from schedule")
    for slot, certification in zip(schedule, certifications, strict=True):
        if not isinstance(certification, PlannerGeometryCertification) or (
            certification.step,
            certification.source_id,
            certification.target_id,
            certification.defect_type,
            certification.seed,
            certification.relationship,
        ) != (
            slot.step,
            slot.source.sample_id,
            slot.target.sample_id,
            slot.defect_type,
            slot.seed,
            "unpaired_geometry_transfer",
        ):
            raise RuntimeError("planner geometry certification order differs from schedule")
        if len(certification.pair_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in certification.pair_sha256
        ):
            raise RuntimeError("planner geometry certification digest is invalid")


def _append_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _metrics_summary(
    records: list[dict[str, object]],
    names: tuple[str, ...],
) -> dict[str, object]:
    if not records:
        raise RuntimeError("training metrics must be non-empty")
    for record in records:
        if record.get("finite") is not True:
            raise RuntimeError("training metric records must declare finite true")
        for name in names:
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RuntimeError("training metric values must be non-boolean numeric")
            try:
                finite = math.isfinite(value)
            except OverflowError as error:
                raise RuntimeError("training metric values must be finite") from error
            if not finite:
                raise RuntimeError("training metric values must be finite")
        if "grad_l2" in names and float(record["grad_l2"]) <= 0.0:
            raise RuntimeError("training gradient norm must be positive")
    means = {name: sum(float(record[name]) for record in records) / len(records) for name in names}
    if not all(math.isfinite(value) for value in means.values()):
        raise RuntimeError("training metric summary must be finite")
    window = min(100, len(records))
    sampled_stride = max(1, math.ceil(len(records) / 50))
    curve_names = tuple(name for name in names if name != "grad_l2")
    sampled_curve: list[dict[str, object]] = []
    seen_steps: set[object] = set()
    for record in (*records[::sampled_stride], records[-1]):
        step = record["step"]
        if step in seen_steps:
            continue
        seen_steps.add(step)
        sampled_curve.append({"step": step, **{name: record[name] for name in curve_names}})
    return {
        "diagnostic_only": True,
        "steps": len(records),
        "mean_losses": means,
        "final_losses": {name: records[-1][name] for name in names},
        "first_losses": {name: records[0][name] for name in names},
        "early_window_mean": {
            name: sum(float(record[name]) for record in records[:window]) / window for name in names
        },
        "late_window_mean": {
            name: sum(float(record[name]) for record in records[-window:]) / window
            for name in names
        },
        "sampled_curve": sampled_curve,
    }


def _run_planner_training_and_calibration(
    preflight: MaskFirstPreflight,
    *,
    device: torch.device,
    metrics_path: Path,
    pair_builder: Callable[[UnpairedSlot], GeometryPair] | None = None,
    require_geometry_certifications: bool = True,
    proposal_must_subset_support: bool = True,
) -> tuple[PlannerNetwork, ThresholdCalibration, dict[str, object]]:
    config = preflight.context.experiment
    if require_geometry_certifications:
        _validate_planner_geometry_certifications(
            preflight.planner_schedule,
            preflight.planner_geometry_certifications,
        )
    planner = PlannerNetwork(
        defect_types=config.defect_types,
        base_channels=config.planner_base_channels,
        type_embedding_dim=config.planner_type_embedding_dim,
    ).to(device=device, dtype=torch.float32)
    planner.train()
    optimizer = build_planner_optimizer(network=planner, config=config)
    records: list[dict[str, object]] = []
    for index, slot in enumerate(preflight.planner_schedule):
        if pair_builder is None:
            pair = build_geometry_pair(
                slot.source,
                slot.target,
                image_size=config.image_size,
                proposal_radius=config.proposal_radius,
                support_corner_width=config.support_corner_width,
            )
        else:
            pair = pair_builder(slot)
        if require_geometry_certifications:
            expected_certification = preflight.planner_geometry_certifications[index]
            actual_certification = _certify_planner_geometry(
                slot,
                pair,
                image_size=config.image_size,
                proposal_must_subset_support=proposal_must_subset_support,
            )
            if actual_certification != expected_certification:
                raise RuntimeError("planner geometry differs from its preflight certification")
        else:
            _validate_planner_geometry_pair(
                slot,
                pair,
                image_size=config.image_size,
                proposal_must_subset_support=proposal_must_subset_support,
            )
        metrics = planner_optimizer_step(
            network=planner,
            optimizer=optimizer,
            batch=PlannerBatch(
                normal_rgb=pair.target_pixels.unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                ),
                proposal=pair.proposal.unsqueeze(0).to(device=device),
                target=pair.ground_truth.unsqueeze(0).to(device=device),
                type_indices=torch.tensor(
                    [config.defect_types.index(slot.defect_type)],
                    dtype=torch.long,
                    device=device,
                ),
            ),
            config=config,
        )
        record: dict[str, object] = {
            "diagnostic_only": True,
            "step": slot.step,
            "source_id": slot.source.sample_id,
            "target_id": slot.target.sample_id,
            "defect_type": slot.defect_type,
            "seed": slot.seed,
            "relationship": "unpaired_geometry_transfer",
            "total": metrics.total,
            "bce": metrics.bce,
            "dice": metrics.dice,
            "grad_l2": metrics.grad_l2,
            "finite": metrics.finite,
        }
        _append_json(metrics_path, record)
        records.append(record)
    del optimizer

    calibration_records: list[ThresholdRecord] = []
    was_training = planner.training
    planner.eval()
    try:
        with torch.no_grad():
            for pair in preflight.calibration_pairs:
                proposal = pair.proposal.unsqueeze(0).to(device=device)
                prediction = planner(
                    pair.target_pixels.unsqueeze(0).to(
                        device=device,
                        dtype=torch.float32,
                    ),
                    proposal,
                    torch.tensor(
                        [config.defect_types.index(pair.defect_type)],
                        dtype=torch.long,
                        device=device,
                    ),
                )
                support_key = (f"{pair.source_id}\n{pair.target_id}\n{pair.defect_type}").encode()
                calibration_records.append(
                    ThresholdRecord(
                        record_id=(
                            "fewshot-train-geometry-support-v1:"
                            + hashlib.sha256(support_key).hexdigest()
                        ),
                        role="fewshot_train_geometry_support",
                        defect_type=pair.defect_type,
                        source_id=pair.source_id,
                        target_id=pair.target_id,
                        q=np.ascontiguousarray(prediction.q[0, 0].float().cpu().numpy()),
                        target=np.ascontiguousarray(pair.ground_truth[0].cpu().numpy(), dtype=bool),
                        proposal=np.ascontiguousarray(pair.proposal[0].cpu().numpy(), dtype=bool),
                    )
                )
    finally:
        planner.train(was_training)
    calibration_target_ids = tuple(
        target.sample_id for target in preflight.planner_targets.calibration
    )
    calibration = calibrate_realization_threshold(
        tuple(calibration_records),
        candidates=config.planner_threshold_candidates,
        calibration_target_ids=calibration_target_ids,
        defect_types=config.defect_types,
    )
    return (
        planner,
        calibration,
        _metrics_summary(records, _PLANNER_LOSS_FIELDS),
    )


def _ordinary_prompt(value: Tensor) -> Tensor:
    prompt = value.detach().clone()
    if (
        prompt.is_inference()
        or prompt.requires_grad
        or not prompt.is_floating_point()
        or not torch.isfinite(prompt).all().item()
    ):
        raise ValueError("P2-B2 prompt embedding must be ordinary, detached, and finite")
    return prompt


def _renderer_batch(
    *,
    prepared: PreparedSample,
    sample: FewshotSample,
    prompts: dict[str, Tensor],
    scheduler: Any,
    config: V06BMaskFirstConfig,
    device: torch.device,
    step: int,
) -> ExactMaskRendererBatch:
    if prepared.sample is not sample and prepared.sample.sample_id != sample.sample_id:
        raise RuntimeError("prepared renderer sample differs from schedule")
    image_mask = sample.mask.unsqueeze(0).to(device=device, dtype=torch.float32)
    masks = build_latent_masks(image_mask, tuple(prepared.target.shape[2:]))
    if not torch.equal(masks.binary, prepared.masks.binary) or not torch.equal(
        masks.soft, prepared.masks.soft
    ):
        raise RuntimeError("prepared renderer masks differ from original exact GT")
    context_seed = config.seed + 70_000_000 + step
    context = build_masked_context(
        prepared.target,
        masks.binary,
        torch.Generator(device=device).manual_seed(context_seed),
    )
    try:
        timestep_count = int(scheduler.config.num_train_timesteps)
    except AttributeError as error:
        raise ValueError("scheduler.config.num_train_timesteps is required") from error
    if timestep_count <= 0:
        raise ValueError("scheduler timestep count must be positive")
    generator = torch.Generator(device=device).manual_seed(config.seed + 80_000_000 + step)
    timesteps = torch.randint(
        0,
        timestep_count,
        (prepared.target.shape[0],),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn(
        prepared.target.shape,
        generator=generator,
        device=device,
        dtype=prepared.target.dtype,
    )
    defect_type = sample.defect_type
    wrong_type = config.defect_types[
        (config.defect_types.index(defect_type) + 1) % len(config.defect_types)
    ]
    target_image = (
        sample.pixels.unsqueeze(0).to(device=device, dtype=torch.float32).add(1.0).mul(0.5)
    )
    return ExactMaskRendererBatch(
        target_latent=prepared.target,
        target_image=target_image,
        context=context,
        binary_gt_mask=masks.binary,
        soft_gt_mask=masks.soft,
        image_gt_mask=image_mask,
        prompt_embeds=prompts[defect_type],
        wrong_prompt_embeds=prompts[wrong_type],
        defect_type=defect_type,
        wrong_defect_type=wrong_type,
        timesteps=timesteps,
        noise=noise,
    )


def _run_renderer_training(
    preflight: MaskFirstPreflight,
    *,
    device: torch.device,
    metrics_path: Path,
    parent_checkpoint: Path | None = None,
    parent_config: V06BMaskFirstConfig | None = None,
    planner: PlannerNetwork | None = None,
) -> tuple[Any, dict[str, object]]:
    context = preflight.context
    config = context.experiment
    components = load_frozen_components(
        context.paths.model_root,
        device,
        torch.float16,
    )
    prompts = {
        defect_type: _ordinary_prompt(
            encode_prompt(
                components,
                context.source_prompts.prompts[defect_type],
                device,
            )
        )
        for defect_type in config.defect_types
    }
    fresh = load_fresh_p2_b2_renderer(
        manifest_path=context.cwd / config.source_renderer_manifest,
        runs_root=context.paths.runs_root,
        category="cable" if is_cable_scheme2(config) else _ARCHITECTURE_CATEGORY,
        source_config=context.source_model,
        unet=components.unet,
        device=device,
    )
    fresh = bind_mask_first_renderer(
        fresh,
        config=config,
        unet=components.unet,
        device=device,
    )
    if fresh.source != preflight.source_renderer:
        raise RuntimeError("fresh P2-B2 reload differs from authenticated preflight source")
    if parent_checkpoint is not None:
        if parent_config is None or planner is None:
            raise ValueError("renderer continue requires parent config and frozen planner")
        load_mask_first_checkpoint(
            parent_checkpoint,
            planner=planner,
            adapter=fresh.adapter,
            conditioner=fresh.conditioner,
            config=parent_config,
            source=fresh.source,
        )
        planner.eval()
        for parameter in planner.parameters():
            parameter.requires_grad_(False)
    optimizer = build_exact_renderer_optimizer(
        model=fresh.model,
        vae=components.vae,
        text_encoder=components.text_encoder,
        config=config,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    prepared_samples = _prepare_samples(
        vae=components.vae,
        samples=preflight.split.train,
        device=device,
        config=mask_first_data_authority_model(config, context.source_model),
        seed_offset=60_000_000,
    )
    prepared_by_id = {prepared.sample.sample_id: prepared for prepared in prepared_samples}
    if set(prepared_by_id) != {sample.sample_id for sample in preflight.split.train}:
        raise RuntimeError("prepared renderer samples differ from fewshot_train authority")
    records: list[dict[str, object]] = []
    for step, sample in enumerate(preflight.renderer_schedule, start=1):
        if (
            sample.role != "fewshot_train"
            or sample.defect_type != config.defect_types[(step - 1) % len(config.defect_types)]
        ):
            raise RuntimeError("renderer step is outside balanced train-only schedule")
        batch = _renderer_batch(
            prepared=prepared_by_id[sample.sample_id],
            sample=sample,
            prompts=prompts,
            scheduler=components.scheduler,
            config=config,
            device=device,
            step=step,
        )
        metrics = exact_renderer_optimizer_step(
            model=fresh.model,
            vae=components.vae,
            scheduler=components.scheduler,
            optimizer=optimizer,
            scaler=scaler,
            batch=batch,
            config=config,
        )
        record: dict[str, object] = {
            "diagnostic_only": True,
            "exact_ground_truth": True,
            "relationship": "real_anomaly_exact_ground_truth",
            "step": step,
            "sample_id": sample.sample_id,
            "defect_type": sample.defect_type,
            "wrong_defect_type": batch.wrong_defect_type,
            "seed": config.seed + 80_000_000 + step,
            "total": metrics.total,
            "epsilon_gt": metrics.epsilon_gt,
            "x0_gt": metrics.x0_gt,
            "rgb_gt": metrics.rgb_gt,
            "edge_gt": metrics.edge_gt,
            "type_margin": metrics.type_margin,
            "wrong_semantic": metrics.wrong_semantic,
            "grad_l2": metrics.grad_l2,
            "finite": metrics.finite,
        }
        _append_json(metrics_path, record)
        records.append(record)
    return (
        fresh,
        _metrics_summary(records, _RENDERER_LOSS_FIELDS),
    )


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _git_head(cwd: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("Git HEAD must be a full lowercase commit hash")
    return head


def _threshold_payload(calibration: ThresholdCalibration) -> dict[str, object]:
    return {
        "schema_version": "v06_b_threshold_calibration_v1",
        "calibration_policy": _CALIBRATION_POLICY,
        "threshold": calibration.threshold,
        "macro_dice": calibration.macro_dice,
        "support_record_ids": list(calibration.support_record_ids),
    }


def _metric_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate metric key: {key}")
        record[key] = value
    return record


def _read_metric_stream(path: Path) -> tuple[dict[str, object], ...]:
    try:
        parsed = tuple(
            json.loads(line, object_pairs_hook=_metric_json_object)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid metric stream: {path.name}") from error
    if any(not isinstance(record, dict) for record in parsed):
        raise RuntimeError(f"invalid metric stream: {path.name}")
    return parsed  # type: ignore[return-value]


def _validate_metric_record(
    record: dict[str, object],
    *,
    fields: frozenset[str],
    numeric_fields: tuple[str, ...],
    path: Path,
) -> None:
    if set(record) != fields:
        raise RuntimeError(f"invalid metric schema: {path.name}")
    if record["diagnostic_only"] is not True or record["finite"] is not True:
        raise RuntimeError(f"invalid metric status: {path.name}")
    for name in numeric_fields:
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuntimeError(f"invalid metric numeric value: {path.name}")
        try:
            finite = math.isfinite(value)
        except OverflowError as error:
            raise RuntimeError(f"non-finite metric value: {path.name}") from error
        if not finite:
            raise RuntimeError(f"non-finite metric value: {path.name}")
    if float(record["grad_l2"]) <= 0.0:
        raise RuntimeError(f"non-positive metric gradient norm: {path.name}")


def _validate_planner_metric_stream(
    path: Path,
    *,
    schedule: tuple[UnpairedSlot, ...],
) -> tuple[dict[str, object], ...]:
    records = _read_metric_stream(path)
    if len(records) != len(schedule):
        raise RuntimeError(f"incomplete metric stream: {path.name}")
    for record, slot in zip(records, schedule, strict=True):
        _validate_metric_record(
            record,
            fields=_PLANNER_METRIC_FIELDS,
            numeric_fields=_PLANNER_LOSS_FIELDS,
            path=path,
        )
        expected = {
            "step": slot.step,
            "source_id": slot.source.sample_id,
            "target_id": slot.target.sample_id,
            "defect_type": slot.defect_type,
            "seed": slot.seed,
            "relationship": "unpaired_geometry_transfer",
        }
        if (
            type(record["step"]) is not int
            or type(record["seed"]) is not int
            or any(record[name] != value for name, value in expected.items())
        ):
            raise RuntimeError(f"unordered planner metric stream: {path.name}")
    return records


def _validate_renderer_metric_stream(
    path: Path,
    *,
    schedule: tuple[FewshotSample, ...],
    defect_types: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, object], ...]:
    records = _read_metric_stream(path)
    if len(records) != len(schedule):
        raise RuntimeError(f"incomplete metric stream: {path.name}")
    for step, (record, sample) in enumerate(
        zip(records, schedule, strict=True),
        start=1,
    ):
        _validate_metric_record(
            record,
            fields=_RENDERER_METRIC_FIELDS,
            numeric_fields=_RENDERER_LOSS_FIELDS,
            path=path,
        )
        try:
            defect_index = defect_types.index(sample.defect_type)
        except ValueError as error:
            raise RuntimeError("renderer schedule contains an unknown defect type") from error
        expected = {
            "step": step,
            "sample_id": sample.sample_id,
            "defect_type": sample.defect_type,
            "wrong_defect_type": defect_types[(defect_index + 1) % len(defect_types)],
            "seed": seed + 80_000_000 + step,
            "relationship": "real_anomaly_exact_ground_truth",
        }
        if (
            record["exact_ground_truth"] is not True
            or type(record["step"]) is not int
            or type(record["seed"]) is not int
            or any(record[name] != value for name, value in expected.items())
        ):
            raise RuntimeError(f"unordered renderer metric stream: {path.name}")
    return records


def run_mask_first_training(
    context: MaskFirstV06BContext,
) -> dict[str, object]:
    preflight = validate_mask_first_preflight(context)
    device = _require_cuda()
    output = context.output
    output.mkdir(parents=True, exist_ok=False)
    planner_metrics = output / "planner_metrics.jsonl"
    renderer_metrics = output / "renderer_metrics.jsonl"
    planner_metrics.touch(exist_ok=False)
    renderer_metrics.touch(exist_ok=False)

    torch.use_deterministic_algorithms(False)
    torch.manual_seed(context.experiment.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(context.experiment.seed)
    planner, calibration, planner_summary = _run_planner_training_and_calibration(
        preflight,
        device=device,
        metrics_path=planner_metrics,
    )
    threshold_path = output / "threshold_calibration.json"
    write_json_atomic(threshold_path, _threshold_payload(calibration))
    if not threshold_path.is_file():
        raise RuntimeError("threshold calibration was not persisted before R")
    # Persist completed H before starting the independent renderer stage.
    stage_path = output / "planner_stage.pt"
    stage_tmp = output / "planner_stage.pt.tmp"
    torch.save(
        {
            "schema_version": "v06_b_planner_stage_v1",
            "category": context.experiment.category,
            "defect_types": context.experiment.defect_types,
            "seed": context.experiment.seed,
            "planner_steps": context.experiment.planner_training_steps,
            "planner_state": {k: v.detach().cpu() for k, v in planner.state_dict().items()},
            "calibration": asdict(calibration),
            "planner_summary": planner_summary,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        stage_tmp,
    )
    stage_tmp.replace(stage_path)
    fresh, renderer_summary = _run_renderer_training(
        preflight,
        device=device,
        metrics_path=renderer_metrics,
    )
    persisted_planner_records = _validate_planner_metric_stream(
        planner_metrics,
        schedule=preflight.planner_schedule,
    )
    persisted_renderer_records = _validate_renderer_metric_stream(
        renderer_metrics,
        schedule=preflight.renderer_schedule,
        defect_types=context.experiment.defect_types,
        seed=context.experiment.seed,
    )
    persisted_planner_summary = _metrics_summary(
        list(persisted_planner_records),
        _PLANNER_LOSS_FIELDS,
    )
    persisted_renderer_summary = _metrics_summary(
        list(persisted_renderer_records),
        _RENDERER_LOSS_FIELDS,
    )
    if planner_summary != persisted_planner_summary:
        raise RuntimeError("planner metric summary differs from validated persisted records")
    if renderer_summary != persisted_renderer_summary:
        raise RuntimeError("renderer metric summary differs from validated persisted records")
    planner_summary = persisted_planner_summary
    renderer_summary = persisted_renderer_summary
    checkpoint = output / "checkpoint"
    save_mask_first_checkpoint(
        checkpoint,
        planner=planner,
        adapter=fresh.adapter,
        conditioner=fresh.conditioner,
        config=context.experiment,
        threshold=calibration,
        source=fresh.source,
        git_head=_git_head(context.cwd),
    )
    if not checkpoint.is_dir():
        raise RuntimeError("mask-first checkpoint was not persisted")
    summary: dict[str, object] = {
        "schema_version": "v06_b_mask_first_training_summary_v1",
        "diagnostic_only": True,
        "planner": planner_summary,
        "threshold_calibration": {
            "diagnostic_only": True,
            "threshold": calibration.threshold,
            "macro_dice": calibration.macro_dice,
            "support_record_ids": list(calibration.support_record_ids),
        },
        "renderer": renderer_summary,
        "source_lineage": {
            "diagnostic_only": True,
            "planner_initialization": "fresh",
            "renderer_initialization": renderer_initialization_label(context.experiment),
            "type_condition_initialization": type_condition_initialization_label(
                context.experiment
            ),
            "renderer_id": fresh.source.renderer_id,
            "manifest_sha256": fresh.source.manifest_sha256,
            "source_git_head": fresh.source.source_git_head,
            "checkpoint_hashes": dict(fresh.source.checkpoint_hashes),
            "source_model_config": context.experiment.source_model_config,
            "source_prompt_config": context.experiment.source_experiment_config,
        },
        "checkpoint": str(checkpoint),
        "visual_review_status": "not_performed_training_diagnostics_only",
        "visible_defects_passed": None,
    }
    write_json_atomic(output / "training_summary.json", summary)
    return summary


def _threshold_from_parent(parent_output: Path) -> ThresholdCalibration:
    payload = json.loads((parent_output / "threshold_calibration.json").read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "v06_b_threshold_calibration_v1"
        or payload.get("calibration_policy") != _CALIBRATION_POLICY
    ):
        raise ValueError("parent threshold calibration schema is incompatible")
    support_ids = payload.get("support_record_ids")
    if not isinstance(support_ids, list) or any(not isinstance(item, str) for item in support_ids):
        raise ValueError("parent support record IDs must be strings")
    return ThresholdCalibration(
        threshold=float(payload["threshold"]),
        macro_dice=float(payload["macro_dice"]),
        support_record_ids=tuple(support_ids),
    )


def run_mask_first_renderer_continue(
    context: MaskFirstV06BContext,
    *,
    parent_output: Path,
    parent_experiment: V06BMaskFirstConfig,
) -> dict[str, object]:
    if not isinstance(parent_output, Path) or not isinstance(
        parent_experiment, V06BMaskFirstConfig
    ):
        raise ValueError("parent output and experiment are required")
    validate_mask_first_renderer_continue_configs(context.experiment, parent_experiment)
    expected_parent = (context.paths.runs_root / parent_experiment.output_relpath).resolve(
        strict=False
    )
    if parent_output.resolve(strict=False) != expected_parent:
        raise ValueError("parent output must exactly match runs_root/parent output_relpath")
    parent_summary_path = parent_output / "training_summary.json"
    parent_checkpoint = parent_output / "checkpoint"
    if parent_summary_path.is_symlink() or not parent_summary_path.is_file():
        raise ValueError("parent training_summary.json is required")
    if parent_checkpoint.is_symlink() or not parent_checkpoint.is_dir():
        raise ValueError("parent checkpoint is required")
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    if (
        parent_summary.get("planner", {}).get("steps") != parent_experiment.planner_training_steps
        or parent_summary.get("renderer", {}).get("steps")
        != parent_experiment.renderer_training_steps
    ):
        raise ValueError("parent Cable scheme 2 training is incomplete")
    calibration = _threshold_from_parent(parent_output)
    preflight = validate_mask_first_preflight(context, skip_planner_geometry=True)
    device = _require_cuda()
    output = context.output
    output.mkdir(parents=True, exist_ok=False)
    renderer_metrics = output / "renderer_metrics.jsonl"
    renderer_metrics.touch(exist_ok=False)
    write_json_atomic(output / "threshold_calibration.json", _threshold_payload(calibration))

    torch.use_deterministic_algorithms(False)
    torch.manual_seed(context.experiment.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(context.experiment.seed)
    planner = PlannerNetwork(
        defect_types=context.experiment.defect_types,
        base_channels=context.experiment.planner_base_channels,
        type_embedding_dim=context.experiment.planner_type_embedding_dim,
    )
    fresh, renderer_summary = _run_renderer_training(
        preflight,
        device=device,
        metrics_path=renderer_metrics,
        parent_checkpoint=parent_checkpoint,
        parent_config=parent_experiment,
        planner=planner,
    )
    persisted_renderer_records = _validate_renderer_metric_stream(
        renderer_metrics,
        schedule=preflight.renderer_schedule,
        defect_types=context.experiment.defect_types,
        seed=context.experiment.seed,
    )
    persisted_renderer_summary = _metrics_summary(
        list(persisted_renderer_records),
        _RENDERER_LOSS_FIELDS,
    )
    if renderer_summary != persisted_renderer_summary:
        raise RuntimeError("renderer metric summary differs from validated persisted records")
    renderer_summary = persisted_renderer_summary
    checkpoint = output / "checkpoint"
    save_mask_first_checkpoint(
        checkpoint,
        planner=planner,
        adapter=fresh.adapter,
        conditioner=fresh.conditioner,
        config=context.experiment,
        threshold=calibration,
        source=fresh.source,
        git_head=_git_head(context.cwd),
    )
    if not checkpoint.is_dir():
        raise RuntimeError("mask-first checkpoint was not persisted")
    summary: dict[str, object] = {
        "schema_version": "v06_b_mask_first_training_summary_v1",
        "diagnostic_only": True,
        "planner": parent_summary["planner"],
        "threshold_calibration": {
            "diagnostic_only": True,
            "threshold": calibration.threshold,
            "macro_dice": calibration.macro_dice,
            "support_record_ids": list(calibration.support_record_ids),
            "reused_from_parent": True,
        },
        "renderer": renderer_summary,
        "renderer_continue": {
            "diagnostic_only": True,
            "parent_output_relpath": parent_experiment.output_relpath,
            "parent_renderer_steps": parent_experiment.renderer_training_steps,
            "continue_renderer_steps": context.experiment.renderer_training_steps,
            "cumulative_renderer_steps": (
                parent_experiment.renderer_training_steps
                + context.experiment.renderer_training_steps
            ),
            "planner_frozen": True,
            "threshold_reused": True,
            "not_a_to_h": True,
        },
        "source_lineage": {
            "diagnostic_only": True,
            "planner_initialization": "fresh",
            "renderer_initialization": renderer_initialization_label(context.experiment),
            "type_condition_initialization": type_condition_initialization_label(
                context.experiment
            ),
            "renderer_id": fresh.source.renderer_id,
            "manifest_sha256": fresh.source.manifest_sha256,
            "source_git_head": fresh.source.source_git_head,
            "checkpoint_hashes": dict(fresh.source.checkpoint_hashes),
            "source_model_config": context.experiment.source_model_config,
            "source_prompt_config": context.experiment.source_experiment_config,
        },
        "checkpoint": str(checkpoint),
        "visual_review_status": "not_performed_training_diagnostics_only",
        "visible_defects_passed": None,
    }
    write_json_atomic(output / "training_summary.json", summary)
    return summary


__all__ = [
    "MaskFirstPreflight",
    "MaskFirstV06BContext",
    "PlannerGeometryCertification",
    "bind_mask_first_renderer",
    "load_mask_first_source_prompts",
    "mask_first_data_authority_model",
    "run_mask_first_renderer_continue",
    "run_mask_first_training",
    "validate_mask_first_preflight",
    "validate_mask_first_source_binding",
]

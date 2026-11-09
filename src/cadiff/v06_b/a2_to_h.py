"""Cable scheme-2 a2→H: train H on official a2 proposals, freeze scheme-2 R.

Does not change the historical atlas A→H runner or its frozen relpath.
Empty transferred GT ∩ a2 proposal fails; no projection, no SquareDilate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from cadiff.diffusion.components import load_frozen_components
from cadiff.p1_v1.dataset import TrainingItem
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator, A2Result, load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import IdentityManifest, IdentityRecord, load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, estimate_normal_structure
from cadiff.v06_atlas_diag.features import FrozenDinoEncoder
from cadiff.v06_b.a_to_h import CachedRgbEncoder, homologous_atlas_seed
from cadiff.v06_b.config import (
    CABLE_SCHEME2_A2_TO_H_OUTPUT,
    CABLE_SCHEME2_OUTPUT,
    V06BMaskFirstConfig,
    is_cable_a2_to_h,
)
from cadiff.v06_b.dataset import GeometryPair, UnpairedSlot, build_geometry_pair, build_unpaired_schedule
from cadiff.v06_b.mask_first_checkpoint import (
    load_mask_first_checkpoint,
    renderer_initialization_label,
    save_mask_first_checkpoint,
    type_condition_initialization_label,
    validate_mask_first_a2_to_h_configs,
)
from cadiff.v06_b.planner import PlannerNetwork

_A2_EXPERIMENT = "configs/experiments/v06_a_cable_rc1.yaml"


@dataclass(frozen=True)
class A2PairDiagnostics:
    source_id: str
    target_id: str
    defect_type: str
    a2_seed: int
    a2_source_id: str
    alignment: str
    a2_pixels: int
    transferred_gt_pixels: int
    clipped_gt_pixels: int
    used_geometry_fallback: bool
    collapsed: bool


def clip_transferred_gt_to_a2(
    ground_truth: np.ndarray,
    proposal: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    if ground_truth.shape != proposal.shape or proposal.shape != valid.shape:
        raise ValueError("transferred GT, a2 proposal, and V_x must share one grid")
    if (proposal & ~valid).any():
        raise ValueError("a2 proposal must stay inside V_x")
    if not proposal.any() or bool(proposal.all()):
        raise ValueError("a2 proposal must be a non-empty non-full mask")
    clipped = np.asarray(ground_truth & proposal, dtype=bool)
    if not clipped.any():
        raise ValueError("transferred GT ∩ a2 proposal is empty")
    if not (proposal & ~clipped).any():
        raise ValueError("a2 proposal must remain a strict superset of clipped GT")
    return clipped


def _identity_record_for_target(
    target: TrainingItem,
    manifest: IdentityManifest,
) -> IdentityRecord:
    records = (*manifest.normal_fit, *manifest.normal_dev, *manifest.open_normals)
    matches = [record for record in records if record.image_relpath == target.image_relpath]
    if len(matches) == 1:
        return matches[0]
    return IdentityRecord(
        category=target.category,
        defect_type="good",
        item_id=PurePosixPath(target.image_relpath).stem,
        image_relpath=target.image_relpath,
        mask_relpath=None,
        role="normal_fit",
        sample_id=target.sample_id,
    )


class A2GeometryBuilder:
    """Build H training pairs whose proposal is official a2."""

    def __init__(
        self,
        *,
        generator: A2ProposalGenerator,
        experiment: Any,
        dataset_root: Path,
        identity_manifest: IdentityManifest,
        image_size: int,
        proposal_radius: int,
        support_corner_width: int,
    ) -> None:
        self._generator = generator
        self._experiment = experiment
        self._dataset_root = dataset_root
        self._manifest = identity_manifest
        self._image_size = image_size
        self._proposal_radius = proposal_radius
        self._support_corner_width = support_corner_width
        self._structures: dict[str, NormalStructure] = {}
        self._proposals: dict[tuple[str, str, int], A2Result] = {}
        self._pairs: dict[tuple[str, str, str, int], tuple[GeometryPair, A2PairDiagnostics]] = {}

    def pair_for_slot(self, slot: UnpairedSlot) -> GeometryPair:
        pair, _diagnostics = self.pair_and_diagnostics(slot)
        return pair

    def pair_and_diagnostics(self, slot: UnpairedSlot) -> tuple[GeometryPair, A2PairDiagnostics]:
        key = (
            slot.source.sample_id,
            slot.target.sample_id,
            slot.defect_type,
            homologous_atlas_seed(
                self._generator,
                slot.source.image_relpath,
                slot.defect_type,
            ),
        )
        cached = self._pairs.get(key)
        if cached is not None:
            return cached
        built = self._build(slot, a2_seed=key[3])
        self._pairs[key] = built
        return built

    def _structure(self, target: TrainingItem) -> NormalStructure:
        cached = self._structures.get(target.sample_id)
        if cached is not None:
            return cached
        record = _identity_record_for_target(target, self._manifest)
        loaded = load_identity_record(
            record,
            dataset_root=self._dataset_root,
            config=self._experiment,
        )
        structure = estimate_normal_structure(loaded, config=self._experiment)
        self._structures[target.sample_id] = structure
        return structure

    def _a2_result(
        self,
        structure: NormalStructure,
        defect_type: str,
        a2_seed: int,
    ) -> A2Result:
        key = (structure.normal.record.sample_id, defect_type, a2_seed)
        cached = self._proposals.get(key)
        if cached is not None:
            return cached
        result = self._generator.generate(structure, defect_type, a2_seed)
        self._proposals[key] = result
        return result

    def _build(
        self,
        slot: UnpairedSlot,
        *,
        a2_seed: int,
    ) -> tuple[GeometryPair, A2PairDiagnostics]:
        base = build_geometry_pair(
            slot.source,
            slot.target,
            image_size=self._image_size,
            proposal_radius=self._proposal_radius,
            support_corner_width=self._support_corner_width,
        )
        structure = self._structure(slot.target)
        a2 = self._a2_result(structure, slot.defect_type, a2_seed)
        transferred = np.ascontiguousarray(base.ground_truth[0].cpu().numpy(), dtype=bool)
        proposal = np.ascontiguousarray(a2.proposal, dtype=bool)
        valid = np.ascontiguousarray(base.target_valid[0].cpu().numpy(), dtype=bool)
        clipped = clip_transferred_gt_to_a2(transferred, proposal, valid)
        pair = replace(
            base,
            ground_truth=torch.from_numpy(np.array(clipped, dtype=bool, copy=True)).unsqueeze(0),
            proposal=torch.from_numpy(np.array(proposal, dtype=bool, copy=True)).unsqueeze(0),
        )
        diagnostics = A2PairDiagnostics(
            source_id=slot.source.sample_id,
            target_id=slot.target.sample_id,
            defect_type=slot.defect_type,
            a2_seed=a2_seed,
            a2_source_id=a2.source_id,
            alignment="subset",
            a2_pixels=int(proposal.sum()),
            transferred_gt_pixels=int(transferred.sum()),
            clipped_gt_pixels=int(clipped.sum()),
            used_geometry_fallback=bool(a2.used_geometry_fallback),
            collapsed=bool(a2.collapsed),
        )
        return pair, diagnostics


def _load_a2_builder(
    *,
    repository_root: Path,
    dataset_root: Path,
    device: torch.device,
    image_size: int,
    proposal_radius: int,
    support_corner_width: int,
) -> A2GeometryBuilder:
    experiment_path = repository_root / _A2_EXPERIMENT
    experiment = load_v06_a_experiment(experiment_path)
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    encoder = CachedRgbEncoder(FrozenDinoEncoder(device=str(device)))
    generator = load_a2_generator(
        experiment_path=experiment_path,
        dataset_root=dataset_root,
        device=str(device),
        encoder=encoder,
    )
    return A2GeometryBuilder(
        generator=generator,
        experiment=experiment,
        dataset_root=dataset_root,
        identity_manifest=manifest,
        image_size=image_size,
        proposal_radius=proposal_radius,
        support_corner_width=support_corner_width,
    )


def _unique_slots(slots: tuple[UnpairedSlot, ...]) -> tuple[UnpairedSlot, ...]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[UnpairedSlot] = []
    for slot in slots:
        key = (slot.source.sample_id, slot.target.sample_id, slot.defect_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(slot)
    return tuple(unique)


def _slot_key(slot: UnpairedSlot) -> tuple[str, str, str]:
    return (slot.source.sample_id, slot.target.sample_id, slot.defect_type)


def _scan_alignment(
    builder: A2GeometryBuilder,
    slots: tuple[UnpairedSlot, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, tuple[UnpairedSlot, ...]]]:
    rows: list[dict[str, object]] = []
    empty: list[dict[str, object]] = []
    valid_by_type: dict[str, list[UnpairedSlot]] = {}
    unique = _unique_slots(slots)
    for index, slot in enumerate(unique, start=1):
        try:
            _pair, diagnostics = builder.pair_and_diagnostics(slot)
        except ValueError as error:
            if "empty" not in str(error):
                raise
            empty.append(
                {
                    "source_id": slot.source.sample_id,
                    "target_id": slot.target.sample_id,
                    "defect_type": slot.defect_type,
                    "reason": str(error),
                }
            )
            print(
                f"a2→H pair FAIL empty {index}/{len(unique)} "
                f"{slot.defect_type} src={slot.source.sample_id} tgt={slot.target.sample_id}",
                flush=True,
            )
            continue
        rows.append(asdict(diagnostics))
        valid_by_type.setdefault(slot.defect_type, []).append(slot)
        if index == 1 or index == len(unique) or index % 50 == 0:
            print(
                f"a2→H pairs {index}/{len(unique)} empty={len(empty)}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("no valid transferred GT ∩ a2 proposal pairs")
    return rows, empty, {kind: tuple(pool) for kind, pool in valid_by_type.items()}


def replace_empty_slots(
    slots: tuple[UnpairedSlot, ...],
    *,
    empty_keys: set[tuple[str, str, str]],
    valid_by_type: dict[str, tuple[UnpairedSlot, ...]],
) -> tuple[UnpairedSlot, ...]:
    replaced: list[UnpairedSlot] = []
    cursor = {kind: 0 for kind in valid_by_type}
    for slot in slots:
        if _slot_key(slot) not in empty_keys:
            replaced.append(slot)
            continue
        pool = valid_by_type.get(slot.defect_type, ())
        if not pool:
            raise RuntimeError(f"no valid a2 pair remains for {slot.defect_type}")
        donor = pool[cursor[slot.defect_type] % len(pool)]
        cursor[slot.defect_type] = cursor[slot.defect_type] + 1
        replaced.append(replace(slot, source=donor.source, target=donor.target))
    return tuple(replaced)


def run_mask_first_a2_to_h_training(
    context: Any,
    *,
    parent_output: Path,
    parent_experiment: V06BMaskFirstConfig,
) -> dict[str, object]:
    from cadiff.v06_b.mask_first_runner import (
        MaskFirstV06BContext,
        _PLANNER_LOSS_FIELDS,
        _git_head,
        _metrics_summary,
        _require_cuda,
        _run_planner_training_and_calibration,
        _threshold_payload,
        _validate_planner_metric_stream,
        bind_mask_first_renderer,
        validate_mask_first_preflight,
    )
    from cadiff.v06_b.source import load_fresh_p2_b2_renderer

    if not isinstance(context, MaskFirstV06BContext):
        raise ValueError("context must be a MaskFirstV06BContext")
    if not isinstance(parent_output, Path) or not isinstance(
        parent_experiment, V06BMaskFirstConfig
    ):
        raise ValueError("parent output and experiment are required")
    if not is_cable_a2_to_h(context.experiment):
        raise ValueError("a2→H runner requires the authorized scheme2-a2-to-h config")
    validate_mask_first_a2_to_h_configs(context.experiment, parent_experiment)
    expected_parent = (context.paths.runs_root / CABLE_SCHEME2_OUTPUT).resolve(strict=False)
    if parent_output.resolve(strict=False) != expected_parent:
        raise ValueError("a2→H parent output must be the frozen Cable scheme-2 run")
    if parent_experiment.output_relpath != CABLE_SCHEME2_OUTPUT:
        raise ValueError("a2→H parent experiment must be the frozen Cable scheme-2 config")
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

    preflight = validate_mask_first_preflight(context, skip_planner_geometry=True)
    device = _require_cuda()
    config = context.experiment
    builder = _load_a2_builder(
        repository_root=context.cwd,
        dataset_root=context.paths.dataset_root,
        device=device,
        image_size=config.image_size,
        proposal_radius=config.proposal_radius,
        support_corner_width=config.support_corner_width,
    )
    calibration_schedule = build_unpaired_schedule(
        sources=preflight.split.train,
        targets=preflight.planner_targets.calibration,
        defect_types=config.defect_types,
        steps=max(len(preflight.planner_targets.calibration), len(config.defect_types)),
        seed=config.seed + 100_000_000,
    )
    alignment_rows, empty_rows, valid_by_type = _scan_alignment(
        builder,
        preflight.planner_schedule + calibration_schedule,
    )
    empty_keys = {
        (str(row["source_id"]), str(row["target_id"]), str(row["defect_type"]))
        for row in empty_rows
    }
    if empty_keys:
        by_type: dict[str, int] = {}
        for row in empty_rows:
            kind = str(row["defect_type"])
            by_type[kind] = by_type.get(kind, 0) + 1
        print(
            "a2→H skipping empty pairs without projection: "
            f"{len(empty_rows)}/{len(alignment_rows) + len(empty_rows)} {by_type}",
            flush=True,
        )
        planner_schedule = replace_empty_slots(
            preflight.planner_schedule,
            empty_keys=empty_keys,
            valid_by_type=valid_by_type,
        )
        calibration_schedule = replace_empty_slots(
            calibration_schedule,
            empty_keys=empty_keys,
            valid_by_type=valid_by_type,
        )
        preflight = replace(preflight, planner_schedule=planner_schedule)
    calibration_pairs = tuple(builder.pair_for_slot(slot) for slot in calibration_schedule)
    calibration_ids = {target.sample_id for target in preflight.planner_targets.calibration}
    train_ids = {sample.sample_id for sample in preflight.split.train}
    if (
        {pair.target_id for pair in calibration_pairs} != calibration_ids
        or {pair.defect_type for pair in calibration_pairs} != set(config.defect_types)
        or any(pair.source_id not in train_ids for pair in calibration_pairs)
    ):
        raise RuntimeError("a2→H calibration pairs do not cover exact support authority")
    preflight = replace(preflight, calibration_pairs=calibration_pairs)

    output = context.output
    output.mkdir(parents=True, exist_ok=False)
    planner_metrics = output / "planner_metrics.jsonl"
    planner_metrics.touch(exist_ok=False)
    write_json_atomic(
        output / "a2_pair_diagnostics.json",
        {
            "schema_version": "v06_b_a2_to_h_pair_diagnostics_v1",
            "diagnostic_only": True,
            "unique_pairs": len(alignment_rows) + len(empty_rows),
            "valid_pairs": len(alignment_rows),
            "empty_pairs": len(empty_rows),
            "empty_rows": empty_rows,
            "projected_pairs": 0,
            "skipped_empty_without_projection": True,
            "rows": alignment_rows,
        },
    )

    torch.use_deterministic_algorithms(False)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    planner, calibration, planner_summary = _run_planner_training_and_calibration(
        preflight,
        device=device,
        metrics_path=planner_metrics,
        pair_builder=builder.pair_for_slot,
        require_geometry_certifications=False,
        proposal_must_subset_support=False,
    )
    write_json_atomic(output / "threshold_calibration.json", _threshold_payload(calibration))
    persisted_planner_records = _validate_planner_metric_stream(
        planner_metrics,
        schedule=preflight.planner_schedule,
    )
    persisted_planner_summary = _metrics_summary(
        list(persisted_planner_records),
        _PLANNER_LOSS_FIELDS,
    )
    if planner_summary != persisted_planner_summary:
        raise RuntimeError("planner metric summary differs from validated persisted records")

    components = load_frozen_components(
        context.paths.model_root,
        device,
        torch.float16,
    )
    throwaway_planner = PlannerNetwork(
        defect_types=config.defect_types,
        base_channels=config.planner_base_channels,
        type_embedding_dim=config.planner_type_embedding_dim,
    ).to(device=device, dtype=torch.float32)
    fresh = load_fresh_p2_b2_renderer(
        manifest_path=context.cwd / config.source_renderer_manifest,
        runs_root=context.paths.runs_root,
        category="cable",
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
    load_mask_first_checkpoint(
        parent_checkpoint,
        planner=throwaway_planner,
        adapter=fresh.adapter,
        conditioner=fresh.conditioner,
        config=parent_experiment,
        source=fresh.source,
    )
    del throwaway_planner

    checkpoint = output / "checkpoint"
    save_mask_first_checkpoint(
        checkpoint,
        planner=planner,
        adapter=fresh.adapter,
        conditioner=fresh.conditioner,
        config=config,
        threshold=calibration,
        source=fresh.source,
        git_head=_git_head(context.cwd),
    )
    if not checkpoint.is_dir():
        raise RuntimeError("mask-first checkpoint was not persisted")
    summary: dict[str, object] = {
        "schema_version": "v06_b_mask_first_training_summary_v1",
        "diagnostic_only": True,
        "planner": persisted_planner_summary,
        "threshold_calibration": {
            "diagnostic_only": True,
            "threshold": calibration.threshold,
            "macro_dice": calibration.macro_dice,
            "support_record_ids": list(calibration.support_record_ids),
            "reused_from_parent": False,
        },
        "renderer": parent_summary["renderer"],
        "a2_to_h": {
            "diagnostic_only": True,
            "parent_output_relpath": parent_experiment.output_relpath,
            "planner_proposal": "idea_a2",
            "supervision": "transferred_gt_intersect_a2",
            "empty_pairs": len(empty_rows),
            "empty_skipped_without_projection": True,
            "no_projection": True,
            "no_square_dilate": True,
            "renderer_frozen_from_parent": True,
            "threshold_recalibrated": True,
            "not_official_pass": True,
            "output_relpath": CABLE_SCHEME2_A2_TO_H_OUTPUT,
        },
        "source_lineage": {
            "diagnostic_only": True,
            "planner_initialization": "fresh",
            "renderer_initialization": renderer_initialization_label(config),
            "type_condition_initialization": type_condition_initialization_label(config),
            "renderer_id": fresh.source.renderer_id,
            "manifest_sha256": fresh.source.manifest_sha256,
            "source_git_head": fresh.source.source_git_head,
            "checkpoint_hashes": dict(fresh.source.checkpoint_hashes),
            "source_model_config": config.source_model_config,
            "source_prompt_config": config.source_experiment_config,
        },
        "checkpoint": str(checkpoint),
        "visual_review_status": "not_performed_training_diagnostics_only",
        "visible_defects_passed": None,
    }
    write_json_atomic(output / "training_summary.json", summary)
    return summary


__all__ = [
    "A2GeometryBuilder",
    "A2PairDiagnostics",
    "clip_transferred_gt_to_a2",
    "replace_empty_slots",
    "run_mask_first_a2_to_h_training",
]

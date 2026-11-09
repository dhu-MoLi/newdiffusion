"""Cable scheme-2 A→H: train H on atlas proposals, freeze scheme-2 R."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from cadiff.diffusion.components import load_frozen_components
from cadiff.p1_v1.dataset import TrainingItem
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import IdentityManifest, IdentityRecord, load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, estimate_normal_structure
from cadiff.v06_atlas_diag.atlas_a import AtlasAResult, AtlasProposalGenerator, load_atlas_a_generator
from cadiff.v06_atlas_diag.features import FrozenDinoEncoder
from cadiff.v06_b.config import (
    CABLE_SCHEME2_A_TO_H_OUTPUT,
    CABLE_SCHEME2_OUTPUT,
    V06BMaskFirstConfig,
    is_cable_a_to_h,
)
from cadiff.v06_b.dataset import GeometryPair, UnpairedSlot, build_geometry_pair, build_unpaired_schedule
from cadiff.v06_b.mask_first_checkpoint import (
    load_mask_first_checkpoint,
    renderer_initialization_label,
    save_mask_first_checkpoint,
    type_condition_initialization_label,
    validate_mask_first_a_to_h_configs,
)
from cadiff.v06_b.planner import PlannerNetwork

_ATLAS_EXPERIMENT = "configs/experiments/v06_a_cable_rc1.yaml"
_MAX_PROJECTED_FRACTION = 0.20


class _RgbEncoder(Protocol):
    model_name: str

    def encode(self, rgb: NDArray[np.uint8]) -> Any:
        ...


@dataclass(frozen=True)
class AtlasPairDiagnostics:
    source_id: str
    target_id: str
    defect_type: str
    atlas_seed: int
    atlas_source_id: str
    alignment: str
    atlas_pixels: int
    transferred_gt_pixels: int
    clipped_gt_pixels: int
    used_geometry_fallback: bool
    collapsed: bool


class CachedRgbEncoder:
    """Reuse frozen DINO encodings of identical RGB bytes."""

    def __init__(self, inner: FrozenDinoEncoder) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self._cache: dict[str, Any] = {}

    def encode(self, rgb: NDArray[np.uint8]) -> Any:
        key = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        cached = self._cache.get(key)
        if cached is None:
            cached = self._inner.encode(rgb)
            self._cache[key] = cached
        return cached


def homologous_atlas_seed(
    generator: AtlasProposalGenerator,
    source_relpath: str,
    defect_type: str,
) -> int:
    sources = generator._sources.get(defect_type)
    if sources is None:
        raise ValueError(f"unknown atlas A defect type: {defect_type}")
    matches = [
        index
        for index, loaded in enumerate(sources)
        if loaded.record.image_relpath == source_relpath
    ]
    if len(matches) != 1:
        raise ValueError(f"few-shot source is not a unique atlas A identity: {source_relpath}")
    return matches[0]


def project_gt_onto_atlas(ground_truth: np.ndarray, atlas: np.ndarray) -> np.ndarray:
    gt_rows, gt_cols = np.nonzero(ground_truth)
    atlas_rows, atlas_cols = np.nonzero(atlas)
    if gt_rows.size == 0 or atlas_rows.size == 0:
        raise ValueError("projection requires non-empty transferred GT and atlas proposal")
    center_row = float(gt_rows.mean())
    center_col = float(gt_cols.mean())
    nearest = int(
        np.argmin((atlas_rows - center_row) ** 2 + (atlas_cols - center_col) ** 2)
    )
    row = int(atlas_rows[nearest])
    col = int(atlas_cols[nearest])
    target = np.zeros_like(atlas, dtype=bool)
    height, width = atlas.shape
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            candidate_row = row + delta_row
            candidate_col = col + delta_col
            if (
                0 <= candidate_row < height
                and 0 <= candidate_col < width
                and atlas[candidate_row, candidate_col]
            ):
                target[candidate_row, candidate_col] = True
    if not target.any():
        target[row, col] = True
    return target


def clip_transferred_gt_to_atlas(
    ground_truth: np.ndarray,
    atlas: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, str]:
    if ground_truth.shape != atlas.shape or atlas.shape != valid.shape:
        raise ValueError("transferred GT, atlas proposal, and V_x must share one grid")
    if (atlas & ~valid).any():
        raise ValueError("atlas proposal must stay inside V_x")
    if not atlas.any() or bool(atlas.all()):
        raise ValueError("atlas proposal must be a non-empty non-full mask")
    clipped = np.asarray(ground_truth & atlas, dtype=bool)
    alignment = "subset"
    if not clipped.any():
        clipped = project_gt_onto_atlas(ground_truth, atlas)
        alignment = "projected"
    if not (atlas & ~clipped).any():
        raise ValueError("atlas proposal must remain a strict superset of clipped GT")
    return clipped, alignment


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


class AtlasGeometryBuilder:
    """Build H training pairs whose proposal is official atlas A."""

    def __init__(
        self,
        *,
        generator: AtlasProposalGenerator,
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
        self._atlas: dict[tuple[str, str, int], AtlasAResult] = {}
        self._pairs: dict[tuple[str, str, str, int], tuple[GeometryPair, AtlasPairDiagnostics]] = {}

    def pair_for_slot(self, slot: UnpairedSlot) -> GeometryPair:
        pair, _diagnostics = self.pair_and_diagnostics(slot)
        return pair

    def pair_and_diagnostics(self, slot: UnpairedSlot) -> tuple[GeometryPair, AtlasPairDiagnostics]:
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
        built = self._build(slot, atlas_seed=key[3])
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

    def _atlas_result(
        self,
        structure: NormalStructure,
        defect_type: str,
        atlas_seed: int,
    ) -> AtlasAResult:
        key = (structure.normal.record.sample_id, defect_type, atlas_seed)
        cached = self._atlas.get(key)
        if cached is not None:
            return cached
        result = self._generator.generate(structure, defect_type, atlas_seed)
        self._atlas[key] = result
        return result

    def _build(
        self,
        slot: UnpairedSlot,
        *,
        atlas_seed: int,
    ) -> tuple[GeometryPair, AtlasPairDiagnostics]:
        base = build_geometry_pair(
            slot.source,
            slot.target,
            image_size=self._image_size,
            proposal_radius=self._proposal_radius,
            support_corner_width=self._support_corner_width,
        )
        structure = self._structure(slot.target)
        atlas = self._atlas_result(structure, slot.defect_type, atlas_seed)
        transferred = np.ascontiguousarray(base.ground_truth[0].cpu().numpy(), dtype=bool)
        proposal = np.ascontiguousarray(atlas.proposal, dtype=bool)
        valid = np.ascontiguousarray(base.target_valid[0].cpu().numpy(), dtype=bool)
        clipped, alignment = clip_transferred_gt_to_atlas(transferred, proposal, valid)
        pair = replace(
            base,
            ground_truth=torch.from_numpy(np.array(clipped, dtype=bool, copy=True)).unsqueeze(0),
            proposal=torch.from_numpy(np.array(proposal, dtype=bool, copy=True)).unsqueeze(0),
        )
        diagnostics = AtlasPairDiagnostics(
            source_id=slot.source.sample_id,
            target_id=slot.target.sample_id,
            defect_type=slot.defect_type,
            atlas_seed=atlas_seed,
            atlas_source_id=atlas.source_id,
            alignment=alignment,
            atlas_pixels=int(proposal.sum()),
            transferred_gt_pixels=int(transferred.sum()),
            clipped_gt_pixels=int(clipped.sum()),
            used_geometry_fallback=bool(atlas.used_geometry_fallback),
            collapsed=bool(atlas.collapsed),
        )
        return pair, diagnostics


def _load_atlas_builder(
    *,
    repository_root: Path,
    dataset_root: Path,
    device: torch.device,
    image_size: int,
    proposal_radius: int,
    support_corner_width: int,
) -> AtlasGeometryBuilder:
    experiment_path = repository_root / _ATLAS_EXPERIMENT
    experiment = load_v06_a_experiment(experiment_path)
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    encoder = CachedRgbEncoder(FrozenDinoEncoder(device=str(device)))
    generator = load_atlas_a_generator(
        experiment_path=experiment_path,
        dataset_root=dataset_root,
        device=str(device),
        encoder=encoder,
    )
    return AtlasGeometryBuilder(
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


def _scan_alignment(
    builder: AtlasGeometryBuilder,
    slots: tuple[UnpairedSlot, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    unique = _unique_slots(slots)
    for index, slot in enumerate(unique, start=1):
        _pair, diagnostics = builder.pair_and_diagnostics(slot)
        rows.append(asdict(diagnostics))
        if index == 1 or index == len(unique) or index % 50 == 0:
            print(
                f"A→H atlas pairs {index}/{len(unique)} "
                f"projected={sum(1 for row in rows if row['alignment'] == 'projected')}",
                flush=True,
            )
    projected = sum(1 for row in rows if row["alignment"] == "projected")
    if rows and projected / len(rows) > _MAX_PROJECTED_FRACTION:
        raise RuntimeError(
            "too many A→H slots needed GT projection onto atlas: "
            f"{projected}/{len(rows)} exceeds {_MAX_PROJECTED_FRACTION}"
        )
    return rows


def run_mask_first_a_to_h_training(
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
    if not is_cable_a_to_h(context.experiment):
        raise ValueError("A→H runner requires the authorized scheme2-a-to-h config")
    validate_mask_first_a_to_h_configs(context.experiment, parent_experiment)
    expected_parent = (context.paths.runs_root / CABLE_SCHEME2_OUTPUT).resolve(strict=False)
    if parent_output.resolve(strict=False) != expected_parent:
        raise ValueError("A→H parent output must be the frozen Cable scheme-2 run")
    if parent_experiment.output_relpath != CABLE_SCHEME2_OUTPUT:
        raise ValueError("A→H parent experiment must be the frozen Cable scheme-2 config")
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
    builder = _load_atlas_builder(
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
    alignment_rows = _scan_alignment(
        builder,
        preflight.planner_schedule + calibration_schedule,
    )
    calibration_pairs = tuple(builder.pair_for_slot(slot) for slot in calibration_schedule)
    calibration_ids = {target.sample_id for target in preflight.planner_targets.calibration}
    train_ids = {sample.sample_id for sample in preflight.split.train}
    if (
        {pair.target_id for pair in calibration_pairs} != calibration_ids
        or {pair.defect_type for pair in calibration_pairs} != set(config.defect_types)
        or any(pair.source_id not in train_ids for pair in calibration_pairs)
    ):
        raise RuntimeError("A→H calibration pairs do not cover exact support authority")
    preflight = replace(preflight, calibration_pairs=calibration_pairs)

    output = context.output
    output.mkdir(parents=True, exist_ok=False)
    planner_metrics = output / "planner_metrics.jsonl"
    planner_metrics.touch(exist_ok=False)
    write_json_atomic(
        output / "atlas_pair_diagnostics.json",
        {
            "schema_version": "v06_b_a_to_h_pair_diagnostics_v1",
            "diagnostic_only": True,
            "unique_pairs": len(alignment_rows),
            "projected_pairs": sum(1 for row in alignment_rows if row["alignment"] == "projected"),
            "max_projected_fraction": _MAX_PROJECTED_FRACTION,
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
        "a_to_h": {
            "diagnostic_only": True,
            "parent_output_relpath": parent_experiment.output_relpath,
            "planner_proposal": "atlas_a",
            "supervision": "transferred_gt_intersect_atlas",
            "renderer_frozen_from_parent": True,
            "threshold_recalibrated": True,
            "not_official_pass": True,
            "output_relpath": CABLE_SCHEME2_A_TO_H_OUTPUT,
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
    "AtlasGeometryBuilder",
    "AtlasPairDiagnostics",
    "CachedRgbEncoder",
    "clip_transferred_gt_to_atlas",
    "homologous_atlas_seed",
    "project_gt_onto_atlas",
    "run_mask_first_a_to_h_training",
]

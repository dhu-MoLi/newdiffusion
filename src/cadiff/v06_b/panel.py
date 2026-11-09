from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cadiff.p1_v1.dataset import TrainingItem
from cadiff.p2_a0.fewshot import FewshotSample
from cadiff.provenance import write_json_atomic

from .backend import BackendDiagnosticOutput, ProposalConditionedBackend
from .checkpoint import load_v06_b_checkpoint
from .dataset import build_geometry_pair
from .evaluation import evaluate_realization, evaluate_seed_response
from .renderer import compose_causal
from .runner import (
    V06BContext,
    _load_runtime_assets,
    _new_training_state,
    _require_cuda,
    validate_v06_b_preflight,
)


@dataclass(frozen=True, eq=False)
class PanelSlot:
    defect_type: str
    source: FewshotSample
    target: TrainingItem
    normal_index: int
    seed: int


def build_fixed_panel_schedule(
    *,
    sources: tuple[FewshotSample, ...],
    targets: tuple[TrainingItem, ...],
    defect_types: tuple[str, ...],
    normal_count: int,
    seeds: tuple[int, ...],
) -> tuple[PanelSlot, ...]:
    if (
        isinstance(normal_count, bool)
        or not isinstance(normal_count, int)
        or normal_count <= 0
        or normal_count > len(targets)
    ):
        raise ValueError("normal_count must fit the normal_train pool")
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("panel seeds must be unique")
    if not defect_types or len(set(defect_types)) != len(defect_types):
        raise ValueError("defect_types must be non-empty and unique")
    source_by_type: dict[str, FewshotSample] = {}
    for defect_type in defect_types:
        candidates = tuple(
            sorted(
                (
                    source
                    for source in sources
                    if source.role == "fewshot_train"
                    and source.defect_type == defect_type
                ),
                key=lambda source: source.image_relpath,
            )
        )
        if not candidates:
            raise ValueError("every panel defect type needs a fewshot_train source")
        source_by_type[defect_type] = candidates[0]
    ordered_targets = tuple(sorted(targets, key=lambda target: target.image_relpath))
    if any(
        target.role != "normal_train" or target.defect_type != "good"
        for target in ordered_targets
    ):
        raise ValueError("panel targets must be normal_train/good")
    if normal_count == 1:
        indices = (0,)
    else:
        indices = tuple(
            round(index * (len(ordered_targets) - 1) / (normal_count - 1))
            for index in range(normal_count)
        )
    selected = tuple(ordered_targets[index] for index in indices)
    return tuple(
        PanelSlot(
            defect_type=defect_type,
            source=source_by_type[defect_type],
            target=target,
            normal_index=normal_index,
            seed=seed,
        )
        for defect_type in defect_types
        for normal_index, target in enumerate(selected)
        for seed in seeds
    )


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.astype(np.float32, copy=True)
    color_array = np.asarray(color, dtype=np.float32)
    result[mask] = 0.55 * result[mask] + 0.45 * color_array
    return np.rint(result).clip(0, 255).astype(np.uint8)


def _q_heat(image: np.ndarray, q: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32, copy=True)
    alpha = (0.70 * q.astype(np.float32))[..., None]
    heat = np.zeros_like(result)
    heat[..., 0] = 255.0
    heat[..., 2] = 80.0
    result = result * (1.0 - alpha) + heat * alpha
    return np.rint(result).clip(0, 255).astype(np.uint8)


def panel_columns(
    *,
    normal: np.ndarray,
    real_reference: np.ndarray,
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    proposal: np.ndarray,
    q: np.ndarray,
    threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if (
        normal.dtype != np.uint8
        or rendered.dtype != np.uint8
        or normal.shape != rendered.shape
        or normal.ndim != 3
        or normal.shape[2] != 3
    ):
        raise ValueError("normal and rendered must be matching uint8 RGB arrays")
    if (
        not isinstance(real_reference, np.ndarray)
        or real_reference.dtype != np.uint8
        or real_reference.ndim != 3
        or real_reference.shape[2] != 3
    ):
        raise ValueError("real_reference must be a uint8 RGB array")
    shape = normal.shape[:2]
    for name, mask in (("ground_truth", ground_truth), ("proposal", proposal)):
        if not isinstance(mask, np.ndarray) or mask.shape != shape or not np.isin(
            mask, (0, 1)
        ).all():
            raise ValueError(f"{name} must be a matching binary mask")
    ground_truth = ground_truth.astype(bool, copy=False)
    proposal = proposal.astype(bool, copy=False)
    if (ground_truth & ~proposal).any():
        raise ValueError("ground_truth must be a subset of proposal")
    if (
        not isinstance(q, np.ndarray)
        or q.shape != shape
        or not np.issubdtype(q.dtype, np.floating)
        or not np.isfinite(q).all()
        or np.any((q < 0.0) | (q > 1.0))
        or np.any(q[~proposal] != 0.0)
    ):
        raise ValueError("q must be finite, continuous, and proposal-gated")
    final = proposal & (q >= float(threshold))
    columns = {
        "x": normal.copy(),
        "real_ref": real_reference.copy(),
        "gt": _overlay(normal, ground_truth, (0, 255, 0)),
        "proposal": _overlay(normal, proposal, (255, 210, 0)),
        "raw_r": rendered.copy(),
        "proposal_y": compose_causal(normal, rendered, proposal),
        "q": _q_heat(normal, q),
        "m_final": _overlay(normal, final, (255, 0, 0)),
        "final_y": compose_causal(normal, rendered, final),
    }
    return columns, final


def _normal_rgb(pixels: torch.Tensor) -> np.ndarray:
    return (
        pixels.add(1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .copy()
    )


def _real_reference_crop(
    *,
    pixels: torch.Tensor,
    ground_truth: torch.Tensor,
    padding: int,
    size: int,
) -> np.ndarray:
    if (
        not isinstance(pixels, torch.Tensor)
        or pixels.ndim != 3
        or pixels.shape[0] != 3
        or not isinstance(ground_truth, torch.Tensor)
        or ground_truth.ndim != 3
        or ground_truth.shape[0] != 1
        or ground_truth.shape[1:] != pixels.shape[1:]
        or isinstance(padding, bool)
        or not isinstance(padding, int)
        or padding < 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ValueError("real reference inputs must be a valid RGB image and GT mask")
    mask = ground_truth.detach().to(device="cpu", dtype=torch.bool)[0].numpy()
    if not mask.any():
        raise ValueError("real reference ground truth must be non-empty")
    rows, columns = np.nonzero(mask)
    height, width = mask.shape
    side = min(
        max(int(rows.max() - rows.min() + 1), int(columns.max() - columns.min() + 1))
        + 2 * padding,
        height,
        width,
    )
    row_center = int((int(rows.min()) + int(rows.max()) + 1) // 2)
    column_center = int((int(columns.min()) + int(columns.max()) + 1) // 2)
    top = min(max(row_center - side // 2, 0), height - side)
    left = min(max(column_center - side // 2, 0), width - side)
    reference = _normal_rgb(pixels)[top : top + side, left : left + side]
    return np.asarray(
        Image.fromarray(reference).resize((size, size), Image.Resampling.LANCZOS)
    ).copy()


def _thumbnail(image: np.ndarray, size: int = 128) -> Image.Image:
    return Image.fromarray(image).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def _save_contact_sheet(
    path: Path,
    rows: list[tuple[str, dict[str, np.ndarray]]],
) -> None:
    names = (
        "x",
        "real_ref",
        "gt",
        "proposal",
        "raw_r",
        "proposal_y",
        "q",
        "m_final",
        "final_y",
    )
    size = 128
    header = 24
    canvas = Image.new("RGB", (len(names) * size, header + len(rows) * size), "black")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(names):
        draw.text((column * size + 4, 5), name, fill="white")
    for row_index, (label, columns) in enumerate(rows):
        top = header + row_index * size
        for column_index, name in enumerate(names):
            thumbnail = _thumbnail(columns[name], size)
            if column_index == 0:
                ImageDraw.Draw(thumbnail).text((4, 4), label, fill="white")
            canvas.paste(thumbnail, (column_index * size, top))
    canvas.save(path)


def save_blind_final_y_panel(
    image_path: Path,
    *,
    key_path: Path,
    final_y_by_type: Mapping[str, Sequence[np.ndarray]],
    defect_types: tuple[str, ...],
    seed: int,
    tile_size: int,
) -> dict[str, object]:
    if (
        not isinstance(image_path, Path)
        or image_path.name != "review.png"
        or not isinstance(key_path, Path)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(tile_size, bool)
        or not isinstance(tile_size, int)
        or tile_size <= 0
        or len(defect_types) != 4
        or len(set(defect_types)) != 4
        or set(final_y_by_type) != set(defect_types)
    ):
        raise ValueError("blind panel requires four uniquely keyed final_y columns")
    if image_path.parent.exists() or image_path.parent.is_symlink():
        raise FileExistsError("blind panel output already exists")
    if key_path.exists() or key_path.is_symlink():
        raise FileExistsError("blind panel mapping output already exists")

    columns = tuple(random.Random(seed).sample(defect_types, k=len(defect_types)))
    canvas = np.empty((8 * tile_size, 4 * tile_size, 3), dtype=np.uint8)
    for column, defect_type in enumerate(columns):
        values = final_y_by_type[defect_type]
        if len(values) != 8:
            raise ValueError("each blind panel type must contain exactly eight final_y images")
        for row, final_y in enumerate(values):
            if (
                not isinstance(final_y, np.ndarray)
                or final_y.dtype != np.uint8
                or final_y.shape != (tile_size, tile_size, 3)
            ):
                raise ValueError("blind panel final_y images must be matching uint8 RGB tiles")
            canvas[
                row * tile_size : (row + 1) * tile_size,
                column * tile_size : (column + 1) * tile_size,
            ] = final_y
    mapping: dict[str, object] = {
        "schema_version": "v06_b_blind_panel_mapping_v1",
        "seed": seed,
        "columns": [
            {"alias": chr(ord("A") + index), "defect_type": defect_type}
            for index, defect_type in enumerate(columns)
        ],
    }
    image_path.parent.mkdir(parents=True, exist_ok=False)
    Image.fromarray(canvas).save(image_path)
    write_json_atomic(key_path, mapping)
    return mapping


def run_v06_b_panel(context: V06BContext) -> dict[str, object]:
    preflight = validate_v06_b_preflight(context)
    checkpoint = context.output / "checkpoint"
    panel_root = context.output / "panel"
    if panel_root.exists() or panel_root.is_symlink():
        raise FileExistsError("fixed Module-B panel output already exists")
    device = _require_cuda()
    assets = _load_runtime_assets(preflight, device)
    state = _new_training_state(preflight, assets, context.experiment, device)
    load_v06_b_checkpoint(
        checkpoint,
        adapter=state.adapter,
        conditioner=state.conditioner,
        realization_head=state.head,
        config=context.experiment,
        source_renderer_manifest_sha256=(
            preflight.source_renderer.manifest_sha256
        ),
        expected_step=context.experiment.final_step,
    )
    state.model.eval()
    state.head.eval()
    backend = ProposalConditionedBackend(
        components=assets.components,
        model=state.model,
        realization_head=state.head,
        prompts=assets.prompts,
        config=context.experiment,
        device=device,
    )
    schedule = build_fixed_panel_schedule(
        sources=preflight.split.train,
        targets=preflight.training_index.normals,
        defect_types=context.experiment.defect_types,
        normal_count=context.experiment.panel_normal_count,
        seeds=context.experiment.panel_seeds,
    )
    if len(schedule) != 32:
        raise RuntimeError("fixed Module-B panel must contain exactly 32 renders")
    samples_root = panel_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    contact_rows: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = defaultdict(list)
    seed_groups: dict[
        tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = defaultdict(list)
    source_by_type: dict[str, FewshotSample] = {}
    for slot in schedule:
        source_by_type.setdefault(slot.defect_type, slot.source)
    real_references = {
        defect_type: _real_reference_crop(
            pixels=source_by_type[defect_type].pixels,
            ground_truth=source_by_type[defect_type].mask,
            padding=32,
            size=context.experiment.image_size,
        )
        for defect_type in context.experiment.defect_types
    }
    final_y_by_type: dict[str, list[np.ndarray]] = {
        defect_type: [] for defect_type in context.experiment.defect_types
    }
    ground_truth_subset_of_proposal = True
    final_subset_of_proposal = True
    for slot in schedule:
        geometry = build_geometry_pair(
            slot.source,
            slot.target,
            image_size=context.experiment.image_size,
            proposal_radius=context.experiment.proposal_radius,
            support_corner_width=context.experiment.support_corner_width,
        )
        normal = _normal_rgb(geometry.target_pixels)
        proposal = geometry.proposal[0].numpy().copy()
        ground_truth = geometry.ground_truth[0].numpy().copy()
        diagnostic: BackendDiagnosticOutput = backend.diagnose(
            normal,
            slot.defect_type,
            proposal,
            slot.seed,
        )
        rendered = diagnostic.rendered.copy()
        rendered[~proposal] = normal[~proposal]
        columns, final = panel_columns(
            normal=normal,
            real_reference=real_references[slot.defect_type],
            rendered=rendered,
            ground_truth=ground_truth,
            proposal=proposal,
            q=diagnostic.q,
            threshold=context.experiment.realization_threshold,
        )
        ground_truth_subset_of_proposal &= bool(
            np.all(~ground_truth.astype(bool) | proposal.astype(bool))
        )
        final_subset_of_proposal &= bool(
            np.all(~final.astype(bool) | proposal.astype(bool))
        )
        comparison = evaluate_realization(
            normal=normal,
            rendered=rendered,
            ground_truth=ground_truth,
            proposal=proposal,
            q=diagnostic.q,
            q_without_generation=diagnostic.q_without_generation,
            threshold=context.experiment.realization_threshold,
        )
        identity = (
            f"{slot.defect_type}-n{slot.normal_index:02d}-seed{slot.seed}"
        )
        sample_root = samples_root / identity
        sample_root.mkdir()
        for name in ("real_ref", "raw_r", "proposal_y", "q", "m_final", "final_y"):
            filename = "m_final_overlay.png" if name == "m_final" else f"{name}.png"
            Image.fromarray(columns[name]).save(sample_root / filename)
        Image.fromarray(proposal.astype(np.uint8) * 255).save(
            sample_root / "proposal.png"
        )
        Image.fromarray(ground_truth.astype(np.uint8) * 255).save(
            sample_root / "transferred_gt.png"
        )
        Image.fromarray(final.astype(np.uint8) * 255).save(
            sample_root / "m_final.png"
        )
        record: dict[str, object] = {
            "identity": identity,
            "defect_type": slot.defect_type,
            "source_id": slot.source.sample_id,
            "target_id": slot.target.sample_id,
            "seed": slot.seed,
            "transfer_retained_fraction": geometry.transfer_retained_fraction,
            "transfer_initial_retained_fraction": (
                geometry.transfer_initial_retained_fraction
            ),
            "transfer_recentered": geometry.transfer_recentered,
            "transfer_row_shift": geometry.transfer_row_shift,
            "transfer_column_shift": geometry.transfer_column_shift,
            "unet_feature_ablation_l1": float(
                np.abs(
                    diagnostic.q[proposal]
                    - diagnostic.q_without_unet_feature[proposal]
                ).mean()
            ),
            **asdict(comparison),
        }
        records.append(record)
        contact_rows[slot.defect_type].append(
            (f"n{slot.normal_index}/s{slot.seed}", columns)
        )
        final_y_by_type[slot.defect_type].append(columns["final_y"])
        seed_groups[(slot.defect_type, slot.target.sample_id)].append(
            (rendered, diagnostic.q, proposal)
        )
    records_path = panel_root / "records.jsonl"
    with records_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    for defect_type in context.experiment.defect_types:
        _save_contact_sheet(
            panel_root / f"panel_{defect_type}.png",
            contact_rows[defect_type],
        )
    save_blind_final_y_panel(
        panel_root / "blind" / "review.png",
        key_path=panel_root / "blind_mapping.json",
        final_y_by_type=final_y_by_type,
        defect_types=context.experiment.defect_types,
        seed=context.experiment.blind_panel_seed,
        tile_size=context.experiment.image_size,
    )

    seed_records: list[dict[str, object]] = []
    for (defect_type, target_id), values in sorted(seed_groups.items()):
        if len(values) != 2:
            raise RuntimeError("fixed panel seed group must contain exactly two outputs")
        first, second = values
        response = evaluate_seed_response(
            first_rendered=first[0],
            second_rendered=second[0],
            first_q=first[1],
            second_q=second[1],
            proposal=first[2],
            threshold=context.experiment.realization_threshold,
        )
        seed_records.append(
            {
                "defect_type": defect_type,
                "target_id": target_id,
                **asdict(response),
            }
        )
    with (panel_root / "seed_response.jsonl").open("x", encoding="utf-8") as handle:
        for record in seed_records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    mean_final_dice = sum(float(record["final_dice"]) for record in records) / len(records)
    mean_proposal_dice = sum(float(record["proposal_dice"]) for record in records) / len(
        records
    )
    per_type_mask_metrics = {
        defect_type: {
            "sample_count": len(type_records),
            "mean_final_dice": sum(
                float(record["final_dice"]) for record in type_records
            )
            / len(type_records),
            "mean_proposal_dice": sum(
                float(record["proposal_dice"]) for record in type_records
            )
            / len(type_records),
            "mean_dice_gain_over_proposal": sum(
                float(record["dice_gain_over_proposal"]) for record in type_records
            )
            / len(type_records),
        }
        for defect_type in context.experiment.defect_types
        for type_records in (
            [record for record in records if record["defect_type"] == defect_type],
        )
    }
    summary: dict[str, object] = {
        "schema_version": "v06_b_fixed_panel_v1",
        "sample_count": len(records),
        "defect_types": list(context.experiment.defect_types),
        "normal_count": context.experiment.panel_normal_count,
        "seeds": list(context.experiment.panel_seeds),
        "mean_final_dice": mean_final_dice,
        "mean_proposal_dice": mean_proposal_dice,
        "mean_dice_gain_over_proposal": mean_final_dice - mean_proposal_dice,
        "mean_transfer_retained_fraction": sum(
            float(record["transfer_retained_fraction"]) for record in records
        )
        / len(records),
        "mean_transfer_initial_retained_fraction": sum(
            float(record["transfer_initial_retained_fraction"])
            for record in records
        )
        / len(records),
        "min_transfer_retained_fraction": min(
            float(record["transfer_retained_fraction"]) for record in records
        ),
        "transfer_recentered_fraction": sum(
            bool(record["transfer_recentered"]) for record in records
        )
        / len(records),
        "mean_q_ablation_l1": sum(float(record["q_ablation_l1"]) for record in records)
        / len(records),
        "mean_unet_feature_ablation_l1": sum(
            float(record["unet_feature_ablation_l1"]) for record in records
        )
        / len(records),
        "generation_dependent_fraction": sum(
            bool(record["generation_dependent"]) for record in records
        )
        / len(records),
        "all_outside_final_rgb_exact": all(
            bool(record["outside_final_rgb_exact"]) for record in records
        ),
        "per_type_mask_metrics": per_type_mask_metrics,
        "subset_contract": {
            "ground_truth_subset_of_proposal": ground_truth_subset_of_proposal,
            "final_subset_of_proposal": final_subset_of_proposal,
        },
        "mask_cleaner_than_proposal": mean_final_dice > mean_proposal_dice,
        "visual_review_required": True,
        "visible_defects_passed": None,
    }
    write_json_atomic(panel_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


__all__ = [
    "PanelSlot",
    "build_fixed_panel_schedule",
    "panel_columns",
    "save_blind_final_y_panel",
    "run_v06_b_panel",
]

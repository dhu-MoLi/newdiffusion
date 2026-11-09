from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p1_v1.dataset import TrainingIndex, TrainingItem, load_training_index
from cadiff.p2_a0.fewshot import FewshotSample, FewshotSplit, load_fewshot_split
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_b2.config import P2B2Config
from cadiff.provenance import write_json_atomic

from .config import V06BMaskFirstConfig, is_cable_scheme2
from .dataset import build_geometry_pair
from .evaluation import evaluate_mask_first_diagnostic
from .exact_renderer import ExactMaskRenderer
from .inference import MaskFirstCausalRenderer
from .mask_first_checkpoint import (
    ResolvedMaskFirstInference,
    load_mask_first_checkpoint,
)
from .prompts import V06BTypePrompts
from .mask_first_runner import (
    MaskFirstV06BContext,
    bind_mask_first_renderer,
    load_mask_first_source_prompts,
    mask_first_data_authority_model,
    validate_mask_first_source_binding,
)
from .planner import PlannerNetwork, RealizationPlanner
from .renderer import compose_causal
from .source import (
    SourceRendererManifest,
    load_fresh_p2_b2_renderer,
    load_source_renderer_manifest,
)


def mask_first_source_renderer_category(config: V06BMaskFirstConfig) -> str:
    return "cable" if is_cable_scheme2(config) else "hazelnut"


_PANEL_COLUMNS = (
    "normal",
    "proposal",
    "m_final",
    "raw_r",
    "final_y",
    "proposal_as_label",
    "real_anomaly_reference",
)


@dataclass(frozen=True)
class MaskFirstPanelPreflight:
    context: MaskFirstV06BContext
    source_renderer: SourceRendererManifest
    split: FewshotSplit
    training_index: TrainingIndex


@dataclass(frozen=True)
class MaskFirstPanelRuntime:
    module_b: MaskFirstCausalRenderer
    inference: ResolvedMaskFirstInference


@dataclass(frozen=True, eq=False)
class _PanelSlot:
    defect_type: str
    source: FewshotSample
    target: TrainingItem
    normal_index: int
    seed: int


def _build_fixed_panel_schedule(
    *,
    sources: tuple[FewshotSample, ...],
    targets: tuple[TrainingItem, ...],
    defect_types: tuple[str, ...],
    normal_count: int,
    seeds: tuple[int, ...],
) -> tuple[_PanelSlot, ...]:
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
                    if source.role == "fewshot_train" and source.defect_type == defect_type
                ),
                key=lambda source: source.image_relpath,
            )
        )
        if not candidates:
            raise ValueError("every panel defect type needs a fewshot_train source")
        source_by_type[defect_type] = candidates[0]
    ordered_targets = tuple(sorted(targets, key=lambda target: target.image_relpath))
    if any(
        target.role != "normal_train" or target.defect_type != "good" for target in ordered_targets
    ):
        raise ValueError("panel targets must be normal_train/good")
    indices = (
        (0,)
        if normal_count == 1
        else tuple(
            round(index * (len(ordered_targets) - 1) / (normal_count - 1))
            for index in range(normal_count)
        )
    )
    selected = tuple(ordered_targets[index] for index in indices)
    return tuple(
        _PanelSlot(
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
        max(int(rows.max() - rows.min() + 1), int(columns.max() - columns.min() + 1)) + 2 * padding,
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


def _save_blind_final_y_panel(
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


def _rgb(value: object, *, name: str, shape: tuple[int, int, int] | None = None) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 3
        or value.shape[2] != 3
        or value.dtype != np.uint8
        or (shape is not None and value.shape != shape)
    ):
        raise ValueError(f"{name} must be a matching uint8 HxWx3 array")
    return np.ascontiguousarray(value)


def _mask(value: object, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or not np.isin(value, (0, 1)).all()
    ):
        raise ValueError(f"{name} must be a matching binary HxW array")
    return np.ascontiguousarray(value, dtype=bool)


def mask_first_panel_columns(
    *,
    normal: np.ndarray,
    proposal: np.ndarray,
    m_final: np.ndarray,
    raw_r: np.ndarray,
    real_reference: np.ndarray,
) -> dict[str, np.ndarray]:
    normal_array = _rgb(normal, name="normal")
    raw_array = _rgb(raw_r, name="raw_r", shape=normal_array.shape)
    reference = _rgb(
        real_reference,
        name="real_reference",
        shape=normal_array.shape,
    )
    proposal_array = _mask(
        proposal,
        name="proposal",
        shape=normal_array.shape[:2],
    )
    final_array = _mask(
        m_final,
        name="m_final",
        shape=normal_array.shape[:2],
    )
    if (final_array & ~proposal_array).any():
        raise ValueError("m_final must be a subset of proposal")
    final_y = compose_causal(normal_array, raw_array, final_array)
    proposal_as_label = compose_causal(normal_array, raw_array, proposal_array)
    return {
        "normal": normal_array.copy(),
        "proposal": proposal_array.astype(np.uint8) * 255,
        "m_final": final_array.astype(np.uint8) * 255,
        "raw_r": raw_array.copy(),
        "final_y": final_y,
        "proposal_as_label": proposal_as_label,
        "real_anomaly_reference": reference.copy(),
    }


def validate_mask_first_panel_preflight(
    context: MaskFirstV06BContext,
) -> MaskFirstPanelPreflight:
    if not isinstance(context, MaskFirstV06BContext):
        raise ValueError("context must be a MaskFirstV06BContext")
    config = context.experiment
    if (
        not isinstance(config, V06BMaskFirstConfig)
        or config.schema_version != "v06_b_mask_first_v1"
    ):
        raise ValueError("experiment must be the mask-first Module-B schema")
    if not isinstance(context.source_model, P2A1Config):
        raise ValueError("source_model must be a P2A1Config")
    if not isinstance(context.source_prompts, (P2B2Config, V06BTypePrompts)):
        raise ValueError("source_prompts must be P2B2Config or V06BTypePrompts")
    if not isinstance(context.cwd, Path) or not context.cwd.is_dir():
        raise ValueError("cwd must be an existing directory")
    if not isinstance(context.manifest, Path) or not isinstance(context.output, Path):
        raise ValueError("manifest and output must be pathlib.Path values")

    runs_root = context.paths.runs_root.resolve(strict=True)
    expected_output = (runs_root / config.output_relpath).resolve(strict=False)
    try:
        expected_output.relative_to(runs_root)
    except ValueError as error:
        raise ValueError("output_relpath escapes runs_root") from error
    if context.output.resolve(strict=False) != expected_output:
        raise ValueError("output must exactly match runs_root/output_relpath")
    if context.output.is_symlink() or not context.output.is_dir():
        raise ValueError("mask-first training output must be a regular directory")
    checkpoint = context.output / "checkpoint"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("mask-first checkpoint must be a regular directory")
    panel_root = context.output / "panel"
    if panel_root.exists() or panel_root.is_symlink():
        raise FileExistsError("fixed mask-first panel output already exists")

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

    source_renderer = load_source_renderer_manifest(
        context.cwd / config.source_renderer_manifest,
        runs_root=runs_root,
        category=mask_first_source_renderer_category(config),
    )
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
        raise ValueError("panel reference authority must be unique and fewshot_train only")
    if (
        training_index.category != config.category
        or len(training_index.normals) < config.panel_normal_count
        or any(
            target.role != "normal_train" or target.defect_type != "good"
            for target in training_index.normals
        )
    ):
        raise ValueError("panel target authority must be normal_train/good only")
    return MaskFirstPanelPreflight(
        context=context,
        source_renderer=source_renderer,
        split=split,
        training_index=training_index,
    )


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _load_panel_runtime(
    preflight: MaskFirstPanelPreflight,
    device: torch.device,
) -> MaskFirstPanelRuntime:
    context = preflight.context
    config = context.experiment
    planner_network = PlannerNetwork(
        defect_types=config.defect_types,
        base_channels=config.planner_base_channels,
        type_embedding_dim=config.planner_type_embedding_dim,
    )
    components = load_frozen_components(
        context.paths.model_root,
        device,
        torch.float16,
    )
    prompts = {
        defect_type: encode_prompt(
            components,
            context.source_prompts.prompts[defect_type],
            device,
        )
        for defect_type in config.defect_types
    }
    fresh = load_fresh_p2_b2_renderer(
        manifest_path=context.cwd / config.source_renderer_manifest,
        runs_root=context.paths.runs_root,
        category=mask_first_source_renderer_category(config),
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
        raise RuntimeError("authenticated P2-B2 source changed before panel inference")
    loaded = load_mask_first_checkpoint(
        context.output / "checkpoint",
        planner=planner_network,
        adapter=fresh.adapter,
        conditioner=fresh.conditioner,
        config=config,
        source=fresh.source,
    )
    if loaded.inference.ddim_steps != config.ddim_steps:
        raise RuntimeError("checkpoint DDIM steps differ from the configured full rollout")
    fresh.model.eval()
    planner = RealizationPlanner(
        network=planner_network,
        defect_types=config.defect_types,
        threshold=loaded.inference.realization_threshold,
        device=device,
    )
    renderer = ExactMaskRenderer(
        components=components,
        model=fresh.model,
        prompts=prompts,
        config=config,
        device=device,
    )
    return MaskFirstPanelRuntime(
        module_b=MaskFirstCausalRenderer(
            planner=planner,
            renderer=renderer,
            defect_types=config.defect_types,
        ),
        inference=loaded.inference,
    )


def _thumbnail(image: np.ndarray, *, size: int) -> Image.Image:
    return (
        Image.fromarray(image)
        .convert("RGB")
        .resize(
            (size, size),
            Image.Resampling.NEAREST if image.ndim == 2 else Image.Resampling.LANCZOS,
        )
    )


def _save_contact_sheet(
    path: Path,
    rows: list[tuple[str, dict[str, np.ndarray]]],
) -> None:
    tile_size = 128
    header_height = 24
    canvas = Image.new(
        "RGB",
        (len(_PANEL_COLUMNS) * tile_size, header_height + len(rows) * tile_size),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(_PANEL_COLUMNS):
        draw.text((column * tile_size + 4, 5), name, fill="white")
    for row_index, (label, columns) in enumerate(rows):
        if tuple(columns) != _PANEL_COLUMNS:
            raise RuntimeError("mask-first contact sheet columns changed")
        top = header_height + row_index * tile_size
        for column_index, name in enumerate(_PANEL_COLUMNS):
            tile = _thumbnail(columns[name], size=tile_size)
            if column_index == 0:
                ImageDraw.Draw(tile).text((4, 4), label, fill="white")
            canvas.paste(tile, (column_index * tile_size, top))
    canvas.save(path)


def _per_type_mask_metrics(
    records: list[dict[str, object]],
    defect_types: tuple[str, ...],
) -> dict[str, object]:
    def mean(
        selected_records: list[dict[str, object]],
        section: str,
        metric: str,
    ) -> float:
        values: list[float] = []
        for record in selected_records:
            mask_metrics = record["mask_metrics"]
            if not isinstance(mask_metrics, dict):
                raise RuntimeError("mask metric record must be a mapping")
            comparison = mask_metrics.get(section)
            if not isinstance(comparison, dict):
                raise RuntimeError("mask comparison record must be a mapping")
            value = comparison.get(metric)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise RuntimeError("mask comparison value must be numeric")
            values.append(float(value))
        return sum(values) / len(values)

    result: dict[str, object] = {}
    for defect_type in defect_types:
        selected = [record for record in records if record["defect_type"] == defect_type]
        if not selected:
            raise RuntimeError("every configured type must appear in panel diagnostics")

        result[defect_type] = {
            "diagnostic_only": True,
            "sample_count": len(selected),
            "m_final_vs_transferred_gt": {
                "mean_dice": mean(selected, "m_final_vs_transferred_gt", "dice"),
                "mean_iou": mean(selected, "m_final_vs_transferred_gt", "iou"),
            },
            "proposal_vs_transferred_gt": {
                "mean_dice": mean(selected, "proposal_vs_transferred_gt", "dice"),
                "mean_iou": mean(selected, "proposal_vs_transferred_gt", "iou"),
            },
        }
    return result


def run_mask_first_panel(
    context: MaskFirstV06BContext,
) -> dict[str, object]:
    preflight = validate_mask_first_panel_preflight(context)
    config = context.experiment
    schedule = _build_fixed_panel_schedule(
        sources=preflight.split.train,
        targets=preflight.training_index.normals,
        defect_types=config.defect_types,
        normal_count=config.panel_normal_count,
        seeds=config.panel_seeds,
    )
    if len(schedule) != 32:
        raise RuntimeError("fixed mask-first panel must contain exactly 32 identities")
    runtime = _load_panel_runtime(preflight, _require_cuda())
    panel_root = context.output / "panel"
    samples_root = panel_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=False)

    source_by_type = {
        defect_type: next(slot.source for slot in schedule if slot.defect_type == defect_type)
        for defect_type in config.defect_types
    }
    real_references = {
        defect_type: _real_reference_crop(
            pixels=source_by_type[defect_type].pixels,
            ground_truth=source_by_type[defect_type].mask,
            padding=32,
            size=config.image_size,
        )
        for defect_type in config.defect_types
    }
    contact_rows: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = defaultdict(list)
    final_y_by_type: dict[str, list[np.ndarray]] = {
        defect_type: [] for defect_type in config.defect_types
    }
    records: list[dict[str, object]] = []
    for slot in schedule:
        geometry = build_geometry_pair(
            slot.source,
            slot.target,
            image_size=config.image_size,
            proposal_radius=config.proposal_radius,
            support_corner_width=config.support_corner_width,
        )
        if (
            geometry.source_id != slot.source.sample_id
            or geometry.target_id != slot.target.sample_id
            or geometry.defect_type != slot.defect_type
        ):
            raise RuntimeError("panel geometry differs from its fixed identity")
        normal = _normal_rgb(geometry.target_pixels)
        proposal = np.ascontiguousarray(geometry.proposal[0].cpu().numpy(), dtype=bool)
        transferred_gt = np.ascontiguousarray(
            geometry.ground_truth[0].cpu().numpy(),
            dtype=bool,
        )
        raw_r, m_final = runtime.module_b(
            normal,
            slot.defect_type,
            proposal,
            slot.seed,
        )
        columns = mask_first_panel_columns(
            normal=normal,
            proposal=proposal,
            m_final=m_final,
            raw_r=raw_r,
            real_reference=real_references[slot.defect_type],
        )
        final_mask = columns["m_final"] > 0
        diagnostic = evaluate_mask_first_diagnostic(
            normal=normal,
            final_y=columns["final_y"],
            transferred_ground_truth=transferred_gt,
            proposal=proposal,
            m_final=final_mask,
        )
        causal_composition = bool(
            diagnostic.outside_rgb_byte_exact
            and np.array_equal(columns["final_y"][final_mask], raw_r[final_mask])
        )
        configured_steps = config.ddim_steps
        executed_steps = runtime.inference.ddim_steps
        candidate_count = 1
        normal_input = slot.target.role == "normal_train" and slot.target.defect_type == "good"
        predicted_m_final = True
        eligible = bool(
            normal_input
            and predicted_m_final
            and diagnostic.m_final_subset_of_proposal
            and diagnostic.outside_rgb_byte_exact
            and executed_steps == configured_steps
            and candidate_count == 1
            and causal_composition
        )
        identity = f"{slot.defect_type}-n{slot.normal_index:02d}-seed{slot.seed}"
        sample_root = samples_root / identity
        sample_root.mkdir(exist_ok=False)
        for name in _PANEL_COLUMNS:
            Image.fromarray(columns[name]).save(sample_root / f"{name}.png")
        record: dict[str, object] = {
            "identity": identity,
            "defect_type": slot.defect_type,
            "source_id": slot.source.sample_id,
            "target_id": slot.target.sample_id,
            "normal_index": slot.normal_index,
            "seed": slot.seed,
            "normal_input": normal_input,
            "predicted_m_final": predicted_m_final,
            "configured_ddim_steps": configured_steps,
            "executed_ddim_steps": executed_steps,
            "candidate_count": candidate_count,
            "causal_composition": causal_composition,
            "m_final_subset_of_proposal": diagnostic.m_final_subset_of_proposal,
            "outside_rgb_byte_exact": diagnostic.outside_rgb_byte_exact,
            "eligible_visual_evidence": eligible,
            "mask_metrics": {
                "diagnostic_only": True,
                "m_final_vs_transferred_gt": {
                    "dice": diagnostic.m_final_dice,
                    "iou": diagnostic.m_final_iou,
                },
                "proposal_vs_transferred_gt": {
                    "dice": diagnostic.proposal_dice,
                    "iou": diagnostic.proposal_iou,
                },
                "dice_gain_over_proposal": diagnostic.dice_gain_over_proposal,
            },
        }
        records.append(record)
        contact_rows[slot.defect_type].append((f"n{slot.normal_index}/s{slot.seed}", columns))
        final_y_by_type[slot.defect_type].append(columns["final_y"])

    with (panel_root / "records.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    mask_diagnostics: dict[str, object] = {
        "schema_version": "v06_b_mask_diagnostics_v1",
        "diagnostic_only": True,
        "sample_count": len(records),
        "per_type_mask_metrics": _per_type_mask_metrics(records, config.defect_types),
    }
    write_json_atomic(panel_root / "mask_diagnostics.json", mask_diagnostics)
    for defect_type in config.defect_types:
        _save_contact_sheet(
            panel_root / f"panel_{defect_type}.png",
            contact_rows[defect_type],
        )
    _save_blind_final_y_panel(
        panel_root / "blind" / "review.png",
        key_path=panel_root / "blind_mapping.json",
        final_y_by_type=final_y_by_type,
        defect_types=config.defect_types,
        seed=config.blind_panel_seed,
        tile_size=config.image_size,
    )
    summary: dict[str, object] = {
        "schema_version": "v06_b_mask_first_fixed_panel_v1",
        "sample_count": len(records),
        "defect_types": list(config.defect_types),
        "normal_count": config.panel_normal_count,
        "seeds": list(config.panel_seeds),
        "eligible_visual_evidence_count": sum(
            record["eligible_visual_evidence"] is True for record in records
        ),
        "engineering_contracts": {
            "m_final_subset_of_proposal": all(
                record["m_final_subset_of_proposal"] is True for record in records
            ),
            "outside_rgb_byte_exact": all(
                record["outside_rgb_byte_exact"] is True for record in records
            ),
        },
        "visual_review_status": "awaiting_human_review",
        "visible_defects_passed": None,
    }
    write_json_atomic(panel_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


__all__ = [
    "mask_first_panel_columns",
    "mask_first_source_renderer_category",
    "run_mask_first_panel",
]

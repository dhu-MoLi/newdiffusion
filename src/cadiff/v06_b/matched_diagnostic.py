from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image, ImageDraw
from torch import Tensor, nn

from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.dataset import TrainingItem
from cadiff.p2_a0.fewshot import FewshotSample
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.runner import _load_p2_a1_checkpoint
from cadiff.provenance import sha256_file, write_json_atomic

from .checkpoint import load_v06_b_checkpoint
from .dataset import build_geometry_pair
from .panel import PanelSlot, _normal_rgb, build_fixed_panel_schedule
from .realization import RealizationHead
from .renderer import compose_causal
from .rollout import truncated_denoise_rollout
from .runner import (
    V06BContext,
    V06BPreflight,
    _load_runtime_assets,
    _require_cuda,
    validate_v06_b_preflight,
)

RendererRole = Literal["p2_b2", "attempt2"]
ConditioningMask = Literal["ground_truth", "proposal"]


@dataclass(frozen=True)
class MatchedVariant:
    label: str
    renderer: RendererRole
    conditioning_mask: ConditioningMask
    ddim_steps: int
    has_realization_head: bool


MATCHED_VARIANTS = (
    MatchedVariant("A", "p2_b2", "ground_truth", 50, False),
    MatchedVariant("B", "p2_b2", "proposal", 50, False),
    MatchedVariant("C", "p2_b2", "proposal", 20, False),
    MatchedVariant("D", "attempt2", "proposal", 20, True),
)


@dataclass(frozen=True, eq=False)
class MatchedRolloutInputs:
    normal_latent: Tensor
    prompt_embeds: Tensor
    noise: Tensor
    binary_mask: Tensor
    context: Tensor


@dataclass(frozen=True)
class _InferenceRuntime:
    components: Any
    prompts: Mapping[str, Tensor]
    p2_b2_model: P2InfillNoisePredictor
    attempt2_model: P2InfillNoisePredictor
    attempt2_head: RealizationHead
    attempt2_metadata: Mapping[str, Any]


def build_matched_schedule(
    *,
    sources: tuple[FewshotSample, ...],
    targets: tuple[TrainingItem, ...],
    defect_types: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[PanelSlot, ...]:
    """Freeze the diagnostic to n00 and both pre-existing panel seeds."""

    if len(defect_types) != 4 or len(seeds) != 2:
        raise ValueError("matched diagnostic requires four types and two seeds")
    schedule = build_fixed_panel_schedule(
        sources=sources,
        targets=targets,
        defect_types=defect_types,
        normal_count=1,
        seeds=seeds,
    )
    if len(schedule) != 8 or any(slot.normal_index != 0 for slot in schedule):
        raise RuntimeError("matched diagnostic schedule must be n00 times two seeds")
    return schedule


def _identity_record(slot: PanelSlot) -> dict[str, object]:
    return {
        "identity": f"{slot.defect_type}-n{slot.normal_index:02d}-seed{slot.seed}",
        "defect_type": slot.defect_type,
        "source_id": slot.source.sample_id,
        "target_id": slot.target.sample_id,
        "seed": slot.seed,
    }


def validate_attempt2_identity_binding(
    schedule: Sequence[PanelSlot],
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    fields = ("identity", "defect_type", "source_id", "target_id", "seed")
    expected = tuple(_identity_record(slot) for slot in schedule)
    try:
        actual = tuple({field: record[field] for field in fields} for record in records)
    except (KeyError, TypeError) as error:
        raise ValueError("attempt2 identity records have an invalid schema") from error
    if actual != expected:
        raise ValueError("attempt2 records do not match the exact fixed identities")
    return expected


def validate_strict_pairing(inputs: Mapping[str, MatchedRolloutInputs]) -> None:
    """Require identity-shared tensors, with only A allowed a different spatial mask."""

    if tuple(inputs) != tuple(variant.label for variant in MATCHED_VARIANTS):
        raise ValueError("matched rollout inputs must be ordered A, B, C, D")
    values = tuple(inputs.values())
    if any(value.normal_latent is not values[0].normal_latent for value in values[1:]):
        raise ValueError("A-D must use the same normal latent object")
    if any(value.prompt_embeds is not values[0].prompt_embeds for value in values[1:]):
        raise ValueError("A-D must use the same prompt object")
    if any(value.noise is not values[0].noise for value in values[1:]):
        raise ValueError("A-D must use the same noise object")
    proposal_values = values[1:]
    if any(
        value.binary_mask is not proposal_values[0].binary_mask
        for value in proposal_values[1:]
    ):
        raise ValueError("B-D must use the same proposal-mask object")
    if any(value.context is not proposal_values[0].context for value in proposal_values[1:]):
        raise ValueError("B-D must use the same proposal-context object")


def _binary_mask(value: object, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must match the RGB spatial shape")
    if not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return value.astype(bool, copy=True)


def matched_columns(
    *,
    normal: np.ndarray,
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    proposal: np.ndarray,
    m_final: np.ndarray | None,
) -> dict[str, np.ndarray | None]:
    if (
        not isinstance(normal, np.ndarray)
        or normal.dtype != np.uint8
        or normal.ndim != 3
        or normal.shape[2] != 3
        or not isinstance(rendered, np.ndarray)
        or rendered.dtype != np.uint8
        or rendered.shape != normal.shape
    ):
        raise ValueError("normal and rendered must be matching uint8 RGB arrays")
    shape = normal.shape[:2]
    ground_truth_mask = _binary_mask(ground_truth, name="ground_truth", shape=shape)
    proposal_mask = _binary_mask(proposal, name="proposal", shape=shape)
    if (ground_truth_mask & ~proposal_mask).any():
        raise ValueError("ground_truth must be a subset of proposal")
    final_composite: np.ndarray | None = None
    if m_final is not None:
        final_mask = _binary_mask(m_final, name="m_final", shape=shape)
        if (final_mask & ~proposal_mask).any():
            raise ValueError("m_final must be a subset of proposal")
        final_composite = compose_causal(normal, rendered, final_mask)
    return {
        "normal": normal.copy(),
        "raw_r": rendered.copy(),
        "gt_composite": compose_causal(normal, rendered, ground_truth_mask),
        "proposal_composite": compose_causal(normal, rendered, proposal_mask),
        "m_final_composite": final_composite,
    }


def _freeze_eval(module: nn.Module) -> None:
    module.eval()
    if module.training:
        raise RuntimeError("inference module did not enter eval mode")
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise RuntimeError("inference module already has parameter gradients")


def _load_inference_runtime(
    preflight: V06BPreflight,
    *,
    attempt2_checkpoint: Path,
    device: torch.device,
) -> _InferenceRuntime:
    config = preflight.context.experiment
    if config.schema_version != "v06_b_causal_renderer_v1":
        raise ValueError("matched diagnostic must use the frozen attempt2 v1 configuration")
    expected_checkpoint = (
        preflight.context.paths.runs_root / config.output_relpath / "checkpoint"
    ).resolve(strict=True)
    if (
        attempt2_checkpoint.is_symlink()
        or attempt2_checkpoint.resolve(strict=True) != expected_checkpoint
    ):
        raise ValueError("attempt2 checkpoint must match the explicit v1 output checkpoint")

    assets = _load_runtime_assets(preflight, device)
    _, source_conditioner, source_adapter, _ = _load_p2_a1_checkpoint(
        preflight.source_renderer.checkpoint,
        config=preflight.context.source_model,
        device=device,
    )
    source_model = P2InfillNoisePredictor(
        assets.components.unet,
        source_conditioner,
        source_adapter,
    )

    _, attempt2_conditioner, attempt2_adapter, _ = _load_p2_a1_checkpoint(
        preflight.source_renderer.checkpoint,
        config=preflight.context.source_model,
        device=device,
    )
    attempt2_head = RealizationHead(
        feature_channels=config.feature_channels,
        hidden_channels=config.head_hidden_channels,
    ).to(device=device, dtype=torch.float32)
    attempt2_metadata = load_v06_b_checkpoint(
        expected_checkpoint,
        adapter=attempt2_adapter,
        conditioner=attempt2_conditioner,
        realization_head=attempt2_head,
        config=config,
        source_renderer_manifest_sha256=preflight.source_renderer.manifest_sha256,
        expected_step=config.final_step,
    )
    attempt2_model = P2InfillNoisePredictor(
        assets.components.unet,
        attempt2_conditioner,
        attempt2_adapter,
    )
    for module in (source_model, attempt2_model, attempt2_head):
        _freeze_eval(module)
    return _InferenceRuntime(
        components=assets.components,
        prompts=assets.prompts,
        p2_b2_model=source_model,
        attempt2_model=attempt2_model,
        attempt2_head=attempt2_head,
        attempt2_metadata=attempt2_metadata,
    )


def _encode_normal(vae: nn.Module, normal: np.ndarray, device: torch.device) -> Tensor:
    parameters = tuple(
        parameter for parameter in vae.parameters() if parameter.is_floating_point()
    )
    if not parameters:
        raise ValueError("VAE must expose floating parameters")
    pixels = (
        torch.from_numpy(normal.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=parameters[0].dtype)
        .div(127.5)
        .sub(1.0)
    )
    with torch.inference_mode():
        encoded = encode_mode(vae, pixels)
    result = encoded.detach().clone()
    if result.requires_grad or result.is_inference():
        raise RuntimeError("normal latent must be an ordinary detached tensor")
    return result


def _image_mask(mask: np.ndarray, device: torch.device) -> Tensor:
    return (
        torch.from_numpy(mask.astype(np.float32, copy=True))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device)
    )


def _render_uint8(vae: nn.Module, latent: Tensor) -> np.ndarray:
    decoded = decode_latent(vae, latent)
    return (
        decoded[0]
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .copy()
    )


def _na_tile(shape: tuple[int, int, int]) -> np.ndarray:
    tile = Image.new("RGB", (shape[1], shape[0]), (28, 28, 28))
    draw = ImageDraw.Draw(tile)
    draw.text((shape[1] // 2 - 12, shape[0] // 2 - 6), "N/A", fill="white")
    return np.asarray(tile).copy()


def _save_contact_sheet(
    path: Path,
    rows: list[tuple[str, Mapping[str, np.ndarray | None]]],
    *,
    tile_size: int,
) -> None:
    names = (
        "normal",
        "decoded raw r",
        "GT-mask composite",
        "proposal composite",
        "m_final composite",
    )
    keys = (
        "normal",
        "raw_r",
        "gt_composite",
        "proposal_composite",
        "m_final_composite",
    )
    header = 28
    canvas = Image.new(
        "RGB", (len(keys) * tile_size, header + len(rows) * tile_size), "black"
    )
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(names):
        draw.text((column * tile_size + 4, 7), name, fill="white")
    for row_index, (label, columns) in enumerate(rows):
        top = header + row_index * tile_size
        for column_index, key in enumerate(keys):
            value = columns[key]
            if value is None:
                value = _na_tile(columns["normal"].shape)  # type: ignore[union-attr]
            tile = Image.fromarray(value).resize(
                (tile_size, tile_size), Image.Resampling.LANCZOS
            )
            if column_index == 0:
                ImageDraw.Draw(tile).text(
                    (4, 4),
                    label,
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
            canvas.paste(tile, (column_index * tile_size, top))
    canvas.save(path)


def _existing_attempt2_image(
    checkpoint: Path, identity: str, filename: str
) -> np.ndarray:
    path = checkpoint.parent / "panel" / "samples" / identity / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"attempt2 regression image is unavailable: {path}")
    return np.asarray(Image.open(path).convert("RGB")).copy()


def _bind_attempt2_panel(
    checkpoint: Path,
    schedule: Sequence[PanelSlot],
) -> tuple[tuple[dict[str, object], ...], str]:
    records_path = checkpoint.parent / "panel" / "records.jsonl"
    if records_path.is_symlink() or not records_path.is_file():
        raise ValueError("attempt2 panel records must be a regular file")
    try:
        all_records = tuple(
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("attempt2 panel records must be valid UTF-8 JSONL") from error
    expected_names = {_identity_record(slot)["identity"] for slot in schedule}
    fixed_records = tuple(
        record
        for record in all_records
        if isinstance(record, Mapping) and record.get("identity") in expected_names
    )
    identities = validate_attempt2_identity_binding(schedule, fixed_records)
    for identity in identities:
        name = str(identity["identity"])
        for filename in ("raw_r.png", "final_y.png"):
            path = checkpoint.parent / "panel" / "samples" / name / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"attempt2 regression image is unavailable: {path}")
    return identities, sha256_file(records_path)


def run_matched_diagnostic(
    context: V06BContext,
    *,
    attempt2_checkpoint: Path,
) -> dict[str, Any]:
    """Run the fixed eight-identity A-D inference comparison without training."""

    preflight = validate_v06_b_preflight(context)
    config = context.experiment
    if config.rollout_steps != 20:
        raise ValueError("attempt2 configuration must retain its frozen 20-step rollout")
    output = context.output
    if output.exists() or output.is_symlink():
        raise FileExistsError("matched diagnostic output already exists")
    schedule = build_matched_schedule(
        sources=preflight.split.train,
        targets=preflight.training_index.normals,
        defect_types=config.defect_types,
        seeds=config.panel_seeds,
    )
    expected_checkpoint = (
        context.paths.runs_root / config.output_relpath / "checkpoint"
    ).resolve(strict=True)
    if (
        attempt2_checkpoint.is_symlink()
        or attempt2_checkpoint.resolve(strict=True) != expected_checkpoint
    ):
        raise ValueError("attempt2 checkpoint must match the explicit v1 output checkpoint")
    frozen_identities, source_records_sha256 = _bind_attempt2_panel(
        expected_checkpoint, schedule
    )
    device = _require_cuda()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    runtime = _load_inference_runtime(
        preflight,
        attempt2_checkpoint=expected_checkpoint,
        device=device,
    )
    output.mkdir(parents=True, exist_ok=False)
    samples_root = output / "samples"
    samples_root.mkdir()
    rows_by_type: dict[
        str, list[tuple[str, Mapping[str, np.ndarray | None]]]
    ] = defaultdict(list)
    all_rows: list[tuple[str, Mapping[str, np.ndarray | None]]] = []
    records: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    normal_latents: dict[str, tuple[np.ndarray, Tensor]] = {}
    attempt2_regression_passed = True

    for slot in schedule:
        geometry = build_geometry_pair(
            slot.source,
            slot.target,
            image_size=config.image_size,
            proposal_radius=config.proposal_radius,
            support_corner_width=config.support_corner_width,
        )
        normal = _normal_rgb(geometry.target_pixels)
        cached = normal_latents.get(slot.target.sample_id)
        if cached is None:
            normal_latent = _encode_normal(runtime.components.vae, normal, device)
            normal_latents[slot.target.sample_id] = (normal.copy(), normal_latent)
        else:
            cached_normal, normal_latent = cached
            if not np.array_equal(normal, cached_normal):
                raise RuntimeError("fixed target identity produced different normal pixels")
        ground_truth = geometry.ground_truth[0].cpu().numpy().astype(bool, copy=True)
        proposal = geometry.proposal[0].cpu().numpy().astype(bool, copy=True)
        ground_truth_tensor = _image_mask(ground_truth, device)
        proposal_tensor = _image_mask(proposal, device)
        ground_truth_latent = build_latent_masks(
            ground_truth_tensor, normal_latent.shape[2:]
        ).binary
        proposal_latent = build_latent_masks(
            proposal_tensor, normal_latent.shape[2:]
        ).binary
        ground_truth_context = build_masked_context(
            normal_latent,
            ground_truth_latent,
            torch.Generator(device=device).manual_seed(slot.seed + 10_000_000),
        )
        proposal_context = build_masked_context(
            normal_latent,
            proposal_latent,
            torch.Generator(device=device).manual_seed(slot.seed + 10_000_000),
        )
        noise = torch.randn(
            normal_latent.shape,
            generator=torch.Generator(device=device).manual_seed(slot.seed),
            device=device,
            dtype=normal_latent.dtype,
        )
        prompt = runtime.prompts[slot.defect_type]
        if prompt.dtype != normal_latent.dtype:
            raise ValueError("P2-B2 prompt dtype must match normal latent")
        proposal_inputs = MatchedRolloutInputs(
            normal_latent=normal_latent,
            prompt_embeds=prompt,
            noise=noise,
            binary_mask=proposal_latent,
            context=proposal_context,
        )
        rollout_inputs = {
            "A": MatchedRolloutInputs(
                normal_latent=normal_latent,
                prompt_embeds=prompt,
                noise=noise,
                binary_mask=ground_truth_latent,
                context=ground_truth_context,
            ),
            "B": proposal_inputs,
            "C": proposal_inputs,
            "D": proposal_inputs,
        }
        validate_strict_pairing(rollout_inputs)
        identity = f"{slot.defect_type}-n{slot.normal_index:02d}-seed{slot.seed}"
        identity_root = samples_root / identity
        identity_root.mkdir()
        Image.fromarray(normal).save(identity_root / "normal.png")
        Image.fromarray(ground_truth.astype(np.uint8) * 255).save(
            identity_root / "transferred_gt.png"
        )
        Image.fromarray(proposal.astype(np.uint8) * 255).save(
            identity_root / "proposal.png"
        )
        identities.append(_identity_record(slot))

        for variant in MATCHED_VARIANTS:
            inputs = rollout_inputs[variant.label]
            model = (
                runtime.p2_b2_model
                if variant.renderer == "p2_b2"
                else runtime.attempt2_model
            )
            scheduler = DDIMScheduler.from_config(runtime.components.scheduler.config)
            rollout = truncated_denoise_rollout(
                model=model,
                scheduler=scheduler,
                normal_latent=inputs.normal_latent,
                context=inputs.context,
                binary_mask=inputs.binary_mask,
                prompt_embeds=inputs.prompt_embeds,
                defect_type=slot.defect_type,
                noise=inputs.noise,
                steps=variant.ddim_steps,
                backprop_steps=0,
            )
            rendered = _render_uint8(runtime.components.vae, rollout.latent)
            final: np.ndarray | None = None
            if variant.has_realization_head:
                with torch.no_grad():
                    q = runtime.attempt2_head(
                        generated_latent=rollout.latent,
                        normal_latent=normal_latent,
                        generation_feature=rollout.generation_feature,
                        proposal=proposal_tensor,
                    ).q
                final = proposal & (
                    q[0, 0].float().cpu().numpy() >= config.realization_threshold
                )
            columns = matched_columns(
                normal=normal,
                rendered=rendered,
                ground_truth=ground_truth,
                proposal=proposal,
                m_final=final,
            )
            variant_root = identity_root / variant.label
            variant_root.mkdir()
            for name in ("raw_r", "gt_composite", "proposal_composite"):
                value = columns[name]
                if value is None:
                    raise RuntimeError("required matched diagnostic column is absent")
                Image.fromarray(value).save(variant_root / f"{name}.png")
            if columns["m_final_composite"] is not None:
                Image.fromarray(columns["m_final_composite"]).save(
                    variant_root / "m_final_composite.png"
                )
            if variant.label == "D":
                expected_raw = _existing_attempt2_image(
                    expected_checkpoint, identity, "raw_r.png"
                )
                expected_final = _existing_attempt2_image(
                    expected_checkpoint, identity, "final_y.png"
                )
                matches = bool(
                    np.array_equal(columns["proposal_composite"], expected_raw)
                    and np.array_equal(columns["m_final_composite"], expected_final)
                )
                attempt2_regression_passed &= matches
                if not matches:
                    raise RuntimeError(
                        f"attempt2 matched regression differs for fixed identity {identity}"
                    )
            row_label = f"{slot.defect_type} s{slot.seed} / {variant.label}"
            rows_by_type[slot.defect_type].append((row_label, columns))
            all_rows.append((row_label, columns))
            records.append(
                {
                    "identity": identity,
                    "variant": variant.label,
                    "renderer": variant.renderer,
                    "conditioning_mask": variant.conditioning_mask,
                    "ddim_steps": variant.ddim_steps,
                    "has_realization_head": variant.has_realization_head,
                    "prompt": preflight.context.source_prompts.prompts[
                        slot.defect_type
                    ],
                }
            )
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("inference-only rollout created parameter gradients")

    if tuple(identities) != frozen_identities:
        raise RuntimeError("rendered identities drifted from the pre-bound attempt2 panel")
    records_path = output / "records.jsonl"
    with records_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    for defect_type in config.defect_types:
        _save_contact_sheet(
            output / f"panel_{defect_type}.png",
            rows_by_type[defect_type],
            tile_size=192,
        )
    _save_contact_sheet(output / "panel_all.png", all_rows, tile_size=128)
    summary: dict[str, Any] = {
        "schema_version": "v06_b_matched_diagnostic_v1",
        "training_performed": False,
        "candidate_selection": False,
        "identity_count": len(identities),
        "record_count": len(records),
        "identity_rule": "existing_fixed_panel_n00_times_both_frozen_seeds",
        "identities": identities,
        "source_records_sha256": source_records_sha256,
        "variants": [
            {
                "label": variant.label,
                "renderer": variant.renderer,
                "conditioning_mask": variant.conditioning_mask,
                "ddim_steps": variant.ddim_steps,
                "has_realization_head": variant.has_realization_head,
            }
            for variant in MATCHED_VARIANTS
        ],
        "strict_pairing": {
            "same_normal_type_seed_prompt_noise_across_A_D": True,
            "same_proposal_context_across_B_D": True,
            "same_spatial_transfer_across_A_D": True,
        },
        "p2_b2_checkpoint": str(preflight.source_renderer.checkpoint),
        "attempt2_checkpoint": str(expected_checkpoint),
        "attempt2_checkpoint_step": runtime.attempt2_metadata["step"],
        "attempt2_panel_byte_regression_passed": attempt2_regression_passed,
        "attempt2_panel_regression_definition": (
            "D proposal_composite equals legacy attempt2 raw_r; "
            "D m_final_composite equals legacy attempt2 final_y"
        ),
        "raw_r_definition": "full_vae_decode_before_pixel_mask_composition",
        "source_prompt_config_sha256": sha256_file(
            context.cwd / config.source_experiment_config
        ),
        "visible_defects_passed": False,
        "diagnosis": None,
        "human_interpretation_required": True,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


__all__ = [
    "MATCHED_VARIANTS",
    "MatchedRolloutInputs",
    "MatchedVariant",
    "build_matched_schedule",
    "matched_columns",
    "run_matched_diagnostic",
    "validate_attempt2_identity_binding",
    "validate_strict_pairing",
]

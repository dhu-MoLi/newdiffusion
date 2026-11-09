from __future__ import annotations

import gc
import json
import math
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cadiff.config import ProjectConfig
from cadiff.diffusion.components import FrozenComponents, load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import encode_mode
from cadiff.p1_v1.dataset import TrainingIndex, load_training_index
from cadiff.p2_a0.fewshot import (
    FewshotSplit,
    PreparedSample,
    _prepare_samples,
    load_fewshot_split,
)
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_a1.runner import _load_p2_a1_checkpoint
from cadiff.p2_b2.config import P2B2Config
from cadiff.provenance import write_json_atomic

from .checkpoint import load_v06_b_parent_checkpoint, save_v06_b_checkpoint
from .config import V06BConfig, load_v06_b_config
from .dataset import (
    GeometryPair,
    UnpairedSlot,
    build_geometry_pair,
    build_source_training_proposal,
    build_unpaired_schedule,
)
from .realization import RealizationHead
from .source import SourceRendererManifest, load_source_renderer_manifest
from .training import (
    DiffusionBatch,
    ModuleBStepMetrics,
    ProposalConditionedDiffusionBatch,
    RealizationBatch,
    build_module_b_optimizer,
    cyclic_wrong_type,
    module_b_optimizer_step,
)


@dataclass(frozen=True)
class V06BContext:
    paths: ProjectConfig
    experiment: V06BConfig
    source_model: P2A1Config
    source_prompts: P2B2Config
    manifest: Path
    output: Path
    cwd: Path


@dataclass(frozen=True)
class V06BPreflight:
    context: V06BContext
    source_renderer: SourceRendererManifest
    split: FewshotSplit
    training_index: TrainingIndex
    schedule: tuple[UnpairedSlot, ...]
    parent_config: V06BConfig | None
    parent_checkpoint: Path | None


@dataclass(frozen=True)
class RuntimeAssets:
    components: FrozenComponents
    prompts: dict[str, Tensor]


@dataclass(frozen=True)
class TrainingState:
    vocabulary_sha256: str
    conditioner: nn.Module
    adapter: nn.Module
    model: P2InfillNoisePredictor
    head: RealizationHead
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    parent_checkpoint_metadata: dict[str, Any] | None


@dataclass(frozen=True)
class PreparedProposalSample:
    sample: Any
    target_latent: Tensor
    target_image: Tensor
    context: Tensor
    binary_proposal: Tensor
    soft_proposal: Tensor
    soft_ground_truth: Tensor
    proposal: Tensor
    ground_truth: Tensor


def probe_attempt_configs(config: V06BConfig) -> tuple[V06BConfig, ...]:
    if not isinstance(config, V06BConfig):
        raise ValueError("config must be a V06BConfig")
    candidates = (
        replace(config, checkpoint_tracked_steps=False),
        replace(config, checkpoint_tracked_steps=True),
        replace(
            config,
            backprop_steps=min(2, config.backprop_steps),
            checkpoint_tracked_steps=True,
        ),
        replace(config, backprop_steps=1, checkpoint_tracked_steps=True),
        replace(
            config,
            backprop_steps=1,
            checkpoint_tracked_steps=True,
            head_hidden_channels=max(8, config.head_hidden_channels // 2),
        ),
    )
    unique: list[V06BConfig] = []
    keys: set[tuple[int, bool, int]] = set()
    for candidate in candidates:
        key = (
            candidate.backprop_steps,
            candidate.checkpoint_tracked_steps,
            candidate.head_hidden_channels,
        )
        if key not in keys:
            keys.add(key)
            unique.append(candidate)
    return tuple(unique)


def validate_v06_b_preflight(context: V06BContext) -> V06BPreflight:
    if not isinstance(context, V06BContext):
        raise ValueError("context must be a V06BContext")
    if not isinstance(context.paths, ProjectConfig):
        raise ValueError("paths must be a ProjectConfig")
    if not isinstance(context.experiment, V06BConfig):
        raise ValueError("experiment must be a V06BConfig")
    if not isinstance(context.source_model, P2A1Config):
        raise ValueError("source_model must be a P2A1Config")
    if not isinstance(context.source_prompts, P2B2Config):
        raise ValueError("source_prompts must be a P2B2Config")
    config = context.experiment
    source_model = context.source_model
    source_prompts = context.source_prompts
    if context.paths.image_size != config.image_size:
        raise ValueError("project and Module-B image_size must match")
    if (
        source_model.category != config.category
        or source_prompts.category != config.category
        or source_model.defect_types != config.defect_types
        or source_prompts.defect_types != config.defect_types
    ):
        raise ValueError("source P2-B2 lineage must match the Module-B category and vocabulary")
    if source_model.image_size != config.image_size:
        raise ValueError("source model and Module-B image_size must match")
    if source_model.type_rank != source_prompts.type_rank:
        raise ValueError("source model and prompt adapter rank must match")

    renderer_manifest = context.cwd / config.source_renderer_manifest
    source_renderer = load_source_renderer_manifest(
        renderer_manifest,
        runs_root=context.paths.runs_root,
        category=config.category,
    )
    split = load_fewshot_split(
        context.manifest,
        context.paths.dataset_root,
        source_model,
    )
    training_index = load_training_index(
        context.manifest,
        context.paths.dataset_root,
        source_model,
    )
    parent_config: V06BConfig | None = None
    parent_checkpoint: Path | None = None
    if config.schema_version == "v06_b_causal_renderer_v2":
        if config.parent_experiment_config is None or config.parent_checkpoint_relpath is None:
            raise ValueError("v2 requires explicit parent experiment and checkpoint paths")
        parent_path = context.cwd / config.parent_experiment_config
        if parent_path.is_symlink() or not parent_path.is_file():
            raise ValueError("parent experiment configuration must be a regular file")
        parent_config = load_v06_b_config(parent_path)
        root = context.paths.runs_root.resolve(strict=True)
        entry = root / config.parent_checkpoint_relpath
        if entry.is_symlink() or not entry.is_dir():
            raise ValueError("parent checkpoint must be a regular directory")
        parent_checkpoint = entry.resolve(strict=True)
        try:
            parent_checkpoint.relative_to(root)
        except ValueError as error:
            raise ValueError("parent checkpoint escapes runs_root") from error
        if config.parent_checkpoint_step != parent_config.final_step:
            raise ValueError("parent checkpoint step differs from parent configuration")
    complete_schedule = build_unpaired_schedule(
        sources=split.train,
        targets=training_index.normals,
        defect_types=config.defect_types,
        steps=config.final_step,
        seed=config.seed,
    )
    schedule = complete_schedule[config.parent_checkpoint_step :]
    if len(schedule) != config.training_steps:
        raise RuntimeError("Module-B continuation schedule has the wrong length")
    return V06BPreflight(
        context=context,
        source_renderer=source_renderer,
        split=split,
        training_index=training_index,
        schedule=schedule,
        parent_config=parent_config,
        parent_checkpoint=parent_checkpoint,
    )


def _ordinary_prompt(value: Tensor) -> Tensor:
    prompt = value.detach().clone()
    if (
        prompt.is_inference()
        or prompt.requires_grad
        or not prompt.is_floating_point()
        or not torch.isfinite(prompt).all().item()
    ):
        raise ValueError("prompt embedding must be an ordinary detached finite tensor")
    return prompt


def _load_runtime_assets(preflight: V06BPreflight, device: torch.device) -> RuntimeAssets:
    components = load_frozen_components(
        preflight.context.paths.model_root,
        device,
        torch.float16,
    )
    prompt_text = (
        preflight.context.experiment.prompts
        if preflight.context.experiment.schema_version == "v06_b_causal_renderer_v2"
        else preflight.context.source_prompts.prompts
    )
    prompts = {
        defect_type: _ordinary_prompt(
            encode_prompt(
                components,
                prompt_text[defect_type],
                device,
            )
        )
        for defect_type in preflight.context.experiment.defect_types
    }
    if tuple(prompts) != preflight.context.experiment.defect_types:
        raise RuntimeError("prompt order differs from the Module-B vocabulary")
    return RuntimeAssets(components=components, prompts=prompts)


def _new_training_state(
    preflight: V06BPreflight,
    assets: RuntimeAssets,
    config: V06BConfig,
    device: torch.device,
) -> TrainingState:
    vocabulary, conditioner, adapter, _ = _load_p2_a1_checkpoint(
        preflight.source_renderer.checkpoint,
        config=preflight.context.source_model,
        device=device,
    )
    model = P2InfillNoisePredictor(
        assets.components.unet,
        conditioner,
        adapter,
    ).train()
    head = RealizationHead(
        feature_channels=config.feature_channels,
        hidden_channels=config.head_hidden_channels,
    ).to(device=device, dtype=torch.float32)
    head.train()
    parent_checkpoint_metadata: dict[str, Any] | None = None
    if config.schema_version == "v06_b_causal_renderer_v2":
        if preflight.parent_config is None or preflight.parent_checkpoint is None:
            raise ValueError("v2 training state requires its validated parent checkpoint")
        parent_checkpoint_metadata = load_v06_b_parent_checkpoint(
            preflight.parent_checkpoint,
            adapter=adapter,
            conditioner=conditioner,
            realization_head=head,
            parent_config=preflight.parent_config,
            config=config,
            source_renderer_manifest_sha256=preflight.source_renderer.manifest_sha256,
        )
        optimizer = build_module_b_optimizer(model=model, head=head, config=config)
    else:
        parameters = (*model.trainable_parameters(), *head.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=0.0,
        )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
        init_scale=config.amp_initial_scale,
    )
    return TrainingState(
        vocabulary_sha256=vocabulary.sha256,
        conditioner=conditioner,
        adapter=adapter,
        model=model,
        head=head,
        optimizer=optimizer,
        scaler=scaler,
        parent_checkpoint_metadata=parent_checkpoint_metadata,
    )


def _diffusion_batch(
    *,
    prepared: PreparedSample,
    prompt: Tensor,
    scheduler: Any,
    seed: int,
) -> DiffusionBatch:
    device = prepared.target.device
    try:
        timestep_count = int(scheduler.config.num_train_timesteps)
    except AttributeError as error:
        raise ValueError("scheduler.config.num_train_timesteps is required") from error
    generator = torch.Generator(device=device).manual_seed(seed)
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
    return DiffusionBatch(
        target_latent=prepared.target,
        context=prepared.context,
        binary_mask=prepared.masks.binary,
        soft_mask=prepared.masks.soft,
        prompt_embeds=prompt,
        defect_type=prepared.sample.defect_type,
        timesteps=timesteps,
        noise=noise,
    )


@torch.no_grad()
def _prepare_proposal_conditioned_samples(
    *,
    vae: Any,
    samples: tuple[Any, ...],
    device: torch.device,
    source_config: P2A1Config,
    config: V06BConfig,
    seed_offset: int,
) -> tuple[PreparedProposalSample, ...]:
    encoded = _prepare_samples(
        vae=vae,
        samples=samples,
        device=device,
        config=source_config,
        seed_offset=seed_offset,
    )
    prepared: list[PreparedProposalSample] = []
    for index, base in enumerate(encoded):
        source_proposal = build_source_training_proposal(
            base.sample,
            image_size=config.image_size,
            proposal_radius=config.proposal_radius,
            support_corner_width=config.support_corner_width,
        )
        proposal = source_proposal.proposal.unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        )
        ground_truth = source_proposal.ground_truth.unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        )
        proposal_masks = build_latent_masks(proposal, base.target.shape[2:])
        ground_truth_masks = build_latent_masks(ground_truth, base.target.shape[2:])
        context = build_masked_context(
            base.target,
            proposal_masks.binary,
            torch.Generator(device=device).manual_seed(
                source_config.context_fill_seed + seed_offset + index
            ),
        )
        target_image = (
            base.sample.pixels.float()
            .add(1.0)
            .mul(0.5)
            .clamp(0.0, 1.0)
            .unsqueeze(0)
            .to(device=device)
        )
        prepared.append(
            PreparedProposalSample(
                sample=base.sample,
                target_latent=base.target,
                target_image=target_image,
                context=context,
                binary_proposal=proposal_masks.binary,
                soft_proposal=proposal_masks.soft,
                soft_ground_truth=ground_truth_masks.soft,
                proposal=proposal,
                ground_truth=ground_truth,
            )
        )
    return tuple(prepared)


def _proposal_conditioned_diffusion_batch(
    *,
    prepared: PreparedProposalSample,
    prompts: dict[str, Tensor],
    defect_types: tuple[str, ...],
    scheduler: Any,
    seed: int,
) -> ProposalConditionedDiffusionBatch:
    device = prepared.target_latent.device
    try:
        timestep_count = int(scheduler.config.num_train_timesteps)
    except AttributeError as error:
        raise ValueError("scheduler.config.num_train_timesteps is required") from error
    generator = torch.Generator(device=device).manual_seed(seed)
    timesteps = torch.randint(
        0,
        timestep_count,
        (prepared.target_latent.shape[0],),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn(
        prepared.target_latent.shape,
        generator=generator,
        device=device,
        dtype=prepared.target_latent.dtype,
    )
    defect_type = prepared.sample.defect_type
    wrong_type = cyclic_wrong_type(defect_type, defect_types)
    return ProposalConditionedDiffusionBatch(
        target_latent=prepared.target_latent,
        target_image=prepared.target_image,
        context=prepared.context,
        binary_proposal=prepared.binary_proposal,
        soft_proposal=prepared.soft_proposal,
        soft_ground_truth=prepared.soft_ground_truth,
        proposal=prepared.proposal,
        ground_truth=prepared.ground_truth,
        prompt_embeds=prompts[defect_type],
        wrong_prompt_embeds=prompts[wrong_type],
        defect_type=defect_type,
        wrong_defect_type=wrong_type,
        timesteps=timesteps,
        noise=noise,
    )


def _realization_batch(
    *,
    pair: GeometryPair,
    prompt: Tensor,
    components: FrozenComponents,
    device: torch.device,
    seed: int,
) -> RealizationBatch:
    vae_parameters = tuple(
        parameter
        for parameter in components.vae.parameters()
        if parameter.is_floating_point()
    )
    if not vae_parameters:
        raise ValueError("VAE must expose floating parameters")
    vae_dtype = vae_parameters[0].dtype
    vae_pixels = pair.target_pixels.unsqueeze(0).to(device=device, dtype=vae_dtype)
    encoded = encode_mode(components.vae, vae_pixels)
    normal_latent = encoded.detach().clone()
    if normal_latent.is_inference() or normal_latent.requires_grad:
        raise ValueError("normal latent must be an ordinary detached tensor")
    normal_image = (
        pair.target_pixels.float()
        .add(1.0)
        .mul(0.5)
        .clamp(0.0, 1.0)
        .unsqueeze(0)
        .to(device=device)
    )
    proposal = pair.proposal.unsqueeze(0).to(device=device, dtype=torch.float32)
    ground_truth = pair.ground_truth.unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    latent_masks = build_latent_masks(proposal, normal_latent.shape[2:])
    context = build_masked_context(
        normal_latent,
        latent_masks.binary,
        torch.Generator(device=device).manual_seed(seed + 10_000_000),
    )
    noise = torch.randn(
        normal_latent.shape,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
        dtype=normal_latent.dtype,
    )
    return RealizationBatch(
        normal_latent=normal_latent,
        normal_image=normal_image,
        context=context,
        binary_proposal=latent_masks.binary,
        proposal=proposal,
        ground_truth=ground_truth,
        prompt_embeds=prompt,
        defect_type=pair.defect_type,
        noise=noise,
    )


def _geometry_pair(slot: UnpairedSlot, config: V06BConfig) -> GeometryPair:
    return build_geometry_pair(
        slot.source,
        slot.target,
        image_size=config.image_size,
        proposal_radius=config.proposal_radius,
        support_corner_width=config.support_corner_width,
    )


def _git_head(cwd: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    if len(head) != 40:
        raise RuntimeError("Git HEAD must be a full commit hash")
    return head


def _metrics_record(
    step: int,
    slot: UnpairedSlot,
    pair: GeometryPair,
    metrics: ModuleBStepMetrics,
) -> dict[str, object]:
    return {
        "step": step,
        "source_id": slot.source.sample_id,
        "target_id": slot.target.sample_id,
        "relationship": "unpaired_geometry_transfer",
        "transfer_retained_fraction": pair.transfer_retained_fraction,
        "transfer_initial_retained_fraction": (
            pair.transfer_initial_retained_fraction
        ),
        "transfer_recentered": pair.transfer_recentered,
        "transfer_row_shift": pair.transfer_row_shift,
        "transfer_column_shift": pair.transfer_column_shift,
        "defect_type": slot.defect_type,
        "seed": slot.seed,
        "total": metrics.total,
        "diffusion": metrics.diffusion,
        "visible": metrics.visible,
        "ring": metrics.ring,
        "mask": metrics.mask,
        "consistency": metrics.consistency,
        "grad_l2": metrics.grad_l2,
        "epsilon_gt": metrics.epsilon_gt,
        "x0_gt": metrics.x0_gt,
        "x0_ring": metrics.x0_ring,
        "epsilon_outside": metrics.epsilon_outside,
        "rgb_gt": metrics.rgb_gt,
        "sobel_gt": metrics.sobel_gt,
        "type_margin": metrics.type_margin,
        "wrong_semantic": metrics.wrong_semantic,
        "appearance_active": metrics.appearance_active,
        "renderer_grad_l2": metrics.renderer_grad_l2,
        "head_grad_l2": metrics.head_grad_l2,
        "finite": metrics.finite,
    }


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device("cuda:0")


def _memory_settings(config: V06BConfig) -> dict[str, int | bool]:
    return {
        "rollout_steps": config.rollout_steps,
        "backprop_steps": config.backprop_steps,
        "checkpoint_tracked_steps": config.checkpoint_tracked_steps,
        "feature_channels": config.feature_channels,
        "head_hidden_channels": config.head_hidden_channels,
        "image_size": config.image_size,
    }


def run_v06_b_probe(context: V06BContext) -> dict[str, object]:
    """Run the single bounded VRAM/runtime probe and discard its updated weights."""

    if context.experiment.schema_version == "v06_b_causal_renderer_v2":
        raise ValueError("v2 must reuse the parent GPU probe; a second probe is forbidden")
    preflight = validate_v06_b_preflight(context)
    claim_path = context.output / "probe_started.json"
    probe_path = context.output / "probe.json"
    context.output.mkdir(parents=True, exist_ok=True)
    if (
        probe_path.exists()
        or probe_path.is_symlink()
        or claim_path.exists()
        or claim_path.is_symlink()
    ):
        raise FileExistsError("the one Module-B GPU probe has already been claimed")
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": "v06_b_gpu_probe_claim_v1",
                "purpose": "one real diffusion update plus one normal 20-step rollout",
                "requested_settings": _memory_settings(context.experiment),
            },
            handle,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    device = _require_cuda()
    torch.manual_seed(context.experiment.seed)
    torch.cuda.manual_seed_all(context.experiment.seed)
    assets = _load_runtime_assets(preflight, device)
    slot = preflight.schedule[0]
    pair = _geometry_pair(slot, context.experiment)
    prepared = _prepare_samples(
        vae=assets.components.vae,
        samples=(slot.source,),
        device=device,
        config=context.source_model,
        seed_offset=60_000_000,
    )[0]
    attempts: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for attempt_index, attempt_config in enumerate(
        probe_attempt_configs(context.experiment), start=1
    ):
        state: TrainingState | None = None
        diffusion_batch: DiffusionBatch | None = None
        realization_batch: RealizationBatch | None = None
        metrics: ModuleBStepMetrics | None = None
        try:
            state = _new_training_state(preflight, assets, attempt_config, device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            diffusion_batch = _diffusion_batch(
                prepared=prepared,
                prompt=assets.prompts[slot.defect_type],
                scheduler=assets.components.scheduler,
                seed=slot.seed + 20_000_000,
            )
            realization_batch = _realization_batch(
                pair=pair,
                prompt=assets.prompts[slot.defect_type],
                components=assets.components,
                device=device,
                seed=slot.seed,
            )
            metrics = module_b_optimizer_step(
                model=state.model,
                vae=assets.components.vae,
                scheduler=assets.components.scheduler,
                head=state.head,
                optimizer=state.optimizer,
                scaler=state.scaler,
                diffusion_batch=diffusion_batch,
                realization_batch=realization_batch,
                source_config=context.source_model,
                config=attempt_config,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            selected = {
                "attempt": attempt_index,
                "settings": _memory_settings(attempt_config),
                "wall_seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "metrics": {
                    "total": metrics.total,
                    "diffusion": metrics.diffusion,
                    "visible": metrics.visible,
                    "ring": metrics.ring,
                    "mask": metrics.mask,
                    "consistency": metrics.consistency,
                    "grad_l2": metrics.grad_l2,
                    "finite": metrics.finite,
                },
            }
            attempts.append(
                {
                    "attempt": attempt_index,
                    "settings": _memory_settings(attempt_config),
                    "status": "passed",
                }
            )
            break
        except torch.cuda.OutOfMemoryError as error:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "settings": _memory_settings(attempt_config),
                    "status": "oom",
                    "error": str(error),
                }
            )
        finally:
            del diffusion_batch, realization_batch, metrics
            if state is not None:
                del state
            gc.collect()
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("Module-B GPU probe exhausted the approved OOM adjustments")
    report: dict[str, object] = {
        "schema_version": "v06_b_gpu_probe_v1",
        "purpose": "one real diffusion update plus one normal 20-step rollout",
        "source_renderer_id": preflight.source_renderer.renderer_id,
        "source_id": slot.source.sample_id,
        "target_id": slot.target.sample_id,
        "defect_type": slot.defect_type,
        "attempts": attempts,
        "selected": selected,
        "weights_discarded": True,
    }
    write_json_atomic(probe_path, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def _load_probe_for_training(context: V06BContext) -> tuple[Path, dict[str, object]]:
    if context.experiment.schema_version == "v06_b_causal_renderer_v2":
        if context.experiment.parent_checkpoint_relpath is None:
            raise ValueError("v2 parent checkpoint relpath is missing")
        probe_root = (
            context.paths.runs_root / context.experiment.parent_checkpoint_relpath
        ).parent
    else:
        probe_root = context.output
    claim_path = probe_root / "probe_started.json"
    if claim_path.is_symlink() or not claim_path.is_file():
        raise ValueError("formal Module-B training requires the one-probe claim")
    try:
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("probe claim must be valid UTF-8 JSON") from error
    if (
        not isinstance(claim, dict)
        or set(claim) != {"schema_version", "purpose", "requested_settings"}
        or claim.get("schema_version") != "v06_b_gpu_probe_claim_v1"
    ):
        raise ValueError("probe claim schema is incompatible")
    path = probe_root / "probe.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("formal Module-B training requires the recorded GPU probe")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("probe report must be valid UTF-8 JSON") from error
    if (
        not isinstance(report, dict)
        or set(report)
        != {
            "schema_version",
            "purpose",
            "source_renderer_id",
            "source_id",
            "target_id",
            "defect_type",
            "attempts",
            "selected",
            "weights_discarded",
        }
        or report.get("schema_version") != "v06_b_gpu_probe_v1"
        or report.get("weights_discarded") is not True
        or not isinstance(report.get("attempts"), list)
        or not report["attempts"]
    ):
        raise ValueError("probe report schema is incompatible")
    selected = report.get("selected")
    if not isinstance(selected, dict) or selected.get("settings") != _memory_settings(
        context.experiment
    ):
        raise ValueError("experiment memory settings must match the successful GPU probe")
    return probe_root.resolve(strict=True), report


def _save_training_checkpoint(
    path: Path,
    *,
    preflight: V06BPreflight,
    state: TrainingState,
    step: int,
    git_head: str,
) -> dict[str, Any]:
    return save_v06_b_checkpoint(
        path,
        adapter=state.adapter,
        conditioner=state.conditioner,
        realization_head=state.head,
        config=preflight.context.experiment,
        git_head=git_head,
        source_renderer_manifest_sha256=(
            preflight.source_renderer.manifest_sha256
        ),
        step=step,
    )


def run_v06_b_training(context: V06BContext) -> dict[str, object]:
    preflight = validate_v06_b_preflight(context)
    probe_root, _ = _load_probe_for_training(context)
    context.output.mkdir(parents=True, exist_ok=True)
    metrics_path = context.output / "training_metrics.jsonl"
    report_path = context.output / "training_report.json"
    final_checkpoint = context.output / "checkpoint"
    if any(
        path.exists() or path.is_symlink()
        for path in (metrics_path, report_path, final_checkpoint)
    ):
        raise FileExistsError("formal Module-B training output already exists")
    device = _require_cuda()
    torch.manual_seed(context.experiment.seed)
    torch.cuda.manual_seed_all(context.experiment.seed)
    assets = _load_runtime_assets(preflight, device)
    state = _new_training_state(preflight, assets, context.experiment, device)
    if context.experiment.schema_version == "v06_b_causal_renderer_v2":
        prepared_samples = _prepare_proposal_conditioned_samples(
            vae=assets.components.vae,
            samples=preflight.split.train,
            device=device,
            source_config=context.source_model,
            config=context.experiment,
            seed_offset=60_000_000,
        )
    else:
        prepared_samples = _prepare_samples(
            vae=assets.components.vae,
            samples=preflight.split.train,
            device=device,
            config=context.source_model,
            seed_offset=60_000_000,
        )
    prepared_by_id = {
        prepared.sample.sample_id: prepared for prepared in prepared_samples
    }
    git_head = _git_head(context.cwd)
    records: list[dict[str, object]] = []
    metrics_path.touch(exist_ok=False)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for additional_step, slot in enumerate(preflight.schedule, start=1):
        step = slot.step
        pair = _geometry_pair(slot, context.experiment)
        prepared = prepared_by_id[slot.source.sample_id]
        if isinstance(prepared, PreparedProposalSample):
            diffusion_batch = _proposal_conditioned_diffusion_batch(
                prepared=prepared,
                prompts=assets.prompts,
                defect_types=context.experiment.defect_types,
                scheduler=assets.components.scheduler,
                seed=slot.seed + 20_000_000,
            )
        else:
            diffusion_batch = _diffusion_batch(
                prepared=prepared,
                prompt=assets.prompts[slot.defect_type],
                scheduler=assets.components.scheduler,
                seed=slot.seed + 20_000_000,
            )
        realization_batch = _realization_batch(
            pair=pair,
            prompt=assets.prompts[slot.defect_type],
            components=assets.components,
            device=device,
            seed=slot.seed,
        )
        metrics = module_b_optimizer_step(
            model=state.model,
            vae=assets.components.vae,
            scheduler=assets.components.scheduler,
            head=state.head,
            optimizer=state.optimizer,
            scaler=state.scaler,
            diffusion_batch=diffusion_batch,
            realization_batch=realization_batch,
            source_config=context.source_model,
            config=context.experiment,
        )
        record = _metrics_record(step, slot, pair, metrics)
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            )
        if step % 10 == 0 or step == 1 or step == len(preflight.schedule):
            print(
                json.dumps(
                    {
                        "event": "v06_b_training",
                        "step": step,
                        "additional_step": additional_step,
                        "additional_steps": len(preflight.schedule),
                        "final_step": context.experiment.final_step,
                        "defect_type": slot.defect_type,
                        "total": metrics.total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (
            additional_step % context.experiment.checkpoint_every == 0
            and additional_step < len(preflight.schedule)
        ):
            _save_training_checkpoint(
                context.output / f"checkpoint-step-{step:04d}",
                preflight=preflight,
                state=state,
                step=step,
                git_head=git_head,
            )
    _save_training_checkpoint(
        final_checkpoint,
        preflight=preflight,
        state=state,
        step=context.experiment.final_step,
        git_head=git_head,
    )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    metric_names = (
        "total",
        "diffusion",
        "visible",
        "ring",
        "mask",
        "consistency",
        "grad_l2",
        "epsilon_gt",
        "x0_gt",
        "x0_ring",
        "epsilon_outside",
        "rgb_gt",
        "sobel_gt",
        "type_margin",
        "wrong_semantic",
        "appearance_active",
        "renderer_grad_l2",
        "head_grad_l2",
    )
    means = {
        name: sum(float(record[name]) for record in records) / len(records)
        for name in metric_names
    }
    if not all(math.isfinite(value) for value in means.values()):
        raise RuntimeError("training summary contains non-finite metrics")
    report: dict[str, object] = {
        "schema_version": "v06_b_training_report_v2",
        "steps": len(records),
        "parent_step": context.experiment.parent_checkpoint_step,
        "additional_steps": context.experiment.training_steps,
        "total_step": context.experiment.final_step,
        "source_renderer_id": preflight.source_renderer.renderer_id,
        "real_relationship": "same_image_proposal_conditioning",
        "normal_relationship": "unpaired_geometry_transfer",
        "weight_initialization": (
            "parent_checkpoint_with_fresh_optimizer"
            if context.experiment.parent_checkpoint_step
            else "source_renderer_with_fresh_optimizer"
        ),
        "parent_checkpoint": (
            str(preflight.parent_checkpoint)
            if preflight.parent_checkpoint is not None
            else None
        ),
        "probe_reused_from": str(probe_root),
        "amp_initial_scale": context.experiment.amp_initial_scale,
        "means": means,
        "wall_seconds": wall_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "checkpoint": str(final_checkpoint),
        "visual_review_required": True,
        "visible_defects_passed": None,
        "state": "awaiting_fixed_panel_and_manual_visual_review",
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


__all__ = [
    "V06BContext",
    "V06BPreflight",
    "probe_attempt_configs",
    "run_v06_b_probe",
    "run_v06_b_training",
    "validate_v06_b_preflight",
]

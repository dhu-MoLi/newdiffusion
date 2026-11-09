"""v07 exact-mask R training.

Same recipe as the baseline (few-shot split, masked infill, type ranking) plus
an optional visibility regularizer and an optional inpainting backbone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.fewshot import (
    _prepare_samples,
    _resampled_training_steps,
    balanced_training_order,
)
from cadiff.p2_a1.model import FactorizedMaskedInfillAdapter
from cadiff.v06_c.can_exact_mask_r import _save_checkpoint
from cadiff.v06_c.visa_exact_mask_r import (
    CHECKPOINT_SCHEMA,
    load_visa_fewshot_split,
    visa_p2_a1_config,
)
from cadiff.v06_c.visa_pilot import load_visa_inventory_for_split
from cadiff.v07.backbones import BackboneSpec, load_backbone_components
from cadiff.v07.model import V07InfillPredictor
from cadiff.v07.objective import ranking_visibility_step
from cadiff.v07.runmeta import write_json
from cadiff.v07.visibility import latent_lowfreq_loss, rgb_lowfreq_loss

SUMMARY_SCHEMA = "v07_exact_mask_r_v1"


def train_exact_mask_r_v07(
    *,
    paths_yaml: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    category: str,
    split: str,
    backbone: BackboneSpec,
    device: str | None = None,
    visibility: str = "none",
    visibility_weight: float = 0.05,
    visibility_factor: int = 2,
    visibility_every: int = 50,
    visibility_size: int = 256,
    steps: int | None = None,
    seed: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if visibility not in {"none", "latent", "rgb"}:
        raise ValueError("visibility must be one of none/latent/rgb")
    if not 0.0 <= float(visibility_weight):
        raise ValueError("visibility_weight must be non-negative")
    if type(visibility_every) is not int or visibility_every < 1:
        raise ValueError("visibility_every must be a positive integer")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite v07 training output: {output}")

    inventory = load_visa_inventory_for_split(dataset_root, category, split, None)
    config = visa_p2_a1_config(category, tuple(item.name for item in inventory.types))
    train_steps = config.steps if steps is None else steps
    train_seed = config.seed if seed is None else seed
    if type(train_steps) is not int or train_steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(train_seed) is not int or train_seed < 0:
        raise ValueError("seed must be a non-negative integer")
    # The checkpoint stores the canonical config so that generation can reload
    # it unchanged. Training-only overrides live in ``train_config`` and in the
    # checkpoint's extra metadata.
    train_config = replace(
        config,
        steps=train_steps,
        seed=train_seed,
        noise_seed=train_seed,
    )
    split_data = load_visa_fewshot_split(manifest, dataset_root, train_config)

    torch_device = torch.device(device or "cuda:0")
    output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)

    components = load_backbone_components(backbone, torch_device, torch.float16)
    prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
    train_prepared = _prepare_samples(
        vae=components.vae,
        samples=split_data.train,
        device=torch_device,
        config=train_config,
        seed_offset=0,
    )
    prepared_by_id = {item.sample.sample_id: item for item in train_prepared}
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    conditioner = DefectConditioner(vocabulary, 128).to(device=torch_device, dtype=torch.float32)
    adapter = FactorizedMaskedInfillAdapter(config).to(device=torch_device, dtype=torch.float32)
    model = V07InfillPredictor(
        components.unet,
        conditioner,
        adapter,
        inpaint=backbone.inpaint,
        latent_channels=backbone.latent_channels,
        latent_size=backbone.latent_size,
    ).train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    schedule = balanced_training_order(
        split_data.train, train_config.steps, train_config.defect_types
    )
    generator = torch.Generator(device=torch_device).manual_seed(
        train_config.noise_seed + 10_000_000
    )
    training: list[dict[str, Any]] = []
    num_train_timesteps = int(components.scheduler.alphas_cumprod.numel())

    for step, sample, prepared, context_seed, context_sha256 in _resampled_training_steps(
        schedule, prepared_by_id, train_config
    ):
        timesteps = torch.randint(
            0,
            num_train_timesteps,
            (1,),
            generator=generator,
            device=torch_device,
            dtype=torch.long,
        )
        noise = torch.randn(
            prepared.target.shape,
            generator=generator,
            device=torch_device,
            dtype=prepared.target.dtype,
        )
        masked_image_latents = None
        if backbone.inpaint:
            masked_image_latents = prepared.target * (
                1.0 - prepared.masks.binary.to(dtype=prepared.target.dtype)
            )
        visibility_fn = None
        if visibility != "none" and (
            step % visibility_every == 1 or step == train_config.steps
        ):
            if visibility == "latent":
                visibility_fn = lambda predicted, target, soft: latent_lowfreq_loss(
                    predicted, target, soft, factor=visibility_factor
                )
            else:
                target_image = prepared.sample.pixels.unsqueeze(0).to(torch_device)
                target_mask = prepared.sample.mask.unsqueeze(0).to(torch_device)
                visibility_fn = lambda predicted, target, soft, _vae=components.vae, _img=target_image, _mask=target_mask: rgb_lowfreq_loss(
                    _vae, predicted, _img, _mask, size=visibility_size
                )
        metrics = ranking_visibility_step(
            model=model,
            unet=components.unet,
            scheduler=components.scheduler,
            optimizer=optimizer,
            scaler=scaler,
            target_latent=prepared.target,
            context=prepared.context,
            binary_mask=prepared.masks.binary,
            soft_mask=prepared.masks.soft,
            prompt_embeds=prompt,
            defect_type=sample.defect_type,
            timesteps=timesteps,
            noise=noise,
            config=train_config,
            masked_image_latents=masked_image_latents,
            visibility_fn=visibility_fn,
            visibility_weight=float(visibility_weight) if visibility != "none" else 0.0,
        )
        training.append(
            {
                "step": step,
                "sample_id": sample.sample_id,
                "defect_type": sample.defect_type,
                "timestep": int(timesteps.item()),
                **asdict(metrics),
                "context_seed": context_seed,
                "context_sha256": context_sha256,
            }
        )
        if step == 1 or step % 100 == 0 or step == train_config.steps:
            print(
                json.dumps(
                    {
                        "stage": "v07_exact_mask_r",
                        "step": step,
                        "steps": train_config.steps,
                        "total": metrics.total,
                        "visibility": metrics.visibility,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    model.eval()
    metadata = {
        "backbone": backbone.name,
        "backbone_kind": backbone.kind,
        "backbone_in_channels": backbone.in_channels,
        "visibility": visibility,
        "visibility_weight": float(visibility_weight) if visibility != "none" else 0.0,
        "visibility_factor": int(visibility_factor),
        "visibility_every": int(visibility_every),
        "trained_steps": int(train_steps),
        "trained_seed": int(train_seed),
        "v07": True,
    }
    if extra_metadata:
        overlap = set(extra_metadata) & set(metadata)
        if overlap:
            raise ValueError(f"v07 metadata keys collide: {sorted(overlap)}")
        metadata.update(extra_metadata)
    checkpoint = _save_checkpoint(
        output / "checkpoint",
        adapter=adapter,
        conditioner=conditioner,
        config=config,
        vocabulary_sha256=vocabulary.sha256,
        checkpoint_schema=CHECKPOINT_SCHEMA,
        extra_metadata=metadata,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "category": config.category,
        "defect_types": list(config.defect_types),
        "steps": train_config.steps,
        "seed": train_config.seed,
        "fewshot_train": len(split_data.train),
        "open_dev_report": len(split_data.evaluation),
        "backbone": backbone.name,
        "visibility": visibility,
        "visibility_weight": float(visibility_weight) if visibility != "none" else 0.0,
        "checkpoint": checkpoint,
        "last_total": training[-1]["total"],
        "last_visibility": training[-1]["visibility"],
        "output": str(output),
    }
    write_json(output / "summary.json", summary)
    write_json(output / "training.json", {"records": training})
    return summary


__all__ = ["SUMMARY_SCHEMA", "train_exact_mask_r_v07"]

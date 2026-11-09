"""Independent MVTec AD 2 can exact-mask R. Not official P2-A1 and not Cable R.

Trains a fresh SD1.5 adapter on DualAnoDiff few-shot (30 images). Ranking is
correct-type vs frozen normal U-Net only because can has a single type `bad`.
Does not modify frozen P2-A1 yaml profiles or borrow Cable/Hazelnut weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import Tensor

from cadiff.config import load_config
from cadiff.data.geometry import resize_pad_image, resize_pad_mask
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.fewshot import (
    FewshotSample,
    FewshotSplit,
    _prepare_samples,
    _resampled_training_steps,
    balanced_training_order,
)
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_a1.model import FactorizedMaskedInfillAdapter
from cadiff.p2_a1.ranking import ranking_optimizer_step
from cadiff.p2_b0.generation import denoise_masked_infill
from cadiff.provenance import sha256_file, write_json_atomic
from cadiff.v06_c.can_mvtec2 import CATEGORY, DEFECT_TYPE, IMAGE_SIZE

CHECKPOINT_SCHEMA = "v06_c_can_exact_mask_r_checkpoint_v1"
DDIM_STEPS = 50


def can_p2_a1_config() -> P2A1Config:
    return P2A1Config(
        schema_version="p2_a1_factorized_v1",
        category=CATEGORY,
        defect_types=(DEFECT_TYPE,),
        normal_prompt="a photo of a normal can",
        image_size=IMAGE_SIZE,
        dtype="float16",
        seed=20260912,
        learning_rate=1e-4,
        steps=1200,
        capacity_defect_type=DEFECT_TYPE,
        context_fill_seed=202609120,
        noise_seed=202609121,
        eval_timesteps=(100, 400, 700),
        inside_noise_weight=4.0,
        lambda_inside=1.0,
        lambda_outside=0.1,
        train_role="fewshot_train",
        eval_role="open_dev_report",
        wrong_type_policy="cyclic_next",
        correct_wrong_ratio_limit=1.0,
        correct_normal_ratio_limit=1.0,
        type_rank=8,
        ranking_target_ratio=0.95,
        ranking_weight=0.25,
    )


def _load_sample(raw: dict[str, object], root: Path, role: str, config: P2A1Config) -> FewshotSample:
    defect_type = raw.get("defect_type")
    image_relpath = raw.get("image_relpath")
    mask_relpath = raw.get("mask_relpath")
    image_sha256 = raw.get("image_sha256")
    mask_sha256 = raw.get("mask_sha256")
    if defect_type != DEFECT_TYPE:
        raise ValueError(f"can R only trains {DEFECT_TYPE}")
    if not isinstance(image_relpath, str) or not image_relpath:
        raise ValueError("image_relpath must be a non-empty string")
    if not isinstance(mask_relpath, str) or not mask_relpath:
        raise ValueError("mask_relpath must be a non-empty string")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("image_sha256 must be a SHA256 string")
    if not isinstance(mask_sha256, str) or len(mask_sha256) != 64:
        raise ValueError("mask_sha256 must be a SHA256 string")
    image_path = root / image_relpath
    mask_path = root / mask_relpath
    if ".." in PurePosixPath(image_relpath).parts or ".." in PurePosixPath(mask_relpath).parts:
        raise ValueError("can few-shot paths must stay under the layout root")
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(image_relpath)
    if sha256_file(image_path) != image_sha256 or sha256_file(mask_path) != mask_sha256:
        raise ValueError("can few-shot SHA256 mismatch")
    with Image.open(image_path) as image:
        transformed, _, geometry = resize_pad_image(image, config.image_size)
    with Image.open(mask_path) as mask_image:
        prepared = Image.fromarray(
            ((np.asarray(mask_image.convert("L")) > 0).astype(np.uint8) * 255)
        )
        if prepared.size != (geometry.source_width, geometry.source_height):
            prepared = prepared.resize(
                (geometry.source_width, geometry.source_height),
                Image.Resampling.NEAREST,
            )
        transformed_mask = resize_pad_mask(prepared, geometry)
    pixels = (
        torch.from_numpy(np.array(transformed, dtype=np.uint8, copy=True))
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(127.5)
        .sub_(1.0)
        .contiguous()
    )
    mask = (
        torch.from_numpy(np.array(transformed_mask, dtype=np.uint8, copy=True))
        .ge(128)
        .to(dtype=torch.float32)
        .unsqueeze(0)
        .contiguous()
    )
    if not bool(mask.any()):
        raise ValueError(f"empty can mask after letterbox: {image_relpath}")
    sample_id = hashlib.sha256(
        f"{config.category}\n{defect_type}\n{image_relpath}".encode()
    ).hexdigest()
    return FewshotSample(
        sample_id=sample_id,
        defect_type=str(defect_type),
        role=role,
        image_relpath=image_relpath,
        mask_relpath=mask_relpath,
        image_path=image_path,
        mask_path=mask_path,
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        pixels=pixels,
        mask=mask,
    )


def load_can_fewshot_split(manifest: Path, dataset_root: Path, config: P2A1Config) -> FewshotSplit:
    root = dataset_root.resolve(strict=True)
    selected: dict[str, list[FewshotSample]] = {config.train_role: [], config.eval_role: []}
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("category") != CATEGORY:
                continue
            roles = raw.get("roles")
            if not isinstance(roles, list):
                continue
            matched = [role for role in selected if role in roles]
            if not matched:
                continue
            if len(matched) != 1:
                raise ValueError(f"manifest line {line_number} has mixed train/eval roles")
            selected[matched[0]].append(_load_sample(raw, root, matched[0], config))
    train = tuple(sorted(selected[config.train_role], key=lambda item: item.image_relpath))
    evaluation = tuple(sorted(selected[config.eval_role], key=lambda item: item.image_relpath))
    if len(train) != 30 or len(evaluation) != 60:
        raise ValueError(f"can few-shot split must be 30/60, got {len(train)}/{len(evaluation)}")
    if any(item.defect_type != DEFECT_TYPE for item in (*train, *evaluation)):
        raise ValueError("can few-shot split must be type bad only")
    return FewshotSplit(train=train, evaluation=evaluation)


def _save_checkpoint(
    output: Path,
    *,
    adapter: torch.nn.Module,
    conditioner: torch.nn.Module,
    config: P2A1Config,
    vocabulary_sha256: str,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"checkpoint already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, int | str]] = {}
    for name, module in (("adapter", adapter), ("conditioner", conditioner)):
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in module.state_dict().items()
        }
        if not state or any(not torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError(f"{name} checkpoint state must be non-empty and finite")
        path = output / f"{name}.safetensors"
        save_file(state, path)
        files[path.name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    metadata: dict[str, Any] = {
        "schema_version": checkpoint_schema,
        "vocabulary_sha256": vocabulary_sha256,
        "config": asdict(config),
        "files": files,
        "diagnostic_only": True,
        "not_official_p2_a1": True,
        "not_cable_or_hazelnut": True,
    }
    if extra_metadata:
        overlap = set(extra_metadata) & set(metadata)
        if overlap:
            raise ValueError(f"checkpoint metadata keys collide: {sorted(overlap)}")
        metadata.update(extra_metadata)
    write_json_atomic(output / "metadata.json", metadata)
    return metadata


def load_can_r_modules(
    checkpoint: Path,
    *,
    config: P2A1Config,
    device: torch.device,
    expected_schema: str = CHECKPOINT_SCHEMA,
) -> tuple[DefectVocabulary, DefectConditioner, FactorizedMaskedInfillAdapter, dict[str, Any]]:
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != expected_schema:
        raise ValueError("checkpoint is not the independent can exact-mask R")
    if metadata.get("config") != json.loads(
        json.dumps(asdict(config), ensure_ascii=False, sort_keys=True, allow_nan=False)
    ):
        raise ValueError("can R checkpoint config does not match")
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    if metadata.get("vocabulary_sha256") != vocabulary.sha256:
        raise ValueError("can R vocabulary changed")
    conditioner = DefectConditioner(vocabulary, 128).to(device=device, dtype=torch.float32)
    adapter = FactorizedMaskedInfillAdapter(config).to(device=device, dtype=torch.float32)
    conditioner.load_state_dict(
        load_file(checkpoint / "conditioner.safetensors", device="cpu"), strict=True
    )
    adapter.load_state_dict(load_file(checkpoint / "adapter.safetensors", device="cpu"), strict=True)
    conditioner.eval()
    adapter.eval()
    return vocabulary, conditioner, adapter, metadata


class CanExactMaskRenderer:
    """50-step DDIM infill given a placed binary mask. Independent of mask-first B."""

    def __init__(
        self,
        *,
        components: Any,
        model: P2InfillNoisePredictor,
        prompt: Tensor,
        device: torch.device,
        image_size: int = IMAGE_SIZE,
        ddim_steps: int = DDIM_STEPS,
        allowed_types: tuple[str, ...] | None = None,
    ) -> None:
        self.components = components
        self.model = model
        self.prompt = prompt
        self.device = device
        self.image_size = image_size
        self.ddim_steps = ddim_steps
        self.allowed_types = (DEFECT_TYPE,) if allowed_types is None else tuple(allowed_types)
        self._scheduler_config = components.scheduler.config
        self._vae_dtype = next(
            parameter for parameter in components.vae.parameters() if parameter.is_floating_point()
        ).dtype

    @torch.inference_mode()
    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        realization_mask: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        if defect_type not in self.allowed_types:
            raise ValueError(f"renderer vocabulary does not include {defect_type}")
        if (
            not isinstance(normal, np.ndarray)
            or normal.shape != (self.image_size, self.image_size, 3)
            or normal.dtype != np.uint8
        ):
            raise ValueError("normal must be 512x512 uint8 RGB")
        mask = np.ascontiguousarray(realization_mask, dtype=bool)
        if mask.shape != (self.image_size, self.image_size) or not mask.any() or mask.all():
            raise ValueError("realization mask must be a non-empty, non-full 512 binary")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        pixels = (
            torch.from_numpy(np.ascontiguousarray(normal))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self._vae_dtype)
            .div(127.5)
            .sub(1.0)
        )
        normal_latent = encode_mode(self.components.vae, pixels).detach()
        source_mask = (
            torch.from_numpy(mask.copy())
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        latent_mask = build_latent_masks(source_mask, tuple(normal_latent.shape[2:])).binary
        context = build_masked_context(
            normal_latent,
            latent_mask,
            torch.Generator(device=self.device).manual_seed(seed + 10_000_000),
        )
        noise = torch.randn(
            normal_latent.shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=normal_latent.dtype,
        )
        generated_latent = denoise_masked_infill(
            model=self.model,
            scheduler=DDIMScheduler.from_config(self._scheduler_config),
            normal_latent=normal_latent,
            context=context,
            binary_mask=latent_mask,
            prompt_embeds=self.prompt,
            defect_type=defect_type,
            noise=noise,
            steps=self.ddim_steps,
        )
        decoded = decode_latent(self.components.vae, generated_latent).float()
        raw = (
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
        return np.ascontiguousarray(raw)


def load_can_renderer(
    *,
    checkpoint: Path,
    model_root: Path,
    device: str | None = None,
) -> CanExactMaskRenderer:
    config = can_p2_a1_config()
    torch_device = torch.device(device or "cuda:0")
    components = load_frozen_components(model_root, torch_device, torch.float16)
    _, conditioner, adapter, _metadata = load_can_r_modules(
        checkpoint, config=config, device=torch_device
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).eval()
    prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
    return CanExactMaskRenderer(
        components=components,
        model=model,
        prompt=prompt,
        device=torch_device,
    )


def _write_visual(
    *,
    renderer: CanExactMaskRenderer,
    sample: FewshotSample,
    output: Path,
) -> str:
    rgb = (
        ((sample.pixels.float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    mask = sample.mask[0].numpy().astype(bool)
    raw = renderer(np.ascontiguousarray(rgb), sample.defect_type, mask, seed=20260912)
    visual_root = output / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(visual_root / "fewshot_rgb.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(visual_root / "fewshot_mask.png")
    Image.fromarray(raw).save(visual_root / "fewshot_infill.png")
    return "visuals/fewshot_infill.png"


def train_can_exact_mask_r(
    *,
    paths_yaml: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    device: str | None = None,
    load_split: Any = None,
    checkpoint_schema: str = CHECKPOINT_SCHEMA,
    summary_schema: str = "v06_c_can_exact_mask_r_v1",
    extra_metadata: dict[str, Any] | None = None,
    stage_name: str = "can_exact_mask_r",
    config: P2A1Config | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite can R output: {output}")
    config = can_p2_a1_config() if config is None else config
    split_loader = load_can_fewshot_split if load_split is None else load_split
    split = split_loader(manifest, dataset_root, config)
    paths = load_config(paths_yaml)
    torch_device = torch.device(device or "cuda:0")
    output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    components = load_frozen_components(paths.model_root, torch_device, torch.float16)
    prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
    train_prepared = _prepare_samples(
        vae=components.vae,
        samples=split.train,
        device=torch_device,
        config=config,
        seed_offset=0,
    )
    prepared_by_id = {item.sample.sample_id: item for item in train_prepared}
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    conditioner = DefectConditioner(vocabulary, 128).to(device=torch_device, dtype=torch.float32)
    adapter = FactorizedMaskedInfillAdapter(config).to(device=torch_device, dtype=torch.float32)
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    schedule = balanced_training_order(split.train, config.steps, config.defect_types)
    generator = torch.Generator(device=torch_device).manual_seed(config.noise_seed + 10_000_000)
    training: list[dict[str, Any]] = []
    num_train_timesteps = int(components.scheduler.alphas_cumprod.numel())
    for step, sample, prepared, context_seed, context_sha256 in _resampled_training_steps(
        schedule, prepared_by_id, config
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
        metrics = ranking_optimizer_step(
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
            config=config,
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
        if step == 1 or step % 100 == 0 or step == config.steps:
            print(
                json.dumps(
                    {
                        "stage": stage_name,
                        "step": step,
                        "steps": config.steps,
                        "total": metrics.total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval()
    checkpoint = _save_checkpoint(
        output / "checkpoint",
        adapter=adapter,
        conditioner=conditioner,
        config=config,
        vocabulary_sha256=vocabulary.sha256,
        checkpoint_schema=checkpoint_schema,
        extra_metadata=extra_metadata,
    )
    renderer = CanExactMaskRenderer(
        components=components,
        model=model,
        prompt=prompt,
        device=torch_device,
        allowed_types=config.defect_types,
    )
    visual = _write_visual(renderer=renderer, sample=split.train[0], output=output)
    summary = {
        "schema_version": summary_schema,
        "diagnostic_only": True,
        "not_official_p2_a1": True,
        "not_learned_h": True,
        "category": config.category,
        "defect_types": list(config.defect_types),
        "defect_type": config.defect_types[0] if len(config.defect_types) == 1 else None,
        "steps": config.steps,
        "fewshot_train": len(split.train),
        "open_dev_report": len(split.evaluation),
        "ranking_negative": "frozen_normal_unet",
        "checkpoint": checkpoint,
        "visual": visual,
        "last_total": training[-1]["total"],
        "output": str(output),
    }
    if extra_metadata:
        summary.update(extra_metadata)
    write_json_atomic(output / "summary.json", summary)
    write_json_atomic(output / "training.json", {"records": training})
    return summary


def train_can_exact_mask_r_from_cli(
    *,
    paths: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    del cwd
    return train_can_exact_mask_r(
        paths_yaml=paths,
        manifest=manifest,
        dataset_root=dataset_root,
        output=output,
        device=device,
    )


__all__ = [
    "CanExactMaskRenderer",
    "can_p2_a1_config",
    "load_can_fewshot_split",
    "load_can_renderer",
    "train_can_exact_mask_r",
    "train_can_exact_mask_r_from_cli",
]

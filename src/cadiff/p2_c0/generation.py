from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.dataset import load_training_image, load_training_index
from cadiff.p2_a0.fewshot import load_fewshot_split
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_a1.runner import _load_p2_a1_checkpoint
from cadiff.p2_b0.generation import denoise_masked_infill
from cadiff.p2_b1.runner import _prepare_normal_tensors
from cadiff.provenance import sha256_file

from .contracts import GenerationJob, TrainingMask, build_generation_schedule


@dataclass(frozen=True)
class GenerationSummary:
    record_count: int
    per_type: dict[str, int]
    manifest_sha256: str


@dataclass(frozen=True)
class GenerationPools:
    normals: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    training_masks: tuple[TrainingMask, ...]


RenderFunction = Callable[[np.ndarray, np.ndarray, GenerationJob], np.ndarray]


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 array with shape {shape}")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    payload = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_synthetic_dataset(
    *,
    jobs: tuple[GenerationJob, ...],
    normals: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    output: Path,
    render: RenderFunction,
    checkpoint_hashes: Mapping[str, str],
) -> GenerationSummary:
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("jobs must be non-empty with unique job IDs")
    if any(job.mask_type != job.defect_type for job in jobs):
        raise ValueError("every synthetic job must use a type-compatible mask")
    if not checkpoint_hashes or any(
        not isinstance(name, str) or not name or not _valid_sha(value)
        for name, value in checkpoint_hashes.items()
    ):
        raise ValueError("checkpoint_hashes must contain lowercase SHA256 values")
    if os.path.lexists(output):
        raise FileExistsError(f"synthetic output already exists: {output}")
    records: list[dict[str, Any]] = []
    try:
        images = output / "images"
        saved_masks = output / "masks"
        images.mkdir(parents=True, exist_ok=False)
        saved_masks.mkdir(exist_ok=False)
        for job in jobs:
            if job.normal_id not in normals or job.mask_id not in masks:
                raise ValueError("generation job references an unknown normal or mask")
            normal = np.asarray(normals[job.normal_id])
            if normal.ndim != 3 or normal.shape[2] != 3:
                raise ValueError("normal images must have H x W x 3 shape")
            normal = _array(normal, normal.shape, "normal")
            mask = _array(
                masks[job.mask_id], (normal.shape[0], normal.shape[1]), "mask"
            )
            if not np.isin(mask, (0, 1, 255)).all():
                raise ValueError("masks must be binary")
            binary = mask != 0
            if not binary.any() or binary.all():
                raise ValueError("masks must contain foreground and background")
            rendered = _array(render(normal.copy(), binary.copy(), job), normal.shape, "rendered")
            if not np.array_equal(rendered[~binary], normal[~binary]):
                raise ValueError("renderer changed pixels outside the conditioning mask")
            image_relpath = f"images/{job.job_id}.png"
            mask_relpath = f"masks/{job.job_id}.png"
            image_path = output / image_relpath
            mask_path = output / mask_relpath
            Image.fromarray(rendered).save(image_path)
            Image.fromarray(binary.astype(np.uint8) * 255).save(mask_path)
            image_bytes = image_path.read_bytes()
            mask_bytes = mask_path.read_bytes()
            records.append(
                {
                    "schema_version": "p2_c0_synthetic_record_v1",
                    "role": "synthetic_train",
                    "job_id": job.job_id,
                    "normal_id": job.normal_id,
                    "mask_id": job.mask_id,
                    "mask_type": job.mask_type,
                    "defect_type": job.defect_type,
                    "seed": job.seed,
                    "image_relpath": image_relpath,
                    "mask_relpath": mask_relpath,
                    "image_sha256": _sha256_bytes(image_bytes),
                    "mask_sha256": _sha256_bytes(mask_bytes),
                    "checkpoint_hashes": dict(checkpoint_hashes),
                }
            )
        manifest = output / "synthetic_manifest.jsonl"
        _atomic_jsonl(manifest, records)
        counts = Counter(record["defect_type"] for record in records)
        return GenerationSummary(
            record_count=len(records),
            per_type={name: counts[name] for name in sorted(counts)},
            manifest_sha256=sha256_file(manifest),
        )
    except BaseException:
        if output.exists() and output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


def load_generation_pools(
    *,
    manifest: Path,
    dataset_root: Path,
    source_config: P2A1Config,
    normal_count: int,
    image_size: int,
) -> GenerationPools:
    index = load_training_index(manifest, dataset_root, source_config)  # type: ignore[arg-type]
    split = load_fewshot_split(manifest, dataset_root, source_config)
    if type(normal_count) is not int or normal_count <= 0 or len(index.normals) < normal_count:
        raise ValueError("normal_count must fit the frozen normal_train pool")
    indices = [
        round(index_value * (len(index.normals) - 1) / (normal_count - 1))
        for index_value in range(normal_count)
    ]
    selected_normals = tuple(index.normals[index_value] for index_value in indices)
    normals: dict[str, np.ndarray] = {}
    for item in selected_normals:
        loaded = load_training_image(item, image_size)
        array = (
            ((loaded.pixels + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        normals[item.sample_id] = np.ascontiguousarray(array)
    masks = {
        sample.sample_id: np.ascontiguousarray(
            sample.mask[0].mul(255).to(torch.uint8).numpy()
        )
        for sample in split.train
    }
    training_masks = tuple(
        TrainingMask(
            sample_id=sample.sample_id,
            defect_type=sample.defect_type,
            role=sample.role,
        )
        for sample in split.train
    )
    return GenerationPools(
        normals=normals,
        masks=masks,
        training_masks=training_masks,
    )


class FrozenP2B2Renderer:
    def __init__(
        self,
        *,
        model_root: Path,
        checkpoint: Path,
        checkpoint_hashes: Mapping[str, str],
        source_config: P2A1Config,
        prompts: Mapping[str, str],
        device: torch.device,
        denoise_steps: int = 50,
    ) -> None:
        for name, expected in checkpoint_hashes.items():
            path = checkpoint / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"frozen renderer checkpoint mismatch: {name}")
        self.device = device
        self.denoise_steps = denoise_steps
        self.components = load_frozen_components(model_root, device, torch.float16)
        _, conditioner, adapter, _ = _load_p2_a1_checkpoint(
            checkpoint, config=source_config, device=device
        )
        self.model = P2InfillNoisePredictor(
            self.components.unet, conditioner, adapter
        ).eval()
        with torch.inference_mode():
            self.prompts = {
                defect_type: encode_prompt(self.components, prompt, device).detach()
                for defect_type, prompt in prompts.items()
            }
        self._normal_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.inference_mode()
    def __call__(
        self, normal: np.ndarray, mask: np.ndarray, job: GenerationJob
    ) -> np.ndarray:
        if job.normal_id not in self._normal_cache:
            pixels = (
                torch.from_numpy(normal.copy())
                .permute(2, 0, 1)
                .to(dtype=torch.float32)
                .div(127.5)
                .sub(1.0)
            )
            vae_dtype = next(self.components.vae.parameters()).dtype
            vae_pixels, normal_image = _prepare_normal_tensors(
                pixels, self.device, vae_dtype=vae_dtype
            )
            normal_latent = encode_mode(self.components.vae, vae_pixels).detach()
            self._normal_cache[job.normal_id] = (normal_latent, normal_image)
        normal_latent, normal_image = self._normal_cache[job.normal_id]
        source_mask = (
            torch.from_numpy(mask.copy())
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        latent_masks = build_latent_masks(source_mask, normal_latent.shape[2:])
        context = build_masked_context(
            normal_latent,
            latent_masks.binary,
            torch.Generator(device=self.device).manual_seed(job.seed + 10_000_000),
        )
        noise = torch.randn(
            normal_latent.shape,
            generator=torch.Generator(device=self.device).manual_seed(job.seed),
            device=self.device,
            dtype=normal_latent.dtype,
        )
        scheduler = DDIMScheduler.from_config(self.components.scheduler.config)
        generated_latent = denoise_masked_infill(
            model=self.model,
            scheduler=scheduler,
            normal_latent=normal_latent,
            context=context,
            binary_mask=latent_masks.binary,
            prompt_embeds=self.prompts[job.defect_type],
            defect_type=job.defect_type,
            noise=noise,
            steps=self.denoise_steps,
        )
        raw = decode_latent(self.components.vae, generated_latent).float()
        composite = raw * source_mask + normal_image * (1.0 - source_mask)
        array = (
            composite[0]
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        result = np.ascontiguousarray(array)
        result[~mask] = normal[~mask]
        return result


def build_formal_generation_schedule(
    *, pools: GenerationPools, defect_types: tuple[str, ...], samples_per_type: int, seed: int
) -> tuple[GenerationJob, ...]:
    return build_generation_schedule(
        normal_ids=tuple(pools.normals),
        masks=pools.training_masks,
        defect_types=defect_types,
        samples_per_type=samples_per_type,
        seed=seed,
    )


__all__ = [
    "FrozenP2B2Renderer",
    "GenerationPools",
    "GenerationSummary",
    "build_formal_generation_schedule",
    "load_generation_pools",
    "materialize_synthetic_dataset",
]

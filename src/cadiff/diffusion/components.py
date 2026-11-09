from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

REQUIRED = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


@dataclass(frozen=True)
class FrozenComponents:
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    unet: UNet2DConditionModel
    scheduler: DDIMScheduler


def validate_model_layout(root: Path) -> None:
    resolved_root = Path(root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"model root is not a directory: {resolved_root}")

    invalid: list[str] = []
    for relative in REQUIRED:
        candidate = resolved_root / relative
        try:
            resolved_file = candidate.resolve(strict=True)
        except FileNotFoundError:
            invalid.append(relative)
            continue
        try:
            resolved_file.relative_to(resolved_root)
        except ValueError:
            invalid.append(relative)
            continue
        if not resolved_file.is_file() or resolved_file.stat().st_size == 0:
            invalid.append(relative)
    if invalid:
        raise FileNotFoundError(f"invalid SD1.5 files: {invalid}")


def load_frozen_components(
    model_root: Path, device: torch.device, dtype: torch.dtype
) -> FrozenComponents:
    validate_model_layout(model_root)
    vae = AutoencoderKL.from_pretrained(
        model_root, subfolder="vae", local_files_only=True, torch_dtype=dtype
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_root, subfolder="text_encoder", local_files_only=True, dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_root, subfolder="tokenizer", local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_root, subfolder="unet", local_files_only=True, torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(
        model_root, subfolder="scheduler", local_files_only=True
    )
    for module in (vae, text_encoder, unet):
        module.requires_grad_(False)
        module.eval()
        module.to(device)
    return FrozenComponents(vae, text_encoder, tokenizer, unet, scheduler)

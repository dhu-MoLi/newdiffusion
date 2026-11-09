"""Backbone registry.

Currently supported:

* ``sd15``         - the frozen SD1.5 U-Net used by the baseline (4-channel).
* ``sd2-inpaint``  - Stable Diffusion 2 inpainting (9-channel U-Net). The
  predictor concatenates ``(noisy, mask, masked_image_latents)`` as diffusers
  inpainting pipelines do.

SDXL/FLUX are intentionally not wired yet: their U-Net/DiT block layout does
not match the residual adapter, so they need a separate adapter design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from cadiff.diffusion.components import FrozenComponents, load_frozen_components


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    kind: str
    root: Path
    in_channels: int
    latent_channels: int = 4
    latent_size: int = 64

    @property
    def inpaint(self) -> bool:
        return self.kind == "sd2_inpaint"


def default_sd2_inpaint_root() -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--stable-diffusion-2-inpainting"
        / "snapshots/sd2-inpaint-local"
    )


def make_backbone_spec(
    name: str,
    *,
    sd15_root: Path,
    sd2_inpaint_root: Path | None = None,
) -> BackboneSpec:
    if name == "sd15":
        return BackboneSpec(
            name=name,
            kind="sd15",
            root=Path(sd15_root).resolve(strict=True),
            in_channels=4,
        )
    if name in {"sd2-inpaint", "sd2_inpaint"}:
        root = Path(sd2_inpaint_root or default_sd2_inpaint_root()).expanduser().resolve(strict=True)
        return BackboneSpec(
            name="sd2-inpaint",
            kind="sd2_inpaint",
            root=root,
            in_channels=9,
        )
    raise ValueError(f"unknown backbone: {name}")


def _load_sd2_inpaint(root: Path, device: torch.device, dtype: torch.dtype) -> FrozenComponents:
    vae = AutoencoderKL.from_pretrained(
        root, subfolder="vae", local_files_only=True, torch_dtype=dtype
    )
    text_encoder = CLIPTextModel.from_pretrained(
        root, subfolder="text_encoder", local_files_only=True, dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        root, subfolder="tokenizer", local_files_only=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        root, subfolder="unet", local_files_only=True, torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(
        root, subfolder="scheduler", local_files_only=True
    )
    if int(unet.config.in_channels) != 9:
        raise ValueError(
            f"inpainting backbone must have 9 input channels, got {unet.config.in_channels}"
        )
    if int(unet.config.sample_size) != 64:
        raise ValueError("inpainting backbone must be a 512px model")
    for module in (vae, text_encoder, unet):
        module.requires_grad_(False)
        module.eval()
        module.to(device)
    return FrozenComponents(vae, text_encoder, tokenizer, unet, scheduler)


def load_backbone_components(
    spec: BackboneSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> FrozenComponents:
    if spec.kind == "sd15":
        if spec.in_channels != 4:
            raise ValueError("sd15 backbone must have four input channels")
        return load_frozen_components(spec.root, device, dtype)
    if spec.kind == "sd2_inpaint":
        return _load_sd2_inpaint(spec.root, device, dtype)
    raise ValueError(f"unknown backbone kind: {spec.kind}")


__all__ = [
    "BackboneSpec",
    "default_sd2_inpaint_root",
    "load_backbone_components",
    "make_backbone_spec",
]

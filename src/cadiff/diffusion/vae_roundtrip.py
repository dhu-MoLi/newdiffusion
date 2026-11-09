from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def _validate_image_tensor(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 4:
        raise ValueError("image must be a 4D NCHW tensor")
    if image.shape[1] != 3:
        raise ValueError("image must have three RGB channels")
    if not image.is_floating_point():
        raise TypeError("image must have a floating-point dtype")
    if not torch.isfinite(image).all():
        raise ValueError("image must contain only finite values")


def _validate_latent_tensor(latent: torch.Tensor) -> None:
    if not isinstance(latent, torch.Tensor):
        raise TypeError("latent must be a torch.Tensor")
    if latent.ndim != 4:
        raise ValueError("latent must be a 4D tensor")
    if not latent.is_floating_point():
        raise TypeError("latent must have a floating-point dtype")
    if not torch.isfinite(latent).all():
        raise ValueError("latent must contain only finite values")


def image_to_tensor(
    image: Image.Image, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    tensor = pil_to_tensor(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype) / 127.5 - 1.0


@torch.inference_mode()
def encode_mode(vae: object, image: torch.Tensor) -> torch.Tensor:
    _validate_image_tensor(image)
    scaling = float(vae.config.scaling_factor)  # type: ignore[attr-defined]
    latent = vae.encode(image).latent_dist.mode() * scaling  # type: ignore[attr-defined]
    _validate_latent_tensor(latent)
    return latent


@torch.inference_mode()
def decode_latent(vae: object, latent: torch.Tensor) -> torch.Tensor:
    _validate_latent_tensor(latent)
    scaling = float(vae.config.scaling_factor)  # type: ignore[attr-defined]
    image = vae.decode(latent / scaling).sample  # type: ignore[attr-defined]
    _validate_image_tensor(image)
    decoded = (image / 2 + 0.5).clamp(0, 1)
    if not torch.isfinite(decoded).all():
        raise ValueError("decoded image must contain only finite values")
    return decoded

"""Frozen DINOv2 patch features for the atlas diagnostic only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class DinoFeatures:
    tokens: NDArray[np.float32]
    grid_height: int
    grid_width: int


class FrozenDinoEncoder:
    """Load a frozen dense encoder. Prefer DINOv2; fall back to local ResNet-18."""

    def __init__(self, *, model_name: str = "dinov2_vits14", device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.backend, self.model, self.input_size = _load_encoder(model_name, self.device)
        self.model_name = self.backend

    @torch.inference_mode()
    def encode(self, rgb: NDArray[np.uint8]) -> DinoFeatures:
        if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
            raise ValueError("encoder expects a 512x512 uint8 RGB image")
        image = Image.fromarray(rgb, mode="RGB").resize(
            (self.input_size, self.input_size), resample=Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.backend.startswith("dinov2") or self.backend.startswith("transformers_"):
            tokens = self.model.forward_features(tensor)["x_norm_patchtokens"]
            grid = tokens.shape[1]
            side = int(round(grid**0.5))
            mapped = tokens[0].detach().float().cpu().numpy().reshape(side, side, -1)
            return DinoFeatures(tokens=mapped.astype(np.float32), grid_height=side, grid_width=side)
        features = self.model(tensor)
        mapped = features[0].detach().float().cpu().permute(1, 2, 0).numpy()
        return DinoFeatures(
            tokens=mapped.astype(np.float32),
            grid_height=mapped.shape[0],
            grid_width=mapped.shape[1],
        )


def _load_encoder(model_name: str, device: torch.device):
    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        model.eval()
        model.to(device)
        patch = int(getattr(model, "patch_size", 14))
        return model_name, model, 37 * patch
    except Exception:
        pass
    try:
        from transformers import Dinov2Model

        hf_name = "facebook/dinov2-small" if "vits" in model_name else "facebook/dinov2-base"
        model = Dinov2Model.from_pretrained(hf_name)
        model.eval()
        model.to(device)

        class _HFWrapper(torch.nn.Module):
            def __init__(self, inner: Dinov2Model) -> None:
                super().__init__()
                self.inner = inner

            def forward_features(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
                outputs = self.inner(pixel_values=tensor)
                return {"x_norm_patchtokens": outputs.last_hidden_state[:, 1:, :]}

        return f"transformers_{hf_name}", _HFWrapper(model), 518
    except Exception:
        pass
    from torchvision.models import resnet18

    try:
        from torchvision.models import ResNet18_Weights

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        backbone = resnet18(pretrained=True)
    encoder = torch.nn.Sequential(*list(backbone.children())[:-2])
    encoder.eval()
    encoder.to(device)
    return "resnet18_imagenet_fallback", encoder, 512


def downsample_mask(mask: NDArray[np.bool_], *, height: int, width: int) -> NDArray[np.bool_]:
    """Mark a feature cell true when any 512-pixel inside it is true."""

    if mask.shape != (512, 512):
        raise ValueError("mask must be 512x512")
    cell_h = 512 / height
    cell_w = 512 / width
    reduced = np.zeros((height, width), dtype=bool)
    for row in range(height):
        top = int(np.floor(row * cell_h))
        bottom = int(np.ceil((row + 1) * cell_h))
        for column in range(width):
            left = int(np.floor(column * cell_w))
            right = int(np.ceil((column + 1) * cell_w))
            reduced[row, column] = bool(mask[top:bottom, left:right].any())
    return reduced

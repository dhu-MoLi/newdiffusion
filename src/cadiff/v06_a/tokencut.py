"""Official a2 object support: TokenCut-style spectral bipartition of frozen patch tokens.

Defines S_x^0 only. Not m_proposal, not m_final, not an external segmenter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from scipy.ndimage import label

from cadiff.v06_a.structure import _fill_binary_holes, _largest_connected_component
from cadiff.v06_atlas_diag.features import FrozenDinoEncoder

_RIDGE = 1e-3
RELATIVE_AREA_TAU = 0.2
_EIGHT = np.ones((3, 3), dtype=np.uint8)


def _upsample_mask(grid: NDArray[np.bool_], size: int = 512) -> NDArray[np.bool_]:
    image = Image.fromarray(grid.astype(np.uint8) * 255)
    return np.asarray(image.resize((size, size), resample=Image.Resampling.NEAREST)) > 127


def _finalize_largest_component(raw: NDArray[np.bool_], valid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Historical TokenCut finalize (through 2026-09-16): keep the largest 8-connected CC."""

    clipped = np.asarray(raw, dtype=bool) & np.asarray(valid, dtype=bool)
    if not clipped.any():
        return np.zeros(valid.shape, dtype=bool)
    component = _largest_connected_component(clipped)
    filled = _fill_binary_holes(component) & valid
    return filled if filled.any() else component


def _relative_area_union(clipped: NDArray[np.bool_], *, tau: float) -> NDArray[np.bool_]:
    binary = np.asarray(clipped, dtype=bool)
    kept = np.zeros(binary.shape, dtype=bool)
    if not binary.any():
        return kept
    labels, count = label(binary, structure=_EIGHT)
    areas = [int((labels == raw_id).sum()) for raw_id in range(1, count + 1)]
    largest = float(max(areas))
    for raw_id, area in enumerate(areas, start=1):
        if float(area) >= tau * largest:
            kept |= labels == raw_id
    return kept


def _finalize_object(
    raw: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    *,
    tau: float = RELATIVE_AREA_TAU,
) -> NDArray[np.bool_]:
    """Official finalize: union CCs with area >= tau * largest, then fill holes."""

    if not np.isfinite(tau) or tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must be in (0, 1]")
    clipped = np.asarray(raw, dtype=bool) & np.asarray(valid, dtype=bool)
    if not clipped.any():
        return np.zeros(valid.shape, dtype=bool)
    union = _relative_area_union(clipped, tau=tau)
    filled = _fill_binary_holes(union) & valid
    return filled if filled.any() else union


def _less_border_partition(flags: NDArray[np.bool_]) -> NDArray[np.bool_]:
    height, width = flags.shape
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    other = ~flags
    border_a = int((flags & border).sum())
    border_b = int((other & border).sum())
    if border_a != border_b:
        return flags if border_a < border_b else other
    area_a = int(flags.sum())
    area_b = int(other.sum())
    if area_a != area_b:
        return flags if area_a < area_b else other
    return flags


def tokencut_bipartition_upsampled(
    tokens: NDArray[np.floating],
    size: int = 512,
) -> NDArray[np.bool_]:
    """Upsampled spectral bipartition before relative-area finalize. Not a label."""

    grid = np.asarray(tokens, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError("tokencut expects (H, W, C) tokens")
    height, width, channels = grid.shape
    del channels
    vectors = grid.reshape(height * width, grid.shape[-1])
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    desc = vectors / np.maximum(norms, 1e-12)
    affinity = np.clip(desc @ desc.T, 0.0, None)
    np.fill_diagonal(affinity, 0.0)
    affinity = affinity + _RIDGE
    degree = affinity.sum(axis=1)
    scale = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = (scale[:, None] * affinity) * scale[None, :]
    _vals, vecs = np.linalg.eigh(
        np.eye(height * width, dtype=np.float64) - normalized.astype(np.float64)
    )
    partition = _less_border_partition((vecs[:, 1] > 0.0).reshape(height, width))
    return _upsample_mask(partition, size=size)


def tokencut_object_mask(
    tokens: NDArray[np.floating],
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Official S_x^0: spectral bipartition then relative-area union (tau=0.2) + holes."""

    return _finalize_object(tokencut_bipartition_upsampled(tokens), valid)


def load_official_dinov2_encoder(*, device: str | None = None) -> FrozenDinoEncoder:
    """Load cached facebookresearch DINOv2 without Hugging Face; refuse ResNet fallback."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    hub_dir = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    if hub_dir.is_dir():
        model = torch.hub.load(str(hub_dir), "dinov2_vits14", source="local", pretrained=True)
        model.eval()
        model.to(torch_device)
        encoder = FrozenDinoEncoder.__new__(FrozenDinoEncoder)
        encoder.device = torch_device
        encoder.backend = "dinov2_vits14"
        encoder.model = model
        encoder.input_size = 37 * int(getattr(model, "patch_size", 14))
        encoder.model_name = "dinov2_vits14"
        return encoder
    encoder = FrozenDinoEncoder(device=device)
    backend = str(getattr(encoder, "backend", encoder.model_name))
    if not backend.startswith("dinov2") and not backend.startswith("transformers_"):
        raise RuntimeError("official a2 TokenCut requires DINOv2; encoder fell back to ResNet")
    return encoder

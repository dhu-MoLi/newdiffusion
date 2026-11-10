"""Reconstruction backends for the A2 material prototype.

The baseline fills the source defect hole with a per-pixel nearest-neighbour
copy. These backends make that step swappable so the proposal diagnostic can
compare classical, atlas-based and learned completion without touching v06.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source

IMAGE_SIZE = 512


class Reconstructor(Protocol):
    name: str

    def __call__(
        self, rgb: np.ndarray, hole: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        ...


def _validate(
    rgb: np.ndarray, hole: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("rgb must be 512x512 uint8 RGB")
    if hole.shape != (IMAGE_SIZE, IMAGE_SIZE) or hole.dtype != bool:
        raise ValueError("hole must be a 512x512 bool mask")
    if valid.shape != (IMAGE_SIZE, IMAGE_SIZE) or valid.dtype != bool:
        raise ValueError("valid must be a 512x512 bool mask")
    masked = hole & valid
    if not masked.any():
        raise ValueError("hole must intersect valid pixels")
    return masked


@dataclass(frozen=True)
class NeighborReconstructor:
    name: str = "neighbor"

    def __call__(
        self, rgb: np.ndarray, hole: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        masked = _validate(rgb, hole, valid)
        return neighbor_fill_source(rgb, masked, valid)


@dataclass(frozen=True)
class OpenCVReconstructor:
    name: str
    flag: int
    radius: int = 3

    def __post_init__(self) -> None:
        if type(self.radius) is not int or self.radius < 1:
            raise ValueError("radius must be a positive integer")

    def __call__(
        self, rgb: np.ndarray, hole: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        masked = _validate(rgb, hole, valid)
        result = cv2.inpaint(
            np.ascontiguousarray(rgb),
            (masked.astype(np.uint8) * 255),
            self.radius,
            self.flag,
        )
        # Keep pixels outside the hole identical to the source.
        out = rgb.copy()
        out[masked] = result[masked]
        return out


def make_telea(radius: int = 3) -> OpenCVReconstructor:
    return OpenCVReconstructor("telea", cv2.INPAINT_TELEA, radius)


def make_ns(radius: int = 3) -> OpenCVReconstructor:
    return OpenCVReconstructor("ns", cv2.INPAINT_NS, radius)


def _align_reference(
    reference: np.ndarray,
    source: np.ndarray,
    source_valid: np.ndarray,
    reference_valid: np.ndarray | None,
) -> np.ndarray:
    """Warp a normal reference to the source frame with ECC (affine)."""

    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    template = source_gray.copy()
    template[~source_valid] = 0.0
    moving = reference_gray.copy()
    if reference_valid is not None:
        moving[~reference_valid] = 0.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        60,
        1e-4,
    )
    input_mask = None
    if reference_valid is not None:
        input_mask = (reference_valid.astype(np.uint8) * 255)
    try:
        _cc, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            input_mask,
            5,
        )
    except cv2.error:
        warp = np.eye(2, 3, dtype=np.float32)
    return cv2.warpAffine(
        reference,
        warp,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


@dataclass(frozen=True)
class AtlasReconstructor:
    """Copy the corresponding normal material from a same-category reference."""

    references: tuple[np.ndarray, ...]
    reference_valids: tuple[np.ndarray, ...] | None = None
    name: str = "atlas"
    cache_alignment: bool = False

    def __post_init__(self) -> None:
        if not self.references:
            raise ValueError("atlas reconstruction requires references")
        if self.reference_valids is not None and len(self.reference_valids) != len(
            self.references
        ):
            raise ValueError("reference_valids must match references")

    def __call__(
        self, rgb: np.ndarray, hole: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        masked = _validate(rgb, hole, valid)
        outside = valid & ~masked
        if not outside.any():
            outside = ~masked
        best: tuple[float, np.ndarray] | None = None
        for index, reference in enumerate(self.references):
            reference_valid = (
                None
                if self.reference_valids is None
                else self.reference_valids[index]
            )
            aligned = _align_reference(reference, rgb, valid, reference_valid)
            error = float(
                np.abs(
                    aligned[outside].astype(np.float32)
                    - rgb[outside].astype(np.float32)
                ).mean()
            )
            if best is None or error < best[0]:
                best = (error, aligned)
        if best is None:
            return neighbor_fill_source(rgb, masked, valid)
        out = rgb.copy()
        out[masked] = best[1][masked]
        return out


@dataclass
class SD2InpaintReconstructor:
    """Lazy Stable Diffusion 2 inpainting backend (GPU recommended)."""

    model_root: Path
    device: str = "cuda:0"
    prompt: str = ""
    steps: int = 50
    guidance: float = 5.0
    name: str = "sd2"
    _cached_pipeline: object | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _pipeline(self):
        if self._cached_pipeline is not None:
            return self._cached_pipeline
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_root,
            local_files_only=True,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        self._cached_pipeline = pipeline
        return pipeline

    def __call__(
        self, rgb: np.ndarray, hole: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        import torch
        from PIL import Image

        masked = _validate(rgb, hole, valid)
        pipeline = self._pipeline()
        with torch.inference_mode():
            image = pipeline(
                prompt=self.prompt,
                image=Image.fromarray(rgb),
                mask_image=Image.fromarray((masked.astype(np.uint8) * 255)),
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
            ).images[0]
        generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
        out = rgb.copy()
        out[masked] = generated[masked]
        return out


def make_reconstructor(
    name: str,
    *,
    references: Sequence[np.ndarray] | None = None,
    reference_valids: Sequence[np.ndarray] | None = None,
    sd2_root: Path | None = None,
    learned_checkpoint: Path | None = None,
    device: str | None = None,
    prompt: str = "",
    steps: int = 50,
    guidance: float = 5.0,
) -> Reconstructor:
    if name == "neighbor":
        return NeighborReconstructor()
    if name == "telea":
        return make_telea()
    if name == "ns":
        return make_ns()
    if name == "atlas":
        if not references:
            raise ValueError("atlas backend requires references")
        return AtlasReconstructor(
            references=tuple(np.asarray(item, dtype=np.uint8) for item in references),
            reference_valids=(
                None
                if reference_valids is None
                else tuple(np.asarray(item, dtype=bool) for item in reference_valids)
            ),
        )
    if name == "sd2":
        if sd2_root is None:
            raise ValueError("sd2 backend requires sd2_root")
        return SD2InpaintReconstructor(
            model_root=Path(sd2_root),
            device=device or "cuda:0",
            prompt=prompt,
            steps=steps,
            guidance=guidance,
        )
    if name == "learned":
        if learned_checkpoint is None:
            raise ValueError("learned backend requires learned_checkpoint")
        from cadiff.v07.reconstructor import LearnedReconstructor

        return LearnedReconstructor(
            Path(learned_checkpoint), device=device or "cuda:0"
        )
    raise ValueError(f"unknown reconstruction backend: {name}")


def ring_mask(
    hole: np.ndarray,
    valid: np.ndarray,
    *,
    radius: int = 12,
) -> np.ndarray:
    """Normal annulus around the hole; used by the no-reconstruction baseline."""

    if hole.shape != (IMAGE_SIZE, IMAGE_SIZE) or hole.dtype != bool:
        raise ValueError("hole must be a 512x512 bool mask")
    if valid.shape != (IMAGE_SIZE, IMAGE_SIZE) or valid.dtype != bool:
        raise ValueError("valid must be a 512x512 bool mask")
    if type(radius) is not int or radius < 1:
        raise ValueError("radius must be a positive integer")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    dilated = cv2.dilate(hole.astype(np.uint8), kernel) > 0
    ring = dilated & ~hole & valid
    if not ring.any():
        ring = valid & ~hole
    return np.ascontiguousarray(ring)


__all__ = [
    "AtlasReconstructor",
    "IMAGE_SIZE",
    "NeighborReconstructor",
    "OpenCVReconstructor",
    "Reconstructor",
    "SD2InpaintReconstructor",
    "make_ns",
    "make_reconstructor",
    "make_telea",
    "ring_mask",
]

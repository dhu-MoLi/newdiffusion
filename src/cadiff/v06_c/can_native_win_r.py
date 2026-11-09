"""Train can exact-mask R on native 512 windows around few-shot GT.

Does not load the letterbox-512 can R. Ranking is still frozen normal U-Net.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p2_a0.fewshot import FewshotSample, FewshotSplit
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config
from cadiff.provenance import sha256_file
from cadiff.v06_c.can_exact_mask_r import (
    CanExactMaskRenderer,
    can_p2_a1_config,
    load_can_r_modules,
    train_can_exact_mask_r,
)
from cadiff.v06_c.can_mvtec2 import CATEGORY, DEFECT_TYPE
from cadiff.v06_c.can_native_window import (
    WINDOW_MARGIN,
    WINDOW_SIZE,
    extract_window,
    load_native_mask,
    load_native_rgb,
    window_from_mask,
)

NATIVE_WIN_CHECKPOINT_SCHEMA = "v06_c_can_native_win_r_checkpoint_v1"
NATIVE_WIN_SUMMARY_SCHEMA = "v06_c_can_native_win_r_v1"


def _load_native_win_sample(
    raw: dict[str, object], root: Path, role: str, config: P2A1Config
) -> FewshotSample:
    defect_type = raw.get("defect_type")
    image_relpath = raw.get("image_relpath")
    mask_relpath = raw.get("mask_relpath")
    image_sha256 = raw.get("image_sha256")
    mask_sha256 = raw.get("mask_sha256")
    if defect_type != DEFECT_TYPE:
        raise ValueError(f"can native-win R only trains {DEFECT_TYPE}")
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
    rgb = load_native_rgb(image_path)
    mask = load_native_mask(mask_path, shape=rgb.shape[:2])
    window = window_from_mask(mask, margin=WINDOW_MARGIN, size=WINDOW_SIZE)
    crop_rgb, crop_mask = extract_window(rgb, mask, window)
    if crop_rgb.shape != (config.image_size, config.image_size, 3):
        raise ValueError("native window RGB must be 512x512")
    pixels = (
        torch.from_numpy(crop_rgb)
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(127.5)
        .sub_(1.0)
        .contiguous()
    )
    mask_tensor = (
        torch.from_numpy(crop_mask.astype(np.uint8, copy=False))
        .to(dtype=torch.float32)
        .unsqueeze(0)
        .contiguous()
    )
    if not bool(mask_tensor.any()):
        raise ValueError(f"empty can mask after native window: {image_relpath}")
    sample_id = hashlib.sha256(
        f"{config.category}\nnative-win\n{defect_type}\n{image_relpath}".encode()
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
        mask=mask_tensor,
    )


def load_can_native_win_fewshot_split(
    manifest: Path, dataset_root: Path, config: P2A1Config
) -> FewshotSplit:
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
            selected[matched[0]].append(_load_native_win_sample(raw, root, matched[0], config))
    train = tuple(sorted(selected[config.train_role], key=lambda item: item.image_relpath))
    evaluation = tuple(sorted(selected[config.eval_role], key=lambda item: item.image_relpath))
    if len(train) != 30 or len(evaluation) != 60:
        raise ValueError(f"can few-shot split must be 30/60, got {len(train)}/{len(evaluation)}")
    if any(item.defect_type != DEFECT_TYPE for item in (*train, *evaluation)):
        raise ValueError("can few-shot split must be type bad only")
    return FewshotSplit(train=train, evaluation=evaluation)


def train_can_native_win_r(
    *,
    paths_yaml: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    device: str | None = None,
) -> dict[str, Any]:
    return train_can_exact_mask_r(
        paths_yaml=paths_yaml,
        manifest=manifest,
        dataset_root=dataset_root,
        output=output,
        device=device,
        load_split=load_can_native_win_fewshot_split,
        checkpoint_schema=NATIVE_WIN_CHECKPOINT_SCHEMA,
        summary_schema=NATIVE_WIN_SUMMARY_SCHEMA,
        extra_metadata={
            "geometry": "native_window",
            "window_margin": WINDOW_MARGIN,
            "window_size": WINDOW_SIZE,
            "not_letterbox_r": True,
        },
        stage_name="can_native_win_r",
    )


def train_can_native_win_r_from_cli(
    *,
    paths: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    del cwd
    return train_can_native_win_r(
        paths_yaml=paths,
        manifest=manifest,
        dataset_root=dataset_root,
        output=output,
        device=device,
    )


def load_can_native_win_renderer(
    *,
    checkpoint: Path,
    model_root: Path,
    device: str | None = None,
) -> CanExactMaskRenderer:
    config = can_p2_a1_config()
    torch_device = torch.device(device or "cuda:0")
    components = load_frozen_components(model_root, torch_device, torch.float16)
    _, conditioner, adapter, _metadata = load_can_r_modules(
        checkpoint,
        config=config,
        device=torch_device,
        expected_schema=NATIVE_WIN_CHECKPOINT_SCHEMA,
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).eval()
    prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
    return CanExactMaskRenderer(
        components=components,
        model=model,
        prompt=prompt,
        device=torch_device,
    )


__all__ = [
    "NATIVE_WIN_CHECKPOINT_SCHEMA",
    "load_can_native_win_fewshot_split",
    "load_can_native_win_renderer",
    "train_can_native_win_r",
    "train_can_native_win_r_from_cli",
]

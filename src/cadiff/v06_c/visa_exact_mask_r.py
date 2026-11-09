"""Independent VisA exact-mask R. Not official P2-A1 and not Cable/can R.

Trains a fresh SD1.5 adapter per VisA object category on DualAnoDiff few-shot
(floor(n/3) lexicographic). Ranking uses cyclic_next plus frozen normal U-Net;
single-type capsules therefore rank only against the frozen normal U-Net.
Does not modify frozen P2-A1 yaml profiles or borrow Cable/Hazelnut/can weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image

from cadiff.data.geometry import resize_pad_image, resize_pad_mask
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p2_a0.fewshot import FewshotSample, FewshotSplit
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config
from cadiff.provenance import sha256_file
from cadiff.v06_c.can_exact_mask_r import (
    CanExactMaskRenderer,
    load_can_r_modules,
    train_can_exact_mask_r,
)
from cadiff.v06_c.visa_pilot import (
    COLLAPSED_TYPE,
    IMAGE_SIZE,
    MAGIC_CATEGORIES,
    load_visa_inventory_for_split,
)

CHECKPOINT_SCHEMA = "v06_c_visa_exact_mask_r_checkpoint_v1"
SUMMARY_SCHEMA = "v06_c_visa_exact_mask_r_v1"


def visa_p2_a1_config(category: str, defect_types: tuple[str, ...]) -> P2A1Config:
    from cadiff.v06_c.mvtec_pilot import MVTEC_CATEGORIES

    if category not in MAGIC_CATEGORIES and category not in MVTEC_CATEGORIES:
        raise ValueError(
            f"R category must stay in VisA {MAGIC_CATEGORIES} or MVTec {MVTEC_CATEGORIES}, got {category}"
        )
    if not defect_types or defect_types != tuple(sorted(defect_types)):
        raise ValueError("VisA defect_types must be a non-empty lexicographic tuple")
    return P2A1Config(
        schema_version="p2_a1_factorized_v1",
        category=category,
        defect_types=defect_types,
        normal_prompt=f"a photo of a normal {category}",
        image_size=IMAGE_SIZE,
        dtype="float16",
        seed=20260913,
        learning_rate=1e-4,
        steps=1200,
        capacity_defect_type=defect_types[0],
        context_fill_seed=202609130,
        noise_seed=202609131,
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
    if defect_type not in config.defect_types:
        raise ValueError(f"VisA R vocabulary does not include {defect_type}")
    if not isinstance(image_relpath, str) or not image_relpath:
        raise ValueError("image_relpath must be a non-empty string")
    if not isinstance(mask_relpath, str) or not mask_relpath:
        raise ValueError("mask_relpath must be a non-empty string")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("image_sha256 must be a SHA256 string")
    if not isinstance(mask_sha256, str) or len(mask_sha256) != 64:
        raise ValueError("mask_sha256 must be a SHA256 string")
    image_path = root.joinpath(*PurePosixPath(image_relpath).parts)
    mask_path = root.joinpath(*PurePosixPath(mask_relpath).parts)
    if ".." in PurePosixPath(image_relpath).parts or ".." in PurePosixPath(mask_relpath).parts:
        raise ValueError("VisA few-shot paths must stay under the dataset root")
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(image_relpath)
    if sha256_file(image_path) != image_sha256 or sha256_file(mask_path) != mask_sha256:
        raise ValueError("VisA few-shot SHA256 mismatch")
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
        raise ValueError(f"empty VisA mask after letterbox: {image_relpath}")
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


def load_visa_fewshot_split(manifest: Path, dataset_root: Path, config: P2A1Config) -> FewshotSplit:
    root = dataset_root.resolve(strict=True)
    selected: dict[str, list[FewshotSample]] = {config.train_role: [], config.eval_role: []}
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("category") != config.category:
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
    if not train or not evaluation:
        raise ValueError(
            f"VisA few-shot split is empty: train={len(train)} eval={len(evaluation)}"
        )
    train_types = {item.defect_type for item in train}
    eval_types = {item.defect_type for item in evaluation}
    if train_types != set(config.defect_types) or eval_types != set(config.defect_types):
        raise ValueError("VisA few-shot split must cover the configured vocabulary")
    return FewshotSplit(train=train, evaluation=evaluation)


def load_visa_renderer(
    *,
    checkpoint: Path,
    model_root: Path,
    config: P2A1Config,
    device: str | None = None,
) -> CanExactMaskRenderer:
    torch_device = torch.device(device or "cuda:0")
    components = load_frozen_components(model_root, torch_device, torch.float16)
    _, conditioner, adapter, _metadata = load_can_r_modules(
        checkpoint,
        config=config,
        device=torch_device,
        expected_schema=CHECKPOINT_SCHEMA,
    )
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).eval()
    prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
    return CanExactMaskRenderer(
        components=components,
        model=model,
        prompt=prompt,
        device=torch_device,
        allowed_types=config.defect_types,
    )


def _visa_r_inventory(
    dataset_root: Path, category: str, collapse_type: str | None, split: str | None = None
):
    return load_visa_inventory_for_split(dataset_root, category, split, collapse_type)


def train_visa_exact_mask_r(
    *,
    paths_yaml: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    category: str,
    device: str | None = None,
    collapse_type: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    inventory = _visa_r_inventory(dataset_root, category, collapse_type, split)
    config = visa_p2_a1_config(category, tuple(item.name for item in inventory.types))
    ranking = (
        "frozen_normal_unet"
        if len(config.defect_types) == 1
        else "cyclic_next_and_frozen_normal_unet"
    )
    extra = {
        "dataset": "VisA",
        "not_native_win": True,
        "ranking_negative": ranking,
        "split_id": inventory.split_id,
    }
    if inventory.collapsed_from is not None:
        extra["collapsed_from"] = list(inventory.collapsed_from)
        extra["collapsed_type"] = COLLAPSED_TYPE
    summary = train_can_exact_mask_r(
        paths_yaml=paths_yaml,
        manifest=manifest,
        dataset_root=dataset_root,
        output=output,
        device=device,
        load_split=load_visa_fewshot_split,
        checkpoint_schema=CHECKPOINT_SCHEMA,
        summary_schema=SUMMARY_SCHEMA,
        extra_metadata=extra,
        stage_name="visa_exact_mask_r",
        config=config,
    )
    return summary


def train_visa_exact_mask_r_from_cli(
    *,
    paths: Path,
    manifest: Path,
    dataset_root: Path,
    output: Path,
    category: str,
    cwd: Path,
    device: str | None = None,
    collapse_type: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    del cwd
    return train_visa_exact_mask_r(
        paths_yaml=paths,
        manifest=manifest,
        dataset_root=dataset_root,
        output=output,
        category=category,
        device=device,
        collapse_type=collapse_type,
        split=split,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "SUMMARY_SCHEMA",
    "load_visa_fewshot_split",
    "load_visa_renderer",
    "train_visa_exact_mask_r",
    "train_visa_exact_mask_r_from_cli",
    "visa_p2_a1_config",
]

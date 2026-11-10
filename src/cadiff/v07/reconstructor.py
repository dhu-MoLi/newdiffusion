"""Category-specific masked reconstructor.

Trained only on normal images: B0 (the parametric defect-shape prior) punches
defect-shaped holes, and the original normal pixels provide the supervision.
The loss combines masked L1, a gradient term and a frozen DINOv2 feature term
so the completion is optimised for the A2 material prototype, not just pixels.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_atlas_diag.features import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    downsample_mask,
)
from cadiff.v06_c.visa_pilot import (
    fewshot_identity_records,
    load_visa_identity,
    load_visa_inventory_for_split,
    normal_identity_records,
)
from cadiff.v07.reconstruction import IMAGE_SIZE, make_telea
from cadiff.v07.runmeta import write_json
from cadiff.v07.shape_prior import ParametricShapePrior


@dataclass(frozen=True)
class ReconstructorConfig:
    image_size: int = IMAGE_SIZE
    base_channels: int = 48
    depth: int = 4
    steps: int = 6000
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20261110
    val_fraction: float = 0.1
    real_mask_fraction: float = 0.5
    feature_every: int = 10
    feature_weight: float = 0.1
    gradient_weight: float = 0.05
    feature_batch: int = 2
    val_size: int = 32
    log_every: int = 50
    val_every: int = 500
    device: str = "cuda:0"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.block(features)


class MaskedUNet(nn.Module):
    """RGB + hole mask in, RGB completion out. The caller composites outside."""

    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 48,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least two")
        channels = [base_channels * (2**index) for index in range(depth + 1)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for channel in channels[:-1]:
            self.encoders.append(ConvBlock(previous, channel))
            previous = channel
        self.bottleneck = ConvBlock(channels[-2], channels[-1])
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(depth - 1, -1, -1):
            self.ups.append(
                nn.ConvTranspose2d(channels[index + 1], channels[index], 2, stride=2)
            )
            self.decoders.append(ConvBlock(channels[index] * 2, channels[index]))
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, rgb: Tensor, hole: Tensor) -> Tensor:
        features = torch.cat((rgb, hole), dim=1)
        skips: list[Tensor] = []
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = F.avg_pool2d(features, 2)
        features = self.bottleneck(features)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            features = up(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(
                    features, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            features = torch.cat((features, skip), dim=1)
            features = decoder(features)
        return torch.tanh(self.head(features))


def _crop_to_content(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(binary)
    if rows.size == 0:
        raise ValueError("empty shape")
    return np.ascontiguousarray(
        binary[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    )


def place_shape_in_support(
    shape: np.ndarray,
    support: np.ndarray,
    valid: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray | None:
    """Place a B0 shape fully inside the object support; None when impossible."""

    try:
        component = _crop_to_content(shape)
    except ValueError:
        return None
    rows, columns = np.nonzero(support & valid)
    if rows.size == 0:
        return None
    top_bound = int(rows.min())
    bottom_bound = int(rows.max()) + 1
    left_bound = int(columns.min())
    right_bound = int(columns.max()) + 1
    height, width = component.shape
    if height > bottom_bound - top_bound or width > right_bound - left_bound:
        scale = min(
            (bottom_bound - top_bound) / height,
            (right_bound - left_bound) / width,
        ) * 0.9
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        component = cv2.resize(
            component.astype(np.uint8),
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        height, width = component.shape
    for _ in range(60):
        if bottom_bound - height + 1 <= top_bound:
            top = top_bound
        else:
            top = int(rng.randint(top_bound, bottom_bound - height + 1))
        if right_bound - width + 1 <= left_bound:
            left = left_bound
        else:
            left = int(rng.randint(left_bound, right_bound - width + 1))
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        canvas[top : top + height, left : left + width] = component
        outside_support = int((canvas & ~support).sum())
        outside_valid = int((canvas & ~valid).sum())
        if outside_valid == 0 and outside_support <= max(1, int(0.01 * canvas.sum())):
            return canvas
    # Deterministic fallback: a small disk at the support centroid.
    center_y = int(rows.mean())
    center_x = int(columns.mean())
    radius = max(4, int(round(math.sqrt(max(int(support.sum()), 1) / math.pi) * 0.05)))
    disk = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cv2.circle(disk, (center_x, center_y), radius, 1, -1)
    disk = disk > 0
    disk &= support & valid
    return disk if disk.any() else None


def sample_hole(
    prior: ParametricShapePrior,
    defect_types: Sequence[str],
    support: np.ndarray,
    valid: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    for _ in range(40):
        defect_type = defect_types[int(rng.randint(0, len(defect_types)))]
        shape = prior.sample(defect_type, int(rng.randint(0, 2**31 - 1)))
        placed = place_shape_in_support(shape, support, valid, rng)
        if placed is not None and placed.sum() >= 64:
            return placed
    raise RuntimeError("could not place a training hole inside the support")


class NormalHoleDataset(Dataset):
    def __init__(
        self,
        *,
        images: tuple[np.ndarray, ...],
        supports: tuple[np.ndarray, ...],
        valids: tuple[np.ndarray, ...],
        prior: ParametricShapePrior,
        defect_types: Sequence[str],
        length: int,
        seed: int,
        real_masks: Sequence[np.ndarray] = (),
        real_mask_fraction: float = 0.0,
    ) -> None:
        if not images or len(images) != len(supports) or len(images) != len(valids):
            raise ValueError("images, supports and valids must be non-empty and aligned")
        if not 0.0 <= float(real_mask_fraction) <= 1.0:
            raise ValueError("real_mask_fraction must be in [0, 1]")
        self.images = images
        self.supports = supports
        self.valids = valids
        self.prior = prior
        self.defect_types = tuple(defect_types)
        self.length = int(length)
        self.seed = int(seed)
        self.real_masks = tuple(
            np.ascontiguousarray(mask, dtype=bool) for mask in real_masks
        )
        self.real_mask_fraction = float(real_mask_fraction)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        rng = np.random.RandomState((self.seed + index) % (2**32))
        image_index = index % len(self.images)
        rgb = self.images[image_index]
        support = self.supports[image_index]
        valid = self.valids[image_index]
        hole = None
        if self.real_masks and rng.rand() < self.real_mask_fraction:
            shape = self.real_masks[int(rng.randint(0, len(self.real_masks)))]
            hole = place_shape_in_support(shape, support, valid, rng)
            if hole is None or int(hole.sum()) < 64:
                hole = None
        if hole is None:
            hole = sample_hole(self.prior, self.defect_types, support, valid, rng)
        if rng.rand() < 0.5:
            rgb = rgb[:, ::-1]
            hole = hole[:, ::-1]
        if rng.rand() < 0.5:
            rgb = rgb[::-1]
            hole = hole[::-1]
        image = (
            torch.from_numpy(np.array(rgb, copy=True))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(127.5)
            .sub_(1.0)
        )
        mask = torch.from_numpy(np.ascontiguousarray(hole)).to(dtype=torch.float32).unsqueeze(0)
        return image, mask


class DinoFeatureLoss(nn.Module):
    def __init__(self, encoder: Any, device: torch.device) -> None:
        super().__init__()
        if not str(encoder.backend).startswith("dinov2"):
            raise ValueError("DINOv2 encoder is required for the feature loss")
        self.model = encoder.model
        self.input_size = int(encoder.input_size)
        self.device = device
        self.register_buffer(
            "mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, composite: Tensor, target: Tensor, hole: Tensor) -> Tensor:
        def prepare(image: Tensor) -> Tensor:
            values = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
            values = F.interpolate(
                values,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
            return (values - self.mean) / self.std

        source = prepare(composite)
        reference = prepare(target)
        tokens_source = self.model.forward_features(source)["x_norm_patchtokens"]
        with torch.no_grad():
            tokens_reference = self.model.forward_features(reference)["x_norm_patchtokens"]
        tokens_source = F.normalize(tokens_source, dim=-1)
        tokens_reference = F.normalize(tokens_reference, dim=-1)
        side = int(round(tokens_source.shape[1] ** 0.5))
        weights = F.adaptive_avg_pool2d(hole, (side, side)).flatten(1)
        selected = weights > 0.25
        if int(selected.sum()) == 0:
            return composite.sum() * 0.0
        cosine = (tokens_source * tokens_reference).sum(dim=-1)
        return (1.0 - cosine)[selected].mean()


def _masked_l1(prediction: Tensor, target: Tensor, hole: Tensor) -> Tensor:
    channels = prediction.shape[1]
    return ((prediction - target).abs() * hole).sum() / (
        hole.sum() * channels
    ).clamp_min(1e-6)


def _gradient(image: Tensor) -> Tensor:
    horizontal = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
    vertical = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
    return horizontal + vertical


def _to_uint8(image: Tensor) -> np.ndarray:
    values = ((image.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round()
    return values.to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _fast_neighbor_fill(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Linear-time nearest valid pixel fill (for validation baselines)."""

    source = ~hole
    if not source.any():
        return rgb.copy()
    _distances, indices = distance_transform_edt(
        ~source, return_distances=True, return_indices=True
    )
    filled = rgb.copy()
    filled[hole] = rgb[indices[0][hole], indices[1][hole]]
    return filled


def _feature_cosine(
    encoder: Any, composite: Tensor, target: Tensor, hole: Tensor
) -> float:
    source = encoder.encode(_to_uint8(composite))
    reference = encoder.encode(_to_uint8(target))
    source_tokens = source.tokens.reshape(-1, source.tokens.shape[-1])
    reference_tokens = reference.tokens.reshape(-1, reference.tokens.shape[-1])
    source_tokens = source_tokens / np.maximum(
        np.linalg.norm(source_tokens, axis=-1, keepdims=True), 1e-12
    )
    reference_tokens = reference_tokens / np.maximum(
        np.linalg.norm(reference_tokens, axis=-1, keepdims=True), 1e-12
    )
    mask_grid = downsample_mask(
        hole.detach().cpu().numpy()[0] > 0.5,
        height=source.grid_height,
        width=source.grid_width,
    ).reshape(-1)
    if not mask_grid.any():
        return float("nan")
    cosine = (source_tokens * reference_tokens).sum(axis=-1)
    return float(cosine[mask_grid].mean())


@dataclass(frozen=True)
class ValidationMetrics:
    masked_l1: float
    psnr: float
    feature_cos: float


def evaluate_reconstructor(
    model: MaskedUNet,
    pairs: Sequence[tuple[Tensor, Tensor]],
    encoder: Any,
    device: torch.device,
) -> ValidationMetrics:
    model.eval()
    l1_values: list[float] = []
    psnr_values: list[float] = []
    cosine_values: list[float] = []
    with torch.no_grad():
        for image, hole in pairs:
            image = image.unsqueeze(0).to(device)
            hole = hole.unsqueeze(0).to(device)
            prediction = model(image, hole)
            composite = image * (1.0 - hole) + prediction * hole
            l1 = float(_masked_l1(prediction, image, hole).item())
            mse = float((((composite - image) ** 2) * hole).sum().item() / max(
                float(hole.sum().item()) * 3.0, 1.0
            ))
            psnr = 10.0 * math.log10(4.0 / max(mse, 1e-12))
            l1_values.append(l1)
            psnr_values.append(psnr)
            cosine_values.append(_feature_cosine(encoder, composite[0], image[0], hole[0]))
    model.train()
    return ValidationMetrics(
        masked_l1=float(np.mean(l1_values)),
        psnr=float(np.mean(psnr_values)),
        feature_cos=float(np.nanmean(cosine_values)),
    )


def _baseline_metrics(
    pairs: Sequence[tuple[Tensor, Tensor]], encoder: Any, name: str
) -> ValidationMetrics:
    if name == "neighbor":
        reconstructor = None
    else:
        reconstructor = make_telea()
    l1_values: list[float] = []
    psnr_values: list[float] = []
    cosine_values: list[float] = []
    for image, hole in pairs:
        rgb = _to_uint8(image)
        mask = hole.numpy()[0] > 0.5
        if reconstructor is None:
            filled = _fast_neighbor_fill(rgb, mask)
        else:
            filled = reconstructor(rgb, mask, np.ones_like(mask))
        filled_t = (
            torch.from_numpy(filled)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(127.5)
            .sub_(1.0)
        )
        composite = image * (1.0 - hole) + filled_t * hole
        l1 = float(_masked_l1(filled_t, image, hole).item())
        mse = float((((composite - image) ** 2) * hole).sum().item() / max(
            float(hole.sum().item()) * 3.0, 1.0
        ))
        psnr = 10.0 * math.log10(4.0 / max(mse, 1e-12))
        l1_values.append(l1)
        psnr_values.append(psnr)
        cosine_values.append(_feature_cosine(encoder, composite, image, hole))
    return ValidationMetrics(
        masked_l1=float(np.mean(l1_values)),
        psnr=float(np.mean(psnr_values)),
        feature_cos=float(np.nanmean(cosine_values)),
    )


def _compute_supports(
    images: Sequence[np.ndarray],
    valids: Sequence[np.ndarray],
    encoder: Any,
    cache_path: Path,
) -> tuple[np.ndarray, ...]:
    if cache_path.is_file():
        payload = np.load(cache_path)
        supports = payload["supports"]
        if supports.shape == (len(images), IMAGE_SIZE, IMAGE_SIZE):
            return tuple(np.ascontiguousarray(item, dtype=bool) for item in supports)
    supports: list[np.ndarray] = []
    for index, (rgb, valid) in enumerate(zip(images, valids, strict=True)):
        features = encoder.encode(rgb)
        support = tokencut_object_mask(features.tokens, valid)
        if not support.any():
            support = valid
        supports.append(np.ascontiguousarray(support))
        if (index + 1) % 20 == 0:
            print(f"supports {index + 1}/{len(images)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, supports=np.stack(supports))
    return tuple(supports)


def _build_pairs(
    images: Sequence[np.ndarray],
    supports: Sequence[np.ndarray],
    valids: Sequence[np.ndarray],
    prior: ParametricShapePrior,
    defect_types: Sequence[str],
    *,
    count: int,
    seed: int,
    real_masks: Sequence[np.ndarray] = (),
    real_mask_fraction: float = 0.0,
) -> list[tuple[Tensor, Tensor]]:
    pairs: list[tuple[Tensor, Tensor]] = []
    for index in range(min(count, len(images))):
        rng = np.random.RandomState((seed + index) % (2**32))
        hole = None
        if real_masks and rng.rand() < real_mask_fraction:
            shape = real_masks[int(rng.randint(0, len(real_masks)))]
            hole = place_shape_in_support(
                shape, supports[index], valids[index], rng
            )
            if hole is None or int(hole.sum()) < 64:
                hole = None
        if hole is None:
            hole = sample_hole(
                prior, defect_types, supports[index], valids[index], rng
            )
        image = (
            torch.from_numpy(np.array(images[index], copy=True))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(127.5)
            .sub_(1.0)
        )
        mask = torch.from_numpy(np.ascontiguousarray(hole)).to(dtype=torch.float32).unsqueeze(0)
        pairs.append((image, mask))
    return pairs


def train_reconstructor(
    *,
    category: str,
    paths_yaml: Path,
    split: str,
    output: Path,
    config: ReconstructorConfig,
    supports_cache: Path | None = None,
) -> dict[str, Any]:
    from cadiff.config import load_config

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite reconstructor output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    project = load_config(paths_yaml)
    inventory = load_visa_inventory_for_split(project.dataset_root, category, split, None)

    normal_records = normal_identity_records(inventory)
    images: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    for record in normal_records:
        loaded = load_visa_identity(record, dataset_root=inventory.source_root)
        images.append(np.ascontiguousarray(loaded.rgb))
        valids.append(np.ascontiguousarray(loaded.valid))

    masks_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in fewshot_identity_records(inventory):
        loaded = load_visa_identity(record, dataset_root=inventory.source_root)
        if loaded.anomaly_gt is None:
            continue
        masks_by_type[record.defect_type].append(
            np.ascontiguousarray(np.asarray(loaded.anomaly_gt, dtype=bool))
        )
    defect_types = tuple(sorted(masks_by_type))
    if not defect_types:
        raise ValueError("no few-shot masks for the shape prior")
    raw_masks = tuple(
        mask for defect_type in defect_types for mask in masks_by_type[defect_type]
    )
    prior = ParametricShapePrior(
        masks_by_type,
        size=IMAGE_SIZE,
        elastic_probability=0.5,
        multi_component_probability=0.1,
        max_components=3,
        rng_seed=config.seed,
    )
    encoder = load_official_dinov2_encoder(device=str(device))
    supports = _compute_supports(
        images,
        valids,
        encoder,
        supports_cache if supports_cache is not None else output / "supports.npz",
    )

    order = np.random.RandomState(config.seed).permutation(len(images))
    val_count = max(1, min(len(images) - 1, int(round(len(images) * config.val_fraction))))
    val_indices = order[:val_count]
    train_indices = order[val_count:]
    train_images = tuple(images[index] for index in train_indices)
    train_supports = tuple(supports[index] for index in train_indices)
    train_valids = tuple(valids[index] for index in train_indices)
    val_images = tuple(images[index] for index in val_indices)
    val_supports = tuple(supports[index] for index in val_indices)
    val_valids = tuple(valids[index] for index in val_indices)

    dataset = NormalHoleDataset(
        images=train_images,
        supports=train_supports,
        valids=train_valids,
        prior=prior,
        defect_types=defect_types,
        length=config.steps * config.batch_size,
        seed=config.seed,
        real_masks=raw_masks,
        real_mask_fraction=config.real_mask_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    model = MaskedUNet(
        base_channels=config.base_channels,
        depth=config.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.steps, 1)
    )
    feature_loss = DinoFeatureLoss(encoder, device).to(device)
    val_pairs = _build_pairs(
        val_images,
        val_supports,
        val_valids,
        prior,
        defect_types,
        count=min(config.val_size, len(val_images) * 4),
        seed=config.seed + 999,
        real_masks=raw_masks,
        real_mask_fraction=config.real_mask_fraction,
    )
    print(json.dumps({"val_pairs": len(val_pairs)}), flush=True)
    baseline_neighbor = _baseline_metrics(val_pairs, encoder, "neighbor")
    print(json.dumps({"baseline_neighbor": asdict(baseline_neighbor)}), flush=True)
    baseline_telea = _baseline_metrics(val_pairs, encoder, "telea")
    print(json.dumps({"baseline_telea": asdict(baseline_telea)}), flush=True)
    print(
        json.dumps(
            {
                "baseline_neighbor": asdict(baseline_neighbor),
                "baseline_telea": asdict(baseline_telea),
                "val_pairs": len(val_pairs),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best = {"score": float("inf"), "step": -1, "metrics": None}
    history: list[dict[str, Any]] = []
    started = time.time()
    model.train()
    for step, (image, hole) in enumerate(loader, start=1):
        image = image.to(device, non_blocking=True)
        hole = hole.to(device, non_blocking=True)
        prediction = model(image, hole)
        composite = image * (1.0 - hole) + prediction * hole
        l1 = _masked_l1(prediction, image, hole)
        gradient = _gradient(composite)
        loss = l1 + config.gradient_weight * gradient
        feature_value = 0.0
        if config.feature_every > 0 and step % config.feature_every == 0:
            batch = min(config.feature_batch, image.shape[0])
            feature_term = feature_loss(
                composite[:batch], image[:batch], hole[:batch]
            )
            feature_value = float(feature_term.item())
            loss = loss + config.feature_weight * feature_term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % config.log_every == 0 or step == 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "steps": config.steps,
                        "l1": float(l1.item()),
                        "gradient": float(gradient.item()),
                        "feature": feature_value,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "elapsed": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % config.val_every == 0 or step == config.steps:
            metrics = evaluate_reconstructor(model, val_pairs, encoder, device)
            cosine = metrics.feature_cos if math.isfinite(metrics.feature_cos) else 0.0
            score = metrics.masked_l1 - 0.02 * cosine
            record = {
                "step": step,
                **asdict(metrics),
                "score": score,
            }
            history.append(record)
            print(json.dumps({"validation": record}, sort_keys=True), flush=True)
            if score < best["score"]:
                best = {"score": score, "step": step, "metrics": asdict(metrics)}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(config),
                        "category": category,
                        "step": step,
                        "metrics": asdict(metrics),
                    },
                    output / "checkpoint.pt",
                )
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "category": category,
            "step": config.steps,
            "metrics": best["metrics"],
        },
        output / "final.pt",
    )
    if not (output / "checkpoint.pt").is_file():
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "category": category,
                "step": config.steps,
                "metrics": best["metrics"],
            },
            output / "checkpoint.pt",
        )
    summary = {
        "schema": "v07_reconstructor_v1",
        "category": category,
        "split": split,
        "defect_types": list(defect_types),
        "normal_images": len(images),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "val_pairs": len(val_pairs),
        "config": asdict(config),
        "baseline_neighbor": asdict(baseline_neighbor),
        "baseline_telea": asdict(baseline_telea),
        "best": best,
        "history": history,
        "checkpoint": str(output / "checkpoint.pt"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    write_json(output / "summary.json", summary)
    return summary


class LearnedReconstructor:
    """Inference backend loading a trained category-specific checkpoint."""

    name = "learned"

    def __init__(self, checkpoint: Path, device: str = "cuda:0") -> None:
        payload = torch.load(checkpoint, map_location="cpu")
        self.config = ReconstructorConfig(**payload["config"])
        self.device = torch.device(device)
        self.model = MaskedUNet(
            base_channels=self.config.base_channels,
            depth=self.config.depth,
        ).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.metadata = {key: value for key, value in payload.items() if key != "model"}

    @torch.inference_mode()
    def __call__(
        self, rgb: np.ndarray, hole: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
            raise ValueError("rgb must be 512x512 uint8 RGB")
        mask = np.ascontiguousarray(hole & valid, dtype=bool)
        if not mask.any():
            return rgb.copy()
        image = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
            .div_(127.5)
            .sub_(1.0)
        )
        mask_tensor = (
            torch.from_numpy(mask)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        prediction = self.model(image, mask_tensor)
        composite = image * (1.0 - mask_tensor) + prediction * mask_tensor
        output = (
            ((composite[0].cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
            .copy()
        )
        output[~mask] = rgb[~mask]
        return np.ascontiguousarray(output)


__all__ = [
    "DinoFeatureLoss",
    "LearnedReconstructor",
    "MaskedUNet",
    "NormalHoleDataset",
    "ReconstructorConfig",
    "ValidationMetrics",
    "evaluate_reconstructor",
    "place_shape_in_support",
    "sample_hole",
    "train_reconstructor",
]

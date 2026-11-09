from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional

from cadiff.provenance import sha256_file

from .dataset import ArmDataset, SegmentationExample
from .metrics import BinaryMetrics, PixelMetrics, binary_metrics, pixel_metrics
from .model import focal_cross_entropy


@dataclass(frozen=True)
class ArmResult:
    name: str
    steps: int
    final_loss: float
    pixel: PixelMetrics
    image: BinaryMetrics
    checkpoint_path: Path
    checkpoint_sha256: str

    def metrics_dict(self) -> dict[str, float]:
        return {
            "pixel_auroc": self.pixel.auroc,
            "pixel_ap": self.pixel.average_precision,
            "pixel_f1_max": self.pixel.f1_max,
            "au_pro": self.pixel.au_pro,
            "image_auroc": self.image.auroc,
            "image_ap": self.image.average_precision,
            "image_f1_max": self.image.f1_max,
        }


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in model.state_dict().items()
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    evaluation: tuple[SegmentationExample, ...],
    *,
    device: torch.device,
    max_fpr: float,
    batch_size: int = 8,
) -> tuple[PixelMetrics, BinaryMetrics]:
    if not evaluation:
        raise ValueError("evaluation set must be non-empty")
    model.eval()
    score_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_scores: list[np.ndarray] = []
    image_labels: list[np.ndarray] = []
    for start in range(0, len(evaluation), batch_size):
        items = evaluation[start : start + batch_size]
        images = torch.stack([item.image for item in items]).to(device)
        logits = model(images)
        probability = logits.softmax(dim=1)[:, 1]
        pooled = functional.avg_pool2d(
            probability.unsqueeze(1), kernel_size=21, stride=1, padding=10
        )[:, 0]
        score_maps.append(probability.float().cpu().numpy())
        masks.append(torch.stack([item.mask for item in items]).cpu().numpy())
        image_scores.append(pooled.flatten(1).amax(dim=1).float().cpu().numpy())
        image_labels.append(
            np.asarray([int(item.mask.bool().any()) for item in items], dtype=np.uint8)
        )
    scores = np.concatenate(score_maps)
    labels = np.concatenate(masks).astype(np.uint8)
    return (
        pixel_metrics(scores, labels, max_fpr=max_fpr),
        binary_metrics(np.concatenate(image_scores), np.concatenate(image_labels)),
    )


def train_arm(
    *,
    arm: ArmDataset,
    evaluation: tuple[SegmentationExample, ...],
    model_factory: Callable[[], nn.Module],
    initial_state: Mapping[str, Tensor] | None,
    steps: int,
    batch_size: int,
    learning_rate: float,
    focal_gamma: float,
    max_fpr: float,
    device: torch.device,
    checkpoint_path: Path,
) -> ArmResult:
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    model = model_factory()
    if initial_state is not None:
        model.load_state_dict(dict(initial_state), strict=True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    final_loss = float("nan")
    for step in range(1, steps + 1):
        images, masks, _ = arm.batch(step=step, batch_size=batch_size)
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = focal_cross_entropy(model(images), masks, gamma=focal_gamma)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise RuntimeError("training gradients must be present and finite")
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_cpu_state(model), checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = model_factory()
    reloaded.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)
    reloaded.to(device)
    pixel, image = evaluate_model(
        reloaded, evaluation, device=device, max_fpr=max_fpr, batch_size=batch_size
    )
    return ArmResult(
        name=arm.name,
        steps=steps,
        final_loss=final_loss,
        pixel=pixel,
        image=image,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )


def decide_promotion(
    *,
    real_fs: Mapping[str, float],
    real_plus_gen: Mapping[str, float],
    pixel_ap_delta: float,
    au_pro_delta: float,
    maximum_image_auroc_drop: float,
) -> bool:
    required = {"pixel_ap", "au_pro", "image_auroc"}
    if set(real_fs) != required or set(real_plus_gen) != required:
        raise ValueError(f"promotion metrics must be exactly {sorted(required)}")
    values = tuple(real_fs.values()) + tuple(real_plus_gen.values())
    if not all(np.isfinite(value) for value in values):
        raise ValueError("promotion metrics must be finite")
    return (
        real_plus_gen["pixel_ap"] - real_fs["pixel_ap"] >= pixel_ap_delta
        and real_plus_gen["au_pro"] - real_fs["au_pro"] >= au_pro_delta
        and real_plus_gen["image_auroc"]
        >= real_fs["image_auroc"] - maximum_image_auroc_drop
    )

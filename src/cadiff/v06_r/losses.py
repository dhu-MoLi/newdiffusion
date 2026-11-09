"""GT-interior appearance loss with a complete-stencil edge domain."""
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor
from cadiff.p2_a0.objective import predict_clean_latent
from cadiff.v06_b.appearance import decode_latent_for_training

@dataclass(frozen=True)
class GatedAppearance:
    rgb: Tensor
    edge: Tensor
    decode_count: int
    edge_pixels: int

def freeze_appearance_vae(vae):
    """Call only after legacy FP16 encoding and matched adapter initialization."""
    if not isinstance(vae, torch.nn.Module):
        raise ValueError("VAE must be a torch module")
    vae.to(dtype=torch.float32).eval().requires_grad_(False)
    _validate_appearance_vae(vae)
    return vae

def _validate_appearance_vae(vae):
    if not isinstance(vae, torch.nn.Module):
        raise ValueError("appearance VAE must be a frozen FP32 module")
    parameters = tuple(p for p in vae.parameters() if p.is_floating_point())
    if not parameters or any(p.dtype != torch.float32 or p.requires_grad or p.grad is not None for p in parameters) or any(m.training for m in vae.modules()):
        raise ValueError("appearance VAE must remain eval, frozen, gradient-free FP32")

def _charbonnier(error: Tensor) -> Tensor:
    return (error.square() + 1e-6).sqrt() - 1e-3

def decode_gate(scheduler: Any, timesteps: Tensor, minimum_snr: float) -> Tensor:
    if timesteps.dtype != torch.long or timesteps.ndim != 1 or timesteps.requires_grad:
        raise ValueError("timesteps must be detached long batch vector")
    alphas = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    if torch.any(timesteps < 0) or torch.any(timesteps >= len(alphas)):
        raise ValueError("timestep outside scheduler")
    alpha = alphas[timesteps]
    if not torch.isfinite(alpha).all() or torch.any(alpha <= 0) or torch.any(alpha > 1):
        raise ValueError("invalid scheduler alpha")
    return alpha / (1 - alpha).clamp_min(1e-12) >= minimum_snr

def _mask(mask: Tensor, shape, device, name: str) -> Tensor:
    if mask.shape != shape or mask.device != device or mask.requires_grad or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be detached matching binary mask")
    return mask.float()

def edge_valid_domain(ground_truth: Tensor, valid: Tensor) -> Tensor:
    if ground_truth.ndim != 4 or ground_truth.shape[1] != 1:
        raise ValueError("GT must be B1HW")
    shape = ground_truth.shape
    gt = _mask(ground_truth, shape, ground_truth.device, "GT")
    legal = _mask(valid, shape, ground_truth.device, "valid")
    if torch.any(gt > legal):
        raise ValueError("GT must lie inside valid image domain")
    # Zero padding excludes the image perimeter: every one of nine pixels is GT.
    return (F.conv2d(gt * legal, gt.new_ones(1, 1, 3, 3), padding=1) == 9).float()

def _sobel(image: Tensor) -> Tensor:
    gray = image[:, 0:1] * .299 + image[:, 1:2] * .587 + image[:, 2:3] * .114
    kernel = image.new_tensor(((1, 0, -1), (2, 0, -2), (1, 0, -1))).view(1, 1, 3, 3) / 8
    return torch.cat((F.conv2d(gray, kernel, padding=1), F.conv2d(gray, kernel.transpose(2, 3), padding=1)), 1)

def masked_appearance(rendered: Tensor, target: Tensor, ground_truth: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, int]:
    if rendered.ndim != 4 or rendered.shape[1] != 3 or rendered.shape != target.shape or target.requires_grad or target.device != rendered.device:
        raise ValueError("appearance needs matching RGB tensors and detached target")
    if not torch.isfinite(rendered).all() or not torch.isfinite(target).all():
        raise ValueError("appearance RGB must be finite")
    shape = (rendered.shape[0], 1, *rendered.shape[2:])
    mask = _mask(ground_truth, shape, rendered.device, "GT")
    legal = _mask(valid, shape, rendered.device, "valid")
    domain = edge_valid_domain(mask, legal)
    counts = mask.sum((1, 2, 3))
    if torch.any(counts <= 0):
        raise ValueError("every GT region must be nonempty")
    rgb = (_charbonnier(rendered.float() - target.float()) * mask).sum((1, 2, 3)) / (counts * 3)
    edge_counts = domain.sum((1, 2, 3))
    # Compute edges on raw images; masking before Sobel would invent GT boundaries.
    edges = (_charbonnier(_sobel(rendered.float()) - _sobel(target.float())) * domain).sum((1, 2, 3)) / (edge_counts.clamp_min(1) * 2)
    return rgb.mean(), edges.mean(), int(edge_counts.sum().item())

def gated_appearance_loss(*, vae, scheduler, noisy, prediction, timesteps, target_image, image_gt_mask, image_valid_mask, minimum_snr) -> GatedAppearance:
    _validate_appearance_vae(vae)
    active = decode_gate(scheduler, timesteps, minimum_snr)
    zero = prediction.sum() * 0
    if not active.any():
        return GatedAppearance(zero, zero, 0, 0)
    # Only the selected low-noise batch items ever enter the frozen VAE decode.
    clean = predict_clean_latent(scheduler, noisy[active], prediction[active], timesteps[active])
    rendered = decode_latent_for_training(vae, clean)
    # Per-item SNR weighting follows the frozen Hazelnut appearance schedule.
    alpha = scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)[timesteps]
    weight = (alpha / (1 - alpha).clamp_min(1e-12)).sqrt().clamp(max=1)
    rgb, edge, count = zero, zero, 0
    selected = torch.nonzero(active).flatten()
    for local, original in enumerate(selected):
        r, e, n = masked_appearance(rendered[local:local+1], target_image[original:original+1], image_gt_mask[original:original+1], image_valid_mask[original:original+1])
        rgb = rgb + r * weight[original] / len(timesteps)
        edge = edge + e * weight[original] / len(timesteps)
        count += n
    return GatedAppearance(rgb, edge, int(active.sum().item()), count)

def compose_exact(normal: Tensor, raw: Tensor, mask: Tensor) -> Tensor:
    if normal.shape != raw.shape or normal.ndim != 4:
        raise ValueError("composition RGB shapes differ")
    _mask(mask, (normal.shape[0], 1, *normal.shape[2:]), normal.device, "composition mask")
    return torch.where(mask.bool(), raw, normal)

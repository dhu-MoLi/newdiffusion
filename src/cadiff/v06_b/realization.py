from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class RealizationPrediction:
    logits: Tensor
    q: Tensor


@dataclass(frozen=True)
class RendererRealizationLoss:
    visible: Tensor
    ring: Tensor


@dataclass(frozen=True)
class RealizationHeadLoss:
    mask: Tensor
    mask_bce: Tensor
    mask_dice: Tensor
    consistency: Tensor


@dataclass(frozen=True)
class ModuleBLoss:
    total: Tensor
    diffusion: Tensor
    visible: Tensor
    ring: Tensor
    mask: Tensor
    consistency: Tensor


class _UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, features: Tensor) -> Tensor:
        features = functional.interpolate(
            features, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return self.activation(self.convolution(features))


class RealizationHead(nn.Module):
    """Decode a continuous realization field from the final generation state."""

    def __init__(self, *, feature_channels: int = 320, hidden_channels: int = 64) -> None:
        super().__init__()
        if (
            isinstance(feature_channels, bool)
            or not isinstance(feature_channels, int)
            or feature_channels <= 0
        ):
            raise ValueError("feature_channels must be a positive integer")
        if (
            isinstance(hidden_channels, bool)
            or not isinstance(hidden_channels, int)
            or hidden_channels < 8
            or hidden_channels % 8
        ):
            raise ValueError("hidden_channels must be a positive multiple of eight")
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.latent_stem = nn.Conv2d(8, hidden_channels, 3, padding=1)
        self.feature_stem = nn.Conv2d(feature_channels, hidden_channels, 1)
        self.fusion = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(2 * hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            _UpsampleBlock(hidden_channels, hidden_channels // 2),
            _UpsampleBlock(hidden_channels // 2, hidden_channels // 4),
            _UpsampleBlock(hidden_channels // 4, hidden_channels // 8),
        )
        self.output = nn.Conv2d(hidden_channels // 8, 1, 1)

    def _validate_latent(self, value: object, *, name: str, batch: int | None = None) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 4:
            raise ValueError(f"{name} must be a four-dimensional tensor")
        if value.shape[0] <= 0 or value.shape[1] != 4:
            raise ValueError(f"{name} must have shape [batch, 4, height, width]")
        if batch is not None and value.shape[0] != batch:
            raise ValueError(f"{name} batch must match generated_latent")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain finite floating values")
        return value

    def forward(
        self,
        *,
        generated_latent: Tensor,
        normal_latent: Tensor,
        generation_feature: Tensor,
        proposal: Tensor,
    ) -> RealizationPrediction:
        generated_latent = self._validate_latent(
            generated_latent, name="generated_latent"
        )
        normal_latent = self._validate_latent(
            normal_latent, name="normal_latent", batch=generated_latent.shape[0]
        )
        if normal_latent.shape != generated_latent.shape:
            raise ValueError("normal_latent shape must match generated_latent")
        if (
            not isinstance(generation_feature, Tensor)
            or generation_feature.shape
            != (
                generated_latent.shape[0],
                self.feature_channels,
                *generated_latent.shape[2:],
            )
            or not generation_feature.is_floating_point()
            or not torch.isfinite(generation_feature).all().item()
        ):
            raise ValueError("generation_feature must match the configured latent feature shape")
        expected_proposal_shape = (
            generated_latent.shape[0],
            1,
            generated_latent.shape[2] * 8,
            generated_latent.shape[3] * 8,
        )
        if not isinstance(proposal, Tensor) or proposal.shape != expected_proposal_shape:
            raise ValueError("proposal must have eight times the latent spatial resolution")
        if not torch.isfinite(proposal).all().item() or not torch.all(
            (proposal == 0) | (proposal == 1)
        ).item():
            raise ValueError("proposal must be finite and binary")
        parameter = self.latent_stem.weight
        values = (generated_latent, normal_latent, generation_feature, proposal)
        if any(value.device != parameter.device for value in values):
            raise ValueError("head inputs must be on the realization head device")

        proposal_full = proposal.to(dtype=parameter.dtype)
        # Head losses must not update the renderer through its generated state.
        generated = generated_latent.detach().to(dtype=parameter.dtype)
        normal = normal_latent.detach().to(dtype=parameter.dtype)
        feature = generation_feature.detach().to(dtype=parameter.dtype)
        latent_features = self.latent_stem(torch.cat((generated, normal), dim=1))
        generation_features = self.feature_stem(feature)
        fused = self.fusion(torch.cat((latent_features, generation_features), dim=1))
        logits = self.output(self.decoder(fused))
        q = logits.sigmoid() * proposal_full
        return RealizationPrediction(logits=logits, q=q)

    def forward_without_generation(
        self,
        *,
        normal_latent: Tensor,
        proposal: Tensor,
    ) -> RealizationPrediction:
        """Proposal-only ablation: replace z_r with z_x and zero the U-Net feature."""

        if not isinstance(normal_latent, Tensor) or normal_latent.ndim != 4:
            raise ValueError("normal_latent must be a four-dimensional tensor")
        feature = normal_latent.new_zeros(
            (
                normal_latent.shape[0],
                self.feature_channels,
                *normal_latent.shape[2:],
            )
        )
        return self(
            generated_latent=normal_latent,
            normal_latent=normal_latent,
            generation_feature=feature,
            proposal=proposal,
        )

    def forward_without_unet_feature(
        self,
        *,
        generated_latent: Tensor,
        normal_latent: Tensor,
        proposal: Tensor,
    ) -> RealizationPrediction:
        """Isolate F64 usage while retaining the generated latent state."""

        if not isinstance(generated_latent, Tensor) or generated_latent.ndim != 4:
            raise ValueError("generated_latent must be a four-dimensional tensor")
        feature = generated_latent.new_zeros(
            (
                generated_latent.shape[0],
                self.feature_channels,
                *generated_latent.shape[2:],
            )
        )
        return self(
            generated_latent=generated_latent,
            normal_latent=normal_latent,
            generation_feature=feature,
            proposal=proposal,
        )


def _image_pair(rendered: object, normal: object) -> tuple[Tensor, Tensor]:
    if not isinstance(rendered, Tensor) or not isinstance(normal, Tensor):
        raise ValueError("rendered and normal must be tensors")
    if (
        rendered.ndim != 4
        or rendered.shape != normal.shape
        or rendered.shape[0] <= 0
        or rendered.shape[1] != 3
    ):
        raise ValueError("rendered and normal must have identical non-empty [B,3,H,W] shapes")
    if (
        not rendered.is_floating_point()
        or normal.dtype != rendered.dtype
        or normal.device != rendered.device
        or not torch.isfinite(rendered).all().item()
        or not torch.isfinite(normal).all().item()
    ):
        raise ValueError("rendered and normal must be matching finite floating tensors")
    return rendered, normal


def _mask_pair(
    ground_truth: object,
    proposal: object,
    *,
    images: Tensor,
) -> tuple[Tensor, Tensor]:
    expected = (images.shape[0], 1, *images.shape[2:])
    if (
        not isinstance(ground_truth, Tensor)
        or not isinstance(proposal, Tensor)
        or ground_truth.shape != expected
        or proposal.shape != expected
        or ground_truth.device != images.device
        or proposal.device != images.device
    ):
        raise ValueError("ground_truth and proposal must match image batch and geometry")
    if not torch.isfinite(ground_truth).all().item() or not torch.isfinite(proposal).all().item():
        raise ValueError("ground_truth and proposal must be finite")
    if not torch.all((ground_truth == 0) | (ground_truth == 1)).item() or not torch.all(
        (proposal == 0) | (proposal == 1)
    ).item():
        raise ValueError("ground_truth and proposal must be binary")
    ground_truth_float = ground_truth.to(dtype=images.dtype)
    proposal_float = proposal.to(dtype=images.dtype)
    if torch.any(ground_truth_float > proposal_float).item():
        raise ValueError("ground_truth must be a subset of proposal")
    inside_count = ground_truth_float.sum(dim=(1, 2, 3))
    proposal_count = proposal_float.sum(dim=(1, 2, 3))
    if not torch.all(inside_count > 0).item():
        raise ValueError("ground_truth must be non-empty for every sample")
    if not torch.all(proposal_count > inside_count).item():
        raise ValueError("proposal must contain a non-empty normal ring")
    return ground_truth_float, proposal_float


def _positive_finite(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _non_negative_finite(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def renderer_realization_loss(
    *,
    rendered: Tensor,
    normal: Tensor,
    ground_truth: Tensor,
    proposal: Tensor,
    visibility_margin: float = 0.05,
) -> RendererRealizationLoss:
    rendered, normal = _image_pair(rendered, normal)
    ground_truth, proposal = _mask_pair(ground_truth, proposal, images=rendered)
    margin = _positive_finite(visibility_margin, name="visibility_margin")
    energy = (rendered - normal).abs().mean(dim=1, keepdim=True)
    inside_count = ground_truth.sum(dim=(1, 2, 3))
    visible_energy = (energy * ground_truth).sum(dim=(1, 2, 3)) / inside_count
    visible = functional.relu(margin - visible_energy).mean()
    ring = proposal - ground_truth
    ring_count = ring.sum(dim=(1, 2, 3))
    ring_loss = ((energy * ring).sum(dim=(1, 2, 3)) / ring_count).mean()
    return RendererRealizationLoss(visible=visible, ring=ring_loss)


def realization_head_loss(
    *,
    logits: Tensor,
    rendered: Tensor,
    normal: Tensor,
    ground_truth: Tensor,
    proposal: Tensor,
) -> RealizationHeadLoss:
    rendered, normal = _image_pair(rendered, normal)
    ground_truth, proposal = _mask_pair(ground_truth, proposal, images=rendered)
    if (
        not isinstance(logits, Tensor)
        or logits.shape != proposal.shape
        or logits.device != rendered.device
        or not logits.is_floating_point()
        or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("logits must be finite and match proposal")
    active = proposal.bool()
    bce_values = []
    for sample in range(logits.shape[0]):
        positive = ground_truth[sample].bool()
        negative = active[sample] & ~positive
        positive_bce = functional.binary_cross_entropy_with_logits(
            logits[sample][positive].float(),
            torch.ones_like(logits[sample][positive], dtype=torch.float32),
        )
        negative_bce = functional.binary_cross_entropy_with_logits(
            logits[sample][negative].float(),
            torch.zeros_like(logits[sample][negative], dtype=torch.float32),
        )
        bce_values.append(0.5 * (positive_bce + negative_bce))
    mask_bce = torch.stack(bce_values).mean()
    q = logits.sigmoid() * proposal.to(dtype=logits.dtype)
    q_float = q.float()
    ground_truth_float = ground_truth.float()
    intersection = (q_float * ground_truth_float).sum(dim=(1, 2, 3))
    denominator = q_float.sum(dim=(1, 2, 3)) + ground_truth_float.sum(dim=(1, 2, 3))
    mask_dice = (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()

    # Pixel difference remains a continuous, detached loss signal. It is never a mask.
    energy = (rendered - normal).abs().mean(dim=1, keepdim=True).detach().float()
    consistency_values = []
    for sample in range(logits.shape[0]):
        selected = active[sample]
        q_vector = q_float[sample][selected].reshape(1, -1)
        energy_vector = energy[sample][selected].reshape(1, -1)
        cosine = functional.cosine_similarity(q_vector, energy_vector, dim=1, eps=1e-8)
        consistency_values.append(1.0 - cosine.squeeze(0))
    consistency = torch.stack(consistency_values).mean()
    return RealizationHeadLoss(
        mask=mask_bce + mask_dice,
        mask_bce=mask_bce,
        mask_dice=mask_dice,
        consistency=consistency,
    )


def compose_module_b_loss(
    *,
    diffusion: Tensor,
    renderer: RendererRealizationLoss,
    head: RealizationHeadLoss,
    diffusion_weight: float = 1.0,
    visible_weight: float = 1.0,
    ring_weight: float = 1.0,
    mask_weight: float = 1.0,
    consistency_weight: float = 0.25,
) -> ModuleBLoss:
    if not isinstance(diffusion, Tensor) or diffusion.ndim != 0 or not torch.isfinite(diffusion):
        raise ValueError("diffusion must be a finite scalar tensor")
    weights = {
        "diffusion_weight": diffusion_weight,
        "visible_weight": visible_weight,
        "ring_weight": ring_weight,
        "mask_weight": mask_weight,
        "consistency_weight": consistency_weight,
    }
    validated = {
        name: _non_negative_finite(value, name=name) for name, value in weights.items()
    }
    total = (
        validated["diffusion_weight"] * diffusion
        + validated["visible_weight"] * renderer.visible
        + validated["ring_weight"] * renderer.ring
        + validated["mask_weight"] * head.mask
        + validated["consistency_weight"] * head.consistency
    )
    return ModuleBLoss(
        total=total,
        diffusion=diffusion,
        visible=renderer.visible,
        ring=renderer.ring,
        mask=head.mask,
        consistency=head.consistency,
    )


def threshold_realization(q: Tensor, proposal: Tensor, *, threshold: float = 0.5) -> Tensor:
    if not isinstance(q, Tensor) or not isinstance(proposal, Tensor) or q.shape != proposal.shape:
        raise ValueError("q and proposal must be tensors with identical shapes")
    if not q.is_floating_point() or not torch.isfinite(q).all().item():
        raise ValueError("q must contain finite floating values")
    if not torch.isfinite(proposal).all().item() or not torch.all(
        (proposal == 0) | (proposal == 1)
    ).item():
        raise ValueError("proposal must be finite and binary")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not 0.0 < float(threshold) < 1.0
    ):
        raise ValueError("threshold must lie strictly between zero and one")
    return proposal.bool() & (q >= float(threshold))

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .planner_threshold import finalize_realization


@dataclass(frozen=True)
class PlannerPrediction:
    logits: Tensor
    q: Tensor


def _group_count(channels: int) -> int:
    return next(group for group in (8, 4, 2, 1) if channels % group == 0)


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.normalization = nn.GroupNorm(
            _group_count(output_channels), output_channels
        )
        self.activation = nn.SiLU()

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(self.normalization(self.convolution(features)))


class PlannerNetwork(nn.Module):
    def __init__(
        self,
        *,
        defect_types: tuple[str, ...],
        base_channels: int,
        type_embedding_dim: int,
    ) -> None:
        super().__init__()
        if (
            not isinstance(defect_types, tuple)
            or not defect_types
            or any(not isinstance(value, str) or not value for value in defect_types)
            or len(set(defect_types)) != len(defect_types)
        ):
            raise ValueError("defect_types must contain unique non-empty strings")
        if (
            isinstance(base_channels, bool)
            or not isinstance(base_channels, int)
            or base_channels <= 0
        ):
            raise ValueError("base_channels must be a positive integer")
        if (
            isinstance(type_embedding_dim, bool)
            or not isinstance(type_embedding_dim, int)
            or type_embedding_dim <= 0
        ):
            raise ValueError("type_embedding_dim must be a positive integer")
        self.defect_types = defect_types
        self.base_channels = base_channels
        self.type_embedding_dim = type_embedding_dim
        self.scale_channels = (
            base_channels,
            2 * base_channels,
            4 * base_channels,
        )
        full, half, quarter = self.scale_channels
        self.encoder_full = _ConvBlock(4, full)
        self.encoder_half = _ConvBlock(full, half, stride=2)
        self.encoder_quarter = _ConvBlock(half, quarter, stride=2)
        self.type_embedding = nn.Embedding(len(defect_types), type_embedding_dim)
        self.bottleneck_film = nn.Linear(type_embedding_dim, 2 * quarter)
        self.decoder_half = _ConvBlock(quarter + half, half)
        self.decoder_full = _ConvBlock(half + full, full)
        self.output = nn.Conv2d(full, 1, kernel_size=1)

    def forward(
        self,
        normal_rgb: Tensor,
        proposal: Tensor,
        type_indices: Tensor,
    ) -> PlannerPrediction:
        if (
            not isinstance(normal_rgb, Tensor)
            or normal_rgb.ndim != 4
            or normal_rgb.shape[0] <= 0
            or normal_rgb.shape[1] != 3
            or not normal_rgb.is_floating_point()
            or not torch.isfinite(normal_rgb).all().item()
        ):
            raise ValueError("normal_rgb must be a finite floating [B,3,H,W] tensor")
        expected_proposal = (normal_rgb.shape[0], 1, *normal_rgb.shape[2:])
        if not isinstance(proposal, Tensor) or proposal.shape != expected_proposal:
            raise ValueError("proposal must match normal_rgb batch and geometry")
        if (
            proposal.dtype is not torch.bool
            and not proposal.is_floating_point()
        ) or (
            not torch.isfinite(proposal).all().item()
            or not torch.all((proposal == 0) | (proposal == 1)).item()
        ):
            raise ValueError("proposal must be a finite binary tensor")
        if normal_rgb.shape[2] % 4 or normal_rgb.shape[3] % 4:
            raise ValueError("planner height and width must be divisible by four")
        if normal_rgb.device != proposal.device:
            raise ValueError("normal_rgb and proposal must share a device")
        if (
            not isinstance(type_indices, Tensor)
            or type_indices.shape != (normal_rgb.shape[0],)
            or type_indices.dtype is not torch.long
            or type_indices.device != normal_rgb.device
            or torch.any(type_indices < 0).item()
            or torch.any(type_indices >= len(self.defect_types)).item()
        ):
            raise ValueError("type_indices must identify one configured type per sample")
        parameter = self.output.weight
        if normal_rgb.device != parameter.device or normal_rgb.dtype != parameter.dtype:
            raise ValueError("planner inputs must match network device and dtype")

        proposal_float = proposal.to(dtype=normal_rgb.dtype)
        full = self.encoder_full(torch.cat((normal_rgb, proposal_float), dim=1))
        half = self.encoder_half(full)
        quarter = self.encoder_quarter(half)
        film = self.bottleneck_film(self.type_embedding(type_indices))
        gamma, beta = film.chunk(2, dim=1)
        quarter = quarter * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        decoded_half = functional.interpolate(
            quarter,
            size=half.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_half = self.decoder_half(torch.cat((decoded_half, half), dim=1))
        decoded_full = functional.interpolate(
            decoded_half,
            size=full.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_full = self.decoder_full(torch.cat((decoded_full, full), dim=1))
        raw_logits = self.output(decoded_full)
        logits = raw_logits * proposal_float
        q = torch.sigmoid(logits) * proposal_float
        return PlannerPrediction(logits=logits, q=q)


class RealizationPlanner:
    planner_seed_mode = "deterministic_invariant_v1"

    def __init__(
        self,
        *,
        network: PlannerNetwork,
        defect_types: tuple[str, ...],
        threshold: float,
        device: torch.device,
    ) -> None:
        if not isinstance(network, PlannerNetwork):
            raise ValueError("network must be a PlannerNetwork")
        if defect_types != network.defect_types:
            raise ValueError("defect_types must match the planner network vocabulary")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(float(threshold))
            or not 0.0 < float(threshold) < 1.0
        ):
            raise ValueError("threshold must lie strictly between zero and one")
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        self.network = network.to(device)
        self.defect_types = defect_types
        self.threshold = float(threshold)
        self.device = device

    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        proposal: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            not isinstance(normal, np.ndarray)
            or normal.ndim != 3
            or normal.shape[2] != 3
            or normal.dtype != np.uint8
        ):
            raise ValueError("normal must be an HxWx3 uint8 numpy array")
        if (
            not isinstance(proposal, np.ndarray)
            or proposal.shape != normal.shape[:2]
            or not np.isin(proposal, (0, 1)).all()
        ):
            raise ValueError("proposal must be a matching binary HxW numpy array")
        if not proposal.any() or proposal.all():
            raise ValueError("proposal must contain allowed and disallowed pixels")
        if defect_type not in self.defect_types:
            raise ValueError("defect_type is outside the configured vocabulary")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        normal_tensor = (
            torch.from_numpy(np.ascontiguousarray(normal))
            .to(device=self.device, dtype=self.network.output.weight.dtype)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(127.5)
            .sub(1.0)
        )
        proposal_array = proposal.astype(bool, copy=True)
        proposal_tensor = (
            torch.from_numpy(np.ascontiguousarray(proposal_array))
            .to(device=self.device, dtype=normal_tensor.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        type_indices = torch.tensor(
            [self.defect_types.index(defect_type)],
            dtype=torch.long,
            device=self.device,
        )
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.no_grad():
                prediction = self.network(normal_tensor, proposal_tensor, type_indices)
        finally:
            self.network.train(was_training)
        q = np.ascontiguousarray(prediction.q[0, 0].float().cpu().numpy())
        final = finalize_realization(q, proposal_array, threshold=self.threshold)
        return q, final


__all__ = ["PlannerNetwork", "PlannerPrediction", "RealizationPlanner"]

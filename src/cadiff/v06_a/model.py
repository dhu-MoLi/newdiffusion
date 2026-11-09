"""Three-scale mask diffusion predictors for Module A RC1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .config import V06AExperimentConfig


SUPPORTED_A1_VARIANTS = ("a1_gv", "a1_sv_plain")
SPATIAL_INPUT_CHANNELS = ("z_k", "valid", "sdf", "u_x", "u_y")
_CATEGORY_VOCABULARIES = {
    "hazelnut": ("crack", "cut", "hole", "print"),
    "cable": (
        "bent_wire",
        "cut_inner_insulation",
        "cut_outer_insulation",
        "poke_insulation",
    ),
}


@dataclass(frozen=True)
class _A1ConstructionContract:
    architecture_revision: str
    category: str
    defect_types: tuple[str, ...]
    variant: str
    base_channels: int
    type_embedding_dim: int
    timestep_embedding_dim: int
    normal_feature_channels: int
    diffusion_steps: int
    proposal_size: int
    image_size: int
    spatial_input_channels: tuple[str, ...]


def _construction_signature(contract: _A1ConstructionContract) -> str:
    payload = json.dumps(
        asdict(contract), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def require_supported_variant(variant: object) -> str:
    if not isinstance(variant, str) or variant not in SUPPORTED_A1_VARIANTS:
        raise ValueError(f"variant {variant!r} is not approved; a1_ctx is gated and unavailable")
    return variant


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conditioning_channels: int) -> None:
        super().__init__()
        self.input_projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.norm1 = nn.GroupNorm(_normalization_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_normalization_groups(out_channels), out_channels)
        self.film = nn.Linear(conditioning_channels, 2 * out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features: Tensor, conditioning: Tensor) -> Tensor:
        residual = self.input_projection(features)
        features = self.conv1(F.silu(self.norm1(features)))
        scale, shift = self.film(conditioning).chunk(2, dim=1)
        features = self.norm2(features)
        features = features * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return residual + self.conv2(F.silu(features))


class _NormalRGBEncoder(nn.Module):
    """Small spatial encoder whose outputs align to the proposal U-Net scales."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.full_path = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.half_path = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
        )
        self.quarter_path = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(2 * channels, 4 * channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, normal_rgb: Tensor, *, proposal_size: int) -> tuple[Tensor, Tensor, Tensor]:
        resized = F.interpolate(
            normal_rgb,
            size=(proposal_size, proposal_size),
            mode="bilinear",
            align_corners=False,
        )
        full = self.full_path(resized)
        half = self.half_path(full)
        return full, half, self.quarter_path(half)


class A1MaskUNet(nn.Module):
    """Small three-scale U-Net with vector-conditioned FiLM blocks."""

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_construction_contract", "_construction_signature"} and name in self.__dict__:
            raise AttributeError(f"{name} is immutable after model construction")
        super().__setattr__(name, value)

    def __init__(
        self,
        *,
        category: str,
        defect_types: tuple[str, ...],
        variant: str,
        base_channels: int = 32,
        type_embedding_dim: int = 64,
        timestep_embedding_dim: int = 64,
        normal_feature_channels: int = 16,
        diffusion_steps: int = 100,
        proposal_size: int = 128,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        variant = require_supported_variant(variant)
        expected_vocabulary = _CATEGORY_VOCABULARIES.get(category)
        if expected_vocabulary is None or defect_types != expected_vocabulary:
            raise ValueError("category and defect_types must equal one frozen RC1 vocabulary")
        base_channels = _positive_integer(base_channels, name="base_channels")
        type_embedding_dim = _positive_integer(type_embedding_dim, name="type_embedding_dim")
        timestep_embedding_dim = _positive_integer(
            timestep_embedding_dim, name="timestep_embedding_dim"
        )
        normal_feature_channels = _positive_integer(
            normal_feature_channels, name="normal_feature_channels"
        )
        diffusion_steps = _positive_integer(diffusion_steps, name="diffusion_steps")
        if proposal_size != 128 or image_size != 512:
            raise ValueError("Module A grids are frozen at proposal=128 and image=512")

        construction_contract = _A1ConstructionContract(
            architecture_revision="v06_a_three_scale_unet_rc1",
            category=category,
            defect_types=defect_types,
            variant=variant,
            base_channels=base_channels,
            type_embedding_dim=type_embedding_dim,
            timestep_embedding_dim=timestep_embedding_dim,
            normal_feature_channels=normal_feature_channels,
            diffusion_steps=diffusion_steps,
            proposal_size=proposal_size,
            image_size=image_size,
            spatial_input_channels=SPATIAL_INPUT_CHANNELS,
        )
        self._construction_contract = construction_contract
        self._construction_signature = _construction_signature(construction_contract)
        self.category = category
        self.defect_types = defect_types
        self.variant = variant
        self.base_channels = base_channels
        self.type_embedding_dim = type_embedding_dim
        self.timestep_embedding_dim = timestep_embedding_dim
        self.normal_feature_channels = normal_feature_channels
        self.diffusion_steps = diffusion_steps
        self.proposal_size = proposal_size
        self.image_size = image_size
        self.spatial_input_channels = SPATIAL_INPUT_CHANNELS

        conditioning_channels = 4 * base_channels
        self.type_embedding = nn.Embedding(len(defect_types), type_embedding_dim)
        self.timestep_embedding = nn.Embedding(diffusion_steps, timestep_embedding_dim)
        self.type_conditioning = nn.Linear(type_embedding_dim, conditioning_channels)
        self.timestep_conditioning = nn.Linear(timestep_embedding_dim, conditioning_channels)

        self.input_projection = nn.Conv2d(5, base_channels, kernel_size=3, padding=1)
        self.encoder_full = _FiLMResidualBlock(base_channels, base_channels, conditioning_channels)
        self.downsample_half = nn.Conv2d(
            base_channels, 2 * base_channels, kernel_size=3, stride=2, padding=1
        )
        self.encoder_half = _FiLMResidualBlock(
            2 * base_channels, 2 * base_channels, conditioning_channels
        )
        self.downsample_quarter = nn.Conv2d(
            2 * base_channels, 4 * base_channels, kernel_size=3, stride=2, padding=1
        )
        self.encoder_quarter = _FiLMResidualBlock(
            4 * base_channels, 4 * base_channels, conditioning_channels
        )
        self.bottleneck = _FiLMResidualBlock(
            4 * base_channels, 4 * base_channels, conditioning_channels
        )
        self.upsample_half = nn.ConvTranspose2d(
            4 * base_channels, 2 * base_channels, kernel_size=2, stride=2
        )
        self.decoder_half = _FiLMResidualBlock(
            4 * base_channels, 2 * base_channels, conditioning_channels
        )
        self.upsample_full = nn.ConvTranspose2d(
            2 * base_channels, base_channels, kernel_size=2, stride=2
        )
        self.decoder_full = _FiLMResidualBlock(
            2 * base_channels, base_channels, conditioning_channels
        )
        self.output_projection = nn.Conv2d(base_channels, 1, kernel_size=1)

        if variant == "a1_sv_plain":
            self.normal_encoder = _NormalRGBEncoder(normal_feature_channels)
            self.normal_fusion_full = nn.Conv2d(
                normal_feature_channels, base_channels, kernel_size=1
            )
            self.normal_fusion_half = nn.Conv2d(
                2 * normal_feature_channels, 2 * base_channels, kernel_size=1
            )
            self.normal_fusion_quarter = nn.Conv2d(
                4 * normal_feature_channels, 4 * base_channels, kernel_size=1
            )
        else:
            self.normal_encoder = None
            self.normal_fusion_full = None
            self.normal_fusion_half = None
            self.normal_fusion_quarter = None

    @classmethod
    def from_experiment(cls, config: V06AExperimentConfig, *, variant: str) -> A1MaskUNet:
        require_supported_variant(variant)
        return cls(
            category=config.category,
            defect_types=config.defect_types,
            variant=variant,
            base_channels=config.base_channels,
            type_embedding_dim=config.type_embedding_dim,
            timestep_embedding_dim=config.timestep_embedding_dim,
            normal_feature_channels=config.normal_feature_channels,
            diffusion_steps=config.diffusion_steps,
            proposal_size=config.proposal_size,
            image_size=config.image_size,
        )

    def _validate_inputs(
        self,
        spatial: Tensor,
        type_ids: Tensor,
        timesteps: Tensor,
        normal_rgb: Tensor | None,
    ) -> None:
        if not isinstance(spatial, Tensor) or spatial.shape[1:] != (5, 128, 128):
            raise ValueError("spatial input must have shape [B,5,128,128]")
        if spatial.shape[0] <= 0 or not spatial.is_floating_point():
            raise ValueError("spatial input must have a non-empty floating batch")
        parameter = self.input_projection.weight
        if spatial.device != parameter.device or spatial.dtype != parameter.dtype:
            raise ValueError("spatial input must match model device and dtype")
        if not torch.isfinite(spatial).all().item():
            raise ValueError("spatial input must contain only finite values")
        valid = spatial[:, 1:2]
        if not torch.all((valid == 0) | (valid == 1)).item():
            raise ValueError("valid spatial channel must be exactly binary")
        expected_vector_shape = (spatial.shape[0],)
        for value, name, upper_bound in (
            (type_ids, "type_ids", len(self.defect_types)),
            (timesteps, "timesteps", self.diffusion_steps),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.long
                or value.shape != expected_vector_shape
                or value.device != spatial.device
            ):
                raise ValueError(f"{name} must be a matching one-dimensional long tensor")
            if not torch.all((0 <= value) & (value < upper_bound)).item():
                raise ValueError(f"{name} is outside its frozen vocabulary")
        if self.variant == "a1_gv":
            if normal_rgb is not None:
                raise ValueError("a1_gv must not receive normal_rgb")
            return
        if normal_rgb is None:
            raise ValueError("a1_sv_plain requires normal_rgb")
        if (
            not isinstance(normal_rgb, Tensor)
            or normal_rgb.shape != (spatial.shape[0], 3, 512, 512)
            or not normal_rgb.is_floating_point()
            or normal_rgb.device != spatial.device
            or normal_rgb.dtype != spatial.dtype
            or not torch.isfinite(normal_rgb).all().item()
        ):
            raise ValueError(
                "normal_rgb must be a finite [B,3,512,512] tensor matching model dtype/device"
            )

    def forward(
        self,
        spatial: Tensor,
        *,
        type_ids: Tensor,
        timesteps: Tensor,
        normal_rgb: Tensor | None = None,
    ) -> Tensor:
        validate_a1_model_contract(self)
        self._validate_inputs(spatial, type_ids, timesteps, normal_rgb)
        conditioning = F.silu(
            self.type_conditioning(self.type_embedding(type_ids))
            + self.timestep_conditioning(self.timestep_embedding(timesteps))
        )
        full = self.input_projection(spatial)
        normal_features: tuple[Tensor, Tensor, Tensor] | None = None
        if self.normal_encoder is not None:
            assert normal_rgb is not None
            normal_features = self.normal_encoder(normal_rgb, proposal_size=self.proposal_size)
            assert self.normal_fusion_full is not None
            full = full + self.normal_fusion_full(normal_features[0])
        full = self.encoder_full(full, conditioning)

        half = self.downsample_half(full)
        if normal_features is not None:
            assert self.normal_fusion_half is not None
            half = half + self.normal_fusion_half(normal_features[1])
        half = self.encoder_half(half, conditioning)

        quarter = self.downsample_quarter(half)
        if normal_features is not None:
            assert self.normal_fusion_quarter is not None
            quarter = quarter + self.normal_fusion_quarter(normal_features[2])
        quarter = self.encoder_quarter(quarter, conditioning)
        quarter = self.bottleneck(quarter, conditioning)

        decoded_half = self.upsample_half(quarter)
        decoded_half = self.decoder_half(torch.cat((decoded_half, half), dim=1), conditioning)
        decoded_full = self.upsample_full(decoded_half)
        decoded_full = self.decoder_full(torch.cat((decoded_full, full), dim=1), conditioning)
        predicted_noise = self.output_projection(F.silu(decoded_full))
        return predicted_noise * spatial[:, 1:2]


def validate_a1_model_contract(
    model: object,
    *,
    expected_variant: str | None = None,
    expected_category: str | None = None,
    expected_defect_types: tuple[str, ...] | None = None,
) -> str:
    """Verify immutable construction identity and variant-specific RGB topology."""

    if not isinstance(model, A1MaskUNet):
        raise TypeError("model must be a genuine A1MaskUNet")
    contract = getattr(model, "_construction_contract", None)
    signature = getattr(model, "_construction_signature", None)
    if not isinstance(contract, _A1ConstructionContract) or signature != _construction_signature(
        contract
    ):
        raise ValueError("A1MaskUNet construction contract signature is invalid")
    mismatches = {
        field: (getattr(model, field, None), getattr(contract, field))
        for field in (
            "category",
            "defect_types",
            "variant",
            "base_channels",
            "type_embedding_dim",
            "timestep_embedding_dim",
            "normal_feature_channels",
            "diffusion_steps",
            "proposal_size",
            "image_size",
            "spatial_input_channels",
        )
        if getattr(model, field, None) != getattr(contract, field)
    }
    if mismatches:
        raise ValueError(f"A1MaskUNet construction contract mismatch: {mismatches}")
    if expected_variant is not None and contract.variant != require_supported_variant(
        expected_variant
    ):
        raise ValueError(
            "A1MaskUNet construction contract variant mismatch: "
            f"constructed={contract.variant!r}, expected={expected_variant!r}"
        )
    if expected_category is not None and contract.category != expected_category:
        raise ValueError("A1MaskUNet construction contract category mismatch")
    if expected_defect_types is not None and contract.defect_types != expected_defect_types:
        raise ValueError("A1MaskUNet construction contract vocabulary mismatch")

    common_topology_valid = (
        model.input_projection.in_channels == 5
        and model.input_projection.out_channels == contract.base_channels
        and model.type_embedding.num_embeddings == len(contract.defect_types)
        and model.type_embedding.embedding_dim == contract.type_embedding_dim
        and model.timestep_embedding.num_embeddings == contract.diffusion_steps
        and model.timestep_embedding.embedding_dim == contract.timestep_embedding_dim
        and model.output_projection.in_channels == contract.base_channels
        and model.output_projection.out_channels == 1
    )
    if not common_topology_valid:
        raise ValueError("A1MaskUNet construction contract topology mismatch")
    rgb_modules = (
        model.normal_encoder,
        model.normal_fusion_full,
        model.normal_fusion_half,
        model.normal_fusion_quarter,
    )
    if contract.variant == "a1_gv":
        if any(module is not None for module in rgb_modules) or any(
            name.startswith(("normal_encoder.", "normal_fusion_")) for name in model.state_dict()
        ):
            raise ValueError("A1MaskUNet GV construction contract forbids RGB modules")
    else:
        if not isinstance(model.normal_encoder, _NormalRGBEncoder) or not all(
            isinstance(module, nn.Conv2d) for module in rgb_modules[1:]
        ):
            raise ValueError("A1MaskUNet SV construction contract requires three-scale RGB modules")
        expected_fusion_shapes = (
            (contract.normal_feature_channels, contract.base_channels),
            (2 * contract.normal_feature_channels, 2 * contract.base_channels),
            (4 * contract.normal_feature_channels, 4 * contract.base_channels),
        )
        observed_fusion_shapes = tuple(
            (module.in_channels, module.out_channels)  # type: ignore[union-attr]
            for module in rgb_modules[1:]
        )
        if observed_fusion_shapes != expected_fusion_shapes:
            raise ValueError("A1MaskUNet SV construction contract RGB topology mismatch")
    return signature

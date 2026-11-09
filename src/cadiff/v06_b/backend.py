from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from torch import Tensor, nn

from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context

from .config import V06BConfig
from .realization import RealizationHead
from .renderer import BackendOutput
from .rollout import truncated_denoise_rollout


@dataclass(frozen=True, eq=False)
class BackendDiagnosticOutput:
    rendered: np.ndarray
    q: np.ndarray
    q_without_unet_feature: np.ndarray
    q_without_generation: np.ndarray


class ProposalConditionedBackend:
    """SD1.5/P2-B2 backend for the public causal-renderer contract."""

    def __init__(
        self,
        *,
        components: Any,
        model: nn.Module,
        realization_head: RealizationHead,
        prompts: Mapping[str, Tensor],
        config: V06BConfig,
        device: torch.device,
        scheduler_factory: Callable[[object], Any] | None = None,
    ) -> None:
        if not isinstance(config, V06BConfig):
            raise ValueError("config must be a V06BConfig")
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a torch module")
        if not isinstance(realization_head, RealizationHead):
            raise ValueError("realization_head must be a RealizationHead")
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        try:
            vae = components.vae
            scheduler_config = components.scheduler.config
        except AttributeError as error:
            raise ValueError("components must expose vae and scheduler.config") from error
        if not isinstance(vae, nn.Module):
            raise ValueError("components.vae must be a torch module")
        if tuple(prompts) != config.defect_types:
            raise ValueError("prompts must match the ordered Module-B vocabulary")
        prompt_values = dict(prompts)
        if any(
            not isinstance(value, Tensor)
            or value.ndim != 3
            or value.shape[0] != 1
            or value.device != device
            or not value.is_floating_point()
            or not torch.isfinite(value).all().item()
            or value.requires_grad
            or value.is_inference()
            for value in prompt_values.values()
        ):
            raise ValueError("prompts must be ordinary detached finite tensors on device")
        if realization_head.feature_channels != config.feature_channels:
            raise ValueError("realization head feature channels must match config")
        self.components = components
        self.model = model.eval()
        self.realization_head = realization_head.eval()
        self.prompts = prompt_values
        self.config = config
        self.device = device
        self.scheduler_factory = scheduler_factory or DDIMScheduler.from_config
        self._scheduler_config = scheduler_config
        self._normal_cache: dict[str, Tensor] = {}

    def _normal_latent(self, normal: np.ndarray) -> Tensor:
        identity = hashlib.sha256(normal.tobytes()).hexdigest()
        cached = self._normal_cache.get(identity)
        if cached is not None:
            return cached
        parameters = tuple(
            parameter
            for parameter in self.components.vae.parameters()
            if parameter.is_floating_point()
        )
        if not parameters:
            raise ValueError("VAE must expose floating parameters")
        vae_dtype = parameters[0].dtype
        pixels = (
            torch.from_numpy(normal.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=vae_dtype)
            .div(127.5)
            .sub(1.0)
        )
        with torch.inference_mode():
            encoded = encode_mode(self.components.vae, pixels)
        latent = encoded.detach().clone()
        if latent.is_inference() or latent.requires_grad:
            raise ValueError("cached normal latent must be an ordinary detached tensor")
        self._normal_cache.clear()
        self._normal_cache[identity] = latent
        return latent

    def diagnose(
        self,
        normal: np.ndarray,
        defect_type: str,
        proposal: np.ndarray,
        seed: int,
    ) -> BackendDiagnosticOutput:
        if (
            not isinstance(normal, np.ndarray)
            or normal.shape != (self.config.image_size, self.config.image_size, 3)
            or normal.dtype != np.uint8
        ):
            raise ValueError("normal must be configured-size uint8 HxWx3")
        if (
            not isinstance(proposal, np.ndarray)
            or proposal.shape != normal.shape[:2]
            or not np.isin(proposal, (0, 1)).all()
            or not proposal.any()
            or proposal.all()
        ):
            raise ValueError("proposal must be a non-trivial binary HxW array")
        if defect_type not in self.config.defect_types:
            raise ValueError("defect_type is outside the Module-B vocabulary")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        normal_latent = self._normal_latent(normal)
        proposal_tensor = (
            torch.from_numpy(proposal.astype(np.float32, copy=True))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=self.device)
        )
        latent_masks = build_latent_masks(proposal_tensor, normal_latent.shape[2:])
        context = build_masked_context(
            normal_latent,
            latent_masks.binary,
            torch.Generator(device=self.device).manual_seed(seed + 10_000_000),
        )
        noise = torch.randn(
            normal_latent.shape,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            device=self.device,
            dtype=normal_latent.dtype,
        )
        scheduler = self.scheduler_factory(self._scheduler_config)
        prompt = self.prompts[defect_type]
        if prompt.dtype != normal_latent.dtype:
            raise ValueError("prompt dtype must match normal latent")
        rollout = truncated_denoise_rollout(
            model=self.model,
            scheduler=scheduler,
            normal_latent=normal_latent,
            context=context,
            binary_mask=latent_masks.binary,
            prompt_embeds=prompt,
            defect_type=defect_type,
            noise=noise,
            steps=self.config.rollout_steps,
            backprop_steps=0,
        )
        raw = decode_latent(self.components.vae, rollout.latent)
        with torch.no_grad():
            full = self.realization_head(
                generated_latent=rollout.latent,
                normal_latent=normal_latent,
                generation_feature=rollout.generation_feature,
                proposal=proposal_tensor,
            )
            without_unet_feature = (
                self.realization_head.forward_without_unet_feature(
                    generated_latent=rollout.latent,
                    normal_latent=normal_latent,
                    proposal=proposal_tensor,
                )
            )
            without_generation = self.realization_head.forward_without_generation(
                normal_latent=normal_latent,
                proposal=proposal_tensor,
            )
        rendered = (
            raw[0]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .copy()
        )
        q = full.q[0, 0].float().cpu().numpy().copy()
        q_without_unet_feature = (
            without_unet_feature.q[0, 0].float().cpu().numpy().copy()
        )
        q_without_generation = (
            without_generation.q[0, 0].float().cpu().numpy().copy()
        )
        return BackendDiagnosticOutput(
            rendered=np.ascontiguousarray(rendered),
            q=np.ascontiguousarray(q),
            q_without_unet_feature=np.ascontiguousarray(q_without_unet_feature),
            q_without_generation=np.ascontiguousarray(q_without_generation),
        )

    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        proposal: np.ndarray,
        seed: int,
    ) -> BackendOutput:
        diagnostic = self.diagnose(normal, defect_type, proposal, seed)
        return BackendOutput(rendered=diagnostic.rendered, q=diagnostic.q)


__all__ = ["BackendDiagnosticOutput", "ProposalConditionedBackend"]

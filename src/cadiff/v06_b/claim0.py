"""Claim-0 identity planner and diagnostic SeaS renderer boundary.

This source has no eager torch or model-loader import; the legacy package
initializer may import those dependencies. Exact exterior
composition belongs to the caller; this boundary returns raw RGB plus an
unchanged, independently owned realization mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import re

import numpy as np

TYPE_MAP = {"scratches": "scratches", "similar_colour_spot": "similar colour spot"}
IMAGE_SIZE = 512
MIN_SHAPE_PIXELS = 64
SCHEMA = "claim0_chewinggum_frozen_b_v1"

_EXPECTED_CONFIG = {
    "capacity_defect_type": "chunk of gum missing", "category": "chewinggum",
    "context_fill_seed": 202609130, "correct_normal_ratio_limit": 1.0,
    "correct_wrong_ratio_limit": 1.0,
    "defect_types": ["chunk of gum missing", "combined", "corner missing",
                     "scratches", "similar colour spot"],
    "dtype": "float16", "eval_role": "open_dev_report",
    "eval_timesteps": [100, 400, 700], "image_size": 512,
    "inside_noise_weight": 4.0, "lambda_inside": 1.0, "lambda_outside": 0.1,
    "learning_rate": 0.0001, "noise_seed": 202609131,
    "normal_prompt": "a photo of a normal chewinggum",
    "ranking_target_ratio": 0.95, "ranking_weight": 0.25,
    "schema_version": "p2_a1_factorized_v1", "seed": 20260913, "steps": 1200,
    "train_role": "fewshot_train", "type_rank": 8, "wrong_type_policy": "cyclic_next",
}
_CHECKPOINT_SHA = {
    "adapter.safetensors": "e2302a44fef076be4aaeaa6572ed306c9453ddd266fcf46d9154a91399ef04a6",
    "conditioner.safetensors": "748f8c8d93c8653038550a6fd8554c50afc3e90ed58aa3993682ca0b94773b86",
    "metadata.json": "a89c9ae55274ae785cb6ee87c3f684034c475225cd408ec846c3e40eaaf913fe",
}
_BASE_SHA = {
    "model_index.json": "49b63f5a8004b64ff4422daae3ee86c8b49e6a65aa9baaa5e3e356274eef662f",
    "scheduler/scheduler_config.json": "699cce92eb7c122e2eb7dfdea78e6187fda76a5ed4a8e42319b85610e620e091",
    "text_encoder/config.json": "845df614cb9327ae7bbea027316246fae917827407da6df13572e41b5f93b4cc",
    "text_encoder/model.safetensors": "d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334",
    "tokenizer/tokenizer_config.json": "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882",
    "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer/special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
    "unet/config.json": "78f474de6bab3d893868f37be97b636ae65c0df3073ed3256ca458ff599b5f96",
    "unet/diffusion_pytorch_model.safetensors": "19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1",
    "vae/config.json": "786a7d21647ddea6a04b9675c03d3cb45e90a2f3c6da5fbda2c54ade040036de",
    "vae/diffusion_pytorch_model.safetensors": "a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c",
}
_CODE_DEPENDENCIES = (
    "src/cadiff/v06_c/visa_exact_mask_r.py",
    "src/cadiff/v06_c/can_exact_mask_r.py",
    "src/cadiff/diffusion/components.py",
    "src/cadiff/p1_v1/conditioning.py",
    "src/cadiff/p2_a0/model.py",
    "src/cadiff/p2_a1/model.py",
    "src/cadiff/p2_a1/config.py",
    "src/cadiff/v06_b/renderer.py",
    "src/cadiff/p2_b0/generation.py",
    "src/cadiff/p2_a0/masking.py",
    "src/cadiff/diffusion/vae_roundtrip.py",
    "src/cadiff/diffusion/ddim_inversion.py",
    "src/cadiff/v06_b/claim0.py",
    "src/cadiff/v06_c/claim0_generate.py",
)


@dataclass(frozen=True)
class FrozenAsset:
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class FrozenSpec:
    manifest_path: Path
    repository_root: Path
    manifest_sha256: str
    checkpoint_root: Path
    model_root: Path
    config: dict[str, Any]
    inference: dict[str, Any]
    assets: tuple[FrozenAsset, ...]
    accepted_sha256: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _absolute(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return path.resolve(strict=True)


def _validate_spec(spec: FrozenSpec) -> None:
    if _canonical(spec.config) != _canonical(_EXPECTED_CONFIG):
        raise ValueError("the complete SeaS five-type metadata config must remain frozen")
    required_inference = {
        "normal_prompt": "a photo of a normal chewinggum", "dtype": "float16",
        "context_seed_offset": 10_000_000, "posthoc_relabel": False,
    }
    for key, value in required_inference.items():
        if key not in spec.inference or _canonical(spec.inference[key]) != _canonical(value):
            raise ValueError(f"inference.{key} must remain frozen")
    if not spec.assets:
        raise ValueError("assets cannot be empty")
    paths: dict[Path, FrozenAsset] = {}
    roles: set[str] = set()
    for asset in spec.assets:
        if not isinstance(asset.role, str) or not asset.role or asset.role in roles:
            raise ValueError("asset roles must be nonempty and unique")
        if not isinstance(asset.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", asset.sha256):
            raise ValueError("asset sha256 must be lowercase hexadecimal")
        path = asset.path.resolve(strict=True)
        if not path.is_file() or path in paths:
            raise ValueError("asset paths must be unique existing files")
        paths[path] = asset
        roles.add(asset.role)
    for root, required in ((spec.checkpoint_root, _CHECKPOINT_SHA), (spec.model_root, _BASE_SHA)):
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("checkpoint_root/model_root must be existing absolute directories")
        for relative, expected in required.items():
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root.resolve(strict=True)):
                raise ValueError("component escapes its declared root")
            if path not in paths or paths[path].sha256 != expected:
                raise ValueError(f"missing or altered required component binding: {relative}")
    for relative in _CODE_DEPENDENCIES:
        if (spec.repository_root / relative).resolve(strict=True) not in paths:
            raise ValueError(f"missing frozen renderer code dependency: {relative}")


def load_freeze_manifest(
    path: str | Path, repository_root: str | Path | None = None, *,
    expected_sha256: str | None = None,
) -> FrozenSpec:
    """Read a freeze specification; model weights are not loaded or hashed here."""
    import yaml

    manifest_path = Path(path).resolve(strict=True)
    actual_sha256 = _sha256(manifest_path)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be lowercase SHA256")
        if actual_sha256 != expected_sha256:
            raise ValueError("freeze manifest differs from externally accepted SHA256")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("freeze manifest must be a mapping")
    fixed = {
        "schema_version": SCHEMA, "diagnostic_only": True, "category": "chewinggum",
        "planner": "identity_v1", "min_shape_pixels": 64, "image_size": 512,
        "ddim_steps": 50, "type_mapping": TYPE_MAP,
    }
    for key, value in fixed.items():
        if key not in raw or _canonical(raw[key]) != _canonical(value):
            raise ValueError(f"freeze manifest {key} must remain frozen")
    if repository_root is None:
        candidates = [parent for parent in manifest_path.parents if (parent / "src/cadiff").is_dir()]
        if not candidates:
            raise ValueError("repository_root required when manifest is outside the repository")
        repo = candidates[0]
    else:
        repo = Path(repository_root).resolve(strict=True)
    if not (repo / "src/cadiff").is_dir():
        raise ValueError("repository_root must contain src/cadiff")
    config, inference = raw.get("config"), raw.get("inference")
    if not isinstance(config, dict) or not isinstance(inference, dict):
        raise ValueError("config and inference mappings are required")
    asset_rows = raw.get("assets")
    if not isinstance(asset_rows, list):
        raise ValueError("assets must be a list")
    assets = []
    for row in asset_rows:
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256"}:
            raise ValueError("each asset must contain role, path, sha256")
        assets.append(FrozenAsset(row["role"], _absolute(row["path"], "asset.path"), row["sha256"]))
    spec = FrozenSpec(
        manifest_path, repo, actual_sha256,
        _absolute(raw.get("checkpoint_root"), "checkpoint_root"),
        _absolute(raw.get("model_root"), "model_root"),
        json.loads(_canonical(config)), json.loads(_canonical(inference)), tuple(assets), expected_sha256,
    )
    _validate_spec(spec)
    return spec


def verify_freeze(spec: FrozenSpec) -> None:
    """Verify manifest, all assets, and metadata contracts without loading a model."""
    _validate_spec(spec)
    if _sha256(spec.manifest_path) != spec.manifest_sha256:
        raise ValueError("freeze manifest changed after loading")
    for asset in spec.assets:
        if _sha256(asset.path) != asset.sha256:
            raise ValueError(f"frozen asset SHA mismatch: {asset.path}")
    metadata = json.loads((spec.checkpoint_root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "v06_c_visa_exact_mask_r_checkpoint_v1":
        raise ValueError("unexpected renderer checkpoint schema")
    if metadata.get("diagnostic_only") is not True or metadata.get("not_official_p2_a1") is not True:
        raise ValueError("SeaS diagnostic status must be preserved")
    if _canonical(metadata.get("config")) != _canonical(spec.config):
        raise ValueError("metadata config differs from frozen complete five-type config")
    for name in ("adapter.safetensors", "conditioner.safetensors"):
        if metadata.get("files", {}).get(name, {}).get("sha256") != _CHECKPOINT_SHA[name]:
            raise ValueError(f"checkpoint metadata binding changed: {name}")


def _validate_valid(valid: np.ndarray) -> None:
    if not isinstance(valid, np.ndarray) or valid.shape != (IMAGE_SIZE, IMAGE_SIZE) or valid.dtype != np.bool_:
        raise ValueError("valid must be a 512x512 boolean array")
    if not valid.any():
        raise ValueError("valid must contain an effective domain")


def validate_normal(normal_rgb: np.ndarray, valid: np.ndarray) -> None:
    _validate_valid(valid)
    if not isinstance(normal_rgb, np.ndarray) or normal_rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or normal_rgb.dtype != np.uint8:
        raise ValueError("normal_rgb must be 512x512 uint8 RGB")


def validate_proposal(proposal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a boolean identity copy; never clip, threshold, repair or fall back."""
    _validate_valid(valid)
    if not isinstance(proposal, np.ndarray) or proposal.shape != valid.shape or proposal.dtype.kind not in "biuf":
        raise ValueError("proposal must be a 512x512 boolean or binary numeric array")
    if not np.isfinite(proposal).all() or not ((proposal == 0) | (proposal == 1)).all():
        raise ValueError("proposal values must be exactly finite zero or one")
    mask = np.array(proposal, dtype=bool, order="C", copy=True)
    count = int(mask.sum())
    if count < MIN_SHAPE_PIXELS:
        raise ValueError("proposal must contain at least 64 pixels")
    if (mask & ~valid).any():
        raise ValueError("proposal must be a subset of valid")
    if count == int(valid.sum()):
        raise ValueError("proposal must not fill the valid domain")
    return mask


def validate_request_type_seed(type_slug: str, seed: int) -> None:
    if not isinstance(type_slug, str) or type_slug not in TYPE_MAP:
        raise ValueError("unsupported Claim-0 type slug")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63), excluding bool")


class Claim0ModuleB:
    def __init__(self, renderer: Callable[..., np.ndarray], spec: FrozenSpec | None = None) -> None:
        if not callable(renderer):
            raise ValueError("renderer must be callable")
        self.renderer = renderer
        self.spec = spec
        self._verified = False

    @property
    def verified(self) -> bool:
        """Whether the real frozen loader completed with an accepted manifest."""
        return self._verified

    def __call__(self, normal_rgb: np.ndarray, type_slug: str, proposal: np.ndarray,
                 valid: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        validate_normal(normal_rgb, valid)
        validate_request_type_seed(type_slug, seed)
        m_final = validate_proposal(proposal, valid)
        snapshot = m_final.copy()
        renderer_mask = m_final.copy()
        renderer_normal = normal_rgb.copy()
        normal_snapshot = renderer_normal.copy()
        raw = self.renderer(renderer_normal, TYPE_MAP[type_slug], renderer_mask, seed)
        if not np.array_equal(renderer_normal, normal_snapshot):
            raise ValueError("renderer mutated its normal-RGB input")
        if not np.array_equal(renderer_mask, snapshot):
            raise ValueError("renderer mutated its realization-mask input")
        if not isinstance(raw, np.ndarray) or raw.shape != normal_rgb.shape or raw.dtype != np.uint8:
            raise ValueError("renderer raw RGB must be 512x512 uint8")
        return np.array(raw, dtype=np.uint8, order="C", copy=True), snapshot


def _verify_import_source(path: Path, spec: FrozenSpec, relative: str) -> None:
    actual = path.resolve(strict=True)
    expected = (spec.repository_root / relative).resolve(strict=True)
    if actual != expected or not actual.is_relative_to(spec.repository_root.resolve(strict=True)):
        raise ValueError(f"imported code is outside the frozen repository: {relative}")
    bindings = {asset.path.resolve(strict=True): asset.sha256 for asset in spec.assets}
    if actual not in bindings or _sha256(actual) != bindings[actual]:
        raise ValueError(f"imported code is not bound to frozen assets: {relative}")


def assert_frozen_inference(renderer: Any, *, device: str) -> None:
    """Check read-only inference state without changing parameters or their flags.

    requires_grad=True on adapter/conditioner is a residual P2 inference
    contract, not authorization to train. The existing renderer inference_mode
    disables autograd; this boundary creates no optimizer and permits no grads.
    """
    import torch

    base = (renderer.components.unet, renderer.components.vae,
            renderer.components.text_encoder)
    roots = (renderer.model, *base)
    modules = {id(module): module for root in roots for module in root.modules()}
    if any(module.training for module in modules.values()):
        raise ValueError("Claim-0 inference requires every module in eval mode")

    seen_containers: set[int] = set()

    def owns_optimizer(value: Any) -> bool:
        if isinstance(value, torch.optim.Optimizer):
            return True
        if isinstance(value, (dict, list, tuple, set, frozenset)):
            if id(value) in seen_containers:
                return False
            seen_containers.add(id(value))
            children = value.values() if isinstance(value, dict) else value
            return any(owns_optimizer(child) for child in children)
        return False

    if any(owns_optimizer(value)
           for owner in (renderer, *modules.values())
           for value in vars(owner).values()):
        raise ValueError("Claim-0 inference must not own an optimizer; training is not authorized")
    if any(parameter.requires_grad for component in base for parameter in component.parameters()):
        raise ValueError("Claim-0 base UNet/VAE/text_encoder must have requires_grad=False")
    if any(parameter.grad is not None for root in roots for parameter in root.parameters()):
        raise ValueError("Claim-0 inference must not accumulate parameter gradients")

    expected_device = torch.device(device)
    trainable = tuple(renderer.model.trainable_parameters())
    if not trainable or any(
        parameter.dtype is not torch.float32
        or parameter.device != expected_device
        or not parameter.requires_grad
        for parameter in trainable
    ):
        raise ValueError(
            "Claim-0 adapter/conditioner must retain FP32 + requires_grad=True "
            "on the inference device for the residual P2 inference contract; "
            "this does not authorize training"
        )


def load_frozen_b(spec: FrozenSpec, device: str = "cuda:0") -> Claim0ModuleB:
    """Explicit opt-in model loading; hash validation always precedes loader import."""
    if (not isinstance(spec.accepted_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", spec.accepted_sha256)
            or spec.accepted_sha256 != spec.manifest_sha256):
        raise ValueError("real model loading requires an externally accepted manifest SHA256")
    verify_freeze(spec)
    _verify_import_source(Path(__file__), spec, "src/cadiff/v06_b/claim0.py")
    from cadiff.p2_a1.config import P2A1Config
    from cadiff.v06_c import visa_exact_mask_r as loader_module
    _verify_import_source(Path(loader_module.__file__), spec,
                          "src/cadiff/v06_c/visa_exact_mask_r.py")

    config_values = json.loads(_canonical(spec.config))
    config_values["defect_types"] = tuple(config_values["defect_types"])
    config_values["eval_timesteps"] = tuple(config_values["eval_timesteps"])
    renderer = loader_module.load_visa_renderer(
        checkpoint=spec.checkpoint_root, model_root=spec.model_root,
        config=P2A1Config(**config_values), device=device,
    )
    renderer.ddim_steps = 50
    # Keep P2 adapter/conditioner FP32 masters and requires_grad=True: this is
    # a residual inference contract, not permission to train. Do not recursively
    # clear renderer.model's flags; its existing inference_mode disables autograd.
    base = (renderer.components.unet, renderer.components.vae,
            renderer.components.text_encoder)
    for component in base:
        component.requires_grad_(False)
    for module in (renderer.model, *base):
        module.eval()
    assert_frozen_inference(renderer, device=device)
    module_b = Claim0ModuleB(renderer, spec)
    module_b._verified = True
    return module_b


__all__ = ["TYPE_MAP", "FrozenAsset", "FrozenSpec", "load_freeze_manifest",
           "verify_freeze", "load_frozen_b", "assert_frozen_inference", "Claim0ModuleB", "validate_proposal",
           "validate_normal", "validate_request_type_seed"]

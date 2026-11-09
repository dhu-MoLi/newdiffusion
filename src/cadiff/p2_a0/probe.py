from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from cadiff.config import ProjectConfig
from cadiff.data.geometry import GeometryMeta, resize_pad_image, resize_pad_mask
from cadiff.diffusion.components import REQUIRED, load_frozen_components, validate_model_layout
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.config import P2A0Config
from cadiff.p2_a0.masking import LatentMasks, build_latent_masks, build_masked_context
from cadiff.p2_a0.model import MaskedInfillAdapter, P2InfillNoisePredictor
from cadiff.p2_a0.objective import InfillLoss, masked_infill_loss
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic


def _capture_parameter_versions(module: nn.Module) -> tuple[int, ...]:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch module")
    return tuple(parameter._version for parameter in module.parameters())


def _assert_parameter_versions_unchanged(
    module: nn.Module,
    expected: tuple[int, ...],
) -> None:
    if not isinstance(expected, tuple) or any(type(value) is not int for value in expected):
        raise TypeError("expected parameter versions must be a tuple of integers")
    actual = _capture_parameter_versions(module)
    if len(actual) != len(expected):
        raise AssertionError("parameter count changed while checking versions")
    changed = tuple(
        index
        for index, pair in enumerate(zip(actual, expected, strict=True))
        if pair[0] != pair[1]
    )
    if changed:
        raise AssertionError(f"parameter version changed at indices {changed}")


@dataclass(frozen=True)
class CapacityProbeContext:
    paths: ProjectConfig
    experiment: P2A0Config
    manifest: Path
    lock: Path
    output: Path
    config_paths: dict[str, Path]
    cwd: Path


@dataclass(frozen=True)
class CapacitySample:
    sample_id: str
    image_relpath: str
    mask_relpath: str
    image_path: Path
    mask_path: Path
    image_sha256: str
    mask_sha256: str
    pixels: Tensor
    mask: Tensor
    valid: Tensor
    geometry: GeometryMeta


@dataclass(frozen=True)
class CapacityStepMetrics:
    total: float
    noise: float
    inside: float
    outside: float
    grad_l2: float
    finite: bool


@dataclass(frozen=True)
class CapacityPreflight:
    context: CapacityProbeContext
    sample: CapacitySample
    git_head: str
    input_hashes: dict[str, str]
    source_snapshot: dict[str, dict[str, int | str]]
    model_snapshot: dict[str, dict[str, int | str]]


def _strict_object(line: str, line_number: int) -> dict[str, object]:
    def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"manifest line {line_number} has duplicate key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            line,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"manifest line {line_number} has non-finite {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"manifest line {line_number} is invalid JSON: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest line {line_number} must be a JSON object")
    return value


def _contained_file(root: Path, relpath: str, *, name: str) -> Path:
    if not isinstance(relpath, str) or not relpath:
        raise ValueError(f"{name} relpath must be a non-empty string")
    path = (root / relpath).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes dataset root: {path}") from error
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file: {path}")
    return path


def load_capacity_sample(
    manifest: Path,
    dataset_root: Path,
    config: P2A0Config,
) -> CapacitySample:
    if manifest.is_symlink():
        raise ValueError(f"manifest symlink is not allowed: {manifest}")
    manifest = manifest.resolve(strict=True)
    if not manifest.is_file():
        raise ValueError(f"manifest must be a regular file: {manifest}")
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dataset root must be a directory: {root}")
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = _strict_object(line, line_number)
            relpath = raw.get("image_relpath")
            if not isinstance(relpath, str) or not relpath:
                raise ValueError(
                    f"manifest line {line_number} image_relpath must be non-empty"
                )
            if relpath in seen:
                raise ValueError(f"duplicate image_relpath: {relpath}")
            seen.add(relpath)
            roles = raw.get("roles")
            if (
                raw.get("category") == config.category
                and raw.get("defect_type") == config.capacity_defect_type
                and isinstance(roles, list)
                and "fewshot_train" in roles
            ):
                candidates.append(raw)
    if not candidates:
        raise ValueError("manifest has no broken_large fewshot_train capacity sample")
    raw = min(candidates, key=lambda value: str(value["image_relpath"]))
    image_relpath = str(raw["image_relpath"])
    mask_relpath = raw.get("mask_relpath")
    image_sha256 = raw.get("image_sha256")
    mask_sha256 = raw.get("mask_sha256")
    if not isinstance(mask_relpath, str) or not mask_relpath:
        raise ValueError("capacity sample must have a mask_relpath")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("capacity sample must have an image SHA256")
    if not isinstance(mask_sha256, str) or len(mask_sha256) != 64:
        raise ValueError("capacity sample must have a mask SHA256")
    image_path = _contained_file(root, image_relpath, name="image")
    mask_path = _contained_file(root, mask_relpath, name="mask")
    if sha256_file(image_path) != image_sha256:
        raise ValueError("capacity image SHA256 mismatch")
    if sha256_file(mask_path) != mask_sha256:
        raise ValueError("capacity mask SHA256 mismatch")
    with Image.open(image_path) as image:
        transformed, valid_image, geometry = resize_pad_image(image, config.image_size)
    with Image.open(mask_path) as mask_image:
        aligned_mask = resize_pad_mask(mask_image, geometry)
    pixel_array = np.array(transformed, dtype=np.uint8, copy=True)
    pixels = (
        torch.from_numpy(pixel_array)
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(127.5)
        .sub_(1.0)
        .contiguous()
    )
    mask_array = np.array(aligned_mask, dtype=np.uint8, copy=True)
    mask = torch.from_numpy(mask_array).ge(128).to(dtype=torch.float32).unsqueeze(0)
    valid_array = np.array(valid_image, dtype=np.uint8, copy=True)
    valid = torch.from_numpy(valid_array).ge(128).unsqueeze(0)
    sample_id = hashlib.sha256(
        f"{config.category}\n{config.capacity_defect_type}\n{image_relpath}".encode()
    ).hexdigest()
    return CapacitySample(
        sample_id=sample_id,
        image_relpath=image_relpath,
        mask_relpath=mask_relpath,
        image_path=image_path,
        mask_path=mask_path,
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        pixels=pixels,
        mask=mask.contiguous(),
        valid=valid.contiguous(),
        geometry=geometry,
    )


def _snapshot(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _model_snapshot(root: Path) -> dict[str, dict[str, int | str]]:
    return {relative: _snapshot(root / relative) for relative in REQUIRED}


def _git_state(cwd: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return head, dirty


def _validate_gpu_contract() -> None:
    required = {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must be {expected}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must enable deterministic CuBLAS")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("exact NVIDIA GeForce RTX 3090 is required")


def _regular_input(path: Path, *, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} symlink is not allowed: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file: {resolved}")
    return resolved


def _validate_hf_home(cwd: Path) -> Path:
    raw = os.environ.get("HF_HOME")
    if not raw:
        raise RuntimeError("HF_HOME must be set")
    candidate = Path(raw)
    if candidate.is_symlink():
        raise RuntimeError("HF_HOME symlink is not allowed")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError("HF_HOME must be a directory")
    root = cwd.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError("HF_HOME must be inside the current worktree") from error
    return resolved


def _validated_output(output: Path, runs_root: Path) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    root = runs_root.resolve(strict=True)
    resolved = output.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("output must be strictly inside runs_root") from error
    if relative == Path("."):
        raise ValueError("output must be strictly inside runs_root")
    return resolved


def validate_capacity_preflight(context: CapacityProbeContext) -> CapacityPreflight:
    _validate_gpu_contract()
    _validate_hf_home(context.cwd)
    head, dirty = _git_state(context.cwd)
    if dirty:
        raise RuntimeError("Git worktree must be clean")
    output = _validated_output(context.output, context.paths.runs_root)
    if output != context.output.resolve(strict=False):
        raise AssertionError("output resolution changed")
    validate_model_layout(context.paths.model_root)
    manifest = _regular_input(context.manifest, name="manifest")
    lock = _regular_input(context.lock, name="lock")
    config_paths = {
        name: _regular_input(path, name=f"config {name}")
        for name, path in sorted(context.config_paths.items())
    }
    sample = load_capacity_sample(manifest, context.paths.dataset_root, context.experiment)
    source_snapshot = {
        sample.image_relpath: _snapshot(sample.image_path),
        sample.mask_relpath: _snapshot(sample.mask_path),
    }
    model_snapshot = _model_snapshot(context.paths.model_root)
    input_hashes = {
        "manifest": sha256_file(manifest),
        "lock": sha256_file(lock),
        **{f"config:{name}": sha256_file(path) for name, path in config_paths.items()},
        "image": sample.image_sha256,
        "mask": sample.mask_sha256,
    }
    return CapacityPreflight(
        context=context,
        sample=sample,
        git_head=head,
        input_hashes=input_hashes,
        source_snapshot=source_snapshot,
        model_snapshot=model_snapshot,
    )


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, nn.Parameter)
    )


def capacity_optimizer_step(
    *,
    model: Any,
    scheduler: object,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    target_latent: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    soft_mask: Tensor,
    prompt_embeds: Tensor,
    defect_type: str,
    timesteps: Tensor,
    noise: Tensor,
    config: P2A0Config,
) -> CapacityStepMetrics:
    trainable = tuple(model.trainable_parameters())
    optimized = _optimizer_parameters(optimizer)
    if not trainable or {id(value) for value in trainable} != {
        id(value) for value in optimized
    } or len(trainable) != len(optimized):
        raise ValueError("optimizer must contain exactly model trainable_parameters")
    optimizer.zero_grad(set_to_none=True)
    noisy = scheduler.add_noise(target_latent, noise, timesteps)
    prediction = model(
        noisy,
        context,
        binary_mask,
        timesteps,
        prompt_embeds,
        defect_type,
    )
    losses: InfillLoss = masked_infill_loss(
        scheduler=scheduler,
        noisy=noisy,
        epsilon_hat=prediction,
        epsilon_target=noise,
        target_latent=target_latent,
        soft_mask=soft_mask,
        timesteps=timesteps,
        inside_noise_weight=config.inside_noise_weight,
        lambda_inside=config.lambda_inside,
        lambda_outside=config.lambda_outside,
    )
    scaler.scale(losses.total).backward()
    scaler.unscale_(optimizer)
    selector = getattr(model, "gradient_parameters", None)
    required = tuple(selector(defect_type)) if callable(selector) else trainable
    trainable_ids = {id(parameter) for parameter in trainable}
    required_ids = {id(parameter) for parameter in required}
    if (
        not required
        or len(required_ids) != len(required)
        or not required_ids.issubset(trainable_ids)
    ):
        raise ValueError("gradient_parameters must be a unique subset of trainable parameters")
    required_gradients = tuple(parameter.grad for parameter in required)
    if any(gradient is None for gradient in required_gradients):
        raise ValueError("every active trainable parameter must receive a gradient")
    inactive = tuple(
        parameter for parameter in trainable if id(parameter) not in required_ids
    )
    if any(parameter.grad is not None for parameter in inactive):
        raise ValueError("inactive routed parameters must not receive gradients")
    concrete = tuple(
        gradient for gradient in required_gradients if gradient is not None
    )
    if any(not torch.isfinite(gradient).all().item() for gradient in concrete):
        raise ValueError("trainable gradients must be finite")
    grad_l2 = math.sqrt(
        sum(float(gradient.float().square().sum().item()) for gradient in concrete)
    )
    if not math.isfinite(grad_l2) or grad_l2 <= 0.0:
        raise ValueError("at least one trainable gradient must be nonzero and finite")
    scaler.step(optimizer)
    scaler.update()
    if any(not torch.isfinite(parameter).all().item() for parameter in trainable):
        raise ValueError("trainable parameters must remain finite")
    model.assert_frozen_backbone()
    values = (losses.total, losses.noise, losses.inside, losses.outside)
    return CapacityStepMetrics(
        total=float(values[0].detach().item()),
        noise=float(values[1].detach().item()),
        inside=float(values[2].detach().item()),
        outside=float(values[3].detach().item()),
        grad_l2=grad_l2,
        finite=all(torch.isfinite(value).all().item() for value in values),
    )


@torch.no_grad()
def _evaluate_bank(
    *,
    model: P2InfillNoisePredictor,
    scheduler: object,
    target: Tensor,
    context: Tensor,
    masks: LatentMasks,
    prompt: Tensor,
    config: P2A0Config,
    bank: tuple[tuple[Tensor, Tensor], ...],
) -> dict[str, float]:
    totals: dict[str, list[float]] = {
        "total": [],
        "noise": [],
        "inside": [],
        "outside": [],
    }
    for timesteps, noise in bank:
        noisy = scheduler.add_noise(target, noise, timesteps)
        prediction = model(
            noisy,
            context,
            masks.binary,
            timesteps,
            prompt,
            config.capacity_defect_type,
        )
        losses = masked_infill_loss(
            scheduler=scheduler,
            noisy=noisy,
            epsilon_hat=prediction,
            epsilon_target=noise,
            target_latent=target,
            soft_mask=masks.soft,
            timesteps=timesteps,
            inside_noise_weight=config.inside_noise_weight,
            lambda_inside=config.lambda_inside,
            lambda_outside=config.lambda_outside,
        )
        for name in totals:
            totals[name].append(float(getattr(losses, name).item()))
    return {name: float(np.mean(values, dtype=np.float64)) for name, values in totals.items()}


def _tensor_to_pil(image: Tensor) -> Image.Image:
    array = (
        image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
    ).astype(np.uint8)
    return Image.fromarray(array)


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError("prediction must be a floating tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError("prediction must contain only finite values")
    canonical = value.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(canonical.dtype), "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = canonical.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(metadata + b"\n" + payload).hexdigest()


@torch.no_grad()
def _evaluate_type_controls(
    *,
    model: Any,
    unet: Any,
    noisy: Tensor,
    context: Tensor,
    binary_mask: Tensor,
    timesteps: Tensor,
    prompt: Tensor,
    correct_type: str,
    wrong_type: str,
) -> dict[str, Any]:
    correct = model(
        noisy,
        context,
        binary_mask,
        timesteps,
        prompt,
        correct_type,
    )
    wrong = model(
        noisy,
        context,
        binary_mask,
        timesteps,
        prompt,
        wrong_type,
    )
    normal = unet(noisy, timesteps, encoder_hidden_states=prompt).sample
    if correct.shape != wrong.shape or correct.shape != normal.shape:
        raise ValueError("type-control predictions must have identical shapes")
    hashes = {
        "correct": _tensor_sha256(correct),
        "wrong": _tensor_sha256(wrong),
        "normal": _tensor_sha256(normal),
    }
    return {
        "prediction_sha256": hashes,
        "correct_vs_wrong_different": hashes["correct"] != hashes["wrong"],
        "correct_vs_normal_different": hashes["correct"] != hashes["normal"],
    }


def _execute_capacity(preflight: CapacityPreflight) -> dict[str, Any]:
    context = preflight.context
    config = context.experiment
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats()
    components = load_frozen_components(context.paths.model_root, device, torch.float16)
    unet_versions = _capture_parameter_versions(components.unet)
    pixels = preflight.sample.pixels.unsqueeze(0).to(device=device, dtype=torch.float16)
    target = encode_mode(components.vae, pixels).detach().clone()
    masks = build_latent_masks(
        preflight.sample.mask.unsqueeze(0).to(device=device),
        (target.shape[2], target.shape[3]),
    )
    context_latent = build_masked_context(
        target,
        masks.binary,
        torch.Generator(device=device).manual_seed(config.context_fill_seed),
    )
    prompt = encode_prompt(components, config.normal_prompt, device).detach().clone()
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    conditioner = DefectConditioner(vocabulary, 128).to(device=device, dtype=torch.float32)
    adapter = MaskedInfillAdapter(config).to(device=device, dtype=torch.float32)
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).train()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=config.learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    bank: list[tuple[Tensor, Tensor]] = []
    for timestep in config.eval_timesteps:
        timesteps = torch.tensor([timestep], device=device, dtype=torch.long)
        noise = torch.randn(
            target.shape,
            generator=torch.Generator(device=device).manual_seed(
                config.noise_seed + timestep
            ),
            device=device,
            dtype=target.dtype,
        )
        bank.append((timesteps, noise))
    fixed_bank = tuple(bank)
    model.eval()
    step0 = _evaluate_bank(
        model=model,
        scheduler=components.scheduler,
        target=target,
        context=context_latent,
        masks=masks,
        prompt=prompt,
        config=config,
        bank=fixed_bank,
    )
    model.train()
    last_metrics: CapacityStepMetrics | None = None
    for step in range(config.steps):
        timesteps, noise = fixed_bank[step % len(fixed_bank)]
        last_metrics = capacity_optimizer_step(
            model=model,
            scheduler=components.scheduler,
            optimizer=optimizer,
            scaler=scaler,
            target_latent=target,
            context=context_latent,
            binary_mask=masks.binary,
            soft_mask=masks.soft,
            prompt_embeds=prompt,
            defect_type=config.capacity_defect_type,
            timesteps=timesteps,
            noise=noise,
            config=config,
        )
    if last_metrics is None:
        raise RuntimeError("capacity training completed no steps")
    model.eval()
    final = _evaluate_bank(
        model=model,
        scheduler=components.scheduler,
        target=target,
        context=context_latent,
        masks=masks,
        prompt=prompt,
        config=config,
        bank=fixed_bank,
    )
    timestep, noise = fixed_bank[1]
    with torch.no_grad():
        noisy = components.scheduler.add_noise(target, noise, timestep)
        prediction = model(
            noisy,
            context_latent,
            masks.binary,
            timestep,
            prompt,
            config.capacity_defect_type,
        )
        diagnostic = masked_infill_loss(
            scheduler=components.scheduler,
            noisy=noisy,
            epsilon_hat=prediction,
            epsilon_target=noise,
            target_latent=target,
            soft_mask=masks.soft,
            timesteps=timestep,
            inside_noise_weight=config.inside_noise_weight,
            lambda_inside=config.lambda_inside,
            lambda_outside=config.lambda_outside,
        )
        composite = (
            masks.binary * diagnostic.predicted_clean
            + (1.0 - masks.binary) * target.float()
        ).to(dtype=target.dtype)
        target_rgb = decode_latent(components.vae, target)[0]
        context_rgb = decode_latent(components.vae, context_latent)[0]
        predicted_rgb = decode_latent(components.vae, composite)[0]
        wrong_type = next(
            defect_type
            for defect_type in config.defect_types
            if defect_type != config.capacity_defect_type
        )
        type_controls = _evaluate_type_controls(
            model=model,
            unet=components.unet,
            noisy=noisy,
            context=context_latent,
            binary_mask=masks.binary,
            timesteps=timestep,
            prompt=prompt,
            correct_type=config.capacity_defect_type,
            wrong_type=wrong_type,
        )
    cells = [
        _tensor_to_pil(((pixels[0].float().cpu() + 1.0) / 2.0).clamp(0, 1)),
        _tensor_to_pil(target_rgb),
        _tensor_to_pil(context_rgb),
        _tensor_to_pil(predicted_rgb),
    ]
    grid = Image.new("RGB", (config.image_size * len(cells), config.image_size))
    for index, cell in enumerate(cells):
        grid.paste(cell, (index * config.image_size, 0))
    grid_path = context.output / "capacity_grid.png"
    grid.save(grid_path)
    mask_path = context.output / "capacity_mask.png"
    Image.fromarray(
        (preflight.sample.mask[0].numpy() * 255).astype(np.uint8), mode="L"
    ).save(mask_path)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated()
    _assert_parameter_versions_unchanged(components.unet, unet_versions)
    if any(parameter.grad is not None for parameter in components.unet.parameters()):
        raise RuntimeError("frozen U-Net accumulated gradients")
    return {
        "step0": step0,
        "final": final,
        "last_train_step": asdict(last_metrics),
        "inside_decreased": final["inside"] < step0["inside"],
        "noise_decreased": final["noise"] < step0["noise"],
        "type_controls": type_controls,
        "visuals": [grid_path.name, mask_path.name],
        "peak_vram_bytes": peak,
    }


def _output_identity(path: Path) -> tuple[int, int]:
    information = os.lstat(path)
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISDIR(information.st_mode):
        raise RuntimeError("output must remain a real directory")
    return information.st_dev, information.st_ino


def _verify_output_identity(
    preflight: CapacityPreflight, expected: tuple[int, int]
) -> None:
    path = preflight.context.output
    if _output_identity(path) != expected:
        raise RuntimeError("output identity changed during capacity run")
    resolved = path.resolve(strict=True)
    root = preflight.context.paths.runs_root.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError("output escaped runs_root during capacity run") from error
    if relative == Path("."):
        raise RuntimeError("output must remain strictly inside runs_root")


def _verify_integrity(
    preflight: CapacityPreflight, output_identity: tuple[int, int]
) -> str:
    head, dirty = _git_state(preflight.context.cwd)
    if dirty or head != preflight.git_head:
        raise RuntimeError("Git changed during capacity run")
    current_sources = {
        preflight.sample.image_relpath: _snapshot(preflight.sample.image_path),
        preflight.sample.mask_relpath: _snapshot(preflight.sample.mask_path),
    }
    if current_sources != preflight.source_snapshot:
        raise RuntimeError("capacity source changed during run")
    if _model_snapshot(preflight.context.paths.model_root) != preflight.model_snapshot:
        raise RuntimeError("model files changed during capacity run")
    current_hashes = {
        "manifest": sha256_file(preflight.context.manifest.resolve(strict=True)),
        "lock": sha256_file(preflight.context.lock.resolve(strict=True)),
        **{
            f"config:{name}": sha256_file(path.resolve(strict=True))
            for name, path in sorted(preflight.context.config_paths.items())
        },
        "image": sha256_file(preflight.sample.image_path),
        "mask": sha256_file(preflight.sample.mask_path),
    }
    if current_hashes != preflight.input_hashes:
        raise RuntimeError("capacity input hash changed during run")
    _verify_output_identity(preflight, output_identity)
    return head


def run_capacity_probe(context: CapacityProbeContext) -> int:
    try:
        preflight = validate_capacity_preflight(context)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        print(
            json.dumps(
                {"reason": "preflight_failure", "status": "failed"},
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    context.output.mkdir(parents=True, exist_ok=False)
    output_identity = _output_identity(context.output)
    try:
        result = _execute_capacity(preflight)
        head_end = _verify_integrity(preflight, output_identity)
        report = {
            "schema_version": "p2_a0_capacity_report_v1",
            "status": "complete",
            "reason": None,
            "category": context.experiment.category,
            "defect_type": context.experiment.capacity_defect_type,
            "sample": {
                "sample_id": preflight.sample.sample_id,
                "image_relpath": preflight.sample.image_relpath,
                "mask_relpath": preflight.sample.mask_relpath,
            },
            "capacity_diagnostic": result,
            "input_hashes": preflight.input_hashes,
        }
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": "p2_a0_capacity_provenance_v1",
                "git_head_start": preflight.git_head,
                "git_head_end": head_end,
                "git_dirty": False,
                "input_hashes": preflight.input_hashes,
                "source_snapshot_before": preflight.source_snapshot,
                "source_snapshot_after": preflight.source_snapshot,
                "model_snapshot_before": preflight.model_snapshot,
                "model_snapshot_after": preflight.model_snapshot,
            }
        )
        write_json_atomic(context.output / "provenance.json", provenance)
        _verify_integrity(preflight, output_identity)
        write_json_atomic(context.output / "report.json", report)
        _verify_integrity(preflight, output_identity)
        return 0
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError):
        failed = {
            "schema_version": "p2_a0_capacity_report_v1",
            "status": "failed",
            "reason": "runtime_or_integrity_failure",
        }
        try:
            _verify_output_identity(preflight, output_identity)
        except (OSError, RuntimeError):
            pass
        else:
            try:
                write_json_atomic(context.output / "report.json", failed)
            except OSError:
                pass
        print(json.dumps(failed, sort_keys=True), flush=True)
        return 1

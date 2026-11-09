"""Fresh, matched-budget VisA R runs; all file writes require a new output root."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import math
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from cadiff.config import load_config
from cadiff.data.geometry import resize_pad_image
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.fewshot import _prepare_samples, _resampled_training_steps, balanced_training_order
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_a1.model import FactorizedMaskedInfillAdapter
from cadiff.provenance import sha256_file
from cadiff.v06_c.visa_exact_mask_r import _load_sample, CHECKPOINT_SCHEMA
from .config import RExperimentConfig
from .training import r_optimizer_step
from .losses import freeze_appearance_vae

def _json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

def write_new_json(path: Path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

def claim_output(output: Path):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite R experiment: {output}")
    output.mkdir(parents=True, exist_ok=False)

def validate_training_config(p2: P2A1Config, *, smoke: bool, forced_timestep: int | None):
    from cadiff.v06_c.visa_exact_mask_r import visa_p2_a1_config
    frozen = visa_p2_a1_config(p2.category, p2.defect_types)
    allowed_changes = {"seed", "context_fill_seed", "noise_seed", "steps"}
    baseline, current = asdict(frozen), asdict(p2)
    if any(current[k] != baseline[k] for k in baseline if k not in allowed_changes):
        raise ValueError("R experiment may only vary seeds/steps, retaining frozen P2 protocol")
    if type(p2.steps) is not int or p2.steps <= 0 or (not smoke and p2.steps != 1200):
        raise ValueError("production uses 1200 successful steps; smoke needs positive steps")
    if forced_timestep is not None and (not smoke or type(forced_timestep) is not int or forced_timestep < 0):
        raise ValueError("forced_timestep is nonnegative and smoke-only")
    if smoke and p2.steps > 4:
        raise ValueError("GPU smoke budget is at most four steps")

def load_train_manifest(manifest: Path, dataset_root: Path, config: P2A1Config):
    """Open only fewshot_train image/mask files; never materialize held-out samples."""
    root = dataset_root.resolve(strict=True)
    train, valid_by_id, records = [], {}, []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("category") != config.category or config.train_role not in raw.get("roles", []):
            continue
        if raw.get("roles") != [config.train_role]:
            raise ValueError(f"manifest {line_number} mixes train/other roles")
        for key in ("image_relpath", "mask_relpath"):
            path = root / raw[key]
            try:
                path.resolve(strict=True).relative_to(root)
            except ValueError as e:
                raise ValueError("train paths escape dataset root") from e
        sample = _load_sample(raw, root, config.train_role, config)
        with Image.open(sample.image_path) as image:
            _, valid, _ = resize_pad_image(image, config.image_size)
        valid_tensor = torch.from_numpy(np.array(valid, copy=True)).ge(128).float().unsqueeze(0)
        if torch.any(sample.mask > valid_tensor):
            raise ValueError("GT intersects invalid letterbox pixels")
        train.append(sample)
        valid_by_id[sample.sample_id] = valid_tensor
        records.append({k: getattr(sample, k) for k in ("sample_id", "defect_type", "role", "image_relpath", "mask_relpath", "image_sha256", "mask_sha256")})
    train = tuple(sorted(train, key=lambda s: s.image_relpath))
    if not train or {s.defect_type for s in train} != set(config.defect_types):
        raise ValueError("training manifest must cover the complete vocabulary")
    if len({s.sample_id for s in train}) != len(train):
        raise ValueError("duplicate training sample")
    records = sorted(records, key=lambda r: r["image_relpath"])
    return train, valid_by_id, records

class AppliedAdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied_steps = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.applied_steps += 1
        return result

def save_training_checkpoint(path: Path, *, model, optimizer, scaler, generator, p2, experiment, identity, completed_steps):
    claim_output(path)
    files = {}
    for name in ("adapter", "conditioner"):
        module = getattr(model, name)
        state = {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}
        if not state or any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("checkpoint must contain finite adapter/conditioner state")
        f = path / f"{name}.safetensors"
        save_file(state, f)
        files[f.name] = {"sha256": sha256_file(f), "size": f.stat().st_size}
    state = {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "generator_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if generator.device.type == "cuda" else [],
             "completed_steps": completed_steps, "identity": identity}
    torch.save(state, path / "resume.pt")
    files["resume.pt"] = {"sha256": sha256_file(path / "resume.pt"), "size": (path / "resume.pt").stat().st_size}
    vocabulary = DefectVocabulary.from_config(p2.category, p2.defect_types)
    metadata = {"schema_version": CHECKPOINT_SCHEMA, "config": _json_value(asdict(p2)),
                "vocabulary_sha256": vocabulary.sha256, "files": files,
                "experiment": asdict(experiment), "completed_steps": completed_steps,
                "ranking_negative": "all_wrong_types_and_normal", "ranking_selection": "hardest_min",
                "diagnostic_only": True, "not_official_p2_a1": True, "identity": identity}
    write_new_json(path / "metadata.json", metadata)
    return metadata

def restore_training_checkpoint(path: Path, *, model, optimizer, scaler, generator, identity):
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("identity") != identity:
        raise ValueError("resume identity differs (data/config/seeds/mode must match)")
    for name, info in metadata["files"].items():
        if sha256_file(path / name) != info["sha256"]:
            raise ValueError("resume checkpoint hash mismatch")
    for name in ("adapter", "conditioner"):
        getattr(model, name).load_state_dict(load_file(path / f"{name}.safetensors"), strict=True)
    state = torch.load(path / "resume.pt", map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("resume state identity differs")
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    generator.set_state(state["generator_state"])
    torch.set_rng_state(state["torch_rng"])
    if generator.device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["completed_steps"])

def train_r_experiment(*, paths_yaml: Path, manifest: Path, dataset_root: Path, output: Path,
                       config: P2A1Config, experiment: RExperimentConfig, device: str = "cuda:0",
                       smoke: bool = False, forced_timestep: int | None = None,
                       resume: Path | None = None, stop_after: int | None = None):
    validate_training_config(config, smoke=smoke, forced_timestep=forced_timestep)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite R experiment: {output}")
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= config.steps):
        raise ValueError("stop_after must be within the configured training budget")
    samples, valid_masks, records = load_train_manifest(manifest, dataset_root, config)
    identity = _json_value({"p2_config": asdict(config), "experiment": asdict(experiment), "train": records,
                           "manifest_sha256": sha256_file(manifest), "smoke": smoke,
                           "forced_timestep": forced_timestep, "paths_sha256": sha256_file(paths_yaml)})
    paths = load_config(paths_yaml)
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("full SD1.5 runner requires CUDA; loss and artifact tests run on CPU")
    claim_output(output)
    write_new_json(output / "run_manifest.json", {**identity, "resume_from": str(resume) if resume else None,
                   "ranking_negative": "all_wrong_types_and_normal", "ranking_selection": "hardest_min",
                   "gt_only": True, "no_test_loading": True, "outside_composition": "torch.where(binary_gt, raw, normal)",
                   "loss": "legacy_base_plus_legacy_ranking_plus_correct_gated_appearance"})
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    components = load_frozen_components(paths.model_root, torch_device, torch.float16)
    prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
    prepared = _prepare_samples(vae=components.vae, samples=samples, device=torch_device, config=config, seed_offset=0)
    by_id = {p.sample.sample_id: p for p in prepared}
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    conditioner = DefectConditioner(vocabulary, 128).to(device=torch_device, dtype=torch.float32)
    adapter = FactorizedMaskedInfillAdapter(config).to(device=torch_device, dtype=torch.float32)
    # Preserve legacy FP16 encoding and initializer RNG; all three arms then share
    # a frozen FP32 VAE, with safe decode gradients for the two appearance arms.
    freeze_appearance_vae(components.vae)
    model = P2InfillNoisePredictor(components.unet, conditioner, adapter).train()
    optimizer = AppliedAdamW(model.trainable_parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    generator = torch.Generator(device=torch_device).manual_seed(config.noise_seed + 10_000_000)
    completed = restore_training_checkpoint(resume, model=model, optimizer=optimizer, scaler=scaler,
                 generator=generator, identity=identity) if resume else 0
    if completed >= config.steps or (stop_after is not None and completed >= stop_after):
        raise ValueError("resume has no remaining requested training steps")
    schedule = balanced_training_order(samples, config.steps, config.defect_types)
    history = []
    num_timesteps = len(components.scheduler.alphas_cumprod)
    if forced_timestep is not None and forced_timestep >= num_timesteps:
        raise ValueError("forced_timestep outside scheduler")
    for step, sample, batch, context_seed, context_sha in _resampled_training_steps(schedule, by_id, config):
        if step <= completed:
            continue
        # Always draw the production random timestep, including smoke overrides.
        t = torch.randint(0, num_timesteps, (1,), generator=generator, device=torch_device, dtype=torch.long)
        if forced_timestep is not None:
            t.fill_(forced_timestep)
        noise = torch.randn(batch.target.shape, generator=generator, device=torch_device, dtype=batch.target.dtype)
        before, scale = optimizer.applied_steps, scaler.get_scale()
        metrics = r_optimizer_step(experiment=experiment, vae=components.vae,
              target_image=((sample.pixels + 1) / 2).unsqueeze(0).to(torch_device),
              image_gt_mask=sample.mask.unsqueeze(0).to(torch_device),
              image_valid_mask=valid_masks[sample.sample_id].unsqueeze(0).to(torch_device),
              model=model, unet=components.unet, scheduler=components.scheduler, optimizer=optimizer, scaler=scaler,
              target_latent=batch.target, context=batch.context, binary_mask=batch.masks.binary,
              soft_mask=batch.masks.soft, prompt_embeds=prompt, defect_type=sample.defect_type,
              timesteps=t, noise=noise, config=config)
        if optimizer.applied_steps != before + 1:
            raise RuntimeError("optimizer step was skipped; successful-step budget cannot advance")
        completed = step
        row = {"step": step, "sample_id": sample.sample_id, "defect_type": sample.defect_type,
               "timestep": int(t.item()), "context_seed": context_seed, "context_sha256": context_sha,
               "applied_optimizer_steps": completed, "amp_scale_before": scale, "amp_scale_after": scaler.get_scale(),
               "amp_overflow_skips": max(0, round(math.log2(scale / scaler.get_scale()))), **asdict(metrics)}
        history.append(row)
        with (output / "training.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        end = step == config.steps or (stop_after is not None and step == stop_after)
        if step % experiment.save_every == 0 or end:
            save_training_checkpoint(output / f"checkpoint-step-{step:06d}", model=model, optimizer=optimizer,
                scaler=scaler, generator=generator, p2=config, experiment=experiment, identity=identity, completed_steps=step)
        if step == 1 or step % 100 == 0 or end:
            print(json.dumps({"stage": "v06_r", "arm": experiment.arm, "step": step, "total": metrics.total}), flush=True)
        if end:
            break
    summary = {"schema_version": "v06_r_training_summary_v1", "arm": experiment.arm,
               "completed_steps": completed, "target_steps": config.steps, "complete": completed == config.steps,
               "checkpoint": str(output / f"checkpoint-step-{completed:06d}"), "fewshot_train": len(samples),
               "no_test_loading": True, "last": history[-1], "output": str(output), "smoke": smoke,
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch_device),
               "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch_device),
               "appearance_decode_updates_this_segment": sum(r["decode_count"] > 0 for r in history),
               "empty_edge_updates_this_segment": sum(r["decode_count"] > 0 and r["edge_pixels"] == 0 for r in history),
               "frozen_unet_unchanged_by_optimizer": all(not p.requires_grad and p.grad is None for p in components.unet.parameters()),
               "frozen_vae_gradient_free": all(not p.requires_grad and p.grad is None for p in components.vae.parameters()),
               "vae_decode_dtype": str(next(components.vae.parameters()).dtype)}
    write_new_json(output / "summary.json", summary)
    return summary

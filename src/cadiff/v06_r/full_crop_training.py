"""Fresh single-view Full versus Crop32; legacy control optimizer path unchanged."""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from cadiff.v06_r.runner import (
    _json_value,write_new_json,claim_output,validate_training_config,load_train_manifest,
    AppliedAdamW,save_training_checkpoint,restore_training_checkpoint,
    load_config,load_frozen_components,encode_prompt,_prepare_samples,
    _resampled_training_steps,balanced_training_order,DefectVocabulary,
    DefectConditioner,FactorizedMaskedInfillAdapter,P2InfillNoisePredictor,
    freeze_appearance_vae,sha256_file,P2A1Config,r_optimizer_step,
)
from cadiff.v06_r.config import RExperimentConfig
from cadiff.v06_r.square_roi_bridge import square_roi_geometry,forward_square,SOURCES as BRIDGE_SOURCES

VIEWS=("full","crop32")
PARAMETERS={"arms":list(VIEWS),"crop_margin":32,"steps":1200,"type_rank":8,"seed":20260913,"context_fill_seed":202609130,"noise_seed":202609131,"learning_rate":1e-4,"loss_arm":"latent_control","rgb_weight":0.,"edge_weight":0.,"save_every":100,"train_encode_dtype":"float16","post_init_vae_dtype":"float32","inference_vae_load_dtype":"float32","inference_encode_dtype":"float32","predictor_latent_dtype":"float16","inference_decode_dtype":"float32","ddim_steps":50,"eta":0,"candidate_count":1}

def validate_protocol(path):
    data=json.loads(path.read_text())
    if data.get("schema_version")!="v06_r_full_crop32_protocol_v1" or data.get("parameters")!=PARAMETERS: raise ValueError("registered full/crop32 protocol differs")
    return data

def load_reference_config(checkpoint,*,smoke=False,steps=1200):
    metadata=json.loads((checkpoint/"metadata.json").read_text())
    if metadata.get("experiment",{}).get("arm")!="latent_control" or metadata.get("completed_steps")!=1200: raise ValueError("reference is not frozen old finalcontrol")
    raw=metadata["config"]; raw["defect_types"]=tuple(raw["defect_types"]); raw["eval_timesteps"]=tuple(raw["eval_timesteps"])
    p2=P2A1Config(**raw)
    for key in ("seed","context_fill_seed","noise_seed","learning_rate","type_rank"):
        if getattr(p2,key)!=PARAMETERS[key]: raise ValueError("reference seed/capacity differs")
    if smoke:
        if type(steps)!=int or not 1<=steps<=4: raise ValueError("smoke at most4 successfulsteps")
        p2=replace(p2,steps=steps)
    elif steps!=1200: raise ValueError("production exactly1200")
    return p2

def tensor_sha256(tensor):
    value=tensor.detach().cpu().contiguous()
    header=json.dumps({"shape":list(value.shape),"dtype":str(value.dtype)},sort_keys=True).encode()
    return hashlib.sha256(header+value.numpy().tobytes()).hexdigest()

def module_sha256(module):
    h=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()): h.update(key.encode()); h.update(tensor_sha256(value).encode())
    return h.hexdigest()

def training_sources():
    root=Path(__file__).resolve().parents[3]
    paths=tuple(dict.fromkeys((*BRIDGE_SOURCES,"src/cadiff/v06_r/full_crop_training.py","tools/v06_r/train_full_crop.py","src/cadiff/v06_r/training.py","src/cadiff/v06_r/losses.py","src/cadiff/p2_a1/ranking.py","src/cadiff/p2_a0/model.py","src/cadiff/p2_a1/model.py","src/cadiff/diffusion/components.py","src/cadiff/diffusion/vae_roundtrip.py")))
    return {p:sha256_file(root/p) for p in paths}

def transform_samples(samples,valid_masks,view):
    if view not in VIEWS: raise ValueError("unregistered view")
    output=[]; transformed_valid={}; records=[]
    for sample in samples:
        legal=valid_masks[sample.sample_id]
        original=(sample.pixels+1)/2
        record={"sample_id":sample.sample_id,"defect_type":sample.defect_type,"source_image_sha256":sample.image_sha256,"source_mask_sha256":sample.mask_sha256,"physical_gt_tensor_sha256":tensor_sha256(sample.mask),"source_pixels_tensor_sha256":tensor_sha256(sample.pixels),"view":view}
        if view=="crop32":
            geometry=square_roi_geometry(sample.mask[None],legal[None],32)
            square,mask,valid=forward_square(original[None],sample.mask[None],geometry)
            sample=replace(sample,pixels=(square[0]*2-1).contiguous(),mask=mask[0].float().contiguous())
            legal=valid[0].float().contiguous(); record["geometry"]=asdict(geometry)
        else: record["geometry"]=None
        record.update({"model_input_pixels_tensor_sha256":tensor_sha256(sample.pixels),"model_input_mask_tensor_sha256":tensor_sha256(sample.mask),"model_input_valid_tensor_sha256":tensor_sha256(legal),"physical_source_hashes_are_not_transformed_hashes":True})
        output.append(sample); transformed_valid[sample.sample_id]=legal; records.append(record)
    return tuple(output),transformed_valid,records

def train_full_crop(*, view: str, reference_checkpoint: Path, protocol: Path, paths_yaml: Path, manifest: Path, dataset_root: Path, output: Path,
                       config: P2A1Config, experiment: RExperimentConfig, device: str = "cuda:0",
                       smoke: bool = False, forced_timestep: int | None = None,
                       resume: Path | None = None, stop_after: int | None = None):
    if view not in VIEWS: raise ValueError("view must be full/crop32")
    protocol_data=validate_protocol(protocol)
    expected=load_reference_config(reference_checkpoint, smoke=smoke, steps=config.steps)
    if asdict(expected)!=asdict(config): raise ValueError("config must exactly inherit old control seeds/protocol")
    if experiment != RExperimentConfig("latent_control",.1,0.,0.,100): raise ValueError("only unchanged legacy loss/save100 is authorized")
    validate_training_config(config, smoke=smoke, forced_timestep=forced_timestep)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite R experiment: {output}")
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= config.steps):
        raise ValueError("stop_after must be within the configured training budget")
    samples, valid_masks, records = load_train_manifest(manifest, dataset_root, config)
    samples,valid_masks,transform_records=transform_samples(samples,valid_masks,view)
    identity = _json_value({"view":view,"protocol_sha256":sha256_file(protocol),"reference_config_sha256":sha256_file(reference_checkpoint/"metadata.json"),"source_sha256":training_sources(),"transform_records":transform_records,"p2_config": asdict(config), "experiment": asdict(experiment), "train": records,
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
                   "loss": "unchanged_legacy_base_plus_legacy_hardest_ranking"})
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
    initial_state={name:module_sha256(getattr(model,name)) for name in ("adapter","conditioner")}
    write_new_json(output/"initial_state_sha256.json",initial_state)
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
               "timestep": int(t.item()), "context_seed": context_seed, "context_sha256": context_sha,"noise_sha256":tensor_sha256(noise),"view":view,
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
            print(json.dumps({"stage": "v06_r", "arm": view, "step": step, "total": metrics.total}), flush=True)
        if end:
            break
    summary = {"schema_version": "v06_r_training_summary_v1", "arm": view,
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

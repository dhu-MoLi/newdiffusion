"""pcb4 control: gray the letterboxed hole before VAE, then keep latent fill.

The official VisA R path encodes the full RGB and only then replaces the latent
hole with noise. This arm first writes uint8 128 into the full H on the existing
512 letterbox, encodes that image for context, and still applies the original
latent fill. The training target and the generation normal reference stay on the
unerased image. Crop32, D4, and the official checkpoint directory are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from cadiff.config import load_config
from cadiff.diffusion.components import load_frozen_components
from cadiff.diffusion.ddim_inversion import encode_prompt
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode
from cadiff.p1_v1.conditioning import DefectConditioner, DefectVocabulary
from cadiff.p2_a0.fewshot import (
    FewshotSample,
    PreparedSample,
    _prepare_samples,
    balanced_training_order,
)
from cadiff.p2_a0.masking import build_latent_masks, build_masked_context
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.p2_a0.probe import _tensor_sha256
from cadiff.p2_a1.model import FactorizedMaskedInfillAdapter
from cadiff.p2_a1.ranking import ranking_optimizer_step
from cadiff.p2_b0.generation import denoise_masked_infill
from cadiff.provenance import write_json_atomic
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.can_exact_mask_r import CanExactMaskRenderer, _save_checkpoint
from cadiff.v06_c.visa_exact_mask_r import (
    CHECKPOINT_SCHEMA,
    load_visa_fewshot_split,
    load_visa_renderer,
    visa_p2_a1_config,
)
from cadiff.v06_c.visa_pilot import load_visa_inventory_for_split
from cadiff.v06_c.visa_place_bulk import pack_visa_eval_generated

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RUNS = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs")
SOURCE_GENERATE = RUNS / "v06_c/visa/pcb4/generate/20260927-magic-500"
EXPERIMENT = RUNS / "v06_c/visa/pcb4/context-erasure-20261102"
TRAIN_OUTPUT = EXPERIMENT / "train"
GENERATE_OUTPUT = EXPERIMENT / "generate"
LAYOUT_OUTPUT = EXPERIMENT / "generated_layout"
DOWNSTREAM_OUTPUT = EXPERIMENT / "downstream"
MVTEC_PATH = RUNS / "v06_c/visa/pcb4/mvtec_layout_magic_20260927"
PATHS_YAML = PACKAGE_ROOT / "configs/paths/omen-home.yaml"
MANIFEST = PACKAGE_ROOT / "splits/v06_b/visa/pcb4/20260927-magic/manifest.jsonl"
DATASET_ROOT = Path("/home/dell/2T/数据集/AD/VisA")
PYTHON = Path(
    "/home/dell/2T/lm/new_projects/category_anomaly_diffusion/.conda/envs/cad-v0/bin/python"
)
GRAY_UINT8 = 128
ANOMALY_NAMES = ("damage", "extra", "missing")
EVAL_ANOMALY_NAMES = ("burnt", "damage", "extra", "missing", "scratch", "wrong_place")


def _uint8_canvas(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("training RGB must be CHW")
    restored = (pixels.detach().float().cpu() + 1.0) * 127.5
    canvas = restored.round().clamp(0, 255).to(torch.uint8)
    restored_pixels = canvas.float().div(127.5).sub(1.0)
    if not torch.allclose(restored_pixels, pixels.detach().float().cpu(), atol=1e-5, rtol=0):
        raise ValueError("letterbox RGB does not round-trip through uint8")
    return canvas


def _erase_uint8(canvas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[-2:] != canvas.shape[-2:]:
        raise ValueError("H must match the 512 letterbox")
    hole = mask.detach().reshape(mask.shape[-2], mask.shape[-1]).cpu() > 0
    if not bool(hole.any()) or bool(hole.all()):
        raise ValueError("H must be a non-empty partial hole on the letterbox")
    erased = canvas.clone()
    erased[:, hole] = GRAY_UINT8
    if not torch.equal(erased[:, ~hole], canvas[:, ~hole]):
        raise ValueError("erasure changed RGB outside H")
    return erased


def _model_pixels(canvas: torch.Tensor, device: torch.device) -> torch.Tensor:
    return canvas.float().div(127.5).sub(1.0).unsqueeze(0).to(device=device, dtype=torch.float16)


def _capture_rng(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(device)


def _restore_rng(device: torch.device, state: tuple[torch.Tensor, torch.Tensor]) -> None:
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1], device)


def _encode_erased_contexts(
    samples: tuple[PreparedSample, ...],
    vae: object,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    contexts: dict[str, torch.Tensor] = {}
    report: list[dict[str, float]] = []
    for index, prepared in enumerate(samples):
        sample = prepared.sample
        canvas = _uint8_canvas(sample.pixels)
        erased = _erase_uint8(canvas, sample.mask)
        latent = encode_mode(vae, _model_pixels(erased, device)).detach()
        if latent.shape != prepared.target.shape or latent.dtype != prepared.target.dtype:
            raise ValueError("erased context latent does not match the training target")
        seed = 20_260_9130 + index
        official = build_masked_context(
            prepared.target,
            prepared.masks.binary,
            torch.Generator(device=device).manual_seed(seed),
        )
        erased_context = build_masked_context(
            latent,
            prepared.masks.binary,
            torch.Generator(device=device).manual_seed(seed),
        )
        hole = prepared.masks.binary.bool().expand_as(latent)
        inside = (erased_context - official).abs()[hole].max().item()
        outside = (latent - prepared.target).abs()[~hole].max().item()
        if inside != 0.0:
            raise ValueError(f"latent fill inside H changed for {sample.image_relpath}")
        report.append(
            {
                "image_relpath": sample.image_relpath,
                "inside_context_absmax": float(inside),
                "outside_latent_absmax": float(outside),
            }
        )
        contexts[sample.sample_id] = latent
    if not any(row["outside_latent_absmax"] > 0.0 for row in report):
        raise ValueError("graying H left every outside latent unchanged")
    return contexts, report


def _erasure_steps(
    schedule: tuple[FewshotSample, ...],
    prepared_by_id: dict[str, PreparedSample],
    contexts: dict[str, torch.Tensor],
    config: Any,
):
    for step, sample in enumerate(schedule, start=1):
        original = prepared_by_id[sample.sample_id]
        context_seed = config.context_fill_seed + 20_000_000 + step
        context = build_masked_context(
            contexts[sample.sample_id],
            original.masks.binary,
            torch.Generator(device=original.target.device).manual_seed(context_seed),
        )
        prepared = PreparedSample(
            sample=original.sample,
            target=original.target,
            context=context,
            masks=original.masks,
        )
        yield step, sample, prepared, context_seed, _tensor_sha256(context)


def _write_erasure_visual(
    renderer: CanExactMaskRenderer,
    sample: FewshotSample,
    output: Path,
) -> str:
    rgb = (
        ((sample.pixels.float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    mask = sample.mask[0].numpy().astype(bool)
    raw = render_erased_context(renderer, np.ascontiguousarray(rgb), sample.defect_type, mask, 20260912)
    visual_root = output / "visuals"
    visual_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(visual_root / "fewshot_rgb.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(visual_root / "fewshot_mask.png")
    Image.fromarray(raw).save(visual_root / "fewshot_infill.png")
    return "visuals/fewshot_infill.png"


def train_pcb4_context_erasure(output: Path = TRAIN_OUTPUT, device: str | None = None) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite context-erasure training: {output}")
    inventory = load_visa_inventory_for_split(DATASET_ROOT, "pcb4", "magic", None)
    config = visa_p2_a1_config("pcb4", tuple(item.name for item in inventory.types))
    if config.defect_types != ANOMALY_NAMES or config.steps != 1200:
        raise ValueError("pcb4 magic inventory no longer matches the frozen baseline")
    split = load_visa_fewshot_split(MANIFEST, DATASET_ROOT, config)
    paths = load_config(PATHS_YAML)
    torch_device = torch.device(device or "cuda:0")
    output.mkdir(parents=True, exist_ok=False)
    try:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        components = load_frozen_components(paths.model_root, torch_device, torch.float16)
        prompt = encode_prompt(components, config.normal_prompt, torch_device).detach().clone()
        train_prepared = _prepare_samples(
            vae=components.vae,
            samples=split.train,
            device=torch_device,
            config=config,
            seed_offset=0,
        )
        rng_state = _capture_rng(torch_device)
        contexts, preflight = _encode_erased_contexts(train_prepared, components.vae, torch_device)
        _restore_rng(torch_device, rng_state)
        preflight_summary = {
            "rows": preflight,
            "gray_uint8": GRAY_UINT8,
            "canvas": "letterbox_512",
            "crop32": False,
            "d4": False,
            "target": "unerased_anomaly_latent",
            "latent_fill": "original_eta",
            "outside_latent_absmax": max(row["outside_latent_absmax"] for row in preflight),
        }
        write_json_atomic(output / "context_erasure_preflight.json", preflight_summary)
        print(json.dumps({"stage": "preflight", **{k: preflight_summary[k] for k in ("outside_latent_absmax", "gray_uint8", "canvas")}}, sort_keys=True), flush=True)
        prepared_by_id = {item.sample.sample_id: item for item in train_prepared}
        vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
        conditioner = DefectConditioner(vocabulary, 128).to(device=torch_device, dtype=torch.float32)
        adapter = FactorizedMaskedInfillAdapter(config).to(device=torch_device, dtype=torch.float32)
        model = P2InfillNoisePredictor(components.unet, conditioner, adapter).train()
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=config.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        schedule = balanced_training_order(split.train, config.steps, config.defect_types)
        generator = torch.Generator(device=torch_device).manual_seed(config.noise_seed + 10_000_000)
        training: list[dict[str, Any]] = []
        num_train_timesteps = int(components.scheduler.alphas_cumprod.numel())
        for step, sample, prepared, context_seed, context_sha256 in _erasure_steps(
            schedule, prepared_by_id, contexts, config
        ):
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (1,),
                generator=generator,
                device=torch_device,
                dtype=torch.long,
            )
            noise = torch.randn(
                prepared.target.shape,
                generator=generator,
                device=torch_device,
                dtype=prepared.target.dtype,
            )
            metrics = ranking_optimizer_step(
                model=model,
                unet=components.unet,
                scheduler=components.scheduler,
                optimizer=optimizer,
                scaler=scaler,
                target_latent=prepared.target,
                context=prepared.context,
                binary_mask=prepared.masks.binary,
                soft_mask=prepared.masks.soft,
                prompt_embeds=prompt,
                defect_type=sample.defect_type,
                timesteps=timesteps,
                noise=noise,
                config=config,
            )
            training.append(
                {
                    "step": step,
                    "sample_id": sample.sample_id,
                    "defect_type": sample.defect_type,
                    "timestep": int(timesteps.item()),
                    **asdict(metrics),
                    "context_seed": context_seed,
                    "context_sha256": context_sha256,
                }
            )
            if step == 1 or step % 100 == 0 or step == config.steps:
                progress = {
                    "stage": "pcb4_context_erasure",
                    "step": step,
                    "steps": config.steps,
                    "total": metrics.total,
                }
                print(json.dumps(progress, sort_keys=True), flush=True)
                write_json_atomic(output / "progress.json", progress)
        model.eval()
        extra = {
            "dataset": "VisA",
            "not_native_win": True,
            "ranking_negative": "cyclic_next_and_frozen_normal_unet",
            "split_id": inventory.split_id,
            "context_erasure": True,
            "context_construction": "uint8_128_on_letterbox_then_vae_then_latent_fill",
            "erasure_canvas": "letterbox_512",
            "not_crop32": True,
            "not_d4": True,
        }
        checkpoint = _save_checkpoint(
            output / "checkpoint",
            adapter=adapter,
            conditioner=conditioner,
            config=config,
            vocabulary_sha256=vocabulary.sha256,
            checkpoint_schema=CHECKPOINT_SCHEMA,
            extra_metadata=extra,
        )
        renderer = CanExactMaskRenderer(
            components=components,
            model=model,
            prompt=prompt,
            device=torch_device,
            allowed_types=config.defect_types,
        )
        visual = _write_erasure_visual(renderer, split.train[0], output)
        summary = {
            "schema_version": "v06_c_visa_pcb4_context_erasure_v1",
            "diagnostic_only": True,
            "not_official_p2_a1": True,
            "not_learned_h": True,
            "category": config.category,
            "defect_types": list(config.defect_types),
            "steps": config.steps,
            "fewshot_train": len(split.train),
            "open_dev_report": len(split.evaluation),
            "checkpoint": checkpoint,
            "visual": visual,
            "last_total": training[-1]["total"],
            "output": str(output),
            **extra,
        }
        write_json_atomic(output / "summary.json", summary)
        write_json_atomic(output / "training.json", {"records": training})
        return summary
    except Exception:
        if output.exists() and not (output / "checkpoint" / "metadata.json").is_file():
            shutil.rmtree(output)
        raise


def _rgb_tensor(image: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        .div(127.5)
        .sub(1.0)
    )


@torch.inference_mode()
def render_erased_context(
    renderer: CanExactMaskRenderer,
    normal: np.ndarray,
    defect_type: str,
    realization_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    if defect_type not in renderer.allowed_types:
        raise ValueError(f"renderer vocabulary does not include {defect_type}")
    if (
        not isinstance(normal, np.ndarray)
        or normal.shape != (renderer.image_size, renderer.image_size, 3)
        or normal.dtype != np.uint8
    ):
        raise ValueError("normal must be 512x512 uint8 RGB")
    mask = np.ascontiguousarray(realization_mask, dtype=bool)
    if mask.shape != (renderer.image_size, renderer.image_size) or not mask.any() or mask.all():
        raise ValueError("realization mask must be a non-empty, non-full 512 binary")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    erased = np.array(normal, copy=True)
    erased[mask] = GRAY_UINT8
    if not np.array_equal(erased[~mask], normal[~mask]):
        raise ValueError("generation erasure changed RGB outside H")
    normal_latent = encode_mode(renderer.components.vae, _rgb_tensor(normal, renderer.device, renderer._vae_dtype)).detach()
    clean_context = encode_mode(
        renderer.components.vae, _rgb_tensor(erased, renderer.device, renderer._vae_dtype)
    ).detach()
    source_mask = (
        torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0).to(device=renderer.device, dtype=torch.float32)
    )
    latent_mask = build_latent_masks(source_mask, tuple(normal_latent.shape[2:])).binary
    context = build_masked_context(
        clean_context,
        latent_mask,
        torch.Generator(device=renderer.device).manual_seed(seed + 10_000_000),
    )
    noise = torch.randn(
        normal_latent.shape,
        generator=torch.Generator(device=renderer.device).manual_seed(seed),
        device=renderer.device,
        dtype=normal_latent.dtype,
    )
    generated_latent = denoise_masked_infill(
        model=renderer.model,
        scheduler=DDIMScheduler.from_config(renderer._scheduler_config),
        normal_latent=normal_latent,
        context=context,
        binary_mask=latent_mask,
        prompt_embeds=renderer.prompt,
        defect_type=defect_type,
        noise=noise,
        steps=renderer.ddim_steps,
    )
    decoded = decode_latent(renderer.components.vae, generated_latent).float()
    raw = (
        decoded[0]
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .copy()
    )
    return np.ascontiguousarray(raw)


def _source_records() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (SOURCE_GENERATE / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [row for row in rows if not row["failed"]]
    counts = {name: sum(row["defect_type"] == name for row in kept) for name in ANOMALY_NAMES}
    if counts != {name: 500 for name in ANOMALY_NAMES} or len(kept) != 1500:
        raise ValueError(f"existing pcb4 masks are not 500 per type: {counts}")
    return kept


def _link(source: Path, destination: Path) -> None:
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source)


def generate_from_existing_masks(
    checkpoint: Path = TRAIN_OUTPUT / "checkpoint",
    output: Path = GENERATE_OUTPUT,
    device: str | None = None,
) -> dict[str, Any]:
    if not (checkpoint / "metadata.json").is_file():
        raise FileNotFoundError(checkpoint)
    records = _source_records()
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    inventory = load_visa_inventory_for_split(DATASET_ROOT, "pcb4", "magic", None)
    config = visa_p2_a1_config("pcb4", tuple(item.name for item in inventory.types))
    model_root = load_config(PATHS_YAML).model_root
    renderer = load_visa_renderer(
        checkpoint=checkpoint,
        model_root=model_root,
        config=config,
        device=device,
    )
    written: list[dict[str, Any]] = []
    for index, source in enumerate(records, start=1):
        job_id = str(source["job_id"])
        sample_root = output / "samples" / job_id
        record_path = sample_root / "record.json"
        image_path = output / "images" / f"{job_id}.png"
        if record_path.is_file() and image_path.is_file() and (sample_root / "final_y.png").is_file():
            written.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        normal_path = SOURCE_GENERATE / "samples" / job_id / "normal.png"
        mask_path = SOURCE_GENERATE / str(source["mask_relpath"])
        sample_mask = SOURCE_GENERATE / "samples" / job_id / "m_final.png"
        normal = np.asarray(Image.open(normal_path).convert("RGB"))
        mask_image = np.asarray(Image.open(mask_path).convert("L"))
        sample_mask_image = np.asarray(Image.open(sample_mask).convert("L"))
        if normal.shape != (512, 512, 3) or not np.array_equal(mask_image, sample_mask_image):
            raise ValueError(f"existing normal/mask pair changed: {job_id}")
        mask = mask_image >= 128
        raw = render_erased_context(renderer, normal, str(source["defect_type"]), mask, int(source["seed"]))
        final = compose_causal(normal, raw, mask)
        sample_root.mkdir(parents=True, exist_ok=True)
        _link(normal_path, sample_root / "normal.png")
        _link(SOURCE_GENERATE / "samples" / job_id / "proposal.png", sample_root / "proposal.png")
        _link(sample_mask, sample_root / "m_final.png")
        _link(mask_path, output / "masks" / f"{job_id}.png")
        Image.fromarray(final).save(sample_root / "final_y.png")
        Image.fromarray(final).save(image_path)
        record = {
            **source,
            "context_erasure": True,
            "normal_reference": "unerased",
            "reused_mask": str(mask_path),
            "image_relpath": f"images/{job_id}.png",
            "mask_relpath": f"masks/{job_id}.png",
        }
        write_json_atomic(record_path, record)
        written.append(record)
        if index == 1 or index % 50 == 0 or index == len(records):
            print(json.dumps({"stage": "generate", "done": index, "total": len(records), "job_id": job_id}), flush=True)
            write_json_atomic(output / "progress.json", {"done": len(written), "total": len(records)})
    if len(written) != 1500:
        raise ValueError(f"generation kept {len(written)} rows")
    (output / "synthetic_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in written), encoding="utf-8"
    )
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in written), encoding="utf-8"
    )
    summary = {
        "schema_version": "v06_c_visa_pcb4_context_erasure_generate_v1",
        "category": "pcb4",
        "kept_rows": len(written),
        "per_type_kept": {name: 500 for name in ANOMALY_NAMES},
        "source_generate": str(SOURCE_GENERATE),
        "h_method": "reused_20260927_magic_500_masks",
        "normal_reference": "unerased",
        "context": "uint8_128_then_vae_then_latent_fill",
        "ddim_steps": 50,
        "destination": str(output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def run_downstream() -> None:
    if not MVTEC_PATH.is_dir():
        raise FileNotFoundError(MVTEC_PATH)
    pack_visa_eval_generated(GENERATE_OUTPUT, LAYOUT_OUTPUT, "pcb4", include_slugs=ANOMALY_NAMES)
    DOWNSTREAM_OUTPUT.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(PACKAGE_ROOT / "eval_kit/evaluate.py"),
        "--class_name",
        "pcb4",
        "--anomaly_names",
        *ANOMALY_NAMES,
        "--eval_anomaly_names",
        *EVAL_ANOMALY_NAMES,
        "--metrics",
        "localization",
        "classification",
        "--generated_root",
        str(LAYOUT_OUTPUT),
        "--mvtec_path",
        str(MVTEC_PATH),
        "--image_subdir",
        "composed",
        "--mask_subdir",
        "mask_defect",
        "--loc_epochs",
        "200",
        "--cls_epochs",
        "30",
        "--img_size",
        "256",
        "--loc_bs",
        "16",
        "--loc_accum",
        "1",
        "--output_dir",
        str(DOWNSTREAM_OUTPUT),
    ]
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pcb4 letterbox context-erasure control")
    parser.add_argument("mode", choices=("train", "generate", "downstream", "all"))
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.mode in ("train", "all"):
        train_pcb4_context_erasure(device=args.device)
    if args.mode in ("generate", "all"):
        generate_from_existing_masks(device=args.device)
    if args.mode in ("downstream", "all"):
        run_downstream()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

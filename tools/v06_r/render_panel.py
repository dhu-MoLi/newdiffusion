"""Fixed 32-identity, single-candidate R comparison with anonymous final-y panels.

Run from the isolated worktree with PYTHONPATH=src:.; inference is explicit.
The --self-check path exercises real causal composition and layout on CPU.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
import random
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from cadiff.config import load_config
from cadiff.p2_a1.config import P2A1Config
from cadiff.p2_a0.model import P2InfillNoisePredictor
from cadiff.provenance import sha256_file
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.visa_exact_mask_r import CHECKPOINT_SCHEMA, load_visa_renderer
from cadiff.v06_c.can_exact_mask_r import load_can_r_modules
from cadiff.v06_r.runner import load_train_manifest, validate_training_config, write_new_json

ARMS = ("latent_control", "rgb", "rgb_edge")
BLIND_SEED = 2026101203
DEFAULT_ROOT = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs/v06_r/visa/chewinggum/20261012-strong-r")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("unsafe panel identity or type slug")
    return value

def checked_inputs(row):
    for kind in ("normal", "mask"):
        if sha256_file(Path(row[f"{kind}_path"])) != row[f"{kind}_sha256"]:
            raise ValueError(f"frozen {kind} changed for {row['identity']}")
    with Image.open(row["normal_path"]) as f:
        normal = np.array(f.convert("RGB"), dtype=np.uint8, copy=True)
    with Image.open(row["mask_path"]) as f:
        mask_pixels = np.array(f.convert("L"), copy=True)
    if normal.shape != (512, 512, 3) or mask_pixels.shape != (512, 512):
        raise ValueError("panel assets must already be frozen on the 512 canvas")
    if not np.isin(mask_pixels, [0, 255]).all():
        raise ValueError("H mask PNG must be binary")
    mask = mask_pixels == 255
    if mask.sum() != row["mask_pixels"] or mask.sum() < 64 or mask.all():
        raise ValueError("frozen H mask pixels/64px contract changed")
    return normal, np.ascontiguousarray(mask)

def crop_bbox(mask, margin=32):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("crop domain needs a nonempty frozen mask")
    return [max(0, int(xs.min())-margin), max(0, int(ys.min())-margin),
            min(mask.shape[1], int(xs.max())+1+margin), min(mask.shape[0], int(ys.max())+1+margin)]

def overlay(normal, mask, bbox):
    out = normal.copy()
    out[mask] = np.round(.6 * normal[mask] + .4 * np.array([0, 235, 255])).astype(np.uint8)
    image = Image.fromarray(out)
    ImageDraw.Draw(image).rectangle((bbox[0], bbox[1], bbox[2]-1, bbox[3]-1), outline=(255, 220, 0), width=2)
    return image

def font(size=18):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def image_tile(image, size):
    # Nearest interpolation preserves displayed tiny scratches without invented detail.
    return ImageOps.pad(image.convert("RGB"), (size, size), method=Image.Resampling.NEAREST,
                        color=(24, 28, 36), centering=(.5, .5))

def row_panel(row, normal, mask, finals, reference, *, size):
    bbox = row["crop_bbox"]
    columns = [Image.fromarray(normal), overlay(normal, mask, bbox)]
    names = ["Normal", "H mask / fixed ROI"]
    for letter in "ABC":
        if letter in finals:
            columns.append(Image.fromarray(finals[letter]))
            names.append(letter)
    columns.append(reference["image"])
    names.append("Train reference (unpaired)")
    gap, label, top = 12, 30, 36
    width = gap + len(columns) * (size + gap)
    panel = Image.new("RGB", (width, top + 2*(size+label+gap)), (18, 22, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((gap, 6), f"{row['identity']} | crop = H-mask bbox + 32 px", font=font(16), fill="white")
    for i, (name, image) in enumerate(zip(names, columns)):
        x = gap + i*(size+gap)
        draw.text((x, top), name, font=font(15), fill=(215, 225, 240))
        panel.paste(image_tile(image, size), (x, top+label))
        crop = reference["bbox"] if i == len(columns)-1 else bbox
        close = image.crop(tuple(crop))
        y = top+size+label+gap
        draw.text((x, y), "Fixed crop + 32 px", font=font(15), fill=(215, 225, 240))
        panel.paste(image_tile(close, size), (x, y+label))
    return panel

def metadata_config(checkpoint, arm, types):
    metadata = read_json(checkpoint / "metadata.json")
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA or metadata.get("completed_steps") != 1200:
        raise ValueError("panel requires the final 1200-step independent VisA R checkpoint")
    if metadata.get("experiment", {}).get("arm") != arm:
        raise ValueError("checkpoint arm differs from the declared arm")
    raw = metadata["config"]
    if set(raw) != set(P2A1Config.__dataclass_fields__):
        raise ValueError("checkpoint P2 config keys differ")
    converted = {**raw, "defect_types": tuple(raw["defect_types"]), "eval_timesteps": tuple(raw["eval_timesteps"])}
    config = P2A1Config(**converted)
    validate_training_config(config, smoke=False, forced_timestep=None)
    if config.category != "chewinggum" or config.defect_types != tuple(types):
        raise ValueError("panel checkpoint vocabulary differs from frozen inputs")
    for name in ("adapter.safetensors", "conditioner.safetensors"):
        if sha256_file(checkpoint / name) != metadata["files"][name]["sha256"]:
            raise ValueError("checkpoint weights changed")
    return config, metadata

def preflight(root, checkpoints):
    inputs, manifest = read_json(root / "inputs.json"), read_json(root / "panel_manifest.json")
    if inputs.get("schema_version") != "v06_r_frozen_inputs_v1" or manifest.get("schema_version") != "v06_r_fixed_panel_v1":
        raise ValueError("unknown frozen panel/input schema")
    if sha256_file(Path(inputs["training_manifest"])) != inputs["training_manifest_sha256"]:
        raise ValueError("frozen training manifest changed")
    worktree = Path(inputs["worktree"])
    for relative, digest in inputs["formal_a_h_and_composition_sha256"].items():
        if sha256_file(worktree / relative) != digest:
            raise ValueError(f"formal A/H/composition changed: {relative}")
    rows = manifest["rows"]
    if len(rows) != 32 or len({_safe_id(row["identity"]) for row in rows}) != 32:
        raise ValueError("panel must contain exactly 32 distinct frozen identities")
    types = inputs["defect_types"]
    if len(types) != 4 or "combined" in types or Counter(r["defect_type"] for r in rows) != Counter({t: 8 for t in types}):
        raise ValueError("panel must cover four single-label types, eight identities each")
    for row in rows:
        _safe_id(row["type_slug"])
        if row.get("uses_test_mask") is not False or row.get("normal_role") != "amazon_1cls_train":
            raise ValueError("panel must use frozen training normals and H masks")
        if row["renderer_seed"] not in (2026101201, 2026101202) or row["normal_id"] not in ("000", "001", "002", "003"):
            raise ValueError("panel identity schedule changed")
        _, mask = checked_inputs(row)
        row["crop_bbox"] = crop_bbox(mask)
    configs, metadata = {}, {}
    for arm, checkpoint in checkpoints.items():
        configs[arm], metadata[arm] = metadata_config(checkpoint, arm, types)
    first = next(iter(configs.values()))
    if any(asdict(c) != asdict(first) for c in configs.values()):
        raise ValueError("comparison arms must have identical P2 config, budget and seed")
    return inputs, manifest, configs, metadata

def training_references(inputs, config):
    samples, _, records = load_train_manifest(Path(inputs["training_manifest"]), Path(inputs["dataset_root"]), config)
    if len(samples) != 18:
        raise ValueError("training reference population must stay the frozen 18 shots")
    refs = {}
    for label in config.defect_types:
        sample = min((s for s in samples if s.defect_type == label), key=lambda s: s.image_relpath)
        rgb = ((sample.pixels+1)*127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
        refs[label] = {"image": Image.fromarray(rgb), "bbox": crop_bbox(sample.mask[0].numpy().astype(bool)),
            "image_relpath": sample.image_relpath, "mask_relpath": sample.mask_relpath,
            "image_sha256": sample.image_sha256, "mask_sha256": sample.mask_sha256,
            "role": "fewshot_train", "selection": "first image_relpath before rendering"}
    return refs

def generate_panel(*, root, output, paths_yaml, checkpoints, device="cuda:0"):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite panel: {output}")
    inputs, manifest, configs, metadata = preflight(root, checkpoints)
    frozen_manifest_sha256 = sha256_file(root / "panel_manifest.json")
    frozen_inputs_sha256 = sha256_file(root / "inputs.json")
    refs = training_references(inputs, next(iter(configs.values())))
    paths = load_config(paths_yaml)
    shuffled = list(ARMS)
    random.Random(BLIND_SEED).shuffle(shuffled)
    blind = dict(zip("ABC", shuffled))
    inverse = {arm: letter for letter, arm in blind.items()}
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir()
    (private / "raw").mkdir()
    images = output / "blind" / "images"
    images.mkdir(parents=True)
    write_new_json(private / "blind_key.json", {"seed": BLIND_SEED, "mapping": blind})
    write_new_json(private / "checkpoint_manifest.json", {arm: {"path": str(checkpoints[arm]),
        "metadata_sha256": sha256_file(checkpoints[arm] / "metadata.json"), "metadata": metadata[arm]} for arm in checkpoints})
    # Crop boxes and identities are frozen before the first inference call.
    write_new_json(output / "identities.json", {"manifest_sha256": sha256_file(root / "panel_manifest.json"),
        "rows": [{k: r[k] for k in ("identity", "defect_type", "normal_id", "renderer_seed", "normal_sha256", "mask_sha256", "crop_bbox")} for r in manifest["rows"]]})
    reference_dir = output / "references"
    reference_dir.mkdir()
    for label, reference in refs.items():
        slug = next(r["type_slug"] for r in manifest["rows"] if r["defect_type"] == label)
        reference["image"].save(reference_dir / f"{slug}-train-first.png")
    write_new_json(reference_dir / "references.json", {label: {k: v for k, v in r.items() if k != "image"} for label, r in refs.items()})
    renderer = None
    contracts = []
    for arm in ARMS:
        if arm not in checkpoints:
            continue
        letter = inverse[arm]
        (images / letter).mkdir()
        (private / "raw" / arm).mkdir()
        if renderer is None:
            renderer = load_visa_renderer(checkpoint=checkpoints[arm], model_root=paths.model_root, config=configs[arm], device=device)
        else:
            # Frozen SD/VAE/text components and prompt are shared across arms.
            _, conditioner, adapter, _ = load_can_r_modules(checkpoints[arm], config=configs[arm], device=torch.device(device), expected_schema=CHECKPOINT_SCHEMA)
            renderer.model = P2InfillNoisePredictor(renderer.components.unet, conditioner, adapter).eval()
            del conditioner, adapter
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        if renderer.ddim_steps != 50:
            raise ValueError("panel requires full 50-step DDIM")
        for index, row in enumerate(manifest["rows"]):
            normal, mask = checked_inputs(row)
            raw = renderer(normal.copy(), row["defect_type"], mask.copy(), row["renderer_seed"])
            if raw.shape != normal.shape or raw.dtype != np.uint8:
                raise ValueError("renderer must return uint8 RGB on frozen canvas")
            final = compose_causal(normal, raw, mask)
            exact = np.array_equal(final[~mask], normal[~mask])
            inside = np.array_equal(final[mask], raw[mask])
            # Rehash original assets after the call to detect accidental mutations.
            checked_inputs(row)
            if not exact or not inside:
                raise RuntimeError("causal composition contract failed")
            final_path = images / letter / f"{row['identity']}.png"
            raw_path = private / "raw" / arm / f"{row['identity']}.png"
            Image.fromarray(final).save(final_path)
            Image.fromarray(raw).save(raw_path)
            contract = {"identity": row["identity"], "arm": arm, "checkpoint": str(checkpoints[arm]),
                "checkpoint_metadata_sha256": sha256_file(checkpoints[arm] / "metadata.json"),
                "normal_sha256": row["normal_sha256"], "mask_sha256": row["mask_sha256"],
                "normal_path": row["normal_path"], "mask_path": row["mask_path"], "crop_bbox": row["crop_bbox"],
                "h_job_id": row["h_job_id"], "h_source_id": row["h_source_id"], "h_source_run": row["h_source_run"],
                "normal_role": row["normal_role"], "uses_test_mask": False,
                "mask_pixels": row["mask_pixels"], "renderer_seed": row["renderer_seed"],
                "candidate_count": 1, "ddim_steps": 50, "outside_rgb_byte_exact": exact,
                "inside_copies_raw": inside, "frozen_mask_unchanged": True,
                "final_path": str(final_path.relative_to(output)), "final_sha256": sha256_file(final_path),
                "raw_path": str(raw_path.relative_to(output)), "raw_sha256": sha256_file(raw_path)}
            contracts.append(contract)
            with (private / "contracts.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(contract, ensure_ascii=False) + "\n")
            print(json.dumps({"stage": "fixed_panel_inference", "completed": len(contracts),
                "total": len(checkpoints)*32, "identity": row["identity"]}), flush=True)
    del renderer
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    if sha256_file(root / "panel_manifest.json") != frozen_manifest_sha256 or sha256_file(root / "inputs.json") != frozen_inputs_sha256:
        raise RuntimeError("frozen panel/input manifest changed during rendering")
    panels = []
    if len(checkpoints) == 3:
        row_dir = output / "blind" / "rows"
        row_dir.mkdir()
        for row in manifest["rows"]:
            normal, mask = checked_inputs(row)
            finals = {letter: np.array(Image.open(images / letter / f"{row['identity']}.png").convert("RGB")) for letter in "ABC"}
            big = row_panel(row, normal, mask, finals, refs[row["defect_type"]], size=384)
            big.save(row_dir / f"{row['identity']}.png")
        for label in configs[next(iter(configs))].defect_types:
            rows = [r for r in manifest["rows"] if r["defect_type"] == label]
            pieces = []
            for row in rows:
                normal, mask = checked_inputs(row)
                finals = {letter: np.array(Image.open(images / letter / f"{row['identity']}.png").convert("RGB")) for letter in "ABC"}
                pieces.append(row_panel(row, normal, mask, finals, refs[label], size=256))
            sheet = Image.new("RGB", (pieces[0].width, sum(p.height for p in pieces)), (18, 22, 30))
            y = 0
            for piece in pieces:
                sheet.paste(piece, (0, y))
                y += piece.height
            path = output / "blind" / f"{rows[0]['type_slug']}-8rows.png"
            sheet.save(path)
            panels.append(str(path.relative_to(output)))
    public = {"schema_version": "v06_r_fixed_panel_results_v1", "identity_count": 32,
        "candidate_count_per_identity_per_arm": 1, "generated_count": len(contracts), "ddim_steps": 50,
        "outside_rgb_byte_exact": all(c["outside_rgb_byte_exact"] for c in contracts),
        "mask_sha256_matches_frozen_manifest": all(c["frozen_mask_unchanged"] for c in contracts),
        "comparison_complete": len(checkpoints) == 3, "panels": panels,
        "row_panels": "blind/rows/" if len(checkpoints) == 3 else None,
        "reference_selection": "first fewshot_train image_relpath per type; 18 shots only; unpaired",
        "crop_rule": "same frozen H-mask bounding box plus 32 px on every side, clipped to canvas",
        "visible_defects_passed": None, "human_review": "pending", "output": str(output)}
    write_new_json(output / "summary.json", public)
    return public

def cpu_self_check():
    normal = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=bool)
    mask[240:256, 248:264] = True
    raw = np.full_like(normal, 151)
    final = compose_causal(normal, raw, mask)
    assert np.array_equal(final[~mask], normal[~mask]) and np.all(final[mask] == 151)
    bbox = crop_bbox(mask)
    assert bbox == [216, 208, 296, 288]
    row = {"identity": "cpu-check-fixed", "crop_bbox": bbox}
    reference = {"image": Image.fromarray(final), "bbox": bbox}
    panel = row_panel(row, normal, mask, {l: final for l in "ABC"}, reference, size=256)
    assert panel.size == (1620, 632)
    # The crop center in all three anonymous columns contains actual final RGB.
    for column in (2, 3, 4):
        assert panel.getpixel((12 + column*268 + 128, 36+256+30+12+30+128)) == (151, 151, 151)
    assert normal.max() == 0 and mask.sum() == 256
    return {"cpu_self_check": "passed", "composition": "exact", "crop": "bbox_plus_32", "layout_size": panel.size}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--paths-yaml", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--checkpoints", help="JSON object text or JSON file: arm -> final checkpoint directory")
    p.add_argument("--arm", choices=ARMS, help="Run exactly 32 outputs for this arm; no three-arm board")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        print(json.dumps(cpu_self_check()))
        return
    if args.paths_yaml is None or args.output is None:
        p.error("--paths-yaml and --output are required for inference")
    if args.checkpoints:
        raw = json.loads(args.checkpoints) if args.checkpoints.lstrip().startswith("{") else read_json(args.checkpoints)
        if not isinstance(raw, dict) or not raw or set(raw) - set(ARMS):
            p.error("--checkpoints must map known arm names to checkpoint paths")
        checkpoints = {arm: Path(path) for arm, path in raw.items()}
    else:
        checkpoints = {arm: args.root / "training" / arm / "checkpoint-step-001200" for arm in ARMS}
    if args.arm:
        if args.arm not in checkpoints:
            p.error("requested --arm has no checkpoint")
        checkpoints = {args.arm: checkpoints[args.arm]}
    elif set(checkpoints) != set(ARMS):
        p.error("default comparison requires all three arms; use --arm for one")
    result = generate_panel(root=args.root, output=args.output, paths_yaml=args.paths_yaml,
                            checkpoints=checkpoints, device=args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()

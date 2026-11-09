"""Generate pcb4 defects on AnomalyDiffusion masks with the official exact-mask R.

Normals are drawn with replacement from Amazon 1cls train/good only. Test/good
is left untouched. Empty masks are recorded and skipped. Outside the mask, the
sampled normal is restored pixel for pixel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from cadiff.config import load_config
from cadiff.data.geometry import resize_pad_image
from cadiff.provenance import write_json_atomic
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.visa_exact_mask_r import load_visa_renderer, visa_p2_a1_config
from cadiff.v06_c.visa_pilot import load_visa_inventory_for_split
from cadiff.v06_c.visa_place_bulk import pack_visa_eval_generated

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MASK_ROOT = Path(
    "/home/dell/2T/lm/new_projects/category_anomaly_diffusion/data/anomalydiffusion/export_pcb4_anomalydiffusion_masks"
)
DATASET_ROOT = Path("/home/dell/2T/数据集/AD/VisA")
PATHS_YAML = PACKAGE_ROOT / "configs/paths/omen-home.yaml"
CHECKPOINT = (
    PACKAGE_ROOT
    / "checkpoints/v06_b/visa/pcb4/exact-mask-r/seed-20260927-magic/checkpoint"
)
RUNS = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/runs")
EXPERIMENT = RUNS / "v06_c/visa/pcb4/ad_mask_official_r_20261103"
GENERATE_OUTPUT = EXPERIMENT / "generate"
LAYOUT_OUTPUT = EXPERIMENT / "generated_layout"
DOWNSTREAM_OUTPUT = EXPERIMENT / "downstream"
MVTEC_PATH = RUNS / "v06_c/visa/pcb4/mvtec_layout_magic_20260927"
PYTHON = Path(
    "/home/dell/2T/lm/new_projects/category_anomaly_diffusion/.conda/envs/cad-v0/bin/python"
)
ANOMALY_NAMES = ("damage", "extra", "missing")
EVAL_ANOMALY_NAMES = ("burnt", "damage", "extra", "missing", "scratch", "wrong_place")
NORMAL_SAMPLE_SEED = 20261103
RENDER_SEED_BASE = 202611030000


def _letterbox(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        canvas, _, _ = resize_pad_image(image, 512)
    return np.asarray(canvas, dtype=np.uint8).copy()


def build_assignments() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (MASK_ROOT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    inventory = load_visa_inventory_for_split(DATASET_ROOT, "pcb4", "magic", None)
    normals = inventory.normals
    if len(normals) != 904:
        raise ValueError(f"pcb4 Amazon train/good count changed: {len(normals)}")
    draw = np.random.Generator(np.random.PCG64(NORMAL_SAMPLE_SEED)).integers(0, len(normals), size=len(rows))
    assignments = []
    for index, row in enumerate(rows):
        mask_path = MASK_ROOT / row["mask_relpath"]
        mask = np.asarray(Image.open(mask_path).convert("L"))
        hole = mask >= 128
        normal_path = normals[int(draw[index])]
        assignments.append(
            {
                "index": index,
                "layout_index": int(row["layout_index"]),
                "defect_type": row["defect_type"],
                "type_slug": row["type_slug"],
                "job_id": f"{row['type_slug']}-{int(row['layout_index']):04d}",
                "mask_src": str(mask_path),
                "normal_src": str(normal_path),
                "normal_name": normal_path.name,
                "placed_pixels": int(hole.sum()),
                "empty": not bool(hole.any()),
                "full": bool(hole.all()),
                "render_seed": RENDER_SEED_BASE + index,
            }
        )
    return assignments


def generate(device: str | None = None) -> dict:
    GENERATE_OUTPUT.mkdir(parents=True, exist_ok=True)
    assignment_path = EXPERIMENT / "assignments.json"
    if assignment_path.is_file():
        assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
    else:
        assignments = build_assignments()
        EXPERIMENT.mkdir(parents=True, exist_ok=True)
        write_json_atomic(assignment_path, assignments)
    inventory = load_visa_inventory_for_split(DATASET_ROOT, "pcb4", "magic", None)
    config = visa_p2_a1_config("pcb4", tuple(item.name for item in inventory.types))
    renderer = load_visa_renderer(
        checkpoint=CHECKPOINT,
        model_root=load_config(PATHS_YAML).model_root,
        config=config,
        device=device,
    )
    cache: dict[str, np.ndarray] = {}
    written: list[dict] = []
    (GENERATE_OUTPUT / "images").mkdir(exist_ok=True)
    (GENERATE_OUTPUT / "masks").mkdir(exist_ok=True)
    for number, job in enumerate(assignments, start=1):
        image_path = GENERATE_OUTPUT / "images" / f"{job['job_id']}.png"
        record_path = GENERATE_OUTPUT / "samples" / job["job_id"] / "record.json"
        if record_path.is_file() and (job["empty"] or job["full"] or image_path.is_file()):
            written.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        mask = np.asarray(Image.open(job["mask_src"]).convert("L")) >= 128
        record = {
            "schema_version": "pcb4_ad_mask_official_r_v1",
            "job_id": job["job_id"],
            "defect_type": job["defect_type"],
            "type_slug": job["type_slug"],
            "layout_index": job["layout_index"],
            "normal_src": job["normal_src"],
            "normal_name": job["normal_name"],
            "mask_src": job["mask_src"],
            "placed_pixels": job["placed_pixels"],
            "render_seed": job["render_seed"],
            "image_relpath": f"images/{job['job_id']}.png",
            "mask_relpath": f"masks/{job['job_id']}.png",
            "failed": False,
            "failure_reason": None,
        }
        sample_root = GENERATE_OUTPUT / "samples" / job["job_id"]
        sample_root.mkdir(parents=True, exist_ok=True)
        if job["empty"] or job["full"] or not mask.any() or mask.all():
            record.update(failed=True, failure_reason="mask is empty or full")
        else:
            normal = cache.get(job["normal_src"])
            if normal is None:
                normal = _letterbox(Path(job["normal_src"]))
                cache[job["normal_src"]] = normal
            raw = renderer(normal, job["defect_type"], mask, int(job["render_seed"]))
            final = compose_causal(normal, raw, mask)
            Image.fromarray(normal).save(sample_root / "normal.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(sample_root / "m_final.png")
            Image.fromarray(final).save(sample_root / "final_y.png")
            Image.fromarray(final).save(image_path)
            mask_link = GENERATE_OUTPUT / "masks" / f"{job['job_id']}.png"
            if mask_link.exists() or mask_link.is_symlink():
                mask_link.unlink()
            mask_link.symlink_to(job["mask_src"])
        write_json_atomic(record_path, record)
        written.append(record)
        if number == 1 or number % 50 == 0 or number == len(assignments):
            failed = sum(row["failed"] for row in written)
            print(
                json.dumps({"stage": "generate", "done": number, "total": len(assignments), "failed": failed}),
                flush=True,
            )
    kept = [row for row in written if not row["failed"]]
    (GENERATE_OUTPUT / "synthetic_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8"
    )
    (GENERATE_OUTPUT / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in written), encoding="utf-8"
    )
    per_type = {name: sum(row["defect_type"] == name for row in kept) for name in ANOMALY_NAMES}
    summary = {
        "schema_version": "pcb4_ad_mask_official_r_generate_v1",
        "checkpoint": str(CHECKPOINT),
        "normal_pool": "amazon_1cls_train_good",
        "normal_sample_seed": NORMAL_SAMPLE_SEED,
        "normal_draw": "with_replacement",
        "render_seed_base": RENDER_SEED_BASE,
        "requested": len(assignments),
        "kept_rows": len(kept),
        "failed_rows": len(written) - len(kept),
        "per_type_kept": per_type,
        "destination": str(GENERATE_OUTPUT),
    }
    write_json_atomic(GENERATE_OUTPUT / "summary.json", summary)
    return summary


def run_downstream() -> None:
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
    parser = argparse.ArgumentParser(description="pcb4 official R on AnomalyDiffusion masks")
    parser.add_argument("mode", choices=("generate", "downstream", "all"))
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.mode in ("generate", "all"):
        generate(device=args.device)
    if args.mode in ("downstream", "all"):
        run_downstream()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

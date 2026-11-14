#!/usr/bin/env python3
"""Run a v07 experiment end to end for one or more categories.

Stages: train -> generate -> pack -> eval. Every stage is skipped when its
output already exists, so a run can be resumed after an interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.provenance import sha256_file
from cadiff.v06_c.mvtec_pilot import load_mvtec_inventory, write_mvtec_manifest
from cadiff.v06_c.visa_pilot import load_visa_inventory_for_split
from cadiff.v06_c.visa_place_bulk import pack_visa_eval_generated
from cadiff.v07.backbones import make_backbone_spec
from cadiff.v07.generate import (
    build_location_prior,
    build_shape_prior,
    run_visa_place_bulk_v07,
)
from cadiff.v07.runmeta import collect_run_meta, write_json
from cadiff.v07.train_r import train_exact_mask_r_v07

DEFAULTS: dict[str, Any] = {
    "name": "v07",
    "paths": "configs/paths/asus-mvtec.yaml",
    "dataset": "mvtec",
    "split": "mvtec",
    "backbone": "sd15",
    "sd2_inpaint_root": None,
    "manifest": None,
    "visibility": "none",
    "visibility_weight": 0.05,
    "visibility_factor": 2,
    "visibility_every": 50,
    "visibility_size": 256,
    "shape_prior": "none",
    "site_method": "baseline",
    "fill_backend": "neighbor",
    "learned_checkpoint": None,
    "location_prior": "none",
    "learned_prior_root": None,
    "learned_prior_weight": 0.5,
    "shape_elastic_probability": 0.5,
    "shape_multi_component_probability": 0.0,
    "shape_max_components": 3,
    "pairs_per_type": 200,
    "loc_epochs": 50,
    "cls_epochs": 0,
    "img_size": 256,
    "loc_bs": 16,
    "steps": None,
    "seed": 20261109,
    "output_root": None,
}


def _log(handle, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def _load_experiment(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a YAML mapping")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
    return {**DEFAULTS, **raw}


def _ensure_mvtec_manifest(dataset_root: Path, category: str) -> Path:
    manifest = ROOT / "splits" / "mvtec" / category / "manifest.jsonl"
    if manifest.is_file():
        return manifest
    inventory = load_mvtec_inventory(dataset_root, category)
    if not manifest.parent.exists():
        write_mvtec_manifest(inventory, manifest)
        return manifest
    # Parent exists but the file does not: write the same payload ourselves.
    lines: list[str] = []
    for item in inventory.types:
        for role, group in (
            ("fewshot_train", item.fewshot),
            ("open_dev_report", item.heldout),
        ):
            for row in group:
                image = inventory.source_root.joinpath(*Path(row["image"]).parts)
                mask = inventory.source_root.joinpath(*Path(row["mask"]).parts)
                payload = {
                    "category": inventory.category,
                    "dataset": "MVTec",
                    "defect_type": item.name,
                    "image_relpath": row["image"],
                    "image_sha256": sha256_file(image),
                    "mask_relpath": row["mask"],
                    "mask_sha256": sha256_file(mask),
                    "roles": [role],
                    "split_id": inventory.split_id,
                }
                lines.append(json.dumps(payload, sort_keys=True))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _manifest_for(experiment: dict[str, Any], dataset_root: Path, category: str) -> Path:
    template = experiment.get("manifest")
    if template:
        return (ROOT / str(template).format(category=category)).resolve()
    if experiment["dataset"] == "mvtec":
        return _ensure_mvtec_manifest(dataset_root, category)
    raise ValueError(
        "non-MVTec experiments must set 'manifest' to a jsonl template containing {category}"
    )


def _latest_results(folder: Path, category: str) -> Path | None:
    target = folder / category
    files = sorted(target.glob("results_*.json")) if target.is_dir() else []
    return files[-1] if files else None


def _run_stage(
    *,
    experiment: dict[str, Any],
    category: str,
    run_dir: Path,
    project: Any,
    backbone: Any,
    seed: int,
    device: str,
    stage: str,
    log_handle: Any,
) -> dict[str, Any]:
    dataset_root = Path(project.dataset_root)
    split = str(experiment["split"])
    train_out = run_dir / "train"
    generate_out = run_dir / "generate"
    layout = run_dir / "generated_layout"
    eval_dir = run_dir / "eval"

    if stage in {"all", "train"}:
        if (train_out / "summary.json").is_file():
            _log(log_handle, f"{category}: train exists, skip")
        else:
            manifest = _manifest_for(experiment, dataset_root, category)
            _log(log_handle, f"{category}: train (manifest={manifest})")
            summary = train_exact_mask_r_v07(
                paths_yaml=ROOT / str(experiment["paths"]),
                manifest=manifest,
                dataset_root=dataset_root,
                output=train_out,
                category=category,
                split=split,
                backbone=backbone,
                device=device,
                visibility=str(experiment["visibility"]),
                visibility_weight=float(experiment["visibility_weight"]),
                visibility_factor=int(experiment["visibility_factor"]),
                visibility_every=int(experiment["visibility_every"]),
                visibility_size=int(experiment["visibility_size"]),
                steps=experiment["steps"],
                seed=seed,
            )
            write_json(run_dir / "train_summary.json", summary)

    if stage in {"all", "generate"}:
        if (generate_out / "summary.json").is_file():
            _log(log_handle, f"{category}: generate exists, skip")
        else:
            inventory = load_visa_inventory_for_split(dataset_root, category, split, None)
            shape_prior = None
            if str(experiment["shape_prior"]) == "parametric":
                shape_prior = build_shape_prior(
                    inventory,
                    elastic_probability=float(experiment["shape_elastic_probability"]),
                    multi_component_probability=float(
                        experiment["shape_multi_component_probability"]
                    ),
                    max_components=int(experiment["shape_max_components"]),
                    rng_seed=seed,
                )
                write_json(
                    run_dir / "shape_prior.json",
                    {
                        defect_type: shape_prior.describe(defect_type)
                        for defect_type in shape_prior.stats_by_type
                    },
                )
            a2_fill_backend = None
            fill_backend_name = str(experiment["fill_backend"])
            if fill_backend_name != "neighbor":
                from cadiff.v07.reconstruction import make_reconstructor

                learned_checkpoint = experiment["learned_checkpoint"]
                a2_fill_backend = make_reconstructor(
                    fill_backend_name,
                    learned_checkpoint=(
                        None
                        if learned_checkpoint in (None, "", "null")
                        else Path(str(learned_checkpoint))
                    ),
                    device=device or "cuda:0",
                )
            location_prior = None
            if str(experiment["location_prior"]) == "histogram":
                location_prior = build_location_prior(inventory, device=device)
                write_json(run_dir / "location_prior.json", location_prior.describe())
            if str(experiment["site_method"]) == "learned_prior":
                from cadiff.v06_a.tokencut import load_official_dinov2_encoder
                from cadiff.v07.placement_scorer import (
                    LearnedPlacementPrior,
                    PlacementScorer,
                )

                prior_root = experiment["learned_prior_root"]
                root = (
                    Path(str(prior_root))
                    if prior_root
                    else Path(project.runs_root) / "v07" / "placement_scorer"
                )
                model_path = root / category / "model.pkl"
                if not model_path.is_file():
                    raise FileNotFoundError(f"missing L2 scorer model: {model_path}")
                scorer = PlacementScorer.load(model_path)
                location_prior = LearnedPlacementPrior(
                    scorer,
                    load_official_dinov2_encoder(device=device),
                    defect_types=tuple(sorted(item.name for item in inventory.types)),
                    histogram=location_prior,
                    learned_weight=float(experiment["learned_prior_weight"]),
                )
                write_json(
                    run_dir / "learned_prior.json",
                    {
                        "model": str(model_path),
                        "learned_weight": float(experiment["learned_prior_weight"]),
                        "fused_with": "histogram" if location_prior.histogram else None,
                    },
                )
            _log(log_handle, f"{category}: generate")
            summary = run_visa_place_bulk_v07(
                inventory=inventory,
                checkpoint=train_out / "checkpoint",
                model_root=Path(project.model_root),
                output=generate_out,
                device=device,
                pairs_per_type=int(experiment["pairs_per_type"]),
                shape_prior=shape_prior,
                backbone=backbone,
                site_method=str(experiment["site_method"]),
                location_prior=location_prior,
                a2_fill_backend=a2_fill_backend,
            )
            write_json(run_dir / "generate_summary.json", summary)

    if stage in {"all", "pack"}:
        if (layout / category).is_dir():
            _log(log_handle, f"{category}: pack exists, skip")
        else:
            inventory = load_visa_inventory_for_split(dataset_root, category, split, None)
            _log(log_handle, f"{category}: pack")
            pack_visa_eval_generated(
                generate_out,
                layout,
                category,
                include_slugs=tuple(item.slug for item in inventory.types),
            )

    if stage in {"all", "eval"}:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            _log(log_handle, f"{category}: eval metrics exist, skip")
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        inventory = load_visa_inventory_for_split(dataset_root, category, split, None)
        names = [item.slug for item in inventory.types]
        metric_names = ["localization"]
        if int(experiment["cls_epochs"]) > 0:
            metric_names.append("classification")
        command = [
            sys.executable,
            str(ROOT / "eval_kit" / "evaluate.py"),
            "--class_name",
            category,
            "--anomaly_names",
            *names,
            "--eval_anomaly_names",
            *names,
            "--metrics",
            *metric_names,
            "--generated_root",
            str(layout),
            "--mvtec_path",
            str(dataset_root),
            "--image_subdir",
            "composed",
            "--mask_subdir",
            "mask_defect",
            "--loc_epochs",
            str(int(experiment["loc_epochs"])),
            "--cls_epochs",
            str(int(experiment["cls_epochs"])),
            "--img_size",
            str(int(experiment["img_size"])),
            "--loc_bs",
            str(int(experiment["loc_bs"])),
            "--loc_accum",
            "1",
            "--seed",
            str(int(experiment["seed"])),
            "--output_dir",
            str(eval_dir),
        ]
        _log(log_handle, f"{category}: eval")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SRC)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with (run_dir / "eval.log").open("a", encoding="utf-8") as eval_log:
            eval_log.write("\n$ " + " ".join(command) + "\n")
            eval_log.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{category}: eval failed with {completed.returncode}")
        results_path = _latest_results(eval_dir, category)
        if results_path is None:
            raise RuntimeError(f"{category}: eval wrote no results")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        metrics = {
            key: payload[key]
            for key in payload
            if key.startswith("loc_") or key == "cls_accuracy"
        }
        record = {
            "schema": "v07_metrics_v1",
            "experiment": experiment["name"],
            "category": category,
            "seed": seed,
            "backbone": backbone.name,
            "visibility": experiment["visibility"],
            "shape_prior": experiment["shape_prior"],
            "site_method": experiment["site_method"],
            "fill_backend": experiment["fill_backend"],
            "location_prior": experiment["location_prior"],
            "pairs_per_type": int(experiment["pairs_per_type"]),
            "loc_epochs": int(experiment["loc_epochs"]),
            "results": str(results_path),
            **metrics,
        }
        write_json(metrics_path, record)
        _log(log_handle, f"{category}: metrics {json.dumps(metrics, sort_keys=True)}")
        return record
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument(
        "--stage",
        choices=["all", "train", "generate", "pack", "eval"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pairs-per-type", type=int, default=None)
    parser.add_argument("--loc-epochs", type=int, default=None)
    arguments = parser.parse_args(argv)

    config_path = arguments.config if arguments.config.is_absolute() else ROOT / arguments.config
    experiment = _load_experiment(config_path)
    project = load_config(ROOT / str(experiment["paths"]))
    seed = int(arguments.seed if arguments.seed is not None else experiment["seed"])
    if arguments.pairs_per_type is not None:
        experiment["pairs_per_type"] = int(arguments.pairs_per_type)
    if arguments.loc_epochs is not None:
        experiment["loc_epochs"] = int(arguments.loc_epochs)
    categories = arguments.categories or ["bottle", "carpet", "grid", "transistor"]
    backbone = make_backbone_spec(
        str(experiment["backbone"]),
        sd15_root=Path(project.model_root),
        sd2_inpaint_root=(
            Path(str(experiment["sd2_inpaint_root"]))
            if experiment["sd2_inpaint_root"]
            else None
        ),
    )
    output_root = experiment.get("output_root")
    if output_root:
        root = Path(str(output_root))
        if not root.is_absolute():
            root = ROOT / root
    else:
        root = Path(project.runs_root) / "v07" / str(experiment["name"])
    root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for category in categories:
        run_dir = root / category / f"seed-{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "run_meta.json"
        if not meta_path.is_file():
            write_json(
                meta_path,
                collect_run_meta(
                    repo_root=ROOT,
                    config_path=config_path,
                    seed=seed,
                    extra={
                        "experiment": experiment["name"],
                        "category": category,
                        "backbone": backbone.name,
                        "device": arguments.device,
                    },
                ),
            )
        with (run_dir / "run.log").open("a", encoding="utf-8") as log_handle:
            try:
                _run_stage(
                    experiment=experiment,
                    category=category,
                    run_dir=run_dir,
                    project=project,
                    backbone=backbone,
                    seed=seed,
                    device=arguments.device,
                    stage=arguments.stage,
                    log_handle=log_handle,
                )
            except Exception as error:
                failures.append(f"{category}: {error}")
                _log(log_handle, f"{category}: FAIL {error}")
    if failures:
        raise SystemExit("\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

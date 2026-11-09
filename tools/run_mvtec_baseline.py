"""Run the VisA exact-mask R baseline on MVTec AD.

Protocol matches the packaged VisA run: floor(n/3) few-shot, 1200 R steps,
500 generated pairs per defect type, then DualAnoDiff localization for 200
epochs and classification for 30 epochs at 256. Real test images are the
MVTec tree itself. A finished stage is skipped when its summary or results
file is already present.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(
    "/mnt/C4400D13400D0DBC/lm/code/category_anomaly_diffusion/runs/mvtec_baseline/20261108"
)
DATASET = ROOT / "MVTec"
PATHS = ROOT / "configs/paths/asus-mvtec.yaml"
PAIRS = 500


def _python() -> str:
    return sys.executable


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONNOUSERSITE"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with {completed.returncode}: {command[0:4]}")


def _prepare_manifest(category: str) -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    from cadiff.v06_c.mvtec_pilot import load_mvtec_inventory, write_mvtec_manifest

    inventory = load_mvtec_inventory(DATASET, category)
    manifest = ROOT / "splits" / "mvtec" / category / "manifest.jsonl"
    if not manifest.is_file():
        write_mvtec_manifest(inventory, manifest)
    return manifest


def _latest_results(eval_dir: Path, category: str) -> Path | None:
    folder = eval_dir / category
    files = sorted(folder.glob("results_*.json")) if folder.is_dir() else []
    return files[-1] if files else None


def run_category(category: str) -> None:
    manifest = _prepare_manifest(category)
    category_root = RUN_ROOT / category
    train_out = category_root / "exact-mask-r"
    generate_out = category_root / "generate"
    layout = category_root / "generated_layout"
    eval_dir = category_root / "eval"
    log_path = category_root / "run.log"
    py = _python()
    if not (train_out / "summary.json").is_file():
        _run(
            [
                py, "-m", "cadiff.cli", "v06-c-visa-train-r",
                "--paths", str(PATHS),
                "--manifest", str(manifest),
                "--dataset-root", str(DATASET),
                "--category", category,
                "--split", "mvtec",
                "--output", str(train_out),
            ],
            log_path,
        )
    if not (generate_out / "summary.json").is_file():
        _run(
            [
                py, "-m", "cadiff.cli", "v06-c-visa-place-bulk",
                "--paths", str(PATHS),
                "--source-root", str(DATASET),
                "--category", category,
                "--split", "mvtec",
                "--checkpoint", str(train_out / "checkpoint"),
                "--pairs-per-type", str(PAIRS),
                "--output", str(generate_out),
            ],
            log_path,
        )
    summary = json.loads((generate_out / "summary.json").read_text(encoding="utf-8"))
    if int(summary["kept_rows"]) < 1:
        raise RuntimeError(f"{category} generated no kept pairs")
    if not (layout / category).is_dir():
        sys.path.insert(0, str(ROOT / "src"))
        from cadiff.v06_c.mvtec_pilot import load_mvtec_inventory
        from cadiff.v06_c.visa_place_bulk import pack_visa_eval_generated

        inventory = load_mvtec_inventory(DATASET, category)
        pack_visa_eval_generated(
            generate_out,
            layout,
            category,
            include_slugs=tuple(item.slug for item in inventory.types),
        )
    if _latest_results(eval_dir, category) is None:
        sys.path.insert(0, str(ROOT / "src"))
        from cadiff.v06_c.mvtec_pilot import load_mvtec_inventory

        inventory = load_mvtec_inventory(DATASET, category)
        names = [item.slug for item in inventory.types]
        _run(
            [
                py, str(ROOT / "eval_kit" / "evaluate.py"),
                "--class_name", category,
                "--anomaly_names", *names,
                "--eval_anomaly_names", *names,
                "--metrics", "localization", "classification",
                "--generated_root", str(layout),
                "--mvtec_path", str(DATASET),
                "--image_subdir", "composed",
                "--mask_subdir", "mask_defect",
                "--loc_epochs", "200",
                "--cls_epochs", "30",
                "--img_size", "256",
                "--loc_bs", "16",
                "--loc_accum", "1",
                "--output_dir", str(eval_dir),
            ],
            log_path,
        )
    results_path = _latest_results(eval_dir, category)
    if results_path is None:
        raise RuntimeError(f"{category} evaluation wrote no results")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    kept = {
        key: payload[key]
        for key in payload
        if key.startswith("loc_") or key == "cls_accuracy"
    }
    report = {"category": category, "results": str(results_path), **kept}
    (category_root / "baseline_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", required=True)
    args = parser.parse_args()
    failures: list[str] = []
    for category in args.categories:
        try:
            run_category(category)
        except Exception as error:
            failures.append(f"{category}: {error}")
            print(f"FAIL {category}: {error}", flush=True)
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()

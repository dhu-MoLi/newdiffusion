#!/usr/bin/env python3
"""Train a category-specific masked reconstructor from normal images.

The holes come from the few-shot shape prior (B0); the normal image provides
the supervision. Nothing here uses defect images at training time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.v07.reconstructor import ReconstructorConfig, train_reconstructor
from cadiff.v07.runmeta import collect_run_meta, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--paths", default="configs/paths/asus-mvtec.yaml")
    parser.add_argument("--split", default="mvtec")
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--feature-every", type=int, default=10)
    parser.add_argument("--feature-weight", type=float, default=0.1)
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--real-mask-fraction", type=float, default=0.5)
    parser.add_argument("--supports-cache", type=Path, default=None)
    arguments = parser.parse_args(argv)

    project = load_config(ROOT / arguments.paths)
    name = arguments.name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = (
        arguments.output
        if arguments.output is not None
        else Path(project.runs_root) / "v07" / "reconstructors" / arguments.category / name
    )
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = ReconstructorConfig(
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        base_channels=arguments.base_channels,
        depth=arguments.depth,
        learning_rate=arguments.lr,
        seed=arguments.seed,
        feature_every=arguments.feature_every,
        feature_weight=arguments.feature_weight,
        gradient_weight=arguments.gradient_weight,
        val_every=arguments.val_every,
        val_size=arguments.val_size,
        real_mask_fraction=arguments.real_mask_fraction,
        device=arguments.device,
    )
    config_path = run_dir / "config.json"
    write_json(
        config_path,
        {
            "category": arguments.category,
            "paths": arguments.paths,
            "split": arguments.split,
            "name": name,
            **asdict(config),
        },
    )
    write_json(
        run_dir / "run_meta.json",
        collect_run_meta(
            repo_root=ROOT,
            config_path=config_path,
            seed=config.seed,
            extra={"category": arguments.category, "device": arguments.device},
        ),
    )
    summary = train_reconstructor(
        category=arguments.category,
        paths_yaml=ROOT / arguments.paths,
        split=arguments.split,
        output=run_dir / "model",
        config=config,
        supports_cache=arguments.supports_cache,
    )
    print(
        json.dumps(
            {
                "category": summary["category"],
                "best": summary["best"],
                "baseline_neighbor": summary["baseline_neighbor"],
                "baseline_telea": summary["baseline_telea"],
                "checkpoint": summary["checkpoint"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

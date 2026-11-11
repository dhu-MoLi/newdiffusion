#!/usr/bin/env python3
"""Train and leave-one-mask-out evaluate the L2 placement scorer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.config import load_config
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_c.visa_pilot import load_visa_inventory_for_split
from cadiff.v07.placement_scorer import (
    PlacementScorer,
    ScorerConfig,
    build_projected_dataset,
    lomo_evaluate,
)


def _stack(dataset) -> tuple[np.ndarray, np.ndarray]:
    xs = [pair.positive_features for pair in dataset.pairs] + [
        pair.negative_features for pair in dataset.pairs
    ]
    ys = [np.ones(len(pair.positive_features)) for pair in dataset.pairs] + [
        np.zeros(len(pair.negative_features)) for pair in dataset.pairs
    ]
    return np.concatenate(xs), np.concatenate(ys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--normals", type=int, default=24)
    parser.add_argument("--positives", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=320)
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--no-lomo", action="store_true")
    arguments = parser.parse_args(argv)

    project = load_config(arguments.paths)
    output_root = arguments.output or (project.runs_root / "v07" / "placement_scorer")
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = load_official_dinov2_encoder(device=arguments.device)
    cfg = ScorerConfig(
        normals_limit=arguments.normals,
        positives_per_pair=arguments.positives,
        negatives_per_pair=arguments.negatives,
        pca_components=arguments.pca_components,
    )

    summaries: dict[str, dict] = {}
    for category in arguments.categories:
        started = time.time()
        inventory = load_visa_inventory_for_split(
            project.dataset_root, category, arguments.dataset, None
        )
        if arguments.no_lomo:
            summary: dict = {"schema": "skipped"}
        else:
            summary = lomo_evaluate(inventory, category, encoder, cfg)
        dataset = build_projected_dataset(inventory, category, encoder, cfg)
        x, y = _stack(dataset)
        scorer = PlacementScorer(cfg).fit(x, y)
        target = output_root / category
        scorer.save(target / "model.pkl")
        summary["category"] = category
        summary["pairs"] = len(dataset.pairs)
        summary["samples"] = int(x.shape[0])
        summary["seconds"] = round(time.time() - started, 1)
        target.mkdir(parents=True, exist_ok=True)
        (target / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summaries[category] = summary
        print(
            json.dumps(
                {
                    "category": category,
                    "pairs": summary["pairs"],
                    "samples": summary["samples"],
                    "learned_auc": round(summary.get("learned_auc_mean", float("nan")), 4),
                    "l1_auc": round(summary.get("l1_auc_mean", float("nan")), 4),
                    "learned_top10": round(summary.get("learned_top10_mean", float("nan")), 4),
                    "l1_top10": round(summary.get("l1_top10_mean", float("nan")), 4),
                    "seconds": summary["seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    (output_root / "summary_all.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

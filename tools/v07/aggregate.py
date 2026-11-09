#!/usr/bin/env python3
"""Aggregate v07 metrics.json files into a per-category table and CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "loc_auroc_image",
    "loc_ap_image",
    "loc_auroc_pixel",
    "loc_ap_pixel",
    "loc_f1_image",
    "loc_f1_pixel",
    "loc_pro_pixel",
    "cls_accuracy",
)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return (mean, std)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/v07"))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    records: list[dict[str, Any]] = []
    for path in sorted(arguments.root.glob("**/metrics.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if arguments.experiment and record.get("experiment") != arguments.experiment:
            continue
        records.append(record)
    if not records:
        raise SystemExit(f"no metrics.json under {arguments.root}")

    by_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[(record.get("experiment", "?"), record.get("category", "?"))].append(record)
        by_experiment[record.get("experiment", "?")].append(record)

    lines = ["# v07 metrics", ""]
    lines.append("## Overall")
    lines.append("")
    header = "| experiment | n | " + " | ".join(METRICS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(METRICS) + 2))
    for experiment, group in sorted(by_experiment.items()):
        cells = []
        for metric in METRICS:
            values = [float(item[metric]) for item in group if metric in item]
            mean, std = _mean_std(values)
            cells.append("n/a" if not values else f"{mean:.4f}±{std:.4f}")
        lines.append(f"| {experiment} | {len(group)} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Per category")
    lines.append("")
    lines.append("| experiment | category | n | " + " | ".join(METRICS) + " |")
    lines.append("|" + "---|" * (len(METRICS) + 3))
    for (experiment, category), group in sorted(by_category.items()):
        cells = []
        for metric in METRICS:
            values = [float(item[metric]) for item in group if metric in item]
            mean, std = _mean_std(values)
            cells.append("n/a" if not values else f"{mean:.4f}±{std:.4f}")
        lines.append(
            f"| {experiment} | {category} | {len(group)} | " + " | ".join(cells) + " |"
        )

    report = "\n".join(lines) + "\n"
    print(report)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report, encoding="utf-8")

    csv_path = arguments.output.with_suffix(".csv") if arguments.output else None
    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["experiment", "category", "seed", *METRICS])
            for record in sorted(
                records, key=lambda item: (item.get("experiment"), item.get("category"), item.get("seed"))
            ):
                writer.writerow(
                    [
                        record.get("experiment"),
                        record.get("category"),
                        record.get("seed"),
                        *[record.get(metric) for metric in METRICS],
                    ]
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

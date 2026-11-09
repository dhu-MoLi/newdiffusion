#!/usr/bin/env python3
"""Compare generated mask statistics with a reference mask directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.v07.shape_prior import mask_metrics  # noqa: E402


def _load(folder: Path, limit: int | None) -> dict[str, np.ndarray]:
    files = sorted(folder.glob("*.png"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no PNG masks under {folder}")
    rows: list[dict[str, float]] = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        metrics = mask_metrics(image > 127)
        rows.append(metrics)
    if not rows:
        raise SystemExit(f"no readable masks under {folder}")
    return {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    generated = _load(arguments.generated, arguments.limit)
    reference = _load(arguments.reference, arguments.limit)
    report = {
        "generated": str(arguments.generated),
        "reference": str(arguments.reference),
        "generated_count": int(len(generated["area"])),
        "reference_count": int(len(reference["area"])),
        "metrics": {},
    }
    for key in ("area", "aspect", "solidity", "components"):
        statistic, pvalue = ks_2samp(generated[key], reference[key])
        report["metrics"][key] = {
            "generated_mean": float(generated[key].mean()),
            "generated_std": float(generated[key].std()),
            "reference_mean": float(reference[key].mean()),
            "reference_std": float(reference[key].std()),
            "ks_statistic": float(statistic),
            "ks_pvalue": float(pvalue),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

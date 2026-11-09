#!/usr/bin/env python3
"""Tensor-level smoke test for v07. Does not load any diffusion weights."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadiff.v07.shape_prior import ParametricShapePrior, mask_metrics
from cadiff.v07.visibility import latent_lowfreq_loss


def _synthetic_mask(size: int = 512, center=(256, 256), radius=40) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    return (grid_y - center[0]) ** 2 + (grid_x - center[1]) ** 2 <= radius**2


def test_visibility() -> None:
    predicted = torch.zeros(1, 4, 64, 64)
    target = torch.zeros(1, 4, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1.0
    same = latent_lowfreq_loss(predicted, target, mask, factor=2)
    assert float(same) == 0.0, same
    predicted[:, :, 20:40, 20:40] = 1.0
    different = latent_lowfreq_loss(predicted, target, mask, factor=2)
    assert float(different) > 0.0, different
    empty = latent_lowfreq_loss(predicted, target, torch.zeros_like(mask), factor=2)
    assert float(empty) == 0.0
    print("visibility ok", float(different))


def test_shape_prior() -> None:
    masks = [_synthetic_mask(radius=radius) for radius in (20, 30, 45, 60)]
    prior = ParametricShapePrior(
        {"scratch": masks},
        multi_component_probability=0.5,
        max_components=3,
        rng_seed=7,
    )
    areas = []
    for seed in range(20):
        sampled = prior.sample("scratch", seed)
        assert sampled.shape == (512, 512)
        assert sampled.dtype == np.bool_
        assert sampled.any()
        areas.append(int(sampled.sum()))
        metrics = mask_metrics(sampled)
        assert metrics["area"] > 0
    print("shape prior ok", min(areas), max(areas), prior.describe("scratch"))


def test_runmeta() -> None:
    from cadiff.v07.runmeta import collect_run_meta, write_json

    config = ROOT / "configs/experiments/v07/mini_baseline.yaml"
    meta = collect_run_meta(repo_root=ROOT, config_path=config, seed=1)
    assert meta["config_sha256"]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "run_meta.json"
        write_json(path, meta)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["seed"] == 1
    print("runmeta ok", meta["git_commit"])


def main() -> int:
    test_visibility()
    test_shape_prior()
    test_runmeta()
    print("v07 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

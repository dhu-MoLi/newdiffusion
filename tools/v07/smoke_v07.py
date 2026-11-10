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

from cadiff.v06_atlas_diag.features import DinoFeatures
from cadiff.v07.location import ObjectLocationPrior
from cadiff.v07.placement import instantiate_site_scale_v07
from cadiff.v07.proposal_probe import (
    fill_consistency_from_features,
    location_stats,
    probe_from_features,
    site_probe,
)
from cadiff.v07.reconstruction import make_ns, make_reconstructor, make_telea, ring_mask
from cadiff.v07.shape_prior import ParametricShapePrior, mask_metrics
from cadiff.v07.site_selection import proposal_sites, select_site
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


def test_reconstruction() -> None:
    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    hole = np.zeros((512, 512), dtype=bool)
    hole[200:260, 220:280] = True
    valid = np.ones((512, 512), dtype=bool)
    reference = np.random.RandomState(42).randint(
        0, 255, (512, 512, 3), dtype=np.uint8
    )
    backends = [
        make_reconstructor("neighbor"),
        make_telea(),
        make_ns(),
        make_reconstructor("atlas", references=(reference,)),
    ]
    for backend in backends:
        output = backend(rgb, hole, valid)
        assert output.shape == rgb.shape and output.dtype == np.uint8
        assert np.array_equal(output[~hole], rgb[~hole])
        assert not np.array_equal(output[hole], rgb[hole])
    ring = ring_mask(hole, valid, radius=12)
    assert ring.any() and not (ring & hole).any()
    print("reconstruction ok", [item.name for item in backends])


def test_probe() -> None:
    rng = np.random.RandomState(1)
    source_features = DinoFeatures(
        tokens=rng.rand(8, 8, 16).astype(np.float32),
        grid_height=8,
        grid_width=8,
    )
    target_features = DinoFeatures(
        tokens=rng.rand(8, 8, 16).astype(np.float32),
        grid_height=8,
        grid_width=8,
    )
    valid = np.ones((512, 512), dtype=bool)
    support = np.ones((512, 512), dtype=bool)
    source_mask = np.zeros((512, 512), dtype=bool)
    source_mask[100:200, 100:200] = True
    rgb = rng.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    filled = rgb.copy()
    filled[source_mask] = 127
    probe = probe_from_features(
        source_features=source_features,
        source_mask=source_mask,
        source_valid=valid,
        source_support=support,
        target_features=target_features,
        target_valid=valid,
        target_support=support,
        quantile=0.60,
    )
    assert probe.proposal.shape == (512, 512)
    assert all(np.isfinite(list(probe.stats.values())))
    site = site_probe(target_support=support, proposal=probe.proposal, seed=1)
    assert site.pixel_count >= 0
    location = location_stats(mask=source_mask, support=support)
    assert np.isfinite(location["sdf_median"])
    fill = fill_consistency_from_features(
        source_rgb=rgb,
        filled_rgb=filled,
        hole=source_mask,
        valid=valid,
        features=source_features,
        ring_radius=64,
    )
    assert np.isfinite(fill["fill_feature_cos"])
    print(
        "probe ok",
        f"proposal={int(probe.proposal.sum())}",
        f"site={site.found}",
        f"fill_cos={fill['fill_feature_cos']:.3f}",
    )


def test_reconstructor_model() -> None:
    import torch

    from cadiff.v07.reconstructor import MaskedUNet, place_shape_in_support

    model = MaskedUNet(base_channels=8, depth=2)
    image = torch.randn(1, 3, 64, 64)
    hole = torch.zeros(1, 1, 64, 64)
    hole[:, :, 20:44, 20:44] = 1.0
    output = model(image, hole)
    assert output.shape == image.shape
    loss = ((output - image).abs() * hole).mean()
    loss.backward()
    support = np.ones((512, 512), dtype=bool)
    valid = np.ones((512, 512), dtype=bool)
    shape = _synthetic_mask(radius=40)
    placed = place_shape_in_support(
        shape, support, valid, np.random.RandomState(0)
    )
    assert placed is not None and placed.any()
    print("reconstructor model ok", float(loss), int(placed.sum()))


def test_reconstructor_dataset() -> None:
    from cadiff.v07.reconstructor import NormalHoleDataset

    masks = [_synthetic_mask(radius=radius) for radius in (20, 30, 45, 60)]
    prior = ParametricShapePrior({"scratch": masks})
    images = (
        np.random.RandomState(3).randint(0, 255, (512, 512, 3), dtype=np.uint8),
    )
    supports = (np.ones((512, 512), dtype=bool),)
    valids = (np.ones((512, 512), dtype=bool),)
    dataset = NormalHoleDataset(
        images=images,
        supports=supports,
        valids=valids,
        prior=prior,
        defect_types=("scratch",),
        length=4,
        seed=1,
        real_masks=masks,
        real_mask_fraction=1.0,
    )
    for index in range(4):
        image, hole = dataset[index]
        assert image.shape == (3, 512, 512)
        assert hole.shape == (1, 512, 512)
        assert float(hole.sum()) > 0
    print("reconstructor dataset ok")


def test_site_and_location() -> None:
    support = _synthetic_mask(radius=220)
    valid = np.ones((512, 512), dtype=bool)
    proposal = np.zeros((512, 512), dtype=bool)
    proposal[150:250, 150:250] = True
    proposal[300:380, 300:380] = True
    proposal &= support
    similarity = np.random.RandomState(5).rand(512, 512).astype(np.float32)
    sites = proposal_sites(
        proposal, support, valid, similarity=similarity, min_pixels=64
    )
    assert len(sites) >= 2
    chosen = select_site(sites, seed=1)
    assert 0 < chosen.index <= len(sites)
    assert chosen.pixels < int(proposal.sum())

    mask_a = _synthetic_mask(center=(256, 256), radius=40)
    mask_b = _synthetic_mask(center=(300, 300), radius=40)
    prior = ObjectLocationPrior.from_masks(
        {"scratch": [mask_a, mask_b]},
        {"scratch": [support, support]},
        bins=32,
        sigma=1.0,
    )
    score = prior.score_mask(mask_a, support, "scratch")
    assert np.isfinite(score)
    heat = prior.heatmap(support, "scratch")
    assert heat.shape == (512, 512) and float(heat.max()) > 0.0

    template, placement = instantiate_site_scale_v07(
        mask_a,
        support,
        support,
        proposal,
        similarity,
        seed=1,
        fewshot_areas=(int(mask_a.sum()),),
        min_shape_pixels=16,
        site_method="proposal_cc",
    )
    assert template.shape == (512, 512)
    assert placement.placed_pixels >= 16
    assert not (placement.mask & ~proposal).any()
    print(
        "site/location/placement ok",
        len(sites),
        f"score={score:.3f}",
        f"placed={placement.placed_pixels}",
    )


def main() -> int:
    test_visibility()
    test_shape_prior()
    test_reconstruction()
    test_probe()
    test_reconstructor_model()
    test_reconstructor_dataset()
    test_site_and_location()
    test_runmeta()
    print("v07 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

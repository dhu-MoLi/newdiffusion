"""v07 generation wrapper.

Reuses the v06 bulk-dump driver, patching three call sites:

* placement -> parametric shape prior (instead of a direct few-shot mask copy);
* placement -> proposal-component sites with an optional location prior (L0/L1);
* renderer -> v07 renderer when the backbone is an inpainting UNet.

The A2 material-prototype fill backend (neighbor / telea / learned) is passed
through to the bulk driver.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cadiff.v06_b.a2_component_place import instantiate_site_scale
from cadiff.v06_c.visa_pilot import (
    fewshot_identity_records,
    load_visa_identity,
)
from cadiff.v07.backbones import BackboneSpec
from cadiff.v07.placement import (
    instantiate_site_scale_location_weighted,
    instantiate_site_scale_v07,
)
from cadiff.v07.scale_calibration import instantiate_scale_calibrated
from cadiff.v07.shape_prior import ParametricShapePrior

V07_SITE_METHODS = frozenset(
    {
        "proposal_cc",
        "proposal_cc_prior",
        "location_weighted",
        "learned_prior",
        "scale_calibrated",
        "scale_calibrated_v4",
    }
)


def build_shape_prior(
    inventory: Any,
    *,
    size: int = 512,
    elastic_alpha: float = 6.0,
    elastic_sigma: float = 8.0,
    elastic_probability: float = 0.5,
    multi_component_probability: float = 0.0,
    max_components: int = 3,
    rng_seed: int = 0,
) -> ParametricShapePrior:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in fewshot_identity_records(inventory):
        identity = load_visa_identity(record, dataset_root=inventory.source_root)
        if identity.anomaly_gt is None:
            raise ValueError(f"few-shot record has no mask: {record.item_id}")
        grouped[record.defect_type].append(
            np.ascontiguousarray(np.asarray(identity.anomaly_gt, dtype=bool))
        )
    return ParametricShapePrior(
        grouped,
        size=size,
        elastic_alpha=elastic_alpha,
        elastic_sigma=elastic_sigma,
        elastic_probability=elastic_probability,
        multi_component_probability=multi_component_probability,
        max_components=max_components,
        rng_seed=rng_seed,
    )


def build_location_prior(inventory: Any, *, device: str | None = None) -> Any:
    """L1: object-frame histogram of few-shot defect positions."""

    from cadiff.v06_a.tokencut import load_official_dinov2_encoder
    from cadiff.v07.location import ObjectLocationPrior

    encoder = load_official_dinov2_encoder(device=device)
    return ObjectLocationPrior.from_inventory(inventory, encoder)


def placement_from_official_a2_v07(
    generator: Any,
    target: Any,
    defect_type: str,
    seed: int,
    *,
    shape_prior: ParametricShapePrior | None = None,
    fewshot_areas_transform: Any = None,
    return_template: bool = False,
    site_method: str = "baseline",
    location_prior: Any = None,
    **place_kwargs: Any,
):
    a2 = generator.generate(target, defect_type, seed)
    if getattr(a2, "object_support_method", None) != "tokencut":
        raise ValueError("v07 bulk dump requires official TokenCut a2")
    if (
        a2.similarity is None
        or a2.source_gt is None
        or a2.source_support is None
        or a2.target_support is None
    ):
        raise ValueError("official a2 maps required")

    source_gt = np.ascontiguousarray(np.asarray(a2.source_gt, dtype=bool))
    if shape_prior is not None:
        source_gt = shape_prior.sample(defect_type, seed)
        fewshot_areas = (int(source_gt.sum()),)
    else:
        fewshot = getattr(a2, "fewshot_gt_areas", None)
        if fewshot is None:
            fewshot_areas = (int(source_gt.sum()),)
        else:
            fewshot_areas = tuple(int(value) for value in fewshot)
    if fewshot_areas_transform is not None:
        fewshot_areas = fewshot_areas_transform(defect_type, fewshot_areas)

    target_valid = np.ascontiguousarray(target.valid, dtype=bool)
    target_support = np.ascontiguousarray(a2.target_support & target.valid, dtype=bool)
    if site_method in {"scale_calibrated", "scale_calibrated_v4"}:
        template, placement = instantiate_scale_calibrated(
            source_gt,
            a2.source_support,
            target_support,
            a2.proposal,
            a2.similarity,
            seed=seed,
            fewshot_areas=fewshot_areas,
            min_shape_pixels=int(place_kwargs.get("min_shape_pixels", 64)),
            location_prior=location_prior,
            defect_type=defect_type,
            target_valid=target_valid,
            policy=(
                "baseline_small"
                if site_method == "scale_calibrated_v4"
                else "calibrated"
            ),
        )
    elif site_method in {"location_weighted", "learned_prior"}:
        prior = location_prior
        if site_method == "learned_prior" and hasattr(prior, "bind"):
            prior = prior.bind(target.normal.rgb, target.normal.valid)
        template, placement = instantiate_site_scale_location_weighted(
            source_gt,
            a2.source_support,
            target_support,
            a2.proposal,
            a2.similarity,
            seed=seed,
            fewshot_areas=fewshot_areas,
            min_shape_pixels=int(place_kwargs.get("min_shape_pixels", 64)),
            location_prior=prior,
            defect_type=defect_type,
            target_valid=target_valid,
        )
    elif site_method in V07_SITE_METHODS:
        template, placement = instantiate_site_scale_v07(
            source_gt,
            a2.source_support,
            target_support,
            a2.proposal,
            a2.similarity,
            seed=seed,
            fewshot_areas=fewshot_areas,
            min_shape_pixels=int(place_kwargs.get("min_shape_pixels", 64)),
            site_method=site_method,
            location_prior=location_prior,
            defect_type=defect_type,
            target_valid=target_valid,
        )
    else:
        place_kwargs.pop("allow_proposal_floor", None)
        template, placement = instantiate_site_scale(
            source_gt,
            a2.source_support,
            target_support,
            a2.proposal,
            a2.similarity,
            seed=seed,
            fewshot_areas=fewshot_areas,
            **place_kwargs,
        )
    if return_template:
        return a2, placement, template
    return a2, placement


@contextmanager
def _patched_generation(
    *,
    shape_prior: ParametricShapePrior | None,
    backbone: BackboneSpec | None,
    site_method: str = "baseline",
    location_prior: Any = None,
):
    import cadiff.v06_c.visa_place_bulk as bulk

    original_placement = bulk.placement_from_official_a2
    original_renderer = bulk.load_visa_renderer
    if shape_prior is not None or site_method in V07_SITE_METHODS:
        bulk.placement_from_official_a2 = partial(
            placement_from_official_a2_v07,
            shape_prior=shape_prior,
            site_method=site_method,
            location_prior=location_prior,
        )
    if backbone is not None and backbone.inpaint:
        from cadiff.v07.render import load_v07_renderer

        bulk.load_visa_renderer = partial(load_v07_renderer, spec=backbone)
    try:
        yield
    finally:
        bulk.placement_from_official_a2 = original_placement
        bulk.load_visa_renderer = original_renderer


def run_visa_place_bulk_v07(
    *,
    inventory: Any,
    checkpoint: Path,
    model_root: Path,
    output: Path,
    device: str | None = None,
    pairs_per_type: int = 500,
    index_start: int = 0,
    only_types: Sequence[str] | None = None,
    shape_prior: ParametricShapePrior | None = None,
    backbone: BackboneSpec | None = None,
    site_method: str = "baseline",
    location_prior: Any = None,
    a2_fill_backend: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    import cadiff.v06_c.visa_place_bulk as bulk

    with _patched_generation(
        shape_prior=shape_prior,
        backbone=backbone,
        site_method=site_method,
        location_prior=location_prior,
    ):
        return bulk.run_visa_place_bulk(
            inventory=inventory,
            checkpoint=checkpoint,
            model_root=model_root,
            output=output,
            device=device,
            pairs_per_type=pairs_per_type,
            index_start=index_start,
            only_types=only_types,
            a2_fill_backend=a2_fill_backend,
            **kwargs,
        )


__all__ = [
    "build_location_prior",
    "build_shape_prior",
    "placement_from_official_a2_v07",
    "run_visa_place_bulk_v07",
]

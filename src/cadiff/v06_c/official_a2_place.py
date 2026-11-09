"""Official TokenCut a2 + TokenCut-CC site-scale placement for diagnostic bulk dumps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from cadiff.v06_b.a2_component_place import instantiate_site_scale

_OFFICIAL_OBJECT = "tokencut"
FewshotAreaTransform = Callable[[str, tuple[int, ...]], tuple[int, ...]]


def large_cluster_fewshot_areas(areas: Sequence[int]) -> tuple[int, ...]:
    """Keep the high mode after the largest log-gap in that type's few-shot areas."""

    ranked = tuple(sorted(int(area) for area in areas if int(area) > 0))
    if not ranked:
        raise ValueError("few-shot areas are empty")
    if len(ranked) == 1:
        return ranked
    logs = np.log(np.asarray(ranked, dtype=np.float64))
    cut = int(np.argmax(logs[1:] - logs[:-1])) + 1
    kept = ranked[cut:]
    if not kept:
        raise ValueError("large-cluster few-shot areas are empty")
    return kept


def large_cluster_source_items(items: Sequence[Any]) -> tuple[Any, ...]:
    """Keep few-shot identities whose GT area sits in the high log-gap mode."""

    ordered = tuple(items)
    if not ordered:
        raise ValueError("few-shot sources are empty")
    areas: list[int] = []
    for item in ordered:
        gt = getattr(item, "anomaly_gt", None)
        if gt is None:
            raise ValueError("few-shot sources require ground-truth masks")
        area = int(np.asarray(gt, dtype=bool).sum())
        if area < 1:
            raise ValueError("few-shot source mask is empty")
        areas.append(area)
    kept_areas = set(large_cluster_fewshot_areas(areas))
    kept = tuple(item for item, area in zip(ordered, areas, strict=True) if area in kept_areas)
    if not kept:
        raise ValueError("large-cluster few-shot sources are empty")
    return kept


def mixed_stamp_count(seed: int) -> int:
    """Deterministic 1-or-2 from the job seed. Combined diagnostic only."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return 1 + (seed % 2)


def placement_from_official_a2(
    generator: Any,
    target: Any,
    defect_type: str,
    seed: int,
    *,
    fewshot_areas_transform: FewshotAreaTransform | None = None,
    return_template: bool = False,
    **place_kwargs: Any,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
    a2 = generator.generate(target, defect_type, seed)
    if getattr(a2, "object_support_method", None) != _OFFICIAL_OBJECT:
        raise ValueError("bulk dump requires official TokenCut a2")
    if (
        a2.similarity is None
        or a2.source_gt is None
        or a2.source_support is None
        or a2.target_support is None
    ):
        raise ValueError("official a2 maps required")
    fewshot = getattr(a2, "fewshot_gt_areas", None)
    if fewshot is None:
        fewshot_areas = (int(np.asarray(a2.source_gt, dtype=bool).sum()),)
    else:
        fewshot_areas = tuple(int(value) for value in fewshot)
    if fewshot_areas_transform is not None:
        fewshot_areas = fewshot_areas_transform(defect_type, fewshot_areas)
    place_kwargs.pop("allow_proposal_floor", None)
    template, placement = instantiate_site_scale(
        a2.source_gt,
        a2.source_support,
        np.ascontiguousarray(a2.target_support & target.valid, dtype=bool),
        a2.proposal,
        a2.similarity,
        seed=seed,
        fewshot_areas=fewshot_areas,
        **place_kwargs,
    )
    if return_template:
        return a2, placement, template
    return a2, placement


__all__ = [
    "large_cluster_fewshot_areas",
    "large_cluster_source_items",
    "mixed_stamp_count",
    "placement_from_official_a2",
]

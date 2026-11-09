from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HeldoutGenerationSlot:
    slot_id: str
    normal_id: str
    mask_id: str
    source_mask_type: str
    seed: int
    defect_type: str


def _sample_id(item: Any) -> str:
    value = getattr(item, "sample_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("probe samples must have non-empty sample_id values")
    return value


def _ordered(items: tuple[Any, ...], name: str) -> list[Any]:
    if not items:
        raise ValueError(f"{name} must be non-empty")
    result = sorted(items, key=lambda item: str(getattr(item, "image_relpath", "")))
    identities = [_sample_id(item) for item in result]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} sample IDs must be unique")
    return result


def _evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return items
    if count == 1:
        raise ValueError("subsampling count must be at least two")
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("even-spacing produced duplicate indices")
    return [items[index] for index in indices]


def select_heldout_probe_inputs(
    *,
    normals: tuple[Any, ...],
    heldout_masks: tuple[Any, ...],
    defect_types: tuple[str, ...],
    masks_per_type: int,
    normal_count: int,
    excluded_normal_ids: frozenset[str],
    excluded_mask_ids: frozenset[str],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if type(masks_per_type) is not int or masks_per_type <= 0:
        raise ValueError("masks_per_type must be a positive integer")
    if type(normal_count) is not int or normal_count <= 0:
        raise ValueError("normal_count must be a positive integer")
    if normal_count != len(defect_types) * masks_per_type:
        raise ValueError("normal_count must equal defect type count times masks_per_type")
    ordered_normals = [
        item for item in _ordered(normals, "normals") if _sample_id(item) not in excluded_normal_ids
    ]
    if len(ordered_normals) < normal_count:
        raise ValueError("normal pool is smaller than normal_count after exclusions")
    selected_normals = _evenly_spaced(ordered_normals, normal_count)
    ordered_masks = _ordered(heldout_masks, "heldout_masks")
    unknown = sorted(
        {getattr(item, "defect_type", None) for item in ordered_masks} - set(defect_types),
        key=repr,
    )
    if unknown:
        raise ValueError(f"heldout_masks contain unexpected defect types: {unknown!r}")
    selected_masks: list[Any] = []
    for defect_type in defect_types:
        candidates = [
            item
            for item in ordered_masks
            if item.defect_type == defect_type and _sample_id(item) not in excluded_mask_ids
        ]
        if len(candidates) < masks_per_type:
            raise ValueError(f"heldout pool for {defect_type} is smaller than masks_per_type")
        selected_masks.extend(_evenly_spaced(candidates, masks_per_type))
    return tuple(selected_normals), tuple(selected_masks)


def _unique_strings(values: tuple[str, ...], name: str) -> None:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")


def build_heldout_generation_schedule(
    *,
    normal_ids: tuple[str, ...],
    mask_ids: tuple[str, ...],
    source_mask_types: tuple[str, ...],
    defect_types: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[HeldoutGenerationSlot, ...]:
    _unique_strings(normal_ids, "normal_ids")
    _unique_strings(mask_ids, "mask_ids")
    _unique_strings(defect_types, "defect_types")
    if len(normal_ids) != len(mask_ids) or len(mask_ids) != len(source_mask_types):
        raise ValueError("source_mask_types must align with normal_ids and mask_ids")
    if any(value not in defect_types for value in source_mask_types):
        raise ValueError("source_mask_types must belong to defect_types")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain unique values")
    slots: list[HeldoutGenerationSlot] = []
    for normal_id, mask_id, source_type in zip(
        normal_ids, mask_ids, source_mask_types, strict=True
    ):
        for seed in seeds:
            for defect_type in defect_types:
                identity = f"{normal_id}\n{mask_id}\n{source_type}\n{seed}\n{defect_type}"
                slots.append(
                    HeldoutGenerationSlot(
                        slot_id=hashlib.sha256(identity.encode()).hexdigest(),
                        normal_id=normal_id,
                        mask_id=mask_id,
                        source_mask_type=source_type,
                        seed=seed,
                        defect_type=defect_type,
                    )
                )
    return tuple(slots)


__all__ = [
    "HeldoutGenerationSlot",
    "build_heldout_generation_schedule",
    "select_heldout_probe_inputs",
]

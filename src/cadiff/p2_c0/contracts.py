from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingMask:
    sample_id: str
    defect_type: str
    role: str


@dataclass(frozen=True)
class GenerationJob:
    job_id: str
    normal_id: str
    mask_id: str
    mask_type: str
    defect_type: str
    seed: int


def _unique_strings(values: tuple[str, ...], name: str) -> None:
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def build_generation_schedule(
    *,
    normal_ids: tuple[str, ...],
    masks: tuple[TrainingMask, ...],
    defect_types: tuple[str, ...],
    samples_per_type: int,
    seed: int,
) -> tuple[GenerationJob, ...]:
    _unique_strings(normal_ids, "normal_ids")
    _unique_strings(defect_types, "defect_types")
    if type(samples_per_type) is not int or samples_per_type <= 0:
        raise ValueError("samples_per_type must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not masks:
        raise ValueError("masks must be non-empty")
    if any(mask.role != "fewshot_train" for mask in masks):
        raise ValueError("all generation masks must have role fewshot_train")
    mask_ids = tuple(mask.sample_id for mask in masks)
    _unique_strings(mask_ids, "mask sample IDs")
    unknown = {mask.defect_type for mask in masks} - set(defect_types)
    if unknown:
        raise ValueError(f"unsupported mask types: {sorted(unknown)}")
    by_type = {
        defect_type: tuple(
            sorted(
                (mask for mask in masks if mask.defect_type == defect_type),
                key=lambda item: item.sample_id,
            )
        )
        for defect_type in defect_types
    }
    if any(not candidates for candidates in by_type.values()):
        raise ValueError("each defect type must have a fewshot_train mask")
    jobs: list[GenerationJob] = []
    for type_index, defect_type in enumerate(defect_types):
        candidates = by_type[defect_type]
        for index in range(samples_per_type):
            normal_id = normal_ids[(type_index * samples_per_type + index) % len(normal_ids)]
            mask = candidates[index % len(candidates)]
            job_seed = seed + type_index * 1_000_000 + index
            identity = f"{normal_id}\n{mask.sample_id}\n{defect_type}\n{job_seed}"
            jobs.append(
                GenerationJob(
                    job_id=hashlib.sha256(identity.encode()).hexdigest(),
                    normal_id=normal_id,
                    mask_id=mask.sample_id,
                    mask_type=mask.defect_type,
                    defect_type=defect_type,
                    seed=job_seed,
                )
            )
    return tuple(jobs)

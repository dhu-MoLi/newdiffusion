from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

_TYPES = ("crack", "cut", "hole", "print")
_REFERENCE_RECORD_COUNT = 64
_RECORD_KEYS = {
    "schema_version",
    "slot_id",
    "normal_id",
    "mask_id",
    "source_mask_type",
    "seed",
    "defect_type",
    "source_normal_sha256",
    "source_mask_sha256",
    "image_relpath",
    "mask_relpath",
    "image_sha256",
    "mask_sha256",
    "generated_tensor_sha256",
    "inside_l1",
    "outside_l1",
    "outside_to_inside_ratio",
    "change_concentration",
    "composite_outside_l1",
    "finite",
}
_SHA_FIELDS = (
    "slot_id",
    "source_normal_sha256",
    "source_mask_sha256",
    "image_sha256",
    "mask_sha256",
    "generated_tensor_sha256",
)
_METRIC_FIELDS = (
    "inside_l1",
    "outside_l1",
    "outside_to_inside_ratio",
    "change_concentration",
    "composite_outside_l1",
)
_RATING_KEYS = {
    "pair_id",
    "semantic_a",
    "semantic_b",
    "realism_a",
    "realism_b",
    "boundary_a",
    "boundary_b",
}


@dataclass(frozen=True)
class TypePromptConfig:
    schema_version: str
    category: str
    normal_prompt: str
    prompts: Mapping[str, str]
    reference_git_head: str
    reference_hashes: Mapping[str, str]
    sample_count: int
    blind_seed: int


@dataclass(frozen=True)
class TypePromptJob:
    pair_id: str
    baseline_slot_id: str
    normal_id: str
    mask_id: str
    source_mask_type: str
    seed: int
    defect_type: str
    source_normal_sha256: str
    source_mask_sha256: str
    baseline_image_relpath: str
    baseline_mask_relpath: str
    baseline_image_sha256: str
    baseline_mask_sha256: str
    type_prompt: str


@dataclass(frozen=True)
class BlindAssignment:
    pair_id: str
    defect_type: str
    panel_a_source: str
    panel_b_source: str
    type_prompt_panel: str


_EXPECTED_PROMPTS = {
    "crack": "a photo of a hazelnut with thin cracks in the shell and fractured edges",
    "cut": "a photo of a hazelnut with a sharp cut in the shell",
    "hole": "a photo of a hazelnut with a puncture hole in the shell",
    "print": "a photo of a hazelnut with a dark printed mark on the shell surface",
}
_EXPECTED_REFERENCE_HASHES = {
    "metrics.jsonl": "5f08113369adff8597c9cc47ece8e30ae67414ce84cd16ba477dc7633267b1af",
    "pairwise_metrics.jsonl": "64207a3adfd59b38b703c80c40d1a457dbd9a686dad6adc872e66a720a3052be",
    "provenance.json": "3776a5d7a6bce2931db48e4b35e01859786466f24000a24c519f646acfac5a8f",
    "report.json": "9cbd22cc3380b4bca97ebb1614e832a985ea74da78440d46d9335e450ece6f93",
}


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _git_head(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase 40-character Git object ID")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_type_prompt_config(path: str | Path) -> TypePromptConfig:
    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid Type-Prompt YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("Type-Prompt configuration must be a YAML mapping")
    expected_keys = frozenset(TypePromptConfig.__dataclass_fields__)
    if set(loaded) != expected_keys:
        missing = sorted(expected_keys - set(loaded))
        extra = sorted(repr(key) for key in set(loaded) - expected_keys)
        raise ValueError(f"Type-Prompt schema mismatch: missing={missing}, extra={extra}")
    prompts = loaded["prompts"]
    if not isinstance(prompts, Mapping) or tuple(prompts) != _TYPES:
        raise ValueError("prompts must contain the exact ordered Hazelnut defect vocabulary")
    prompt_values = {key: _string(value, f"prompts.{key}") for key, value in prompts.items()}
    hashes = loaded["reference_hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(_EXPECTED_REFERENCE_HASHES):
        raise ValueError("reference_hashes must contain the exact accepted artifact names")
    hash_values = {key: _sha256(value, f"reference_hashes.{key}") for key, value in hashes.items()}
    sample_count = loaded["sample_count"]
    blind_seed = loaded["blind_seed"]
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(blind_seed) is not int or blind_seed < 0:
        raise ValueError("blind_seed must be a non-negative integer")
    config = TypePromptConfig(
        schema_version=_string(loaded["schema_version"], "schema_version"),
        category=_string(loaded["category"], "category"),
        normal_prompt=_string(loaded["normal_prompt"], "normal_prompt"),
        prompts=MappingProxyType(prompt_values),
        reference_git_head=_git_head(loaded["reference_git_head"], "reference_git_head"),
        reference_hashes=MappingProxyType(hash_values),
        sample_count=sample_count,
        blind_seed=blind_seed,
    )
    expected = {
        "schema_version": "p2_b1_type_prompt_probe_v1",
        "category": "hazelnut",
        "normal_prompt": "a photo of a normal hazelnut",
        "prompts": _EXPECTED_PROMPTS,
        "reference_git_head": "4bd6d47d83292107c9c68f99f841eb46d000789e",
        "reference_hashes": _EXPECTED_REFERENCE_HASHES,
        "sample_count": 16,
        "blind_seed": 202608177,
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"{field} must equal the frozen Type-Prompt value {value!r}")
    return config


def _validate_reference_record(record: object, index: int) -> dict[str, object]:
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise ValueError(f"reference record {index} has an invalid exact schema")
    if record["schema_version"] != "p2_b1_generation_record_v1":
        raise ValueError(f"reference record {index} has an invalid schema version")
    for field in _SHA_FIELDS:
        _sha256(record[field], f"reference record {index} {field}")
    for field in ("normal_id", "mask_id", "source_mask_type", "defect_type"):
        _string(record[field], f"reference record {index} {field}")
    for field in ("image_relpath", "mask_relpath"):
        _string(record[field], f"reference record {index} {field}")
    if record["source_mask_type"] not in _TYPES or record["defect_type"] not in _TYPES:
        raise ValueError(f"reference record {index} has an unknown defect type")
    if type(record["seed"]) is not int or record["seed"] < 0:
        raise ValueError(f"reference record {index} seed must be a non-negative integer")
    if record["finite"] is not True:
        raise ValueError(f"reference record {index} must be finite")
    for field in _METRIC_FIELDS:
        value = record[field]
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"reference record {index} {field} must be finite")
    return record


def select_type_prompt_jobs(
    records: Sequence[object], config: TypePromptConfig
) -> tuple[TypePromptJob, ...]:
    if len(records) != _REFERENCE_RECORD_COUNT:
        raise ValueError("reference metrics must contain exactly 64 records and 16 matches")
    validated = tuple(
        _validate_reference_record(record, index)
        for index, record in enumerate(records)
    )
    slot_ids = [str(record["slot_id"]) for record in validated]
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("reference metrics contain a duplicate slot_id")
    matched = [
        record
        for record in validated
        if record["source_mask_type"] == record["defect_type"]
    ]
    if len(matched) != config.sample_count:
        raise ValueError("reference metrics must contain exactly 16 matched records")
    jobs: list[TypePromptJob] = []
    for record in matched:
        defect_type = str(record["defect_type"])
        prompt = config.prompts[defect_type]
        baseline_slot_id = str(record["slot_id"])
        pair_id = hashlib.sha256(f"{baseline_slot_id}\n{prompt}".encode()).hexdigest()
        jobs.append(
            TypePromptJob(
                pair_id=pair_id,
                baseline_slot_id=baseline_slot_id,
                normal_id=str(record["normal_id"]),
                mask_id=str(record["mask_id"]),
                source_mask_type=str(record["source_mask_type"]),
                seed=int(record["seed"]),
                defect_type=defect_type,
                source_normal_sha256=str(record["source_normal_sha256"]),
                source_mask_sha256=str(record["source_mask_sha256"]),
                baseline_image_relpath=str(record["image_relpath"]),
                baseline_mask_relpath=str(record["mask_relpath"]),
                baseline_image_sha256=str(record["image_sha256"]),
                baseline_mask_sha256=str(record["mask_sha256"]),
                type_prompt=prompt,
            )
        )
    jobs.sort(key=lambda item: item.pair_id)
    if len({item.pair_id for item in jobs}) != config.sample_count:
        raise ValueError("matched Type-Prompt jobs contain a duplicate pair_id")
    if any(sum(item.defect_type == label for item in jobs) != 4 for label in _TYPES):
        raise ValueError("matched Type-Prompt jobs must contain four records per type")
    if len({item.normal_id for item in jobs}) != 8 or len({item.mask_id for item in jobs}) != 8:
        raise ValueError("matched Type-Prompt jobs must contain eight contexts")
    return tuple(jobs)


def build_blind_assignments(
    jobs: Sequence[TypePromptJob], blind_seed: int
) -> tuple[BlindAssignment, ...]:
    if type(blind_seed) is not int or blind_seed < 0:
        raise ValueError("blind_seed must be a non-negative integer")
    if len(jobs) != 16 or len({item.pair_id for item in jobs}) != 16:
        raise ValueError("blind assignment requires 16 unique jobs")
    assignments: list[BlindAssignment] = []
    for defect_type in _TYPES:
        ranked = sorted(
            (job for job in jobs if job.defect_type == defect_type),
            key=lambda item: hashlib.sha256(
                f"{blind_seed}\n{defect_type}\n{item.pair_id}".encode()
            ).hexdigest(),
        )
        if len(ranked) != 4:
            raise ValueError("blind assignment requires four jobs per defect type")
        for index, job in enumerate(ranked):
            type_panel = "A" if index < 2 else "B"
            assignments.append(
                BlindAssignment(
                    pair_id=job.pair_id,
                    defect_type=job.defect_type,
                    panel_a_source="type_prompt" if type_panel == "A" else "baseline",
                    panel_b_source="baseline" if type_panel == "A" else "type_prompt",
                    type_prompt_panel=type_panel,
                )
            )
    return tuple(sorted(assignments, key=lambda item: item.pair_id))


def evaluate_visual_ratings(
    ratings: Sequence[object],
    jobs: Sequence[TypePromptJob],
    assignments: Sequence[BlindAssignment],
    config: TypePromptConfig,
) -> dict[str, object]:
    jobs_by_pair = {item.pair_id: item for item in jobs}
    assignment_by_pair = {item.pair_id: item for item in assignments}
    if (
        len(jobs_by_pair) != config.sample_count
        or len(assignment_by_pair) != config.sample_count
        or set(assignment_by_pair) != set(jobs_by_pair)
        or len(ratings) != config.sample_count
    ):
        raise ValueError("visual ratings must cover the exact 16 Type-Prompt pairs")
    for pair_id, assignment in assignment_by_pair.items():
        job = jobs_by_pair[pair_id]
        if (
            assignment.defect_type != job.defect_type
            or assignment.type_prompt_panel not in {"A", "B"}
            or {assignment.panel_a_source, assignment.panel_b_source}
            != {"baseline", "type_prompt"}
            or (
                assignment.type_prompt_panel == "A"
                and assignment.panel_a_source != "type_prompt"
            )
            or (
                assignment.type_prompt_panel == "B"
                and assignment.panel_b_source != "type_prompt"
            )
        ):
            raise ValueError("visual ratings use an invalid blind assignment")
    rating_by_pair: dict[str, dict[str, object]] = {}
    for index, value in enumerate(ratings):
        if not isinstance(value, dict) or set(value) != _RATING_KEYS:
            raise ValueError(f"visual rating {index} has an invalid exact schema")
        pair_id = _sha256(value["pair_id"], f"visual rating {index} pair_id")
        if pair_id not in jobs_by_pair or pair_id in rating_by_pair:
            raise ValueError(f"visual rating {index} has an unknown or duplicate pair_id")
        for field in _RATING_KEYS - {"pair_id"}:
            score = value[field]
            if type(score) is not int or not 1 <= score <= 5:
                raise ValueError(f"visual rating {index} {field} must be an integer from 1 to 5")
        rating_by_pair[pair_id] = value
    semantic_by_type: dict[str, float] = {}
    realism_deltas: list[int] = []
    boundary_deltas: list[int] = []
    for defect_type in _TYPES:
        semantic_deltas: list[int] = []
        for pair_id, job in jobs_by_pair.items():
            if job.defect_type != defect_type:
                continue
            row = rating_by_pair[pair_id]
            assignment = assignment_by_pair[pair_id]
            type_panel = assignment.type_prompt_panel.lower()
            baseline_panel = "b" if type_panel == "a" else "a"
            semantic_deltas.append(
                int(row[f"semantic_{type_panel}"])
                - int(row[f"semantic_{baseline_panel}"])
            )
            realism_deltas.append(
                int(row[f"realism_{type_panel}"])
                - int(row[f"realism_{baseline_panel}"])
            )
            boundary_deltas.append(
                int(row[f"boundary_{type_panel}"])
                - int(row[f"boundary_{baseline_panel}"])
            )
        semantic_by_type[defect_type] = float(statistics.median(semantic_deltas))
    realism = float(statistics.median(realism_deltas))
    boundary = float(statistics.median(boundary_deltas))
    improved = sum(value > 0.0 for value in semantic_by_type.values())
    positive = improved >= 3 and realism >= 0.0 and boundary >= 0.0
    return {
        "schema_version": "p2_b1_type_prompt_visual_summary_v1",
        "status": "positive" if positive else "negative",
        "improved_type_count": improved,
        "semantic_median_delta_by_type": semantic_by_type,
        "realism_median_delta": realism,
        "boundary_median_delta": boundary,
    }


__all__ = [
    "BlindAssignment",
    "TypePromptConfig",
    "TypePromptJob",
    "build_blind_assignments",
    "evaluate_visual_ratings",
    "load_type_prompt_config",
    "select_type_prompt_jobs",
]

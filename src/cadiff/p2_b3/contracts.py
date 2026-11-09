from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cadiff.p2_b1.type_prompt import _validate_reference_record
from cadiff.p2_b3.config import P2B3Config

_TYPES = ("crack", "cut", "hole", "print")
_SHARED_CONDITION_FIELDS = (
    "data_role",
    "normal_id",
    "mask_id",
    "seed",
    "source_normal_sha256",
    "source_mask_sha256",
    "reference_image_sha256",
    "prompt_sha256",
    "prompt_embedding_sha256",
    "normal_latent_sha256",
    "context_latent_sha256",
    "binary_mask_sha256",
    "soft_mask_sha256",
    "noise_sha256",
    "sampler_sha256",
    "timestep_schedule_sha256",
    "scheduler_config",
    "scheduler_timesteps",
    "scheduler_set_timesteps_calls",
    "scheduler_step_calls",
    "model_forward_calls",
)
_CONDITION_KEYS = {
    "checkpoint_role",
    "checkpoint_relpath",
    "checkpoint_hashes",
    *_SHARED_CONDITION_FIELDS,
    "prediction_sha256",
    "conditioner_state_sha256",
    "adapter_state_sha256",
    "model_binding_sha256",
    "image_relpath",
    "image_sha256",
    "composite_outside_l1",
    "sampler_calls",
}
_RECORD_KEYS = {
    "schema_version",
    "pair_id",
    "reference_slot_id",
    "normal_id",
    "mask_id",
    "source_mask_type",
    "seed",
    "defect_type",
    "source_normal_sha256",
    "source_mask_sha256",
    "reference_image_relpath",
    "reference_image_sha256",
    "mask_relpath",
    "mask_sha256",
    "prompt_sha256",
    "baseline",
    "candidate",
}


@dataclass(frozen=True)
class CheckpointProbeJob:
    pair_id: str
    reference_slot_id: str
    normal_id: str
    mask_id: str
    source_mask_type: str
    seed: int
    defect_type: str
    source_normal_sha256: str
    source_mask_sha256: str
    accepted_image_relpath: str
    accepted_image_sha256: str
    accepted_mask_relpath: str
    reference_mask_sha256: str
    type_prompt: str


@dataclass(frozen=True)
class BlindAssignment:
    pair_id: str
    defect_type: str
    panel_a_condition: str
    panel_b_condition: str
    candidate_panel: str


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _canonical_json_sha256(value: object, field: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be canonical finite JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def select_checkpoint_probe_jobs(
    records: Sequence[object], config: P2B3Config
) -> tuple[CheckpointProbeJob, ...]:
    if len(records) != 64:
        raise ValueError("accepted P2-B1 reference must contain exactly 64 records")
    validated = tuple(
        _validate_reference_record(record, index)
        for index, record in enumerate(records)
    )
    slot_ids = [str(record["slot_id"]) for record in validated]
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("accepted P2-B1 reference contains a duplicate slot_id")
    matched = [
        record
        for record in validated
        if record["source_mask_type"] == record["defect_type"]
    ]
    if len(matched) != config.sample_count:
        raise ValueError("checkpoint probe requires exactly 16 true-type matches")
    jobs: list[CheckpointProbeJob] = []
    for record in matched:
        defect_type = str(record["defect_type"])
        prompt = config.prompts[defect_type]
        slot_id = str(record["slot_id"])
        identity = f"checkpoint-paired\n{slot_id}\n{prompt}"
        jobs.append(
            CheckpointProbeJob(
                pair_id=hashlib.sha256(identity.encode()).hexdigest(),
                reference_slot_id=slot_id,
                normal_id=str(record["normal_id"]),
                mask_id=str(record["mask_id"]),
                source_mask_type=str(record["source_mask_type"]),
                seed=int(record["seed"]),
                defect_type=defect_type,
                source_normal_sha256=str(record["source_normal_sha256"]),
                source_mask_sha256=str(record["source_mask_sha256"]),
                accepted_image_relpath=str(record["image_relpath"]),
                accepted_image_sha256=str(record["image_sha256"]),
                accepted_mask_relpath=str(record["mask_relpath"]),
                reference_mask_sha256=str(record["mask_sha256"]),
                type_prompt=prompt,
            )
        )
    jobs.sort(key=lambda item: item.pair_id)
    if len({item.pair_id for item in jobs}) != config.sample_count:
        raise ValueError("checkpoint probe jobs contain a duplicate pair_id")
    if any(sum(item.defect_type == label for item in jobs) != 4 for label in _TYPES):
        raise ValueError("checkpoint probe requires exactly four jobs per defect type")
    if len({item.normal_id for item in jobs}) != 8:
        raise ValueError("checkpoint probe requires exactly eight heldout normal contexts")
    if len({item.mask_id for item in jobs}) != 8:
        raise ValueError("checkpoint probe requires exactly eight heldout masks")
    return tuple(jobs)


def build_blind_assignments(
    jobs: Sequence[CheckpointProbeJob], blind_seed: int
) -> tuple[BlindAssignment, ...]:
    if type(blind_seed) is not int or blind_seed < 0:
        raise ValueError("blind_seed must be a non-negative integer")
    if len(jobs) != 16 or len({job.pair_id for job in jobs}) != 16:
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
            candidate_panel = "A" if index < 2 else "B"
            assignments.append(
                BlindAssignment(
                    pair_id=job.pair_id,
                    defect_type=defect_type,
                    panel_a_condition=(
                        "candidate" if candidate_panel == "A" else "baseline"
                    ),
                    panel_b_condition=(
                        "baseline" if candidate_panel == "A" else "candidate"
                    ),
                    candidate_panel=candidate_panel,
                )
            )
    assignments.sort(key=lambda item: item.pair_id)
    return tuple(assignments)


def _validate_condition(
    value: object,
    *,
    role: str,
    job: CheckpointProbeJob,
    record: Mapping[str, object],
    config: P2B3Config,
    index: int,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CONDITION_KEYS:
        raise ValueError(f"paired record {index} {role} has an invalid exact schema")
    if value["checkpoint_role"] != role:
        raise ValueError(f"paired record {index} {role} checkpoint role mismatch")
    expected_relpath = getattr(config, f"{role}_checkpoint_relpath")
    expected_hashes = dict(getattr(config, f"{role}_checkpoint_hashes"))
    if value["checkpoint_relpath"] != expected_relpath:
        raise ValueError(f"paired record {index} {role} checkpoint path mismatch")
    if value["checkpoint_hashes"] != expected_hashes:
        raise ValueError(f"paired record {index} {role} checkpoint hashes mismatch")
    if value["data_role"] != "heldout":
        raise ValueError(f"paired record {index} must use only the heldout data role")
    identities = {
        "normal_id": job.normal_id,
        "mask_id": job.mask_id,
        "seed": job.seed,
        "source_normal_sha256": job.source_normal_sha256,
        "source_mask_sha256": job.source_mask_sha256,
        "reference_image_sha256": record["reference_image_sha256"],
        "prompt_sha256": record["prompt_sha256"],
    }
    if any(value[field] != expected for field, expected in identities.items()):
        raise ValueError(
            f"paired record {index} {role} shared causal input identity mismatch"
        )
    for field in (
        "source_normal_sha256",
        "source_mask_sha256",
        "reference_image_sha256",
        "prompt_sha256",
        "prompt_embedding_sha256",
        "normal_latent_sha256",
        "context_latent_sha256",
        "binary_mask_sha256",
        "soft_mask_sha256",
        "noise_sha256",
        "sampler_sha256",
        "timestep_schedule_sha256",
        "prediction_sha256",
        "conditioner_state_sha256",
        "adapter_state_sha256",
        "model_binding_sha256",
        "image_sha256",
    ):
        _sha256(value[field], f"paired record {index} {role} {field}")
    timesteps = value["scheduler_timesteps"]
    scheduler_config = value["scheduler_config"]
    if (
        not isinstance(scheduler_config, Mapping)
        or not isinstance(timesteps, list)
        or len(timesteps) != config.denoise_steps
        or any(type(timestep) is not int or timestep < 0 for timestep in timesteps)
        or value["scheduler_set_timesteps_calls"] != 1
        or value["scheduler_step_calls"] != config.denoise_steps
        or value["model_forward_calls"] != config.denoise_steps
    ):
        raise ValueError(f"paired record {index} {role} scheduler/model trace is invalid")
    expected_timestep_sha = _canonical_json_sha256(
        timesteps, f"paired record {index} {role} scheduler timesteps"
    )
    expected_sampler_sha = _canonical_json_sha256(
        {
            "scheduler": "DDIMScheduler",
            "config": dict(scheduler_config),
            "timesteps": timesteps,
            "set_timesteps_calls": 1,
            "step_calls": config.denoise_steps,
            "model_forward_calls": config.denoise_steps,
        },
        f"paired record {index} {role} sampler trace",
    )
    if (
        value["timestep_schedule_sha256"] != expected_timestep_sha
        or value["sampler_sha256"] != expected_sampler_sha
    ):
        raise ValueError(f"paired record {index} {role} scheduler trace hash mismatch")
    state_hashes = {
        "conditioner": value["conditioner_state_sha256"],
        "adapter": value["adapter_state_sha256"],
    }
    expected_binding = _canonical_json_sha256(
        {
            "checkpoint_hashes": expected_hashes,
            "state_hashes": state_hashes,
        },
        f"paired record {index} {role} model binding",
    )
    if value["model_binding_sha256"] != expected_binding:
        raise ValueError(f"paired record {index} {role} model binding hash mismatch")
    if value["image_relpath"] != f"paired/{job.pair_id}_{role}.png":
        raise ValueError(f"paired record {index} {role} image path mismatch")
    if value["sampler_calls"] != 1:
        raise ValueError(f"paired record {index} {role} must make one full sampler call")
    exterior = value["composite_outside_l1"]
    if type(exterior) is not float or not math.isfinite(exterior) or exterior != 0.0:
        raise ValueError(f"paired record {index} {role} changed the mask exterior")
    return value


def validate_paired_records(
    records: Sequence[object],
    jobs: Sequence[CheckpointProbeJob],
    config: P2B3Config,
) -> None:
    if len(records) != config.sample_count or len(jobs) != config.sample_count:
        raise ValueError("paired records must match the exact 16-job schedule")
    prompt_embedding_by_type: dict[str, str] = {}
    for index, (value, job) in enumerate(zip(records, jobs, strict=True)):
        if not isinstance(value, Mapping) or set(value) != _RECORD_KEYS:
            raise ValueError(f"paired record {index} has an invalid exact schema")
        identity = {
            "pair_id": job.pair_id,
            "reference_slot_id": job.reference_slot_id,
            "normal_id": job.normal_id,
            "mask_id": job.mask_id,
            "source_mask_type": job.source_mask_type,
            "seed": job.seed,
            "defect_type": job.defect_type,
            "source_normal_sha256": job.source_normal_sha256,
            "source_mask_sha256": job.source_mask_sha256,
            "mask_sha256": job.reference_mask_sha256,
        }
        if value["schema_version"] != "p2_b3_paired_record_v1":
            raise ValueError(f"paired record {index} schema version mismatch")
        if any(value[field] != expected for field, expected in identity.items()):
            raise ValueError(f"paired record {index} differs from its frozen job")
        if value["reference_image_relpath"] != f"paired/{job.pair_id}_reference.png":
            raise ValueError(f"paired record {index} reference image path mismatch")
        if value["mask_relpath"] != f"paired/{job.pair_id}_mask.png":
            raise ValueError(f"paired record {index} mask path mismatch")
        for field in (
            "pair_id",
            "reference_slot_id",
            "source_normal_sha256",
            "source_mask_sha256",
            "reference_image_sha256",
            "mask_sha256",
            "prompt_sha256",
        ):
            _sha256(value[field], f"paired record {index} {field}")
        expected_prompt = hashlib.sha256(job.type_prompt.encode()).hexdigest()
        if value["prompt_sha256"] != expected_prompt:
            raise ValueError(f"paired record {index} prompt hash mismatch")
        baseline = _validate_condition(
            value["baseline"], role="baseline", job=job, record=value,
            config=config, index=index,
        )
        candidate = _validate_condition(
            value["candidate"], role="candidate", job=job, record=value,
            config=config, index=index,
        )
        for field in _SHARED_CONDITION_FIELDS:
            if baseline[field] != candidate[field]:
                raise ValueError(
                    f"paired record {index} shared causal input differs for {field}"
                )
        prompt_embedding = str(baseline["prompt_embedding_sha256"])
        previous = prompt_embedding_by_type.setdefault(job.defect_type, prompt_embedding)
        if previous != prompt_embedding:
            raise ValueError(f"paired record {index} prompt embedding changed within type")
    if len(set(prompt_embedding_by_type.values())) != len(_TYPES):
        raise ValueError("prompt embeddings must be distinct across defect types")


def build_blind_records(
    records: Sequence[Mapping[str, object]],
    assignments: Sequence[BlindAssignment],
    *,
    blind_ids: Mapping[str, str],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    record_by_pair = {str(record["pair_id"]): record for record in records}
    if len(record_by_pair) != 16 or len(assignments) != 16:
        raise ValueError("blind records require 16 unique certified pairs")
    if set(blind_ids) != set(record_by_pair):
        raise ValueError("blind aliases must cover exactly the certified pairs")
    for pair_id, blind_id in blind_ids.items():
        _sha256(blind_id, f"blind alias for {pair_id}")
    if len(set(blind_ids.values())) != 16:
        raise ValueError("blind aliases must be unique high-entropy identifiers")
    public: list[dict[str, object]] = []
    reveal: list[dict[str, object]] = []
    for assignment in assignments:
        record = record_by_pair.get(assignment.pair_id)
        if record is None or record["defect_type"] != assignment.defect_type:
            raise ValueError("blind assignment does not match paired records")
        panel_a = record[assignment.panel_a_condition]
        panel_b = record[assignment.panel_b_condition]
        if not isinstance(panel_a, Mapping) or not isinstance(panel_b, Mapping):
            raise ValueError("paired condition records must be mappings")
        blind_id = blind_ids[assignment.pair_id]
        prefix = f"comparisons/{blind_id}"
        public.append(
            {
                "schema_version": "p2_b3_blind_pair_v1",
                "blind_id": blind_id,
                "defect_type": assignment.defect_type,
                "panel_a_relpath": f"{prefix}_A.png",
                "panel_b_relpath": f"{prefix}_B.png",
                "reference_relpath": f"{prefix}_reference.png",
                "mask_relpath": f"{prefix}_mask.png",
                "panel_a_sha256": panel_a["image_sha256"],
                "panel_b_sha256": panel_b["image_sha256"],
                "reference_sha256": record["reference_image_sha256"],
                "mask_sha256": record["mask_sha256"],
            }
        )
        reveal.append(
            {
                "schema_version": "p2_b3_private_reveal_v1",
                "blind_id": blind_id,
                "pair_id": assignment.pair_id,
                "defect_type": assignment.defect_type,
                "candidate_panel": assignment.candidate_panel,
                "panel_a_condition": assignment.panel_a_condition,
                "panel_b_condition": assignment.panel_b_condition,
            }
        )
    return tuple(public), tuple(reveal)


__all__ = [
    "BlindAssignment",
    "CheckpointProbeJob",
    "build_blind_assignments",
    "build_blind_records",
    "select_checkpoint_probe_jobs",
    "validate_paired_records",
]

"""Helpers for Cable scheme 2: P2-A1 -> P2-B2 freeze files. Not used by Hazelnut."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from cadiff.provenance import sha256_file

CABLE_P2_A1_RELPATH = "p2_a1/cable/factorized/seed-20260817"
CABLE_P2_B2_RELPATH = "p2_b2/cable/type-prompt-training/seed-20260817"
CABLE_P2_B2_CHECKPOINT_RELPATH = f"{CABLE_P2_B2_RELPATH}/checkpoint"
CABLE_P2_B2_EXPERIMENT = "configs/experiments/p2_b2_cable_type_prompt_training.yaml"
CABLE_P2_B2_MANIFEST = "configs/checkpoints/p2_b2_cable_renderer.yaml"
_SOURCE_FILES = ("adapter.safetensors", "conditioner.safetensors", "metadata.json")
_CABLE_PROMPTS = {
    "bent_wire": "a photo of a cable with a bent conductor wire",
    "cut_inner_insulation": "a photo of a cable with a cut in the inner insulation",
    "cut_outer_insulation": "a photo of a cable with a cut in the outer insulation",
    "poke_insulation": "a photo of a cable with a poke puncture through the insulation",
}


def write_p2_b2_cable_experiment(a1_checkpoint: Path, destination: Path) -> Path:
    checkpoint = a1_checkpoint.resolve(strict=True)
    hashes = {name: sha256_file(checkpoint / name) for name in _SOURCE_FILES}
    payload = {
        "schema_version": "p2_b2_type_prompt_training_v1",
        "category": "cable",
        "defect_types": [
            "bent_wire",
            "cut_inner_insulation",
            "cut_outer_insulation",
            "poke_insulation",
        ],
        "prompts": _CABLE_PROMPTS,
        "steps": 400,
        "learning_rate": 0.0001,
        "type_rank": 8,
        "source_checkpoint_relpath": f"{CABLE_P2_A1_RELPATH}/checkpoint",
        "source_checkpoint_hashes": hashes,
        "evaluation_mode": "diagnostic_only",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def write_p2_b2_cable_freeze_manifest(b2_run: Path, destination: Path) -> Path:
    run = b2_run.resolve(strict=True)
    checkpoint = (run / "checkpoint").resolve(strict=True)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    git_head = metadata["git_head"]
    files = {
        name: {
            "sha256": sha256_file(checkpoint / name),
            "size": (checkpoint / name).stat().st_size,
        }
        for name in _SOURCE_FILES
    }

    def _hash_if_present(name: str) -> dict[str, str]:
        path = run / name
        if not path.is_file():
            return {}
        return {name: sha256_file(path)}

    run_hashes: dict[str, str] = {}
    for name in (
        "report.json",
        "provenance.json",
        "metrics.jsonl",
        "training_metrics.jsonl",
    ):
        run_hashes.update(_hash_if_present(name))
    payload = {
        "schema_version": "p2_renderer_freeze_v1",
        "renderer_id": "p2_b2_cable_type_prompt_scheme2",
        "category": "cable",
        "status": "frozen_default",
        "decision_date": "2026-09-09",
        "scope": "cable_scheme2_mask_first_initialization",
        "checkpoint": {
            "runs_root_relpath": CABLE_P2_B2_CHECKPOINT_RELPATH,
            "source_git_head": git_head,
            "files": files,
        },
        "source_run": {
            "runs_root_relpath": CABLE_P2_B2_RELPATH,
            "hashes": run_hashes,
        },
        "decision_evidence": {
            "runs_root_relpath": CABLE_P2_B2_RELPATH,
            "hashes": run_hashes,
            "human_review": {
                "status": "not_performed_training_initialization_only",
            },
        },
        "decision": {
            "default_renderer": "p2_b2_cable_type_prompt_scheme2",
            "rationale": (
                "Cable scheme 2 exact-mask P2-B2 is the mask-first initialization; "
                "not a visual PASS and not a replacement of frozen Hazelnut P2-B2."
            ),
            "limitations": [
                "Not visually reviewed as a standalone renderer.",
                "Does not claim Cable inner/outer insulation are separated.",
                "Must not overwrite the frozen Hazelnut P2-B2 checkpoint.",
            ],
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


__all__ = [
    "CABLE_P2_A1_RELPATH",
    "CABLE_P2_B2_CHECKPOINT_RELPATH",
    "CABLE_P2_B2_EXPERIMENT",
    "CABLE_P2_B2_MANIFEST",
    "CABLE_P2_B2_RELPATH",
    "write_p2_b2_cable_experiment",
    "write_p2_b2_cable_freeze_manifest",
]

"""can bulk dump: a2 + largest-CC placement H + independently trained exact-mask R.

Diagnostic generation for MVTec AD 2 can. Not official generate() and not Module C.
Does not borrow Cable/Hazelnut R. Placement does not floor to the whole can.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement, _contained_output, _save_sheet
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.can_exact_mask_r import load_can_renderer
from cadiff.v06_c.can_mvtec2 import (
    CATEGORY,
    DEFECT_TYPE,
    STRUCTURE_CONFIG,
    CanInventory,
    fewshot_identity_records,
    load_can_identity,
    normal_identity_records,
    stage_can_mvtec_layout,
)
from cadiff.v06_c.official_a2_place import placement_from_official_a2

PAIRS = 500
BASE_SEED = 20260912
PREVIEW_COLUMNS = ("normal", "proposal", "m_final", "final_y")


@dataclass(frozen=True)
class BulkJob:
    index: int
    defect_type: str
    target_id: str
    seed: int
    job_id: str


def build_can_place_bulk_jobs(
    normals: tuple[str, ...],
    *,
    pairs: int = PAIRS,
) -> tuple[BulkJob, ...]:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if not normals:
        raise ValueError("can bulk dump requires train/good targets")
    jobs: list[BulkJob] = []
    for index in range(pairs):
        seed = BASE_SEED + index
        if seed < 0 or seed >= 2**32:
            raise ValueError("bulk seed must fit in uint32")
        target_id = normals[index % len(normals)]
        job_id = f"{DEFECT_TYPE}-{target_id}-{seed}"
        jobs.append(
            BulkJob(
                index=index,
                defect_type=DEFECT_TYPE,
                target_id=target_id,
                seed=seed,
                job_id=job_id,
            )
        )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise RuntimeError("can bulk job IDs collided")
    return tuple(jobs)


def _job_done(root: Path, job: BulkJob) -> bool:
    sample = root / "samples" / job.job_id
    record_path = sample / "record.json"
    if not (
        (sample / "final_y.png").is_file()
        and (sample / "m_final.png").is_file()
        and record_path.is_file()
    ):
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return not bool(record.get("failed"))


def run_can_place_bulk(
    *,
    inventory: CanInventory,
    checkpoint: Path,
    model_root: Path,
    output: Path,
    device: str | None = None,
    pairs: int = PAIRS,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if summary_path.is_file():
        raise FileExistsError(f"refuses to overwrite completed can bulk dump: {output}")
    fewshot_records = fewshot_identity_records(inventory)
    normal_records = normal_identity_records(inventory)
    jobs = build_can_place_bulk_jobs(
        tuple(record.item_id for record in normal_records),
        pairs=pairs,
    )
    if output.exists() or output.is_symlink():
        if not (output / "samples").is_dir():
            raise FileExistsError(f"refuses to overwrite existing output: {output}")
    else:
        _contained_output(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "samples").mkdir()
    (output / "images").mkdir(exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    encoder = load_official_dinov2_encoder(device=device)
    sources = {
        DEFECT_TYPE: tuple(
            load_can_identity(record, dataset_root=inventory.layout_root)
            for record in fewshot_records
        )
    }
    generator = A2ProposalGenerator(
        sources,
        config=STRUCTURE_CONFIG,
        identity_manifest=None,
        encoder=encoder,
    )
    renderer = load_can_renderer(checkpoint=checkpoint, model_root=model_root, device=device)
    normals_by_id = {record.item_id: record for record in normal_records}
    structures: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    preview_rows: list[tuple[str, dict[str, np.ndarray]]] = []

    for job in jobs:
        sample_root = output / "samples" / job.job_id
        if _job_done(output, job) and (sample_root / "record.json").is_file():
            record = json.loads((sample_root / "record.json").read_text(encoding="utf-8"))
            records.append(record)
            print(job.job_id, "SKIP", flush=True)
            continue
        if job.target_id not in structures:
            loaded = load_can_identity(
                normals_by_id[job.target_id],
                dataset_root=inventory.layout_root,
            )
            structure = estimate_normal_structure(loaded, config=STRUCTURE_CONFIG)
            structures[job.target_id] = structure
        target = structures[job.target_id]
        sample_root.mkdir(parents=True, exist_ok=True)
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        record = {
            "schema_version": "v06_c_can_place_bulk_v1",
            "role": "synthetic_train",
            "job_id": job.job_id,
            "category": CATEGORY,
            "defect_type": job.defect_type,
            "target_id": job.target_id,
            "target_role": "normal_fit",
            "source_id": None,
            "seed": job.seed,
            "index": job.index,
            "a2_tau": None,
            "object_support_method": "tokencut",
            "h_method": "a2_largest_cc_place",
            "proposal_pixels": 0,
            "failed": False,
            "failure_reason": None,
            "used_site_fallback": False,
            "used_proposal_floor": False,
            "placed_pixels": 0,
            "image_relpath": f"images/{job.job_id}.png",
            "mask_relpath": f"masks/{job.job_id}.png",
        }
        arrays = {
            "normal": normal,
            "proposal": np.zeros(normal.shape[:2], dtype=np.uint8),
            "m_final": np.zeros(normal.shape[:2], dtype=np.uint8),
            "final_y": np.zeros_like(normal),
        }
        try:
            a2, placement = placement_from_official_a2(
                generator,
                target,
                job.defect_type,
                job.seed,
                min_shape_pixels=1,
                allow_proposal_floor=False,
            )
            proposal = np.ascontiguousarray(a2.proposal, dtype=bool)
            arrays["proposal"] = (proposal.astype(np.uint8) * 255)
            record.update(
                source_id=a2.source_id,
                a2_tau=a2.tau,
                proposal_pixels=int(proposal.sum()),
                object_support_method=a2.object_support_method,
            )
            validate_realization(placement.mask, placement.site)
            if (placement.mask & ~proposal).any():
                raise ValueError("placed shape escaped a2")
            raw = renderer(normal, job.defect_type, placement.mask, job.seed)
            final_y = compose_causal(normal, raw, placement.mask)
            if not np.array_equal(final_y[~placement.mask], normal[~placement.mask]):
                raise ValueError("renderer changed pixels outside m_final")
            arrays["m_final"] = placement.mask.astype(np.uint8) * 255
            arrays["final_y"] = np.ascontiguousarray(final_y)
            record.update(
                used_site_fallback=placement.used_site_fallback,
                used_proposal_floor=False,
                placed_pixels=placement.placed_pixels,
                site_pixels=int(placement.site.sum()),
                scale=placement.scale,
                angle_deg=placement.angle_deg,
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        Image.fromarray(arrays["normal"]).save(sample_root / "normal.png")
        Image.fromarray(arrays["proposal"]).save(sample_root / "proposal.png")
        Image.fromarray(arrays["m_final"]).save(sample_root / "m_final.png")
        Image.fromarray(arrays["final_y"]).save(sample_root / "final_y.png")
        if not record["failed"]:
            Image.fromarray(arrays["final_y"]).save(output / record["image_relpath"])
            Image.fromarray(arrays["m_final"]).save(output / record["mask_relpath"])
        write_json_atomic(sample_root / "record.json", record)
        records.append(record)
        if job.index < 4:
            preview_rows.append((job.job_id, arrays))
        status = "FAIL " + str(record["failure_reason"]) if record["failed"] else "OK"
        if not record["failed"] and record.get("used_site_fallback"):
            status = "OK FALLBACK"
        print(job.job_id, status, flush=True)
        write_json_atomic(
            output / "progress.json",
            {
                "done": len(records),
                "total": len(jobs),
                "failed": sum(row["failed"] for row in records),
            },
        )

    if preview_rows:
        _save_sheet(output / "preview.png", preview_rows, PREVIEW_COLUMNS, tile=160)
    kept = [row for row in records if not row["failed"]]
    (output / "synthetic_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in kept),
        encoding="utf-8",
    )
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "v06_c_can_place_bulk_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_module_c": True,
        "not_learned_h": True,
        "method": "tokencut_a2_sampled_place_then_independent_can_r",
        "object_support_method": "tokencut",
        "h_method": "a2_largest_cc_place",
        "category": CATEGORY,
        "pairs_requested": pairs,
        "identity_count": len(records),
        "kept_rows": len(kept),
        "failed_rows": sum(row["failed"] for row in records),
        "floor_rows": 0,
        "fallback_rows": sum(bool(row.get("used_site_fallback")) for row in records),
        "targets": "mvtec2_train_good",
        "normal_count": len(normal_records),
        "fewshot_count": len(fewshot_records),
        "renderer": "independent_can_exact_mask_r",
        "ddim_steps": 50,
        "min_shape_pixels": 1,
        "allow_proposal_floor": False,
        "destination": str(output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def run_can_place_bulk_from_cli(
    *,
    paths: Path,
    source_root: Path,
    layout_root: Path,
    checkpoint: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
    pairs: int = PAIRS,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    inventory = stage_can_mvtec_layout(source_root=source_root, layout_root=layout_root)
    return run_can_place_bulk(
        inventory=inventory,
        checkpoint=checkpoint,
        model_root=project.model_root,
        output=output,
        device=device,
        pairs=pairs,
    )


__all__ = [
    "BASE_SEED",
    "BulkJob",
    "PAIRS",
    "build_can_place_bulk_jobs",
    "run_can_place_bulk",
    "run_can_place_bulk_from_cli",
]

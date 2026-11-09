"""Cable bulk dump: official TokenCut a2 + sampled placement H + frozen scheme-2 R.

Diagnostic generation, not the 32-identity official generate() panel and not
Module C. Extra/cad3 categories have no renderer and are not included.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import IdentityManifest, load_v06_a_identity_manifest
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_b.a2_shape_instantiate import (
    EmptyFeasiblePlacement,
    _contained_output,
    _require_frozen_scheme2_renderer,
    _save_sheet,
)
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_b.renderer import compose_causal
from cadiff.v06_c.cable_generate import load_cable_generate_config, load_cable_runtime
from cadiff.v06_c.official_a2_place import placement_from_official_a2

PAIRS_PER_TYPE = 500
BASE_SEED = 20260912
TYPE_SEED_STRIDE = 10_000
PREVIEW_COLUMNS = ("normal", "proposal", "m_final", "final_y")


@dataclass(frozen=True)
class BulkJob:
    index: int
    defect_type: str
    target_id: str
    seed: int
    job_id: str


def build_cable_place_bulk_jobs(
    manifest: IdentityManifest,
    *,
    pairs_per_type: int = PAIRS_PER_TYPE,
    index_start: int = 0,
) -> tuple[BulkJob, ...]:
    if manifest.category != "cable":
        raise ValueError("bulk dump is Cable-only")
    if pairs_per_type < 1:
        raise ValueError("pairs_per_type must be positive")
    if index_start < 0:
        raise ValueError("index_start must be non-negative")
    if index_start + pairs_per_type > TYPE_SEED_STRIDE:
        raise ValueError("bulk index range collides with the next defect-type seed block")
    normals = tuple(record.item_id for record in manifest.normal_fit)
    if not normals:
        raise ValueError("Cable bulk dump requires frozen normal_fit targets")
    jobs: list[BulkJob] = []
    for type_index, defect_type in enumerate(manifest.defect_types):
        for index in range(index_start, index_start + pairs_per_type):
            seed = BASE_SEED + type_index * TYPE_SEED_STRIDE + index
            if seed < 0 or seed >= 2**32:
                raise ValueError("bulk seed must fit in uint32")
            target_id = normals[index % len(normals)]
            job_id = f"{defect_type}-{target_id}-{seed}"
            jobs.append(
                BulkJob(
                    index=index,
                    defect_type=defect_type,
                    target_id=target_id,
                    seed=seed,
                    job_id=job_id,
                )
            )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise RuntimeError("bulk job IDs collided")
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


def run_cable_place_bulk(
    *,
    repository_root: Path,
    dataset_root: Path,
    output: Path,
    a_experiment: Path,
    generate_config: Path,
    device: str | None = None,
    pairs_per_type: int = PAIRS_PER_TYPE,
    index_start: int = 0,
    renderer: Callable[[np.ndarray, str, np.ndarray, int], np.ndarray] | None = None,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if summary_path.is_file():
        raise FileExistsError(f"refuses to overwrite completed bulk dump: {output}")
    experiment = load_v06_a_experiment(a_experiment)
    manifest = load_v06_a_identity_manifest(
        repository_root / experiment.identity_manifest,
        experiment=experiment,
    )
    jobs = build_cable_place_bulk_jobs(
        manifest,
        pairs_per_type=pairs_per_type,
        index_start=index_start,
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
    generator = load_a2_generator(
        experiment_path=a_experiment,
        dataset_root=dataset_root,
        device=device,
    )
    if renderer is None:
        _require_frozen_scheme2_renderer(generate_config, repository_root)
        runtime = load_cable_runtime(
            load_cable_generate_config(generate_config, repository_root=repository_root),
            repository_root,
            device,
        )
        renderer = runtime.module_b.renderer
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
            target_record = next(
                item for item in manifest.normal_fit if item.item_id == job.target_id
            )
            loaded = load_identity_record(
                target_record,
                dataset_root=dataset_root,
                config=experiment,
            )
            structure = estimate_normal_structure(loaded, config=experiment)
            structures[job.target_id] = structure
        target = structures[job.target_id]
        sample_root.mkdir(parents=True, exist_ok=True)
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        record = {
            "schema_version": "v06_c_cable_place_bulk_v1",
            "role": "synthetic_train",
            "job_id": job.job_id,
            "category": "cable",
            "defect_type": job.defect_type,
            "target_id": job.target_id,
            "target_role": "normal_fit",
            "source_id": None,
            "seed": job.seed,
            "index": job.index,
            "a2_tau": None,
            "object_support_method": "tokencut",
            "h_method": "tokencut_site_scale_place",
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
            raw = renderer(normal, job.defect_type, placement.mask.astype(np.uint8), job.seed)
            final_y = compose_causal(normal, raw, placement.mask)
            if (placement.mask & ~proposal).any():
                raise ValueError("placed shape escaped a2")
            if not np.array_equal(final_y[~placement.mask], normal[~placement.mask]):
                raise ValueError("renderer changed pixels outside m_final")
            arrays["m_final"] = placement.mask.astype(np.uint8) * 255
            arrays["final_y"] = np.ascontiguousarray(final_y)
            record.update(
                used_site_fallback=placement.used_site_fallback,
                used_proposal_floor=placement.used_proposal_floor,
                used_clip=bool(placement.used_clip),
                placed_pixels=placement.placed_pixels,
                site_pixels=int(placement.site.sum()),
                scale=placement.scale,
                factor=float(placement.factor),
                angle_deg=placement.angle_deg,
                site_index=int(placement.site_index),
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
        if job.index < index_start + 2:
            preview_rows.append((job.job_id, arrays))
        status = "FAIL " + str(record["failure_reason"]) if record["failed"] else "OK"
        if not record["failed"] and record.get("used_proposal_floor"):
            status = "OK FLOOR"
        elif not record["failed"] and record.get("used_site_fallback"):
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
    per_type = {
        name: sum(1 for row in kept if row["defect_type"] == name) for name in manifest.defect_types
    }
    summary = {
        "schema_version": "v06_c_cable_place_bulk_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_module_c": True,
        "method": "tokencut_a2_site_scale_place_then_frozen_scheme2_r",
        "object_support_method": "tokencut",
        "h_method": "tokencut_site_scale_place",
        "category": "cable",
        "pairs_per_type_requested": pairs_per_type,
        "index_start": index_start,
        "identity_count": len(records),
        "kept_rows": len(kept),
        "failed_rows": sum(row["failed"] for row in records),
        "floor_rows": sum(bool(row.get("used_proposal_floor")) for row in records),
        "fallback_rows": sum(bool(row.get("used_site_fallback")) for row in records),
        "per_type_kept": per_type,
        "targets": "frozen_normal_fit_only",
        "normal_fit_count": len(manifest.normal_fit),
        "renderer": "frozen_scheme2",
        "ddim_steps": 50,
        "destination": str(output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def pack_cable_eval_generated(
    dump: Path | Sequence[Path],
    layout: Path,
    category: str = "cable",
) -> Path:
    from collections import defaultdict

    dumps = (dump,) if isinstance(dump, (str, Path)) else tuple(dump)
    by_type: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for root in dumps:
        root = Path(root)
        manifest = root / "synthetic_manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing bulk manifest: {manifest}")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            job_id = str(row["job_id"])
            if job_id in seen_ids:
                raise ValueError(f"duplicate generated pair while packing: {job_id}")
            seen_ids.add(job_id)
            by_type[str(row["defect_type"])].append((root, row))
    layout.mkdir(parents=True, exist_ok=True)
    for defect_type, rows in by_type.items():
        image_dir = layout / category / defect_type / "composed"
        mask_dir = layout / category / defect_type / "mask_defect"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index, (root, row) in enumerate(rows):
            src_image = root / row["image_relpath"]
            src_mask = root / row["mask_relpath"]
            if not src_image.is_file() or not src_mask.is_file():
                raise FileNotFoundError(row["job_id"])
            image_link = image_dir / f"{index}.png"
            mask_link = mask_dir / f"{index}.png"
            if image_link.is_symlink() or image_link.exists():
                image_link.unlink()
            if mask_link.is_symlink() or mask_link.exists():
                mask_link.unlink()
            image_link.symlink_to(src_image)
            mask_link.symlink_to(src_mask)
    return layout


def run_cable_place_bulk_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    generate_config: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
    pairs_per_type: int = PAIRS_PER_TYPE,
    index_start: int = 0,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_cable_place_bulk(
        repository_root=cwd,
        dataset_root=project.dataset_root,
        output=output,
        a_experiment=a_experiment if a_experiment.is_absolute() else cwd / a_experiment,
        generate_config=(
            generate_config if generate_config.is_absolute() else cwd / generate_config
        ),
        device=device,
        pairs_per_type=pairs_per_type,
        index_start=index_start,
    )


__all__ = [
    "BASE_SEED",
    "BulkJob",
    "PAIRS_PER_TYPE",
    "TYPE_SEED_STRIDE",
    "build_cable_place_bulk_jobs",
    "pack_cable_eval_generated",
    "run_cable_place_bulk",
    "run_cable_place_bulk_from_cli",
]

"""VisA bulk dump: official TokenCut a2 + site-scale H + independent exact-mask R.

Diagnostic generation for frozen extra-A VisA pilots. Not the 32-identity
official generate() panel and not Module C. Placement is instantiate_site_scale.
Does not use can native-window H/R.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import rmtree
from typing import Any, Sequence

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
from cadiff.v06_c.capsules_combined_two_stamp_panel import place_second_stamp
from cadiff.v06_c.official_a2_place import (
    large_cluster_fewshot_areas,
    large_cluster_source_items,
    mixed_stamp_count,
    placement_from_official_a2,
)
from cadiff.v06_c.visa_exact_mask_r import load_visa_renderer, visa_p2_a1_config
from cadiff.v06_c.visa_pilot import (
    COMBINED_TYPE,
    COLLAPSED_TYPE,
    EXPECTED,
    MAGIC_CATEGORIES,
    SPLIT_ID_MAGIC,
    VisaInventory,
    fewshot_identity_records,
    load_visa_identity,
    load_visa_inventory_for_split,
    normal_identity_records,
    slug_type,
)

PAIRS_PER_TYPE = 500
BASE_SEED = 20260912
TYPE_SEED_STRIDE = 10_000
PREVIEW_COLUMNS = ("normal", "proposal", "m_final", "final_y")
MIN_SHAPE_PIXELS = 64
ALLOW_PROPOSAL_FLOOR = True


@dataclass(frozen=True)
class BulkJob:
    index: int
    defect_type: str
    type_slug: str
    target_id: str
    seed: int
    job_id: str


def build_visa_place_bulk_jobs(
    inventory: VisaInventory,
    *,
    pairs_per_type: int = PAIRS_PER_TYPE,
    index_start: int = 0,
) -> tuple[BulkJob, ...]:
    if inventory.split_id == SPLIT_ID_MAGIC:
        if inventory.category not in MAGIC_CATEGORIES:
            raise ValueError(f"VisA MAGIC bulk category must stay {MAGIC_CATEGORIES}")
        empty = [item.name for item in inventory.types if not item.fewshot]
        if empty:
            raise ValueError(f"MAGIC generate refuses floor-zero types: {empty}")
    elif inventory.split_id == "mvtec_lexicographic_floor_third_v1":
        from cadiff.v06_c.mvtec_pilot import MVTEC_CATEGORIES

        if inventory.category not in MVTEC_CATEGORIES:
            raise ValueError(f"MVTec bulk category must stay {MVTEC_CATEGORIES}")
        empty = [item.name for item in inventory.types if not item.fewshot or not item.heldout]
        if empty:
            raise ValueError(f"MVTec generate refuses types without few-shot and held-out: {empty}")
    elif inventory.category not in EXPECTED:
        raise ValueError(f"VisA bulk dump category must stay {EXPECTED}")
    if pairs_per_type < 1:
        raise ValueError("pairs_per_type must be positive")
    if index_start < 0:
        raise ValueError("index_start must be non-negative")
    if index_start + pairs_per_type > TYPE_SEED_STRIDE:
        raise ValueError("bulk index range collides with the next defect-type seed block")
    normals = tuple(path.stem for path in inventory.normals)
    if not normals:
        raise ValueError("VisA bulk dump requires train/good targets")
    jobs: list[BulkJob] = []
    for type_index, item in enumerate(inventory.types):
        for index in range(index_start, index_start + pairs_per_type):
            seed = BASE_SEED + type_index * TYPE_SEED_STRIDE + index
            if seed < 0 or seed >= 2**32:
                raise ValueError("bulk seed must fit in uint32")
            target_id = normals[index % len(normals)]
            job_id = f"{item.slug}-{target_id}-{seed}"
            jobs.append(
                BulkJob(
                    index=index,
                    defect_type=item.name,
                    type_slug=item.slug,
                    target_id=target_id,
                    seed=seed,
                    job_id=job_id,
                )
            )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise RuntimeError("VisA bulk job IDs collided")
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


def _combined_large_cluster_areas(defect_type: str, areas: tuple[int, ...]) -> tuple[int, ...]:
    if defect_type != COMBINED_TYPE:
        return areas
    return large_cluster_fewshot_areas(areas)


def run_visa_place_bulk(
    *,
    inventory: VisaInventory,
    checkpoint: Path,
    model_root: Path,
    output: Path,
    device: str | None = None,
    pairs_per_type: int = PAIRS_PER_TYPE,
    index_start: int = 0,
    only_types: Sequence[str] | None = None,
    combined_large_cluster_area: bool = False,
    combined_large_cluster_source: bool = False,
    combined_mix_y_stamps: bool = False,
    a2_fill_backend: Any = None,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if summary_path.is_file():
        raise FileExistsError(f"refuses to overwrite completed VisA bulk dump: {output}")
    fewshot_records = fewshot_identity_records(inventory)
    normal_records = normal_identity_records(inventory)
    jobs = build_visa_place_bulk_jobs(
        inventory,
        pairs_per_type=pairs_per_type,
        index_start=index_start,
    )
    if only_types is not None:
        allowed = tuple(only_types)
        if not allowed:
            raise ValueError("only_types must be non-empty when provided")
        unknown = [name for name in allowed if name not in {item.name for item in inventory.types}]
        if unknown:
            raise ValueError(f"unknown VisA types for bulk filter: {unknown}")
        jobs = tuple(job for job in jobs if job.defect_type in allowed)
        if not jobs:
            raise ValueError("only_types removed every bulk job")
    area_transform = _combined_large_cluster_areas if combined_large_cluster_area else None
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
        item.name: tuple(
            load_visa_identity(record, dataset_root=inventory.source_root)
            for record in fewshot_records
            if record.defect_type == item.name
        )
        for item in inventory.types
    }
    if combined_large_cluster_source:
        if COMBINED_TYPE not in sources:
            raise ValueError("combined large-cluster source requires a combined type")
        sources = {
            **sources,
            COMBINED_TYPE: large_cluster_source_items(sources[COMBINED_TYPE]),
        }
    generator = A2ProposalGenerator(
        sources,
        config=inventory.structure,
        identity_manifest=None,
        encoder=encoder,
        fill_backend=a2_fill_backend,
    )
    r_config = visa_p2_a1_config(inventory.category, tuple(item.name for item in inventory.types))
    renderer = load_visa_renderer(
        checkpoint=checkpoint,
        model_root=model_root,
        config=r_config,
        device=device,
    )
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
            loaded = load_visa_identity(
                normals_by_id[job.target_id],
                dataset_root=inventory.source_root,
            )
            structure = estimate_normal_structure(loaded, config=inventory.structure)
            structures[job.target_id] = structure
        target = structures[job.target_id]
        sample_root.mkdir(parents=True, exist_ok=True)
        normal = np.ascontiguousarray(target.normal.rgb.copy())
        record = {
            "schema_version": "v06_c_visa_place_bulk_v1",
            "role": "synthetic_train",
            "job_id": job.job_id,
            "category": inventory.category,
            "defect_type": job.defect_type,
            "type_slug": job.type_slug,
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
            a2, placement, template = placement_from_official_a2(
                generator,
                target,
                job.defect_type,
                job.seed,
                fewshot_areas_transform=area_transform,
                min_shape_pixels=MIN_SHAPE_PIXELS,
                return_template=True,
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
            escaped_proposal = bool((placement.mask & ~proposal).any())
            if escaped_proposal and not getattr(placement, "used_site_fallback", False):
                raise ValueError("placed shape escaped a2")
            n_requested = 1
            n_stamps = 1
            second_pixels = 0
            second_failed = False
            second_reason = None
            second_used_clip = False
            if combined_mix_y_stamps and job.defect_type == COMBINED_TYPE:
                n_requested = mixed_stamp_count(job.seed)
                if n_requested == 2:
                    second = place_second_stamp(
                        template,
                        placement.site,
                        placement.mask,
                        a2.similarity,
                        proposal,
                        seed=job.seed,
                        min_shape_pixels=MIN_SHAPE_PIXELS,
                    )
                    second_failed = bool(second["failed"])
                    second_reason = second.get("failure_reason")
                    second_used_clip = bool(second.get("used_clip"))
                    if not second["failed"]:
                        union = np.ascontiguousarray(placement.mask | second["mask"], dtype=bool)
                        validate_realization(union, placement.site)
                        if (union & ~proposal).any() and not getattr(
                            placement, "used_site_fallback", False
                        ):
                            raise ValueError("placed shape escaped a2")
                        placement = replace(
                            placement,
                            mask=union,
                            placed_pixels=int(union.sum()),
                        )
                        n_stamps = 2
                        second_pixels = int(second["pixels"])
            raw = renderer(normal, job.defect_type, placement.mask.astype(np.uint8), job.seed)
            final_y = compose_causal(normal, raw, placement.mask)
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
                n_stamps_requested=n_requested,
                n_stamps=n_stamps,
                second_stamp_pixels=second_pixels,
                second_stamp_failed=second_failed,
                second_stamp_reason=second_reason,
                second_used_clip=second_used_clip,
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
        elif not record["failed"] and int(record.get("n_stamps") or 1) == 2:
            status = "OK Y2"
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
        item.name: sum(1 for row in kept if row["defect_type"] == item.name)
        for item in inventory.types
    }
    summary = {
        "schema_version": "v06_c_visa_place_bulk_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_module_c": True,
        "not_learned_h": True,
        "not_native_win": True,
        "method": "tokencut_a2_site_scale_place_then_independent_visa_r",
        "object_support_method": "tokencut",
        "h_method": "tokencut_site_scale_place",
        "category": inventory.category,
        "pairs_per_type_requested": pairs_per_type,
        "index_start": index_start,
        "only_types": list(only_types) if only_types is not None else None,
        "combined_large_cluster_area": bool(combined_large_cluster_area),
        "combined_large_cluster_source": bool(combined_large_cluster_source),
        "combined_mix_y_stamps": bool(combined_mix_y_stamps),
        "combined_source_count": len(sources.get(COMBINED_TYPE, ())),
        "combined_n_stamps_two": sum(int(row.get("n_stamps") or 1) == 2 for row in kept),
        "combined_n_stamps_requested_two": sum(
            int(row.get("n_stamps_requested") or 1) == 2 for row in records
        ),
        "not_official_h": bool(
            combined_large_cluster_area or combined_large_cluster_source or combined_mix_y_stamps
        ),
        "identity_count": len(records),
        "kept_rows": len(kept),
        "failed_rows": sum(row["failed"] for row in records),
        "floor_rows": sum(bool(row.get("used_proposal_floor")) for row in records),
        "fallback_rows": sum(bool(row.get("used_site_fallback")) for row in records),
        "per_type_kept": per_type,
        "targets": "visa_train_good",
        "normal_count": len(normal_records),
        "fewshot_count": len(fewshot_records),
        "renderer": "independent_visa_exact_mask_r",
        "ddim_steps": 50,
        "min_shape_pixels": MIN_SHAPE_PIXELS,
        "allow_proposal_floor": False,
        "destination": str(output),
        "split_id": inventory.split_id,
        "collapsed_from": list(inventory.collapsed_from or ()),
        "collapsed_type": COLLAPSED_TYPE if inventory.collapsed_from else None,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def pack_visa_eval_generated(
    dump: Path | Sequence[Path],
    layout: Path,
    category: str,
    *,
    include_slugs: Sequence[str] | None = None,
    exclude_slugs: Sequence[str] | None = None,
) -> Path:
    from collections import defaultdict

    dumps = (dump,) if isinstance(dump, (str, Path)) else tuple(dump)
    by_slug: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
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
            slug = row.get("type_slug") or slug_type(str(row["defect_type"]))
            by_slug[slug].append((root, row))
    if include_slugs is not None:
        allowed = {str(slug) for slug in include_slugs}
        by_slug = {slug: rows for slug, rows in by_slug.items() if slug in allowed}
    if exclude_slugs is not None:
        blocked = {str(slug) for slug in exclude_slugs}
        by_slug = {slug: rows for slug, rows in by_slug.items() if slug not in blocked}
    if not by_slug:
        raise ValueError("pack produced no generated types")
    layout.mkdir(parents=True, exist_ok=True)
    for slug, rows in by_slug.items():
        image_dir = layout / category / slug / "composed"
        mask_dir = layout / category / slug / "mask_defect"
        if image_dir.exists():
            rmtree(image_dir)
        if mask_dir.exists():
            rmtree(mask_dir)
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
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


def run_visa_place_bulk_from_cli(
    *,
    paths: Path,
    source_root: Path,
    category: str,
    checkpoint: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
    pairs_per_type: int = PAIRS_PER_TYPE,
    index_start: int = 0,
    collapse_type: str | None = None,
    split: str | None = None,
    only_types: Sequence[str] | None = None,
    combined_large_cluster_area: bool = False,
    combined_large_cluster_source: bool = False,
    combined_mix_y_stamps: bool = False,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    inventory = load_visa_inventory_for_split(source_root, category, split, collapse_type)
    return run_visa_place_bulk(
        inventory=inventory,
        checkpoint=checkpoint,
        model_root=project.model_root,
        output=output,
        device=device,
        pairs_per_type=pairs_per_type,
        index_start=index_start,
        only_types=only_types,
        combined_large_cluster_area=combined_large_cluster_area,
        combined_large_cluster_source=combined_large_cluster_source,
        combined_mix_y_stamps=combined_mix_y_stamps,
    )


__all__ = [
    "ALLOW_PROPOSAL_FLOOR",
    "BASE_SEED",
    "BulkJob",
    "MIN_SHAPE_PIXELS",
    "PAIRS_PER_TYPE",
    "TYPE_SEED_STRIDE",
    "build_visa_place_bulk_jobs",
    "large_cluster_fewshot_areas",
    "large_cluster_source_items",
    "mixed_stamp_count",
    "pack_visa_eval_generated",
    "run_visa_place_bulk",
    "run_visa_place_bulk_from_cli",
]

"""can native-window bulk dump: letterbox a2 + native placement H + native-window R.

Diagnostic generation. Not official generate() and not Module C.
Does not overwrite the letterbox-512 can 500-pair dump.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import A2ProposalGenerator, _material_proposal_maps
from cadiff.v06_a.structure import _estimate_support, estimate_normal_structure
from cadiff.v06_atlas_diag.features import FrozenDinoEncoder
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source
from cadiff.v06_b.a2_component_place import instantiate_on_component
from cadiff.v06_b.a2_shape_instantiate import EmptyFeasiblePlacement, _contained_output, _save_sheet
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_b.renderer import compose_causal
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
from cadiff.v06_c.can_native_win_r import load_can_native_win_renderer
from cadiff.v06_c.can_native_window import (
    WINDOW_MARGIN,
    WINDOW_SIZE,
    crop_to_hw,
    extract_window,
    geometry_for_rgb,
    letterbox_binary,
    letterbox_rgb,
    lift_binary_mask,
    lift_float_map,
    load_native_mask,
    load_native_rgb,
    pad_to_square,
    paste_window,
    window_from_mask,
)
from cadiff.v06_c.can_place_bulk import PAIRS, build_can_place_bulk_jobs, _job_done

PREVIEW_COLUMNS = ("normal", "proposal", "m_final", "final_y")
MIN_SHAPE_PIXELS = 8


def _layout_path(root: Path, relpath: str) -> Path:
    return root.joinpath(*PurePosixPath(relpath).parts)


def run_can_native_win_bulk(
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
        raise FileExistsError(f"refuses to overwrite completed can native-win dump: {output}")
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
    encoder = FrozenDinoEncoder(device=device)
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
    renderer = load_can_native_win_renderer(
        checkpoint=checkpoint, model_root=model_root, device=device
    )
    normals_by_id = {record.item_id: record for record in normal_records}
    structures: dict[str, Any] = {}
    target_feat: dict[str, Any] = {}
    native_rgb: dict[str, np.ndarray] = {}
    native_meta: dict[str, Any] = {}
    source_pack: dict[str, dict[str, Any]] = {}
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
            target_feat[job.target_id] = generator._encoder.encode(structure.normal.rgb)
            native = load_native_rgb(
                _layout_path(inventory.layout_root, normals_by_id[job.target_id].image_relpath)
            )
            native_rgb[job.target_id] = native
            native_meta[job.target_id] = geometry_for_rgb(native)
        target = structures[job.target_id]
        source = generator.select_source(job.defect_type, job.seed)
        assert source.anomaly_gt is not None
        source_id = source.record.item_id
        if source_id not in source_pack:
            inpainted = neighbor_fill_source(source.rgb, source.anomaly_gt, source.valid)
            inpainted_identity = replace(
                source,
                rgb=np.array(inpainted, copy=True),
                anomaly_gt=None,
                record=replace(source.record, role="normal_fit", defect_type="good"),
            )
            if source.record.mask_relpath is None:
                raise ValueError(f"few-shot source missing mask: {source_id}")
            source_native = load_native_rgb(
                _layout_path(inventory.layout_root, source.record.image_relpath)
            )
            source_gt_native = load_native_mask(
                _layout_path(inventory.layout_root, source.record.mask_relpath),
                shape=source_native.shape[:2],
            )
            source_pack[source_id] = {
                "source": source,
                "inpainted": np.ascontiguousarray(inpainted),
                "support": _estimate_support(inpainted_identity, STRUCTURE_CONFIG),
                "feat": generator._encoder.encode(inpainted),
                "native_gt": source_gt_native,
            }
        packed = source_pack[source_id]
        proposal, tau, similarity = _material_proposal_maps(
            source.anomaly_gt,
            source.valid,
            target,
            source_feat=packed["feat"],
            target_feat=target_feat[job.target_id],
        )
        sample_root.mkdir(parents=True, exist_ok=True)
        native = np.ascontiguousarray(native_rgb[job.target_id])
        meta = native_meta[job.target_id]
        letterbox_normal = np.ascontiguousarray(target.normal.rgb.copy())
        record = {
            "schema_version": "v06_c_can_native_win_bulk_v1",
            "role": "synthetic_train",
            "job_id": job.job_id,
            "category": CATEGORY,
            "defect_type": job.defect_type,
            "target_id": job.target_id,
            "target_role": "normal_fit",
            "source_id": source_id,
            "seed": job.seed,
            "index": job.index,
            "a2_tau": tau,
            "proposal_pixels_letterbox": int(proposal.sum()),
            "failed": False,
            "failure_reason": None,
            "used_site_fallback": False,
            "used_proposal_floor": False,
            "placed_pixels": 0,
            "image_relpath": f"images/{job.job_id}.png",
            "mask_relpath": f"masks/{job.job_id}.png",
            "geometry": "native_window",
            "window_margin": WINDOW_MARGIN,
            "window_size": WINDOW_SIZE,
        }
        arrays = {
            "normal": letterbox_normal,
            "proposal": (proposal.astype(np.uint8) * 255),
            "m_final": np.zeros(letterbox_normal.shape[:2], dtype=np.uint8),
            "final_y": np.zeros_like(letterbox_normal),
        }
        try:
            if (
                not proposal.any()
                or bool(np.all(proposal[target.valid]))
                or bool(np.any(proposal & ~target.valid))
            ):
                raise EmptyFeasiblePlacement("a2 proposal is empty, full within V, or outside V_x")
            proposal_native = lift_binary_mask(proposal, meta)
            similarity_native = lift_float_map(similarity, meta)
            if not proposal_native.any():
                raise EmptyFeasiblePlacement("lifted a2 proposal is empty")
            source_gt_native = packed["native_gt"]
            if source_gt_native.shape != native.shape[:2]:
                raise ValueError("few-shot GT and target native frames differ")
            gt_sq, native_h, native_w = pad_to_square(source_gt_native, False)
            proposal_sq, _, _ = pad_to_square(proposal_native, False)
            similarity_sq, _, _ = pad_to_square(similarity_native, 0.0)
            _template, placement = instantiate_on_component(
                gt_sq,
                packed["support"],
                target.support & target.valid,
                proposal_sq,
                similarity_sq,
                seed=job.seed,
                min_shape_pixels=MIN_SHAPE_PIXELS,
                allow_proposal_floor=False,
            )
            placed_native = crop_to_hw(placement.mask, native_h, native_w)
            site_native = crop_to_hw(placement.site, native_h, native_w)
            validate_realization(placed_native, proposal_native)
            if (placed_native & ~proposal_native).any():
                raise ValueError("placed shape escaped lifted a2")
            window = window_from_mask(placed_native, margin=WINDOW_MARGIN, size=WINDOW_SIZE)
            crop_rgb, crop_mask = extract_window(native, placed_native, window)
            raw_crop = renderer(crop_rgb, job.defect_type, crop_mask, job.seed)
            pasted = paste_window(native, raw_crop, window)
            final_y = compose_causal(native, pasted, placed_native)
            if not np.array_equal(final_y[~placed_native], native[~placed_native]):
                raise ValueError("renderer changed pixels outside m_final")
            y_letterbox, y_meta = letterbox_rgb(final_y, 512)
            arrays["m_final"] = letterbox_binary(placed_native, y_meta).astype(np.uint8) * 255
            arrays["final_y"] = y_letterbox
            record.update(
                used_site_fallback=placement.used_site_fallback,
                used_proposal_floor=False,
                placed_pixels=int(placed_native.sum()),
                site_pixels=int(site_native.sum()),
                proposal_pixels_native=int(proposal_native.sum()),
                scale=placement.scale,
                angle_deg=placement.angle_deg,
                window=window.as_record(),
                native_height=native_h,
                native_width=native_w,
            )
            Image.fromarray(final_y).save(sample_root / "final_y.png")
            Image.fromarray(placed_native.astype(np.uint8) * 255).save(sample_root / "m_final.png")
            Image.fromarray(crop_rgb).save(sample_root / "window_x.png")
            Image.fromarray(raw_crop).save(sample_root / "window_r.png")
            Image.fromarray(crop_mask.astype(np.uint8) * 255).save(sample_root / "window_m.png")
            Image.fromarray(final_y).save(output / record["image_relpath"])
            Image.fromarray(placed_native.astype(np.uint8) * 255).save(
                output / record["mask_relpath"]
            )
        except (EmptyFeasiblePlacement, ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
            Image.fromarray(native).save(sample_root / "final_y.png")
            Image.fromarray(np.zeros(native.shape[:2], dtype=np.uint8)).save(
                sample_root / "m_final.png"
            )
        Image.fromarray(native).save(sample_root / "normal.png")
        Image.fromarray(arrays["normal"]).save(sample_root / "normal_letterbox.png")
        Image.fromarray(arrays["proposal"]).save(sample_root / "proposal_letterbox.png")
        Image.fromarray(arrays["m_final"]).save(sample_root / "m_final_letterbox.png")
        Image.fromarray(arrays["final_y"]).save(sample_root / "final_y_letterbox.png")
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
        "schema_version": "v06_c_can_native_win_bulk_v1",
        "diagnostic_only": True,
        "not_official_generate": True,
        "not_module_c": True,
        "not_learned_h": True,
        "method": "letterbox_a2_native_place_native_window_r",
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
        "renderer": "independent_can_native_win_r",
        "ddim_steps": 50,
        "min_shape_pixels": MIN_SHAPE_PIXELS,
        "allow_proposal_floor": False,
        "window_margin": WINDOW_MARGIN,
        "window_size": WINDOW_SIZE,
        "destination": str(output),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def run_can_native_win_bulk_from_cli(
    *,
    paths: Path,
    source_root: Path,
    layout_root: Path,
    checkpoint: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    inventory = stage_can_mvtec_layout(source_root=source_root, layout_root=layout_root)
    return run_can_native_win_bulk(
        inventory=inventory,
        checkpoint=checkpoint,
        model_root=project.model_root,
        output=output,
        device=device,
    )


__all__ = [
    "run_can_native_win_bulk",
    "run_can_native_win_bulk_from_cli",
]

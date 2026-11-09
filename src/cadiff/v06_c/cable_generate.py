"""Cable a2 A -> official placement H -> independently trained Cable R."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from cadiff.config import load_config
from cadiff.p2_a1.config import load_p2_a1_config
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import load_v06_a_identity_manifest
from cadiff.v06_a.structure import estimate_normal_structure
from cadiff.v06_a.a2 import load_a2_generator
from cadiff.v06_atlas_diag.cable_a import _lookup_p1_target, build_cable_a_slots
from cadiff.v06_b.config import is_cable_scheme2, load_v06_b_config
from cadiff.v06_b.mask_first_panel import _load_panel_runtime
from cadiff.v06_b.mask_first_runner import MaskFirstV06BContext, load_mask_first_source_prompts
from cadiff.v06_b.source import load_source_renderer_manifest
from cadiff.v06_c.generate import (
    _OFFICIAL_H_METHOD,
    _compose_from_proposal,
    _proposal_from_generator,
)

COLUMNS = ("normal", "proposal", "m_final", "final_y")


def load_cable_generate_config(path: Path, *, repository_root: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != "v06_c_cable_generate":
        raise ValueError("Cable generate requires its own schema")
    if data.get("category") != "cable" or "rows" in data:
        raise ValueError("Cable category and frozen identities are required")
    for key in ("experiment", "paths", "b_experiment"):
        rel = PurePosixPath(data[key])
        if rel.is_absolute() or ".." in rel.parts or not (repository_root / str(rel)).is_file():
            raise ValueError(f"invalid repository config: {key}")
    return data


def cable_generate_output(root: Path, relpath: str) -> Path:
    rel = PurePosixPath(relpath)
    if rel.parts[:3] != ("v06_c", "cable", "generate") or len(rel.parts) != 4 or ".." in rel.parts:
        raise ValueError("Cable generation must use v06_c/cable/generate/<run>")
    dest = root.resolve(strict=True).joinpath(*rel.parts)
    dest.resolve().relative_to(root.resolve())
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("refuses to overwrite Cable generation")
    return dest


def load_cable_runtime(config: dict, repository_root: Path, device: str | None = None):
    paths = load_config(repository_root / config["paths"])
    b = load_v06_b_config(repository_root / config["b_experiment"])
    if b.category != "cable":
        raise ValueError("Cable proposal requires independently trained Cable B")
    output = paths.runs_root / b.output_relpath
    summary = json.loads((output / "training_summary.json").read_text())
    if (
        summary["planner"]["steps"] != b.planner_training_steps
        or summary["renderer"]["steps"] != b.renderer_training_steps
    ):
        raise ValueError("Cable B training is incomplete")
    ctx = MaskFirstV06BContext(
        paths=paths,
        experiment=b,
        source_model=load_p2_a1_config(repository_root / b.source_model_config),
        source_prompts=load_mask_first_source_prompts(
            repository_root / b.source_experiment_config, category="cable"
        ),
        manifest=Path(config["manifest"]),
        output=output,
        cwd=repository_root,
    )
    source = load_source_renderer_manifest(
        repository_root / b.source_renderer_manifest,
        runs_root=paths.runs_root,
        category="cable" if is_cable_scheme2(b) else "hazelnut",
    )
    runtime = _load_panel_runtime(
        SimpleNamespace(context=ctx, source_renderer=source), torch.device(device or "cuda:0")
    )
    if runtime.inference.defect_types != b.defect_types or runtime.inference.ddim_steps != 50:
        raise ValueError("Cable inference vocabulary or full sampler mismatch")
    return runtime


def _panel(path: Path, rows: list, *, tile: int = 256) -> None:
    canvas = Image.new("RGB", (tile * 4, 28 + len(rows) * (tile + 22)), "white")
    draw = ImageDraw.Draw(canvas)
    for j, name in enumerate(COLUMNS):
        draw.text((j * tile + 5, 7), name, fill="black")
    for i, (label, arrays) in enumerate(rows):
        top = 28 + i * (tile + 22)
        draw.text((5, top + 3), label, fill="black")
        for j, name in enumerate(COLUMNS):
            a = arrays[name]
            img = (
                Image.fromarray(a)
                .convert("RGB")
                .resize(
                    (tile, tile),
                    Image.Resampling.NEAREST if a.ndim == 2 else Image.Resampling.LANCZOS,
                )
            )
            canvas.paste(img, (j * tile, top + 22))
    canvas.save(path)


def run_cable_generate_panel(
    config: dict,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
    device: str | None = None,
    atlas_generator=None,
    runtime=None,
    load_target=load_identity_record,
    estimate_structure=estimate_normal_structure,
) -> dict:
    destination = cable_generate_output(output_root, output_relpath)
    experiment = load_v06_a_experiment(repository_root / config["experiment"])
    manifest = load_v06_a_identity_manifest(
        repository_root / experiment.identity_manifest, experiment=experiment
    )
    slots = build_cable_a_slots(manifest, experiment=experiment)
    if runtime is None:
        runtime = load_cable_runtime(config, repository_root, device)
    if (
        tuple(runtime.inference.defect_types) != experiment.defect_types
        or runtime.inference.ddim_steps != 50
    ):
        raise ValueError("Cable B vocabulary and 50-step sampler required")
    if atlas_generator is None:
        atlas_generator = load_a2_generator(
            experiment_path=repository_root / config["experiment"],
            dataset_root=dataset_root,
            device=device,
        )
    destination.mkdir(parents=True)
    records, panels, structures = [], [], {}
    per_type = {kind: [] for kind in experiment.defect_types}
    for slot in slots:
        if slot.normal_id not in structures:
            target = load_target(
                _lookup_p1_target(manifest, slot.normal_id),
                dataset_root=dataset_root,
                config=experiment,
            )
            structures[slot.normal_id] = estimate_structure(target, config=experiment)
        structure = structures[slot.normal_id]
        normal = np.ascontiguousarray(structure.normal.rgb.copy())
        identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        sample_root = destination / identity
        sample_root.mkdir()
        failure = np.zeros_like(normal)
        failure[..., 0] = 160
        arrays = {
            "normal": normal,
            "proposal": failure.copy(),
            "m_final": failure.copy(),
            "raw_r": failure.copy(),
            "final_y": failure.copy(),
        }
        record = dict(
            identity=identity,
            category="cable",
            defect_type=slot.defect_type,
            target_id=slot.normal_id,
            target_role="normal_dev",
            source_id=slot.source_id,
            seed=slot.seed,
            failed=False,
            failure_reason=None,
            candidate_count=1,
            retried=False,
            clipped_to_support=False,
            realization_threshold=None,
            ddim_steps=50,
            h_method=_OFFICIAL_H_METHOD,
            used_learned_planner=False,
            used_site_fallback=None,
            used_proposal_floor=None,
        )
        try:
            proposal, extras, maps = _proposal_from_generator(
                atlas_generator, structure, slot.defect_type, slot.seed
            )
            record.update(extras)
            arrays["proposal"] = proposal.astype(np.uint8) * 255
            result, raw, h_meta = _compose_from_proposal(
                structure,
                slot.defect_type,
                slot.seed,
                proposal=proposal,
                maps=maps,
                module_b=runtime.module_b,
            )
            mask = result.anomaly_mask
            if not mask.any() or mask.all():
                raise ValueError("Cable realization mask is empty or full")
            subset = bool(not (mask & ~proposal).any())
            unchanged = bool(np.array_equal(result.anomaly_image[~mask], normal[~mask]))
            if not subset or not unchanged:
                raise ValueError("Cable spatial or exterior RGB contract failed")
            arrays.update(
                m_final=mask.astype(np.uint8) * 255, raw_r=raw, final_y=result.anomaly_image
            )
            record.update(h_meta)
            record.update(
                proposal_pixels=int(proposal.sum()),
                m_final_pixels=int(mask.sum()),
                m_final_subset_of_proposal=subset,
                outside_rgb_unchanged=unchanged,
            )
        except (ValueError, RuntimeError) as error:
            record.update(failed=True, failure_reason=str(error))
        for name, array in arrays.items():
            Image.fromarray(array).save(sample_root / (name + ".png"))
        (sample_root / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        label = f"{slot.defect_type} normal={slot.normal_id} seed={slot.seed}" + (
            " FAIL" if record["failed"] else ""
        )
        panels.append((label, arrays))
        per_type[slot.defect_type].append((label, arrays))
        print(
            identity,
            "FAIL " + str(record["failure_reason"]) if record["failed"] else "OK",
            flush=True,
        )
    _panel(destination / "atlas_generate_cable.png", panels)
    for kind, rows in per_type.items():
        _panel(destination / f"panel_{kind}.png", rows)
    (destination / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = dict(
        schema_version="v06_c_cable_generate",
        category="cable",
        status="awaiting_human_review",
        identity_count=len(records),
        failed_rows=sum(r["failed"] for r in records),
        floor_rows=sum(r.get("used_proposal_floor") is True for r in records),
        fallback_rows=sum(r.get("used_site_fallback") is True for r in records),
        h_method=_OFFICIAL_H_METHOD,
        used_learned_planner=False,
        realization_threshold=None,
        ddim_steps=50,
        candidate_count=1,
        retried=False,
        target_role="normal_dev",
        not_unseen_evaluation=True,
        visible_defects_passed=None,
        not_module_c_downstream_complete=True,
        b_experiment=config["b_experiment"],
        destination=str(destination),
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

"""Cable-only atlas A proposal panel. Does not train or call Module B."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import LoadedIdentity, load_identity_record
from cadiff.v06_a.identities import IdentityManifest, IdentityRecord, load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, estimate_normal_structure

from .atlas_a import AtlasAResult, load_atlas_a_generator
from .panel import _CELL, _overlay_mask, _to_image

_OUTPUT_PREFIX = ("v06_a", "diag_atlas_prop")
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")
_PRESSURE_TYPES = frozenset({"cut_inner_insulation", "cut_outer_insulation"})
_CABLE_TYPES = (
    "bent_wire",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "poke_insulation",
)
_P1_NORMAL_IDS = ("000", "071", "152", "223")
_ATLAS_COLOR = (80, 160, 255)
REVIEW_COLUMNS = ("source", "source_gt", "target", "proposal")


@dataclass(frozen=True)
class CableAConfig:
    schema_version: str
    status: str
    category: str
    experiment: str
    rows: tuple[()] = ()


@dataclass(frozen=True)
class CableASlot:
    category: str
    defect_type: str
    normal_id: str
    seed: int
    source_id: str
    target_role: str
    review_role: str


def build_cable_a_slots(manifest: IdentityManifest, *, experiment: Any) -> tuple[CableASlot, ...]:
    if manifest.category != "cable" or experiment.category != "cable":
        raise ValueError("Cable A panel is Cable-only")
    if manifest.defect_types != _CABLE_TYPES or experiment.defect_types != _CABLE_TYPES:
        raise ValueError("Cable A panel requires the frozen Cable vocabulary")
    if len(manifest.p1_identities) != 32:
        raise ValueError("Cable A panel requires the frozen 32-identity P1 schedule")
    if tuple(record.item_id for record in manifest.p1_normals) != _P1_NORMAL_IDS:
        raise ValueError("Cable A P1 normals must be the frozen 000/071/152/223 set")
    slots: list[CableASlot] = []
    for identity in manifest.p1_identities:
        if identity.phase != "p1":
            raise ValueError("Cable A slots must come from the frozen p1 schedule")
        sources = tuple(
            sorted(
                (record for record in manifest.fewshot if record.defect_type == identity.defect_type),
                key=lambda record: record.sample_id,
            )
        )
        if not sources:
            raise ValueError(f"missing frozen fewshot sources for {identity.defect_type}")
        source = sources[identity.seed % len(sources)]
        slots.append(
            CableASlot(
                category="cable",
                defect_type=identity.defect_type,
                normal_id=identity.normal_id,
                seed=identity.seed,
                source_id=source.item_id,
                target_role="normal_dev",
                review_role="pressure" if identity.defect_type in _PRESSURE_TYPES else "primary",
            )
        )
    if len(slots) != 32:
        raise ValueError("Cable A panel freezes exactly 32 identities")
    return tuple(slots)


def load_cable_a_config(path: Path, *, repository_root: Path) -> CableAConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or str(loaded.get("schema_version")) != "v06_a_diag_atlas_prop_cable_a":
        raise ValueError("Cable A panel requires schema v06_a_diag_atlas_prop_cable_a")
    if "rows" in loaded:
        raise ValueError("Cable A rows must be built from the frozen manifest, not listed in yaml")
    if str(loaded.get("category")) != "cable":
        raise ValueError("Cable A panel is Cable-only")
    relative = loaded.get("experiment")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Cable A experiment must be a repository-relative path")
    resolved = repository_root.joinpath(*PurePosixPath(relative).parts)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("Cable A experiment must be a regular file")
    return CableAConfig(
        schema_version="v06_a_diag_atlas_prop_cable_a",
        status=str(loaded["status"]),
        category="cable",
        experiment=relative,
    )


def cable_a_output_path(output_root: Path, output_relpath: str) -> Path:
    parts = PurePosixPath(output_relpath).parts
    if parts[:2] != _OUTPUT_PREFIX or len(parts) != 3:
        raise ValueError("Cable A output must live under v06_a/diag_atlas_prop/<name>")
    name = parts[2]
    if name.startswith(_FORBIDDEN_OUTPUT_PREFIXES):
        raise ValueError(f"refuses to write f1-f4 outputs: {name}")
    root = output_root.resolve(strict=True)
    destination = root.joinpath(*parts)
    destination.resolve(strict=False).relative_to(root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {destination}")
    return destination


def _lookup_p1_target(manifest: IdentityManifest, normal_id: str) -> IdentityRecord:
    matches = [record for record in manifest.p1_normals if record.item_id == normal_id]
    if len(matches) != 1 or matches[0].role != "normal_dev":
        raise ValueError("Cable A target must be the exact frozen normal_dev")
    return matches[0]


def _lookup_source(manifest: IdentityManifest, slot: CableASlot) -> IdentityRecord:
    matches = [
        record
        for record in manifest.fewshot
        if record.defect_type == slot.defect_type and record.item_id == slot.source_id
    ]
    if len(matches) != 1:
        raise ValueError("Cable A source must be the exact frozen fewshot")
    return matches[0]


def _write_panel(row_images: list[Image.Image], destination: Path) -> None:
    width = row_images[0].width
    header_height = 22
    canvas = Image.new(
        "RGB",
        (width, header_height + sum(image.height for image in row_images)),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 4), "  ".join(REVIEW_COLUMNS), fill=(200, 200, 200), font=font)
    top = header_height
    for image in row_images:
        canvas.paste(image, (0, top))
        top += image.height
    canvas.save(destination)


def _render_row(
    *,
    source: LoadedIdentity,
    target: LoadedIdentity,
    proposal: np.ndarray,
    title: str,
) -> Image.Image:
    if source.anomaly_gt is None:
        raise ValueError("Cable A source requires a ground-truth mask")
    cells = [
        _to_image(source.rgb),
        _to_image(_overlay_mask(source.rgb, source.anomaly_gt, (255, 220, 40))),
        _to_image(target.rgb),
        _to_image(_overlay_mask(target.rgb, proposal, _ATLAS_COLOR)),
    ]
    canvas = Image.new("RGB", (_CELL * 4, _CELL + 28), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 4), title, fill=(240, 240, 240), font=font)
    for index, cell in enumerate(cells):
        canvas.paste(cell, (index * _CELL, 28))
    return canvas


def run_cable_a_panel(
    config: CableAConfig,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
    generator: Any | None = None,
    load_target: Callable[..., LoadedIdentity] | None = None,
    load_source: Callable[..., LoadedIdentity] | None = None,
    estimate_structure: Callable[..., NormalStructure] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if config.category != "cable":
        raise ValueError("Cable A panel is Cable-only")
    destination = cable_a_output_path(output_root, output_relpath)
    experiment_path = repository_root.joinpath(*PurePosixPath(config.experiment).parts)
    experiment = load_v06_a_experiment(experiment_path)
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    slots = build_cable_a_slots(manifest, experiment=experiment)
    if generator is None:
        generator = load_atlas_a_generator(
            experiment_path=experiment_path,
            dataset_root=dataset_root,
            device=device,
        )
    if load_target is None:
        load_target = load_identity_record
    if load_source is None:
        load_source = load_identity_record
    if estimate_structure is None:
        estimate_structure = estimate_normal_structure

    destination.mkdir(parents=True)
    rows_dir = destination / "rows"
    rows_dir.mkdir()
    structures: dict[str, NormalStructure] = {}
    sidecar: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    for index, slot in enumerate(slots):
        if slot.normal_id not in structures:
            target_record = _lookup_p1_target(manifest, slot.normal_id)
            target = load_target(target_record, dataset_root=dataset_root, config=experiment)
            structures[slot.normal_id] = estimate_structure(target, config=experiment)
        structure = structures[slot.normal_id]
        source = load_source(
            _lookup_source(manifest, slot),
            dataset_root=dataset_root,
            config=experiment,
        )
        result = generator.generate(structure, slot.defect_type, slot.seed)
        if not isinstance(result, AtlasAResult):
            raise TypeError("Cable A panel requires AtlasAResult")
        if result.source_id != slot.source_id:
            raise ValueError("proposal source must follow seed % n")
        status = "FALLBACK" if result.used_geometry_fallback else "PROP"
        title = (
            f"{index:02d} cable/{slot.defect_type} "
            f"src={slot.source_id} tgt={slot.normal_id} seed={slot.seed} "
            f"{status} {slot.review_role}"
        )
        image = _render_row(
            source=source,
            target=structure.normal,
            proposal=result.proposal,
            title=title,
        )
        image.save(
            rows_dir / f"{index:02d}_{slot.defect_type}_{slot.normal_id}_{slot.seed}.png"
        )
        images.append(image)
        sidecar.append(
            {
                "index": index,
                "category": "cable",
                "defect_type": slot.defect_type,
                "source_id": slot.source_id,
                "target_id": slot.normal_id,
                "target_role": "normal_dev",
                "seed": slot.seed,
                "review_role": slot.review_role,
                "used_geometry_fallback": result.used_geometry_fallback,
                "collapsed": result.collapsed,
                "collapse_reason": result.collapse_reason,
                "proposal_pixels": int(result.proposal.sum()),
                "source_selected_by_seed_mod_n": True,
            }
        )
        print(title, flush=True)

    panel = destination / "atlas_prop_cable_a_32.png"
    _write_panel(images, panel)
    summary = {
        "schema_version": config.schema_version,
        "status": "awaiting_human_review",
        "category": "cable",
        "identity_count": 32,
        "look_at": "blue proposal column; cut_inner/cut_outer are pressure only",
        "destination": str(destination),
        "panel": str(panel),
        "fallback_rows": sum(1 for row in sidecar if row["used_geometry_fallback"]),
        "collapsed_rows": sum(1 for row in sidecar if row["collapsed"]),
        "rows": sidecar,
        "not_module_a": True,
        "not_module_a_complete": True,
        "not_module_b": True,
        "did_not_train_or_call_b": True,
        "did_not_overwrite_f1_f4": True,
        "used_open_report_or_test": False,
    }
    (destination / "summary.json").write_text(
        json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sidecar),
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cable-only atlas A panel. Does not call Module B.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_cable_a_panel(
        load_cable_a_config(arguments.config, repository_root=arguments.repository_root),
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        output_relpath=arguments.output_relpath,
        device=arguments.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

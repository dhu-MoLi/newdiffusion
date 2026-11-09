"""3CAD atlas A panels. Isolation only; does not train or call Module B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from cadiff.v06_a.identities import IdentityRecord
from cadiff.v06_a.structure import estimate_normal_structure

from .atlas_a import AtlasProposalGenerator
from .extra_a import (
    ExtraACategory,
    ExtraASlot,
    ExtraAStructureConfig,
    _NORMAL_COUNT,
    _PANEL_SEEDS,
    _even_ids,
    _fewshot_ids,
    _item_id,
    _record,
    _render_row,
    _slug,
    _write_panel,
    extra_a_output_path,
    load_extra_identity,
)
from .features import FrozenDinoEncoder

_MAX_TYPES = 4
_MIXED_TYPE = "Multiple-defects"
_EXPECTED_CAD3 = (
    "Aluminum_Camera_Cover",
    "Aluminum_Ipad",
    "Aluminum_Middle_Frame",
)


def cad3_a_output_path(output_root: Path, output_relpath: str) -> Path:
    return extra_a_output_path(output_root, output_relpath)


def select_cad3_categories(cad3_root: Path) -> tuple[str, ...]:
    names = tuple(
        sorted(
            path.name
            for path in cad3_root.iterdir()
            if path.is_dir()
            and (path / "train" / "good").is_dir()
            and (path / "test").is_dir()
            and (path / "ground_truth").is_dir()
        )
    )
    selected = names[:3]
    if selected != _EXPECTED_CAD3:
        raise ValueError(f"3CAD atlas-A selection must stay {_EXPECTED_CAD3}, got {selected}")
    return selected


def inventory_cad3(cad3_root: Path, category: str) -> ExtraACategory:
    test_root = cad3_root / category / "test"
    types = tuple(
        sorted(
            path.name
            for path in test_root.iterdir()
            if path.is_dir() and path.name not in {"good", _MIXED_TYPE}
        )
    )[:_MAX_TYPES]
    if not types:
        raise ValueError(f"3CAD {category} has no single-defect types")
    fewshot: dict[str, tuple[IdentityRecord, ...]] = {}
    for defect_type in types:
        images = tuple(sorted((test_root / defect_type).glob("*.png")))
        ids = tuple(_item_id(path) for path in images)
        chosen = _fewshot_ids(ids)
        fewshot[defect_type] = tuple(
            _record(
                category=category,
                defect_type=defect_type,
                item_id=item_id,
                image_relpath=f"{category}/test/{defect_type}/{item_id}.png",
                mask_relpath=f"{category}/ground_truth/{defect_type}/{item_id}.png",
                role="fewshot",
            )
            for item_id in chosen
        )
    normals = tuple(sorted((cad3_root / category / "train" / "good").glob("*.png")))
    normal_ids = _even_ids(tuple(_item_id(path) for path in normals), _NORMAL_COUNT)
    return ExtraACategory(
        dataset="cad3",
        name=category,
        root=cad3_root,
        defect_types=types,
        fewshot=fewshot,
        normals=tuple(
            _record(
                category=category,
                defect_type="good",
                item_id=item_id,
                image_relpath=f"{category}/train/good/{item_id}.png",
                mask_relpath=None,
                role="normal_dev",
            )
            for item_id in normal_ids
        ),
    )


def build_cad3_a_slots(pack: ExtraACategory) -> tuple[ExtraASlot, ...]:
    slots: list[ExtraASlot] = []
    for defect_type in pack.defect_types:
        sources = pack.fewshot[defect_type]
        for normal in pack.normals:
            for seed in _PANEL_SEEDS:
                source = sources[seed % len(sources)]
                slots.append(
                    ExtraASlot(
                        dataset=pack.dataset,
                        category=pack.name,
                        defect_type=defect_type,
                        normal_id=normal.item_id,
                        source_id=source.item_id,
                        seed=seed,
                    )
                )
    return tuple(slots)


def _local_seed(sources: tuple[Any, ...], source_id: str) -> int:
    for index, item in enumerate(sources):
        if item.record.item_id == source_id:
            return index
    raise ValueError(f"missing loaded 3CAD source {source_id}")


def _run_category(
    pack: ExtraACategory,
    *,
    destination: Path,
    encoder: FrozenDinoEncoder,
) -> dict[str, Any]:
    config = ExtraAStructureConfig(category=pack.name, defect_types=pack.defect_types)
    slots = build_cad3_a_slots(pack)
    needed = {(slot.defect_type, slot.source_id) for slot in slots}
    loaded_by_key = {}
    for defect_type, records in pack.fewshot.items():
        for record in records:
            if (defect_type, record.item_id) in needed:
                loaded_by_key[(defect_type, record.item_id)] = load_extra_identity(
                    record, dataset_root=pack.root
                )
    sources = {
        defect_type: tuple(
            loaded_by_key[(defect_type, record.item_id)]
            for record in records
            if (defect_type, record.item_id) in needed
        )
        for defect_type, records in pack.fewshot.items()
    }
    generator = AtlasProposalGenerator(
        sources,
        config=config,
        identity_manifest=None,
        encoder=encoder,
    )
    structures = {
        record.item_id: estimate_normal_structure(
            load_extra_identity(record, dataset_root=pack.root),
            config=config,
        )
        for record in pack.normals
    }
    source_by_key = loaded_by_key
    rows_dir = destination / pack.dataset / _slug(pack.name) / "rows"
    rows_dir.mkdir(parents=True)
    images = []
    sidecar = []
    for index, slot in enumerate(slots):
        structure = structures[slot.normal_id]
        source = source_by_key[(slot.defect_type, slot.source_id)]
        try:
            result = generator.generate(
                structure,
                slot.defect_type,
                _local_seed(sources[slot.defect_type], slot.source_id),
            )
            if result.source_id != slot.source_id:
                raise ValueError("proposal source must follow seed % n")
            proposal = result.proposal
            status = "FALLBACK" if result.used_geometry_fallback else "PROP"
            used_fallback = result.used_geometry_fallback
            collapsed = result.collapsed
            failure = None
        except ValueError as error:
            proposal = np.zeros(structure.valid.shape, dtype=bool)
            status = "FAIL"
            used_fallback = None
            collapsed = None
            failure = str(error)
        title = (
            f"{index:02d} cad3/{pack.name}/{slot.defect_type} "
            f"src={slot.source_id} tgt={slot.normal_id} seed={slot.seed} {status}"
        )
        image = _render_row(
            source=source,
            target=structure.normal,
            proposal=proposal,
            title=title,
        )
        image.save(rows_dir / f"{index:02d}_{_slug(slot.defect_type)}_{slot.normal_id}_{slot.seed}.png")
        images.append(image)
        sidecar.append(
            {
                "index": index,
                "dataset": pack.dataset,
                "category": pack.name,
                "defect_type": slot.defect_type,
                "source_id": slot.source_id,
                "target_id": slot.normal_id,
                "seed": slot.seed,
                "used_geometry_fallback": used_fallback,
                "collapsed": collapsed,
                "proposal_pixels": int(proposal.sum()),
                "failed": failure is not None,
                "failure_reason": failure,
            }
        )
        print(title, flush=True)
    panel = destination / pack.dataset / _slug(pack.name) / f"atlas_prop_cad3_{_slug(pack.name)}.png"
    _write_panel(images, panel)
    return {
        "dataset": pack.dataset,
        "category": pack.name,
        "defect_types": list(pack.defect_types),
        "identity_count": len(slots),
        "fallback_rows": sum(1 for row in sidecar if row["used_geometry_fallback"]),
        "failed_rows": sum(1 for row in sidecar if row["failed"]),
        "panel": str(panel),
        "rows": sidecar,
    }


def run_cad3_a(
    *,
    cad3_root: Path,
    output_root: Path,
    output_relpath: str,
    device: str | None = None,
) -> dict[str, Any]:
    destination = cad3_a_output_path(output_root, output_relpath)
    packs = [inventory_cad3(cad3_root, name) for name in select_cad3_categories(cad3_root)]
    destination.mkdir(parents=True)
    encoder = FrozenDinoEncoder(device=device)
    summaries = []
    all_rows = []
    for pack in packs:
        summary = _run_category(pack, destination=destination, encoder=encoder)
        summaries.append({key: value for key, value in summary.items() if key != "rows"})
        all_rows.extend(summary["rows"])
    payload = {
        "schema_version": "v06_a_diag_atlas_prop_cad3_a",
        "status": "awaiting_human_review",
        "look_at": "blue proposal column",
        "cad3": list(_EXPECTED_CAD3),
        "selection": {
            "categories": "3CAD folders with train/good+test+ground_truth, lex 3",
            "types": "exclude good and Multiple-defects, lex 4",
            "fewshot": "floor(n/3) by lexical id; only seed-selected sources are loaded",
            "normals": "4 evenly spaced train/good ids",
            "seeds": list(_PANEL_SEEDS),
        },
        "categories": summaries,
        "identity_count": len(all_rows),
        "fallback_rows": sum(1 for row in all_rows if row["used_geometry_fallback"]),
        "failed_rows": sum(1 for row in all_rows if row.get("failed")),
        "destination": str(destination),
        "not_module_a_complete": True,
        "not_module_b": True,
        "did_not_overwrite_f1_f4": True,
    }
    (destination / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    payload["rows"] = all_rows
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3CAD atlas A panels. Not Module B.")
    parser.add_argument("--cad3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_cad3_a(
        cad3_root=arguments.cad3_root,
        output_root=arguments.output_root,
        output_relpath=arguments.output_relpath,
        device=arguments.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

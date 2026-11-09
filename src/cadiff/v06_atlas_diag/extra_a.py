"""Extra-category atlas A panels. Isolation only; does not train or call Module B."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from cadiff.data.geometry import resize_pad_image, resize_pad_mask
from cadiff.v06_a.data import LoadedIdentity
from cadiff.v06_a.identities import IdentityRecord
from cadiff.v06_a.structure import estimate_normal_structure

from .atlas_a import AtlasProposalGenerator
from .features import FrozenDinoEncoder
from .panel import _CELL, _overlay_mask, _to_image

_OUTPUT_PREFIX = ("v06_a", "diag_atlas_prop")
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")
_ATLAS_COLOR = (80, 160, 255)
_PANEL_SEEDS = (2026082801, 2026082802)
_PREVIEW_SEED = 2026082801
_PREVIEW_TYPE_COUNT = 2
_PREVIEW_NORMAL_COUNT = 2
_NORMAL_COUNT = 4
_MAX_TYPES = 4
_VISA_MIN_TYPE_COUNT = 6
_MVTEC_TEXTURE = frozenset({"carpet", "grid", "leather", "tile", "wood"})
_MVTEC_USED = frozenset({"hazelnut", "cable"})
_MVTEC_LOGICAL = frozenset({"transistor"})
_EXPECTED_MVTEC = ("bottle", "capsule")
_EXPECTED_VISA = ("candle", "capsules", "cashew", "chewinggum")
REVIEW_COLUMNS = ("source", "source_gt", "target", "proposal")


@dataclass(frozen=True)
class ExtraAStructureConfig:
    category: str
    defect_types: tuple[str, ...]
    image_size: int = 512
    proposal_size: int = 128
    support_border_width: int = 16
    support_background_margin: float = 0.03


@dataclass(frozen=True)
class ExtraASlot:
    dataset: str
    category: str
    defect_type: str
    normal_id: str
    source_id: str
    seed: int


@dataclass(frozen=True)
class ExtraACategory:
    dataset: str
    name: str
    root: Path
    defect_types: tuple[str, ...]
    fewshot: Mapping[str, tuple[IdentityRecord, ...]]
    normals: tuple[IdentityRecord, ...]


def _item_id(path: Path) -> str:
    return path.stem


def _sample_id(category: str, defect_type: str, relpath: str) -> str:
    return f"{category}/{defect_type}/{relpath}"


def select_mvtec_categories(mvtec_root: Path) -> tuple[str, ...]:
    names = tuple(
        sorted(
            path.name
            for path in mvtec_root.iterdir()
            if path.is_dir()
            and (path / "train" / "good").is_dir()
            and path.name not in _MVTEC_TEXTURE | _MVTEC_USED | _MVTEC_LOGICAL
        )
    )
    selected = names[:2]
    if selected != _EXPECTED_MVTEC:
        raise ValueError(f"MVTec extra-A selection must stay {_EXPECTED_MVTEC}, got {selected}")
    return selected


def select_visa_categories(visa_root: Path) -> tuple[str, ...]:
    names = tuple(
        sorted(
            path.name
            for path in visa_root.iterdir()
            if path.is_dir() and (path / "image_anno.csv").is_file()
        )
    )
    selected = names[:4]
    if selected != _EXPECTED_VISA:
        raise ValueError(f"VisA extra-A selection must stay {_EXPECTED_VISA}, got {selected}")
    return selected


def _even_ids(ids: tuple[str, ...], count: int) -> tuple[str, ...]:
    if len(ids) < count:
        raise ValueError("not enough normal images for the frozen extra-A panel")
    if count == 1:
        return (ids[0],)
    return tuple(ids[round(index * (len(ids) - 1) / (count - 1))] for index in range(count))


def _fewshot_ids(ids: tuple[str, ...]) -> tuple[str, ...]:
    count = len(ids) // 3
    if count < 1:
        raise ValueError("type does not have enough images for floor(n/3) few-shot")
    return ids[:count]


def _record(
    *,
    category: str,
    defect_type: str,
    item_id: str,
    image_relpath: str,
    mask_relpath: str | None,
    role: str,
) -> IdentityRecord:
    return IdentityRecord(
        category=category,
        defect_type=defect_type,
        item_id=item_id,
        image_relpath=image_relpath,
        mask_relpath=mask_relpath,
        role=role,
        sample_id=_sample_id(category, defect_type, image_relpath),
    )


def inventory_mvtec(mvtec_root: Path, category: str) -> ExtraACategory:
    test_root = mvtec_root / category / "test"
    types = tuple(
        sorted(path.name for path in test_root.iterdir() if path.is_dir() and path.name != "good")
    )[:_MAX_TYPES]
    if not types:
        raise ValueError(f"MVTec {category} has no anomaly types")
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
                mask_relpath=f"{category}/ground_truth/{defect_type}/{item_id}_mask.png",
                role="fewshot",
            )
            for item_id in chosen
        )
    normals = tuple(sorted((mvtec_root / category / "train" / "good").glob("*.png")))
    normal_ids = _even_ids(tuple(_item_id(path) for path in normals), _NORMAL_COUNT)
    return ExtraACategory(
        dataset="mvtec",
        name=category,
        root=mvtec_root,
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


def _visa_single_types(visa_root: Path, category: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (visa_root / category / "image_anno.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"]
            if label == "normal" or "," in label or not row["mask"]:
                continue
            grouped[label].append(row)
    kept = {
        label: tuple(sorted(rows, key=lambda item: item["image"]))
        for label, rows in grouped.items()
        if len(rows) >= _VISA_MIN_TYPE_COUNT
    }
    chosen = tuple(sorted(kept))[:_MAX_TYPES]
    if not chosen:
        raise ValueError(f"VisA {category} has no single-label types with >= {_VISA_MIN_TYPE_COUNT} images")
    return {label: list(kept[label]) for label in chosen}


def inventory_visa(visa_root: Path, category: str) -> ExtraACategory:
    grouped = _visa_single_types(visa_root, category)
    types = tuple(grouped)
    fewshot: dict[str, tuple[IdentityRecord, ...]] = {}
    for defect_type, rows in grouped.items():
        ids = tuple(_item_id(Path(row["image"])) for row in rows)
        chosen = set(_fewshot_ids(ids))
        selected = [row for row in rows if _item_id(Path(row["image"])) in chosen]
        fewshot[defect_type] = tuple(
            _record(
                category=category,
                defect_type=defect_type,
                item_id=_item_id(Path(row["image"])),
                image_relpath=row["image"],
                mask_relpath=row["mask"],
                role="fewshot",
            )
            for row in selected
        )
    normals = tuple(sorted((visa_root / category / "Data" / "Images" / "Normal").glob("*.JPG")))
    if not normals:
        normals = tuple(sorted((visa_root / category / "Data" / "Images" / "Normal").glob("*.jpg")))
    normal_ids = _even_ids(tuple(_item_id(path) for path in normals), _NORMAL_COUNT)
    suffix = normals[0].suffix
    return ExtraACategory(
        dataset="visa",
        name=category,
        root=visa_root,
        defect_types=types,
        fewshot=fewshot,
        normals=tuple(
            _record(
                category=category,
                defect_type="good",
                item_id=item_id,
                image_relpath=f"{category}/Data/Images/Normal/{item_id}{suffix}",
                mask_relpath=None,
                role="normal_dev",
            )
            for item_id in normal_ids
        ),
    )


def build_extra_a_slots(pack: ExtraACategory) -> tuple[ExtraASlot, ...]:
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


def build_extra_preview_slots(pack: ExtraACategory) -> tuple[ExtraASlot, ...]:
    """Frozen 20260910 extra a2 preview: first 2 types × first 2 normals × one seed."""

    types = pack.defect_types[:_PREVIEW_TYPE_COUNT]
    normals = pack.normals[:_PREVIEW_NORMAL_COUNT]
    if not types or not normals:
        raise ValueError(f"{pack.dataset}/{pack.name} has no preview types or normals")
    slots: list[ExtraASlot] = []
    for defect_type in types:
        sources = pack.fewshot[defect_type]
        source = sources[_PREVIEW_SEED % len(sources)]
        for normal in normals:
            slots.append(
                ExtraASlot(
                    dataset=pack.dataset,
                    category=pack.name,
                    defect_type=defect_type,
                    normal_id=normal.item_id,
                    source_id=source.item_id,
                    seed=_PREVIEW_SEED,
                )
            )
    return tuple(slots)


def _prepare_mask(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("L"))
    return Image.fromarray(((array > 0).astype(np.uint8) * 255))


def load_extra_identity(record: IdentityRecord, *, dataset_root: Path) -> LoadedIdentity:
    image_path = dataset_root.joinpath(*PurePosixPath(record.image_relpath).parts)
    with Image.open(image_path) as source:
        image, valid, geometry = resize_pad_image(source, 512)
    rgb = np.asarray(image, dtype=np.uint8).copy()
    valid_array = np.asarray(valid, dtype=np.uint8).astype(bool, copy=False)
    anomaly_gt = None
    if record.mask_relpath is not None:
        mask_path = dataset_root.joinpath(*PurePosixPath(record.mask_relpath).parts)
        with Image.open(mask_path) as source_mask:
            prepared = _prepare_mask(source_mask)
            if prepared.size != (geometry.source_width, geometry.source_height):
                prepared = prepared.resize(
                    (geometry.source_width, geometry.source_height),
                    Image.Resampling.NEAREST,
                )
            transformed = resize_pad_mask(prepared, geometry)
        anomaly_gt = np.asarray(transformed, dtype=np.uint8).astype(bool, copy=False)
        if not anomaly_gt.any():
            raise ValueError(f"empty extra-A mask: {record.mask_relpath}")
    return LoadedIdentity(record, rgb, valid_array, anomaly_gt)


def extra_a_output_path(output_root: Path, output_relpath: str) -> Path:
    parts = PurePosixPath(output_relpath).parts
    if parts[:2] != _OUTPUT_PREFIX or len(parts) != 3:
        raise ValueError("extra-A output must live under v06_a/diag_atlas_prop/<name>")
    if parts[2].startswith(_FORBIDDEN_OUTPUT_PREFIXES):
        raise ValueError(f"refuses to write f1-f4 outputs: {parts[2]}")
    root = output_root.resolve(strict=True)
    destination = root.joinpath(*parts)
    destination.resolve(strict=False).relative_to(root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {destination}")
    return destination


def _slug(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def _render_row(*, source: LoadedIdentity, target: LoadedIdentity, proposal: np.ndarray, title: str) -> Image.Image:
    if source.anomaly_gt is None:
        raise ValueError("extra-A source requires a ground-truth mask")
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _run_category(
    pack: ExtraACategory,
    *,
    destination: Path,
    encoder: FrozenDinoEncoder,
) -> dict[str, Any]:
    config = ExtraAStructureConfig(category=pack.name, defect_types=pack.defect_types)
    sources = {
        defect_type: tuple(
            load_extra_identity(record, dataset_root=pack.root) for record in records
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
    source_by_key = {
        (item.record.defect_type, item.record.item_id): item
        for items in sources.values()
        for item in items
    }
    slots = build_extra_a_slots(pack)
    rows_dir = destination / pack.dataset / _slug(pack.name) / "rows"
    rows_dir.mkdir(parents=True)
    images = []
    sidecar = []
    for index, slot in enumerate(slots):
        structure = structures[slot.normal_id]
        source = source_by_key[(slot.defect_type, slot.source_id)]
        try:
            result = generator.generate(structure, slot.defect_type, slot.seed)
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
            f"{index:02d} {pack.dataset}/{pack.name}/{slot.defect_type} "
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
    panel = destination / pack.dataset / _slug(pack.name) / f"atlas_prop_{pack.dataset}_{_slug(pack.name)}.png"
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


def run_extra_a(
    *,
    mvtec_root: Path,
    visa_root: Path,
    output_root: Path,
    output_relpath: str,
    device: str | None = None,
) -> dict[str, Any]:
    destination = extra_a_output_path(output_root, output_relpath)
    mvtec = [inventory_mvtec(mvtec_root, name) for name in select_mvtec_categories(mvtec_root)]
    visa = [inventory_visa(visa_root, name) for name in select_visa_categories(visa_root)]
    destination.mkdir(parents=True)
    encoder = FrozenDinoEncoder(device=device)
    summaries = []
    all_rows = []
    for pack in (*mvtec, *visa):
        summary = _run_category(pack, destination=destination, encoder=encoder)
        summaries.append({key: value for key, value in summary.items() if key != "rows"})
        all_rows.extend(summary["rows"])
    payload = {
        "schema_version": "v06_a_diag_atlas_prop_extra_a",
        "status": "awaiting_human_review",
        "look_at": "blue proposal column",
        "mvtec": list(_EXPECTED_MVTEC),
        "visa": list(_EXPECTED_VISA),
        "selection": {
            "mvtec": "object, not used/texture/transistor, lex 2",
            "visa": "image_anno categories, lex 4; single-label types only, n>=6, lex 4",
            "fewshot": "floor(n/3) by lexical id",
            "normals": "4 evenly spaced train/normal ids",
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
    parser = argparse.ArgumentParser(description="Extra-category atlas A panels. Not Module B.")
    parser.add_argument("--mvtec-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_extra_a(
        mvtec_root=arguments.mvtec_root,
        visa_root=arguments.visa_root,
        output_root=arguments.output_root,
        output_relpath=arguments.output_relpath,
        device=arguments.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

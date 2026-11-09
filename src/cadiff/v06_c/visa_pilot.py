"""VisA DualAnoDiff inventory, MVTec-style eval layout, and extra-style identities.

Frozen extra-A selection: candle / capsules / cashew / chewinggum.
Diagnostic third-dataset arm. Does not add VisA to official P2-A1 / v0.6-A loaders.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from cadiff.provenance import sha256_file, write_json_atomic
from cadiff.v06_a.data import LoadedIdentity
from cadiff.v06_a.identities import IdentityRecord
from cadiff.v06_atlas_diag.extra_a import (
    ExtraAStructureConfig,
    _fewshot_ids,
    _item_id,
    _prepare_mask,
    _record,
    _visa_single_types,
    load_extra_identity,
    select_visa_categories,
)

SOURCE_ROOT = Path("/home/dell/2T/lm/new_projects/category_anomaly_diffusion/data/VisA")
AMAZON_1CLS = (
    Path(__file__).resolve().parents[3]
    / "configs/checkpoints/visa_split_catalog_20260922/amazon_1cls.csv"
)
DATASET_NAME = "VisA"
SPLIT_ID = "visa_lexicographic_floor_third_v1"
SPLIT_ID_COLLAPSED = "visa_collapsed_defect_floor_third_v1"
SPLIT_ID_SEAS = "visa_seas_type_amazon1cls_v1"
SPLIT_ID_MAGIC = "visa_magic_single_anomaly_amazon1cls_v1"
COLLAPSED_TYPE = "defect"
COMBINED_TYPE = "combined"
IMAGE_SIZE = 512
EXPECTED = ("candle", "capsules", "cashew", "chewinggum")
MAGIC_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
VISA_SPLITS = ("extra-a", "defect", "seas", "magic", "mvtec")


def slug_type(name: str) -> str:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("defect type must be a non-empty string")
    return name.replace(" ", "_")


@dataclass(frozen=True)
class VisaTypeSplit:
    name: str
    slug: str
    rows: tuple[dict[str, str], ...]
    fewshot: tuple[dict[str, str], ...]
    heldout: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class VisaInventory:
    category: str
    source_root: Path
    types: tuple[VisaTypeSplit, ...]
    normals: tuple[Path, ...]
    structure: ExtraAStructureConfig
    split_id: str = SPLIT_ID
    collapsed_from: tuple[str, ...] | None = None
    test_normals: tuple[Path, ...] | None = None


def load_visa_inventory(source_root: Path, category: str) -> VisaInventory:
    if category not in EXPECTED:
        raise ValueError(f"VisA pilot category must stay {EXPECTED}, got {category}")
    if select_visa_categories(source_root) != EXPECTED:
        raise ValueError("VisA extra-A selection drifted")
    grouped = _visa_single_types(source_root, category)
    types: list[VisaTypeSplit] = []
    for name, rows in grouped.items():
        ordered = tuple(rows)
        ids = tuple(_item_id(Path(row["image"])) for row in ordered)
        few_ids = set(_fewshot_ids(ids))
        fewshot = tuple(row for row in ordered if _item_id(Path(row["image"])) in few_ids)
        heldout = tuple(row for row in ordered if _item_id(Path(row["image"])) not in few_ids)
        if len(fewshot) + len(heldout) != len(ordered):
            raise RuntimeError(f"{category}/{name} few-shot split collided")
        types.append(
            VisaTypeSplit(
                name=name,
                slug=slug_type(name),
                rows=ordered,
                fewshot=fewshot,
                heldout=heldout,
            )
        )
    normals = tuple(sorted((source_root / category / "Data" / "Images" / "Normal").glob("*.JPG")))
    if not normals:
        normals = tuple(sorted((source_root / category / "Data" / "Images" / "Normal").glob("*.jpg")))
    if len(normals) < 1:
        raise ValueError(f"VisA {category} has no train/good normals")
    return VisaInventory(
        category=category,
        source_root=source_root.resolve(strict=True),
        types=tuple(types),
        normals=normals,
        structure=ExtraAStructureConfig(
            category=category,
            defect_types=tuple(item.name for item in types),
        ),
        split_id=SPLIT_ID,
        collapsed_from=None,
    )


def load_visa_collapsed_inventory(source_root: Path, category: str) -> VisaInventory:
    """Pool extra-A single-label types into one DualAnoDiff class named defect.

    Few-shot is floor(n/3) on the pooled lexicographic list. Does not change
    the per-type inventory used by official-500/1000.
    """
    base = load_visa_inventory(source_root, category)
    source_names = tuple(item.name for item in base.types)
    pooled = tuple(sorted(
        (row for item in base.types for row in item.rows),
        key=lambda row: row["image"],
    ))
    ids = tuple(_item_id(Path(row["image"])) for row in pooled)
    if len(ids) != len(set(ids)):
        raise ValueError(f"VisA {category} collapsed stems collided")
    few_ids = set(_fewshot_ids(ids))
    fewshot = tuple(row for row in pooled if _item_id(Path(row["image"])) in few_ids)
    heldout = tuple(row for row in pooled if _item_id(Path(row["image"])) not in few_ids)
    if len(fewshot) + len(heldout) != len(pooled):
        raise RuntimeError(f"{category}/{COLLAPSED_TYPE} few-shot split collided")
    collapsed = VisaTypeSplit(
        name=COLLAPSED_TYPE,
        slug=slug_type(COLLAPSED_TYPE),
        rows=pooled,
        fewshot=fewshot,
        heldout=heldout,
    )
    return VisaInventory(
        category=base.category,
        source_root=base.source_root,
        types=(collapsed,),
        normals=base.normals,
        structure=ExtraAStructureConfig(
            category=category,
            defect_types=(COLLAPSED_TYPE,),
        ),
        split_id=SPLIT_ID_COLLAPSED,
        collapsed_from=source_names,
    )


def seas_fewshot_ids(ids: tuple[str, ...]) -> tuple[str, ...]:
    if not ids:
        raise ValueError("type does not have images for SeaS few-shot")
    return ids[: max(len(ids) // 3, 1)]


def resolve_visa_split(split: str | None, collapse_type: str | None) -> str:
    chosen = "extra-a" if split in (None, "") else split
    if chosen not in VISA_SPLITS:
        raise ValueError(f"VisA split must be one of {VISA_SPLITS}, got {split!r}")
    if collapse_type in (None, ""):
        return chosen
    if collapse_type != COLLAPSED_TYPE:
        raise ValueError(f"VisA collapse_type must be {COLLAPSED_TYPE!r} or empty")
    if chosen not in ("extra-a", "defect"):
        raise ValueError("SeaS split cannot combine with --collapse-type defect")
    return "defect"


def _amazon_1cls_normals(source_root: Path, category: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    if not AMAZON_1CLS.is_file():
        raise FileNotFoundError(AMAZON_1CLS)
    train: list[Path] = []
    test: list[Path] = []
    with AMAZON_1CLS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [name.strip() for name in (reader.fieldnames or [])]
        for raw in reader:
            row = {key.strip(): (value or "").strip() for key, value in raw.items()}
            if row["object"] != category or row["label"] != "normal":
                continue
            path = source_root.joinpath(*PurePosixPath(row["image"]).parts)
            if not path.is_file():
                raise FileNotFoundError(path)
            if row["split"] == "train":
                train.append(path)
            elif row["split"] == "test":
                test.append(path)
            else:
                raise ValueError(f"unexpected Amazon 1cls split {row['split']!r}")
    if not train or not test:
        raise ValueError(f"VisA {category} Amazon 1cls normals are empty")
    train_ids = {path.name for path in train}
    test_ids = {path.name for path in test}
    if train_ids & test_ids:
        raise RuntimeError(f"VisA {category} Amazon 1cls train/test good overlap")
    return tuple(sorted(train, key=lambda path: path.name)), tuple(sorted(test, key=lambda path: path.name))


def _visa_seas_grouped(source_root: Path, category: str) -> dict[str, list[dict[str, str]]]:
    single: dict[str, list[dict[str, str]]] = defaultdict(list)
    multi: list[dict[str, str]] = []
    with (source_root / category / "image_anno.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["label"]
            if label == "normal" or not row.get("mask"):
                continue
            if "," in label:
                multi.append(row)
            else:
                single[label].append(row)
    grouped = {
        label: sorted(rows, key=lambda item: item["image"])
        for label, rows in single.items()
    }
    if multi:
        grouped[COMBINED_TYPE] = sorted(multi, key=lambda item: item["image"])
    if not grouped:
        raise ValueError(f"VisA {category} has no SeaS defect types")
    return {label: grouped[label] for label in sorted(grouped)}


def magic_fewshot_ids(ids: tuple[str, ...]) -> tuple[str, ...]:
    """AnomalyDiffusion / MAGIC: first floor(n/3). n<3 yields no few-shot images."""
    if not ids:
        raise ValueError("type does not have images for MAGIC few-shot")
    return ids[: len(ids) // 3]


def _type_split(name: str, rows: list[dict[str, str]], *, seas: bool, magic: bool = False) -> VisaTypeSplit:
    if seas and magic:
        raise ValueError("a type split cannot be both SeaS and MAGIC")
    ordered = tuple(rows)
    ids = tuple(_item_id(Path(row["image"])) for row in ordered)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} stems collided")
    if magic:
        chosen = magic_fewshot_ids(ids)
    elif seas:
        chosen = seas_fewshot_ids(ids)
    else:
        chosen = _fewshot_ids(ids)
    few_ids = set(chosen)
    fewshot = tuple(row for row in ordered if _item_id(Path(row["image"])) in few_ids)
    heldout = tuple(row for row in ordered if _item_id(Path(row["image"])) not in few_ids)
    if len(fewshot) + len(heldout) != len(ordered):
        raise RuntimeError(f"{name} few-shot split collided")
    return VisaTypeSplit(
        name=name,
        slug=slug_type(name),
        rows=ordered,
        fewshot=fewshot,
        heldout=heldout,
    )


def load_visa_seas_inventory(source_root: Path, category: str) -> VisaInventory:
    """SeaS type folders + Amazon 1cls train/test good. Pilot categories only."""
    if category not in EXPECTED:
        raise ValueError(f"VisA pilot category must stay {EXPECTED}, got {category}")
    if select_visa_categories(source_root) != EXPECTED:
        raise ValueError("VisA extra-A selection drifted")
    grouped = _visa_seas_grouped(source_root, category)
    types = tuple(_type_split(name, rows, seas=True) for name, rows in grouped.items())
    names = tuple(item.name for item in types)
    if names != tuple(sorted(names)):
        raise RuntimeError(f"SeaS types must be lexicographic, got {names}")
    train_normals, test_normals = _amazon_1cls_normals(source_root, category)
    return VisaInventory(
        category=category,
        source_root=source_root.resolve(strict=True),
        types=types,
        normals=train_normals,
        structure=ExtraAStructureConfig(
            category=category,
            defect_types=names,
        ),
        split_id=SPLIT_ID_SEAS,
        collapsed_from=None,
        test_normals=test_normals,
    )


def load_visa_magic_inventory(
    source_root: Path, category: str, *, generate_only: bool = True
) -> VisaInventory:
    """Single-label VisA + Amazon 1cls. Few-shot is floor(n/3), with no combined bucket.

    generate_only drops types whose floor(n/3) is 0 so R and generate never see them.
    The eval layout must be staged with generate_only=False so those images stay in test.
    """
    if category not in MAGIC_CATEGORIES:
        raise ValueError(f"VisA MAGIC category must stay {MAGIC_CATEGORIES}, got {category}")
    grouped = _visa_seas_grouped(source_root, category)
    grouped.pop(COMBINED_TYPE, None)
    if not grouped:
        raise ValueError(f"VisA {category} has no single-label MAGIC types")
    types = tuple(
        _type_split(name, rows, seas=False, magic=True) for name, rows in grouped.items()
    )
    if generate_only:
        types = tuple(item for item in types if item.fewshot)
    if not types:
        raise ValueError(f"VisA {category} has no MAGIC types left to generate")
    names = tuple(item.name for item in types)
    if names != tuple(sorted(names)):
        raise RuntimeError(f"MAGIC types must be lexicographic, got {names}")
    if any(item.name == COMBINED_TYPE for item in types):
        raise RuntimeError("MAGIC inventory must not contain combined")
    train_normals, test_normals = _amazon_1cls_normals(source_root, category)
    return VisaInventory(
        category=category,
        source_root=source_root.resolve(strict=True),
        types=types,
        normals=train_normals,
        structure=ExtraAStructureConfig(
            category=category,
            defect_types=names,
        ),
        split_id=SPLIT_ID_MAGIC,
        collapsed_from=None,
        test_normals=test_normals,
    )


def load_visa_inventory_for_split(
    source_root: Path, category: str, split: str | None, collapse_type: str | None = None
) -> VisaInventory:
    chosen = resolve_visa_split(split, collapse_type)
    if chosen == "seas":
        return load_visa_seas_inventory(source_root, category)
    if chosen == "magic":
        return load_visa_magic_inventory(source_root, category, generate_only=True)
    if chosen == "mvtec":
        from cadiff.v06_c.mvtec_pilot import load_mvtec_inventory

        return load_mvtec_inventory(source_root, category)
    if chosen == "defect":
        return load_visa_collapsed_inventory(source_root, category)
    return load_visa_inventory(source_root, category)


def fewshot_identity_records(inventory: VisaInventory) -> tuple[IdentityRecord, ...]:
    records: list[IdentityRecord] = []
    for item in inventory.types:
        for row in item.fewshot:
            records.append(
                _record(
                    category=inventory.category,
                    defect_type=item.name,
                    item_id=_item_id(Path(row["image"])),
                    image_relpath=row["image"],
                    mask_relpath=row["mask"],
                    role="fewshot",
                )
            )
    return tuple(records)


def heldout_identity_records(inventory: VisaInventory) -> tuple[IdentityRecord, ...]:
    records: list[IdentityRecord] = []
    for item in inventory.types:
        for row in item.heldout:
            records.append(
                _record(
                    category=inventory.category,
                    defect_type=item.name,
                    item_id=_item_id(Path(row["image"])),
                    image_relpath=row["image"],
                    mask_relpath=row["mask"],
                    role="open_dev_report",
                )
            )
    return tuple(records)


def normal_identity_records(inventory: VisaInventory) -> tuple[IdentityRecord, ...]:
    root = inventory.source_root.resolve()
    return tuple(
        _record(
            category=inventory.category,
            defect_type="good",
            item_id=path.stem,
            image_relpath=path.resolve().relative_to(root).as_posix(),
            mask_relpath=None,
            role="normal_fit",
        )
        for path in inventory.normals
    )


def load_visa_identity(record: IdentityRecord, *, dataset_root: Path) -> LoadedIdentity:
    loaded = load_extra_identity(record, dataset_root=dataset_root)
    if record.role == "fewshot":
        if loaded.anomaly_gt is None or not np.asarray(loaded.anomaly_gt).any():
            raise ValueError(f"empty VisA few-shot mask: {record.mask_relpath}")
    return loaded


def build_visa_manifest(inventory: VisaInventory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = inventory.source_root
    for record in (*fewshot_identity_records(inventory), *heldout_identity_records(inventory)):
        image_path = root.joinpath(*PurePosixPath(record.image_relpath).parts)
        mask_path = root.joinpath(*PurePosixPath(str(record.mask_relpath)).parts)
        role = "fewshot_train" if record.role == "fewshot" else "open_dev_report"
        rows.append(
            {
                "category": inventory.category,
                "dataset": DATASET_NAME,
                "defect_type": record.defect_type,
                "image_relpath": record.image_relpath,
                "image_sha256": sha256_file(image_path),
                "mask_relpath": record.mask_relpath,
                "mask_sha256": sha256_file(mask_path),
                "roles": [role],
                "split_id": inventory.split_id,
            }
        )
    for record in normal_identity_records(inventory):
        image_path = root.joinpath(*PurePosixPath(record.image_relpath).parts)
        rows.append(
            {
                "category": inventory.category,
                "dataset": DATASET_NAME,
                "defect_type": "good",
                "image_relpath": record.image_relpath,
                "image_sha256": sha256_file(image_path),
                "mask_relpath": None,
                "mask_sha256": None,
                "roles": ["normal_train"],
                "split_id": inventory.split_id,
            }
        )
    rows.sort(key=lambda row: (str(row["defect_type"]), str(row["image_relpath"])))
    return rows


def write_visa_manifest(inventory: VisaInventory, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = build_visa_manifest(inventory)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json_atomic(
        destination.with_suffix(".summary.json"),
        {
            "schema_version": "v06_c_visa_split_v1",
            "category": inventory.category,
            "defect_types": [item.name for item in inventory.types],
            "split_id": inventory.split_id,
            "collapsed_from": list(inventory.collapsed_from or ()),
            "fewshot_train": sum(len(item.fewshot) for item in inventory.types),
            "open_dev_report": sum(len(item.heldout) for item in inventory.types),
            "normal_train": len(inventory.normals),
            "manifest": str(destination),
            "source_root": str(inventory.source_root),
        },
    )
    return destination


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())


def split_visa_normals(normals: tuple[Path, ...]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Lexicographic last 1/3 of VisA normals are test/good; the rest stay train/good.

    eval_kit image AUROC needs both classes. VisA has no official MVTec test/good.
    Defect few-shot already uses the first floor(n/3); this holds out the tail.
    """
    if len(normals) < 2:
        raise ValueError("VisA eval layout needs at least 2 normals for train/good and test/good")
    n_test = max(1, len(normals) // 3)
    n_train = len(normals) - n_test
    if n_train < 1:
        raise ValueError("VisA eval layout needs at least 1 train/good after holding out test/good")
    return normals[:n_train], normals[n_train:]


def _stage_visa_normals(
    root: Path, train: tuple[Path, ...], test: tuple[Path, ...]
) -> None:
    train_dir = root / "train" / "good"
    test_dir = root / "test" / "good"
    keep_train = {path.name for path in train}
    keep_test = {path.name for path in test}
    if keep_train & keep_test:
        raise RuntimeError("VisA train/good and test/good must be disjoint")
    if train_dir.is_dir():
        for child in train_dir.iterdir():
            if child.name not in keep_train:
                child.unlink()
    if test_dir.is_dir():
        for child in test_dir.iterdir():
            if child.name not in keep_test:
                child.unlink()
    for path in train:
        _link(path, train_dir / path.name)
    for path in test:
        _link(path, test_dir / path.name)


def _write_visa_layout_json(
    destination: Path,
    inventory: VisaInventory,
    train: tuple[Path, ...],
    test: tuple[Path, ...],
) -> None:
    write_json_atomic(
        destination / "layout.json",
        {
            "schema_version": "v06_c_visa_eval_layout_v1",
            "category": inventory.category,
            "types": [{"name": item.name, "slug": item.slug, "count": len(item.rows)} for item in inventory.types],
            "split_id": inventory.split_id,
            "collapsed_from": list(inventory.collapsed_from or ()),
            "normal_count": len(inventory.normals) + (
                len(inventory.test_normals) if inventory.test_normals is not None else 0
            ),
            "train_good_count": len(train),
            "test_good_count": len(test),
            "normal_split": (
                "amazon_spot_diff_1cls_v1"
                if inventory.test_normals is not None
                else "lexicographic_last_third_test_good_v1"
            ),
        },
    )


def stage_visa_eval_layout(inventory: VisaInventory, destination: Path) -> Path:
    root = destination / inventory.category
    if inventory.test_normals is not None:
        train, test = inventory.normals, inventory.test_normals
        if not train or not test:
            raise ValueError("SeaS eval layout needs Amazon 1cls train/good and test/good")
        overlap = {path.name for path in train} & {path.name for path in test}
        if overlap:
            raise RuntimeError("VisA train/good and test/good must be disjoint")
    else:
        train, test = split_visa_normals(inventory.normals)
    if destination.exists() or destination.is_symlink():
        marker = root / "train" / "good"
        if not marker.is_dir():
            raise FileExistsError(f"refuses to overwrite eval layout: {destination}")
        _stage_visa_normals(root, train, test)
        _write_visa_layout_json(destination, inventory, train, test)
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    _stage_visa_normals(root, train, test)
    for item in inventory.types:
        for row in item.rows:
            image = inventory.source_root.joinpath(*PurePosixPath(row["image"]).parts)
            mask = inventory.source_root.joinpath(*PurePosixPath(row["mask"]).parts)
            stem = Path(row["image"]).stem
            suffix = Path(row["image"]).suffix
            _link(image, root / "test" / item.slug / f"{stem}{suffix}")
            prepared = _prepare_mask(Image.open(mask))
            gt_path = root / "ground_truth" / item.slug / f"{stem}_mask.png"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            if gt_path.exists() or gt_path.is_symlink():
                raise FileExistsError(gt_path)
            prepared.save(gt_path)
    _write_visa_layout_json(destination, inventory, train, test)
    return destination


__all__ = [
    "AMAZON_1CLS",
    "COLLAPSED_TYPE",
    "COMBINED_TYPE",
    "DATASET_NAME",
    "EXPECTED",
    "MAGIC_CATEGORIES",
    "IMAGE_SIZE",
    "SOURCE_ROOT",
    "SPLIT_ID",
    "SPLIT_ID_COLLAPSED",
    "SPLIT_ID_MAGIC",
    "SPLIT_ID_SEAS",
    "VISA_SPLITS",
    "VisaInventory",
    "VisaTypeSplit",
    "build_visa_manifest",
    "fewshot_identity_records",
    "heldout_identity_records",
    "load_visa_collapsed_inventory",
    "load_visa_identity",
    "load_visa_inventory",
    "load_visa_inventory_for_split",
    "load_visa_magic_inventory",
    "load_visa_seas_inventory",
    "magic_fewshot_ids",
    "normal_identity_records",
    "resolve_visa_split",
    "seas_fewshot_ids",
    "slug_type",
    "split_visa_normals",
    "stage_visa_eval_layout",
    "write_visa_manifest",
]

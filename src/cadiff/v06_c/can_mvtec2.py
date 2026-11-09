"""MVTec AD 2 can layout, DualAnoDiff split, and extra-style identities.

Diagnostic third category. Does not add can to frozen v0.6-A / P2-A1 loaders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from cadiff.provenance import sha256_file, write_json_atomic
from cadiff.v06_a.data import LoadedIdentity
from cadiff.v06_a.identities import IdentityRecord
from cadiff.v06_atlas_diag.extra_a import ExtraAStructureConfig, load_extra_identity

SOURCE_ROOT = Path(
    "/home/dell/2T/lm/new_projects/category_anomaly_diffusion/data/AD/mvtec2/"
    "Downloads/can/can"
)
CATEGORY = "can"
DEFECT_TYPE = "bad"
DATASET_NAME = "MVTecAD2"
SPLIT_ID = "mvtec2_can_lexicographic_floor_third_v1"
IMAGE_SIZE = 512
STRUCTURE_CONFIG = ExtraAStructureConfig(category=CATEGORY, defect_types=(DEFECT_TYPE,))


@dataclass(frozen=True)
class CanInventory:
    source_root: Path
    layout_root: Path
    train_good: tuple[Path, ...]
    test_bad: tuple[Path, ...]
    test_good: tuple[Path, ...]
    fewshot: tuple[Path, ...]
    heldout: tuple[Path, ...]


def _pngs(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing can directory: {directory}")
    files = tuple(sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png"))
    if not files:
        raise ValueError(f"no PNG files in {directory}")
    return files


def dualanodiff_split(files: tuple[Path, ...]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    if len(files) < 3:
        raise ValueError("can test_public/bad must have at least 3 images")
    skip = len(files) // 3
    if skip < 1:
        raise ValueError("floor(n/3) few-shot split is empty")
    return files[:skip], files[skip:]


def mask_path_for(image_path: Path, *, ground_truth_root: Path) -> Path:
    candidate = ground_truth_root / f"{image_path.stem}_mask.png"
    if not candidate.is_file():
        raise FileNotFoundError(f"missing can GT: {candidate}")
    return candidate


def stage_can_mvtec_layout(*, source_root: Path, layout_root: Path) -> CanInventory:
    source = source_root.resolve(strict=True)
    if layout_root.exists() or layout_root.is_symlink():
        existing = layout_root / CATEGORY
        if not existing.is_dir():
            raise FileExistsError(f"refuses to overwrite existing layout: {layout_root}")
    else:
        layout_root.mkdir(parents=True, exist_ok=False)
        can_root = layout_root / CATEGORY
        can_root.mkdir()
        (can_root / "train").symlink_to(source / "train")
        test_root = can_root / "test"
        test_root.mkdir()
        (test_root / "bad").symlink_to(source / "test_public" / "bad")
        (test_root / "good").symlink_to(source / "test_public" / "good")
        gt_root = can_root / "ground_truth"
        gt_root.mkdir()
        (gt_root / "bad").symlink_to(source / "test_public" / "ground_truth" / "bad")
    train_good = _pngs(layout_root / CATEGORY / "train" / "good")
    test_bad = _pngs(layout_root / CATEGORY / "test" / "bad")
    test_good = _pngs(layout_root / CATEGORY / "test" / "good")
    fewshot, heldout = dualanodiff_split(test_bad)
    if len(train_good) != 412:
        raise ValueError(f"can train/good must stay 412, got {len(train_good)}")
    if len(test_bad) != 90 or len(fewshot) != 30 or len(heldout) != 60:
        raise ValueError("can DualAnoDiff split must be 30 few-shot / 60 held-out")
    if len(test_good) != 72:
        raise ValueError(f"can test_public/good must stay 72, got {len(test_good)}")
    return CanInventory(
        source_root=source,
        layout_root=layout_root.resolve(strict=True),
        train_good=train_good,
        test_bad=test_bad,
        test_good=test_good,
        fewshot=fewshot,
        heldout=heldout,
    )


def _relpath(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return PurePosixPath(*relative.parts).as_posix()


def build_can_manifest(inventory: CanInventory) -> list[dict[str, Any]]:
    dataset_root = inventory.layout_root
    gt_root = dataset_root / CATEGORY / "ground_truth" / "bad"
    rows: list[dict[str, Any]] = []
    for path, role in (
        *((item, "fewshot_train") for item in inventory.fewshot),
        *((item, "open_dev_report") for item in inventory.heldout),
    ):
        mask = mask_path_for(path, ground_truth_root=gt_root)
        rows.append(
            {
                "category": CATEGORY,
                "dataset": DATASET_NAME,
                "defect_type": DEFECT_TYPE,
                "image_relpath": _relpath(dataset_root, path),
                "image_sha256": sha256_file(path),
                "mask_relpath": _relpath(dataset_root, mask),
                "mask_sha256": sha256_file(mask),
                "roles": [role],
                "split_id": SPLIT_ID,
            }
        )
    for path in inventory.train_good:
        rows.append(
            {
                "category": CATEGORY,
                "dataset": DATASET_NAME,
                "defect_type": "good",
                "image_relpath": _relpath(dataset_root, path),
                "image_sha256": sha256_file(path),
                "mask_relpath": None,
                "mask_sha256": None,
                "roles": ["normal_train"],
                "split_id": SPLIT_ID,
            }
        )
    rows.sort(key=lambda row: str(row["image_relpath"]))
    return rows


def write_can_manifest(inventory: CanInventory, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = build_can_manifest(inventory)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json_atomic(
        destination.with_suffix(".summary.json"),
        {
            "schema_version": "v06_c_can_mvtec2_split_v1",
            "category": CATEGORY,
            "defect_type": DEFECT_TYPE,
            "split_id": SPLIT_ID,
            "fewshot_train": 30,
            "open_dev_report": 60,
            "normal_train": 412,
            "manifest": str(destination),
            "layout_root": str(inventory.layout_root),
            "source_root": str(inventory.source_root),
        },
    )
    return destination


def _record(
    *,
    defect_type: str,
    item_id: str,
    image_relpath: str,
    mask_relpath: str | None,
    role: str,
) -> IdentityRecord:
    return IdentityRecord(
        category=CATEGORY,
        defect_type=defect_type,
        item_id=item_id,
        image_relpath=image_relpath,
        mask_relpath=mask_relpath,
        role=role,
        sample_id=f"{CATEGORY}/{defect_type}/{image_relpath}",
    )


def fewshot_identity_records(inventory: CanInventory) -> tuple[IdentityRecord, ...]:
    dataset_root = inventory.layout_root
    gt_root = dataset_root / CATEGORY / "ground_truth" / "bad"
    return tuple(
        _record(
            defect_type=DEFECT_TYPE,
            item_id=path.stem,
            image_relpath=_relpath(dataset_root, path),
            mask_relpath=_relpath(dataset_root, mask_path_for(path, ground_truth_root=gt_root)),
            role="fewshot",
        )
        for path in inventory.fewshot
    )


def normal_identity_records(inventory: CanInventory) -> tuple[IdentityRecord, ...]:
    dataset_root = inventory.layout_root
    return tuple(
        _record(
            defect_type="good",
            item_id=path.stem,
            image_relpath=_relpath(dataset_root, path),
            mask_relpath=None,
            role="normal_fit",
        )
        for path in inventory.train_good
    )


def load_can_identity(record: IdentityRecord, *, dataset_root: Path) -> LoadedIdentity:
    loaded = load_extra_identity(record, dataset_root=dataset_root)
    if record.role == "fewshot":
        if loaded.anomaly_gt is None or not np.asarray(loaded.anomaly_gt).any():
            raise ValueError(f"empty can few-shot mask: {record.mask_relpath}")
    return loaded


def assert_letterbox_masks_nonempty(inventory: CanInventory) -> None:
    dataset_root = inventory.layout_root
    for record in fewshot_identity_records(inventory):
        loaded = load_can_identity(record, dataset_root=dataset_root)
        assert loaded.anomaly_gt is not None
        if int(loaded.anomaly_gt.sum()) < 1:
            raise ValueError(f"letterbox emptied can GT: {record.item_id}")


__all__ = [
    "CATEGORY",
    "CanInventory",
    "DEFECT_TYPE",
    "IMAGE_SIZE",
    "SOURCE_ROOT",
    "SPLIT_ID",
    "STRUCTURE_CONFIG",
    "assert_letterbox_masks_nonempty",
    "build_can_manifest",
    "dualanodiff_split",
    "fewshot_identity_records",
    "load_can_identity",
    "mask_path_for",
    "normal_identity_records",
    "stage_can_mvtec_layout",
    "write_can_manifest",
]

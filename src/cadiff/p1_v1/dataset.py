from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cadiff.data.geometry import GeometryMeta, resize_pad_image
from cadiff.data.mvtec_manifest import ManifestRecord
from cadiff.p1_v1.conditioning import DefectVocabulary
from cadiff.p1_v1.config import Smoke0Config
from cadiff.provenance import sha256_file


@dataclass(frozen=True)
class TrainingItem:
    sample_id: str
    category: str
    defect_type: str
    role: str
    image_relpath: str
    image_path: Path
    image_sha256: str


@dataclass(frozen=True)
class TrainingIndex:
    category: str
    vocabulary_sha256: str
    normals: tuple[TrainingItem, ...]
    anomalies: tuple[TrainingItem, ...]


@dataclass(frozen=True)
class LoadedTrainingImage:
    item: TrainingItem
    pixels: torch.Tensor
    valid: torch.Tensor
    geometry: GeometryMeta


_MANIFEST_FIELDS = frozenset(ManifestRecord.__dataclass_fields__)


def _manifest_record(raw: object, line_number: int) -> ManifestRecord:
    if not isinstance(raw, Mapping):
        raise ValueError(f"manifest line {line_number} must be a JSON object")
    actual = set(raw)
    missing = sorted(_MANIFEST_FIELDS - actual)
    extra = sorted(str(key) for key in actual - _MANIFEST_FIELDS)
    if missing:
        raise ValueError(
            f"manifest line {line_number} is missing fields: {', '.join(missing)}"
        )
    if extra:
        raise ValueError(
            f"manifest line {line_number} has unexpected fields: {', '.join(extra)}"
        )

    required_strings = (
        "dataset",
        "split_id",
        "category",
        "defect_type",
        "image_relpath",
        "image_sha256",
    )
    for field in required_strings:
        value = raw[field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"manifest line {line_number} field {field} must be a non-empty string"
            )
    roles = raw["roles"]
    if not isinstance(roles, list) or any(
        not isinstance(role, str) or not role for role in roles
    ):
        raise ValueError(
            f"manifest line {line_number} field roles must be a list of non-empty strings"
        )
    if len(set(roles)) != len(roles):
        raise ValueError(f"manifest line {line_number} field roles contains duplicates")
    for field in ("mask_relpath", "mask_sha256"):
        value = raw[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"manifest line {line_number} field {field} must be a string or null"
            )
    image_sha256 = raw["image_sha256"]
    if len(image_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in image_sha256
    ):
        raise ValueError(
            f"manifest line {line_number} field image_sha256 must be lowercase SHA256"
        )

    return ManifestRecord(
        dataset=raw["dataset"],
        split_id=raw["split_id"],
        category=raw["category"],
        defect_type=raw["defect_type"],
        image_relpath=raw["image_relpath"],
        mask_relpath=raw["mask_relpath"],
        roles=tuple(roles),
        image_sha256=image_sha256,
        mask_sha256=raw["mask_sha256"],
    )


def _read_manifest(manifest: Path) -> tuple[ManifestRecord, ...]:
    if manifest.is_symlink():
        raise ValueError(f"manifest symlink is not allowed: {manifest}")
    resolved = manifest.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"manifest must be a file: {resolved}")
    records: list[ManifestRecord] = []
    try:
        with resolved.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"manifest line {line_number} is invalid JSON: {error.msg}"
                    ) from error
                records.append(_manifest_record(raw, line_number))
    except UnicodeDecodeError as error:
        raise ValueError(f"manifest is not valid UTF-8: {resolved}") from error
    return tuple(records)


def _training_item(
    record: ManifestRecord, role: str, dataset_root: Path
) -> TrainingItem:
    image_path = (dataset_root / record.image_relpath).resolve(strict=True)
    try:
        image_path.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(
            f"image {image_path} escapes dataset root {dataset_root}"
        ) from error
    if not image_path.is_file():
        raise ValueError(f"training image must be a file: {image_path}")
    actual_sha256 = sha256_file(image_path)
    if actual_sha256 != record.image_sha256:
        raise ValueError(
            f"image SHA256 mismatch for {record.image_relpath}: "
            f"expected {record.image_sha256}, got {actual_sha256}"
        )
    sample_id = hashlib.sha256(
        f"{record.category}\n{record.defect_type}\n{record.image_relpath}".encode()
    ).hexdigest()
    return TrainingItem(
        sample_id=sample_id,
        category=record.category,
        defect_type=record.defect_type,
        role=role,
        image_relpath=record.image_relpath,
        image_path=image_path,
        image_sha256=record.image_sha256,
    )


def _remember_image_target(
    item: TrainingItem, seen_targets: dict[Path, str]
) -> None:
    previous_relpath = seen_targets.get(item.image_path)
    if previous_relpath is not None:
        raise ValueError(
            f"duplicate resolved image target {item.image_path}: "
            f"{previous_relpath} and {item.image_relpath}"
        )
    seen_targets[item.image_path] = item.image_relpath


def load_training_index(
    manifest: Path,
    dataset_root: Path,
    config: Smoke0Config,
    *,
    ignore_unknown_fewshot_types: bool = False,
) -> TrainingIndex:
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dataset_root must be a directory: {root}")
    vocabulary = DefectVocabulary.from_config(config.category, config.defect_types)
    records = _read_manifest(manifest)
    seen_paths: set[str] = set()
    seen_targets: dict[Path, str] = {}
    normals: list[TrainingItem] = []
    anomalies: list[TrainingItem] = []

    for record in records:
        if record.image_relpath in seen_paths:
            raise ValueError(f"duplicate image_relpath: {record.image_relpath}")
        seen_paths.add(record.image_relpath)
        if record.category != config.category:
            continue

        is_normal_train = "normal_train" in record.roles
        is_fewshot_train = "fewshot_train" in record.roles
        if is_normal_train:
            if record.defect_type != "good" or is_fewshot_train:
                raise ValueError(
                    "normal training records must be normal_train/good only: "
                    f"{record.image_relpath}"
                )
            item = _training_item(record, "normal_train", root)
            _remember_image_target(item, seen_targets)
            normals.append(item)
        elif is_fewshot_train:
            if record.defect_type not in vocabulary.labels:
                if ignore_unknown_fewshot_types:
                    continue
                raise ValueError(
                    "unexpected fewshot_train defect type "
                    f"{record.defect_type}: {record.image_relpath}"
                )
            item = _training_item(record, "fewshot_train", root)
            _remember_image_target(item, seen_targets)
            anomalies.append(item)

    if not normals:
        raise ValueError("manifest has no normal_train/good records for configured category")
    present_types = {item.defect_type for item in anomalies}
    missing_types = [label for label in vocabulary.labels if label not in present_types]
    if missing_types:
        raise ValueError(
            "missing fewshot_train records for configured defect types: "
            f"{', '.join(missing_types)}"
        )
    return TrainingIndex(
        category=config.category,
        vocabulary_sha256=vocabulary.sha256,
        normals=tuple(sorted(normals, key=lambda item: item.image_relpath)),
        anomalies=tuple(sorted(anomalies, key=lambda item: item.image_relpath)),
    )


def select_smoke0_items(
    index: TrainingIndex,
) -> tuple[TrainingItem, tuple[TrainingItem, ...]]:
    if not index.normals:
        raise ValueError("training index has no normal items")
    by_type: dict[str, list[TrainingItem]] = {}
    for item in index.anomalies:
        by_type.setdefault(item.defect_type, []).append(item)
    if not by_type:
        raise ValueError("training index has no anomaly items")
    selected = tuple(
        min(by_type[label], key=lambda item: item.image_relpath)
        for label in sorted(by_type)
    )
    normal = min(index.normals, key=lambda item: item.image_relpath)
    return normal, selected


def load_training_image(item: TrainingItem, image_size: int) -> LoadedTrainingImage:
    actual_sha256 = sha256_file(item.image_path)
    if actual_sha256 != item.image_sha256:
        raise ValueError(
            f"image SHA256 mismatch for {item.image_relpath}: "
            f"expected {item.image_sha256}, got {actual_sha256}"
        )
    with Image.open(item.image_path) as image:
        transformed, valid_image, geometry = resize_pad_image(image, image_size)
    pixel_array = np.array(transformed, dtype=np.uint8, copy=True)
    pixels = (
        torch.from_numpy(pixel_array)
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(127.5)
        .sub_(1.0)
        .contiguous()
    )
    valid_array = np.array(valid_image, dtype=np.uint8, copy=True)
    valid = torch.from_numpy(valid_array).ne(0).unsqueeze(0).contiguous()
    return LoadedTrainingImage(item=item, pixels=pixels, valid=valid, geometry=geometry)


__all__ = [
    "LoadedTrainingImage",
    "TrainingIndex",
    "TrainingItem",
    "load_training_image",
    "load_training_index",
    "select_smoke0_items",
]

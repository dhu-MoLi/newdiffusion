"""Immutable, geometry-normalized inputs for Module A RC1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np
from PIL import Image

from cadiff.data.geometry import resize_pad_image, resize_pad_mask

from .identities import IdentityRecord


class _ImageConfig(Protocol):
    image_size: int


_NORMAL_ROLES = frozenset({"normal_fit", "normal_dev", "open_normal"})
_ANOMALY_ROLES = frozenset({"fewshot", "open_report", "excluded"})
_CATEGORY_TYPES = {
    "hazelnut": frozenset({"good", "crack", "cut", "hole", "print"}),
    "cable": frozenset(
        {
            "good",
            "bent_wire",
            "cut_inner_insulation",
            "cut_outer_insulation",
            "poke_insulation",
            "cable_swap",
            "combined",
            "missing_cable",
            "missing_wire",
        }
    ),
}


@dataclass(frozen=True)
class LoadedIdentity:
    """One v0.6-A identity on the full 512 pixel canvas."""

    record: IdentityRecord
    rgb: np.ndarray
    valid: np.ndarray
    anomaly_gt: np.ndarray | None

    def __post_init__(self) -> None:
        if self.rgb.shape != (512, 512, 3) or self.rgb.dtype != np.uint8:
            raise ValueError("rgb must be a 512x512 RGB uint8 array")
        if self.valid.shape != (512, 512) or self.valid.dtype != bool:
            raise ValueError("valid must be a 512x512 bool array")
        if self.anomaly_gt is not None and (
            self.anomaly_gt.shape != (512, 512) or self.anomaly_gt.dtype != bool
        ):
            raise ValueError("anomaly_gt must be a 512x512 bool array")
        if self.record.role in _NORMAL_ROLES and self.anomaly_gt is not None:
            raise ValueError("normal records cannot carry an anomaly ground truth")
        if self.record.role in _ANOMALY_ROLES and (
            self.anomaly_gt is None or not self.anomaly_gt.any()
        ):
            raise ValueError("anomaly records require a non-empty ground truth")
        for array in (self.rgb, self.valid, self.anomaly_gt):
            if array is not None:
                array.setflags(write=False)


def _checked_path(root: Path, relpath: str, *, kind: str) -> Path:
    pure = PurePosixPath(relpath)
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
        raise ValueError(f"{kind} path must be normalized and root-relative")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {kind}: {candidate}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{kind} escapes dataset root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {kind}: {candidate}")
    return resolved


def _validate_record(record: IdentityRecord) -> None:
    if (
        record.category not in _CATEGORY_TYPES
        or record.defect_type not in _CATEGORY_TYPES[record.category]
    ):
        raise ValueError("record category and defect type must use the frozen v0.6-A vocabulary")
    if record.role in _NORMAL_ROLES:
        if record.defect_type != "good" or record.mask_relpath is not None:
            raise ValueError("normal records must be good images without masks")
    elif record.role in _ANOMALY_ROLES:
        if record.defect_type == "good" or record.mask_relpath is None:
            raise ValueError("anomaly records require a defect type and mask")
    else:
        raise ValueError(f"unsupported identity role: {record.role}")


def _binary_source_mask(mask: Image.Image) -> None:
    values = np.unique(np.asarray(mask.convert("L"), dtype=np.uint8))
    if not set(values).issubset({0, 255}):
        raise ValueError("ground truth mask must be binary 0/255")
    if not np.any(values == 255):
        raise ValueError("ground truth mask must be non-empty")


def load_identity_record(
    record: IdentityRecord, *, dataset_root: Path, config: _ImageConfig
) -> LoadedIdentity:
    """Load one declared identity without accepting paths outside the dataset root."""

    _validate_record(record)
    if getattr(config, "image_size", None) != 512:
        raise ValueError("v0.6-A image_size must be exactly 512")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    root = dataset_root.resolve(strict=True)
    image_path = _checked_path(root, record.image_relpath, kind="image")
    with Image.open(image_path) as source:
        image, valid, geometry = resize_pad_image(source, 512)
    rgb = np.asarray(image, dtype=np.uint8).copy()
    valid_array = np.asarray(valid, dtype=np.uint8).astype(bool, copy=False)
    anomaly_gt: np.ndarray | None = None
    if record.mask_relpath is not None:
        mask_path = _checked_path(root, record.mask_relpath, kind="mask")
        with Image.open(mask_path) as source_mask:
            _binary_source_mask(source_mask)
            transformed_mask = resize_pad_mask(source_mask, geometry)
        anomaly_gt = np.asarray(transformed_mask, dtype=np.uint8).astype(bool, copy=False)
        if not anomaly_gt.any():
            raise ValueError("ground truth mask must remain non-empty after resize")
    return LoadedIdentity(record, rgb, valid_array, anomaly_gt)

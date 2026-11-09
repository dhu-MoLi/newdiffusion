from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from cadiff.data.geometry import resize_pad_image
from cadiff.p1_v1.dataset import TrainingItem, load_training_image, load_training_index
from cadiff.p2_a0.fewshot import FewshotSample, load_fewshot_split
from cadiff.p2_a1.config import P2A1Config
from cadiff.provenance import sha256_file

_MANIFEST_SHA256 = "b93b6c0758645d3beb51ee9e4c3b43e934572c3e5641345a94d6f572acb6d58d"
_ACCEPTED_REPORT_SHA256 = "9cbd22cc3380b4bca97ebb1614e832a985ea74da78440d46d9335e450ece6f93"
_ACCEPTED_REPORT_RELPATH = "p2_b1/hazelnut/heldout-generation/seed-20260817-attempt2/report.json"
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "reason",
        "training_performed",
        "checkpoint_written",
        "evaluation_mode",
        "summary",
        "peak_vram_bytes",
        "input_hashes",
        "normal_ids",
        "mask_ids",
        "source_mask_types",
    }
)
_TYPES = ("crack", "cut", "hole", "print")


@dataclass(frozen=True)
class ResponseInput:
    sample_id: str
    cohort: str
    source_defect_type: str | None
    image_relpath: str
    image_sha256: str
    pixels: Tensor
    valid: Tensor


@dataclass(frozen=True)
class MaskTarget:
    sample_id: str
    defect_type: str
    role: str
    mask_sha256: str
    mask: Tensor
    valid: Tensor


@dataclass(frozen=True)
class P3A0Data:
    normal_inputs: tuple[ResponseInput, ...]
    auxiliary_inputs: tuple[ResponseInput, ...]
    calibration_targets: tuple[MaskTarget, ...]
    auxiliary_targets: tuple[MaskTarget, ...]


def _strict_json_object(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"accepted report contains duplicate JSON key: {key}")
            result[key] = value
        return result

    if path.is_symlink():
        raise ValueError("accepted report symlink is not allowed")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("accepted report must be a regular file")
    try:
        raw = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"accepted report has non-finite JSON value: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("accepted report is invalid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("accepted report must be a JSON object")
    return raw


def _unique_strings(value: object, *, count: int, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"accepted report {field} must contain {count} unique non-empty IDs")
    return tuple(value)


def _accepted_normal_ids(path: Path, source_config: P2A1Config) -> tuple[str, ...]:
    report = _strict_json_object(path)
    if set(report) != _REPORT_KEYS:
        raise ValueError("accepted report schema is invalid")
    if report["schema_version"] != "p2_b1_generation_report_v1":
        raise ValueError("accepted report schema is incompatible")
    if report["status"] != "complete":
        raise ValueError("accepted report status must be complete")
    if (
        report["reason"] is not None
        or report["training_performed"] is not False
        or report["checkpoint_written"] is not False
        or report["evaluation_mode"] != "diagnostic_only"
    ):
        raise ValueError("accepted report identity is incompatible")
    if not isinstance(report["summary"], Mapping):
        raise ValueError("accepted report summary must be a mapping")
    if type(report["peak_vram_bytes"]) is not int or report["peak_vram_bytes"] < 0:
        raise ValueError("accepted report peak_vram_bytes is invalid")
    if not isinstance(report["input_hashes"], Mapping) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for key, value in report["input_hashes"].items()
    ):
        raise ValueError("accepted report input_hashes is invalid")
    normal_ids = _unique_strings(report["normal_ids"], count=8, field="normal_ids")
    _unique_strings(report["mask_ids"], count=8, field="mask_ids")
    source_mask_types = report["source_mask_types"]
    if (
        not isinstance(source_mask_types, list)
        or len(source_mask_types) != 8
        or any(value not in source_config.defect_types for value in source_mask_types)
        or tuple(source_mask_types) != tuple(label for label in _TYPES for _ in range(2))
    ):
        raise ValueError("accepted report source_mask_types is invalid")
    return normal_ids


def _validate_inputs(source_config: P2A1Config, image_size: int) -> None:
    if not isinstance(source_config, P2A1Config):
        raise ValueError("source_config must be a P2A1Config")
    if image_size != 512 or source_config.image_size != 512:
        raise ValueError("P3-A0 response geometry must be 512 pixels")
    if source_config.category != "hazelnut" or source_config.defect_types != _TYPES:
        raise ValueError("source_config must be the frozen Hazelnut P2-A1 profile")


def _validate_manifest_authority(manifest: Path) -> None:
    if sha256_file(manifest) != _MANIFEST_SHA256:
        raise ValueError("manifest SHA256 does not match the frozen P3-A0 authority")


def _input_from_normal(item: TrainingItem, image_size: int) -> ResponseInput:
    loaded = load_training_image(item, image_size)
    return ResponseInput(
        sample_id=item.sample_id,
        cohort="normal_reference",
        source_defect_type=None,
        image_relpath=item.image_relpath,
        image_sha256=item.image_sha256,
        pixels=loaded.pixels,
        valid=loaded.valid,
    )


def _fewshot_pixels_and_valid(sample: FewshotSample, image_size: int) -> tuple[Tensor, Tensor]:
    with Image.open(sample.image_path) as image:
        transformed, valid_image, _ = resize_pad_image(image, image_size)
    pixels = (
        torch.from_numpy(np.array(transformed, dtype=np.uint8, copy=True))
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(127.5)
        .sub_(1.0)
        .contiguous()
    )
    if pixels.shape != sample.pixels.shape or not torch.equal(pixels, sample.pixels):
        raise ValueError("fewshot transformed pixels disagree with the frozen split")
    valid = (
        torch.from_numpy(np.array(valid_image, dtype=np.uint8, copy=True))
        .ne(0)
        .unsqueeze(0)
        .contiguous()
    )
    return pixels, valid


def _from_fewshot(sample: FewshotSample, image_size: int) -> tuple[ResponseInput, MaskTarget]:
    pixels, valid = _fewshot_pixels_and_valid(sample, image_size)
    mask = sample.mask.bool().contiguous()
    if mask.shape != valid.shape:
        raise ValueError("fewshot mask geometry disagrees with its image")
    return (
        ResponseInput(
            sample_id=sample.sample_id,
            cohort="auxiliary_real",
            source_defect_type=sample.defect_type,
            image_relpath=sample.image_relpath,
            image_sha256=sample.image_sha256,
            pixels=pixels,
            valid=valid,
        ),
        MaskTarget(
            sample_id=sample.sample_id,
            defect_type=sample.defect_type,
            role=sample.role,
            mask_sha256=sample.mask_sha256,
            mask=mask,
            valid=valid,
        ),
    )


def load_p3_a0_data(
    *,
    manifest: Path,
    dataset_root: Path,
    source_config: P2A1Config,
    accepted_normal_report: Path,
    image_size: int,
) -> P3A0Data:
    _validate_inputs(source_config, image_size)
    _validate_manifest_authority(manifest)
    normal_ids = _accepted_normal_ids(accepted_normal_report, source_config)
    try:
        index = load_training_index(manifest, dataset_root, source_config)
    except FileNotFoundError as error:
        raise ValueError("manifest sample escapes dataset root or is missing") from error
    _validate_manifest_authority(manifest)
    normal_by_id = {item.sample_id: item for item in index.normals}
    if len(normal_by_id) != len(index.normals):
        raise ValueError("normal candidate IDs are not unique")
    if any(sample_id not in normal_by_id for sample_id in normal_ids):
        raise ValueError("accepted normal_ids are not present in normal candidates")
    normal_inputs = tuple(
        _input_from_normal(normal_by_id[sample_id], image_size) for sample_id in normal_ids
    )
    _validate_manifest_authority(manifest)
    split = load_fewshot_split(manifest, dataset_root, source_config)
    _validate_manifest_authority(manifest)
    calibration = tuple(_from_fewshot(sample, image_size) for sample in split.train)
    auxiliary = tuple(_from_fewshot(sample, image_size) for sample in split.evaluation)
    return P3A0Data(
        normal_inputs=normal_inputs,
        auxiliary_inputs=tuple(input_value for input_value, _ in auxiliary),
        calibration_targets=tuple(target for _, target in calibration),
        auxiliary_targets=tuple(target for _, target in auxiliary),
    )


__all__ = ["MaskTarget", "P3A0Data", "ResponseInput", "load_p3_a0_data"]

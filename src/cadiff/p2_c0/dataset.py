from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from cadiff.data.geometry import resize_pad_image, resize_pad_mask
from cadiff.p2_a0.fewshot import load_fewshot_split
from cadiff.p2_a1.config import P2A1Config
from cadiff.provenance import sha256_file


@dataclass(frozen=True)
class SegmentationExample:
    sample_id: str
    source: str
    defect_type: str | None
    image: Tensor
    mask: Tensor


@dataclass(frozen=True)
class ArmDataset:
    name: str
    normals: tuple[SegmentationExample, ...]
    anomaly_groups: tuple[tuple[SegmentationExample, ...], ...]
    seed: int

    def batch(self, *, step: int, batch_size: int) -> tuple[Tensor, Tensor, list[str]]:
        if type(step) is not int or step <= 0:
            raise ValueError("step must be a positive integer")
        if type(batch_size) is not int or batch_size <= 1 or batch_size % 2:
            raise ValueError("batch_size must be a positive even integer")
        anomaly_count = batch_size // 2
        if anomaly_count % len(self.anomaly_groups):
            raise ValueError("anomaly half-batch must divide evenly across source groups")
        examples: list[SegmentationExample] = []
        per_group = anomaly_count // len(self.anomaly_groups)
        for group_index, group in enumerate(self.anomaly_groups):
            offset = self.seed + step * per_group + group_index * 104729
            examples.extend(group[(offset + index) % len(group)] for index in range(per_group))
        normal_offset = self.seed + step * anomaly_count
        examples.extend(
            self.normals[(normal_offset + index) % len(self.normals)]
            for index in range(anomaly_count)
        )
        return (
            torch.stack([item.image for item in examples]),
            torch.stack([item.mask for item in examples]),
            [item.source for item in examples],
        )


@dataclass(frozen=True)
class DownstreamData:
    arms: dict[str, ArmDataset]
    evaluation: tuple[SegmentationExample, ...]
    counts: dict[str, int]


def _validate_examples(
    values: tuple[SegmentationExample, ...], *, anomalous: bool, name: str
) -> None:
    if not values or len({item.sample_id for item in values}) != len(values):
        raise ValueError(f"{name} must be non-empty with unique sample IDs")
    shape = values[0].image.shape
    for item in values:
        if item.image.shape != shape or item.image.ndim != 3 or shape[0] != 3:
            raise ValueError(f"{name} images must share 3 x H x W shape")
        if item.image.dtype != torch.float32 or not torch.isfinite(item.image).all():
            raise ValueError(f"{name} images must be finite float32 tensors")
        if item.mask.shape != shape[1:] or item.mask.dtype != torch.long:
            raise ValueError(f"{name} masks must be aligned long tensors")
        has_anomaly = bool(item.mask.bool().any())
        if has_anomaly != anomalous:
            raise ValueError(f"{name} anomaly identity disagrees with its mask")


def _cutpaste(
    normals: tuple[SegmentationExample, ...], *, count: int, seed: int
) -> tuple[SegmentationExample, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("cutpaste_count must be positive")
    height, width = normals[0].mask.shape
    if min(height, width) < 8:
        raise ValueError("CutPaste images must be at least 8 x 8")
    result: list[SegmentationExample] = []
    for index in range(count):
        rng = np.random.default_rng(seed + index)
        target = normals[index % len(normals)]
        donor = normals[(index * 17 + 1) % len(normals)]
        patch_height = int(rng.integers(max(2, height // 12), max(3, height // 4)))
        patch_width = int(rng.integers(max(2, width // 12), max(3, width // 4)))
        source_row = int(rng.integers(0, height - patch_height + 1))
        source_column = int(rng.integers(0, width - patch_width + 1))
        target_row = int(rng.integers(0, height - patch_height + 1))
        target_column = int(rng.integers(0, width - patch_width + 1))
        image = target.image.clone()
        image[
            :, target_row : target_row + patch_height, target_column : target_column + patch_width
        ] = donor.image[
            :, source_row : source_row + patch_height, source_column : source_column + patch_width
        ]
        mask = torch.zeros_like(target.mask)
        mask[
            target_row : target_row + patch_height, target_column : target_column + patch_width
        ] = 1
        result.append(
            SegmentationExample(
                sample_id=f"cutpaste-{seed}-{index}",
                source="cutpaste",
                defect_type="cutpaste",
                image=image,
                mask=mask,
            )
        )
    return tuple(result)


def build_arm_datasets(
    *,
    normals: tuple[SegmentationExample, ...],
    real_anomalies: tuple[SegmentationExample, ...],
    generated_anomalies: tuple[SegmentationExample, ...],
    cutpaste_count: int,
    seed: int,
) -> dict[str, ArmDataset]:
    _validate_examples(normals, anomalous=False, name="normals")
    _validate_examples(real_anomalies, anomalous=True, name="real anomalies")
    _validate_examples(generated_anomalies, anomalous=True, name="generated anomalies")
    cutpaste = _cutpaste(normals, count=cutpaste_count, seed=seed + 30_000_000)
    return {
        "real_fs": ArmDataset("real_fs", normals, (real_anomalies,), seed),
        "gen_only": ArmDataset("gen_only", normals, (generated_anomalies,), seed),
        "real_plus_gen": ArmDataset(
            "real_plus_gen", normals, (real_anomalies, generated_anomalies), seed
        ),
        "cutpaste": ArmDataset("cutpaste", normals, (cutpaste,), seed),
    }


def _strict_object(line: str, line_number: int) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key on line {line_number}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            line,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value on line {line_number}: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON on line {line_number}") from error
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number} must contain a JSON object")
    return value


def _contained(root: Path, relpath: str) -> Path:
    path = (root / relpath).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("sample path escapes dataset root") from error
    if not path.is_file():
        raise ValueError("sample path must be a regular file")
    return path


def _load_pixels(image_path: Path, mask_path: Path | None, size: int) -> tuple[Tensor, Tensor]:
    with Image.open(image_path) as image:
        transformed, _, geometry = resize_pad_image(image, size)
    pixels = (
        torch.from_numpy(np.array(transformed, dtype=np.uint8, copy=True))
        .permute(2, 0, 1)
        .to(torch.float32)
        .div_(255.0)
        .contiguous()
    )
    if mask_path is None:
        mask = torch.zeros((size, size), dtype=torch.long)
    else:
        with Image.open(mask_path) as source:
            transformed_mask = resize_pad_mask(source, geometry)
        mask = torch.from_numpy(
            np.array(transformed_mask, dtype=np.uint8, copy=True)
        ).ge(128).to(torch.long)
    return pixels, mask.contiguous()


def _load_normals(
    manifest: Path, dataset_root: Path, *, role: str, size: int
) -> tuple[SegmentationExample, ...]:
    values: list[SegmentationExample] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        raw = _strict_object(line, line_number)
        if raw.get("category") != "hazelnut" or role not in raw.get("roles", []):
            continue
        relpath = raw.get("image_relpath")
        digest = raw.get("image_sha256")
        if (
            raw.get("dataset") != "MVTec"
            or raw.get("split_id") != "mvtec_lexicographic_floor_third_v1"
            or raw.get("defect_type") != "good"
            or not isinstance(relpath, str)
            or not isinstance(digest, str)
        ):
            raise ValueError("normal manifest identity is invalid")
        expected_prefix = (
            "hazelnut/train/good/"
            if role == "normal_train"
            else "hazelnut/test/good/"
        )
        if not relpath.startswith(expected_prefix):
            raise ValueError("normal path does not match its role")
        path = _contained(dataset_root, relpath)
        if sha256_file(path) != digest:
            raise ValueError("normal image SHA256 mismatch")
        image, mask = _load_pixels(path, None, size)
        values.append(SegmentationExample(digest, role, None, image, mask))
    if not values:
        raise ValueError(f"manifest contains no {role} normals")
    return tuple(values)


def _from_fewshot(sample: Any, *, source: str, size: int) -> SegmentationExample:
    image, mask = _load_pixels(sample.image_path, sample.mask_path, size)
    return SegmentationExample(sample.sample_id, source, sample.defect_type, image, mask)


def _load_synthetic(manifest: Path, root: Path, *, size: int) -> tuple[SegmentationExample, ...]:
    values: list[SegmentationExample] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        raw = _strict_object(line, line_number)
        if (
            raw.get("schema_version") != "p2_c0_synthetic_record_v1"
            or raw.get("role") != "synthetic_train"
        ):
            raise ValueError("synthetic record schema is invalid")
        sample_id = raw.get("job_id")
        defect_type = raw.get("defect_type")
        if not isinstance(sample_id, str) or sample_id in seen or not isinstance(defect_type, str):
            raise ValueError("synthetic identity is invalid")
        if raw.get("mask_type") != defect_type:
            raise ValueError("synthetic mask and defect types differ")
        image_path = _contained(root, str(raw.get("image_relpath")))
        mask_path = _contained(root, str(raw.get("mask_relpath")))
        if (
            sha256_file(image_path) != raw.get("image_sha256")
            or sha256_file(mask_path) != raw.get("mask_sha256")
        ):
            raise ValueError("synthetic artifact SHA256 mismatch")
        image, mask = _load_pixels(image_path, mask_path, size)
        values.append(SegmentationExample(sample_id, "generated", defect_type, image, mask))
        seen.add(sample_id)
    if not values:
        raise ValueError("synthetic manifest is empty")
    return tuple(values)


def load_downstream_data(
    *,
    manifest: Path,
    dataset_root: Path,
    source_config: P2A1Config,
    synthetic_root: Path,
    image_size: int,
    seed: int,
) -> DownstreamData:
    root = dataset_root.resolve(strict=True)
    split = load_fewshot_split(manifest, root, source_config)
    real_train = tuple(
        _from_fewshot(item, source="real_fs", size=image_size)
        for item in split.train
    )
    real_eval = tuple(
        _from_fewshot(item, source="heldout_real", size=image_size)
        for item in split.evaluation
    )
    normals = _load_normals(manifest, root, role="normal_train", size=image_size)
    eval_normals = _load_normals(manifest, root, role="open_dev_report_normal", size=image_size)
    generated = _load_synthetic(
        synthetic_root / "synthetic_manifest.jsonl", synthetic_root, size=image_size
    )
    arms = build_arm_datasets(
        normals=normals,
        real_anomalies=real_train,
        generated_anomalies=generated,
        cutpaste_count=len(generated),
        seed=seed,
    )
    evaluation = eval_normals + real_eval
    return DownstreamData(
        arms=arms,
        evaluation=evaluation,
        counts={
            "normal_train": len(normals),
            "real_fs": len(real_train),
            "generated": len(generated),
            "cutpaste": len(generated),
            "heldout_normal": len(eval_normals),
            "heldout_real": len(real_eval),
        },
    )

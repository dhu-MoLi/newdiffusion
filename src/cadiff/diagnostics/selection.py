from __future__ import annotations

import dataclasses
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from cadiff.data.mvtec_manifest import ManifestRecord
from cadiff.p0 import P0_CATEGORIES, P0_SAMPLE_COUNT
from cadiff.provenance import sha256_file


@dataclass(frozen=True)
class SelectedDiagnosticSample:
    category: str
    image_relpath: str
    image_sha256: str
    baseline_extra_l1: float
    selection_role: str


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_report(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(raw, Mapping):
        raise ValueError("frozen metrics must be a JSON mapping")
    if raw.get("schema_version") != "p0_gate_a_metrics_v1":
        raise ValueError("frozen metrics schema_version is invalid")
    if raw.get("gate_revision") != "gate_a_v1":
        raise ValueError("frozen metrics gate revision is invalid")
    return raw


def _canonical_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("metric image_relpath must be a non-empty string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError("metric image_relpath must be canonical and relative")
    return value


def _finite_extra(value: object) -> float:
    if type(value) not in {float, int} or isinstance(value, bool):
        raise ValueError("metric extra_l1 must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("metric extra_l1 must be a finite non-negative number")
    return result


def _source_within_root(dataset_root: Path, relative: str) -> Path:
    candidate = dataset_root.joinpath(*PurePosixPath(relative).parts)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"source {resolved} escapes dataset root {dataset_root}") from error
    if not resolved.is_file():
        raise ValueError(f"source is not a file: {resolved}")
    return resolved


def select_diagnostic_samples(
    metrics_path: Path,
    expected_metrics_sha256: str,
    records: Iterable[ManifestRecord],
    dataset_root: Path,
) -> tuple[SelectedDiagnosticSample, ...]:
    if sha256_file(metrics_path) != expected_metrics_sha256:
        raise ValueError("frozen metrics SHA256 does not match the expected value")
    report = _strict_report(metrics_path)
    raw_images = report.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("frozen metrics images must be a list")
    record_list = list(records)
    by_path = {record.image_relpath: record for record in record_list}
    if len(by_path) != len(record_list):
        raise ValueError("manifest image paths must be unique")

    dataset_root = dataset_root.resolve(strict=True)
    grouped: dict[str, list[tuple[float, str, ManifestRecord]]] = defaultdict(list)
    seen: set[str] = set()
    for index, raw_image in enumerate(raw_images):
        if not isinstance(raw_image, Mapping):
            raise ValueError(f"metrics images[{index}] must be a mapping")
        category = raw_image.get("category")
        if category not in P0_CATEGORIES:
            raise ValueError(f"metrics images[{index}] category is invalid")
        relative = _canonical_relative(raw_image.get("image_relpath"))
        if relative in seen:
            raise ValueError("metric image paths must be unique")
        seen.add(relative)
        if raw_image.get("finite") is not True or raw_image.get("status") != "complete":
            raise ValueError("all frozen metric rows must be finite complete samples")
        extra_l1 = _finite_extra(raw_image.get("extra_l1"))
        record = by_path.get(relative)
        if record is None:
            raise ValueError(f"metric row has no manifest record: {relative}")
        if (
            record.dataset != "MVTec"
            or record.category != category
            or record.defect_type != "good"
            or record.roles != ("normal_train",)
            or record.mask_relpath is not None
        ):
            raise ValueError(f"metric row is not a normal_train/good record: {relative}")
        source = _source_within_root(dataset_root, relative)
        if sha256_file(source) != record.image_sha256:
            raise ValueError(f"source SHA256 does not match manifest: {relative}")
        grouped[str(category)].append((extra_l1, relative, record))

    if set(grouped) != set(P0_CATEGORIES):
        raise ValueError("frozen metrics must contain the exact P0 categories")
    selected: list[SelectedDiagnosticSample] = []
    for category in P0_CATEGORIES:
        candidates = sorted(grouped[category], key=lambda item: (item[0], item[1]))
        if len(candidates) != P0_SAMPLE_COUNT:
            raise ValueError(f"{category} must contain exactly 12 frozen metric rows")
        low = candidates[0]
        high = candidates[-1]
        if low[1] == high[1]:
            raise ValueError(f"{category} diagnostic min/max samples must be distinct")
        for role, candidate in (("min", low), ("max", high)):
            extra_l1, relative, record = candidate
            selected.append(
                SelectedDiagnosticSample(
                    category,
                    relative,
                    record.image_sha256,
                    extra_l1,
                    role,
                )
            )
    return tuple(selected)


def selection_document(
    samples: Sequence[SelectedDiagnosticSample], metrics_sha256: str
) -> dict[str, object]:
    if len(samples) != len(P0_CATEGORIES) * 2:
        raise ValueError("diagnostic selection must contain exactly six samples")
    return {
        "schema_version": "gate_a_diagnostic_selection_v1",
        "frozen_metrics_sha256": metrics_sha256,
        "samples": [dataclasses.asdict(sample) for sample in samples],
    }

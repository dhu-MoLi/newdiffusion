from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cadiff.config import load_config
from cadiff.p2_a0.fewshot import load_fewshot_split
from cadiff.p2_a0.fewshot_config import P2A0FewshotConfig
from cadiff.p2_b2.config import load_p2_b2_config
from cadiff.provenance import sha256_file, write_json_atomic

SCHEMA_VERSION = "p2_b4_real_texture_diagnostic_v1"
METRIC_RECORD_SCHEMA_VERSION = "p2_b4_texture_metric_v1"
METRIC_NAMES = (
    "luminance_std",
    "gradient_energy",
    "laplacian_energy",
    "high_frequency_residual",
    "color_entropy",
)
TEXTURE_TRIGGER_NAMES = (
    "gradient_energy",
    "laplacian_energy",
    "high_frequency_residual",
    "color_entropy",
)
CONTACT_TILE_SIZE = 128
CONTACT_SHEET_SIZE = (CONTACT_TILE_SIZE * 4, CONTACT_TILE_SIZE * 2)
_P2_B3_REQUIRED = frozenset(
    {
        "schema_version",
        "pair_id",
        "defect_type",
        "source_mask_type",
        "mask_relpath",
        "mask_sha256",
        "candidate",
    }
)
_P2_B3_REPORT_KEYS = frozenset(
    {
        "blind_package_relpath",
        "blind_pair_count",
        "causal_difference",
        "checkpoint_effect_detected",
        "checkpoint_lineages",
        "checkpoint_written",
        "evaluation_mode",
        "identical_pair_count",
        "input_hashes",
        "peak_vram_bytes",
        "promotion_rule",
        "reason",
        "record_count",
        "schema_version",
        "status",
        "training_performed",
        "visual_status",
    }
)
_APPROVED_DATASET = "MVTec"
_APPROVED_SPLIT_ID = "mvtec_lexicographic_floor_third_v1"


@dataclass(frozen=True)
class TextureSample:
    source: str
    sample_id: str
    defect_type: str
    image_relpath: str
    mask_relpath: str
    image_path: Path
    mask_path: Path
    image_sha256: str
    mask_sha256: str


def _strict_json_object(line: str, line_number: int, *, name: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(line, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} line {line_number} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} line {line_number} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, name: str) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{name} line {line_number} must not be blank")
            records.append(_strict_json_object(line, line_number, name=name))
    if not records:
        raise ValueError(f"{name} must not be empty")
    return records


def _read_json_file(path: Path, *, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    return _strict_json_object(path.read_text(encoding="utf-8"), 1, name=name)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _relpath(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty normalized relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ValueError(f"{name} must be a non-empty normalized relative path")
    return value


def _contained_file(root: Path, relpath: str, *, name: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{name} root must be a regular directory: {root}")
    root = root.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relpath).parts)
    current = root
    for part in PurePosixPath(relpath).parts:
        current = current / part
        try:
            node = os.lstat(current)
        except OSError as error:
            raise ValueError(f"{name} is unavailable: {candidate}") from error
        if stat.S_ISLNK(node.st_mode):
            raise ValueError(f"{name} contains a symlink: {current}")
    if not candidate.is_file():
        raise ValueError(f"{name} must be a regular file: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes root: {candidate}") from error
    return resolved


def _validate_p2_b3_report(root: Path) -> None:
    report = _read_json_file(root / "report.json", name="P2-B3 report")
    if set(report) != _P2_B3_REPORT_KEYS:
        raise ValueError("P2-B3 report has an invalid exact schema")
    expected = {
        "schema_version": "p2_b3_checkpoint_paired_probe_report_v1",
        "status": "complete",
        "reason": None,
        "evaluation_mode": "inference_only_checkpoint_probe",
        "causal_difference": "checkpoint_weights_only",
        "record_count": 16,
        "blind_pair_count": 16,
        "visual_status": "pending_human_review",
        "training_performed": False,
        "checkpoint_written": False,
        "blind_package_relpath": "blind_package",
    }
    if any(report[field] != value for field, value in expected.items()):
        raise ValueError("P2-B3 report violates the complete successful-run contract")
    identical_count = report["identical_pair_count"]
    effect = report["checkpoint_effect_detected"]
    if (
        type(identical_count) is not int
        or not 0 <= identical_count <= 16
        or type(effect) is not bool
        or effect != (identical_count < 16)
    ):
        raise ValueError("P2-B3 report checkpoint effect summary is invalid")
    if type(report["peak_vram_bytes"]) is not int or report["peak_vram_bytes"] < 0:
        raise ValueError("P2-B3 report peak_vram_bytes is invalid")
    if not isinstance(report["checkpoint_lineages"], Mapping) or set(
        report["checkpoint_lineages"]
    ) != {"baseline", "candidate"}:
        raise ValueError("P2-B3 report checkpoint lineages are invalid")
    if not isinstance(report["input_hashes"], Mapping) or not isinstance(
        report["promotion_rule"], Mapping
    ):
        raise ValueError("P2-B3 report authority fields are invalid")


def _validate_paired_record_paths(
    record: Mapping[str, object], root: Path, pair_id: str, index: int
) -> tuple[str, str, str, str]:
    expected_candidate = f"paired/{pair_id}_candidate.png"
    expected_mask = f"paired/{pair_id}_mask.png"
    expected_baseline = f"paired/{pair_id}_baseline.png"
    expected_reference = f"paired/{pair_id}_reference.png"
    candidate = record["candidate"]
    baseline = record.get("baseline")
    if not isinstance(candidate, Mapping) or not isinstance(baseline, Mapping):
        raise ValueError(f"P2-B3 record {index} paired conditions must be objects")
    candidate_image = _relpath(candidate.get("image_relpath"), name="candidate image_relpath")
    mask_relpath = _relpath(record["mask_relpath"], name="candidate mask_relpath")
    baseline_image = _relpath(baseline.get("image_relpath"), name="baseline image_relpath")
    reference_image = _relpath(
        record.get("reference_image_relpath"), name="reference image_relpath"
    )
    if (
        candidate_image != expected_candidate
        or mask_relpath != expected_mask
        or baseline_image != expected_baseline
        or reference_image != expected_reference
    ):
        raise ValueError(f"P2-B3 record {index} has an invalid paired candidate binding")
    candidate_hash = _sha256(candidate.get("image_sha256"), name="candidate image_sha256")
    mask_hash = _sha256(record["mask_sha256"], name="candidate mask_sha256")
    baseline_hash = _sha256(baseline.get("image_sha256"), name="baseline image_sha256")
    reference_hash = _sha256(record.get("reference_image_sha256"), name="reference image_sha256")
    if candidate_hash == baseline_hash or candidate_hash == reference_hash:
        raise ValueError(f"P2-B3 record {index} candidate hash is not distinct")
    for relpath, digest, name in (
        (candidate_image, candidate_hash, "candidate image"),
        (mask_relpath, mask_hash, "candidate mask"),
        (baseline_image, baseline_hash, "baseline image"),
        (reference_image, reference_hash, "reference image"),
    ):
        if sha256_file(_contained_file(root, relpath, name=name)) != digest:
            raise ValueError(f"P2-B3 record {index} {name} SHA256 mismatch")
    return candidate_image, mask_relpath, candidate_hash, mask_hash


def load_candidate_samples(
    p2_b3_output: str | Path, defect_types: Sequence[str]
) -> tuple[TextureSample, ...]:
    root = Path(p2_b3_output)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("P2-B3 output must be a regular non-symlink directory")
    root = root.resolve(strict=True)
    _validate_p2_b3_report(root)
    records = _read_jsonl(root / "records.jsonl", name="P2-B3 records")
    labels = tuple(defect_types)
    samples: list[TextureSample] = []
    pair_ids: set[str] = set()
    for index, record in enumerate(records):
        if not _P2_B3_REQUIRED <= set(record):
            raise ValueError(f"P2-B3 record {index} is missing required fields")
        if record["schema_version"] != "p2_b3_paired_record_v1":
            raise ValueError(f"P2-B3 record {index} has an unexpected schema_version")
        pair_id = record["pair_id"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise ValueError(f"P2-B3 record {index} has an invalid pair_id")
        pair_ids.add(pair_id)
        defect_type = record["defect_type"]
        if defect_type not in labels or record["source_mask_type"] != defect_type:
            raise ValueError(f"P2-B3 record {index} has an invalid matched defect type")
        candidate = record["candidate"]
        if not isinstance(candidate, Mapping):
            raise ValueError(f"P2-B3 record {index} candidate must be an object")
        if (
            candidate.get("checkpoint_role") != "candidate"
            or candidate.get("data_role") != "heldout"
        ):
            raise ValueError(f"P2-B3 record {index} must be a heldout candidate")
        image_relpath, mask_relpath, image_sha256, mask_sha256 = _validate_paired_record_paths(
            record, root, pair_id, index
        )
        image_path = _contained_file(root, image_relpath, name="candidate image")
        mask_path = _contained_file(root, mask_relpath, name="candidate mask")
        samples.append(
            TextureSample(
                source="candidate",
                sample_id=pair_id,
                defect_type=defect_type,
                image_relpath=image_relpath,
                mask_relpath=mask_relpath,
                image_path=image_path,
                mask_path=mask_path,
                image_sha256=image_sha256,
                mask_sha256=mask_sha256,
            )
        )
    if len(samples) != 16:
        raise ValueError("P2-B3 candidate records must contain exactly 16 samples")
    counts = {label: sum(item.defect_type == label for item in samples) for label in labels}
    if counts != {label: 4 for label in labels}:
        raise ValueError("P2-B3 candidate records must contain exactly four samples per type")
    return tuple(sorted(samples, key=lambda item: item.sample_id))


def _fewshot_config(category: str, defect_types: Sequence[str]) -> P2A0FewshotConfig:
    labels = tuple(defect_types)
    return P2A0FewshotConfig(
        schema_version="p2_b4_fewshot_selector_adapter_v1",
        category=category,
        defect_types=labels,
        normal_prompt="a photo of a normal hazelnut",
        image_size=512,
        dtype="float16",
        seed=20260817,
        learning_rate=1e-4,
        steps=1,
        capacity_defect_type=labels[0],
        context_fill_seed=0,
        noise_seed=0,
        eval_timesteps=(1,),
        inside_noise_weight=1.0,
        lambda_inside=1.0,
        lambda_outside=1.0,
        train_role="fewshot_train",
        eval_role="open_dev_report",
        wrong_type_policy="cyclic_next",
        correct_wrong_ratio_limit=1.0,
        correct_normal_ratio_limit=1.0,
    )


def _bind_real_manifest_identity(manifest: Path, category: str, samples: Sequence[object]) -> None:
    expected = {str(sample.image_relpath): sample for sample in samples}
    bound: set[str] = set()
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        record = _strict_json_object(raw, line_number, name="manifest")
        image_relpath = record.get("image_relpath")
        if image_relpath not in expected:
            continue
        sample = expected[str(image_relpath)]
        defect_type = sample.defect_type
        expected_mask = (
            f"{category}/ground_truth/{defect_type}/{Path(str(image_relpath)).stem}_mask.png"
        )
        expected_image_prefix = f"{category}/test/{defect_type}/"
        if (
            record.get("dataset") != _APPROVED_DATASET
            or record.get("split_id") != _APPROVED_SPLIT_ID
            or not str(image_relpath).startswith(expected_image_prefix)
            or record.get("mask_relpath") != expected_mask
            or record.get("defect_type") != defect_type
            or record.get("image_sha256") != sample.image_sha256
            or record.get("mask_sha256") != sample.mask_sha256
        ):
            raise ValueError("manifest identity binding is invalid")
        bound.add(str(image_relpath))
    if bound != set(expected):
        raise ValueError("manifest identity binding is incomplete")


def load_real_samples(
    manifest: str | Path,
    dataset_root: str | Path,
    category: str,
    defect_types: Sequence[str],
) -> tuple[TextureSample, ...]:
    manifest_path = Path(manifest)
    dataset = Path(dataset_root).resolve(strict=True)
    if not dataset.is_dir():
        raise ValueError("dataset_root must resolve to a directory")
    split = load_fewshot_split(manifest_path, dataset, _fewshot_config(category, defect_types))
    _bind_real_manifest_identity(manifest_path, category, split.evaluation)
    samples = [
        TextureSample(
            source="real",
            sample_id=item.sample_id,
            defect_type=item.defect_type,
            image_relpath=item.image_relpath,
            mask_relpath=item.mask_relpath,
            image_path=item.image_path,
            mask_path=item.mask_path,
            image_sha256=item.image_sha256,
            mask_sha256=item.mask_sha256,
        )
        for item in split.evaluation
    ]
    labels = tuple(defect_types)
    expected = {label: 12 for label in labels}
    actual = {label: sum(item.defect_type == label for item in samples) for label in labels}
    if actual != expected:
        raise ValueError(f"heldout real anomaly counts must be {expected}, got {actual}")
    return tuple(sorted(samples, key=lambda item: (item.defect_type, item.sample_id)))


def eroded_mask_interior(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("mask must be a two-dimensional boolean array")
    if not mask.any():
        raise ValueError("mask must contain at least one foreground pixel")
    if mask.shape[0] < 5 or mask.shape[1] < 5:
        return mask.copy()
    eroded = np.zeros_like(mask)
    windows = [
        mask[row : row + mask.shape[0] - 4, column : column + mask.shape[1] - 4]
        for row in range(5)
        for column in range(5)
    ]
    eroded[2:-2, 2:-2] = np.logical_and.reduce(windows)
    return eroded if eroded.any() else mask.copy()


def compute_texture_metrics(rgb: np.ndarray, interior: np.ndarray) -> dict[str, float]:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != interior.shape:
        raise ValueError("rgb and interior must be aligned RGB and mask arrays")
    if interior.dtype != bool or not interior.any():
        raise ValueError("interior must be a non-empty boolean array")
    image = rgb.astype(np.float64, copy=False)
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.hypot(gradient_y, gradient_x)
    padded = np.pad(gray, 1, mode="edge")
    laplacian = np.abs(
        4.0 * gray - padded[:-2, 1:-1] - padded[2:, 1:-1] - padded[1:-1, :-2] - padded[1:-1, 2:]
    )
    blur = (
        sum(
            padded[row : row + gray.shape[0], column : column + gray.shape[1]]
            for row in range(3)
            for column in range(3)
        )
        / 9.0
    )
    entropy: list[float] = []
    for channel in range(3):
        counts, _ = np.histogram(image[..., channel][interior], bins=32, range=(0, 256))
        probabilities = counts[counts > 0] / counts.sum()
        entropy.append(float(-(probabilities * np.log2(probabilities)).sum()))
    metrics = {
        "luminance_std": float(gray[interior].std()),
        "gradient_energy": float(gradient[interior].mean()),
        "laplacian_energy": float(laplacian[interior].mean()),
        "high_frequency_residual": float(np.abs(gray - blur)[interior].mean()),
        "color_entropy": float(np.mean(entropy)),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in metrics.values()):
        raise ValueError("texture metrics must be finite non-negative values")
    return metrics


def _load_preprocessed(sample: TextureSample) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    with Image.open(sample.mask_path) as opened:
        mask = opened.convert("L").resize((256, 256), Image.Resampling.NEAREST)
    return np.asarray(image), np.asarray(mask) >= 128


def _median(values: Sequence[float]) -> float:
    value = float(np.median(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(value):
        raise ValueError("summary median must be finite")
    return value


def _group_summary(
    rows: Sequence[tuple[TextureSample, Mapping[str, float]]],
) -> dict[str, dict[str, float | int]]:
    if not rows:
        raise ValueError("summary group must not be empty")
    return {
        name: {"count": len(rows), "median": _median([float(metrics[name]) for _, metrics in rows])}
        for name in METRIC_NAMES
    }


def summarize_texture_metrics(
    rows: Sequence[tuple[TextureSample, Mapping[str, float]]], defect_types: Sequence[str]
) -> tuple[dict[str, Any], bool, list[str]]:
    labels = tuple(defect_types)
    result: dict[str, Any] = {"per_type": {}}
    for label in (*labels, "overall"):
        selected = (
            list(rows)
            if label == "overall"
            else [row for row in rows if row[0].defect_type == label]
        )
        candidate = [row for row in selected if row[0].source == "candidate"]
        real = [row for row in selected if row[0].source == "real"]
        candidate_summary = _group_summary(candidate)
        real_summary = _group_summary(real)
        ratios: dict[str, float] = {}
        for name in METRIC_NAMES:
            denominator = float(real_summary[name]["median"])
            if denominator <= 0.0:
                raise ValueError(f"real {name} median must be positive for a ratio")
            ratios[name] = float(float(candidate_summary[name]["median"]) / denominator)
        block = {
            "candidate": candidate_summary,
            "real": real_summary,
            "candidate_to_real_median_ratio": ratios,
        }
        if label == "overall":
            result["overall"] = block
        else:
            result["per_type"][label] = block
    triggers = [
        name
        for name in TEXTURE_TRIGGER_NAMES
        if result["overall"]["candidate_to_real_median_ratio"][name] < 0.80
    ]
    return result, len(triggers) >= 2, triggers


def _crop_tile(sample: TextureSample) -> Image.Image:
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB")
    with Image.open(sample.mask_path) as opened:
        mask = np.asarray(opened.convert("L")) >= 128
    rows, columns = np.where(mask)
    if not len(rows):
        raise ValueError(f"contact-sheet mask is empty: {sample.mask_relpath}")
    crop = image.crop(
        (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)
    )
    tile = crop.resize((CONTACT_TILE_SIZE, CONTACT_TILE_SIZE), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(tile)
    label = f"{'CANDIDATE' if sample.source == 'candidate' else 'REAL'} {sample.sample_id[:8]}"
    draw.rectangle((0, 0, CONTACT_TILE_SIZE, 13), fill=(0, 0, 0))
    draw.text((2, 1), label, fill=(255, 255, 255))
    return tile


def _evenly_pick(samples: Sequence[TextureSample], count: int) -> list[TextureSample]:
    ordered = sorted(samples, key=lambda item: (item.sample_id, item.image_relpath))
    if len(ordered) <= count:
        return ordered
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def _write_contact_sheets(
    output: Path, samples: Sequence[TextureSample], defect_types: Sequence[str]
) -> None:
    contact_sheets = output / "contact_sheets"
    contact_sheets.mkdir()
    for label in defect_types:
        candidates = sorted(
            (item for item in samples if item.source == "candidate" and item.defect_type == label),
            key=lambda item: item.sample_id,
        )
        if len(candidates) != 4:
            raise ValueError("contact sheets require exactly four candidates per type")
        reals = _evenly_pick(
            [item for item in samples if item.source == "real" and item.defect_type == label], 4
        )
        sheet = Image.new("RGB", CONTACT_SHEET_SIZE, "black")
        for index, item in enumerate([*candidates, *reals]):
            sheet.paste(
                _crop_tile(item),
                ((index % 4) * CONTACT_TILE_SIZE, (index // 4) * CONTACT_TILE_SIZE),
            )
        sheet.save(contact_sheets / f"{label}.png")


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                )
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_texture_diagnostic(
    paths_config: str | Path,
    experiment_config: str | Path,
    manifest: str | Path,
    p2_b3_output: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    paths = load_config(Path(paths_config))
    experiment = load_p2_b2_config(Path(experiment_config))
    if experiment.category != "hazelnut":
        raise ValueError("P2-B4 requires the frozen Hazelnut P2-B2 config")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"P2-B4 output must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        candidates = load_candidate_samples(p2_b3_output, experiment.defect_types)
        reals = load_real_samples(
            manifest,
            paths.dataset_root,
            experiment.category,
            experiment.defect_types,
        )
        measured: list[tuple[TextureSample, dict[str, float]]] = []
        for sample in (*candidates, *reals):
            rgb, mask = _load_preprocessed(sample)
            measured.append((sample, compute_texture_metrics(rgb, eroded_mask_interior(mask))))
        summary, detected, triggers = summarize_texture_metrics(measured, experiment.defect_types)
        metric_rows = [
            {
                "schema_version": METRIC_RECORD_SCHEMA_VERSION,
                "source": sample.source,
                "sample_id": sample.sample_id,
                "defect_type": sample.defect_type,
                "image_relpath": sample.image_relpath,
                "mask_relpath": sample.mask_relpath,
                "image_sha256": sample.image_sha256,
                "mask_sha256": sample.mask_sha256,
                "metrics": metrics,
            }
            for sample, metrics in measured
        ]
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "category": experiment.category,
            "defect_types": list(experiment.defect_types),
            "candidate_count": len(candidates),
            "real_count": len(reals),
            "summary": summary,
            "texture_collapse_detected": detected,
            "trigger_metrics": triggers,
            "interpretation": (
                "descriptive texture diagnostic only; not a scientific gate or promotion decision"
            ),
        }
        _write_jsonl_atomic(temporary / "metrics.jsonl", metric_rows)
        write_json_atomic(temporary / "report.json", report)
        _write_contact_sheets(temporary, [*candidates, *reals], experiment.defect_types)
        if destination.exists():
            raise FileExistsError(f"P2-B4 output must not exist: {destination}")
        os.replace(temporary, destination)
        temporary = None
        return report
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CPU-only P2-B4 real-texture diagnostic")
    parser.add_argument("--paths-config", required=True, type=Path)
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--p2-b3-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_texture_diagnostic(
        args.paths_config,
        args.experiment_config,
        args.manifest,
        args.p2_b3_output,
        args.output,
    )
    return 0


__all__ = [
    "CONTACT_SHEET_SIZE",
    "METRIC_NAMES",
    "TextureSample",
    "build_parser",
    "compute_texture_metrics",
    "eroded_mask_interior",
    "load_candidate_samples",
    "load_real_samples",
    "main",
    "run_texture_diagnostic",
    "summarize_texture_metrics",
]

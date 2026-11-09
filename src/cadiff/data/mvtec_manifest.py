from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from cadiff.provenance import sha256_file


def exact_evenly_spaced(items: list[object], count: int) -> list[object]:
    """Return a deterministic, endpoint-preserving sample of ``items``.

    Paths sort by their POSIX spelling so this remains stable across hosts.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("count must be an exact positive integer")
    ordered = sorted(
        items,
        key=lambda item: item.as_posix() if isinstance(item, Path) else str(item),
    )
    if len(ordered) <= count:
        return ordered
    if count == 1:
        raise ValueError("subsampling count must be at least two to preserve both endpoints")
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("even-spacing produced duplicate indices")
    return [ordered[index] for index in indices]


@dataclass(frozen=True)
class ManifestRecord:
    dataset: str
    split_id: str
    category: str
    defect_type: str
    image_relpath: str
    mask_relpath: str | None
    roles: tuple[str, ...]
    image_sha256: str
    mask_sha256: str | None


def _posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _within_root(candidate: Path, root: Path) -> Path:
    if not candidate.exists():
        return candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"candidate {resolved} escapes dataset root {root}"
        ) from error
    return candidate


def _mask_for(image: Path, root: Path, category: str, defect: str) -> Path:
    mask = root / category / "ground_truth" / defect / f"{image.stem}_mask.png"
    if not mask.is_file():
        raise FileNotFoundError(f"missing mask for {image}: {mask}")
    _within_root(mask, root)
    with Image.open(image) as image_file, Image.open(mask) as mask_file:
        if image_file.size != mask_file.size:
            raise ValueError(
                f"image/mask size mismatch for {image}: {image_file.size}; "
                f"{mask}: {mask_file.size}"
            )
    return mask


def build_mvtec_manifest(root: Path, split_id: str) -> list[ManifestRecord]:
    if not isinstance(split_id, str) or not split_id.strip():
        raise ValueError("split_id must be a non-empty string")
    root = root.resolve(strict=True)
    records: list[ManifestRecord] = []
    categories = sorted(
        _within_root(path, root) for path in root.iterdir() if path.is_dir()
    )
    for category_dir in categories:
        category = category_dir.name
        train_good_dir = _within_root(category_dir / "train/good", root)
        test_good_dir = _within_root(category_dir / "test/good", root)
        train_good = sorted(train_good_dir.glob("*.png"))
        test_good = sorted(test_good_dir.glob("*.png"))
        for image in train_good:
            _within_root(image, root)
            records.append(
                ManifestRecord(
                    "MVTec",
                    split_id,
                    category,
                    "good",
                    _posix(image, root),
                    None,
                    ("normal_train",),
                    sha256_file(image),
                    None,
                )
            )
        for image in test_good:
            _within_root(image, root)
            records.append(
                ManifestRecord(
                    "MVTec",
                    split_id,
                    category,
                    "good",
                    _posix(image, root),
                    None,
                    ("open_dev_report_normal",),
                    sha256_file(image),
                    None,
                )
            )
        test_root = _within_root(category_dir / "test", root)
        defect_dirs = sorted(
            _within_root(path, root) for path in test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )
        for defect_dir in defect_dirs:
            images = sorted(defect_dir.glob("*.png"), key=lambda path: _posix(path, root))
            train_count = math.floor(len(images) / 3)
            for index, image in enumerate(images):
                _within_root(image, root)
                mask = _mask_for(image, root, category, defect_dir.name)
                roles = (
                    ("fewshot_train", "validation")
                    if index < train_count
                    else ("open_dev_report",)
                )
                records.append(
                    ManifestRecord(
                        "MVTec",
                        split_id,
                        category,
                        defect_dir.name,
                        _posix(image, root),
                        _posix(mask, root),
                        roles,
                        sha256_file(image),
                        sha256_file(mask),
                    )
                )
    return sorted(records, key=lambda record: record.image_relpath)


def write_manifest(records: list[ManifestRecord], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(output)

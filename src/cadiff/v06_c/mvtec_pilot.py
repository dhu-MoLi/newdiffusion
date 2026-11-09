"""MVTec AD inventory for the VisA exact-mask R baseline.

MVTec is already in the DualAnoDiff directory layout: ``train/good``,
``test/<defect>``, ``ground_truth/<defect>/<stem>_mask.png``. Few-shot is the
lexicographic first ``floor(n/3)`` of each defect type, the same rule as the
existing MVTec manifest and the MAGIC VisA split. Types with an empty few-shot
or held-out side are left out of R, because training requires both roles.
"""

from __future__ import annotations

import json
from pathlib import Path

from cadiff.provenance import sha256_file
from cadiff.v06_atlas_diag.extra_a import ExtraAStructureConfig
from cadiff.v06_c.visa_pilot import (
    VisaInventory,
    _type_split,
)

SPLIT_ID = "mvtec_lexicographic_floor_third_v1"
MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(folder: Path) -> tuple[Path, ...]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return tuple(sorted(files, key=lambda path: path.name))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mask_for(root: Path, category: str, defect: str, image: Path) -> Path:
    folder = root / category / "ground_truth" / defect
    candidates = (
        folder / f"{image.stem}_mask.png",
        folder / f"{image.stem}_mask{image.suffix}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing MVTec mask for {image}")


def load_mvtec_inventory(source_root: Path, category: str) -> VisaInventory:
    if category not in MVTEC_CATEGORIES:
        raise ValueError(f"MVTec category must stay {MVTEC_CATEGORIES}, got {category}")
    root = source_root.resolve(strict=True)
    category_root = root / category
    if not (category_root / "train" / "good").is_dir():
        raise FileNotFoundError(category_root / "train" / "good")
    test_root = category_root / "test"
    defect_names = tuple(
        sorted(
            path.name
            for path in test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )
    )
    if not defect_names:
        raise ValueError(f"MVTec {category} has no defect types")
    grouped: dict[str, list[dict[str, str]]] = {}
    for name in defect_names:
        images = _images(test_root / name)
        if not images:
            raise ValueError(f"MVTec {category}/{name} has no test images")
        stems = [path.stem for path in images]
        if len(stems) != len(set(stems)):
            raise ValueError(f"MVTec {category}/{name} image stems collided")
        grouped[name] = [
            {
                "image": _relative(root, image),
                "mask": _relative(root, _mask_for(root, category, name, image)),
            }
            for image in images
        ]
    types = tuple(
        _type_split(name, rows, seas=False, magic=True) for name, rows in grouped.items()
    )
    types = tuple(item for item in types if item.fewshot and item.heldout)
    if not types:
        raise ValueError(f"MVTec {category} has no defect type with floor(n/3) few-shot and held-out")
    names = tuple(item.name for item in types)
    if names != tuple(sorted(names)):
        raise RuntimeError(f"MVTec types must be lexicographic, got {names}")
    normals = _images(category_root / "train" / "good")
    if not normals:
        raise ValueError(f"MVTec {category} has no train/good images")
    if len({path.stem for path in normals}) != len(normals):
        raise ValueError(f"MVTec {category} train/good stems collided")
    test_normals = _images(category_root / "test" / "good")
    if not test_normals:
        raise ValueError(f"MVTec {category} has no test/good images")
    return VisaInventory(
        category=category,
        source_root=root,
        types=types,
        normals=normals,
        structure=ExtraAStructureConfig(category=category, defect_types=names),
        split_id=SPLIT_ID,
        collapsed_from=None,
        test_normals=test_normals,
    )


def write_mvtec_manifest(inventory: VisaInventory, output: Path) -> None:
    """Write the few-shot jsonl consumed by ``load_visa_fewshot_split``."""

    if inventory.split_id != SPLIT_ID:
        raise ValueError(f"manifest writer expected {SPLIT_ID}, got {inventory.split_id}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite MVTec manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=False)
    lines: list[str] = []
    for item in inventory.types:
        for role, group in (("fewshot_train", item.fewshot), ("open_dev_report", item.heldout)):
            for row in group:
                image = inventory.source_root.joinpath(*Path(row["image"]).parts)
                mask = inventory.source_root.joinpath(*Path(row["mask"]).parts)
                payload = {
                    "category": inventory.category,
                    "dataset": "MVTec",
                    "defect_type": item.name,
                    "image_relpath": row["image"],
                    "image_sha256": sha256_file(image),
                    "mask_relpath": row["mask"],
                    "mask_sha256": sha256_file(mask),
                    "roles": [role],
                    "split_id": inventory.split_id,
                }
                lines.append(json.dumps(payload, sort_keys=True))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

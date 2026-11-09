from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    dataset_root: Path
    model_root: Path
    runs_root: Path
    image_size: int
    num_inversion_steps: int


REQUIRED_KEYS = (
    "dataset_root",
    "model_root",
    "runs_root",
    "image_size",
    "num_inversion_steps",
)


def _existing_dir(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def load_config(path: Path) -> ProjectConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"configuration is missing required keys: {', '.join(missing)}")
    image_size = _integer(raw["image_size"], "image_size")
    if image_size != 512:
        raise ValueError("P0/V0 image_size must be 512")
    steps = _integer(raw["num_inversion_steps"], "num_inversion_steps")
    if steps <= 0:
        raise ValueError("num_inversion_steps must be positive")
    return ProjectConfig(
        dataset_root=_existing_dir(raw["dataset_root"], "dataset_root"),
        model_root=_existing_dir(raw["model_root"], "model_root"),
        runs_root=_existing_dir(raw["runs_root"], "runs_root"),
        image_size=image_size,
        num_inversion_steps=steps,
    )

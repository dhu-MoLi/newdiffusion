from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from cadiff.config import ProjectConfig


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_value(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "numpy",
        "Pillow",
        "PyYAML",
        "safetensors",
        "torchmetrics",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None


def collect_provenance(config: ProjectConfig, *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        git_commit = _git_value("rev-parse", "HEAD", cwd=cwd)
        git_dirty: bool | None = bool(_git_value("status", "--porcelain", cwd=cwd))
    except subprocess.CalledProcessError:
        git_commit = None
        git_dirty = None
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "dataset_root": str(config.dataset_root),
        "model_root": str(config.model_root),
        "runs_root": str(config.runs_root),
        "image_size": config.image_size,
        "num_inversion_steps": config.num_inversion_steps,
        "driver_version": _driver_version(),
        "dependencies": _package_versions(),
    }
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(0)
        payload["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return payload


def capture_provenance(config: ProjectConfig, output: Path) -> dict[str, Any]:
    payload = collect_provenance(config)
    write_json_atomic(output, payload)
    return payload

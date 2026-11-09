"""Run metadata: every v07 result must be traceable to a commit and a config."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).decode("utf-8", "replace").strip()
    except Exception:
        return None


def collect_run_meta(
    *,
    repo_root: Path,
    config_path: Path,
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    raw = config_path.read_bytes()
    meta: dict[str, Any] = {
        "schema": "v07_run_meta_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "seed": int(seed),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool((_git(repo_root, "status", "--porcelain") or "").strip()),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        if torch.cuda.is_available():
            meta["cuda"] = torch.cuda.get_device_name(0)
            meta["cuda_count"] = torch.cuda.device_count()
    except Exception:
        pass
    if extra:
        meta["extra"] = dict(extra)
    return meta


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


__all__ = ["append_jsonl", "collect_run_meta", "write_json"]

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from safetensors.torch import load as load_safetensors

from cadiff.p1_v1.dataset import load_training_image
from cadiff.p2_a0.fewshot import _tensor_sha256, _tensor_to_pil
from cadiff.p2_b3.config import P2B3Config
from cadiff.p2_b3.contracts import (
    build_blind_assignments,
    build_blind_records,
    validate_paired_records,
)
from cadiff.provenance import sha256_file

_PUBLIC_KEYS = {
    "schema_version",
    "blind_id",
    "defect_type",
    "panel_a_relpath",
    "panel_b_relpath",
    "reference_relpath",
    "mask_relpath",
    "panel_a_sha256",
    "panel_b_sha256",
    "reference_sha256",
    "mask_sha256",
}
_REPORT_KEYS = {
    "schema_version",
    "status",
    "reason",
    "visual_status",
    "training_performed",
    "checkpoint_written",
    "evaluation_mode",
    "causal_difference",
    "record_count",
    "blind_pair_count",
    "checkpoint_effect_detected",
    "identical_pair_count",
    "blind_package_relpath",
    "peak_vram_bytes",
    "input_hashes",
    "checkpoint_lineages",
    "promotion_rule",
}
_PROVENANCE_KEYS = {
    "schema_version",
    "git_head_start",
    "git_head_end",
    "git_dirty",
    "visual_status",
    "training_performed",
    "checkpoint_written",
    "checkpoint_effect_detected",
    "identical_pair_count",
    "evaluation_mode",
    "causal_difference",
    "input_hashes",
    "source_snapshot_before",
    "source_snapshot_after",
    "model_snapshot_before",
    "model_snapshot_after",
    "reference_snapshot_before",
    "reference_snapshot_after",
    "baseline_checkpoint_lineage",
    "candidate_checkpoint_lineage",
    "candidate_run_lineage",
    "type_prompts",
}
_REVEAL_KEYS = {
    "schema_version",
    "blind_id",
    "pair_id",
    "defect_type",
    "candidate_panel",
    "panel_a_condition",
    "panel_b_condition",
}


def _sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _strict_object_hook(path: Path):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value
    return hook


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_hook(path),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"strict JSON artifact must be an object: {path}")
    return value


def _strict_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            raise ValueError(f"blank strict JSONL line {line_number} in {path}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object_hook(path),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"invalid strict JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"strict JSONL row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"strict JSONL artifact must not be empty: {path}")
    return tuple(rows)


def _regular_file(
    stack: ExitStack, path: Path, root: Path, label: str
) -> Path:
    if path.parent.parent == Path("/proc/self/fd") and path.parent.name.isdigit():
        parent_fd = int(path.parent.name)
    else:
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            parent_flags |= os.O_CLOEXEC
        try:
            parent_fd = os.open(path.parent, parent_flags)
        except OSError as error:
            raise ValueError(f"{label} parent is not descriptor-stable") from error
        stack.callback(os.close, parent_fd)
    leaf = _leaf_name(path.name)
    try:
        named_before = os.stat(
            leaf, dir_fd=parent_fd, follow_symlinks=False
        )
    except OSError as error:
        raise ValueError(f"{label} must be a regular non-symlink file") from error
    if not stat.S_ISREG(named_before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"{label} escapes the certified tree")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        file_fd = os.open(leaf, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{label} is not descriptor-stable") from error
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        os.close(file_fd)
        raise ValueError(f"{label} must be a regular file authority")
    opened = os.fstat(file_fd)
    identity = (opened.st_dev, opened.st_ino)
    initial_stat = (
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    if (
        (named_before.st_dev, named_before.st_ino) != identity
        or (
            named_before.st_size,
            named_before.st_mtime_ns,
            named_before.st_ctime_ns,
        )
        != initial_stat
    ):
        os.close(file_fd)
        raise ValueError(f"{label} named inode changed while opening")
    initial_sha256 = hashlib.sha256(_read_all_at(file_fd)).hexdigest()

    def assert_named_identity() -> None:
        try:
            named = os.stat(
                leaf, dir_fd=parent_fd, follow_symlinks=False
            )
            current = os.fstat(file_fd)
        except OSError as error:
            raise ValueError(f"{label} named inode changed") from error
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino) != identity
            or (current.st_dev, current.st_ino) != identity
            or (
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            != initial_stat
            or hashlib.sha256(_read_all_at(file_fd)).hexdigest()
            != initial_sha256
        ):
            raise ValueError(
                f"{label} named inode/content/stat changed during validation"
            )

    stack.callback(os.close, file_fd)
    stack.callback(assert_named_identity)
    assert_named_identity()
    return Path("/proc/self/fd") / str(file_fd)


def _real_directory(path: Path, root: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real non-symlink directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"{label} escapes the certified tree")
    return resolved


def _leaf_name(name: str) -> str:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("descriptor publication requires a single leaf name")
    return name


def _temporary_name(name: str) -> str:
    return f".{name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"


def _open_temporary_at(directory_fd: int, name: str) -> tuple[int, str]:
    name = _leaf_name(name)
    temporary = _temporary_name(name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return os.open(temporary, flags, 0o644, dir_fd=directory_fd), temporary


def _unlink_owned_temporary_at(
    directory_fd: int, temporary: str, identity: tuple[int, int]
) -> None:
    try:
        named = os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino) != identity:
        raise RuntimeError("temporary publication inode changed")
    os.unlink(temporary, dir_fd=directory_fd)


def _commit_temporary_at(
    directory_fd: int,
    temporary: str,
    name: str,
    identity: tuple[int, int],
) -> None:
    try:
        os.link(
            temporary,
            _leaf_name(name),
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    finally:
        _unlink_owned_temporary_at(directory_fd, temporary, identity)


def _write_all(file_fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(file_fd, remaining)
        if written <= 0:
            raise OSError("descriptor publication made no progress")
        remaining = remaining[written:]


def _publish_bytes_at(directory_fd: int, name: str, payload: bytes) -> None:
    file_fd, temporary = _open_temporary_at(directory_fd, name)
    opened = os.fstat(file_fd)
    temporary_identity = (opened.st_dev, opened.st_ino)
    try:
        _write_all(file_fd, payload)
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = -1
        _commit_temporary_at(
            directory_fd, temporary, name, temporary_identity
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        _unlink_owned_temporary_at(
            directory_fd, temporary, temporary_identity
        )


def _copy_file_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    source_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        source_flags |= os.O_CLOEXEC
    source_fd = os.open(
        _leaf_name(source_name), source_flags, dir_fd=source_directory_fd
    )
    destination_fd: int | None = None
    temporary: str | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise ValueError("blind source must be a regular non-symlink file")
        destination_fd, temporary = _open_temporary_at(
            destination_directory_fd, destination_name
        )
        opened = os.fstat(destination_fd)
        temporary_identity = (opened.st_dev, opened.st_ino)
        while chunk := os.read(source_fd, 1024 * 1024):
            _write_all(destination_fd, chunk)
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = None
        _commit_temporary_at(
            destination_directory_fd,
            temporary,
            destination_name,
            temporary_identity,
        )
        temporary = None
        temporary_identity = None
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        if temporary is not None and temporary_identity is not None:
            _unlink_owned_temporary_at(
                destination_directory_fd, temporary, temporary_identity
            )


def _write_json_at(directory_fd: int, name: str, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    _publish_bytes_at(directory_fd, name, encoded)


def _write_jsonl_at(
    directory_fd: int, name: str, records: Sequence[Mapping[str, object]]
) -> None:
    encoded = "".join(
        json.dumps(
            record, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        + "\n"
        for record in records
    ).encode("utf-8")
    _publish_bytes_at(directory_fd, name, encoded)


def materialize_blind_package(
    output: Path | int,
    records: Sequence[Mapping[str, object]],
    public: Sequence[Mapping[str, object]],
    reveal: Sequence[Mapping[str, object]],
    *,
    blind_seed: int,
) -> None:
    if len(records) != 16 or len(public) != 16 or len(reveal) != 16:
        raise ValueError("blind publication requires exactly 16 certified pairs")
    with ExitStack() as stack:
        root_fd = _open_root_authority(stack, output, "blind output root")
        try:
            os.mkdir("blind_package", dir_fd=root_fd)
            os.mkdir("private", dir_fd=root_fd)
        except OSError as error:
            raise ValueError("blind directories cannot be created exclusively") from error
        paired_fd = _open_directory_at(stack, root_fd, "paired", "paired source")
        package_fd = _open_directory_at(
            stack, root_fd, "blind_package", "blind package"
        )
        private_fd = _open_directory_at(stack, root_fd, "private", "private reveal")
        try:
            os.mkdir("comparisons", dir_fd=package_fd)
        except OSError as error:
            raise ValueError("comparisons directory cannot be created exclusively") from error
        comparisons_fd = _open_directory_at(
            stack, package_fd, "comparisons", "blind comparisons"
        )
        record_by_pair = {str(record["pair_id"]): record for record in records}
        reveal_by_blind = {str(row["blind_id"]): row for row in reveal}
        for row in public:
            if set(row) != _PUBLIC_KEYS:
                raise ValueError("public blind row has an invalid exact schema")
            blind_id = str(row["blind_id"])
            key = reveal_by_blind.get(blind_id)
            pair_id = "" if key is None else str(key["pair_id"])
            record = record_by_pair.get(pair_id)
            if record is None or key is None:
                raise ValueError("public blind row has no certified private pairing")
            copies = (
                (
                    Path(
                        str(record[str(key["panel_a_condition"])]["image_relpath"])
                    ).name,
                    Path(str(row["panel_a_relpath"])).name,
                ),
                (
                    Path(
                        str(record[str(key["panel_b_condition"])]["image_relpath"])
                    ).name,
                    Path(str(row["panel_b_relpath"])).name,
                ),
                (
                    Path(str(record["reference_image_relpath"])).name,
                    Path(str(row["reference_relpath"])).name,
                ),
                (
                    Path(str(record["mask_relpath"])).name,
                    Path(str(row["mask_relpath"])).name,
                ),
            )
            for source_name, destination_name in copies:
                _copy_file_at(
                    paired_fd, source_name, comparisons_fd, destination_name
                )
                _assert_named_directory(
                    package_fd, "comparisons", comparisons_fd, "blind comparisons"
                )
        _write_jsonl_at(
            package_fd, "blind_manifest.jsonl", tuple(dict(row) for row in public)
        )
        _write_json_at(
            private_fd,
            "reveal_key.json",
            {
                "schema_version": "p2_b3_private_reveal_key_v1",
                "blind_seed": blind_seed,
                "records": [dict(row) for row in reveal],
            },
        )
        _assert_named_directory(root_fd, "paired", paired_fd, "paired source")
        _assert_named_directory(
            root_fd, "blind_package", package_fd, "blind package"
        )
        _assert_named_directory(root_fd, "private", private_fd, "private reveal")
        _assert_named_directory(
            package_fd, "comparisons", comparisons_fd, "blind comparisons"
        )


def _validate_report(report: Mapping[str, object], config: P2B3Config) -> None:
    if set(report) != _REPORT_KEYS:
        raise ValueError("P2-B3 report has an invalid exact schema")
    expected = {
        "schema_version": "p2_b3_checkpoint_paired_probe_report_v1",
        "status": "complete",
        "reason": None,
        "visual_status": "pending_human_review",
        "training_performed": False,
        "checkpoint_written": False,
        "evaluation_mode": config.evaluation_mode,
        "causal_difference": config.causal_difference,
        "record_count": 16,
        "blind_pair_count": 16,
        "blind_package_relpath": "blind_package",
    }
    if any(report[field] != value for field, value in expected.items()):
        raise ValueError("P2-B3 report violates the inference-only pending-review contract")
    if (
        type(report["checkpoint_effect_detected"]) is not bool
        or type(report["identical_pair_count"]) is not int
        or not 0 <= report["identical_pair_count"] <= 16
        or report["checkpoint_effect_detected"]
        != (report["identical_pair_count"] < 16)
    ):
        raise ValueError("P2-B3 report checkpoint effect summary is invalid")
    peak = report["peak_vram_bytes"]
    if type(peak) is not int or peak < 0:
        raise ValueError("P2-B3 report peak_vram_bytes must be non-negative")
    expected_lineages = {
        role: {
            "path": report["checkpoint_lineages"][role].get("path")
            if isinstance(report["checkpoint_lineages"], Mapping)
            and isinstance(report["checkpoint_lineages"].get(role), Mapping)
            else None,
            "hashes": dict(getattr(config, f"{role}_checkpoint_hashes")),
            "snapshot": report["checkpoint_lineages"][role].get("snapshot")
            if isinstance(report["checkpoint_lineages"], Mapping)
            and isinstance(report["checkpoint_lineages"].get(role), Mapping)
            else None,
        }
        for role in ("baseline", "candidate")
    }
    if report["checkpoint_lineages"] != expected_lineages:
        raise ValueError("P2-B3 report checkpoint lineages are invalid")
    expected_rule = {
        "candidate_semantic_improves_at_least_types": 3,
        "required_improvements": ["print", "hole"],
        "no_realism_regression": True,
        "no_boundary_regression": True,
    }
    if report["promotion_rule"] != expected_rule:
        raise ValueError("P2-B3 report promotion rule is invalid")
    if not isinstance(report["input_hashes"], dict):
        raise ValueError("P2-B3 report input_hashes must be a mapping")


def _validate_provenance(
    provenance: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    config: P2B3Config,
) -> None:
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("P2-B3 provenance has an invalid exact schema")
    expected = {
        "schema_version": "p2_b3_checkpoint_paired_probe_provenance_v1",
        "git_dirty": False,
        "visual_status": "pending_human_review",
        "training_performed": False,
        "checkpoint_written": False,
        "evaluation_mode": config.evaluation_mode,
        "causal_difference": config.causal_difference,
    }
    if any(provenance[field] != value for field, value in expected.items()):
        raise ValueError("P2-B3 provenance violates the frozen inference-only contract")
    if (
        type(provenance["checkpoint_effect_detected"]) is not bool
        or type(provenance["identical_pair_count"]) is not int
        or not 0 <= provenance["identical_pair_count"] <= 16
        or provenance["checkpoint_effect_detected"]
        != (provenance["identical_pair_count"] < 16)
    ):
        raise ValueError("P2-B3 provenance checkpoint effect summary is invalid")
    for field in ("git_head_start", "git_head_end"):
        value = provenance[field]
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"P2-B3 provenance {field} is invalid")
    for role in ("baseline", "candidate"):
        lineage = provenance[f"{role}_checkpoint_lineage"]
        if not isinstance(lineage, dict) or set(lineage) != {
            "path", "hashes", "snapshot_before", "snapshot_after"
        }:
            raise ValueError(f"P2-B3 {role} checkpoint lineage schema is invalid")
        if lineage["hashes"] != dict(getattr(config, f"{role}_checkpoint_hashes")):
            raise ValueError(f"P2-B3 {role} checkpoint lineage hashes are invalid")
        if lineage["snapshot_before"] != lineage["snapshot_after"]:
            raise ValueError(f"P2-B3 {role} checkpoint changed during inference")
    prompt_embeddings: dict[str, str] = {}
    for record in records:
        label = str(record["defect_type"])
        prompt_embeddings.setdefault(
            label, str(record["baseline"]["prompt_embedding_sha256"])
        )
    expected_prompts = {
        label: {
            "text": text,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_embedding_sha256": prompt_embeddings[label],
        }
        for label, text in config.prompts.items()
    }
    if provenance["type_prompts"] != expected_prompts:
        raise ValueError("P2-B3 prompt lineage is invalid")
    for field in (
        "input_hashes", "source_snapshot_before", "source_snapshot_after",
        "model_snapshot_before", "model_snapshot_after",
        "reference_snapshot_before", "reference_snapshot_after",
    ):
        if not isinstance(provenance[field], dict):
            raise ValueError(f"P2-B3 provenance {field} must be a mapping")
    for before, after in (
        ("source_snapshot_before", "source_snapshot_after"),
        ("model_snapshot_before", "model_snapshot_after"),
        ("reference_snapshot_before", "reference_snapshot_after"),
    ):
        if provenance[before] != provenance[after]:
            raise ValueError(f"P2-B3 provenance {before} changed")


def _validate_authority_payloads(
    report: Mapping[str, object],
    provenance: Mapping[str, object],
    preflight: Any,
) -> None:
    if (
        provenance.get("git_head_start") != preflight.git_head
        or provenance.get("git_head_end") != preflight.git_head
    ):
        raise ValueError("P2-B3 final Git authority differs from authenticated preflight")
    expected_inputs = dict(preflight.input_hashes)
    if (
        report.get("input_hashes") != expected_inputs
        or provenance.get("input_hashes") != expected_inputs
        or report.get("input_hashes") != provenance.get("input_hashes")
    ):
        raise ValueError("P2-B3 final input hash authority differs from preflight")
    snapshots = {
        "source_snapshot": dict(preflight.base.source_snapshot),
        "model_snapshot": dict(preflight.base.model_snapshot),
        "reference_snapshot": dict(preflight.reference_snapshot),
    }
    for name, expected in snapshots.items():
        if (
            provenance.get(f"{name}_before") != expected
            or provenance.get(f"{name}_after") != expected
        ):
            raise ValueError(f"P2-B3 final {name} authority differs from preflight")
    for role in ("baseline", "candidate"):
        authenticated = getattr(preflight, f"{role}_checkpoint")
        expected = {
            "path": str(authenticated.path),
            "hashes": dict(authenticated.hashes),
            "snapshot_before": dict(authenticated.snapshot),
            "snapshot_after": dict(authenticated.snapshot),
        }
        if provenance.get(f"{role}_checkpoint_lineage") != expected:
            raise ValueError(
                f"P2-B3 final {role} checkpoint authority differs from preflight"
            )
        report_lineage = {
            "path": str(authenticated.path),
            "hashes": dict(authenticated.hashes),
            "snapshot": dict(authenticated.snapshot),
        }
        if report.get("checkpoint_lineages", {}).get(role) != report_lineage:
            raise ValueError(
                f"P2-B3 report {role} checkpoint authority differs from preflight"
            )
    expected_candidate_run = {
        "path": str(preflight.candidate_checkpoint.path.parent),
        "hashes": dict(preflight.context.experiment.candidate_run_hashes),
        "snapshot_before": dict(preflight.candidate_run_snapshot),
        "snapshot_after": dict(preflight.candidate_run_snapshot),
    }
    if provenance.get("candidate_run_lineage") != expected_candidate_run:
        raise ValueError("P2-B3 accepted candidate run authority differs from preflight")


def _validate_frozen_scheduler_traces(
    records: Sequence[Mapping[str, object]], config: P2B3Config
) -> None:
    expected_by_config: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        for role in ("baseline", "candidate"):
            condition = record[role]
            if not isinstance(condition, Mapping):
                raise ValueError("paired scheduler condition must be a mapping")
            scheduler_config = condition["scheduler_config"]
            key = json.dumps(
                scheduler_config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if key not in expected_by_config:
                scheduler = DDIMScheduler.from_config(dict(scheduler_config))
                scheduler.set_timesteps(
                    config.denoise_steps, device=torch.device("cpu")
                )
                expected_by_config[key] = [
                    int(timestep.item()) for timestep in scheduler.timesteps
                ]
            if condition["scheduler_timesteps"] != expected_by_config[key]:
                raise ValueError(
                    f"paired record {index} {role} differs from frozen scheduler schedule"
                )


def _validate_trusted_png_contract(
    stack: ExitStack,
    root: Path,
    paired: Path,
    records: Sequence[Mapping[str, object]],
    preflight: Any,
    *,
    image_size: int,
) -> None:
    normals = {item.sample_id: item for item in preflight.base.normals}
    masks = {item.sample_id: item for item in preflight.base.masks}
    normal_arrays: dict[str, np.ndarray] = {}
    for index, (record, job) in enumerate(
        zip(records, preflight.jobs, strict=True)
    ):
        if job.normal_id not in normal_arrays:
            loaded = load_training_image(normals[job.normal_id], image_size)
            normal_arrays[job.normal_id] = np.asarray(
                _tensor_to_pil(((loaded.pixels + 1.0) / 2.0).clamp(0, 1)),
                dtype=np.uint8,
            )
        expected_normal = normal_arrays[job.normal_id]
        source_mask = masks[job.mask_id].mask
        if (
            not isinstance(source_mask, torch.Tensor)
            or source_mask.shape != (1, image_size, image_size)
        ):
            raise ValueError("authenticated source mask has an invalid shape")
        expected_mask = (
            source_mask[0].detach().cpu().numpy() * 255.0
        ).astype(np.uint8)
        mask_path = _regular_file(
            stack,
            paired / Path(str(record["mask_relpath"])).name,
            root,
            "paired trusted mask",
        )
        with Image.open(mask_path) as image:
            actual_mask = np.asarray(image.convert("L"), dtype=np.uint8)
        if not np.array_equal(actual_mask, expected_mask):
            raise ValueError(f"paired record {index} mask correspondence is invalid")
        trusted_reference = (
            preflight.reference_path / job.accepted_image_relpath
        )
        reference_path = _regular_file(
            stack,
            paired / Path(str(record["reference_image_relpath"])).name,
            root,
            "paired trusted reference",
        )
        if (
            sha256_file(trusted_reference) != job.accepted_image_sha256
            or sha256_file(reference_path) != job.accepted_image_sha256
        ):
            raise ValueError(f"paired record {index} trusted reference differs")
        outside = actual_mask == 0
        if not outside.any() or not (actual_mask != 0).any():
            raise ValueError("authenticated mask must contain inside and outside pixels")
        for role in ("baseline", "candidate"):
            generated_path = _regular_file(
                stack,
                paired / Path(str(record[role]["image_relpath"])).name,
                root,
                f"paired trusted {role}",
            )
            with Image.open(generated_path) as image:
                generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if not np.array_equal(generated[outside], expected_normal[outside]):
                raise ValueError(
                    f"paired record {index} {role} changed trusted normal exterior"
                )
            if record[role]["composite_outside_l1"] != 0.0:
                raise ValueError(f"paired record {index} exterior metric is invalid")


def _canonical_module_state_sha256(module: torch.nn.Module) -> str:
    entries = [
        {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "tensor_sha256": _tensor_sha256(tensor),
        }
        for name, tensor in sorted(module.state_dict().items())
    ]
    if not entries:
        raise ValueError("runtime authenticated module state must not be empty")
    encoded = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_lexical_non_symlink_path(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    if lexical != path:
        raise ValueError(f"{label} path is not the authenticated lexical path")
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            entry = os.stat(current, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{label} lexical ancestry changed") from error
        if stat.S_ISLNK(entry.st_mode):
            raise ValueError(f"{label} lexical ancestry became a symlink")
    return lexical


def _read_all_at(file_fd: int) -> bytes:
    size = os.fstat(file_fd).st_size
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(file_fd, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise ValueError("authenticated file ended before its pinned size")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _canonical_checkpoint_state_sha256(authenticated: Any) -> dict[str, str]:
    checkpoint = _assert_lexical_non_symlink_path(
        authenticated.path, "authenticated checkpoint"
    )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    named_directory = os.stat(checkpoint, follow_symlinks=False)
    directory_fd = os.open(checkpoint, directory_flags)
    result: dict[str, str] = {}
    try:
        opened_directory = os.fstat(directory_fd)
        directory_identity = (
            opened_directory.st_dev,
            opened_directory.st_ino,
        )
        if (
            not stat.S_ISDIR(named_directory.st_mode)
            or (named_directory.st_dev, named_directory.st_ino)
            != directory_identity
            or set(os.listdir(directory_fd))
            != {"adapter.safetensors", "conditioner.safetensors", "metadata.json"}
        ):
            raise ValueError("authenticated checkpoint directory identity changed")
        for state_name, filename in (
            ("conditioner", "conditioner.safetensors"),
            ("adapter", "adapter.safetensors"),
        ):
            expected_snapshot = dict(authenticated.snapshot[filename])
            expected_hash = authenticated.hashes[filename]
            named_before = os.stat(
                filename, dir_fd=directory_fd, follow_symlinks=False
            )
            file_flags = os.O_RDONLY | os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                file_flags |= os.O_CLOEXEC
            file_fd = os.open(filename, file_flags, dir_fd=directory_fd)
            try:
                opened_before = os.fstat(file_fd)
                identity = (opened_before.st_dev, opened_before.st_ino)
                if (
                    not stat.S_ISREG(named_before.st_mode)
                    or (named_before.st_dev, named_before.st_ino) != identity
                ):
                    raise ValueError(
                        f"authenticated checkpoint {filename} named identity changed"
                    )
                raw_before = _read_all_at(file_fd)
                raw_sha = hashlib.sha256(raw_before).hexdigest()
                if (
                    raw_sha != expected_hash
                    or expected_snapshot.get("sha256") != expected_hash
                    or opened_before.st_size != expected_snapshot.get("size")
                    or opened_before.st_mtime_ns
                    != expected_snapshot.get("mtime_ns")
                ):
                    raise ValueError(
                        f"authenticated checkpoint {filename} raw SHA/snapshot mismatch"
                    )
                state = load_safetensors(raw_before)
                raw_after = _read_all_at(file_fd)
                opened_after = os.fstat(file_fd)
                named_after = os.stat(
                    filename, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    raw_after != raw_before
                    or opened_after.st_size != opened_before.st_size
                    or opened_after.st_mtime_ns != opened_before.st_mtime_ns
                    or opened_after.st_ctime_ns != opened_before.st_ctime_ns
                    or (named_after.st_dev, named_after.st_ino) != identity
                ):
                    raise ValueError(
                        f"authenticated checkpoint {filename} changed during decode"
                    )
                entries = [
                    {
                        "name": name,
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                        "tensor_sha256": _tensor_sha256(tensor),
                    }
                    for name, tensor in sorted(state.items())
                ]
                if not entries:
                    raise ValueError(
                        "authenticated checkpoint state must not be empty"
                    )
                encoded = json.dumps(
                    entries,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    ensure_ascii=False,
                ).encode()
                result[state_name] = hashlib.sha256(encoded).hexdigest()
            finally:
                os.close(file_fd)
        named_directory_after = os.stat(checkpoint, follow_symlinks=False)
        if (
            (named_directory_after.st_dev, named_directory_after.st_ino)
            != directory_identity
            or set(os.listdir(directory_fd))
            != {"adapter.safetensors", "conditioner.safetensors", "metadata.json"}
        ):
            raise ValueError("authenticated checkpoint directory changed during decode")
        _assert_lexical_non_symlink_path(
            authenticated.path, "authenticated checkpoint"
        )
        return result
    finally:
        os.close(directory_fd)


def _validate_runtime_authority(
    authority: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    config: P2B3Config,
    preflight: Any,
) -> None:
    expected_keys = {
        "baseline_model",
        "candidate_model",
        "baseline_conditioner",
        "candidate_conditioner",
        "baseline_adapter",
        "candidate_adapter",
        "baseline_state_hashes",
        "candidate_state_hashes",
        "scheduler_config",
        "scheduler_timesteps",
        "actual_traces",
    }
    if set(authority) != expected_keys:
        raise ValueError("runtime model authority has an invalid exact schema")
    baseline_model = authority["baseline_model"]
    candidate_model = authority["candidate_model"]
    if (
        not isinstance(baseline_model, torch.nn.Module)
        or not isinstance(candidate_model, torch.nn.Module)
        or baseline_model is candidate_model
    ):
        raise ValueError("runtime authority requires distinct predictor objects")
    modules = {
        role: {
            "conditioner": authority[f"{role}_conditioner"],
            "adapter": authority[f"{role}_adapter"],
        }
        for role in ("baseline", "candidate")
    }
    if (
        baseline_model.unet is not candidate_model.unet
        or baseline_model.conditioner is not modules["baseline"]["conditioner"]
        or candidate_model.conditioner is not modules["candidate"]["conditioner"]
        or baseline_model.adapter is not modules["baseline"]["adapter"]
        or candidate_model.adapter is not modules["candidate"]["adapter"]
    ):
        raise ValueError("runtime predictors do not bind the exact shared U-Net/modules")
    checkpoint_modules = tuple(
        modules[role][name]
        for role in ("baseline", "candidate")
        for name in ("conditioner", "adapter")
    )
    if (
        any(not isinstance(module, torch.nn.Module) for module in checkpoint_modules)
        or len({id(module) for module in checkpoint_modules}) != 4
    ):
        raise ValueError("runtime checkpoint modules must be distinct objects")
    baseline_parameters = {
        id(parameter)
        for module in modules["baseline"].values()
        for parameter in module.parameters()
    }
    candidate_parameters = {
        id(parameter)
        for module in modules["candidate"].values()
        for parameter in module.parameters()
    }
    if not baseline_parameters or not candidate_parameters or (
        baseline_parameters & candidate_parameters
    ):
        raise ValueError("runtime checkpoint parameter identities are not disjoint")
    checkpoint_states = {
        role: _canonical_checkpoint_state_sha256(
            getattr(preflight, f"{role}_checkpoint")
        )
        for role in ("baseline", "candidate")
    }
    shared_unet = baseline_model.unet
    if (
        baseline_model.training
        or candidate_model.training
        or shared_unet.training
        or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in shared_unet.parameters()
        )
    ):
        raise ValueError("runtime shared U-Net is not frozen eval inference state")
    for module in checkpoint_modules:
        if module.training or any(
            parameter.dtype is not torch.float32
            or not parameter.requires_grad
            or parameter.grad is not None
            for parameter in module.parameters()
        ):
            raise ValueError(
                "runtime checkpoint module is not a clean FP32 master in eval"
            )
    for role in ("baseline", "candidate"):
        checkpoint_state = checkpoint_states[role]
        state_hashes = {
            name: _canonical_module_state_sha256(modules[role][name])
            for name in ("conditioner", "adapter")
        }
        if (
            authority[f"{role}_state_hashes"] != state_hashes
            or state_hashes != checkpoint_state
        ):
            raise ValueError(f"runtime {role} canonical state authority mismatch")
        for record in records:
            condition = record[role]
            if any(
                condition[f"{name}_state_sha256"] != digest
                for name, digest in state_hashes.items()
            ):
                raise ValueError(f"runtime {role} record state binding mismatch")
    traces = authority["actual_traces"]
    if not isinstance(traces, tuple) or len(traces) != len(records) * 2:
        raise ValueError("runtime authority requires exactly 32 immutable traces")
    expected_trace_keys = {
        "pair_id",
        "checkpoint_role",
        "checkpoint_state_identity",
        "set_timesteps_calls",
        "scheduler_steps",
        "model_forward_calls",
        "prediction_sha256",
        "output_sha256",
    }
    for trace, (record, role) in zip(
        traces,
        (
            (record, role)
            for record in records
            for role in ("baseline", "candidate")
        ),
        strict=True,
    ):
        if type(trace).__name__ != "mappingproxy" or set(trace) != expected_trace_keys:
            raise ValueError("runtime trace must be an immutable exact-schema mapping")
        checkpoint_state = checkpoint_states[role]
        condition = record[role]
        expected = {
            "pair_id": record["pair_id"],
            "checkpoint_role": role,
            "checkpoint_state_identity": checkpoint_state,
            "set_timesteps_calls": condition["scheduler_set_timesteps_calls"],
            "scheduler_steps": tuple(condition["scheduler_timesteps"]),
            "model_forward_calls": condition["model_forward_calls"],
            "prediction_sha256": condition["prediction_sha256"],
            "output_sha256": condition["image_sha256"],
        }
        if dict(trace) != expected:
            raise ValueError("runtime actual trace differs from paired record/checkpoint")
    if any(
        record[role]["scheduler_config"] != authority["scheduler_config"]
        or record[role]["scheduler_timesteps"]
        != authority["scheduler_timesteps"]
        for record in records
        for role in ("baseline", "candidate")
    ):
        raise ValueError("runtime scheduler authority differs from actual call traces")


def _validate_p2_b3_artifact_tree_opened(
    stack: ExitStack,
    root: Path,
    paired: Path,
    package: Path,
    private: Path,
    comparisons: Path,
    preflight: Any,
    *,
    runtime_authority: Mapping[str, object],
    image_size: int,
) -> None:
    jobs = preflight.jobs
    config = preflight.context.experiment
    records = _strict_jsonl(
        _regular_file(stack, root / "records.jsonl", root, "records")
    )
    validate_paired_records(records, jobs, config)
    _validate_frozen_scheduler_traces(records, config)
    _validate_runtime_authority(runtime_authority, records, config, preflight)
    _validate_trusted_png_contract(
        stack, root, paired, records, preflight, image_size=image_size
    )
    assignments = build_blind_assignments(jobs, config.blind_seed)
    public = _strict_jsonl(
        _regular_file(
            stack, package / "blind_manifest.jsonl", root, "public blind manifest"
        )
    )
    if any(set(row) != _PUBLIC_KEYS for row in public):
        raise ValueError("public blind manifest has an invalid exact schema or reveal leak")
    serialized_public = json.dumps(public, sort_keys=True)
    if any(token in serialized_public for token in (
        "baseline", "candidate", "checkpoint", "prompt", "/runs/", "source_"
    )):
        raise ValueError("public blind manifest leaks private checkpoint identity")
    reveal_key = _strict_json(
        _regular_file(stack, private / "reveal_key.json", root, "private reveal key")
    )
    if set(reveal_key) != {"schema_version", "blind_seed", "records"}:
        raise ValueError("private reveal key has an invalid exact schema")
    if (
        reveal_key["schema_version"] != "p2_b3_private_reveal_key_v1"
        or reveal_key["blind_seed"] != config.blind_seed
        or any(set(row) != _REVEAL_KEYS for row in reveal_key["records"])
    ):
        raise ValueError("private reveal key differs from the frozen assignment")
    blind_ids = {
        str(row["pair_id"]): str(row["blind_id"])
        for row in reveal_key["records"]
    }
    expected_public, expected_reveal = build_blind_records(
        records, assignments, blind_ids=blind_ids
    )
    if public != expected_public or reveal_key["records"] != list(expected_reveal):
        raise ValueError("blind artifacts differ from the certified private mapping")
    expected_paired: set[str] = set()
    expected_comparisons: set[str] = set()
    actual_output_hashes: dict[tuple[int, str], str] = {}
    for index, (record, public_row) in enumerate(zip(records, public, strict=True)):
        for rel_field, sha_field, mode in (
            ("reference_image_relpath", "reference_image_sha256", "RGB"),
            ("mask_relpath", "mask_sha256", "L"),
        ):
            artifact_name = Path(str(record[rel_field])).name
            path = _regular_file(
                stack, paired / artifact_name, root, rel_field
            )
            expected_paired.add(artifact_name)
            if sha256_file(path) != record[sha_field]:
                raise ValueError(f"paired artifact hash mismatch at record {index}")
            with Image.open(path) as image:
                image.load()
                if image.mode != mode or image.size != (image_size, image_size):
                    raise ValueError(f"paired artifact image contract mismatch at record {index}")
        for role in ("baseline", "candidate"):
            condition = record[role]
            artifact_name = Path(str(condition["image_relpath"])).name
            path = _regular_file(
                stack, paired / artifact_name, root, role
            )
            expected_paired.add(artifact_name)
            if sha256_file(path) != condition["image_sha256"]:
                raise ValueError(f"paired {role} hash mismatch at record {index}")
            actual_output_hashes[(index, role)] = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (image_size, image_size):
                    raise ValueError(f"paired {role} image contract mismatch at record {index}")
        for rel_field, sha_field, mode in (
            ("panel_a_relpath", "panel_a_sha256", "RGB"),
            ("panel_b_relpath", "panel_b_sha256", "RGB"),
            ("reference_relpath", "reference_sha256", "RGB"),
            ("mask_relpath", "mask_sha256", "L"),
        ):
            artifact_name = Path(str(public_row[rel_field])).name
            path = _regular_file(
                stack,
                comparisons / artifact_name,
                root,
                rel_field,
            )
            expected_comparisons.add(artifact_name)
            if sha256_file(path) != public_row[sha_field]:
                raise ValueError(f"public blind artifact hash mismatch at record {index}")
            with Image.open(path) as image:
                image.load()
                if image.mode != mode or image.size != (image_size, image_size):
                    raise ValueError(f"public blind image contract mismatch at record {index}")
    actual_paired = {child.name for child in paired.iterdir()}
    actual_comparisons = {child.name for child in comparisons.iterdir()}
    if actual_paired != expected_paired or actual_comparisons != expected_comparisons:
        raise ValueError("paired or public blind directory is partial or contains extras")
    report = _strict_json(
        _regular_file(stack, root / "report.json", root, "report")
    )
    provenance = _strict_json(
        _regular_file(stack, root / "provenance.json", root, "provenance")
    )
    _validate_report(report, config)
    _validate_provenance(provenance, records, config)
    identical_pair_count = sum(
        actual_output_hashes[(index, "baseline")]
        == actual_output_hashes[(index, "candidate")]
        for index in range(len(records))
    )
    expected_effect = {
        "checkpoint_effect_detected": identical_pair_count < len(records),
        "identical_pair_count": identical_pair_count,
    }
    if any(
        report[field] != value or provenance[field] != value
        for field, value in expected_effect.items()
    ):
        raise ValueError("final PNG checkpoint effect summary is not independently bound")
    _validate_authority_payloads(report, provenance, preflight)
    json.dumps(report, sort_keys=True, allow_nan=False)
    json.dumps(provenance, sort_keys=True, allow_nan=False)


def _open_directory_at(
    stack: ExitStack, parent_fd: int, name: str, label: str
) -> int:
    flags = os.O_RDONLY
    for flag_name in ("O_DIRECTORY", "O_NOFOLLOW"):
        if not hasattr(os, flag_name):
            raise RuntimeError("descriptor-safe directory access is unavailable")
        flags |= getattr(os, flag_name)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{label} is not a stable non-symlink directory") from error
    stack.callback(os.close, fd)
    return fd


def _open_root_authority(stack: ExitStack, output: Path | int, label: str) -> int:
    if isinstance(output, int):
        root_fd = os.dup(output)
    else:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            root_fd = os.open(output, flags)
        except OSError as error:
            raise ValueError(f"{label} is not descriptor-stable") from error
    identity = os.fstat(root_fd)
    if not stat.S_ISDIR(identity.st_mode):
        os.close(root_fd)
        raise ValueError(f"{label} is not a directory authority")
    stack.callback(os.close, root_fd)
    return root_fd


def _assert_named_directory(
    parent_fd: int, name: str, child_fd: int, label: str
) -> None:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(child_fd)
    except OSError as error:
        raise ValueError(f"{label} descriptor changed during validation") from error
    if (
        not stat.S_ISDIR(named.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError(f"{label} descriptor changed or became a symlink")


def validate_p2_b3_artifact_tree(
    output: Path | int,
    preflight: Any,
    *,
    runtime_authority: Mapping[str, object],
    image_size: int,
) -> None:
    with ExitStack() as stack:
        root_fd = _open_root_authority(stack, output, "P2-B3 root")
        expected_top = {
            "paired",
            "blind_package",
            "private",
            "records.jsonl",
            "provenance.json",
            "report.json",
        }
        if set(os.listdir(root_fd)) != expected_top:
            raise ValueError("P2-B3 output does not match the exact certified tree")
        paired_fd = _open_directory_at(stack, root_fd, "paired", "paired")
        package_fd = _open_directory_at(
            stack, root_fd, "blind_package", "blind package"
        )
        private_fd = _open_directory_at(stack, root_fd, "private", "private reveal")
        if set(os.listdir(package_fd)) != {"comparisons", "blind_manifest.jsonl"}:
            raise ValueError("public blind package has an invalid exact tree")
        if set(os.listdir(private_fd)) != {"reveal_key.json"}:
            raise ValueError("private reveal directory has an invalid exact tree")
        comparisons_fd = _open_directory_at(
            stack, package_fd, "comparisons", "blind comparisons"
        )
        proc = Path("/proc/self/fd")
        _validate_p2_b3_artifact_tree_opened(
            stack,
            proc / str(root_fd),
            proc / str(paired_fd),
            proc / str(package_fd),
            proc / str(private_fd),
            proc / str(comparisons_fd),
            preflight,
            runtime_authority=runtime_authority,
            image_size=image_size,
        )
        _assert_named_directory(root_fd, "paired", paired_fd, "paired")
        _assert_named_directory(
            root_fd, "blind_package", package_fd, "blind package"
        )
        _assert_named_directory(root_fd, "private", private_fd, "private reveal")
        _assert_named_directory(
            package_fd, "comparisons", comparisons_fd, "blind comparisons"
        )
        if (
            set(os.listdir(root_fd)) != expected_top
            or set(os.listdir(package_fd))
            != {"comparisons", "blind_manifest.jsonl"}
            or set(os.listdir(private_fd)) != {"reveal_key.json"}
        ):
            raise ValueError("descriptor-certified artifact tree changed")


__all__ = ["materialize_blind_package", "validate_p2_b3_artifact_tree"]

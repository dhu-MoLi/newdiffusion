from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml

from cadiff.config import ProjectConfig, load_config
from cadiff.diffusion.components import REQUIRED, load_frozen_components
from cadiff.p2_a1.config import P2A1Config, load_p2_a1_config
from cadiff.p2_b1.config import P2B1Config, load_p2_b1_config
from cadiff.p2_b2.config import P2B2Config, load_p2_b2_config
from cadiff.p3_a0.artifacts import publish_json_at, write_complete_artifacts
from cadiff.p3_a0.config import P3A0Config, load_p3_a0_config
from cadiff.p3_a0.contracts import ResponseSchedule, build_response_schedule
from cadiff.p3_a0.dataset import P3A0Data, load_p3_a0_data
from cadiff.p3_a0.evaluation import evaluate_p3_a0
from cadiff.p3_a0.response import ResponseBatch, run_paired_prompt_responses
from cadiff.provenance import collect_provenance, sha256_file

_EXPECTED_HASHES = {
    "design": "64c5f8cc60bd9991d70df5ef6abf1c2a0eeaa2e134385689d5f2067199e9330d",
    "spec": "4d899fafbcfba2e3965a01561c3a9a6452a3e1b50aa8e3351fe6bb0ea507abb4",
    "manifest": "b93b6c0758645d3beb51ee9e4c3b43e934572c3e5641345a94d6f572acb6d58d",
    "source_experiment": "8440fd086534b91a42a6fc629bd18dae62dd6014272009fda7c750d067544aaf",
    "generation_experiment": "0d661c20ba0dbdd75111d11a706ba80a2bf3571ebb6de3f5585964e2ed57b610",
    "prompt_experiment": "7bdf9e7f6b80af9301d51388af5c37ef8151904fd289406a7bcec5b95f2f2eb2",
    "renderer_manifest": "f719ee91c67a16e6d3c1658b2195d9c54849a951e21d14cd36689c26b6abbd53",
    "normal_report": "9cbd22cc3380b4bca97ebb1614e832a985ea74da78440d46d9335e450ece6f93",
}


def _git_state(cwd: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return head, dirty


def _git_is_ancestor(cwd: Path, base: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=cwd, check=False
    )
    return completed.returncode == 0


_DESIGN_BASE = "2f29f955b9f8c106b1a7e852652125f3234033b0"
_REPORT_SCHEMA = "p3_a0_hazelnut_spatial_response_report_v1"


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader whose mapping constructor rejects duplicate keys."""


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


@dataclass(frozen=True)
class _FileSnapshot:
    data: bytes
    sha256: str
    device: int
    inode: int
    size: int
    resolved: Path


def _snapshot_file(path: Path) -> _FileSnapshot:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"authority file cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"authority file must be regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            named = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"authority file identity changed while reading: {path}") from error
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or (named.st_dev, named.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(f"authority file identity changed while reading: {path}")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            raise ValueError(f"authority file size changed while reading: {path}")
        resolved = Path(os.path.realpath(f"/proc/self/fd/{descriptor}"))
        return _FileSnapshot(
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            device=before.st_dev,
            inode=before.st_ino,
            size=before.st_size,
            resolved=resolved,
        )
    finally:
        os.close(descriptor)


def _strict_yaml_bytes(data: bytes) -> dict[str, Any]:
    try:
        raw = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"renderer manifest is invalid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("renderer manifest must be a YAML mapping")
    return raw


def _strict_json_bytes(data: bytes) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"source provenance contains non-finite value: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source provenance is invalid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("source provenance must be a JSON object")
    return raw


def _snapshot_model(root: Path) -> dict[str, dict[str, int | str]]:
    resolved_root = root.resolve(strict=True)
    snapshot: dict[str, dict[str, int | str]] = {}
    for relative in REQUIRED:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"model snapshot file must be regular: {relative}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"model snapshot file escapes model_root: {relative}")
        snapshot[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return snapshot


@dataclass
class _ModelAuthority:
    root: Path
    snapshot: dict[str, dict[str, int | str]]
    temporary: tempfile.TemporaryDirectory[str]
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            self.temporary.cleanup()
            self._closed = True


def _snapshot_copy(source: int, destination: int) -> tuple[str, int]:
    try:
        fcntl.ioctl(destination, 0x40049409, source)
    except OSError as error:
        if error.errno not in {
            errno.EXDEV,
            errno.EINVAL,
            errno.ENOTTY,
            errno.EOPNOTSUPP,
        }:
            raise
        os.ftruncate(destination, 0)
        os.lseek(source, 0, os.SEEK_SET)
        os.lseek(destination, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while data := os.read(source, 1024 * 1024):
            digest.update(data)
            size += len(data)
            view = memoryview(data)
            while view:
                written = os.write(destination, view)
                if written <= 0:
                    raise OSError("model snapshot copy made no progress") from None
                view = view[written:]
        return digest.hexdigest(), size
    os.lseek(destination, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while data := os.read(destination, 1024 * 1024):
        digest.update(data)
        size += len(data)
    return digest.hexdigest(), size


def _capture_model_authority(
    root: Path, authorized: Mapping[str, Mapping[str, int | str]]
) -> _ModelAuthority:
    resolved_root = root.resolve(strict=True)
    temporary = tempfile.TemporaryDirectory(prefix=".cadiff-p3-a0-model-", dir=resolved_root.parent)
    private_root = Path(temporary.name)
    snapshot: dict[str, dict[str, int | str]] = {}
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for relative in REQUIRED:
            source_path = root / relative
            source = os.open(source_path, flags)
            try:
                before = os.fstat(source)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"model snapshot file must be regular: {relative}")
                resolved = Path(os.path.realpath(f"/proc/self/fd/{source}"))
                if not resolved.is_relative_to(resolved_root):
                    raise ValueError(f"model snapshot file escapes model_root: {relative}")
                destination_path = private_root / relative
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                destination = os.open(
                    destination_path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o400,
                )
                try:
                    digest, size = _snapshot_copy(source, destination)
                    os.fsync(destination)
                finally:
                    os.close(destination)
                after = os.fstat(source)
                try:
                    named = os.stat(source_path, follow_symlinks=False)
                except OSError as error:
                    raise ValueError(
                        f"model snapshot file identity changed while reading: {relative}"
                    ) from error
                identity_before = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                identity_after = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if identity_before != identity_after or (
                    named.st_dev,
                    named.st_ino,
                ) != (before.st_dev, before.st_ino):
                    raise ValueError(
                        f"model snapshot file identity changed while reading: {relative}"
                    )
                value = {"sha256": digest, "size": size}
                if value != dict(authorized[relative]):
                    raise ValueError("current model snapshot does not match renderer authority")
                snapshot[relative] = value
            finally:
                os.close(source)
        return _ModelAuthority(private_root, snapshot, temporary)
    except Exception:
        temporary.cleanup()
        raise


def _authorized_model_snapshot(raw: object) -> dict[str, dict[str, int | str]]:
    if not isinstance(raw, Mapping) or set(raw) != set(REQUIRED):
        raise ValueError("source provenance model snapshot has an invalid file set")
    result: dict[str, dict[str, int | str]] = {}
    for relative in REQUIRED:
        value = raw[relative]
        if not isinstance(value, Mapping):
            raise ValueError("source provenance model snapshot entry is invalid")
        digest, size = value.get("sha256"), value.get("size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size) is not int
            or size < 0
        ):
            raise ValueError("source provenance model snapshot hash or size is invalid")
        result[relative] = {"sha256": digest, "size": size}
    return result


def _renderer_authority(
    context: P3A0Context,
    renderer_snapshot: _FileSnapshot,
) -> tuple[dict[str, dict[str, int | str]], str]:
    raw = _strict_yaml_bytes(renderer_snapshot.data)
    identity = {
        "schema_version": "p2_renderer_freeze_v1",
        "renderer_id": "p2_b2_hazelnut_type_prompt_attempt1",
        "category": "hazelnut",
        "status": "frozen_default",
        "scope": "project_renderer_and_downstream_evaluation",
    }
    if any(raw.get(key) != value for key, value in identity.items()):
        raise ValueError("renderer manifest does not match the frozen Hazelnut P2-B2 identity")
    decision = raw.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("default_renderer") != identity["renderer_id"]
    ):
        raise ValueError("renderer manifest default decision is incompatible")
    source_run = raw.get("source_run")
    if not isinstance(source_run, Mapping):
        raise ValueError("renderer manifest source_run is invalid")
    relative = source_run.get("runs_root_relpath")
    hashes = source_run.get("hashes")
    if not isinstance(relative, str) or not relative:
        raise ValueError("renderer source_run path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        raise ValueError("renderer source provenance escapes runs_root")
    if not isinstance(hashes, Mapping) or not isinstance(hashes.get("provenance.json"), str):
        raise ValueError("renderer source provenance SHA256 is missing")
    root = context.paths.runs_root.resolve(strict=True)
    provenance_path = context.paths.runs_root / relative / "provenance.json"
    provenance_snapshot = _snapshot_file(provenance_path)
    if not provenance_snapshot.resolved.is_relative_to(root):
        raise ValueError("renderer source provenance escapes runs_root")
    provenance_digest = provenance_snapshot.sha256
    if provenance_digest != hashes["provenance.json"]:
        raise ValueError("renderer source provenance SHA256 changed")
    provenance = _strict_json_bytes(provenance_snapshot.data)
    if provenance.get("schema_version") != "p2_b2_type_prompt_training_provenance_v1":
        raise ValueError("renderer source provenance schema is incompatible")
    authorized = _authorized_model_snapshot(provenance.get("model_snapshot_after"))
    return authorized, provenance_digest


def _authority_paths(context: P3A0Context) -> dict[str, Path]:
    return {
        **_input_paths(context),
        "paths": context.config_paths["paths"],
        "experiment": context.config_paths["experiment"],
        "lock": context.lock,
    }


def _snapshot_authorities(context: P3A0Context) -> dict[str, _FileSnapshot]:
    return {name: _snapshot_file(path) for name, path in _authority_paths(context).items()}


@contextmanager
def _materialized_snapshots(snapshots: Mapping[str, _FileSnapshot], names: tuple[str, ...]):
    with tempfile.TemporaryDirectory(prefix="cadiff-p3-a0-authority-") as temporary:
        root = Path(temporary)
        paths: dict[str, Path] = {}
        for name in names:
            suffix = (
                ".jsonl" if name == "manifest" else ".json" if name == "normal_report" else ".yaml"
            )
            path = root / f"{name}{suffix}"
            path.write_bytes(snapshots[name].data)
            path.chmod(0o400)
            paths[name] = path
        yield paths


def _bind_configuration_objects(
    context: P3A0Context, snapshots: Mapping[str, _FileSnapshot]
) -> None:
    names = (
        "paths",
        "experiment",
        "source_experiment",
        "generation_experiment",
        "prompt_experiment",
    )
    with _materialized_snapshots(snapshots, names) as paths:
        loaded = (
            load_config(paths["paths"]),
            load_p3_a0_config(paths["experiment"]),
            load_p2_a1_config(paths["source_experiment"]),
            load_p2_b1_config(paths["generation_experiment"]),
            load_p2_b2_config(paths["prompt_experiment"]),
        )
    supplied = (
        context.paths,
        context.experiment,
        context.source_experiment,
        context.generation_experiment,
        context.prompt_experiment,
    )
    if loaded != supplied:
        raise ValueError("supplied configuration object differs from its authoritative YAML")


def _validate_cross_links(context: P3A0Context) -> None:
    experiment = context.experiment
    source = context.source_experiment
    generation = context.generation_experiment
    prompt = context.prompt_experiment
    compatible = (
        context.paths.image_size
        == experiment.image_size
        == source.image_size
        == generation.image_size
        and experiment.category
        == source.category
        == generation.category
        == prompt.category
        == "hazelnut"
        and experiment.defect_types
        == source.defect_types
        == generation.defect_types
        == prompt.defect_types
        and experiment.normal_prompt == source.normal_prompt == generation.normal_prompt
        and dict(experiment.defect_prompts) == dict(prompt.prompts)
        and experiment.dtype == source.dtype == generation.dtype == "float16"
        and experiment.timesteps == source.eval_timesteps
        and experiment.seeds == generation.sample_seeds
        and experiment.normal_count == generation.normal_reference_count
        and source.type_rank == prompt.type_rank == 8
    )
    if not compatible:
        raise ValueError("P3-A0 configuration cross-links are incompatible")


def _input_paths(context: P3A0Context) -> dict[str, Path]:
    required = {
        "paths",
        "experiment",
        "source_experiment",
        "generation_experiment",
        "prompt_experiment",
    }
    if set(context.config_paths) != required:
        raise ValueError("config_paths must contain the exact P3-A0 inputs")
    return {
        "design": context.cwd / "docs/design/research_design_v0.5_zh.md",
        "spec": context.cwd
        / "docs/superpowers/specs/2026-08-24-p3-a0-hazelnut-mask-free-spatial-response-design.md",
        "manifest": context.manifest,
        "source_experiment": context.config_paths["source_experiment"],
        "generation_experiment": context.config_paths["generation_experiment"],
        "prompt_experiment": context.config_paths["prompt_experiment"],
        "renderer_manifest": context.renderer_manifest,
        "normal_report": context.normal_report,
    }


def _output_parts(context: P3A0Context) -> tuple[str, ...]:
    if context.output.is_symlink():
        raise FileExistsError("output symlink is forbidden")
    if context.output.exists():
        raise FileExistsError("output must be absent")
    root = context.paths.runs_root.resolve(strict=True)
    candidate = Path(os.path.abspath(context.output))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("output must be a contained child of runs_root") from error
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError("output must be a contained child of runs_root")
    return relative.parts


@dataclass
class _DirectoryIdentity:
    device: int
    inode: int

    @classmethod
    def from_fd(cls, descriptor: int) -> _DirectoryIdentity:
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode):
            raise RuntimeError("retained filesystem authority is not a directory")
        return cls(value.st_dev, value.st_ino)

    def matches(self, value: os.stat_result) -> bool:
        return stat.S_ISDIR(value.st_mode) and (value.st_dev, value.st_ino) == (
            self.device,
            self.inode,
        )


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _rename_between_noreplace(
    source_fd: int, source: str, destination_fd: int, destination: str
) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_fd,
        os.fsencode(source),
        destination_fd,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    _rename_between_noreplace(parent_fd, source, parent_fd, destination)


def _verify_named_root(path: Path, descriptor: int, identity: _DirectoryIdentity) -> None:
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("configured runs_root identity changed") from error
    if not identity.matches(os.fstat(descriptor)) or not identity.matches(named):
        raise RuntimeError("configured runs_root identity changed")


def _verify_ancestry(
    descriptor: int,
    levels: int,
    root_identity: _DirectoryIdentity,
) -> None:
    current = os.dup(descriptor)
    seen: set[tuple[int, int]] = set()
    try:
        for _ in range(levels):
            value = os.fstat(current)
            identity = (value.st_dev, value.st_ino)
            if (
                not stat.S_ISDIR(value.st_mode)
                or value.st_dev != root_identity.device
                or identity in seen
                or root_identity.matches(value)
            ):
                raise RuntimeError("output descriptor ancestry escaped runs_root")
            seen.add(identity)
            parent = os.open("..", _directory_flags(), dir_fd=current)
            os.close(current)
            current = parent
        if not root_identity.matches(os.fstat(current)):
            raise RuntimeError("output descriptor ancestry escaped runs_root")
    finally:
        os.close(current)


def _verify_named_chain(
    root_fd: int,
    parts: tuple[str, ...],
    root_identity: _DirectoryIdentity,
    leaf_identity: _DirectoryIdentity,
) -> None:
    current = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(part, _directory_flags(), dir_fd=current)
            os.close(current)
            current = child
            value = os.fstat(current)
            if value.st_dev != root_identity.device:
                raise RuntimeError("configured output path crosses a filesystem boundary")
        if not leaf_identity.matches(os.fstat(current)):
            raise RuntimeError("configured output path identity changed")
    except OSError as error:
        raise RuntimeError("configured output path identity changed") from error
    finally:
        os.close(current)


def _purge_private_directory(descriptor: int) -> None:
    for name in tuple(os.listdir(descriptor)):
        value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(value.st_mode):
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            try:
                _purge_private_directory(child)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        elif stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode):
            os.unlink(name, dir_fd=descriptor)
        else:
            raise RuntimeError("private garbage contains an unsupported filesystem node")


@dataclass
class _PrivateGarbage:
    root_path: Path
    root_fd: int
    root_identity: _DirectoryIdentity
    name: str
    fd: int
    identity: _DirectoryIdentity
    tainted: bool = False
    _closed: bool = False

    def verify(self) -> None:
        if self._closed:
            raise RuntimeError("private garbage authority is closed")
        _verify_named_root(self.root_path, self.root_fd, self.root_identity)
        opened = os.fstat(self.fd)
        try:
            named = os.stat(self.name, dir_fd=self.root_fd, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError("private garbage identity changed") from error
        if (
            not self.identity.matches(opened)
            or not self.identity.matches(named)
            or opened.st_dev != self.root_identity.device
        ):
            raise RuntimeError("private garbage identity changed")
        _verify_ancestry(self.fd, 1, self.root_identity)

    def isolate(
        self,
        source_fd: int,
        source: str,
        expected: os.stat_result,
    ) -> str:
        self.verify()
        destination = f"entry-{secrets.token_hex(16)}"
        _rename_between_noreplace(source_fd, source, self.fd, destination)
        moved = os.stat(destination, dir_fd=self.fd, follow_symlinks=False)
        if (moved.st_dev, moved.st_ino) != (expected.st_dev, expected.st_ino):
            try:
                _rename_between_noreplace(self.fd, destination, source_fd, source)
            except Exception as error:
                self.tainted = True
                raise RuntimeError(
                    "private garbage retained raced entry "
                    f"{destination!r}; manual recovery required"
                ) from error
            raise RuntimeError("filesystem entry identity changed before isolation")
        self.verify()
        return destination

    def purge(self) -> None:
        self.verify()
        if self.tainted:
            raise RuntimeError("private garbage is tainted; manual recovery required")
        _purge_private_directory(self.fd)
        self.verify()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.purge()
            os.rmdir(self.name, dir_fd=self.root_fd)
        finally:
            os.close(self.fd)
            self._closed = True


@dataclass
class _OutputAuthority:
    path: Path
    fd: int
    device: int
    inode: int
    root_path: Path
    root_fd: int
    root_identity: _DirectoryIdentity
    parts: tuple[str, ...]
    garbage: _PrivateGarbage
    _closed: bool = False

    def verify(self) -> None:
        if self._closed:
            raise RuntimeError("output authority is closed")
        _verify_named_root(self.root_path, self.root_fd, self.root_identity)
        opened = os.fstat(self.fd)
        expected = _DirectoryIdentity(self.device, self.inode)
        if not expected.matches(opened) or opened.st_dev != self.root_identity.device:
            raise RuntimeError("acquired output path identity changed")
        _verify_ancestry(self.fd, len(self.parts), self.root_identity)
        _verify_named_chain(
            self.root_fd,
            self.parts,
            self.root_identity,
            expected,
        )
        self.garbage.verify()

    def retract(self, name: str, device: int, inode: int) -> None:
        if (
            self._closed
            or not isinstance(name, str)
            or not name
            or "/" in name
            or name in (".", "..")
        ):
            raise RuntimeError("publication retraction received an invalid authority")
        output_identity = _DirectoryIdentity(self.device, self.inode)
        if not output_identity.matches(os.fstat(self.fd)):
            raise RuntimeError("publication retraction lost the retained output directory")
        self.garbage.verify()
        try:
            current = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as error:
            raise RuntimeError("published artifact is unavailable for retraction") from error
        if (current.st_dev, current.st_ino) != (device, inode):
            raise RuntimeError("published artifact identity changed before retraction")
        try:
            self.garbage.isolate(self.fd, name, current)
            self.garbage.purge()
        except Exception as error:
            raise RuntimeError(
                "published artifact could not be retracted to root-anchored garbage; "
                "manual recovery required"
            ) from error

    def close(self) -> None:
        if not self._closed:
            errors: list[Exception] = []
            try:
                os.close(self.fd)
            except Exception as error:
                errors.append(error)
            try:
                self.garbage.close()
            except Exception as error:
                errors.append(error)
            try:
                os.close(self.root_fd)
            except Exception as error:
                errors.append(error)
            finally:
                self._closed = True
            if errors:
                raise RuntimeError(
                    "; ".join(f"output authority close failed: {error}" for error in errors)
                ) from errors[0]


def _acquire_private_garbage(
    root_path: Path,
    root_fd: int,
    root_identity: _DirectoryIdentity,
) -> _PrivateGarbage:
    name = f".cadiff-p3-a0-garbage-{secrets.token_hex(16)}"
    os.mkdir(name, mode=0o700, dir_fd=root_fd)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=root_fd)
        garbage = _PrivateGarbage(
            root_path,
            root_fd,
            root_identity,
            name,
            descriptor,
            _DirectoryIdentity.from_fd(descriptor),
        )
        garbage.verify()
    except Exception as error:
        if descriptor is not None:
            os.close(descriptor)
        raise RuntimeError(
            f"private garbage acquisition left {name!r}; manual recovery required"
        ) from error
    else:
        return garbage


def _verify_parent(
    root_path: Path,
    root_fd: int,
    root_identity: _DirectoryIdentity,
    parent_fd: int,
    parent_parts: tuple[str, ...],
) -> None:
    _verify_named_root(root_path, root_fd, root_identity)
    parent_identity = _DirectoryIdentity.from_fd(parent_fd)
    if parent_identity.device != root_identity.device:
        raise RuntimeError("output parent crosses a filesystem boundary")
    _verify_ancestry(parent_fd, len(parent_parts), root_identity)
    _verify_named_chain(root_fd, parent_parts, root_identity, parent_identity)


def _acquire_output(context: P3A0Context, parts: tuple[str, ...]) -> _OutputAuthority:
    root = context.paths.runs_root.resolve(strict=True)
    root_fd = os.open(root, _directory_flags())
    root_identity = _DirectoryIdentity.from_fd(root_fd)
    garbage: _PrivateGarbage | None = None
    current = os.dup(root_fd)
    output_fd: int | None = None
    installed = False
    try:
        _verify_named_root(root, root_fd, root_identity)
        garbage = _acquire_private_garbage(root, root_fd, root_identity)
        for part in parts[:-1]:
            with suppress(FileExistsError):
                os.mkdir(part, mode=0o755, dir_fd=current)
            next_fd = os.open(part, _directory_flags(), dir_fd=current)
            os.close(current)
            current = next_fd
        leaf = parts[-1]
        staging = f"stage-{secrets.token_hex(16)}"
        os.mkdir(staging, mode=0o700, dir_fd=garbage.fd)
        output_fd = os.open(staging, _directory_flags(), dir_fd=garbage.fd)
        opened = os.fstat(output_fd)
        named = os.stat(staging, dir_fd=garbage.fd, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("private output directory identity changed during acquisition")
        _verify_parent(root, root_fd, root_identity, current, parts[:-1])
        garbage.verify()
        _rename_between_noreplace(garbage.fd, staging, current, leaf)
        installed = True
        authority = _OutputAuthority(
            path=root.joinpath(*parts),
            fd=output_fd,
            device=opened.st_dev,
            inode=opened.st_ino,
            root_path=root,
            root_fd=root_fd,
            root_identity=root_identity,
            parts=parts,
            garbage=garbage,
        )
        authority.verify()
        os.close(current)
        return authority
    except Exception as error:
        cleanup_errors: list[Exception] = []
        if installed and output_fd is not None and garbage is not None:
            try:
                garbage.isolate(current, parts[-1], os.fstat(output_fd))
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if output_fd is not None:
            try:
                os.close(output_fd)
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if garbage is not None:
            try:
                garbage.close()
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            os.close(current)
        except Exception as cleanup_error:
            cleanup_errors.append(cleanup_error)
        try:
            os.close(root_fd)
        except Exception as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            details = "; ".join(str(value) for value in cleanup_errors)
            raise RuntimeError(
                f"output acquisition failed: {error}; cleanup failed: {details}"
            ) from error
        if isinstance(error, FileExistsError):
            raise FileExistsError("output must be absent") from error
        raise


@dataclass(frozen=True)
class P3A0Context:
    paths: ProjectConfig
    experiment: P3A0Config
    source_experiment: P2A1Config
    generation_experiment: P2B1Config
    prompt_experiment: P2B2Config
    renderer_manifest: Path
    normal_report: Path
    manifest: Path
    lock: Path
    output: Path
    config_paths: Mapping[str, Path]
    cwd: Path


@dataclass(frozen=True)
class P3A0Preflight:
    context: P3A0Context
    output: Path
    git_head: str
    input_hashes: dict[str, str]
    model_authority: dict[str, dict[str, int | str]]
    data: P3A0Data
    schedule: ResponseSchedule


@dataclass(frozen=True)
class P3A0Backend:
    respond: Callable[[P3A0Preflight, torch.device], ResponseBatch]


def _validate_p3_a0_preflight_retained(
    context: P3A0Context,
) -> tuple[P3A0Preflight, _OutputAuthority, _ModelAuthority]:
    if not isinstance(context, P3A0Context):
        raise ValueError("context must be a P3A0Context")
    if not isinstance(context.paths, ProjectConfig):
        raise ValueError("paths must be a ProjectConfig")
    if not isinstance(context.experiment, P3A0Config):
        raise ValueError("experiment must be a P3A0Config")
    if not isinstance(context.source_experiment, P2A1Config):
        raise ValueError("source_experiment must be a P2A1Config")
    if not isinstance(context.generation_experiment, P2B1Config):
        raise ValueError("generation_experiment must be a P2B1Config")
    if not isinstance(context.prompt_experiment, P2B2Config):
        raise ValueError("prompt_experiment must be a P2B2Config")
    _validate_cross_links(context)
    output_parts = _output_parts(context)
    head, dirty = _git_state(context.cwd)
    if dirty:
        raise ValueError("P3-A0 preflight requires a clean Git worktree")
    if len(head) != 40 or not _git_is_ancestor(context.cwd, _DESIGN_BASE):
        raise ValueError("frozen design commit must be an ancestor of P3-A0 HEAD")

    snapshots = _snapshot_authorities(context)
    _bind_configuration_objects(context, snapshots)
    input_hashes: dict[str, str] = {}
    for name in _input_paths(context):
        digest = snapshots[name].sha256
        if digest != _EXPECTED_HASHES[name]:
            raise ValueError(f"{name} SHA256 does not match the frozen authority")
        input_hashes[name] = digest
    input_hashes["paths"] = snapshots["paths"].sha256
    input_hashes["experiment"] = snapshots["experiment"].sha256
    input_hashes["lock"] = snapshots["lock"].sha256

    authorized_model, source_provenance_hash = _renderer_authority(
        context, snapshots["renderer_manifest"]
    )
    input_hashes["renderer_source_provenance"] = source_provenance_hash
    model_authority = _capture_model_authority(context.paths.model_root, authorized_model)
    try:
        with _materialized_snapshots(snapshots, ("manifest", "normal_report")) as paths:
            data = load_p3_a0_data(
                manifest=paths["manifest"],
                dataset_root=context.paths.dataset_root,
                source_config=context.source_experiment,
                accepted_normal_report=paths["normal_report"],
                image_size=context.experiment.image_size,
            )
        schedule = build_response_schedule(
            tuple(value.sample_id for value in data.normal_inputs),
            {
                source_type: tuple(
                    value.sample_id
                    for value in data.auxiliary_inputs
                    if value.source_defect_type == source_type
                )
                for source_type in context.experiment.defect_types
            },
            context.experiment.defect_types,
            context.experiment.seeds,
        )
        output_authority = _acquire_output(context, output_parts)
        preflight = P3A0Preflight(
            context=context,
            output=output_authority.path,
            git_head=head,
            input_hashes=dict(sorted(input_hashes.items())),
            model_authority=model_authority.snapshot,
            data=data,
            schedule=schedule,
        )
        return preflight, output_authority, model_authority
    except Exception:
        model_authority.close()
        raise


def validate_p3_a0_preflight(context: P3A0Context) -> P3A0Preflight:
    preflight, output_authority, model_authority = _validate_p3_a0_preflight_retained(context)
    try:
        return preflight
    finally:
        output_authority.close()
        model_authority.close()


def _parameter_versions(components: object) -> dict[str, tuple[int, ...]]:
    versions: dict[str, tuple[int, ...]] = {}
    for name in ("unet", "vae", "text_encoder"):
        module = getattr(components, name, None)
        if isinstance(module, torch.nn.Module):
            versions[name] = tuple(parameter._version for parameter in module.parameters())
    return versions


def _respond_default(
    preflight: P3A0Preflight,
    device: torch.device,
    model_authority: _ModelAuthority,
) -> ResponseBatch:
    context = preflight.context
    components = load_frozen_components(model_authority.root, device, torch.float16)
    versions = _parameter_versions(components)
    responses = run_paired_prompt_responses(
        components=components,
        inputs=(*preflight.data.normal_inputs, *preflight.data.auxiliary_inputs),
        jobs=(*preflight.schedule.normal, *preflight.schedule.auxiliary),
        normal_prompt=context.experiment.normal_prompt,
        defect_prompts=context.experiment.defect_prompts,
        timesteps=context.experiment.timesteps,
        seeds=context.experiment.seeds,
        device=device,
    )
    if _parameter_versions(components) != versions:
        raise RuntimeError("base-model parameter versions changed during response execution")
    return responses


def _clear_directory(authority: _OutputAuthority, descriptor: int) -> None:
    for name in tuple(os.listdir(descriptor)):
        authority.verify()
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        authority.garbage.isolate(descriptor, name, before)
        authority.garbage.purge()
        authority.verify()


def _cleanup_failed(preflight: P3A0Preflight, authority: _OutputAuthority, reason: str) -> None:
    del preflight
    authority.verify()
    _clear_directory(authority, authority.fd)
    authority.verify()
    publish_json_at(
        authority,
        "report.json",
        {"schema_version": _REPORT_SCHEMA, "status": "failed", "reason": reason},
    )


def _peak_vram() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def run_p3_a0(context: P3A0Context, *, backend: P3A0Backend | None = None) -> int:
    try:
        preflight, output_authority, model_authority = _validate_p3_a0_preflight_retained(context)
    except Exception as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True), flush=True)
        return 1
    try:
        device = torch.device("cuda:0")
        if backend is not None and not isinstance(backend, P3A0Backend):
            raise ValueError("backend must be a P3A0Backend")
        responses = (
            backend.respond(preflight, device)
            if backend is not None
            else _respond_default(preflight, device, model_authority)
        )
        if (
            not isinstance(responses, ResponseBatch)
            or len(responses.maps) != 448
            or responses.unet_forward_count != 1680
            or responses.parameters_unchanged is not True
        ):
            raise ValueError("response batch violates the frozen 448-record/1,680-forward contract")
        model_after = _snapshot_model(context.paths.model_root)
        if model_after != preflight.model_authority:
            raise RuntimeError("base model snapshot changed during P3-A0 execution")
        end_head, end_dirty = _git_state(context.cwd)
        if end_head != preflight.git_head or end_dirty:
            raise RuntimeError("Git state changed during P3-A0 execution")
        evaluation = evaluate_p3_a0(
            data=preflight.data, responses=responses, config=context.experiment
        )
        peak_vram = _peak_vram()
        next_step = (
            "proceed_p3_a1_raw_response_baseline"
            if evaluation.promoted
            else "stop_autonomous_localization_claim"
        )
        report = {
            "schema_version": _REPORT_SCHEMA,
            "status": "complete",
            "reason": None,
            "promoted": evaluation.promoted,
            "next_step": next_step,
            "evaluation_mode": "diagnostic_only",
            "training_performed": False,
            "checkpoint_written": False,
            "renderer_checkpoint_loaded": False,
            "unet_forward_count": responses.unet_forward_count,
            "peak_vram_bytes": peak_vram,
            "artifact_count": 8,
        }
        provenance = collect_provenance(context.paths, cwd=context.cwd)
        provenance.update(
            {
                "schema_version": "p3_a0_hazelnut_spatial_response_provenance_v1",
                "git_head_start": preflight.git_head,
                "git_head_end": end_head,
                "input_hashes": preflight.input_hashes,
                "model_snapshot_before": preflight.model_authority,
                "model_snapshot_after": model_after,
                "training_performed": False,
                "checkpoint_written": False,
                "renderer_checkpoint_loaded": False,
                "unet_forward_count": responses.unet_forward_count,
                "peak_vram_bytes": peak_vram,
            }
        )
        write_complete_artifacts(
            authority=output_authority,
            data=preflight.data,
            responses=responses,
            evaluation=evaluation,
            defect_types=context.experiment.defect_types,
            seeds=context.experiment.seeds,
            report=report,
            provenance=provenance,
        )
        return 0
    except Exception as error:
        try:
            _cleanup_failed(preflight, output_authority, str(error))
        except Exception as cleanup_error:
            print(
                json.dumps(
                    {"status": "failed", "reason": str(error), "cleanup_error": str(cleanup_error)},
                    sort_keys=True,
                ),
                flush=True,
            )
            return 1
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True), flush=True)
        return 1
    finally:
        output_authority.close()
        model_authority.close()


__all__ = [
    "P3A0Backend",
    "P3A0Context",
    "P3A0Preflight",
    "run_p3_a0",
    "validate_p3_a0_preflight",
]

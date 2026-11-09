from __future__ import annotations

import io
import json
import os
import secrets
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from cadiff.p3_a0.dataset import P3A0Data, ResponseInput
from cadiff.p3_a0.evaluation import EvaluationResult
from cadiff.p3_a0.response import ResponseBatch, ResponseMap

ARTIFACT_NAMES = (
    "records.jsonl",
    "metrics.json",
    "report.json",
    "provenance.json",
    "contact_sheet_crack.png",
    "contact_sheet_cut.png",
    "contact_sheet_hole.png",
    "contact_sheet_print.png",
)


@dataclass(frozen=True)
class _PublishedArtifact:
    name: str
    device: int
    inode: int


class OutputAuthority(Protocol):
    fd: int

    def verify(self) -> None: ...

    def retract(self, name: str, device: int, inode: int) -> None: ...


def _publish_bytes_at(authority: OutputAuthority, name: str, payload: bytes) -> _PublishedArtifact:
    if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
        raise ValueError("artifact name must be a direct output child")
    authority.verify()
    try:
        os.stat(name, dir_fd=authority.fd, follow_symlinks=False)
    except FileNotFoundError:
        artifact_exists = False
    else:
        artifact_exists = True
    if artifact_exists:
        raise FileExistsError(f"artifact already exists: {name}")
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    link_attempted = False
    published_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=authority.fd)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        temporary_stat = os.fstat(descriptor)
        published_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        os.close(descriptor)
        descriptor = None
        authority.verify()
        link_attempted = True
        os.link(
            temporary,
            name,
            src_dir_fd=authority.fd,
            dst_dir_fd=authority.fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=authority.fd)
        os.fsync(authority.fd)
        authority.verify()
    except Exception as error:
        if link_attempted and published_identity is not None:
            try:
                authority.retract(name, *published_identity)
            except Exception as retract_error:
                raise RuntimeError(
                    f"publication retraction blocked for {name!r}: {retract_error}"
                ) from error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=authority.fd)
    if published_identity is None:
        raise RuntimeError("published artifact identity was not captured")
    return _PublishedArtifact(name, *published_identity)


def _write_jsonl_atomic(
    authority: OutputAuthority, name: str, records: tuple[dict[str, Any], ...]
) -> _PublishedArtifact:
    payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for value in records
    ).encode("utf-8")
    return _publish_bytes_at(authority, name, payload)


def publish_json_at(
    authority: OutputAuthority, name: str, payload: dict[str, Any]
) -> _PublishedArtifact:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _publish_bytes_at(authority, name, encoded)


def aggregate_records(
    responses: ResponseBatch, evaluation: EvaluationResult
) -> tuple[dict[str, Any], ...]:
    if len(responses.maps) != 448 or len(evaluation.records) != 448:
        raise ValueError("artifact accounting requires exactly 448 records")
    maps: dict[str, ResponseMap] = {}
    for value in responses.maps:
        record_id = value.job.record_id
        if record_id in maps:
            raise ValueError("response record IDs must be unique")
        if not isinstance(value.timestep_stats, tuple) or len(value.timestep_stats) != 3:
            raise ValueError("each response record must carry exactly three timestep stats")
        maps[record_id] = value
    records: list[dict[str, Any]] = []
    for summary in evaluation.records:
        if not isinstance(summary, dict) or not isinstance(summary.get("record_id"), str):
            raise ValueError("evaluation record schema is invalid")
        response = maps.get(summary["record_id"])
        if response is None:
            raise ValueError("evaluation record does not bind to a response")
        record = {
            **summary,
            "image_latent_sha256": response.image_latent_sha256,
            "noise_sha256": response.noise_sha256,
            "timestep_stats": [asdict(value) for value in response.timestep_stats],
        }
        if any(key in record for key in ("response", "pixels", "image_relpath", "mask_relpath")):
            raise ValueError("aggregated records cannot embed maps or source paths")
        records.append(record)
    if {value["record_id"] for value in records} != set(maps):
        raise ValueError("evaluation records must cover the complete response batch")
    return tuple(
        sorted(
            records,
            key=lambda value: (
                value["cohort"],
                value["sample_id"],
                value["prompt_defect_type"],
                value["seed"],
                value["record_id"],
            ),
        )
    )


def _rgb(item: ResponseInput) -> np.ndarray:
    pixels = item.pixels.detach().cpu().numpy()
    if pixels.ndim != 3 or pixels.shape[0] != 3 or not np.isfinite(pixels).all():
        raise ValueError("contact-sheet source must be a finite CHW RGB tensor")
    return np.moveaxis(np.clip((pixels + 1.0) * 127.5, 0.0, 255.0), 0, 2).astype(np.uint8)


def _overlay(item: ResponseInput, response: np.ndarray) -> Image.Image:
    base = Image.fromarray(_rgb(item)).resize((512, 512), Image.Resampling.BILINEAR)
    score = np.asarray(response, dtype=np.float64)
    if score.ndim != 2 or not np.isfinite(score).all() or np.any(score < 0.0):
        raise ValueError("contact-sheet response must be a finite non-negative 2D map")
    heat = Image.fromarray(np.clip(score * 255.0, 0.0, 255.0).astype(np.uint8)).resize(
        (512, 512), Image.Resampling.BILINEAR
    )
    alpha = np.asarray(heat, dtype=np.float64)[:, :, None] / 255.0 * 0.55
    base_array = np.asarray(base, dtype=np.float64)
    red = np.empty_like(base_array)
    red[:, :, 0] = 255.0
    red[:, :, 1] = np.asarray(heat, dtype=np.float64)
    red[:, :, 2] = 0.0
    composited = np.clip(base_array * (1.0 - alpha) + red * alpha, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(composited)


def _write_contact_sheets(
    authority: OutputAuthority,
    data: P3A0Data,
    responses: ResponseBatch,
    types: tuple[str, ...],
    seeds: tuple[int, ...],
    published: list[_PublishedArtifact],
) -> None:
    if len(data.normal_inputs) != 8 or len(seeds) != 2:
        raise ValueError("contact sheets require eight normal references and two seeds")
    maps = {
        (value.job.cohort, value.job.sample_id, value.job.prompt_defect_type, value.job.seed): value
        for value in responses.maps
    }
    normal_inputs = tuple(sorted(data.normal_inputs, key=lambda value: value.sample_id))
    required = {
        ("normal_reference", item.sample_id, defect_type, seed)
        for defect_type in types
        for item in normal_inputs
        for seed in seeds
    }
    if not required.issubset(maps):
        raise ValueError("contact-sheet response matrix is incomplete")
    for defect_type in types:
        sheet = Image.new("RGB", (1024, 4096))
        for row, item in enumerate(normal_inputs):
            for column, seed in enumerate(seeds):
                key = ("normal_reference", item.sample_id, defect_type, seed)
                if key not in maps:
                    raise ValueError("contact-sheet response matrix is incomplete")
                sheet.paste(_overlay(item, maps[key].response), (column * 512, row * 512))
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG", compress_level=9, optimize=False)
        published.append(
            _publish_bytes_at(
                authority,
                f"contact_sheet_{defect_type}.png",
                buffer.getvalue(),
            )
        )


def _rollback_bundle(
    authority: OutputAuthority,
    published: list[_PublishedArtifact],
    cause: Exception,
) -> None:
    blockers: list[str] = []
    for value in reversed(published):
        try:
            authority.retract(value.name, value.device, value.inode)
        except Exception as error:
            blockers.append(f"{value.name}: {error}")
    if blockers:
        raise RuntimeError(
            f"{cause}; artifact bundle rollback blocked: {'; '.join(blockers)}"
        ) from cause


def write_complete_artifacts(
    *,
    authority: OutputAuthority,
    data: P3A0Data,
    responses: ResponseBatch,
    evaluation: EvaluationResult,
    defect_types: tuple[str, ...],
    seeds: tuple[int, ...],
    report: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    if tuple(defect_types) != ("crack", "cut", "hole", "print"):
        raise ValueError("artifact defect types must match the frozen order")
    records = aggregate_records(responses, evaluation)
    metrics = {
        "schema_version": "p3_a0_hazelnut_spatial_response_metrics_v1",
        "gate_groups": evaluation.gate_groups,
        **evaluation.metrics,
    }
    published: list[_PublishedArtifact] = []
    try:
        authority.verify()
        if os.listdir(authority.fd):
            raise RuntimeError("formal output must be empty before artifact publication")
        _write_contact_sheets(authority, data, responses, defect_types, seeds, published)
        published.append(_write_jsonl_atomic(authority, "records.jsonl", records))
        published.append(publish_json_at(authority, "metrics.json", metrics))
        published.append(publish_json_at(authority, "provenance.json", provenance))
        published.append(publish_json_at(authority, "report.json", report))
        actual = tuple(sorted(os.listdir(authority.fd)))
        if actual != tuple(sorted(ARTIFACT_NAMES)):
            raise RuntimeError("formal output differs from the minimal eight-artifact contract")
        authority.verify()
    except Exception as error:
        _rollback_bundle(authority, published, error)
        raise


__all__ = [
    "ARTIFACT_NAMES",
    "OutputAuthority",
    "aggregate_records",
    "publish_json_at",
    "write_complete_artifacts",
]

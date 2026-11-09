"""Explicit training and single-rollout inference facades for Module A RC1."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .checkpoint import (
    A1CheckpointMetadata,
    a1_experiment_config_sha256,
    load_a1_checkpoint,
    save_a1_checkpoint,
)
from .config import V06AExperimentConfig
from .data import LoadedIdentity
from .dataset import PlainTargetDataset, PlainTargetRequest, _normalise_rgb
from .determinism import (
    A1DeterminismPreflight,
    A1DeterminismProfile,
    cpu_determinism_preflight,
    cpu_determinism_profile,
    cuda_determinism_scope,
    formal_cuda_bootstrap_scope,
    run_cuda_determinism_preflight,
)
from .diffusion import MaskDiffusionSchedule
from .identities import IdentityManifest
from .inference import A1InferenceCondition, A1ProposalSampler
from .model import A1MaskUNet, require_supported_variant
from .structure import estimate_normal_structure
from .targets import downsample_all_4x4, nearest_upsample_4x4
from .training import compute_a1_diffusion_loss, sample_a1_step_randomness


def _validate_authority(config: V06AExperimentConfig, identity_manifest: IdentityManifest) -> None:
    if not isinstance(config, V06AExperimentConfig):
        raise TypeError("config must be a V06AExperimentConfig")
    if not isinstance(identity_manifest, IdentityManifest):
        raise TypeError("identity_manifest must be an IdentityManifest")
    if not identity_manifest.is_authoritative():
        raise ValueError("runner requires a loader-authoritative identity manifest")
    if (
        config.category != identity_manifest.category
        or config.defect_types != identity_manifest.defect_types
    ):
        raise ValueError("config and identity manifest category/vocabulary must agree")
    expected = {
        "schema_version": "v06_a_experiment_rc1",
        "identity_manifest": f"configs/manifests/v06_a_{config.category}_rc1.yaml",
        "image_size": 512,
        "proposal_size": 128,
        "square_dilate_radius": 24,
        "a_variants": ("a0_v", "a1_gv", "a1_sv_plain"),
        "panel_seeds": (2026082801, 2026082802),
        "p0_pairs_per_category": 8,
        "p1_normal_count": 4,
        "support_border_width": 16,
        "support_background_margin": 0.03,
        "atlas_reference_count": 16,
        "atlas_consensus": 0.5,
        "teacher_outer_radius": 72,
        "teacher_blind_gaps": (32, 48),
        "teacher_candidate_stride": 16,
        "teacher_temperature": 0.25,
        "teacher_relation_weight": 2.0,
        "teacher_min_context_fraction": 0.75,
    }
    mismatches = {
        name: (getattr(config, name), value)
        for name, value in expected.items()
        if getattr(config, name) != value
    }
    if mismatches:
        raise ValueError(f"runner config does not equal the frozen RC1 contract: {mismatches}")
    integer_tunables = (
        "base_channels",
        "type_embedding_dim",
        "timestep_embedding_dim",
        "normal_feature_channels",
        "diffusion_steps",
        "sampling_steps",
        "training_steps",
        "batch_size",
    )
    invalid_integers = {
        name: getattr(config, name)
        for name in integer_tunables
        if isinstance(getattr(config, name), bool)
        or not isinstance(getattr(config, name), int)
        or getattr(config, name) <= 0
    }
    if invalid_integers:
        raise ValueError(f"runner tunable integer must be positive: {invalid_integers}")
    if config.sampling_steps > config.diffusion_steps:
        raise ValueError("runner sampling_steps must not exceed diffusion_steps")
    if (
        isinstance(config.training_seed, bool)
        or not isinstance(config.training_seed, int)
        or not 0 <= config.training_seed < 2**63
    ):
        raise ValueError("runner training_seed must be a non-negative signed 64-bit integer")
    for name in ("learning_rate", "gradient_clip"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"runner {name} must be a positive finite number")
    output_relpath = config.output_relpath
    relative = PurePosixPath(output_relpath) if isinstance(output_relpath, str) else None
    if (
        relative is None
        or not output_relpath
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != output_relpath
    ):
        raise ValueError("runner output_relpath must remain normalized even though it is not used")


class DeterministicTrainingSchedule:
    """Local MT19937 protocol: round-robin type, fit-index draw, 63-bit source seed."""

    def __init__(
        self,
        *,
        config: V06AExperimentConfig,
        identity_manifest: IdentityManifest,
    ) -> None:
        _validate_authority(config, identity_manifest)
        if len(identity_manifest.defect_types) != 4:
            raise ValueError("RC1 schedule requires exactly four defect types")
        if not identity_manifest.normal_fit or any(
            record.role != "normal_fit" for record in identity_manifest.normal_fit
        ):
            raise ValueError("RC1 schedule may draw targets only from normal_fit")
        self.config = config
        self.identity_manifest = identity_manifest
        self._rng = random.Random(config.training_seed)
        self._issued_batches = 0

    @property
    def issued_batches(self) -> int:
        return self._issued_batches

    def next_batch(self) -> tuple[PlainTargetRequest, ...]:
        start = self._issued_batches * self.config.batch_size
        requests: list[PlainTargetRequest] = []
        for offset in range(self.config.batch_size):
            defect_type = self.identity_manifest.defect_types[
                (start + offset) % len(self.identity_manifest.defect_types)
            ]
            target_index = self._rng.randrange(len(self.identity_manifest.normal_fit))
            transfer_seed = self._rng.getrandbits(63)
            requests.append(
                PlainTargetRequest(
                    normal_record=self.identity_manifest.normal_fit[target_index],
                    defect_type=defect_type,
                    transfer_seed=transfer_seed,
                )
            )
        self._issued_batches += 1
        return tuple(requests)


def _locally_seeded_model(
    *,
    config: V06AExperimentConfig,
    variant: str,
    device: torch.device,
) -> A1MaskUNet:
    """Construct deterministically while restoring caller CPU/target-CUDA RNG state."""

    cuda_devices: list[int] = []
    cuda_index: int | None = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested CUDA device is unavailable")
        cuda_index = torch.cuda.current_device() if device.index is None else device.index
        cuda_devices.append(cuda_index)
    with torch.random.fork_rng(devices=cuda_devices, device_type="cuda"):
        torch.random.default_generator.manual_seed(config.training_seed)
        if cuda_index is not None:
            with torch.cuda.device(cuda_index):
                torch.cuda.manual_seed(config.training_seed)
        model = A1MaskUNet.from_experiment(config, variant=variant).to(device)
    return model


@dataclass(frozen=True)
class A1TrainStepResult:
    step: int
    loss: float
    gradient_norm: float
    requests: tuple[PlainTargetRequest, ...]


class A1Trainer:
    """One category/variant trainer; construction performs no training by itself."""

    def __init__(
        self,
        *,
        config: V06AExperimentConfig,
        identity_manifest: IdentityManifest,
        dataset: PlainTargetDataset,
        variant: str,
        device: str | torch.device = "cpu",
    ) -> None:
        variant = require_supported_variant(variant)
        _validate_authority(config, identity_manifest)
        if not isinstance(dataset, PlainTargetDataset):
            raise TypeError("dataset must be a PlainTargetDataset")
        if (
            dataset.config != config
            or dataset.identity_manifest.authority_digest != identity_manifest.authority_digest
        ):
            raise ValueError("trainer dataset must use the supplied authoritative config/manifest")
        torch_device = torch.device(device)
        self.config = config
        self.identity_manifest = identity_manifest
        self.dataset = dataset
        self.variant = variant
        self.device = torch_device
        self.model = _locally_seeded_model(config=config, variant=variant, device=torch_device)
        self.diffusion_schedule = MaskDiffusionSchedule(diffusion_steps=config.diffusion_steps)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.data_schedule = DeterministicTrainingSchedule(
            config=config, identity_manifest=identity_manifest
        )
        self._noise_generator = torch.Generator(device=torch_device)
        self._noise_generator.manual_seed(config.training_seed)
        self.completed_steps = 0

    def train_step(self) -> A1TrainStepResult:
        """Run one scheduled optimizer step; an invalid ordinary target aborts directly."""

        requests = self.data_schedule.next_batch()
        batch = self.dataset.build_batch(requests, variant=self.variant, device=self.device)
        start = self.completed_steps * self.config.batch_size
        expected_types = torch.tensor(
            [
                (start + offset) % len(self.identity_manifest.defect_types)
                for offset in range(self.config.batch_size)
            ],
            dtype=torch.long,
            device=self.device,
        )
        if batch.z0.shape[0] != self.config.batch_size or not torch.equal(
            batch.type_ids, expected_types
        ):
            raise ValueError("training batch must preserve the deterministic round-robin sequence")
        randomness = sample_a1_step_randomness(
            batch=batch,
            schedule=self.diffusion_schedule,
            generator=self._noise_generator,
        )
        self.optimizer.zero_grad(set_to_none=True)
        result = compute_a1_diffusion_loss(
            model=self.model,
            schedule=self.diffusion_schedule,
            batch=batch,
            randomness=randomness,
        )
        if not torch.isfinite(result.loss).item():
            raise ValueError("training produced a non-finite loss")
        result.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)).item():
            raise ValueError("training produced a non-finite gradient norm")
        self.optimizer.step()
        step = self.completed_steps
        self.completed_steps += 1
        return A1TrainStepResult(
            step=step,
            loss=float(result.loss.detach().cpu()),
            gradient_norm=float(torch.as_tensor(gradient_norm).detach().cpu()),
            requests=requests,
        )


@dataclass(frozen=True)
class A1TrainingRunResult:
    checkpoint: Path
    provenance_path: Path
    steps: int
    final_loss: float
    metadata: A1CheckpointMetadata
    provenance: A1TrainingProvenance


@dataclass(frozen=True)
class A1TrainingProvenance:
    schema_version: str
    category: str
    variant: str
    run_id: str
    identity_manifest_authority_sha256: str
    training_protocol: str
    training_seed: int
    completed_steps: int
    batch_size: int
    learning_rate: float
    gradient_clip: float
    target_kind: str
    diffusion_steps: int
    sampling_steps: int
    experiment_config_sha256: str
    model_construction_signature: str
    checkpoint_relpath: str
    checkpoint_metadata_sha256: str
    checkpoint_weights_sha256: str
    determinism_profile: A1DeterminismProfile
    determinism_preflight: A1DeterminismPreflight
    final_loss: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_training_provenance(
    *,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
    variant: str,
    run_id: str,
    checkpoint: Path,
    metadata: A1CheckpointMetadata,
    completed_steps: int,
    final_loss: float,
    determinism_profile: A1DeterminismProfile,
    determinism_preflight: A1DeterminismPreflight,
) -> A1TrainingProvenance:
    if completed_steps != config.training_steps or not np.isfinite(final_loss):
        raise ValueError("training provenance requires all configured finite-loss steps")
    metadata_path = checkpoint / "metadata.json"
    weights_path = checkpoint / "model.safetensors"
    metadata_sha256 = _sha256_file(metadata_path)
    weights_sha256 = _sha256_file(weights_path)
    if weights_sha256 != metadata.weights_sha256:
        raise ValueError("saved checkpoint weights do not match returned checkpoint metadata")
    return A1TrainingProvenance(
        schema_version="v06_a_training_run_rc2",
        category=config.category,
        variant=variant,
        run_id=run_id,
        identity_manifest_authority_sha256=identity_manifest.authority_digest,
        training_protocol=(
            "round_robin_type__local_mt19937_fit_index_getrandbits63__"
            "local_mt19937_source_index__plain_center_no_retry"
        ),
        training_seed=config.training_seed,
        completed_steps=completed_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gradient_clip=config.gradient_clip,
        target_kind="plain_geometry_transfer",
        diffusion_steps=config.diffusion_steps,
        sampling_steps=config.sampling_steps,
        experiment_config_sha256=metadata.experiment_config_sha256,
        model_construction_signature=metadata.model_construction_signature,
        checkpoint_relpath="checkpoint",
        checkpoint_metadata_sha256=metadata_sha256,
        checkpoint_weights_sha256=weights_sha256,
        determinism_profile=determinism_profile,
        determinism_preflight=determinism_preflight,
        final_loss=final_loss,
    )


def _execute_training_attempt(
    *,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
    dataset: PlainTargetDataset,
    variant: str,
    run_id: str,
    artifact_directory: Path,
    device: torch.device,
    determinism_profile: A1DeterminismProfile,
    determinism_preflight: A1DeterminismPreflight,
) -> A1TrainingRunResult:
    trainer = A1Trainer(
        config=config,
        identity_manifest=identity_manifest,
        dataset=dataset,
        variant=variant,
        device=device,
    )
    last: A1TrainStepResult | None = None
    for _ in range(config.training_steps):
        last = trainer.train_step()
    if last is None or trainer.completed_steps != config.training_steps:
        raise RuntimeError("formal RC1 training did not complete all configured steps")
    artifact_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{variant}.run-", dir=artifact_directory.parent))
    published = False
    try:
        temporary_checkpoint = temporary / "checkpoint"
        metadata = save_a1_checkpoint(
            checkpoint=temporary_checkpoint,
            model=trainer.model,
            config=config,
        )
        provenance = _build_training_provenance(
            config=config,
            identity_manifest=identity_manifest,
            variant=variant,
            run_id=run_id,
            checkpoint=temporary_checkpoint,
            metadata=metadata,
            completed_steps=trainer.completed_steps,
            final_loss=last.loss,
            determinism_profile=determinism_profile,
            determinism_preflight=determinism_preflight,
        )
        (temporary / "training_run.json").write_text(
            json.dumps(asdict(provenance), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if artifact_directory.exists() or artifact_directory.is_symlink():
            raise FileExistsError(
                f"formal RC1 artifact appeared during publication: {artifact_directory}"
            )
        temporary.rename(artifact_directory)
        published = True
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    checkpoint = artifact_directory / "checkpoint"
    provenance_path = artifact_directory / "training_run.json"
    return A1TrainingRunResult(
        checkpoint=checkpoint,
        provenance_path=provenance_path,
        steps=trainer.completed_steps,
        final_loss=last.loss,
        metadata=metadata,
        provenance=provenance,
    )


_ATTEMPT_CONTRACT_KEYS = {
    "schema_version",
    "category",
    "run_id",
    "identity_manifest_authority_sha256",
    "experiment_config_sha256",
}


def _attempt_contract_payload(
    *,
    run_directory: Path,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
) -> dict[str, object]:
    return {
        "schema_version": "v06_a_attempt_contract_rc2",
        "category": config.category,
        "run_id": run_directory.name,
        "identity_manifest_authority_sha256": identity_manifest.authority_digest,
        "experiment_config_sha256": a1_experiment_config_sha256(config),
    }


def _read_and_validate_attempt_contract(
    *,
    run_directory: Path,
    expected: dict[str, object],
) -> Path:
    contract = run_directory / "attempt_contract.json"
    if contract.is_symlink() or not contract.is_file():
        raise ValueError("same-run GV/SV variants require real adjacent attempt_contract.json")
    try:
        raw = json.loads(
            contract.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except ValueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid attempt_contract.json: {error}") from error
    if not isinstance(raw, dict) or set(raw) != _ATTEMPT_CONTRACT_KEYS:
        raise ValueError("same-run GV/SV attempt_contract.json schema mismatch")
    canonical = lambda value: json.dumps(  # noqa: E731 - local canonical comparator
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical(raw) != canonical(expected):
        raise ValueError("same-run GV/SV variants must use the same training budget")
    return contract


def _claim_gv_sv_attempt_contract(
    *,
    run_directory: Path,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
) -> Path:
    """Atomically establish one shared scientific budget for concurrent GV/SV siblings."""

    run_directory.parent.mkdir(parents=True, exist_ok=True)
    if run_directory.parent.is_symlink() or not run_directory.parent.is_dir():
        raise ValueError("formal category directory must be a real directory")
    run_directory.mkdir(exist_ok=True)
    if run_directory.is_symlink() or not run_directory.is_dir():
        raise ValueError("formal run_id directory must be a real directory")
    expected = _attempt_contract_payload(
        run_directory=run_directory,
        config=config,
        identity_manifest=identity_manifest,
    )
    payload = (json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".attempt_contract-",
        dir=run_directory,
    )
    temporary = Path(temporary_name)
    contract = run_directory / "attempt_contract.json"
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, contract)
        except FileExistsError:
            return _read_and_validate_attempt_contract(
                run_directory=run_directory,
                expected=expected,
            )
        return contract
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _variant_attempt_lock(
    *,
    run_directory: Path,
    variant: str,
) -> Iterator[Path]:
    """Hold one process-level run/variant mutex without consuming its identity."""

    variant = require_supported_variant(variant)
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    if run_directory.parent.is_symlink() or not run_directory.parent.is_dir():
        raise ValueError("formal category directory must be a real directory")
    run_directory.mkdir(exist_ok=True)
    if run_directory.is_symlink() or not run_directory.is_dir():
        raise ValueError("formal run_id directory must be a real directory")
    lock_path = run_directory / f".{variant}.attempt.lock"
    file_descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BlockingIOError(
                f"formal run_id/variant attempt is already active: {run_directory / variant}"
            ) from error
        try:
            yield lock_path
        finally:
            fcntl.flock(file_descriptor, fcntl.LOCK_UN)
    finally:
        os.close(file_descriptor)


def run_training(
    *,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
    dataset: PlainTargetDataset,
    variant: str,
    runs_root: Path,
    run_id: str,
    device: str | torch.device = "cpu",
) -> A1TrainingRunResult:
    """Run one configured attempt; a new contained ``run_id`` is required for another attempt."""

    variant = require_supported_variant(variant)
    _validate_authority(config, identity_manifest)
    if not isinstance(runs_root, Path):
        raise TypeError("runs_root must be a pathlib.Path")
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs_root does not exist: {runs_root}")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None
    ):
        raise ValueError("run_id must be one safe, normalized path component")
    root = runs_root.resolve(strict=True)
    artifact_directory = root / "v06_a" / config.category / run_id / variant
    try:
        artifact_directory.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise ValueError("run_id must resolve within runs_root") from error
    if artifact_directory.exists() or artifact_directory.is_symlink():
        raise FileExistsError(
            f"formal RC1 category/variant artifact already exists: {artifact_directory}"
        )

    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"}:
        raise ValueError("formal Module-A training device must be CPU or CUDA")

    def execute_attempt(
        *,
        profile: A1DeterminismProfile,
        preflight: A1DeterminismPreflight,
    ) -> A1TrainingRunResult:
        _claim_gv_sv_attempt_contract(
            run_directory=artifact_directory.parent,
            config=config,
            identity_manifest=identity_manifest,
        )
        _validate_gv_sv_sibling_budget(
            artifact_directory=artifact_directory,
            config=config,
            identity_manifest=identity_manifest,
            run_id=run_id,
        )
        return _execute_training_attempt(
            config=config,
            identity_manifest=identity_manifest,
            dataset=dataset,
            variant=variant,
            run_id=run_id,
            artifact_directory=artifact_directory,
            device=torch_device,
            determinism_profile=profile,
            determinism_preflight=preflight,
        )

    with _variant_attempt_lock(
        run_directory=artifact_directory.parent,
        variant=variant,
    ):
        if artifact_directory.exists() or artifact_directory.is_symlink():
            raise FileExistsError(
                f"formal RC1 category/variant artifact already exists: {artifact_directory}"
            )
        if torch_device.type == "cuda":
            with formal_cuda_bootstrap_scope(torch_device) as profile:
                preflight = run_cuda_determinism_preflight(
                    config=config,
                    variant=variant,
                    device=torch_device,
                    profile=profile,
                )
                return execute_attempt(profile=profile, preflight=preflight)
        profile = cpu_determinism_profile()
        preflight = cpu_determinism_preflight()
        return execute_attempt(profile=profile, preflight=preflight)


_TRAINING_PROTOCOL = (
    "round_robin_type__local_mt19937_fit_index_getrandbits63__"
    "local_mt19937_source_index__plain_center_no_retry"
)
_PROVENANCE_KEYS = {
    "schema_version",
    "category",
    "variant",
    "run_id",
    "identity_manifest_authority_sha256",
    "training_protocol",
    "training_seed",
    "completed_steps",
    "batch_size",
    "learning_rate",
    "gradient_clip",
    "target_kind",
    "diffusion_steps",
    "sampling_steps",
    "experiment_config_sha256",
    "model_construction_signature",
    "checkpoint_relpath",
    "checkpoint_metadata_sha256",
    "checkpoint_weights_sha256",
    "determinism_profile",
    "determinism_preflight",
    "final_loss",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class A1FormalRun:
    run_directory: Path
    checkpoint: Path
    metadata: A1CheckpointMetadata
    provenance: A1TrainingProvenance
    model: A1MaskUNet


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate formal training run key: {key}")
        result[key] = value
    return result


def _read_training_provenance(path: Path) -> A1TrainingProvenance:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    except ValueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training_run.json: {error}") from error
    if not isinstance(raw, dict) or set(raw) != _PROVENANCE_KEYS:
        raise ValueError("formal training run schema mismatch")

    profile_raw = raw.pop("determinism_profile")
    preflight_raw = raw.pop("determinism_preflight")
    profile_keys = set(asdict(A1DeterminismProfile()))
    if not isinstance(profile_raw, dict) or set(profile_raw) != profile_keys:
        raise ValueError("formal training run determinism_profile mismatch")
    expected_preflight_keys = {
        "device_type",
        "device",
        "device_name",
        "status",
        "error",
        "silent_fallback_used",
    }
    if not isinstance(preflight_raw, dict) or set(preflight_raw) != expected_preflight_keys:
        raise ValueError("formal training run determinism_preflight schema mismatch")
    if (
        preflight_raw["device_type"] not in {"cpu", "cuda"}
        or not isinstance(preflight_raw["device"], str)
        or (
            preflight_raw["device_name"] is not None
            and not isinstance(preflight_raw["device_name"], str)
        )
        or preflight_raw["error"] is not None
        or preflight_raw["silent_fallback_used"] is not False
    ):
        raise ValueError("formal training run determinism_preflight is invalid")
    expected_status = "not_applicable_cpu" if preflight_raw["device_type"] == "cpu" else "passed"
    if preflight_raw["status"] != expected_status:
        raise ValueError("formal training run determinism_preflight status mismatch")
    if preflight_raw["device_type"] == "cuda" and not preflight_raw["device_name"]:
        raise ValueError("formal CUDA training run must record its device name")

    boolean_profile_fields = {
        "local_seeded_model_init",
        "local_seeded_data_schedule",
        "local_seeded_diffusion_noise",
        "torch_deterministic_algorithms",
        "torch_deterministic_warn_only",
        "cross_machine_byte_identity_promised",
    }
    if any(not isinstance(profile_raw[name], bool) for name in boolean_profile_fields):
        raise ValueError("formal training run determinism_profile mismatch")
    if (
        not isinstance(profile_raw["device_type"], str)
        or not isinstance(profile_raw["cuda_policy_status"], str)
        or not isinstance(profile_raw["float32_matmul_precision"], str)
    ):
        raise ValueError("formal training run determinism_profile mismatch")
    try:
        profile = A1DeterminismProfile(**profile_raw)
    except TypeError as error:
        raise ValueError("formal training run determinism_profile mismatch") from error
    if preflight_raw["device_type"] == "cuda":
        if profile != A1DeterminismProfile():
            raise ValueError("formal training run determinism_profile mismatch")
    elif (
        profile.device_type != "cpu"
        or profile.cuda_policy_status != "not_applicable"
        or profile.cudnn_deterministic is not None
        or profile.cudnn_benchmark is not None
        or profile.cuda_matmul_allow_tf32 is not None
        or profile.cudnn_allow_tf32 is not None
        or profile.cublas_workspace_config is not None
        or profile.cross_machine_byte_identity_promised
    ):
        raise ValueError("formal training run determinism_profile mismatch")

    string_fields = {
        "schema_version",
        "category",
        "variant",
        "run_id",
        "identity_manifest_authority_sha256",
        "training_protocol",
        "target_kind",
        "experiment_config_sha256",
        "model_construction_signature",
        "checkpoint_relpath",
        "checkpoint_metadata_sha256",
        "checkpoint_weights_sha256",
    }
    integer_fields = {
        "training_seed",
        "completed_steps",
        "batch_size",
        "diffusion_steps",
        "sampling_steps",
    }
    numeric_fields = {"learning_rate", "gradient_clip", "final_loss"}
    if any(not isinstance(raw[name], str) for name in string_fields):
        raise ValueError("formal training run string field has invalid type")
    if any(
        isinstance(raw[name], bool) or not isinstance(raw[name], int) for name in integer_fields
    ):
        raise ValueError("formal training run integer field has invalid type")
    if any(
        isinstance(raw[name], bool)
        or not isinstance(raw[name], int | float)
        or not math.isfinite(float(raw[name]))
        for name in numeric_fields
    ):
        raise ValueError("formal training run numeric field must be finite")
    for name in (
        "identity_manifest_authority_sha256",
        "experiment_config_sha256",
        "model_construction_signature",
        "checkpoint_metadata_sha256",
        "checkpoint_weights_sha256",
    ):
        if _SHA256_PATTERN.fullmatch(raw[name]) is None:
            raise ValueError(f"formal training run {name} is not SHA-256")

    raw["learning_rate"] = float(raw["learning_rate"])
    raw["gradient_clip"] = float(raw["gradient_clip"])
    raw["final_loss"] = float(raw["final_loss"])
    raw["determinism_profile"] = profile
    raw["determinism_preflight"] = A1DeterminismPreflight(**preflight_raw)
    try:
        return A1TrainingProvenance(**raw)
    except TypeError as error:
        raise ValueError(f"invalid formal training run: {error}") from error


def _validate_gv_sv_sibling_budget(
    *,
    artifact_directory: Path,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
    run_id: str,
) -> None:
    """Require GV/SV siblings under one run_id to share one full training budget."""

    for sibling_variant in ("a1_gv", "a1_sv_plain"):
        sibling_directory = artifact_directory.parent / sibling_variant
        if sibling_directory == artifact_directory or not sibling_directory.exists():
            continue
        if sibling_directory.is_symlink() or not sibling_directory.is_dir():
            raise ValueError("same-run GV/SV sibling must be a formal run directory")
        sibling = _read_training_provenance(sibling_directory / "training_run.json")
        expected = {
            "category": config.category,
            "variant": sibling_variant,
            "run_id": run_id,
            "identity_manifest_authority_sha256": identity_manifest.authority_digest,
            "training_seed": config.training_seed,
            "completed_steps": config.training_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "gradient_clip": config.gradient_clip,
            "diffusion_steps": config.diffusion_steps,
            "experiment_config_sha256": a1_experiment_config_sha256(config),
        }
        mismatches = {
            name: (getattr(sibling, name), value)
            for name, value in expected.items()
            if getattr(sibling, name) != value
        }
        if mismatches:
            raise ValueError(
                f"same-run GV/SV variants must use the same training budget: {mismatches}"
            )


def load_formal_a1_run(
    *,
    run_directory: Path,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
    expected_variant: str,
    device: str | torch.device = "cpu",
) -> A1FormalRun:
    """Load only a completed run whose sidecar and checkpoint cross-validate exactly."""

    expected_variant = require_supported_variant(expected_variant)
    _validate_authority(config, identity_manifest)
    if not isinstance(run_directory, Path):
        raise TypeError("run_directory must be a pathlib.Path")
    try:
        resolved_run = run_directory.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"formal training run directory is missing: {run_directory}") from error
    if not resolved_run.is_dir() or run_directory.is_symlink():
        raise ValueError("formal training run must be a real directory, not a symlink")
    sidecar_path = resolved_run / "training_run.json"
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise ValueError("formal training run requires adjacent training_run.json")
    provenance = _read_training_provenance(sidecar_path)

    expected_run_id = resolved_run.parent.name
    layout = (resolved_run.parent.parent.parent.name, resolved_run.parent.parent.name)
    if provenance.run_id != expected_run_id:
        raise ValueError("formal training run run_id does not match its directory")
    if resolved_run.name != expected_variant or provenance.variant != expected_variant:
        raise ValueError("formal training run variant mismatch")
    if layout != ("v06_a", config.category) or provenance.category != config.category:
        raise ValueError("formal training run category/layout mismatch")
    if provenance.identity_manifest_authority_sha256 != identity_manifest.authority_digest:
        raise ValueError("formal training run manifest digest mismatch")
    _read_and_validate_attempt_contract(
        run_directory=resolved_run.parent,
        expected=_attempt_contract_payload(
            run_directory=resolved_run.parent,
            config=config,
            identity_manifest=identity_manifest,
        ),
    )
    if provenance.checkpoint_relpath != "checkpoint":
        raise ValueError("formal training run checkpoint_relpath must equal checkpoint")
    checkpoint = resolved_run / provenance.checkpoint_relpath
    try:
        resolved_checkpoint = checkpoint.resolve(strict=True)
        resolved_checkpoint.relative_to(resolved_run)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("formal training run checkpoint_relpath escapes or is missing") from error
    if checkpoint.is_symlink() or not resolved_checkpoint.is_dir():
        raise ValueError("formal training run checkpoint must be a real contained directory")

    metadata_path = resolved_checkpoint / "metadata.json"
    weights_path = resolved_checkpoint / "model.safetensors"
    for path in (metadata_path, weights_path):
        try:
            resolved_file = path.resolve(strict=True)
            resolved_file.relative_to(resolved_checkpoint)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError("formal checkpoint requires real contained files") from error
        if path.is_symlink() or not resolved_file.is_file():
            raise ValueError("formal checkpoint requires real contained files")
    try:
        metadata_sha256 = _sha256_file(metadata_path)
        weights_sha256 = _sha256_file(weights_path)
    except OSError as error:
        raise ValueError(f"formal training run checkpoint file is missing: {error}") from error
    if provenance.checkpoint_metadata_sha256 != metadata_sha256:
        raise ValueError("formal training run metadata SHA-256 mismatch")
    if provenance.checkpoint_weights_sha256 != weights_sha256:
        raise ValueError("formal training run weights SHA-256 mismatch")

    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"}:
        raise ValueError("formal Module-A loading device must be CPU or CUDA")

    def load_checkpoint() -> tuple[A1MaskUNet, A1CheckpointMetadata]:
        model = _locally_seeded_model(
            config=config,
            variant=expected_variant,
            device=torch_device,
        )
        metadata = load_a1_checkpoint(
            checkpoint=resolved_checkpoint,
            model=model,
            config=config,
            expected_variant=expected_variant,
        )
        return model, metadata

    if torch_device.type == "cuda":
        with formal_cuda_bootstrap_scope(torch_device):
            model, metadata = load_checkpoint()
    else:
        model, metadata = load_checkpoint()
    expected = {
        "schema_version": "v06_a_training_run_rc2",
        "training_protocol": _TRAINING_PROTOCOL,
        "training_seed": config.training_seed,
        "completed_steps": config.training_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "gradient_clip": config.gradient_clip,
        "target_kind": "plain_geometry_transfer",
        "diffusion_steps": config.diffusion_steps,
        "experiment_config_sha256": metadata.experiment_config_sha256,
        "model_construction_signature": metadata.model_construction_signature,
        "checkpoint_weights_sha256": metadata.weights_sha256,
    }
    mismatches = {
        name: (getattr(provenance, name), value)
        for name, value in expected.items()
        if getattr(provenance, name) != value
    }
    if mismatches:
        raise ValueError(f"formal training run config/model metadata mismatch: {mismatches}")
    model.eval()
    return A1FormalRun(
        resolved_run,
        resolved_checkpoint,
        metadata,
        provenance,
        model,
    )


_GENERATOR_TOKEN = object()


class A1ProposalGenerator:
    """Formal-run-validated, category-specific one-candidate A1 facade."""

    def __init__(
        self,
        token: object,
        *,
        config: V06AExperimentConfig,
        identity_manifest: IdentityManifest,
        variant: str,
        model: A1MaskUNet,
        diffusion_schedule: MaskDiffusionSchedule,
        sampler: A1ProposalSampler,
        metadata: A1CheckpointMetadata,
        provenance: A1TrainingProvenance,
    ) -> None:
        if token is not _GENERATOR_TOKEN:
            raise ValueError("A1ProposalGenerator must be created from a validated formal run")
        self.config = config
        self.identity_manifest = identity_manifest
        self.variant = variant
        self.model = model
        self.diffusion_schedule = diffusion_schedule
        self.sampler = sampler
        self.metadata = metadata
        self.provenance = provenance
        self._formal_load_device = next(model.parameters()).device
        allowed = (
            identity_manifest.normal_fit
            + identity_manifest.normal_dev
            + identity_manifest.open_normals
        )
        self._normal_records = {record.sample_id: record for record in allowed}

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint: Path,
        config: V06AExperimentConfig,
        identity_manifest: IdentityManifest,
        variant: str,
        device: str | torch.device = "cpu",
    ) -> A1ProposalGenerator:
        variant = require_supported_variant(variant)
        _validate_authority(config, identity_manifest)
        if not isinstance(checkpoint, Path):
            raise TypeError("checkpoint must be a pathlib.Path")
        torch.device(device)
        raise ValueError(
            "checkpoint-only loading cannot create a formal proposal generator; "
            "use from_training_run"
        )

    @classmethod
    def from_training_run(
        cls,
        *,
        run_directory: Path,
        config: V06AExperimentConfig,
        identity_manifest: IdentityManifest,
        expected_variant: str,
        device: str | torch.device = "cpu",
    ) -> A1ProposalGenerator:
        """Create the formal inference facade only from a validated completed run."""

        loaded = load_formal_a1_run(
            run_directory=run_directory,
            config=config,
            identity_manifest=identity_manifest,
            expected_variant=expected_variant,
            device=device,
        )
        schedule = MaskDiffusionSchedule(diffusion_steps=config.diffusion_steps)
        sampler = A1ProposalSampler(
            model=loaded.model,
            schedule=schedule,
            sampling_steps=config.sampling_steps,
        )
        return cls(
            _GENERATOR_TOKEN,
            config=config,
            identity_manifest=identity_manifest,
            variant=expected_variant,
            model=loaded.model,
            diffusion_schedule=schedule,
            sampler=sampler,
            metadata=loaded.metadata,
            provenance=loaded.provenance,
        )

    def generate(
        self,
        *,
        normal: LoadedIdentity,
        defect_type: str,
        seed: int,
    ) -> np.ndarray:
        """Return one nearest-upsampled bool mask; sampler failures propagate without retry."""

        try:
            device = next(self.model.parameters()).device
        except StopIteration as error:
            raise ValueError("formal proposal model has no parameters") from error
        if device != self._formal_load_device:
            raise ValueError("proposal model device changed after formal load")
        if device.type == "cuda":
            with cuda_determinism_scope(device):
                return self._generate_once(
                    normal=normal,
                    defect_type=defect_type,
                    seed=seed,
                )
        if device.type != "cpu":
            raise ValueError("formal Module-A proposal device must be CPU or CUDA")
        return self._generate_once(
            normal=normal,
            defect_type=defect_type,
            seed=seed,
        )

    def _generate_once(
        self,
        *,
        normal: LoadedIdentity,
        defect_type: str,
        seed: int,
    ) -> np.ndarray:
        """Execute one formal rollout inside the caller's selected device policy."""

        if not isinstance(normal, LoadedIdentity):
            raise TypeError("normal must be a LoadedIdentity")
        if normal.record.category != self.config.category:
            raise ValueError("normal category does not match the checkpoint category")
        expected = self._normal_records.get(normal.record.sample_id)
        if expected is None or expected != normal.record:
            raise ValueError("inference normal must be an exact fit/dev/open_normal identity")
        if normal.anomaly_gt is not None:
            raise ValueError("inference normal cannot carry anomaly ground truth")
        try:
            type_id = self.identity_manifest.defect_types.index(defect_type)
        except (ValueError, TypeError) as error:
            raise ValueError("defect_type is outside the checkpoint vocabulary") from error

        structure = estimate_normal_structure(normal, config=self.config)
        valid_128 = downsample_all_4x4(normal.valid)
        if not valid_128.any():
            raise ValueError("inference normal has no fully valid proposal cell")
        parameter = next(self.model.parameters())

        def condition_tensor(value: np.ndarray) -> torch.Tensor:
            array = np.array(value, dtype=np.float32, copy=True)[None, None]
            return torch.from_numpy(array).to(device=parameter.device, dtype=parameter.dtype)

        valid = torch.from_numpy(np.array(valid_128, copy=True)[None, None]).to(parameter.device)
        normal_rgb = None
        if self.variant == "a1_sv_plain":
            rgb = _normalise_rgb(normal.rgb)[None]
            normal_rgb = torch.from_numpy(np.ascontiguousarray(rgb)).to(
                device=parameter.device, dtype=parameter.dtype
            )
        condition = A1InferenceCondition(
            valid=valid,
            sdf=condition_tensor(structure.sdf_128),
            u_x=condition_tensor(structure.u_x),
            u_y=condition_tensor(structure.u_y),
            type_id=type_id,
            normal_rgb=normal_rgb,
        )
        sample = self.sampler.sample(condition=condition, seed=seed)
        expected_trajectory = tuple(
            int(value)
            for value in self.diffusion_schedule.sampling_timesteps(
                sampling_steps=self.config.sampling_steps
            ).tolist()
        )
        if sample.seed != seed or sample.ddim_timesteps != expected_trajectory:
            raise ValueError("sampler did not return the configured single trajectory")
        mask_128 = sample.mask_128.detach().to(device="cpu")
        mask_512 = sample.mask_512.detach().to(device="cpu")
        if (
            mask_128.shape != (128, 128)
            or mask_128.dtype != torch.bool
            or mask_512.shape != (512, 512)
            or mask_512.dtype != torch.bool
        ):
            raise ValueError("sampler masks must be bool arrays on the frozen grids")
        mask_128_array = mask_128.numpy()
        mask_512_array = mask_512.numpy()
        expected_512 = nearest_upsample_4x4(mask_128_array)
        valid_512 = nearest_upsample_4x4(valid_128)
        if (
            not mask_128_array.any()
            or np.all(mask_128_array[valid_128])
            or np.any(mask_128_array & ~valid_128)
            or not np.array_equal(mask_512_array, expected_512)
            or np.any(mask_512_array & ~valid_512)
            or np.any(mask_512_array & ~normal.valid)
        ):
            raise ValueError("checkpoint proposal violates nonempty/nonfull/subset-of-V binding")
        return mask_512_array.astype(np.bool_, copy=True)

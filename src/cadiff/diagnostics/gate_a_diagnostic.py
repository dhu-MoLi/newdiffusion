from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from cadiff.config import ProjectConfig
from cadiff.data.geometry import resize_pad_image
from cadiff.data.mvtec_manifest import ManifestRecord, build_mvtec_manifest
from cadiff.diagnostics.config import DiagnosticCase, DiagnosticConfig, PromptMode
from cadiff.diagnostics.noise_replay import (
    invert_with_noise_replay,
    reconstruct_with_noise_replay,
)
from cadiff.diagnostics.selection import (
    SelectedDiagnosticSample,
    select_diagnostic_samples,
    selection_document,
)
from cadiff.diagnostics.trajectory_trace import reconstruct_with_trace
from cadiff.diffusion.components import REQUIRED, FrozenComponents, load_frozen_components
from cadiff.diffusion.ddim_inversion import invert, reconstruct
from cadiff.diffusion.vae_roundtrip import decode_latent, encode_mode, image_to_tensor
from cadiff.p0 import (
    ExperimentConfig,
    GateConfig,
    ImageMetrics,
    _manifest_sha,
    _set_determinism,
    evaluate_gate,
    masked_reconstruction_metrics,
    select_normal_train,
    tree_snapshot,
)
from cadiff.provenance import collect_provenance, sha256_file, write_json_atomic

SAMPLE_SCHEMA = "gate_a_diagnostic_sample_v1"
DIAGNOSTIC_OUTPUT_NAME = "gate_a_rootcause_v1"
SUPPORTED_PHASES = {
    "baseline-repeat": "baseline_repeat",
    "single-factor": "single_factor",
}
NOISE_REPLAY_PHASES = {
    "noise-replay-six-r1": 1,
    "noise-replay-six-r2": 2,
}
NOISE_REPLAY_FULL_PHASES = {
    "noise-replay-full-r1": 1,
    "noise-replay-full-r2": 2,
}
NOISE_REPLAY_CASE_ID = "d5_noise_replay_50_class_fp16"


@dataclass(frozen=True)
class DiagnosticMeasurement:
    metrics: ImageMetrics
    final_latent_sha256: str | None
    decoded_sha256: str | None
    trajectory: Mapping[str, object]
    finite: bool
    runtime_seconds: float
    peak_cuda_memory_allocated: int
    core_trace_bitwise_equal: bool
    inversion_core_bitwise_equal: bool = True
    reconstruction_unet_calls: int | None = None


@dataclass(frozen=True)
class CaseAggregate:
    case_id: str
    factor: str
    steps: int
    prompt_mode: PromptMode
    dtype: str
    target_max_extra_l1: float
    six_image_median_extra_l1: float
    bottle_regression: float
    finite: bool
    deterministic: bool
    compute_cost_rank: int


def _aggregate_ranking_key(result: CaseAggregate) -> tuple[float, float, int, str]:
    return (
        result.target_max_extra_l1,
        result.six_image_median_extra_l1,
        result.compute_cost_rank,
        result.case_id,
    )


def rank_single_factors(
    baseline: CaseAggregate,
    cases: Sequence[CaseAggregate],
    *,
    bottle_regression_max: float,
) -> tuple[CaseAggregate, ...]:
    if baseline.factor != "baseline" or not baseline.finite or not baseline.deterministic:
        raise ValueError("ranking requires a finite deterministic baseline")
    if not math.isfinite(bottle_regression_max) or bottle_regression_max < 0:
        raise ValueError("bottle_regression_max must be finite and non-negative")
    allowed_factors = {"steps", "prompt", "dtype"}
    seen: set[str] = set()
    effective: list[CaseAggregate] = []
    for candidate in cases:
        if candidate.case_id in seen:
            raise ValueError("candidate case IDs must be unique")
        seen.add(candidate.case_id)
        values = (
            candidate.target_max_extra_l1,
            candidate.six_image_median_extra_l1,
            candidate.bottle_regression,
        )
        if candidate.factor not in allowed_factors:
            raise ValueError(f"candidate factor is invalid: {candidate.factor}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"candidate aggregate contains non-finite values: {candidate.case_id}")
        if (
            candidate.finite
            and candidate.deterministic
            and candidate.bottle_regression <= bottle_regression_max
            and candidate.target_max_extra_l1 < baseline.target_max_extra_l1
            and candidate.six_image_median_extra_l1 <= baseline.six_image_median_extra_l1
        ):
            effective.append(candidate)
    return tuple(sorted(effective, key=_aggregate_ranking_key))


def _combined_case(left: CaseAggregate, right: CaseAggregate) -> DiagnosticCase:
    steps = 50
    prompt_mode = PromptMode.CLASS
    dtype = "float16"
    for candidate in (left, right):
        if candidate.factor == "steps":
            steps = candidate.steps
        elif candidate.factor == "prompt":
            prompt_mode = candidate.prompt_mode
        elif candidate.factor == "dtype":
            dtype = candidate.dtype
    identifiers = sorted((left.case_id, right.case_id))
    return DiagnosticCase(
        case_id=f"d4_{identifiers[0]}__{identifiers[1]}",
        phase="minimal_combination",
        steps=steps,
        prompt_mode=prompt_mode,
        dtype=dtype,
        repeat=1,
    )


def build_minimal_combinations(
    baseline: CaseAggregate,
    effective_factors: Sequence[CaseAggregate],
    *,
    maximum: int,
) -> tuple[DiagnosticCase, ...]:
    if type(maximum) is not int or not 0 < maximum <= 3:
        raise ValueError("maximum combinations must be an exact integer in [1, 3]")
    if baseline.factor != "baseline":
        raise ValueError("combination generation requires the baseline aggregate")
    best_by_factor: dict[str, CaseAggregate] = {}
    for candidate in sorted(effective_factors, key=_aggregate_ranking_key):
        if candidate.factor not in {"steps", "prompt", "dtype"}:
            raise ValueError(f"candidate factor is invalid: {candidate.factor}")
        best_by_factor.setdefault(candidate.factor, candidate)
    if len(best_by_factor) < 2:
        return ()
    pairs = list(itertools.combinations(best_by_factor.values(), 2))

    def estimated_key(pair: tuple[CaseAggregate, CaseAggregate]) -> tuple[float, int, str, str]:
        gain = sum(
            baseline.target_max_extra_l1 - candidate.target_max_extra_l1
            for candidate in pair
        )
        cost = sum(candidate.compute_cost_rank for candidate in pair)
        names = sorted(candidate.case_id for candidate in pair)
        return (-gain, cost, names[0], names[1])

    return tuple(_combined_case(*pair) for pair in sorted(pairs, key=estimated_key)[:maximum])


def _phase_root(output: Path, phase: str) -> Path:
    if phase in NOISE_REPLAY_PHASES or phase in NOISE_REPLAY_FULL_PHASES:
        return output / "trajectory_correction" / "noise_replay_v1" / phase
    return output / phase


def _case_documents(output: Path, phase: str, case_id: str) -> list[Mapping[str, Any]]:
    sample_root = _phase_root(output, phase) / case_id / "samples"
    paths = sorted(sample_root.glob("*.json"))
    if len(paths) != 6:
        raise RuntimeError(f"case {case_id} must contain exactly six sample documents")
    documents = [_read_strict_json(path) for path in paths]
    relpaths = [document.get("image_relpath") for document in documents]
    if len(set(relpaths)) != 6:
        raise RuntimeError(f"case {case_id} sample paths must be unique")
    return documents


def _repeat_signature(documents: Sequence[Mapping[str, Any]]) -> str:
    stable = []
    for document in sorted(documents, key=lambda item: str(item.get("image_relpath"))):
        stable.append(
            {
                "image_relpath": document.get("image_relpath"),
                "image_sha256": document.get("image_sha256"),
                "status": document.get("status"),
                "finite": document.get("finite"),
                "metrics": document.get("metrics"),
                "final_latent_sha256": document.get("final_latent_sha256"),
                "decoded_sha256": document.get("decoded_sha256"),
                "trajectory": document.get("trajectory"),
                "core_trace_bitwise_equal": document.get("core_trace_bitwise_equal"),
                "inversion_core_bitwise_equal": document.get(
                    "inversion_core_bitwise_equal"
                ),
                "reconstruction_unet_calls": document.get("reconstruction_unet_calls"),
            }
        )
    return _canonical_sha(stable)


def _extra_by_path(documents: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for document in documents:
        if document.get("status") != "complete" or document.get("finite") is not True:
            raise RuntimeError("candidate ranking requires complete finite sample documents")
        metrics = document.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError("candidate sample metrics must be a mapping")
        value = metrics.get("extra_l1")
        if type(value) not in {float, int} or isinstance(value, bool):
            raise RuntimeError("candidate extra_l1 must be a finite number")
        extra_l1 = float(value)
        if not math.isfinite(extra_l1):
            raise RuntimeError("candidate extra_l1 must be finite")
        relative = document.get("image_relpath")
        if not isinstance(relative, str) or relative in values:
            raise RuntimeError("candidate image_relpath must be a unique string")
        values[relative] = extra_l1
    return values


def _case_factor(case: DiagnosticCase) -> str:
    changes = [
        name
        for name, changed in (
            ("steps", case.steps != 50),
            ("prompt", case.prompt_mode is not PromptMode.CLASS),
            ("dtype", case.dtype != "float16"),
        )
        if changed
    ]
    if len(changes) != 1:
        raise RuntimeError(f"single-factor case must change exactly one factor: {case.case_id}")
    return changes[0]


def _aggregate_documents(
    case: DiagnosticCase,
    documents: Sequence[Mapping[str, Any]],
    baseline_by_path: Mapping[str, float],
    *,
    factor: str,
    deterministic: bool,
) -> CaseAggregate:
    values = _extra_by_path(documents)
    if set(values) != set(baseline_by_path):
        raise RuntimeError(f"case {case.case_id} does not use the frozen six samples")
    by_relative_document = {
        str(document["image_relpath"]): document for document in documents
    }
    target = [
        value
        for relative, value in values.items()
        if by_relative_document[relative].get("category") in {"grid", "transistor"}
    ]
    bottle_regressions = [
        value - baseline_by_path[relative]
        for relative, value in values.items()
        if by_relative_document[relative].get("category") == "bottle"
    ]
    if len(target) != 4 or len(bottle_regressions) != 2:
        raise RuntimeError("case categories do not match the fixed 2+2+2 diagnostic selection")
    return CaseAggregate(
        case.case_id,
        factor,
        case.steps,
        case.prompt_mode,
        case.dtype,
        max(target),
        float(statistics.median(values.values())),
        max(bottle_regressions),
        True,
        deterministic,
        case.steps * (2 if case.dtype == "float32" else 1),
    )


def write_candidate_ranking(output: Path, config: DiagnosticConfig) -> dict[str, object]:
    baseline_cases = [
        case for case in config.single_factor_cases if case.phase == "baseline_repeat"
    ]
    if len(baseline_cases) != 2:
        raise RuntimeError("candidate ranking requires exactly two D0 cases")
    baseline_documents = [
        _case_documents(output, "baseline-repeat", case.case_id) for case in baseline_cases
    ]
    if _repeat_signature(baseline_documents[0]) != _repeat_signature(baseline_documents[1]):
        raise RuntimeError("D0 repeats are not deterministic")
    baseline_values = _extra_by_path(baseline_documents[0])
    baseline = _aggregate_documents(
        DiagnosticCase("d0_baseline", "baseline", 50, PromptMode.CLASS, "float16", 1),
        baseline_documents[0],
        baseline_values,
        factor="baseline",
        deterministic=True,
    )
    single_cases = [case for case in config.single_factor_cases if case.phase == "single_factor"]
    aggregates = [
        _aggregate_documents(
            case,
            _case_documents(output, "single-factor", case.case_id),
            baseline_values,
            factor=_case_factor(case),
            deterministic=True,
        )
        for case in single_cases
    ]
    effective = rank_single_factors(
        baseline,
        aggregates,
        bottle_regression_max=config.bottle_regression_max,
    )
    combinations = build_minimal_combinations(
        baseline,
        effective,
        maximum=config.maximum_combinations,
    )
    next_action = (
        "run_minimal_combinations"
        if combinations
        else "promote_single_factor"
        if effective
        else "trajectory_cycle_correction"
    )
    effective_ids = {item.case_id for item in effective}
    document: dict[str, object] = {
        "schema_version": "gate_a_diagnostic_ranking_v1",
        "baseline_deterministic": True,
        "baseline": dataclasses.asdict(baseline),
        "single_factors": [
            {**dataclasses.asdict(item), "effective": item.case_id in effective_ids}
            for item in sorted(aggregates, key=_aggregate_ranking_key)
        ],
        "effective_single_factors": [dataclasses.asdict(item) for item in effective],
        "minimal_combinations": [dataclasses.asdict(item) for item in combinations],
        "ranking_order": [
            "target_max_extra_l1",
            "six_image_median_extra_l1",
            "compute_cost_rank",
            "case_id",
        ],
        "bottle_regression_max": config.bottle_regression_max,
        "next_action": next_action,
    }
    write_json_atomic(output / "candidate_ranking.json", document)
    return document


def write_noise_replay_promotion(
    output: Path, config: DiagnosticConfig
) -> dict[str, object]:
    baseline_cases = [
        case for case in config.single_factor_cases if case.phase == "baseline_repeat"
    ]
    if len(baseline_cases) != 2:
        raise RuntimeError("noise replay promotion requires exactly two D0 cases")
    baseline_documents = [
        _case_documents(output, "baseline-repeat", case.case_id) for case in baseline_cases
    ]
    if _repeat_signature(baseline_documents[0]) != _repeat_signature(
        baseline_documents[1]
    ):
        raise RuntimeError("D0 repeats are not deterministic")

    repeat_documents = [
        _case_documents(output, phase, NOISE_REPLAY_CASE_ID)
        for phase in NOISE_REPLAY_PHASES
    ]
    repeat_signatures = [_repeat_signature(documents) for documents in repeat_documents]
    if repeat_signatures[0] != repeat_signatures[1]:
        raise RuntimeError("noise replay repeats are not bitwise identical")

    baseline_values = _extra_by_path(baseline_documents[0])
    baseline = _aggregate_documents(
        DiagnosticCase("d0_baseline", "baseline", 50, PromptMode.CLASS, "float16", 1),
        baseline_documents[0],
        baseline_values,
        factor="baseline",
        deterministic=True,
    )
    candidate_case = DiagnosticCase(
        NOISE_REPLAY_CASE_ID,
        "trajectory_correction",
        50,
        PromptMode.CLASS,
        "float16",
        1,
    )
    candidate = _aggregate_documents(
        candidate_case,
        repeat_documents[0],
        baseline_values,
        factor="trajectory_correction",
        deterministic=True,
    )
    inversion_equal = all(
        document.get("inversion_core_bitwise_equal") is True
        for documents in repeat_documents
        for document in documents
    )
    reconstruction_calls = [
        document.get("reconstruction_unet_calls")
        for documents in repeat_documents
        for document in documents
    ]
    no_reconstruction_unet = all(
        type(value) is int and value == 0 for value in reconstruction_calls
    )
    criteria = {
        "repeat_signatures_identical": True,
        "all_complete_finite": candidate.finite,
        "inversion_core_bitwise_equal": inversion_equal,
        "reconstruction_unet_calls_zero": no_reconstruction_unet,
        "bottle_regression_within_limit": (
            candidate.bottle_regression <= config.bottle_regression_max
        ),
        "target_max_improved": (
            candidate.target_max_extra_l1 < baseline.target_max_extra_l1
        ),
        "six_image_median_not_worse": (
            candidate.six_image_median_extra_l1
            <= baseline.six_image_median_extra_l1
        ),
    }
    promoted = all(criteria.values())
    document: dict[str, object] = {
        "schema_version": "gate_a_noise_replay_promotion_v1",
        "promoted": promoted,
        "repeat_signatures_identical": True,
        "repeat_signatures": repeat_signatures,
        "reconstruction_unet_calls": 0 if no_reconstruction_unet else None,
        "baseline": dataclasses.asdict(baseline),
        "candidate": dataclasses.asdict(candidate),
        "criteria": criteria,
        "bottle_regression_max": config.bottle_regression_max,
        "next_action": "run_noise_replay_full_gate" if promoted else "stop",
    }
    target = output / "trajectory_correction" / "noise_replay_v1" / "promotion.json"
    write_json_atomic(target, document)
    return document


def _full_selected_samples(
    context: DiagnosticContext, records: Sequence[ManifestRecord]
) -> tuple[SelectedDiagnosticSample, ...]:
    selected = select_normal_train(records, context.experiment, context.gate)
    report = _read_strict_json(context.frozen_metrics)
    raw_images = report.get("images")
    if not isinstance(raw_images, list):
        raise RuntimeError("frozen P0 report images must be a list")
    by_path: dict[str, Mapping[str, Any]] = {}
    for raw in raw_images:
        if not isinstance(raw, Mapping):
            raise RuntimeError("frozen P0 image rows must be mappings")
        relative = raw.get("image_relpath")
        if not isinstance(relative, str) or relative in by_path:
            raise RuntimeError("frozen P0 image paths must be unique strings")
        by_path[relative] = raw
    samples: list[SelectedDiagnosticSample] = []
    for record in selected:
        raw = by_path.get(record.image_relpath)
        if raw is None or raw.get("category") != record.category:
            raise RuntimeError("full Gate selection is absent from the frozen P0 report")
        value = raw.get("extra_l1")
        if type(value) not in {float, int} or isinstance(value, bool):
            raise RuntimeError("frozen P0 extra_l1 must be numeric")
        baseline = float(value)
        if not math.isfinite(baseline):
            raise RuntimeError("frozen P0 extra_l1 must be finite")
        samples.append(
            SelectedDiagnosticSample(
                record.category,
                record.image_relpath,
                record.image_sha256,
                baseline,
                "full_gate",
            )
        )
    if len(samples) != 36:
        raise RuntimeError("noise replay full Gate requires exactly 36 samples")
    return tuple(samples)


def _load_or_create_full_selection(
    output: Path,
    samples: Sequence[SelectedDiagnosticSample],
    metrics_sha256: str,
) -> str:
    document = {
        "schema_version": "gate_a_noise_replay_full_selection_v1",
        "frozen_metrics_sha256": metrics_sha256,
        "count": 36,
        "samples": [dataclasses.asdict(sample) for sample in samples],
    }
    path = output / "trajectory_correction" / "noise_replay_v1" / "full_selection.json"
    if path.exists():
        if dict(_read_strict_json(path)) != document:
            raise RuntimeError("existing full Gate selection does not match frozen inputs")
    else:
        write_json_atomic(path, document)
    return sha256_file(path)


def _full_phase_documents(
    output: Path, phase: str
) -> list[Mapping[str, Any]]:
    paths = sorted(
        (_phase_root(output, phase) / NOISE_REPLAY_CASE_ID / "samples").glob(
            "*.json"
        )
    )
    if len(paths) != 36:
        raise RuntimeError(f"{phase} must contain exactly 36 sample documents")
    documents = [_read_strict_json(path) for path in paths]
    if len({document.get("image_relpath") for document in documents}) != 36:
        raise RuntimeError(f"{phase} sample paths must be unique")
    return documents


def _write_full_gate_report(
    output: Path,
    phase: str,
    samples: Sequence[SelectedDiagnosticSample],
    gate: GateConfig,
) -> dict[str, Any]:
    documents = _full_phase_documents(output, phase)
    by_path = {str(document["image_relpath"]): document for document in documents}
    metrics: list[ImageMetrics] = []
    for sample in samples:
        document = by_path[sample.image_relpath]
        if document.get("status") == "unrun":
            continue
        raw_metrics = document.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise RuntimeError("full Gate sample metrics must be a mapping")
        values = []
        for name in ("vae_l1", "ddim_l1", "extra_l1", "vae_psnr", "ddim_psnr"):
            value = raw_metrics.get(name)
            values.append(
                float(value)
                if type(value) in {float, int} and not isinstance(value, bool)
                else math.nan
            )
        metrics.append(ImageMetrics(sample.category, sample.image_relpath, *values))
    report = evaluate_gate(metrics, gate)
    report["images"] = [
        {
            "category": sample.category,
            "image_relpath": sample.image_relpath,
            **dict(by_path[sample.image_relpath]["metrics"]),
            "finite": by_path[sample.image_relpath].get("finite"),
            "status": by_path[sample.image_relpath].get("status"),
            "reason": by_path[sample.image_relpath].get("reason"),
            "final_latent_sha256": by_path[sample.image_relpath].get(
                "final_latent_sha256"
            ),
            "decoded_sha256": by_path[sample.image_relpath].get("decoded_sha256"),
        }
        for sample in samples
    ]
    write_json_atomic(_phase_root(output, phase) / "gate_a_metrics.json", report)
    return report


def _write_full_confirmation(output: Path) -> dict[str, object]:
    phases = ("noise-replay-full-r1", "noise-replay-full-r2")
    reports = [
        _read_strict_json(_phase_root(output, phase) / "gate_a_metrics.json")
        for phase in phases
    ]
    if not all(report.get("passed") is True for report in reports):
        raise RuntimeError("both noise replay full Gate repeats must pass")
    signatures = [
        _repeat_signature(_full_phase_documents(output, phase)) for phase in phases
    ]
    if signatures[0] != signatures[1]:
        raise RuntimeError("noise replay full Gate repeats are not identical")
    document = {
        "schema_version": "gate_a_noise_replay_full_confirmation_v1",
        "confirmed": True,
        "repeat_signatures_identical": True,
        "repeat_signatures": signatures,
        "gate_a_metrics_sha256": [
            sha256_file(_phase_root(output, phase) / "gate_a_metrics.json")
            for phase in phases
        ],
    }
    write_json_atomic(
        output / "trajectory_correction" / "noise_replay_v1" / "full_confirmation.json",
        document,
    )
    return document


class DiagnosticRunner(Protocol):
    def reconstruct(
        self, record: ManifestRecord, prompt: str, steps: int
    ) -> DiagnosticMeasurement: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DiagnosticContext:
    paths: ProjectConfig
    experiment: ExperimentConfig
    gate: GateConfig
    diagnostic: DiagnosticConfig
    output: Path
    frozen_metrics: Path
    manifest: Path
    lock: Path
    config_paths: Mapping[str, Path]
    cwd: Path


def prompt_for(case: DiagnosticCase, category: str, generic_prompt: str) -> str:
    if case.prompt_mode is PromptMode.CLASS:
        return f"a photo of a normal {category}"
    if case.prompt_mode is PromptMode.EMPTY:
        return ""
    if case.prompt_mode is PromptMode.GENERIC:
        return generic_prompt
    raise ValueError(f"unsupported prompt mode: {case.prompt_mode}")


def torch_dtype_for(case: DiagnosticCase) -> torch.dtype:
    try:
        return {"float16": torch.float16, "float32": torch.float32}[case.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported diagnostic dtype: {case.dtype}") from error


def prepare_diagnostic_output(output: Path, paths: ProjectConfig) -> Path:
    runs_root = paths.runs_root.resolve(strict=True)
    expected = runs_root / "v0_baseline" / "diagnostics" / DIAGNOSTIC_OUTPUT_NAME
    requested = output.expanduser().absolute()
    resolved = requested.resolve()
    if requested != resolved:
        raise ValueError("diagnostic output must be a canonical non-symlink path")
    if resolved != expected:
        raise ValueError(f"output must equal the isolated diagnostic root: {expected}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as error:
        raise RuntimeError(f"invalid diagnostic JSON: {path}") from error
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"diagnostic JSON must be a mapping: {path}")
    return raw


def claim_sample_output(
    path: Path, fingerprint: str, image_sha256: str
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    document = _read_strict_json(path)
    if document.get("schema_version") != SAMPLE_SCHEMA:
        raise RuntimeError(f"sample schema mismatch: {path}")
    if document.get("status") != "complete":
        raise RuntimeError(f"sample is not safely resumable: {path}")
    if document.get("fingerprint") != fingerprint:
        raise RuntimeError(f"sample fingerprint mismatch: {path}")
    if document.get("image_sha256") != image_sha256:
        raise RuntimeError(f"sample source SHA256 mismatch: {path}")
    return document


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


class RealDiagnosticRunner:
    def __init__(self, config: ProjectConfig, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.config = config
        self.device = torch.device("cuda:0")
        self.dtype = dtype
        self.components: FrozenComponents = load_frozen_components(
            config.model_root, self.device, dtype
        )

    def reconstruct(
        self, record: ManifestRecord, prompt: str, steps: int
    ) -> DiagnosticMeasurement:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        source = self.config.dataset_root / record.image_relpath
        with Image.open(source) as image:
            prepared, valid, _ = resize_pad_image(image, self.config.image_size)
        image_tensor = image_to_tensor(prepared, self.device, self.dtype)
        reference = (image_tensor / 2 + 0.5).to(torch.float32)
        valid_tensor = pil_to_tensor(valid).unsqueeze(0).to(self.device)
        latent = encode_mode(self.components.vae, image_tensor)
        vae = decode_latent(self.components.vae, latent)
        trajectory = invert(self.components, latent, prompt, steps)
        core_latent = reconstruct(self.components, trajectory, prompt)
        traced = reconstruct_with_trace(self.components, trajectory, prompt)
        core_equal = torch.equal(core_latent, traced.final_latent)
        latent_finite = bool(torch.isfinite(traced.final_latent).all())
        if latent_finite and not core_equal:
            raise RuntimeError("core and traced reconstructions are not bitwise equal")
        trajectory_document = {
            "states": [dataclasses.asdict(item) for item in traced.states],
            "cycle": [dataclasses.asdict(item) for item in traced.cycle],
        }
        if latent_finite:
            ddim = decode_latent(self.components.vae, traced.final_latent)
            measured = masked_reconstruction_metrics(reference, vae, ddim, valid_tensor)
            latent_sha = _tensor_sha256(traced.final_latent)
            decoded_sha = _tensor_sha256(ddim)
        else:
            measured = ImageMetrics("", "", *(math.nan for _ in range(5)))
            latent_sha = None
            decoded_sha = None
        metric_finite = all(math.isfinite(value) for value in measured.as_dict().values())
        torch.cuda.synchronize(self.device)
        return DiagnosticMeasurement(
            measured,
            latent_sha,
            decoded_sha,
            trajectory_document,
            latent_finite and metric_finite and core_equal,
            time.perf_counter() - started,
            torch.cuda.max_memory_allocated(self.device),
            core_equal,
        )

    def close(self) -> None:
        del self.components
        torch.cuda.empty_cache()


class NoiseReplayDiagnosticRunner:
    def __init__(self, config: ProjectConfig, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.config = config
        self.device = torch.device("cuda:0")
        self.dtype = dtype
        self.components: FrozenComponents = load_frozen_components(
            config.model_root, self.device, dtype
        )

    @classmethod
    def from_components(
        cls,
        config: ProjectConfig,
        dtype: torch.dtype,
        components: FrozenComponents,
        *,
        device: torch.device,
    ) -> NoiseReplayDiagnosticRunner:
        instance = cls.__new__(cls)
        instance.config = config
        instance.device = device
        instance.dtype = dtype
        instance.components = components
        return instance

    def reconstruct(
        self, record: ManifestRecord, prompt: str, steps: int
    ) -> DiagnosticMeasurement:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        source = self.config.dataset_root / record.image_relpath
        with Image.open(source) as image:
            prepared, valid, _ = resize_pad_image(image, self.config.image_size)
        image_tensor = image_to_tensor(prepared, self.device, self.dtype)
        reference = (image_tensor / 2 + 0.5).to(torch.float32)
        valid_tensor = pil_to_tensor(valid).unsqueeze(0).to(self.device)
        latent = encode_mode(self.components.vae, image_tensor)
        vae = decode_latent(self.components.vae, latent)
        core_trajectory = invert(self.components, latent, prompt, steps)
        replay_trajectory = invert_with_noise_replay(
            self.components, latent, prompt, steps
        )
        inversion_equal = len(core_trajectory) == len(replay_trajectory.latents) and all(
            left.timestep == right.timestep and torch.equal(left.latent, right.latent)
            for left, right in zip(
                core_trajectory, replay_trajectory.latents, strict=True
            )
        )
        if not inversion_equal:
            raise RuntimeError("captured inversion does not match core inversion bitwise")
        replayed = reconstruct_with_noise_replay(self.components, replay_trajectory)
        latent_finite = bool(torch.isfinite(replayed.final_latent).all())
        trajectory_document = {
            "states": [dataclasses.asdict(item) for item in replayed.states],
            "cycle": [dataclasses.asdict(item) for item in replayed.cycle],
        }
        if latent_finite:
            ddim = decode_latent(self.components.vae, replayed.final_latent)
            measured = masked_reconstruction_metrics(reference, vae, ddim, valid_tensor)
            latent_sha = _tensor_sha256(replayed.final_latent)
            decoded_sha = _tensor_sha256(ddim)
        else:
            measured = ImageMetrics("", "", *(math.nan for _ in range(5)))
            latent_sha = None
            decoded_sha = None
        metric_finite = all(math.isfinite(value) for value in measured.as_dict().values())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_memory = 0
        return DiagnosticMeasurement(
            measured,
            latent_sha,
            decoded_sha,
            trajectory_document,
            latent_finite and metric_finite and inversion_equal,
            time.perf_counter() - started,
            peak_memory,
            inversion_equal,
            inversion_equal,
            0,
        )

    def close(self) -> None:
        del self.components
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _clean_git(cwd: Path) -> str:
    import subprocess

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("diagnostics require a clean git worktree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return head.stdout.strip()


def _model_snapshot(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        relative: {
            "size": (root / relative).stat().st_size,
            "mtime_ns": (root / relative).stat().st_mtime_ns,
            "sha256": sha256_file(root / relative),
        }
        for relative in REQUIRED
    }


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_inputs(
    context: DiagnosticContext,
) -> tuple[list[ManifestRecord], tuple[SelectedDiagnosticSample, ...], dict[str, object]]:
    start_commit = _clean_git(context.cwd)
    if sha256_file(context.frozen_metrics) != context.diagnostic.frozen_metrics_sha256:
        raise RuntimeError("frozen metrics hash changed before diagnostics")
    records = build_mvtec_manifest(context.paths.dataset_root, context.experiment.split_id)
    manifest_sha = _manifest_sha(records)
    if sha256_file(context.manifest) != manifest_sha:
        raise RuntimeError("manifest does not match a fresh read-only dataset scan")
    samples = select_diagnostic_samples(
        context.frozen_metrics,
        context.diagnostic.frozen_metrics_sha256,
        records,
        context.paths.dataset_root,
    )
    visa_root = context.paths.dataset_root.parent / "VisA"
    inputs: dict[str, object] = {
        "git_commit": start_commit,
        "frozen_metrics_sha256": context.diagnostic.frozen_metrics_sha256,
        "manifest_sha256": manifest_sha,
        "lock_sha256": sha256_file(context.lock),
        "config_sha256": {
            name: sha256_file(path) for name, path in sorted(context.config_paths.items())
        },
        "model_required": _model_snapshot(context.paths.model_root),
        "mvtec_snapshot": tree_snapshot(context.paths.dataset_root),
        "visa_snapshot": tree_snapshot(visa_root) if visa_root.is_dir() else None,
    }
    return records, samples, inputs


def _verify_inputs(context: DiagnosticContext, expected: Mapping[str, object]) -> None:
    commit = _clean_git(context.cwd)
    if commit != expected["git_commit"]:
        raise RuntimeError("diagnostic Git identity changed during execution")
    records = build_mvtec_manifest(context.paths.dataset_root, context.experiment.split_id)
    visa_root = context.paths.dataset_root.parent / "VisA"
    actual = {
        "frozen_metrics_sha256": sha256_file(context.frozen_metrics),
        "manifest_sha256": _manifest_sha(records),
        "lock_sha256": sha256_file(context.lock),
        "config_sha256": {
            name: sha256_file(path) for name, path in sorted(context.config_paths.items())
        },
        "model_required": _model_snapshot(context.paths.model_root),
        "mvtec_snapshot": tree_snapshot(context.paths.dataset_root),
        "visa_snapshot": tree_snapshot(visa_root) if visa_root.is_dir() else None,
    }
    comparable = {key: value for key, value in expected.items() if key != "git_commit"}
    if actual != comparable:
        raise RuntimeError("diagnostic immutable inputs changed during execution")


def _load_or_create_selection(
    output: Path,
    samples: Sequence[SelectedDiagnosticSample],
    metrics_sha256: str,
) -> tuple[dict[str, object], str]:
    document = selection_document(samples, metrics_sha256)
    path = output / "sample_selection.json"
    if path.exists():
        if dict(_read_strict_json(path)) != document:
            raise RuntimeError("existing diagnostic selection does not match frozen inputs")
    else:
        write_json_atomic(path, document)
    return document, sha256_file(path)


def _cases_for_phase(config: DiagnosticConfig, phase: str) -> tuple[DiagnosticCase, ...]:
    noise_phases = {**NOISE_REPLAY_PHASES, **NOISE_REPLAY_FULL_PHASES}
    if phase in noise_phases:
        return (
            DiagnosticCase(
                NOISE_REPLAY_CASE_ID,
                "trajectory_correction",
                50,
                PromptMode.CLASS,
                "float16",
                noise_phases[phase],
            ),
        )
    try:
        stored_phase = SUPPORTED_PHASES[phase]
    except KeyError as error:
        raise ValueError(f"unsupported diagnostic phase: {phase}") from error
    cases = tuple(case for case in config.single_factor_cases if case.phase == stored_phase)
    if not cases:
        raise ValueError(f"diagnostic phase has no cases: {phase}")
    return cases


def _case_fingerprint(
    case: DiagnosticCase,
    inputs: Mapping[str, object],
    selection_sha256: str,
    seed: int,
    determinism: Mapping[str, object],
) -> str:
    return _canonical_sha(
        {
            "schema_version": "gate_a_diagnostic_fingerprint_v1",
            "case": dataclasses.asdict(case),
            "inputs": inputs,
            "selection_sha256": selection_sha256,
            "seed": seed,
            "determinism": determinism,
        }
    )


def _sample_path(
    output: Path,
    phase: str,
    case: DiagnosticCase,
    sample: SelectedDiagnosticSample,
) -> Path:
    stem = Path(sample.image_relpath).stem
    return _phase_root(output, phase) / case.case_id / "samples" / (
        f"{sample.category}_{stem}.json"
    )


def _metric_document(metrics: ImageMetrics) -> dict[str, float | None]:
    return {
        name: value if math.isfinite(value) else None
        for name, value in metrics.as_dict().items()
    }


def _completed_document(
    case: DiagnosticCase,
    sample: SelectedDiagnosticSample,
    fingerprint: str,
    prompt: str,
    measurement: DiagnosticMeasurement,
) -> dict[str, object]:
    finite = measurement.finite and all(
        math.isfinite(value) for value in measurement.metrics.as_dict().values()
    )
    return {
        "schema_version": SAMPLE_SCHEMA,
        "case_id": case.case_id,
        "category": sample.category,
        "image_relpath": sample.image_relpath,
        "image_sha256": sample.image_sha256,
        "selection_role": sample.selection_role,
        "baseline_extra_l1": sample.baseline_extra_l1,
        "fingerprint": fingerprint,
        "prompt": prompt,
        "steps": case.steps,
        "dtype": case.dtype,
        "status": "complete" if finite else "nonfinite",
        "finite": finite,
        "metrics": _metric_document(measurement.metrics),
        "final_latent_sha256": measurement.final_latent_sha256 if finite else None,
        "decoded_sha256": measurement.decoded_sha256 if finite else None,
        "trajectory": measurement.trajectory,
        "runtime_seconds": (
            measurement.runtime_seconds if math.isfinite(measurement.runtime_seconds) else None
        ),
        "peak_cuda_memory_allocated": measurement.peak_cuda_memory_allocated,
        "core_trace_bitwise_equal": measurement.core_trace_bitwise_equal,
        "inversion_core_bitwise_equal": measurement.inversion_core_bitwise_equal,
        "reconstruction_unet_calls": measurement.reconstruction_unet_calls,
        "reason": None if finite else "non_finite_measurement",
    }


def _unrun_document(
    case: DiagnosticCase,
    sample: SelectedDiagnosticSample,
    fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": SAMPLE_SCHEMA,
        "case_id": case.case_id,
        "category": sample.category,
        "image_relpath": sample.image_relpath,
        "image_sha256": sample.image_sha256,
        "selection_role": sample.selection_role,
        "baseline_extra_l1": sample.baseline_extra_l1,
        "fingerprint": fingerprint,
        "status": "unrun",
        "finite": None,
        "metrics": {name: None for name in ImageMetrics("", "", 0, 0, 0, 0, 0).as_dict()},
        "final_latent_sha256": None,
        "decoded_sha256": None,
        "trajectory": {"states": [], "cycle": []},
        "runtime_seconds": None,
        "peak_cuda_memory_allocated": None,
        "core_trace_bitwise_equal": None,
        "inversion_core_bitwise_equal": None,
        "reconstruction_unet_calls": None,
        "reason": "short_circuited_non_finite",
    }


def _remaining_pairs(
    cases: Sequence[DiagnosticCase],
    samples: Sequence[SelectedDiagnosticSample],
    case_index: int,
    sample_index: int,
) -> list[tuple[DiagnosticCase, SelectedDiagnosticSample]]:
    pairs: list[tuple[DiagnosticCase, SelectedDiagnosticSample]] = []
    for later_case_index in range(case_index, len(cases)):
        start = sample_index + 1 if later_case_index == case_index else 0
        pairs.extend((cases[later_case_index], sample) for sample in samples[start:])
    return pairs


def _write_phase_summary(output: Path, phase: str) -> dict[str, object]:
    phase_root = _phase_root(output, phase)
    paths = sorted(phase_root.glob("*/samples/*.json"))
    documents = [dict(_read_strict_json(path)) for path in paths]
    summary = {
        "schema_version": "gate_a_diagnostic_phase_v1",
        "phase": phase,
        "count": len(documents),
        "status_counts": {
            status: sum(document.get("status") == status for document in documents)
            for status in ("complete", "nonfinite", "unrun")
        },
        "sample_sha256": {str(path.relative_to(output)): sha256_file(path) for path in paths},
    }
    write_json_atomic(phase_root / "summary.json", summary)
    return summary


def run_diagnostic_phase(
    context: DiagnosticContext,
    phase: str,
    runner_factory: Callable[[ProjectConfig, torch.dtype], DiagnosticRunner] | None = None,
) -> int:
    output = prepare_diagnostic_output(context.output, context.paths)
    if phase == "noise-replay-six-r2":
        repeat_one_summary = _phase_root(output, "noise-replay-six-r1") / "summary.json"
        if not repeat_one_summary.is_file():
            raise RuntimeError("noise replay repeat 1 must complete before repeat 2")
    if phase in NOISE_REPLAY_FULL_PHASES:
        promotion_path = (
            output / "trajectory_correction" / "noise_replay_v1" / "promotion.json"
        )
        if not promotion_path.is_file() or _read_strict_json(promotion_path).get(
            "promoted"
        ) is not True:
            raise RuntimeError("noise replay full Gate requires successful promotion")
        if phase == "noise-replay-full-r2":
            repeat_one_report = (
                _phase_root(output, "noise-replay-full-r1") / "gate_a_metrics.json"
            )
            if not repeat_one_report.is_file() or _read_strict_json(
                repeat_one_report
            ).get("passed") is not True:
                raise RuntimeError("noise replay full repeat 1 must pass before repeat 2")
    records, diagnostic_samples, inputs = _load_inputs(context)
    samples = (
        _full_selected_samples(context, records)
        if phase in NOISE_REPLAY_FULL_PHASES
        else diagnostic_samples
    )
    by_path = {record.image_relpath: record for record in records}
    if phase in NOISE_REPLAY_FULL_PHASES:
        selection_sha = _load_or_create_full_selection(
            output, samples, context.diagnostic.frozen_metrics_sha256
        )
    else:
        _, selection_sha = _load_or_create_selection(
            output, samples, context.diagnostic.frozen_metrics_sha256
        )
    determinism = _set_determinism(context.experiment.seed)
    cases = _cases_for_phase(context.diagnostic, phase)
    factory = runner_factory
    if factory is None:
        factory = (
            NoiseReplayDiagnosticRunner
            if phase in NOISE_REPLAY_PHASES or phase in NOISE_REPLAY_FULL_PHASES
            else RealDiagnosticRunner
        )
    fingerprints = {
        case.case_id: _case_fingerprint(
            case, inputs, selection_sha, context.experiment.seed, determinism
        )
        for case in cases
    }
    short_circuited = False
    for case_index, case in enumerate(cases):
        runner = factory(context.paths, torch_dtype_for(case))
        try:
            for sample_index, sample in enumerate(samples):
                path = _sample_path(output, phase, case, sample)
                claimed = claim_sample_output(
                    path, fingerprints[case.case_id], sample.image_sha256
                )
                if claimed is not None:
                    continue
                prompt = prompt_for(case, sample.category, context.diagnostic.generic_prompt)
                measurement = runner.reconstruct(by_path[sample.image_relpath], prompt, case.steps)
                document = _completed_document(
                    case, sample, fingerprints[case.case_id], prompt, measurement
                )
                write_json_atomic(path, document)
                if document["status"] == "nonfinite":
                    for later_case, later_sample in _remaining_pairs(
                        cases, samples, case_index, sample_index
                    ):
                        later_path = _sample_path(output, phase, later_case, later_sample)
                        write_json_atomic(
                            later_path,
                            _unrun_document(
                                later_case,
                                later_sample,
                                fingerprints[later_case.case_id],
                            ),
                        )
                    short_circuited = True
                    break
        finally:
            runner.close()
        if short_circuited:
            break
    summary = _write_phase_summary(output, phase)
    if phase == "single-factor" and not short_circuited:
        write_candidate_ranking(output, context.diagnostic)
        summary["candidate_ranking_sha256"] = sha256_file(output / "candidate_ranking.json")
        write_json_atomic(_phase_root(output, phase) / "summary.json", summary)
    if phase == "noise-replay-six-r2" and not short_circuited:
        promotion = write_noise_replay_promotion(output, context.diagnostic)
        summary["promotion_sha256"] = sha256_file(
            output / "trajectory_correction" / "noise_replay_v1" / "promotion.json"
        )
        summary["promoted"] = promotion["promoted"]
        write_json_atomic(_phase_root(output, phase) / "summary.json", summary)
    full_report: dict[str, Any] | None = None
    if phase in NOISE_REPLAY_FULL_PHASES:
        full_report = _write_full_gate_report(output, phase, samples, context.gate)
        summary["gate_a_metrics_sha256"] = sha256_file(
            _phase_root(output, phase) / "gate_a_metrics.json"
        )
        summary["passed"] = full_report["passed"]
        write_json_atomic(_phase_root(output, phase) / "summary.json", summary)
        if phase == "noise-replay-full-r2" and full_report["passed"] is True:
            confirmation = _write_full_confirmation(output)
            summary["full_confirmation_sha256"] = sha256_file(
                output
                / "trajectory_correction"
                / "noise_replay_v1"
                / "full_confirmation.json"
            )
            summary["confirmed"] = confirmation["confirmed"]
            write_json_atomic(_phase_root(output, phase) / "summary.json", summary)
    _verify_inputs(context, inputs)
    provenance = collect_provenance(context.paths, cwd=context.cwd)
    previous_phases: dict[str, object] = {}
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        old = _read_strict_json(provenance_path).get("completed_phases", {})
        if isinstance(old, Mapping):
            previous_phases = dict(old)
    previous_phases[phase] = summary
    provenance.update(
        {
            "schema_version": "gate_a_diagnostic_provenance_v1",
            "git_commit": inputs["git_commit"],
            "git_dirty": False,
            "inputs": inputs,
            "selection_sha256": selection_sha,
            "determinism": determinism,
            "completed_phases": previous_phases,
            "output": str(output),
        }
    )
    write_json_atomic(provenance_path, provenance)
    if phase in NOISE_REPLAY_FULL_PHASES:
        return 0 if full_report is not None and full_report["passed"] is True else 2
    return 1 if short_circuited else 0

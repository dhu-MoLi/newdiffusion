from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from cadiff.p2_c0.metrics import pixel_metrics
from cadiff.p3_a0.config import P3A0Config
from cadiff.p3_a0.contracts import build_response_schedule
from cadiff.p3_a0.dataset import MaskTarget, P3A0Data, ResponseInput
from cadiff.p3_a0.response import RawTimestepStats, ResponseBatch, ResponseMap
from cadiff.p3_a0.spatial import (
    calibrate_morphology,
    morphology_descriptor,
    morphology_distance,
    normalized_l1,
    otsu_binary,
    product_roi,
    random_baselines,
    sobel_baseline,
    spearman,
)


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    gate_groups: dict[str, dict[str, Any]]
    promoted: bool


def _as_2d(value: object, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return array


def _valid(item: ResponseInput) -> np.ndarray:
    valid = _as_2d(item.valid, name="valid").astype(bool, copy=False)
    if not valid.any():
        raise ValueError("response input valid region must be non-empty")
    return valid


def _rgb(item: ResponseInput) -> np.ndarray:
    pixels = item.pixels.detach().cpu().numpy()
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("response input pixels must have shape 3xHxW")
    rgb = np.moveaxis(pixels, 0, 2).astype(np.float64, copy=False)
    if not np.isfinite(rgb).all():
        raise ValueError("response input pixels must be finite")
    return rgb


def _validate_targets(data: P3A0Data, config: P3A0Config) -> dict[str, MaskTarget]:
    calibration_counts = {value: 0 for value in config.defect_types}
    for target in data.calibration_targets:
        if not isinstance(target, MaskTarget) or target.defect_type not in calibration_counts:
            raise ValueError("calibration targets must use the frozen defect types")
        calibration_counts[target.defect_type] += 1
        mask = _as_2d(target.mask, name="calibration mask")
        valid = _as_2d(target.valid, name="calibration valid")
        if mask.shape != valid.shape or not np.asarray(valid, dtype=bool).any():
            raise ValueError("calibration target geometry is invalid")
    if calibration_counts != config.calibration_counts:
        raise ValueError("calibration target accounting is invalid")

    targets: dict[str, MaskTarget] = {}
    auxiliary_counts = {value: 0 for value in config.defect_types}
    for target in data.auxiliary_targets:
        if not isinstance(target, MaskTarget) or target.sample_id in targets:
            raise ValueError("auxiliary targets must have unique sample IDs")
        if target.defect_type not in auxiliary_counts:
            raise ValueError("auxiliary targets must use the frozen defect types")
        mask = _as_2d(target.mask, name="auxiliary mask").astype(bool, copy=False)
        valid = _as_2d(target.valid, name="auxiliary valid").astype(bool, copy=False)
        if (
            mask.shape != valid.shape
            or not valid.any()
            or not (mask & valid).any()
            or not (~mask & valid).any()
        ):
            raise ValueError("auxiliary targets require aligned positive and negative valid pixels")
        auxiliary_counts[target.defect_type] += 1
        targets[target.sample_id] = target
    if auxiliary_counts != config.auxiliary_counts:
        raise ValueError("auxiliary target accounting is invalid")
    inputs = {value.sample_id: value for value in data.auxiliary_inputs}
    if set(targets) != set(inputs):
        raise ValueError("auxiliary target IDs must exactly match auxiliary input IDs")
    for sample_id, target in targets.items():
        item = inputs[sample_id]
        if target.defect_type != item.source_defect_type:
            raise ValueError("auxiliary target binding must match the input source defect type")
        target_valid = _as_2d(target.valid, name="auxiliary target valid").astype(bool, copy=False)
        if not np.array_equal(target_valid, _valid(item)):
            raise ValueError("auxiliary target valid region must equal the input valid region")
    return targets


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_timestep_stats(response_map: ResponseMap, config: P3A0Config) -> None:
    stats = response_map.timestep_stats
    if (
        not isinstance(stats, tuple)
        or len(stats) != 3
        or tuple(value.timestep for value in stats) != config.timesteps
    ):
        raise ValueError("paired-prompt provenance must contain the three frozen timesteps")
    for value in stats:
        if not isinstance(value, RawTimestepStats):
            raise ValueError("paired-prompt provenance timestep stats are invalid")
        raw_values = (
            value.minimum,
            value.maximum,
            value.mean,
            value.standard_deviation,
            value.q01,
            value.q99,
            value.dynamic_range,
        )
        if (
            value.finite is not True
            or type(value.degenerate) is not bool
            or not np.isfinite(raw_values).all()
            or not _is_sha256(value.noised_latent_sha256)
        ):
            raise ValueError("paired-prompt provenance raw timestep stats must be finite")


def _validate_paired_prompt_provenance(
    maps: dict[tuple[str, str, str, int], ResponseMap], config: P3A0Config
) -> None:
    groups: dict[tuple[str, str, int], list[ResponseMap]] = {}
    for (cohort, sample_id, _prompt_type, seed), response_map in maps.items():
        if not _is_sha256(response_map.image_latent_sha256) or not _is_sha256(
            response_map.noise_sha256
        ):
            raise ValueError("paired-prompt provenance hashes must be canonical SHA256 values")
        _validate_timestep_stats(response_map, config)
        groups.setdefault((cohort, sample_id, seed), []).append(response_map)
    for response_maps in groups.values():
        if len(response_maps) != len(config.defect_types):
            raise ValueError("paired-prompt provenance group is incomplete")
        if len({value.image_latent_sha256 for value in response_maps}) != 1:
            raise ValueError("paired-prompt provenance image latent identity differs by prompt")
        if len({value.noise_sha256 for value in response_maps}) != 1:
            raise ValueError("paired-prompt provenance noise identity differs by prompt")
        for index in range(len(config.timesteps)):
            if (
                len({value.timestep_stats[index].noised_latent_sha256 for value in response_maps})
                != 1
            ):
                raise ValueError(
                    "paired-prompt provenance noised latent identity differs by prompt"
                )


def _expected_response_keys(
    data: P3A0Data, config: P3A0Config
) -> dict[tuple[str, str, str, int], ResponseInput]:
    if len(data.normal_inputs) != config.normal_count:
        raise ValueError("normal input accounting is invalid")
    normal_ids = [value.sample_id for value in data.normal_inputs]
    auxiliary_ids = [value.sample_id for value in data.auxiliary_inputs]
    if len(set(normal_ids)) != len(normal_ids) or len(set(auxiliary_ids)) != len(auxiliary_ids):
        raise ValueError("response input sample IDs must be unique within each cohort")
    if set(normal_ids) & set(auxiliary_ids):
        raise ValueError("response input sample IDs must be unique across cohorts")

    source_counts = {value: 0 for value in config.defect_types}
    result: dict[tuple[str, str, str, int], ResponseInput] = {}
    for item in data.normal_inputs:
        if item.cohort != "normal_reference" or item.source_defect_type is not None:
            raise ValueError("normal input identity is invalid")
        for prompt_type in config.defect_types:
            for seed in config.seeds:
                result[(item.cohort, item.sample_id, prompt_type, seed)] = item
    for item in data.auxiliary_inputs:
        source_type = item.source_defect_type
        if item.cohort != "auxiliary_real" or source_type not in source_counts:
            raise ValueError("auxiliary input identity is invalid")
        source_counts[source_type] += 1
        for prompt_type in config.defect_types:
            for seed in config.seeds:
                result[(item.cohort, item.sample_id, prompt_type, seed)] = item
    if source_counts != config.auxiliary_counts:
        raise ValueError("auxiliary input accounting is invalid")
    return result


def _validate_responses(
    data: P3A0Data, responses: ResponseBatch, config: P3A0Config
) -> tuple[dict[tuple[str, str, str, int], ResponseMap], dict[str, MaskTarget]]:
    if not isinstance(data, P3A0Data) or not isinstance(responses, ResponseBatch):
        raise TypeError("data and responses must use the P3-A0 contracts")
    if not isinstance(config, P3A0Config):
        raise TypeError("config must be P3A0Config")
    expected = _expected_response_keys(data, config)
    schedule = build_response_schedule(
        tuple(value.sample_id for value in data.normal_inputs),
        {
            source_type: tuple(
                value.sample_id
                for value in data.auxiliary_inputs
                if value.source_defect_type == source_type
            )
            for source_type in config.defect_types
        },
        config.defect_types,
        config.seeds,
    )
    canonical_jobs = {
        (job.cohort, job.sample_id, job.prompt_defect_type, job.seed): job
        for job in (*schedule.normal, *schedule.auxiliary)
    }
    expected_normal = config.normal_count * len(config.defect_types) * len(config.seeds)
    expected_auxiliary = (
        sum(config.auxiliary_counts.values()) * len(config.defect_types) * len(config.seeds)
    )
    if expected_normal != 64 or expected_auxiliary != 384 or len(responses.maps) != 448:
        raise ValueError("response accounting requires exactly 64 normal and 384 auxiliary records")
    record_ids = [value.job.record_id for value in responses.maps]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("response record IDs must be unique")

    maps: dict[tuple[str, str, str, int], ResponseMap] = {}
    for response_map in responses.maps:
        if not isinstance(response_map, ResponseMap):
            raise TypeError("responses must contain only ResponseMap values")
        job = response_map.job
        key = (job.cohort, job.sample_id, job.prompt_defect_type, job.seed)
        source = expected.get(key)
        if key in maps:
            raise ValueError("response accounting contains a duplicate matrix key")
        if source is None or job.source_defect_type != source.source_defect_type:
            raise ValueError("response accounting does not match the complete input matrix")
        if job.record_id != canonical_jobs[key].record_id:
            raise ValueError("canonical response record ID does not match the response schedule")
        response = np.asarray(response_map.response)
        valid = _valid(source)
        if response.shape != valid.shape or not np.issubdtype(response.dtype, np.floating):
            raise ValueError("response map geometry or dtype is invalid")
        if not np.isfinite(response).all() or np.any(response < 0.0):
            raise ValueError("response maps must contain finite non-negative values")
        digest = hashlib.sha256(response.tobytes()).hexdigest()
        if response_map.response_sha256 != digest:
            raise ValueError("response SHA256 does not match the response map")
        maps[key] = response_map
    if set(maps) != set(expected):
        raise ValueError("response accounting does not match the complete input matrix")
    if responses.parameters_unchanged is not True:
        raise ValueError("response parameters were not preserved")
    _validate_paired_prompt_provenance(maps, config)
    return maps, _validate_targets(data, config)


def _common_record(response_map: ResponseMap) -> dict[str, Any]:
    job = response_map.job
    return {
        "cohort": job.cohort,
        "prompt_defect_type": job.prompt_defect_type,
        "record_id": job.record_id,
        "response_sha256": response_map.response_sha256,
        "sample_id": job.sample_id,
        "seed": job.seed,
        "source_defect_type": job.source_defect_type,
    }


def _normal_metrics(
    *, data: P3A0Data, maps: dict[tuple[str, str, str, int], ResponseMap], config: P3A0Config
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    calibration = calibrate_morphology(data.calibration_targets)
    if set(calibration.centers) != set(config.defect_types):
        raise ValueError("morphology calibration must cover all defect types")
    records: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {value: [] for value in config.defect_types}
    rois: dict[str, np.ndarray] = {}
    valids: dict[str, np.ndarray] = {}
    for item in sorted(data.normal_inputs, key=lambda value: value.sample_id):
        valid = _valid(item)
        valids[item.sample_id] = valid
        rois[item.sample_id] = product_roi(_rgb(item), valid, border_width=config.roi_border_width)

    for prompt_type in config.defect_types:
        lower, upper = calibration.area_envelopes[prompt_type]
        for item in sorted(data.normal_inputs, key=lambda value: value.sample_id):
            valid = valids[item.sample_id]
            roi = rois[item.sample_id]
            valid_count = int(np.count_nonzero(valid))
            roi_prior = int(np.count_nonzero(roi)) / valid_count
            for seed in config.seeds:
                response_map = maps[("normal_reference", item.sample_id, prompt_type, seed)]
                response = np.asarray(response_map.response, dtype=np.float64)
                valid_values = response[valid]
                binary = otsu_binary(response, valid, bins=config.otsu_bins)
                binary_count = int(np.count_nonzero(binary))
                area_fraction = binary_count / valid_count
                total_mass = float(valid_values.sum())
                degenerate = bool(
                    total_mass == 0.0 or float(valid_values.min()) == float(valid_values.max())
                )
                mass_inside = float(response[roi].sum()) / total_mass if total_mass != 0.0 else 0.0
                roi_precision = (
                    int(np.count_nonzero(binary & roi)) / binary_count if binary_count else 0.0
                )
                descriptor = morphology_descriptor(binary, valid)
                distances = {
                    value: morphology_distance(
                        descriptor, calibration.centers[value], calibration.pooled_mad
                    )
                    for value in config.defect_types
                }
                metric = {
                    "area_envelope_member": bool(lower <= area_fraction <= upper),
                    "degenerate": degenerate,
                    "morphology_distances": dict(sorted(distances.items())),
                    "otsu_area_fraction": float(area_fraction),
                    "response_mass_inside_roi": float(mass_inside),
                    "roi_area_prior": float(roi_prior),
                    "roi_mass_advantage": float(mass_inside - roi_prior),
                    "roi_precision": float(roi_precision),
                }
                grouped[prompt_type].append(metric)
                records.append({**_common_record(response_map), **metric})

    by_type: dict[str, Any] = {}
    localization_gate: dict[str, Any] = {}
    seed_gate: dict[str, Any] = {}
    morphology_gate: dict[str, Any] = {}
    for prompt_type in config.defect_types:
        values = grouped[prompt_type]
        pair_l1: list[float] = []
        pair_spearman: list[float] = []
        different_hashes = 0
        for item in sorted(data.normal_inputs, key=lambda value: value.sample_id):
            left = maps[("normal_reference", item.sample_id, prompt_type, config.seeds[0])]
            right = maps[("normal_reference", item.sample_id, prompt_type, config.seeds[1])]
            valid = valids[item.sample_id]
            pair_l1.append(normalized_l1(left.response, right.response, valid))
            pair_spearman.append(spearman(left.response, right.response, valid))
            different_hashes += int(left.response_sha256 != right.response_sha256)
        matched = [value["morphology_distances"][prompt_type] for value in values]
        unmatched = [
            value["morphology_distances"][other]
            for value in values
            for other in config.defect_types
            if other != prompt_type
        ]
        summary = {
            "area_envelope_fraction": float(
                np.mean([value["area_envelope_member"] for value in values])
            ),
            "different_hash_pairs": int(different_hashes),
            "median_cross_seed_l1": float(np.median(pair_l1)),
            "median_cross_seed_spearman": float(np.median(pair_spearman)),
            "median_matched_morphology_distance": float(np.median(matched)),
            "median_mass_advantage": float(
                np.median([value["roi_mass_advantage"] for value in values])
            ),
            "median_roi_precision": float(np.median([value["roi_precision"] for value in values])),
            "median_unmatched_morphology_distance": float(np.median(unmatched)),
            "seed_pair_count": len(pair_l1),
        }
        by_type[prompt_type] = summary
        localization_gate[prompt_type] = {
            "area_envelope_fraction": summary["area_envelope_fraction"],
            "median_mass_advantage": summary["median_mass_advantage"],
            "median_roi_precision": summary["median_roi_precision"],
            "passed": bool(
                summary["area_envelope_fraction"] >= config.gate.area_envelope_fraction
                and summary["median_roi_precision"] >= config.gate.roi_precision
                and summary["median_mass_advantage"] >= config.gate.roi_mass_advantage
            ),
        }
        seed_gate[prompt_type] = {
            "different_hash_pairs": summary["different_hash_pairs"],
            "median_cross_seed_l1": summary["median_cross_seed_l1"],
            "median_cross_seed_spearman": summary["median_cross_seed_spearman"],
            "passed": bool(
                summary["different_hash_pairs"] == summary["seed_pair_count"]
                and summary["median_cross_seed_l1"] >= config.gate.seed_l1
                and summary["median_cross_seed_spearman"] >= config.gate.seed_spearman
            ),
            "seed_pair_count": summary["seed_pair_count"],
        }
        morphology_gate[prompt_type] = {
            "median_matched_distance": summary["median_matched_morphology_distance"],
            "median_pooled_unmatched_distance": summary["median_unmatched_morphology_distance"],
            "passed": bool(
                summary["median_matched_morphology_distance"]
                < summary["median_unmatched_morphology_distance"]
            ),
        }

    def gate(values: dict[str, dict[str, Any]]) -> dict[str, Any]:
        passing = [value for value in config.defect_types if values[value]["passed"]]
        return {
            "by_type": dict(sorted(values.items())),
            "passed": len(passing) >= config.gate.minimum_passing_types,
            "passing_types": passing,
        }

    return (
        records,
        {"by_type": dict(sorted(by_type.items()))},
        {
            "non_degenerate_on_object": gate(localization_gate),
            "seed_behavior": gate(seed_gate),
            "type_morphology": gate(morphology_gate),
        },
    )


def _binary_diagnostics(
    score_maps: np.ndarray, masks: np.ndarray, valids: np.ndarray, bins: int
) -> dict[str, float]:
    binaries = np.stack(
        [
            otsu_binary(score, valid, bins=bins)
            for score, valid in zip(score_maps, valids, strict=True)
        ]
    )
    labels = masks.astype(bool, copy=False) & valids
    predictions = binaries & valids
    true_positive = int(np.count_nonzero(predictions & labels))
    false_positive = int(np.count_nonzero(predictions & ~labels & valids))
    false_negative = int(np.count_nonzero(~predictions & labels))
    iou_denominator = true_positive + false_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "binary_f1": float(2 * true_positive / f1_denominator) if f1_denominator else 0.0,
        "binary_iou": float(true_positive / iou_denominator) if iou_denominator else 0.0,
    }


def _continuous_metrics(
    score_maps: np.ndarray, masks: np.ndarray, max_fpr: float
) -> dict[str, float]:
    value = pixel_metrics(score_maps, masks, max_fpr=max_fpr)
    return {"au_pro": float(value.au_pro), "average_precision": float(value.average_precision)}


def _metric_bundle(
    score_maps: np.ndarray, masks: np.ndarray, valids: np.ndarray, config: P3A0Config
) -> dict[str, float]:
    return {
        **_continuous_metrics(score_maps, masks, config.max_fpr),
        **_binary_diagnostics(score_maps, masks, valids, config.otsu_bins),
    }


def _uncertainty(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p025": float(np.quantile(array, 0.025, method="linear")),
        "p975": float(np.quantile(array, 0.975, method="linear")),
        "values": [float(value) for value in array],
    }


def _auxiliary_metrics(
    *,
    data: P3A0Data,
    maps: dict[tuple[str, str, str, int], ResponseMap],
    targets: dict[str, MaskTarget],
    config: P3A0Config,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    by_seed: dict[str, Any] = {}
    inputs_by_type = {
        source_type: tuple(
            sorted(
                (
                    value
                    for value in data.auxiliary_inputs
                    if value.source_defect_type == source_type
                ),
                key=lambda value: value.sample_id,
            )
        )
        for source_type in config.defect_types
    }

    for item in sorted(data.auxiliary_inputs, key=lambda value: value.sample_id):
        target = targets[item.sample_id]
        valid = _valid(item)
        mask = _as_2d(target.mask, name="auxiliary mask").astype(bool, copy=False)
        for prompt_type in config.defect_types:
            for seed in config.seeds:
                response_map = maps[("auxiliary_real", item.sample_id, prompt_type, seed)]
                diagnostics = _binary_diagnostics(
                    np.asarray(response_map.response)[None, :, :],
                    mask[None, :, :],
                    valid[None, :, :],
                    config.otsu_bins,
                )
                records.append(
                    {
                        **_common_record(response_map),
                        **diagnostics,
                        "partition": "matched"
                        if prompt_type == item.source_defect_type
                        else "wrong",
                    }
                )

    auxiliary_gate_seeds: dict[str, Any] = {}
    metric_names = ("average_precision", "au_pro")
    for seed in config.seeds:
        per_type: dict[str, Any] = {}
        for source_type in config.defect_types:
            items = inputs_by_type[source_type]
            masks = np.stack(
                [
                    _as_2d(targets[item.sample_id].mask, name="auxiliary mask").astype(np.uint8)
                    for item in items
                ]
            )
            valids = np.stack([_valid(item) for item in items])
            prompt_metrics: dict[str, dict[str, float]] = {}
            for prompt_type in config.defect_types:
                score_maps = np.stack(
                    [
                        maps[("auxiliary_real", item.sample_id, prompt_type, seed)].response
                        for item in items
                    ]
                )
                prompt_metrics[prompt_type] = _metric_bundle(score_maps, masks, valids, config)
            matched = prompt_metrics[source_type]
            wrong = {
                prompt_type: prompt_metrics[prompt_type]
                for prompt_type in config.defect_types
                if prompt_type != source_type
            }
            wrong_strongest = {
                metric_name: max(value[metric_name] for value in wrong.values())
                for metric_name in metric_names
            }
            sobel_maps = np.stack([sobel_baseline(_rgb(item), _valid(item)) for item in items])
            sobel = _continuous_metrics(sobel_maps, masks, config.max_fpr)
            random_by_item = [
                random_baselines(
                    maps[("auxiliary_real", item.sample_id, source_type, seed)].job.record_id,
                    _valid(item),
                    config.random_baseline_count,
                )
                for item in items
            ]
            random_metrics = [
                _continuous_metrics(
                    np.stack([values[index] for values in random_by_item]), masks, config.max_fpr
                )
                for index in range(config.random_baseline_count)
            ]
            random = {
                metric_name: _uncertainty([value[metric_name] for value in random_metrics])
                for metric_name in metric_names
            }
            strongest = {
                metric_name: max(
                    wrong_strongest[metric_name], sobel[metric_name], random[metric_name]["mean"]
                )
                for metric_name in metric_names
            }
            per_type[source_type] = {
                "deltas": {
                    metric_name: float(matched[metric_name] - strongest[metric_name])
                    for metric_name in metric_names
                },
                "matched": dict(sorted(matched.items())),
                "random": dict(sorted(random.items())),
                "sobel": dict(sorted(sobel.items())),
                "strongest_baseline": dict(sorted(strongest.items())),
                "wrong_prompts": dict(sorted(wrong.items())),
                "wrong_strongest": dict(sorted(wrong_strongest.items())),
            }

        macro_matched = {
            metric_name: float(
                np.mean([per_type[value]["matched"][metric_name] for value in config.defect_types])
            )
            for metric_name in metric_names
        }
        macro_wrong = {
            metric_name: float(
                np.mean(
                    [
                        per_type[value]["wrong_strongest"][metric_name]
                        for value in config.defect_types
                    ]
                )
            )
            for metric_name in metric_names
        }
        macro_sobel = {
            metric_name: float(
                np.mean([per_type[value]["sobel"][metric_name] for value in config.defect_types])
            )
            for metric_name in metric_names
        }
        macro_random = {
            metric_name: float(
                np.mean(
                    [
                        per_type[value]["random"][metric_name]["mean"]
                        for value in config.defect_types
                    ]
                )
            )
            for metric_name in metric_names
        }
        macro_strongest = {
            metric_name: max(
                macro_wrong[metric_name], macro_sobel[metric_name], macro_random[metric_name]
            )
            for metric_name in metric_names
        }
        macro = {
            "deltas": {
                metric_name: float(macro_matched[metric_name] - macro_strongest[metric_name])
                for metric_name in metric_names
            },
            "matched": dict(sorted(macro_matched.items())),
            "random_mean": dict(sorted(macro_random.items())),
            "sobel": dict(sorted(macro_sobel.items())),
            "strongest_baseline": dict(sorted(macro_strongest.items())),
            "wrong_strongest": dict(sorted(macro_wrong.items())),
        }
        positive_types = {
            metric_name: [
                value
                for value in config.defect_types
                if per_type[value]["deltas"][metric_name] > 0.0
            ]
            for metric_name in metric_names
        }
        seed_passed = bool(
            all(
                macro["deltas"][metric_name] >= config.gate.auxiliary_margin
                for metric_name in metric_names
            )
            and all(
                len(positive_types[metric_name]) >= config.gate.minimum_passing_types
                for metric_name in metric_names
            )
        )
        seed_key = str(seed)
        by_seed[seed_key] = {
            "by_type": dict(sorted(per_type.items())),
            "macro": macro,
        }
        auxiliary_gate_seeds[seed_key] = {
            "macro_deltas": macro["deltas"],
            "passed": seed_passed,
            "positive_delta_types": dict(sorted(positive_types.items())),
        }
    gate = {
        "by_seed": dict(sorted(auxiliary_gate_seeds.items())),
        "passed": all(value["passed"] for value in auxiliary_gate_seeds.values()),
    }
    return records, {"by_seed": dict(sorted(by_seed.items()))}, gate


def evaluate_p3_a0(
    *, data: P3A0Data, responses: ResponseBatch, config: P3A0Config
) -> EvaluationResult:
    maps, targets = _validate_responses(data, responses, config)
    normal_records, normal_metrics, normal_gates = _normal_metrics(
        data=data, maps=maps, config=config
    )
    auxiliary_records, auxiliary_metrics, auxiliary_gate = _auxiliary_metrics(
        data=data, maps=maps, targets=targets, config=config
    )
    records = tuple(
        sorted(
            (*normal_records, *auxiliary_records),
            key=lambda value: (
                value["cohort"],
                value["sample_id"],
                value["prompt_defect_type"],
                value["seed"],
                value["record_id"],
            ),
        )
    )
    gate_groups = {
        "auxiliary_real_localization": auxiliary_gate,
        "non_degenerate_on_object": normal_gates["non_degenerate_on_object"],
        "seed_behavior": normal_gates["seed_behavior"],
        "type_morphology": normal_gates["type_morphology"],
    }
    metrics = {
        "accounting": {"auxiliary": 384, "normal": 64, "total": 448},
        "auxiliary": auxiliary_metrics,
        "normal": normal_metrics,
    }
    return EvaluationResult(
        records=records,
        metrics=metrics,
        gate_groups=gate_groups,
        promoted=all(value["passed"] for value in gate_groups.values()),
    )


__all__ = ["EvaluationResult", "evaluate_p3_a0"]

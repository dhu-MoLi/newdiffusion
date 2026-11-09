"""Official Hazelnut generate(x, type, seed) -> y, m_final.

Internal: a2 A once (TokenCut relative-union, q=0.60), official site-scale H once, frozen R once, compose_causal once.
Caller-facing result is only the anomaly image and m_final.
Not Module C downstream. Does not overwrite 20260907 panels.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from cadiff.v06_a.a2 import load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import LoadedIdentity, load_identity_record
from cadiff.v06_a.identities import load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, estimate_normal_structure
from cadiff.v06_atlas_diag.p2_cascade import (
    P2CascadeConfig,
    assert_frozen_b_contract,
    build_p2_hazelnut_slots,
    load_frozen_module_b,
    lookup_p2_target,
)
from cadiff.v06_b.inference import validate_realization
from cadiff.v06_b.renderer import compose_causal

_OUTPUT_PREFIX = ("v06_c", "hazelnut", "generate")
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")
_FROZEN_DDIM_STEPS = 50
_OFFICIAL_H_METHOD = "tokencut_site_scale_place"
REVIEW_COLUMNS = (
    "normal",
    "proposal",
    "m_final",
    "final_y",
)


class _InferenceContract(Protocol):
    realization_threshold: float
    ddim_steps: int


class _ProposalGenerator(Protocol):
    def generate(self, target: NormalStructure, defect_type: str, seed: int) -> Any:
        ...


class _ExactRenderer(Protocol):
    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        realization_mask: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        ...


class _ModuleB(Protocol):
    renderer: _ExactRenderer


@dataclass(frozen=True)
class GenerateResult:
    anomaly_image: np.ndarray
    anomaly_mask: np.ndarray


@dataclass(frozen=True)
class GenerateConfig:
    schema_version: str
    status: str
    category: str
    experiment: str
    paths: str
    b_experiment: str
    b_freeze: str
    rows: tuple[()] = ()


def generate(
    x: NormalStructure,
    defect_type: str,
    seed: int,
    *,
    atlas_generator: _ProposalGenerator,
    module_b: _ModuleB,
) -> GenerateResult:
    """Caller-facing generate(x, type, seed) -> y + m_final."""

    proposal, _extras, maps = _proposal_from_generator(atlas_generator, x, defect_type, seed)
    result, _raw_r, _h_meta = _compose_from_proposal(
        x,
        defect_type,
        seed,
        proposal=proposal,
        maps=maps,
        module_b=module_b,
    )
    return result


def load_generate_config(path: Path, *, repository_root: Path) -> GenerateConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("official generate config must be a mapping")
    if str(loaded.get("schema_version")) != "v06_c_hazelnut_generate":
        raise ValueError("official generate requires schema v06_c_hazelnut_generate")
    if "rows" in loaded:
        raise ValueError("official generate rows must be built from the frozen P2 schedule, not listed in yaml")
    if "include_a0v" in loaded:
        raise ValueError("official generate is a2 A only; A0-V is a P2 control, not this interface")
    if str(loaded.get("category")) != "hazelnut":
        raise ValueError("official generate is Hazelnut-only until an independent Cable B exists")
    for key in ("experiment", "paths", "b_experiment", "b_freeze"):
        relative = loaded.get(key)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"official generate {key} must be a repository-relative path")
        resolved = repository_root.joinpath(*PurePosixPath(relative).parts)
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError(f"official generate {key} must be a regular file")
    return GenerateConfig(
        schema_version="v06_c_hazelnut_generate",
        status=str(loaded["status"]),
        category="hazelnut",
        experiment=str(loaded["experiment"]),
        paths=str(loaded["paths"]),
        b_experiment=str(loaded["b_experiment"]),
        b_freeze=str(loaded["b_freeze"]),
    )


def generate_output_path(output_root: Path, output_relpath: str) -> Path:
    parts = PurePosixPath(output_relpath).parts
    if parts[:1] == ("v06_b",) or "v06_b" in parts:
        raise ValueError("refuses to write into Module B runs")
    if any(part.startswith(_FORBIDDEN_OUTPUT_PREFIXES) for part in parts):
        name = parts[-1] if parts else output_relpath
        raise ValueError(f"refuses to write f1-f4 outputs: {name}")
    if parts[:3] != _OUTPUT_PREFIX or len(parts) != 4:
        raise ValueError("official generate output must live under v06_c/hazelnut/generate/<name>")
    root = output_root.resolve(strict=True)
    destination = root.joinpath(*parts)
    resolved = destination.resolve(strict=False)
    resolved.relative_to(root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {destination}")
    return destination


def _as_p2_b_config(config: GenerateConfig) -> P2CascadeConfig:
    return P2CascadeConfig(
        schema_version="v06_a_diag_atlas_prop_p2",
        status="official_generate_b_load",
        category="hazelnut",
        include_a0v=True,
        experiment=config.experiment,
        paths=config.paths,
        b_experiment=config.b_experiment,
        b_freeze=config.b_freeze,
    )


def _copy_map(value: object, *, dtype: type, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    if copied.ndim != 2:
        raise ValueError(f"{name} must be HxW")
    return copied


def _placement_maps_from_result(result: Any) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    similarity = getattr(result, "similarity", None)
    source_gt = getattr(result, "source_gt", None)
    source_support = getattr(result, "source_support", None)
    target_support = getattr(result, "target_support", None)
    if similarity is not None:
        maps["similarity"] = _copy_map(similarity, dtype=np.float32, name="similarity")
    if source_gt is not None:
        maps["source_gt"] = _copy_map(source_gt, dtype=bool, name="source_gt")
    if source_support is not None:
        maps["source_support"] = _copy_map(source_support, dtype=bool, name="source_support")
    if target_support is not None:
        maps["target_support"] = _copy_map(target_support, dtype=bool, name="target_support")
    fewshot_gt_areas = getattr(result, "fewshot_gt_areas", None)
    if fewshot_gt_areas is not None:
        maps["fewshot_gt_areas"] = np.ascontiguousarray(fewshot_gt_areas, dtype=np.int64)
    return maps


def _frozen_renderer(module_b: _ModuleB) -> _ExactRenderer:
    renderer = getattr(module_b, "renderer", None)
    if not callable(renderer):
        raise TypeError("official generate uses frozen R only; module_b.renderer is required")
    return renderer


def _proposal_from_generator(
    generator: _ProposalGenerator,
    structure: NormalStructure,
    defect_type: str,
    seed: int,
) -> tuple[NDArray[np.bool_], dict[str, Any], dict[str, np.ndarray]]:
    result = generator.generate(structure, defect_type, seed)
    if hasattr(result, "proposal"):
        extras = {
            "source_id": getattr(result, "source_id", None),
            "used_geometry_fallback": bool(getattr(result, "used_geometry_fallback", False)),
            "collapsed": bool(getattr(result, "collapsed", False)),
            "collapse_reason": getattr(result, "collapse_reason", None),
        }
        method = getattr(result, "method", None)
        if method is not None:
            extras["method"] = method
        object_support_method = getattr(result, "object_support_method", None)
        if object_support_method is not None:
            extras["object_support_method"] = object_support_method
        proposal = np.ascontiguousarray(result.proposal, dtype=bool).copy()
        return proposal, extras, _placement_maps_from_result(result)
    if isinstance(result, np.ndarray):
        proposal = np.ascontiguousarray(result, dtype=bool)
        return proposal, {
            "source_id": None,
            "used_geometry_fallback": None,
            "collapsed": None,
            "collapse_reason": None,
        }, {}
    raise TypeError("proposal generator must return an A result with proposal or a binary mask")


def _compose_from_proposal(
    structure: NormalStructure,
    defect_type: str,
    seed: int,
    *,
    proposal: np.ndarray,
    maps: Mapping[str, np.ndarray],
    module_b: _ModuleB,
) -> tuple[GenerateResult, np.ndarray, dict[str, Any]]:
    from cadiff.v06_b.a2_component_place import instantiate_site_scale

    similarity = maps.get("similarity")
    source_gt = maps.get("source_gt")
    source_support = maps.get("source_support")
    target_support = maps.get("target_support")
    fewshot_gt_areas = maps.get("fewshot_gt_areas")
    if similarity is None or source_gt is None or source_support is None or target_support is None:
        raise ValueError(
            "official generate requires a2 similarity, source_gt, source_support, and target_support"
        )
    if fewshot_gt_areas is None:
        fewshot_areas = (int(np.asarray(source_gt, dtype=bool).sum()),)
    else:
        fewshot_areas = tuple(int(value) for value in np.asarray(fewshot_gt_areas).ravel())
    normal = np.ascontiguousarray(structure.normal.rgb.copy())
    _template, placement = instantiate_site_scale(
        source_gt,
        source_support,
        np.ascontiguousarray(target_support & structure.valid, dtype=bool),
        proposal,
        similarity,
        seed=seed,
        fewshot_areas=fewshot_areas,
    )
    m_final = np.ascontiguousarray(placement.mask, dtype=bool)
    validate_realization(m_final, placement.site)
    if (m_final & ~proposal).any():
        raise ValueError("placed shape escaped a2")
    raw_r = np.ascontiguousarray(
        _frozen_renderer(module_b)(normal, defect_type, m_final, seed)
    )
    final_y = compose_causal(normal, raw_r, m_final)
    h_meta = {
        "h_method": _OFFICIAL_H_METHOD,
        "used_learned_planner": False,
        "used_site_fallback": bool(placement.used_site_fallback),
        "used_proposal_floor": bool(placement.used_proposal_floor),
        "used_clip": bool(placement.used_clip),
        "site_pixels": int(placement.site.sum()),
        "placed_pixels": int(placement.placed_pixels),
        "scale": float(placement.scale),
        "factor": float(placement.factor),
        "angle_deg": int(placement.angle_deg),
        "component_count": int(placement.component_count),
        "site_index": int(placement.site_index),
    }
    return GenerateResult(anomaly_image=final_y, anomaly_mask=m_final), raw_r, h_meta


def _mask_image(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask.astype(np.uint8) * 255)


def _failure_rgb(shape: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    image[..., 0] = 160
    return image


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def _save_contact_sheet(
    path: Path,
    rows: list[tuple[str, dict[str, np.ndarray]]],
) -> None:
    tile_size = 128
    header_height = 24
    canvas = Image.new(
        "RGB",
        (len(REVIEW_COLUMNS) * tile_size, header_height + len(rows) * tile_size),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(REVIEW_COLUMNS):
        draw.text((column * tile_size + 4, 5), name, fill="white")
    for row_index, (label, columns) in enumerate(rows):
        top = header_height + row_index * tile_size
        for column_index, name in enumerate(REVIEW_COLUMNS):
            source = columns[name]
            tile = Image.fromarray(source).convert("RGB").resize(
                (tile_size, tile_size),
                Image.Resampling.NEAREST if source.ndim == 2 else Image.Resampling.LANCZOS,
            )
            if column_index == 0:
                ImageDraw.Draw(tile).text((4, 4), label, fill="white")
            canvas.paste(tile, (column_index * tile_size, top))
    canvas.save(path)


def run_official_generate_panel(
    config: GenerateConfig,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
    atlas_generator: _ProposalGenerator | None = None,
    module_b: _ModuleB | None = None,
    inference: _InferenceContract | None = None,
    load_target: Callable[..., LoadedIdentity] | None = None,
    estimate_structure: Callable[..., NormalStructure] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if config.category != "hazelnut":
        raise ValueError("official generate is Hazelnut-only until an independent Cable B exists")
    destination = generate_output_path(output_root, output_relpath)
    experiment_path = repository_root.joinpath(*PurePosixPath(config.experiment).parts)
    experiment = load_v06_a_experiment(experiment_path)
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    slots = build_p2_hazelnut_slots(manifest, experiment=experiment)
    if inference is None or module_b is None:
        module_b, inference = load_frozen_module_b(
            repository_root=repository_root,
            config=_as_p2_b_config(config),
            device=device,
        )
    assert_frozen_b_contract(inference)
    if atlas_generator is None:
        atlas_generator = load_a2_generator(
            experiment_path=experiment_path,
            dataset_root=dataset_root,
            device=device,
        )
    if load_target is None:
        load_target = load_identity_record
    if estimate_structure is None:
        estimate_structure = estimate_normal_structure

    destination.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    contact: list[tuple[str, dict[str, np.ndarray]]] = []
    structures: dict[str, NormalStructure] = {}
    for slot in slots:
        if slot.normal_id not in structures:
            record = lookup_p2_target(manifest, slot.normal_id)
            target = load_target(record, dataset_root=dataset_root, config=experiment)
            structures[slot.normal_id] = estimate_structure(target, config=experiment)
        structure = structures[slot.normal_id]
        proposal, extras, maps = _proposal_from_generator(
            atlas_generator,
            structure,
            slot.defect_type,
            slot.seed,
        )
        source_id = extras["source_id"] or slot.source_id
        identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        sample_root = destination / identity
        sample_root.mkdir()
        normal = np.ascontiguousarray(structure.normal.rgb.copy())
        record: dict[str, Any] = {
            "identity": identity,
            "category": "hazelnut",
            "defect_type": slot.defect_type,
            "target_id": slot.normal_id,
            "target_role": "open_normal",
            "source_id": source_id,
            "seed": slot.seed,
            "used_geometry_fallback": extras["used_geometry_fallback"],
            "collapsed": extras["collapsed"],
            "collapse_reason": extras["collapse_reason"],
            "proposal": int(proposal.sum()),
            "proposal_pixels": int(proposal.sum()),
            "m_final_pixels": None,
            "m_final_subset_of_proposal": None,
            "outside_rgb_unchanged": None,
            "failed": False,
            "failure_reason": None,
            "h_method": _OFFICIAL_H_METHOD,
            "used_learned_planner": False,
            "used_site_fallback": None,
            "used_proposal_floor": None,
            "realization_threshold": None,
            "ddim_steps": _FROZEN_DDIM_STEPS,
            "candidate_count": 1,
            "clipped_to_support": False,
            "retried": False,
            "not_module_c_complete": True,
        }
        columns = {
            "normal": normal.copy(),
            "proposal": _mask_image(proposal),
            "m_final": _failure_rgb(normal.shape),
            "raw_r": _failure_rgb(normal.shape),
            "final_y": _failure_rgb(normal.shape),
        }
        try:
            result, raw_r, h_meta = _compose_from_proposal(
                structure,
                slot.defect_type,
                slot.seed,
                proposal=proposal,
                maps=maps,
                module_b=module_b,
            )
            m_final = np.ascontiguousarray(result.anomaly_mask, dtype=bool)
            final_y = np.ascontiguousarray(result.anomaly_image)
            columns["m_final"] = _mask_image(m_final)
            columns["final_y"] = final_y
            columns["raw_r"] = raw_r
            record.update(h_meta)
            record["m_final_pixels"] = int(m_final.sum())
            record["m_final_subset_of_proposal"] = bool(not (m_final & ~proposal).any())
            record["outside_rgb_unchanged"] = bool(
                np.array_equal(final_y[~m_final], normal[~m_final])
            )
        except (ValueError, RuntimeError) as error:
            record["failed"] = True
            record["failure_reason"] = str(error)
            (sample_root / "failure.json").write_text(
                json.dumps(
                    {
                        "failed": True,
                        "failure_reason": str(error),
                        "retried": False,
                        "clipped_to_support": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        for name in (*REVIEW_COLUMNS, "raw_r"):
            _save_png(sample_root / f"{name}.png", columns[name])
        (sample_root / "record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows.append(record)
        contact.append((f"{slot.defect_type}/{slot.normal_id}/{slot.seed}", columns))
        print(
            f"{identity} src={source_id} "
            f"{'FAIL ' + record['failure_reason'] if record['failed'] else 'ok'}",
            flush=True,
        )

    with (destination / "records.jsonl").open("x", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    _save_contact_sheet(destination / "atlas_generate_hazelnut.png", contact)
    summary = {
        "schema_version": config.schema_version,
        "status": "awaiting_human_review",
        "category": "hazelnut",
        "identity_count": 32,
        "failed_rows": sum(1 for row in rows if row["failed"]),
        "floor_rows": sum(1 for row in rows if row.get("used_proposal_floor") is True),
        "fallback_rows": sum(1 for row in rows if row.get("used_site_fallback") is True),
        "h_method": _OFFICIAL_H_METHOD,
        "used_learned_planner": False,
        "realization_threshold": None,
        "ddim_steps": _FROZEN_DDIM_STEPS,
        "candidate_count": 1,
        "clipped_to_support": False,
        "retried": False,
        "look_at": "normal | proposal | m_final | final_y; raw_r is sidecar only",
        "destination": str(destination),
        "rows": rows,
        "not_module_c_complete": True,
        "did_not_overwrite_f1_f4": True,
        "did_not_overwrite_20260907": True,
        "did_not_write_module_b_run": True,
        "used_p2_open_normals": True,
        "include_a0v": False,
    }
    (destination / "summary.json").write_text(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official Hazelnut generate() panel. Not Module C downstream."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-relpath", required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_official_generate_panel(
        load_generate_config(arguments.config, repository_root=arguments.repository_root),
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        output_relpath=arguments.output_relpath,
        device=arguments.device,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Hazelnut P2 cascade: candidate A and A0-V into frozen Module B.

Not a formal Module A completion. Does not unfreeze or rewrite B.
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

from cadiff.config import load_config
from cadiff.v06_a.a0 import A0VGenerator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import LoadedIdentity, load_identity_record
from cadiff.v06_a.identities import IdentityManifest, IdentityRecord, load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, build_normal_atlas, estimate_normal_structure

from .atlas_a import AtlasAResult, load_atlas_a_generator

_OUTPUT_PREFIX = ("v06_a", "diag_atlas_prop")
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")
_FROZEN_B_CHECKPOINT = Path(
    "v06_b/hazelnut/mask-first-causal-renderer/seed-20260826/checkpoint"
)
_FROZEN_THRESHOLD = 0.35
_FROZEN_DDIM_STEPS = 50
_FROZEN_P2_NORMAL_IDS = ("000", "013", "026", "039")
_HAZELNUT_TYPES = ("crack", "cut", "hole", "print")
REVIEW_COLUMNS = (
    "normal",
    "proposal",
    "m_final",
    "raw_r",
    "final_y",
)
_FAMILIES = ("atlas", "a0v")


class _InferenceContract(Protocol):
    realization_threshold: float
    ddim_steps: int


class _ProposalGenerator(Protocol):
    def generate(self, target: NormalStructure, defect_type: str, seed: int) -> Any:
        ...


class _ModuleB(Protocol):
    def __call__(
        self,
        normal: np.ndarray,
        defect_type: str,
        proposal: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class P2CascadeConfig:
    schema_version: str
    status: str
    category: str
    include_a0v: bool
    experiment: str
    paths: str
    b_experiment: str
    b_freeze: str
    rows: tuple[()] = ()


@dataclass(frozen=True)
class P2Slot:
    category: str
    defect_type: str
    normal_id: str
    seed: int
    source_id: str
    target_role: str


def assert_frozen_b_contract(inference: _InferenceContract) -> None:
    if float(inference.realization_threshold) != _FROZEN_THRESHOLD:
        raise ValueError("refuses to change frozen B threshold 0.35")
    if int(inference.ddim_steps) != _FROZEN_DDIM_STEPS:
        raise ValueError("refuses to change frozen B 50-step DDIM")


def resolve_frozen_b_checkpoint(runs_root: Path, freeze: Mapping[str, Any]) -> Path:
    checkpoint = freeze.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("B freeze manifest must contain a checkpoint mapping")
    relative = checkpoint.get("runs_root_relpath")
    if relative != _FROZEN_B_CHECKPOINT.as_posix():
        raise ValueError("P2 cascade accepts only the frozen Hazelnut B checkpoint")
    destination = (runs_root / _FROZEN_B_CHECKPOINT).resolve(strict=True)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("frozen B checkpoint must be a regular directory")
    return destination


def lookup_p2_target(manifest: IdentityManifest, normal_id: str) -> IdentityRecord:
    matches = [record for record in manifest.p2_normals if record.item_id == normal_id]
    if len(matches) != 1:
        raise ValueError("P2 target must be the exact frozen open_normal")
    record = matches[0]
    if record.role != "open_normal":
        raise ValueError("P2 target role must be open_normal")
    return record


def build_p2_hazelnut_slots(
    manifest: IdentityManifest,
    *,
    experiment: Any,
) -> tuple[P2Slot, ...]:
    if manifest.category != "hazelnut" or experiment.category != "hazelnut":
        raise ValueError("P2 cascade is Hazelnut-only")
    if (
        manifest.defect_types != _HAZELNUT_TYPES
        or experiment.defect_types != _HAZELNUT_TYPES
    ):
        raise ValueError("P2 cascade requires the frozen Hazelnut vocabulary")
    if len(manifest.p2_identities) != 32:
        raise ValueError("P2 requires the frozen 32-identity Hazelnut schedule")
    if tuple(record.item_id for record in manifest.p2_normals) != _FROZEN_P2_NORMAL_IDS:
        raise ValueError("P2 open normals must be the frozen 000/013/026/039 set")
    if any(record.role != "open_normal" for record in manifest.p2_normals):
        raise ValueError("P2 normals must be open_normal identities")
    slots: list[P2Slot] = []
    for identity in manifest.p2_identities:
        if identity.phase != "p2":
            raise ValueError("P2 slots must come from the frozen p2 schedule")
        sources = tuple(
            sorted(
                (
                    record
                    for record in manifest.fewshot
                    if record.defect_type == identity.defect_type
                ),
                key=lambda record: record.sample_id,
            )
        )
        if not sources:
            raise ValueError(f"missing frozen fewshot sources for {identity.defect_type}")
        source = sources[identity.seed % len(sources)]
        slots.append(
            P2Slot(
                category="hazelnut",
                defect_type=identity.defect_type,
                normal_id=identity.normal_id,
                seed=identity.seed,
                source_id=source.item_id,
                target_role="open_normal",
            )
        )
    if len(slots) != 32:
        raise ValueError("P2 cascade freezes exactly 32 Hazelnut identities")
    return tuple(slots)


def load_p2_cascade_config(path: Path, *, repository_root: Path) -> P2CascadeConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("P2 cascade config must be a mapping")
    if str(loaded.get("schema_version")) != "v06_a_diag_atlas_prop_p2":
        raise ValueError("P2 cascade requires schema v06_a_diag_atlas_prop_p2")
    if "rows" in loaded:
        raise ValueError("P2 cascade rows must be built from the frozen manifest, not listed in yaml")
    if str(loaded.get("category")) != "hazelnut":
        raise ValueError("P2 cascade is Hazelnut-only")
    if loaded.get("include_a0v") is not True:
        raise ValueError("P2 cascade freezes include_a0v=true so A0-V also enters frozen B")
    for key in ("experiment", "paths", "b_experiment", "b_freeze"):
        relative = loaded.get(key)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"P2 cascade {key} must be a repository-relative path")
        resolved = repository_root.joinpath(*PurePosixPath(relative).parts)
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError(f"P2 cascade {key} must be a regular file")
    return P2CascadeConfig(
        schema_version="v06_a_diag_atlas_prop_p2",
        status=str(loaded["status"]),
        category="hazelnut",
        include_a0v=True,
        experiment=str(loaded["experiment"]),
        paths=str(loaded["paths"]),
        b_experiment=str(loaded["b_experiment"]),
        b_freeze=str(loaded["b_freeze"]),
    )


def p2_cascade_output_path(output_root: Path, output_relpath: str) -> Path:
    parts = PurePosixPath(output_relpath).parts
    if parts[:1] == ("v06_b",) or "v06_b" in parts:
        raise ValueError("refuses to write into Module B runs")
    if parts[:2] != _OUTPUT_PREFIX or len(parts) != 3:
        raise ValueError("P2 cascade output must live under v06_a/diag_atlas_prop/<name>")
    name = parts[2]
    if name.startswith(_FORBIDDEN_OUTPUT_PREFIXES):
        raise ValueError(f"refuses to write f1-f4 outputs: {name}")
    root = output_root.resolve(strict=True)
    destination = root.joinpath(*parts)
    resolved = destination.resolve(strict=False)
    resolved.relative_to(root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {destination}")
    return destination


def load_a0v_generator(
    *,
    experiment_path: Path,
    dataset_root: Path,
) -> A0VGenerator:
    experiment = load_v06_a_experiment(experiment_path)
    repository_root = experiment_path.resolve().parents[2]
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    if manifest.category != "hazelnut":
        raise ValueError("P2 A0-V loader is Hazelnut-only")
    sources = {
        defect_type: tuple(
            load_identity_record(record, dataset_root=dataset_root, config=experiment)
            for record in sorted(
                (item for item in manifest.fewshot if item.defect_type == defect_type),
                key=lambda item: item.sample_id,
            )
        )
        for defect_type in experiment.defect_types
    }
    references = tuple(
        load_identity_record(
            next(record for record in manifest.normal_fit if record.item_id == item_id),
            dataset_root=dataset_root,
            config=experiment,
        )
        for item_id in manifest.atlas_normal_ids
    )
    atlas = build_normal_atlas(
        references,
        config=experiment,
        identity_manifest=manifest,
    )
    return A0VGenerator(
        sources,
        atlas,
        config=experiment,
        identity_manifest=manifest,
    )


def load_frozen_module_b(
    *,
    repository_root: Path,
    config: P2CascadeConfig,
    device: str | None = None,
) -> tuple[_ModuleB, _InferenceContract]:
    import torch

    from cadiff.diffusion.components import load_frozen_components
    from cadiff.diffusion.ddim_inversion import encode_prompt
    from cadiff.p2_a1.config import load_p2_a1_config
    from cadiff.p2_b2.config import load_p2_b2_config
    from cadiff.provenance import sha256_file
    from cadiff.v06_b.config import V06BMaskFirstConfig, load_v06_b_config
    from cadiff.v06_b.exact_renderer import ExactMaskRenderer
    from cadiff.v06_b.inference import MaskFirstCausalRenderer
    from cadiff.v06_b.mask_first_checkpoint import load_mask_first_checkpoint
    from cadiff.v06_b.planner import PlannerNetwork, RealizationPlanner
    from cadiff.v06_b.source import load_fresh_p2_b2_renderer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch_device = torch.device(device or "cuda:0")
    paths = load_config(repository_root.joinpath(*PurePosixPath(config.paths).parts))
    freeze = yaml.safe_load(
        repository_root.joinpath(*PurePosixPath(config.b_freeze).parts).read_text(encoding="utf-8")
    )
    checkpoint = resolve_frozen_b_checkpoint(paths.runs_root, freeze)
    for filename, meta in freeze["checkpoint"]["files"].items():
        payload = checkpoint / filename
        if payload.is_symlink() or not payload.is_file():
            raise ValueError(f"frozen B file missing: {filename}")
        if sha256_file(payload) != meta["sha256"]:
            raise ValueError(f"frozen B hash mismatch: {filename}")
    experiment = load_v06_b_config(
        repository_root.joinpath(*PurePosixPath(config.b_experiment).parts)
    )
    if not isinstance(experiment, V06BMaskFirstConfig):
        raise ValueError("P2 cascade requires the frozen mask-first Module-B config")
    if experiment.ddim_steps != _FROZEN_DDIM_STEPS:
        raise ValueError("refuses to change frozen B 50-step DDIM")
    source_model = load_p2_a1_config(repository_root / experiment.source_model_config)
    source_prompts = load_p2_b2_config(repository_root / experiment.source_experiment_config)
    planner_network = PlannerNetwork(
        defect_types=experiment.defect_types,
        base_channels=experiment.planner_base_channels,
        type_embedding_dim=experiment.planner_type_embedding_dim,
    )
    components = load_frozen_components(paths.model_root, torch_device, torch.float16)
    prompts = {
        defect_type: encode_prompt(
            components,
            source_prompts.prompts[defect_type],
            torch_device,
        )
        for defect_type in experiment.defect_types
    }
    fresh = load_fresh_p2_b2_renderer(
        manifest_path=repository_root / experiment.source_renderer_manifest,
        runs_root=paths.runs_root,
        category=experiment.category,
        source_config=source_model,
        unet=components.unet,
        device=torch_device,
    )
    loaded = load_mask_first_checkpoint(
        checkpoint,
        planner=planner_network,
        adapter=fresh.adapter,
        conditioner=fresh.conditioner,
        config=experiment,
        source=fresh.source,
    )
    assert_frozen_b_contract(loaded.inference)
    fresh.model.eval()
    planner = RealizationPlanner(
        network=planner_network,
        defect_types=experiment.defect_types,
        threshold=loaded.inference.realization_threshold,
        device=torch_device,
    )
    renderer = ExactMaskRenderer(
        components=components,
        model=fresh.model,
        prompts=prompts,
        config=experiment,
        device=torch_device,
    )
    return (
        MaskFirstCausalRenderer(
            planner=planner,
            renderer=renderer,
            defect_types=experiment.defect_types,
        ),
        loaded.inference,
    )


def _proposal_from_generator(
    generator: _ProposalGenerator,
    structure: NormalStructure,
    defect_type: str,
    seed: int,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    result = generator.generate(structure, defect_type, seed)
    if isinstance(result, AtlasAResult):
        return result.proposal.copy(), {
            "source_id": result.source_id,
            "used_geometry_fallback": result.used_geometry_fallback,
            "collapsed": result.collapsed,
            "collapse_reason": result.collapse_reason,
        }
    if isinstance(result, np.ndarray):
        proposal = np.ascontiguousarray(result, dtype=bool)
        return proposal, {
            "source_id": None,
            "used_geometry_fallback": None,
            "collapsed": None,
            "collapse_reason": None,
        }
    raise TypeError("proposal generator must return AtlasAResult or a binary mask")


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


def run_p2_cascade(
    config: P2CascadeConfig,
    *,
    repository_root: Path,
    dataset_root: Path,
    output_root: Path,
    output_relpath: str,
    atlas_generator: _ProposalGenerator | None = None,
    a0v_generator: _ProposalGenerator | None = None,
    module_b: _ModuleB | None = None,
    inference: _InferenceContract | None = None,
    load_target: Callable[..., LoadedIdentity] | None = None,
    estimate_structure: Callable[..., NormalStructure] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if config.category != "hazelnut" or config.include_a0v is not True:
        raise ValueError("P2 cascade is Hazelnut-only and must include A0-V")
    destination = p2_cascade_output_path(output_root, output_relpath)
    experiment_path = repository_root.joinpath(*PurePosixPath(config.experiment).parts)
    experiment = load_v06_a_experiment(experiment_path)
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    slots = build_p2_hazelnut_slots(manifest, experiment=experiment)
    if inference is None or module_b is None:
        module_b, inference = load_frozen_module_b(
            repository_root=repository_root,
            config=config,
            device=device,
        )
    assert_frozen_b_contract(inference)
    if atlas_generator is None:
        atlas_generator = load_atlas_a_generator(
            experiment_path=experiment_path,
            dataset_root=dataset_root,
            device=device,
        )
    if a0v_generator is None:
        a0v_generator = load_a0v_generator(
            experiment_path=experiment_path,
            dataset_root=dataset_root,
        )
    if load_target is None:
        load_target = load_identity_record
    if estimate_structure is None:
        estimate_structure = estimate_normal_structure

    destination.mkdir(parents=True)
    generators: dict[str, _ProposalGenerator] = {
        "atlas": atlas_generator,
        "a0v": a0v_generator,
    }
    rows: list[dict[str, Any]] = []
    contact: dict[str, dict[str, list[tuple[str, dict[str, np.ndarray]]]]] = {
        family: {defect_type: [] for defect_type in _HAZELNUT_TYPES}
        for family in _FAMILIES
    }
    structures: dict[str, NormalStructure] = {}
    for family in _FAMILIES:
        (destination / family).mkdir()
        for slot in slots:
            if slot.normal_id not in structures:
                record = lookup_p2_target(manifest, slot.normal_id)
                target = load_target(record, dataset_root=dataset_root, config=experiment)
                structures[slot.normal_id] = estimate_structure(target, config=experiment)
            structure = structures[slot.normal_id]
            proposal, extras = _proposal_from_generator(
                generators[family],
                structure,
                slot.defect_type,
                slot.seed,
            )
            source_id = extras["source_id"] or slot.source_id
            if extras["source_id"] not in {None, slot.source_id}:
                raise ValueError("proposal source must follow seed % n")
            if (
                not proposal.any()
                or np.all(proposal[structure.valid])
                or not np.all(proposal <= structure.valid)
            ):
                raise ValueError("P2 proposal must be non-empty, non-full within V, and a subset of V")
            identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
            sample_root = destination / family / identity
            sample_root.mkdir()
            normal = np.ascontiguousarray(structure.normal.rgb.copy())
            record: dict[str, Any] = {
                "family": family,
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
                "proposal_pixels": int(proposal.sum()),
                "m_final_pixels": None,
                "m_final_subset_of_proposal": None,
                "outside_rgb_byte_exact": None,
                "compatibility_failed": False,
                "failure_reason": None,
                "realization_threshold": _FROZEN_THRESHOLD,
                "ddim_steps": _FROZEN_DDIM_STEPS,
                "candidate_count": 1,
                "clipped_to_support": False,
                "retried": False,
                "not_module_a_complete": True,
            }
            columns = {
                "normal": normal.copy(),
                "proposal": _mask_image(proposal),
                "m_final": _failure_rgb(normal.shape),
                "raw_r": _failure_rgb(normal.shape),
                "final_y": _failure_rgb(normal.shape),
            }
            try:
                from cadiff.v06_b.renderer import compose_causal

                raw_r, m_final = module_b(normal, slot.defect_type, proposal, slot.seed)
                m_final = np.ascontiguousarray(m_final, dtype=bool)
                if (m_final & ~proposal).any():
                    raise ValueError("realization mask escaped proposal")
                final_y = compose_causal(normal, raw_r, m_final)
                columns["m_final"] = _mask_image(m_final)
                columns["raw_r"] = np.ascontiguousarray(raw_r)
                columns["final_y"] = final_y
                record["m_final_pixels"] = int(m_final.sum())
                record["m_final_subset_of_proposal"] = True
                record["outside_rgb_byte_exact"] = bool(
                    np.array_equal(final_y[~m_final], normal[~m_final])
                )
            except (ValueError, RuntimeError) as error:
                record["compatibility_failed"] = True
                record["failure_reason"] = str(error)
                (sample_root / "failure.json").write_text(
                    json.dumps(
                        {
                            "compatibility_failed": True,
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
            for name in REVIEW_COLUMNS:
                _save_png(sample_root / f"{name}.png", columns[name])
            (sample_root / "record.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rows.append(record)
            contact[family][slot.defect_type].append(
                (f"{slot.normal_id}/{slot.seed}", columns)
            )
            print(
                f"{family} {identity} src={source_id} "
                f"{'FAIL ' + record['failure_reason'] if record['compatibility_failed'] else 'ok'}",
                flush=True,
            )

    with (destination / "records.jsonl").open("x", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    for family in _FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        for defect_type in _HAZELNUT_TYPES:
            _save_contact_sheet(
                destination / f"panel_{family}_{defect_type}.png",
                contact[family][defect_type],
            )
        _save_contact_sheet(
            destination / f"panel_{family}_all.png",
            [item for defect_type in _HAZELNUT_TYPES for item in contact[family][defect_type]],
        )
        (destination / f"{family}_summary.json").write_text(
            json.dumps(
                {
                    "family": family,
                    "sample_count": len(family_rows),
                    "compatibility_failures": sum(
                        1 for row in family_rows if row["compatibility_failed"]
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "schema_version": config.schema_version,
        "status": "awaiting_human_review",
        "category": "hazelnut",
        "identity_count": 32,
        "family_counts": {
            family: sum(1 for row in rows if row["family"] == family) for family in _FAMILIES
        },
        "compatibility_failures": sum(1 for row in rows if row["compatibility_failed"]),
        "realization_threshold": _FROZEN_THRESHOLD,
        "ddim_steps": _FROZEN_DDIM_STEPS,
        "candidate_count": 1,
        "clipped_to_support": False,
        "look_at": "proposal / m_final / raw_r / final_y; red tiles are recorded B failures",
        "destination": str(destination),
        "rows": rows,
        "not_module_a": True,
        "not_module_a_complete": True,
        "did_not_overwrite_f1_f4": True,
        "did_not_write_module_b_run": True,
        "used_open_report_or_test_anomalies": False,
        "used_p2_open_normals": True,
    }
    (destination / "summary.json").write_text(
        json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hazelnut P2 cascade into frozen Module B. Not formal A completion."
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
    summary = run_p2_cascade(
        load_p2_cascade_config(arguments.config, repository_root=arguments.repository_root),
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

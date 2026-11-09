"""Historical Module A: atlas label propagation f4. Official A is a2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import LoadedIdentity, load_identity_record
from cadiff.v06_a.identities import IdentityManifest, load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, estimate_normal_structure

from .correspondence import (
    CollapseLimits,
    geometry_fallback_mask,
    match_feature_maps,
    select_used_mask,
    source_match_support,
    splat_mask,
    square_dilate,
)
from .features import FrozenDinoEncoder, downsample_mask
from .inpaint import inpaint_source
from .support import rgb_support

_SOURCE_MATCH_DILATE_PX = 16
_SPLAT_NEIGHBOR_RADIUS = 1
_SQUARE_DILATE_PX = 24
_INPAINT_RADIUS = 3
_SMOOTH_RADIUS = 1
_LIMITS = CollapseLimits(
    max_cycle_fail_ratio=0.45,
    min_valid_matches=16,
    max_median_displacement_px=256.0,
    wipe_on_collapse=False,
)
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")


class _AtlasAConfig(Protocol):
    category: str
    defect_types: tuple[str, ...]


class _Encoder(Protocol):
    model_name: str

    def encode(self, rgb: NDArray[np.uint8]) -> Any:
        ...


@dataclass(frozen=True)
class AtlasAResult:
    proposal: NDArray[np.bool_]
    source_id: str
    defect_type: str
    seed: int
    used_geometry_fallback: bool
    collapsed: bool
    collapse_reason: str | None
    is_official_module_a: bool = False
    not_learned: bool = True
    not_module_a_complete: bool = False

    def __post_init__(self) -> None:
        self.proposal.setflags(write=False)


class AtlasProposalGenerator:
    """Historical A(x, defect_type, seed) -> m_proposal using frozen f4 atlas rules."""

    def __init__(
        self,
        sources_by_type: Mapping[str, Sequence[LoadedIdentity]],
        *,
        config: _AtlasAConfig,
        identity_manifest: IdentityManifest | None = None,
        encoder: _Encoder,
    ) -> None:
        if identity_manifest is not None and (
            identity_manifest.category != config.category
            or identity_manifest.defect_types != config.defect_types
        ):
            raise ValueError("atlas A manifest must agree with the experiment config")
        if set(sources_by_type) != set(config.defect_types):
            raise ValueError("atlas A sources must cover exactly the configured vocabulary")
        expected_by_type = None
        if identity_manifest is not None:
            expected_by_type = {
                defect_type: tuple(
                    sorted(
                        (
                            record
                            for record in identity_manifest.fewshot
                            if record.defect_type == defect_type
                        ),
                        key=lambda record: record.sample_id,
                    )
                )
                for defect_type in config.defect_types
            }
        normalized: dict[str, tuple[LoadedIdentity, ...]] = {}
        for defect_type, sources in sources_by_type.items():
            ordered = tuple(sorted(sources, key=lambda item: item.record.sample_id))
            if not ordered:
                raise ValueError("atlas A sources must be non-empty same-type ground-truth identities")
            if any(item.record.role != "fewshot" for item in ordered):
                raise ValueError("atlas A sources must be fewshot identities")
            if any(item.record.category != config.category for item in ordered):
                raise ValueError("atlas A source category must match the configured category")
            if any(item.record.defect_type != defect_type for item in ordered):
                raise ValueError("atlas A source type must match its source mapping key")
            if any(item.anomaly_gt is None for item in ordered):
                raise ValueError("atlas A sources require non-empty ground-truth masks")
            if expected_by_type is not None and tuple(item.record for item in ordered) != expected_by_type[defect_type]:
                raise ValueError("atlas A sources must exactly match frozen manifest fewshot records")
            normalized[defect_type] = ordered
        self._sources = normalized
        self._config = config
        self._manifest = identity_manifest
        self._encoder = encoder

    def select_source(self, defect_type: str, seed: int) -> LoadedIdentity:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        sources = self._sources.get(defect_type)
        if sources is None:
            raise ValueError(f"unknown atlas A defect type: {defect_type}")
        return sources[seed % len(sources)]

    def generate(self, target: NormalStructure, defect_type: str, seed: int) -> AtlasAResult:
        source = self.select_source(defect_type, seed)
        if source.anomaly_gt is None:
            raise ValueError("atlas A source requires a ground-truth mask")
        if target.normal.record.role not in {"normal_fit", "normal_dev", "open_normal"}:
            raise ValueError("atlas A target must be a prepared normal identity")
        if not np.array_equal(target.valid, target.normal.valid):
            raise ValueError("target structure must belong to the supplied target normal")
        inpainted = inpaint_source(
            source.rgb, source.anomaly_gt, source.valid, radius=_INPAINT_RADIUS
        )
        source_object = rgb_support(inpainted, source.valid)
        source_feat = self._encoder.encode(inpainted)
        target_feat = self._encoder.encode(target.normal.rgb)
        source_match = source_match_support(
            source.anomaly_gt,
            source_object,
            source.valid,
            dilate_px=_SOURCE_MATCH_DILATE_PX,
        )
        source_grid = downsample_mask(
            source_match,
            height=source_feat.grid_height,
            width=source_feat.grid_width,
        )
        target_grid = downsample_mask(
            target.support & target.valid,
            height=target_feat.grid_height,
            width=target_feat.grid_width,
        )
        matched = match_feature_maps(
            source_feat.tokens,
            target_feat.tokens,
            source_grid,
            target_grid,
            limits=_LIMITS,
            smooth_radius=_SMOOTH_RADIUS,
        )
        propagated = splat_mask(
            source.anomaly_gt,
            matched,
            neighbor_radius=_SPLAT_NEIGHBOR_RADIUS,
        )
        geometry = geometry_fallback_mask(source.anomaly_gt, source_object, target.support)
        used, used_fallback = select_used_mask(
            propagated,
            geometry,
            collapsed=matched.collapsed,
            use_kept_splat=True,
        )
        proposal = square_dilate(used, _SQUARE_DILATE_PX) & target.valid
        if (
            not proposal.any()
            or np.all(proposal[target.valid])
            or not np.all(proposal <= target.valid)
        ):
            raise ValueError("atlas A proposal must be non-empty, non-full within V, and a subset of V")
        return AtlasAResult(
            proposal=proposal.copy(),
            source_id=source.record.item_id,
            defect_type=defect_type,
            seed=seed,
            used_geometry_fallback=used_fallback,
            collapsed=matched.collapsed,
            collapse_reason=matched.collapse_reason,
            is_official_module_a=False,
            not_learned=True,
            not_module_a_complete=False,
        )


def load_atlas_a_generator(
    *,
    experiment_path: Path,
    dataset_root: Path,
    device: str | None = None,
    encoder: _Encoder | None = None,
) -> AtlasProposalGenerator:
    experiment = load_v06_a_experiment(experiment_path)
    repository_root = experiment_path.resolve().parents[2]
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
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
    if encoder is None:
        encoder = FrozenDinoEncoder(device=device)
    return AtlasProposalGenerator(
        sources,
        config=experiment,
        identity_manifest=manifest,
        encoder=encoder,
    )


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(_FORBIDDEN_OUTPUT_PREFIXES) for part in output.parts):
        raise ValueError(f"refuses to write f1-f4 outputs: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Candidate atlas A inference. Not formal completion.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--defect-type", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    destination = _contained_output(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    experiment = load_v06_a_experiment(arguments.experiment)
    repository_root = arguments.experiment.resolve().parents[2]
    manifest_path = repository_root.joinpath(*PurePosixPath(experiment.identity_manifest).parts)
    manifest = load_v06_a_identity_manifest(manifest_path, experiment=experiment)
    generator = load_atlas_a_generator(
        experiment_path=arguments.experiment,
        dataset_root=arguments.dataset_root,
        device=arguments.device,
    )
    targets = [
        record
        for record in manifest.normal_dev
        if record.item_id == arguments.target_id and record.defect_type == "good"
    ]
    if len(targets) != 1:
        raise ValueError("atlas A CLI target must be the exact frozen normal_dev record")
    target = load_identity_record(targets[0], dataset_root=arguments.dataset_root, config=experiment)
    structure = estimate_normal_structure(target, config=experiment)
    result = generator.generate(structure, arguments.defect_type, arguments.seed)
    Image.fromarray((result.proposal.astype(np.uint8) * 255), mode="L").save(destination)
    sidecar = destination.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "source_id": result.source_id,
                "target_id": arguments.target_id,
                "defect_type": result.defect_type,
                "seed": result.seed,
                "used_geometry_fallback": result.used_geometry_fallback,
                "collapsed": result.collapsed,
                "collapse_reason": result.collapse_reason,
                "proposal_pixels": int(result.proposal.sum()),
                "is_official_module_a": False,
                "not_learned": True,
                "not_module_a_complete": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "source_id": result.source_id,
                "is_official_module_a": False,
                "not_learned": True,
                "not_module_a_complete": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

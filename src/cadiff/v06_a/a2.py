"""Official Module A: a2 material prototype. Not learned Mask diffusion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import LoadedIdentity, load_identity_record
from cadiff.v06_a.identities import IdentityManifest, load_v06_a_identity_manifest
from cadiff.v06_a.structure import NormalStructure, estimate_normal_structure
from cadiff.v06_a.tokencut import load_official_dinov2_encoder, tokencut_object_mask
from cadiff.v06_atlas_diag.features import downsample_mask
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source

_SUPPORT_Q = 0.60
_HOLE_Q_FALLBACK = 0.20
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")


class _A2Config(Protocol):
    category: str
    defect_types: tuple[str, ...]


class _Encoder(Protocol):
    model_name: str

    def encode(self, rgb: NDArray[np.uint8]) -> Any:
        ...


@dataclass(frozen=True)
class A2Result:
    proposal: NDArray[np.bool_]
    source_id: str
    defect_type: str
    seed: int
    used_geometry_fallback: bool = False
    collapsed: bool = False
    collapse_reason: str | None = None
    is_official_module_a: bool = True
    not_learned: bool = True
    not_module_a_complete: bool = False
    method: str = "idea_a2"
    tau: float | None = None
    similarity: NDArray[np.float32] | None = None
    source_gt: NDArray[np.bool_] | None = None
    source_support: NDArray[np.bool_] | None = None
    target_support: NDArray[np.bool_] | None = None
    object_support_method: str = "tokencut"
    fewshot_gt_areas: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.proposal.setflags(write=False)
        if self.similarity is not None:
            self.similarity.setflags(write=False)
        if self.source_gt is not None:
            self.source_gt.setflags(write=False)
        if self.source_support is not None:
            self.source_support.setflags(write=False)
        if self.target_support is not None:
            self.target_support.setflags(write=False)


class A2ProposalGenerator:
    """Official A(x, defect_type, seed) -> m_proposal using frozen a2 rules."""

    def __init__(
        self,
        sources_by_type: Mapping[str, Sequence[LoadedIdentity]],
        *,
        config: _A2Config,
        identity_manifest: IdentityManifest | None = None,
        encoder: _Encoder,
        fill_backend: Any | None = None,
    ) -> None:
        """fill_backend(rgb, hole, valid) -> rgb inpaints the few-shot defect
        hole before the material prototype is built. None keeps the official
        nearest-neighbour fill."""
        if identity_manifest is not None and (
            identity_manifest.category != config.category
            or identity_manifest.defect_types != config.defect_types
        ):
            raise ValueError("a2 manifest must agree with the experiment config")
        if set(sources_by_type) != set(config.defect_types):
            raise ValueError("a2 sources must cover exactly the configured vocabulary")
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
                raise ValueError("a2 sources must be non-empty same-type ground-truth identities")
            if any(item.record.role != "fewshot" for item in ordered):
                raise ValueError("a2 sources must be fewshot identities")
            if any(item.record.category != config.category for item in ordered):
                raise ValueError("a2 source category must match the configured category")
            if any(item.record.defect_type != defect_type for item in ordered):
                raise ValueError("a2 source type must match its source mapping key")
            if any(item.anomaly_gt is None for item in ordered):
                raise ValueError("a2 sources require non-empty ground-truth masks")
            if expected_by_type is not None and tuple(item.record for item in ordered) != expected_by_type[defect_type]:
                raise ValueError("a2 sources must exactly match frozen manifest fewshot records")
            normalized[defect_type] = ordered
        self._sources = normalized
        self._config = config
        self._manifest = identity_manifest
        self._encoder = encoder
        self._fill = fill_backend if fill_backend is not None else neighbor_fill_source

    def select_source(self, defect_type: str, seed: int) -> LoadedIdentity:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        sources = self._sources.get(defect_type)
        if sources is None:
            raise ValueError(f"unknown a2 defect type: {defect_type}")
        return sources[seed % len(sources)]

    def generate(self, target: NormalStructure, defect_type: str, seed: int) -> A2Result:
        source = self.select_source(defect_type, seed)
        if source.anomaly_gt is None:
            raise ValueError("a2 source requires a ground-truth mask")
        if target.normal.record.role not in {"normal_fit", "normal_dev", "open_normal"}:
            raise ValueError("a2 target must be a prepared normal identity")
        if not np.array_equal(target.valid, target.normal.valid):
            raise ValueError("target structure must belong to the supplied target normal")
        inpainted = self._fill(source.rgb, source.anomaly_gt, source.valid)
        source_feat = self._encoder.encode(inpainted)
        target_feat = self._encoder.encode(target.normal.rgb)
        target_support = tokencut_object_mask(target_feat.tokens, target.valid)
        source_support = tokencut_object_mask(source_feat.tokens, source.valid)
        if not target_support.any():
            raise ValueError("a2 TokenCut target object is empty")
        if not source_support.any():
            raise ValueError("a2 TokenCut source object is empty")
        clipped_target = replace(target, support=target_support)
        proposal, tau, similarity = _material_proposal_maps(
            source.anomaly_gt,
            source.valid,
            clipped_target,
            source_feat=source_feat,
            target_feat=target_feat,
        )
        if (
            not proposal.any()
            or np.all(proposal[target.valid])
            or not np.all(proposal <= target.valid)
        ):
            raise ValueError("a2 proposal must be non-empty, non-full within V, and a subset of V")
        if (proposal & ~target_support).any():
            raise ValueError("a2 proposal escaped TokenCut object")
        return A2Result(
            proposal=proposal.copy(),
            source_id=source.record.item_id,
            defect_type=defect_type,
            seed=seed,
            used_geometry_fallback=False,
            collapsed=False,
            collapse_reason=None,
            is_official_module_a=True,
            not_learned=True,
            not_module_a_complete=False,
            method="idea_a2",
            tau=tau,
            similarity=np.ascontiguousarray(similarity, dtype=np.float32),
            source_gt=np.ascontiguousarray(source.anomaly_gt, dtype=bool),
            source_support=np.ascontiguousarray(source_support, dtype=bool),
            target_support=np.ascontiguousarray(target_support, dtype=bool),
            object_support_method="tokencut",
            fewshot_gt_areas=tuple(
                int(item.anomaly_gt.sum())
                for item in self._sources[defect_type]
                if item.anomaly_gt is not None
            ),
        )


def load_a2_generator(
    *,
    experiment_path: Path,
    dataset_root: Path,
    device: str | None = None,
    encoder: _Encoder | None = None,
    fill_backend: Any | None = None,
) -> A2ProposalGenerator:
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
        encoder = load_official_dinov2_encoder(device=device)
    return A2ProposalGenerator(
        sources,
        config=experiment,
        identity_manifest=manifest,
        encoder=encoder,
        fill_backend=fill_backend,
    )


def _l2_rows(features: NDArray[np.floating]) -> NDArray[np.float32]:
    vectors = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _upsample_map(grid: NDArray[np.float32], size: int = 512) -> NDArray[np.float32]:
    image = Image.fromarray(grid, mode="F")
    return np.asarray(
        image.resize((size, size), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _material_proposal_maps(
    source_gt: NDArray[np.bool_],
    source_valid: NDArray[np.bool_],
    target: NormalStructure,
    *,
    source_feat: Any,
    target_feat: Any,
) -> tuple[NDArray[np.bool_], float, NDArray[np.float32]]:
    hole = source_gt & source_valid
    hole_grid = downsample_mask(
        hole,
        height=source_feat.grid_height,
        width=source_feat.grid_width,
    )
    target_valid_grid = downsample_mask(
        target.valid,
        height=target_feat.grid_height,
        width=target_feat.grid_width,
    )
    source_desc = _l2_rows(source_feat.tokens)
    target_desc = _l2_rows(target_feat.tokens)
    hole_tokens = source_desc[hole_grid]
    if hole_tokens.shape[0] == 0:
        raise ValueError("a2 source hole has no encoder tokens")
    prototype = hole_tokens.mean(axis=0)
    prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-12)
    hole_self = hole_tokens @ prototype
    grid_sim = target_desc @ prototype
    grid_sim = np.where(target_valid_grid, grid_sim, np.float32(-1.0))
    support_grid = downsample_mask(
        target.support & target.valid,
        height=target_feat.grid_height,
        width=target_feat.grid_width,
    )
    support_sims = grid_sim[support_grid]
    if support_sims.size == 0:
        tau = float(np.quantile(hole_self, _HOLE_Q_FALLBACK))
    else:
        tau = float(np.quantile(support_sims, _SUPPORT_Q))
    similarity = _upsample_map(grid_sim.astype(np.float32))
    object_mask = target.support & target.valid
    proposal = (similarity >= tau) & target.valid & object_mask
    return proposal, tau, np.ascontiguousarray(similarity, dtype=np.float32)


def _material_proposal(
    source_gt: NDArray[np.bool_],
    source_valid: NDArray[np.bool_],
    target: NormalStructure,
    *,
    source_feat: Any,
    target_feat: Any,
) -> tuple[NDArray[np.bool_], float]:
    proposal, tau, _similarity = _material_proposal_maps(
        source_gt,
        source_valid,
        target,
        source_feat=source_feat,
        target_feat=target_feat,
    )
    return proposal, tau


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(_FORBIDDEN_OUTPUT_PREFIXES) for part in output.parts):
        raise ValueError(f"refuses to write f1-f4 outputs: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official a2 Module A inference.")
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
    generator = load_a2_generator(
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
        raise ValueError("a2 CLI target must be the exact frozen normal_dev record")
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
                "method": result.method,
                "tau": result.tau,
                "is_official_module_a": True,
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
                "method": result.method,
                "is_official_module_a": True,
                "not_learned": True,
                "not_module_a_complete": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

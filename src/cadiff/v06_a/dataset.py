"""Authoritative plain-target data path for Module A RC1 training."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import V06AExperimentConfig
from .data import LoadedIdentity, load_identity_record
from .identities import IdentityManifest, IdentityRecord
from .model import require_supported_variant
from .structure import NormalAtlas, NormalStructure, build_normal_atlas, estimate_normal_structure
from .targets import (
    ProposalTarget,
    construct_plain_target,
    downsample_all_4x4,
    nearest_upsample_4x4,
)
from .training import A1DiffusionBatch

RecordLoader = Callable[[IdentityRecord], LoadedIdentity]


@dataclass(frozen=True)
class PlainTargetRequest:
    """One frozen ordinary-transfer request; the seed chooses one few-shot source."""

    normal_record: IdentityRecord
    defect_type: str
    transfer_seed: int


@dataclass(frozen=True)
class PlainTargetExample:
    request: PlainTargetRequest
    source: LoadedIdentity
    normal: LoadedIdentity
    structure: NormalStructure
    target: ProposalTarget
    type_id: int


def _validate_authority(config: V06AExperimentConfig, identity_manifest: IdentityManifest) -> None:
    if not isinstance(config, V06AExperimentConfig):
        raise TypeError("config must be a V06AExperimentConfig")
    if not isinstance(identity_manifest, IdentityManifest):
        raise TypeError("identity_manifest must be an IdentityManifest")
    if not identity_manifest.is_authoritative():
        raise ValueError("plain targets require a loader-authoritative identity manifest")
    if (
        config.category != identity_manifest.category
        or config.defect_types != identity_manifest.defect_types
    ):
        raise ValueError("config and identity manifest category/vocabulary must agree")
    if config.image_size != 512 or config.proposal_size != 128:
        raise ValueError("Module A data grids are frozen at image=512 and proposal=128")
    if config.atlas_reference_count != 16:
        raise ValueError("Module A normal atlas requires exactly 16 references")


def _normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("normal RGB must be uint8 [512,512,3]")
    value = rgb.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def _validate_plain_binding(example: PlainTargetExample) -> None:
    target = example.target
    normal = example.normal
    structure = example.structure
    expected_valid = downsample_all_4x4(normal.valid)
    if (
        target.valid_128.shape != (128, 128)
        or target.valid_128.dtype != bool
        or not np.array_equal(target.valid_128, expected_valid)
    ):
        raise ValueError("plain target valid_128 is not exactly bound to target normal V")
    if (
        target.proposal_128.shape != (128, 128)
        or target.proposal_128.dtype != bool
        or not target.proposal_128.any()
        or np.all(target.proposal_128[target.valid_128])
        or np.any(target.proposal_128 & ~target.valid_128)
    ):
        raise ValueError("plain proposal_128 must be non-empty, non-full, and subset of V_128")
    expected_512 = nearest_upsample_4x4(target.proposal_128)
    if (
        target.proposal_512.shape != (512, 512)
        or target.proposal_512.dtype != bool
        or not np.array_equal(target.proposal_512, expected_512)
        or np.any(target.proposal_512 & ~normal.valid)
    ):
        raise ValueError("plain proposal nearest 128-to-512 binding is invalid")
    expected_z0 = np.where(target.proposal_128, 1, -1).astype(np.int8)
    expected_z0[~target.valid_128] = -1
    if (
        target.z0.shape != (128, 128)
        or target.z0.dtype != np.int8
        or not np.array_equal(target.z0, expected_z0)
    ):
        raise ValueError("plain target z0 is not bound to proposal_128 and V_128")
    if (
        structure.normal.record != normal.record
        or not np.array_equal(structure.valid, normal.valid)
        or target.target_frame != structure.frame
        or not np.array_equal(target.target_support, structure.support)
        or not np.array_equal(target.target_sdf_128, structure.sdf_128)
        or not np.array_equal(target.target_u_x, structure.u_x)
        or not np.array_equal(target.target_u_y, structure.u_y)
    ):
        raise ValueError("plain target structure is not bound to its target normal")
    if (
        target.transferred_gt_512.shape != (512, 512)
        or target.transferred_gt_512.dtype != bool
        or not target.transferred_gt_512.any()
        or np.any(target.transferred_gt_512 & ~normal.valid)
    ):
        raise ValueError("plain transferred ground truth is invalid on the target V")


class PlainTargetDataset:
    """Lazy normal/structure cache over the exact loader-authoritative A-fit schedule."""

    def __init__(
        self,
        *,
        config: V06AExperimentConfig,
        identity_manifest: IdentityManifest,
        record_loader: RecordLoader,
    ) -> None:
        _validate_authority(config, identity_manifest)
        if not callable(record_loader):
            raise TypeError("record_loader must be callable")
        if any(
            record.role != "normal_fit"
            or record.category != config.category
            or record.defect_type != "good"
            for record in identity_manifest.normal_fit
        ):
            raise ValueError("training normals must be exactly the frozen normal_fit identities")
        grouped = {
            defect_type: tuple(
                record for record in identity_manifest.fewshot if record.defect_type == defect_type
            )
            for defect_type in identity_manifest.defect_types
        }
        flattened = tuple(
            record
            for defect_type in identity_manifest.defect_types
            for record in grouped[defect_type]
        )
        if (
            flattened != identity_manifest.fewshot
            or any(
                record.role != "fewshot" or record.category != config.category
                for record in flattened
            )
            or any(
                tuple(record.item_id for record in grouped[defect_type])
                != identity_manifest.fewshot_ids_by_type[defect_type]
                for defect_type in identity_manifest.defect_types
            )
        ):
            raise ValueError("training sources must be the exact frozen fewshot universe")
        expected_source_count = 22 if config.category == "hazelnut" else 14
        if len(flattened) != expected_source_count:
            raise ValueError("training source count is not the frozen category count")
        atlas_records = tuple(
            next(
                (record for record in identity_manifest.normal_fit if record.item_id == item_id),
                None,
            )
            for item_id in identity_manifest.atlas_normal_ids
        )
        if len(atlas_records) != 16 or any(record is None for record in atlas_records):
            raise ValueError("atlas identities must be exactly 16 frozen normal_fit records")

        self.config = config
        self.identity_manifest = identity_manifest
        self._record_loader = record_loader
        self._normal_fit = {record.sample_id: record for record in identity_manifest.normal_fit}
        self._sources = grouped
        self._atlas_records = tuple(atlas_records)
        self._loaded_cache: dict[str, LoadedIdentity] = {}
        self._structure_cache: dict[str, NormalStructure] = {}
        self._atlas: NormalAtlas | None = None

    def sources(self, defect_type: str) -> tuple[IdentityRecord, ...]:
        """Return the immutable frozen source tuple for one approved type."""

        try:
            return self._sources[defect_type]
        except (KeyError, TypeError) as error:
            raise ValueError("defect_type is outside the frozen category vocabulary") from error

    def _load_exact(self, record: IdentityRecord) -> LoadedIdentity:
        cached = self._loaded_cache.get(record.sample_id)
        if cached is not None:
            if cached.record != record:
                raise ValueError("loaded identity cache collision")
            return cached
        loaded = self._record_loader(record)
        if not isinstance(loaded, LoadedIdentity) or loaded.record != record:
            raise ValueError("record loader identity mismatch")
        self._loaded_cache[record.sample_id] = loaded
        return loaded

    def _normal_structure(self, normal: LoadedIdentity) -> NormalStructure:
        cached = self._structure_cache.get(normal.record.sample_id)
        if cached is not None:
            return cached
        structure = estimate_normal_structure(normal, config=self.config)
        if (
            not isinstance(structure, NormalStructure)
            or structure.normal.record != normal.record
            or not np.array_equal(structure.valid, normal.valid)
        ):
            raise ValueError("normal structure estimator returned a mismatched identity")
        self._structure_cache[normal.record.sample_id] = structure
        return structure

    def _normal_atlas(self) -> NormalAtlas:
        if self._atlas is None:
            references = tuple(self._load_exact(record) for record in self._atlas_records)
            self._atlas = build_normal_atlas(
                references,
                config=self.config,
                identity_manifest=self.identity_manifest,
            )
        return self._atlas

    def materialize(self, request: PlainTargetRequest) -> PlainTargetExample:
        """Construct one ordinary centered target exactly once; errors propagate."""

        if not isinstance(request, PlainTargetRequest):
            raise TypeError("request must be a PlainTargetRequest")
        if (
            isinstance(request.transfer_seed, bool)
            or not isinstance(request.transfer_seed, int)
            or not 0 <= request.transfer_seed < 2**63
        ):
            raise ValueError("transfer_seed must be a non-negative signed 64-bit integer")
        expected_normal = self._normal_fit.get(request.normal_record.sample_id)
        if expected_normal is None or expected_normal != request.normal_record:
            raise ValueError("training target must be an exact frozen normal_fit record")
        sources = self.sources(request.defect_type)
        source_rng = random.Random(request.transfer_seed)
        source_record = sources[source_rng.randrange(len(sources))]
        source = self._load_exact(source_record)
        normal = self._load_exact(expected_normal)
        structure = self._normal_structure(normal)
        atlas = self._normal_atlas()
        target = construct_plain_target(
            source,
            atlas,
            normal,
            structure,
            self.config,
            self.identity_manifest,
        )
        if not isinstance(target, ProposalTarget):
            raise TypeError("construct_plain_target must return a ProposalTarget")
        example = PlainTargetExample(
            request=request,
            source=source,
            normal=normal,
            structure=structure,
            target=target,
            type_id=self.identity_manifest.defect_types.index(request.defect_type),
        )
        _validate_plain_binding(example)
        return example

    def build_batch(
        self,
        requests: Sequence[PlainTargetRequest],
        *,
        variant: str,
        device: str | torch.device = "cpu",
    ) -> A1DiffusionBatch:
        """Materialize once per request and tensorize in the frozen channel semantics."""

        variant = require_supported_variant(variant)
        requests = tuple(requests)
        if not requests:
            raise ValueError("a training batch must contain at least one request")
        examples = tuple(self.materialize(request) for request in requests)
        torch_device = torch.device(device)

        def floating(values: Sequence[np.ndarray]) -> torch.Tensor:
            array = np.stack(values).astype(np.float32, copy=False)
            return torch.from_numpy(np.ascontiguousarray(array[:, None])).to(torch_device)

        z0 = floating([example.target.z0 for example in examples])
        valid_array = np.stack([example.target.valid_128 for example in examples])
        valid = torch.from_numpy(np.ascontiguousarray(valid_array[:, None])).to(torch_device)
        type_ids = torch.tensor(
            [example.type_id for example in examples], dtype=torch.long, device=torch_device
        )
        normal_rgb = None
        if variant == "a1_sv_plain":
            rgb = np.stack([_normalise_rgb(example.normal.rgb) for example in examples])
            normal_rgb = torch.from_numpy(np.ascontiguousarray(rgb)).to(torch_device)
        return A1DiffusionBatch(
            z0=z0,
            valid=valid,
            sdf=floating([example.target.target_sdf_128 for example in examples]),
            u_x=floating([example.target.target_u_x for example in examples]),
            u_y=floating([example.target.target_u_y for example in examples]),
            type_ids=type_ids,
            normal_rgb=normal_rgb,
        )


def build_plain_target_dataset(
    *,
    config: V06AExperimentConfig,
    identity_manifest: IdentityManifest,
    dataset_root: Path,
) -> PlainTargetDataset:
    """Bind the authoritative data path to the on-disk MVTec root without loading eagerly."""

    if not isinstance(dataset_root, Path):
        raise TypeError("dataset_root must be a pathlib.Path")
    return PlainTargetDataset(
        config=config,
        identity_manifest=identity_manifest,
        record_loader=lambda record: load_identity_record(
            record, dataset_root=dataset_root, config=config
        ),
    )

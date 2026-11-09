"""The single-transfer A0-V baseline generator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from .data import LoadedIdentity
from .identities import IdentityManifest
from .structure import NormalAtlas, NormalStructure
from .targets import construct_plain_target


class _A0Config(Protocol):
    category: str
    defect_types: tuple[str, ...]
    image_size: int
    proposal_size: int
    square_dilate_radius: int


class A0VGenerator:
    """Select exactly one same-type source and perform exactly one plain transfer."""

    def __init__(
        self,
        sources_by_type: Mapping[str, Sequence[LoadedIdentity]],
        source_atlas: NormalAtlas,
        *,
        config: _A0Config,
        identity_manifest: IdentityManifest,
    ) -> None:
        if not isinstance(source_atlas, NormalAtlas):
            raise ValueError("source atlas must be a validated NormalAtlas")
        if source_atlas.category != config.category:
            raise ValueError("atlas category must match the configured category")
        if (
            identity_manifest.category != config.category
            or identity_manifest.defect_types != config.defect_types
        ):
            raise ValueError("A0 manifest must agree with the experiment config")
        if not source_atlas.matches_manifest(identity_manifest):
            raise ValueError("A0 atlas provenance must match the supplied identity manifest")
        if not sources_by_type:
            raise ValueError("A0 requires at least one same-type source")
        if set(sources_by_type) != set(config.defect_types):
            raise ValueError("A0 sources must cover exactly the configured vocabulary")
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
            if not defect_type or not ordered:
                raise ValueError("A0 sources must be non-empty same-type ground-truth identities")
            if defect_type not in config.defect_types:
                raise ValueError("A0 source type must be in the configured vocabulary")
            if any(item.record.role != "fewshot" for item in ordered):
                raise ValueError("A0 sources must be fewshot identities")
            if any(item.record.category != config.category for item in ordered):
                raise ValueError("A0 source category must match the configured category")
            if any(item.record.defect_type not in config.defect_types for item in ordered):
                raise ValueError("A0 source type must be in the configured vocabulary")
            if any(item.record.defect_type != defect_type for item in ordered):
                raise ValueError("A0 source type must match its source mapping key")
            if any(item.anomaly_gt is None for item in ordered):
                raise ValueError("A0 sources require non-empty ground-truth masks")
            if tuple(item.record for item in ordered) != expected_by_type[defect_type]:
                raise ValueError("A0 sources must exactly match frozen manifest fewshot records")
            normalized[defect_type] = ordered
        self._sources = normalized
        self._source_atlas = source_atlas
        self._config = config
        self._manifest = identity_manifest

    def generate(self, target: NormalStructure, defect_type: str, seed: int) -> np.ndarray:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        sources = self._sources.get(defect_type)
        if sources is None:
            raise ValueError(f"unknown A0 defect type: {defect_type}")
        source = sources[seed % len(sources)]
        outcome = construct_plain_target(
            source,
            self._source_atlas,
            target.normal,
            target,
            self._config,
            self._manifest,
        )
        proposal = outcome.proposal_512
        if (
            not proposal.any()
            or np.all(proposal[target.valid])
            or not np.all(proposal <= target.valid)
        ):
            raise ValueError("A0 proposal must be non-empty, non-full within V, and a subset of V")
        return proposal.copy()

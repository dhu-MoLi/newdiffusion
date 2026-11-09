"""Diagnostic Claim-0 generate boundary; no TokenCut, training or default model.

A proposal provider is injected explicitly. Engineering fixtures are marked in
results and never represented as a learned chewinggum Mod-A checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cadiff.v06_b.claim0 import (
    Claim0ModuleB,
    validate_normal,
    validate_proposal,
    validate_request_type_seed,
)
from cadiff.v06_b.renderer import compose_causal


class ProposalProvider(Protocol):
    source_kind: str

    def generate(self, *, normal_rgb: np.ndarray, valid: np.ndarray,
                 defect_type: str, seed: int) -> np.ndarray: ...


@dataclass(frozen=True, eq=False)
class Claim0Result:
    anomaly_image: np.ndarray
    anomaly_mask: np.ndarray
    diagnostic_only: bool
    proposal_source: str
    frozen_bundle_verified: bool
    learned_provenance_verified: bool


def generate(normal_rgb: np.ndarray, defect_type: str, seed: int, *,
             valid: np.ndarray, proposal_provider: ProposalProvider,
             module_b: Claim0ModuleB) -> Claim0Result:
    """Call one provider and one B; propagate failures without repair or retry.

    The 512 RGB/valid geometry is explicit. The public boundary does not accept
    a user mask or anomaly GT. Fake B objects are permitted for CPU unit tests;
    their results are explicitly marked frozen_bundle_verified=False.
    proposal_source is a declaration, not checkpoint validation; Claim-0 never
    marks learned_provenance_verified=True.
    """
    validate_normal(normal_rgb, valid)
    validate_request_type_seed(defect_type, seed)
    source_kind = getattr(proposal_provider, 'source_kind', None)
    if source_kind not in ('engineering_fixture', 'learned_mod_a'):
        raise ValueError('proposal provider must declare its provenance')
    if not callable(getattr(proposal_provider, 'generate', None)):
        raise ValueError('proposal provider must implement generate')
    if not isinstance(module_b, Claim0ModuleB):
        raise ValueError('module_b must be an explicit Claim0ModuleB')

    normal_snapshot = np.array(normal_rgb, copy=True, order='C')
    valid_snapshot = np.array(valid, copy=True, order='C')
    provider_rgb, provider_valid = normal_snapshot.copy(), valid_snapshot.copy()
    proposal = proposal_provider.generate(
        normal_rgb=provider_rgb, valid=provider_valid,
        defect_type=defect_type, seed=seed,
    )
    if not np.array_equal(provider_rgb, normal_snapshot):
        raise ValueError('proposal provider mutated normal RGB')
    if not np.array_equal(provider_valid, valid_snapshot):
        raise ValueError('proposal provider mutated valid geometry')
    proposal_snapshot = validate_proposal(proposal, valid_snapshot)
    b_rgb, b_valid, b_proposal = (
        normal_snapshot.copy(), valid_snapshot.copy(), proposal_snapshot.copy()
    )
    raw_rgb, m_final = module_b(b_rgb, defect_type, b_proposal, b_valid, seed)
    if not (np.array_equal(b_rgb, normal_snapshot)
            and np.array_equal(b_valid, valid_snapshot)
            and np.array_equal(b_proposal, proposal_snapshot)):
        raise ValueError('Module B mutated its inputs')
    final_snapshot = validate_proposal(m_final, valid_snapshot)
    if not np.array_equal(final_snapshot, proposal_snapshot):
        raise ValueError('identity planner changed its pre-render realization mask')
    final_y = compose_causal(normal_snapshot, raw_rgb, final_snapshot)
    if not np.array_equal(final_y[~final_snapshot], normal_snapshot[~final_snapshot]):
        raise ValueError('RGB outside realization mask changed')
    return Claim0Result(
        anomaly_image=np.ascontiguousarray(final_y),
        anomaly_mask=final_snapshot.copy(),
        diagnostic_only=True,
        proposal_source=source_kind,
        frozen_bundle_verified=bool(module_b.verified),
        learned_provenance_verified=False,
    )


__all__ = ['ProposalProvider', 'Claim0Result', 'generate']

"""Official Module A: a2 material prototype. Historical atlas f4 lives in v06_atlas_diag.atlas_a."""

from cadiff.v06_a.a2 import (
    A2ProposalGenerator,
    A2Result,
    load_a2_generator,
)

AtlasAResult = A2Result
AtlasProposalGenerator = A2ProposalGenerator
load_atlas_a_generator = load_a2_generator

__all__ = [
    "A2ProposalGenerator",
    "A2Result",
    "AtlasAResult",
    "AtlasProposalGenerator",
    "load_a2_generator",
    "load_atlas_a_generator",
]

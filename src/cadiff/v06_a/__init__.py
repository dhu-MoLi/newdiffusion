"""Official Module A: a2 material prototype, plus retained RC1 schedules."""

from .config import V06AExperimentConfig, load_v06_a_experiment
from .identities import IdentityManifest, load_v06_a_identity_manifest

__all__ = [
    "A2ProposalGenerator",
    "A2Result",
    "AtlasAResult",
    "AtlasProposalGenerator",
    "IdentityManifest",
    "V06AExperimentConfig",
    "load_a2_generator",
    "load_atlas_a_generator",
    "load_v06_a_experiment",
    "load_v06_a_identity_manifest",
]


def __getattr__(name: str):
    if name in {
        "A2ProposalGenerator",
        "A2Result",
        "AtlasAResult",
        "AtlasProposalGenerator",
        "load_a2_generator",
        "load_atlas_a_generator",
    }:
        from .atlas import (
            A2ProposalGenerator,
            A2Result,
            AtlasAResult,
            AtlasProposalGenerator,
            load_a2_generator,
            load_atlas_a_generator,
        )

        exports = {
            "A2ProposalGenerator": A2ProposalGenerator,
            "A2Result": A2Result,
            "AtlasAResult": AtlasAResult,
            "AtlasProposalGenerator": AtlasProposalGenerator,
            "load_a2_generator": load_a2_generator,
            "load_atlas_a_generator": load_atlas_a_generator,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

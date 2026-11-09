"""Atlas label-propagation diagnostics. Official A lives at cadiff.v06_a."""

from .atlas_a import AtlasProposalGenerator, load_atlas_a_generator
from .runner import load_atlas_diag_config, run_atlas_diag

__all__ = [
    "AtlasProposalGenerator",
    "load_atlas_a_generator",
    "load_atlas_diag_config",
    "run_atlas_diag",
]

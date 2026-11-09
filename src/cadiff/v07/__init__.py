"""v07: experiment scaffolding for the baseline upgrade.

The v06 package stays frozen so the baseline remains reproducible. Everything
new lives here and reuses v06 helpers by composition.
"""

from __future__ import annotations

__all__ = [
    "backbones",
    "generate",
    "model",
    "objective",
    "render",
    "runmeta",
    "shape_prior",
    "train_r",
    "visibility",
]

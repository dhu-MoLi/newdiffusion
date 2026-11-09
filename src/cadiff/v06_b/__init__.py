"""Public mask-first v0.6 Module B inference API."""

from .inference import MaskFirstCausalRenderer
from .renderer import compose_causal

__all__ = ["MaskFirstCausalRenderer", "compose_causal"]

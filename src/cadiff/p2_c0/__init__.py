"""P2-C0 downstream-value pilot."""

from cadiff.p2_c0.config import P2C0Config, load_p2_c0_config
from cadiff.p2_c0.contracts import GenerationJob, TrainingMask, build_generation_schedule
from cadiff.p2_c0.metrics import BinaryMetrics, PixelMetrics, binary_metrics, pixel_metrics

__all__ = [
    "BinaryMetrics",
    "GenerationJob",
    "P2C0Config",
    "PixelMetrics",
    "TrainingMask",
    "binary_metrics",
    "build_generation_schedule",
    "load_p2_c0_config",
    "pixel_metrics",
]

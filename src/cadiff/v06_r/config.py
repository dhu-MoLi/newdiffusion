"""Explicit loss-only ablation configuration; no changes to existing P2 configs."""
from dataclasses import dataclass, asdict
from math import isfinite
from pathlib import Path
import json

@dataclass(frozen=True)
class RExperimentConfig:
    arm: str
    minimum_snr: float
    rgb_weight: float
    edge_weight: float
    save_every: int = 300
    schema_version: str = "v06_r_loss_ablation_v1"

    def __post_init__(self):
        if self.schema_version != "v06_r_loss_ablation_v1":
            raise ValueError("unknown R experiment schema")
        if self.arm not in ("latent_control", "rgb", "rgb_edge"):
            raise ValueError("unknown experiment arm")
        for name in ("minimum_snr", "rgb_weight", "edge_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite non-negative")
        if self.minimum_snr <= 0:
            raise ValueError("minimum_snr must be positive")
        if type(self.save_every) is not int or self.save_every < 1:
            raise ValueError("save_every must be a positive integer")
        if self.arm == "latent_control" and (self.rgb_weight or self.edge_weight):
            raise ValueError("control must have zero appearance weights")
        if self.arm != "latent_control" and self.rgb_weight <= 0:
            raise ValueError("appearance arms require positive rgb_weight")
        if self.arm == "rgb" and self.edge_weight:
            raise ValueError("rgb arm must have zero edge_weight")
        if self.arm == "rgb_edge" and self.edge_weight <= 0:
            raise ValueError("rgb_edge requires positive edge_weight")

    def to_dict(self):
        return asdict(self)

def load_r_experiment_config(path: Path) -> RExperimentConfig:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("R experiment config must be an object")
    return RExperimentConfig(**raw)

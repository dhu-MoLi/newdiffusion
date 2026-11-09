from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SCHEMA = "p2_c0_hazelnut_downstream_v1"
_TYPES = ("crack", "cut", "hole", "print")
_RENDERER_MANIFEST = "configs/checkpoints/p2_b2_hazelnut_renderer.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True)
class P2C0Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    renderer_manifest: str
    synthetic_per_type: int
    generation_seed: int
    image_size: int
    downstream_image_size: int
    training_steps_per_arm: int
    batch_size: int
    learning_rate: float
    focal_gamma: float
    max_fpr: float
    promotion_pixel_ap_delta: float
    promotion_au_pro_delta: float
    maximum_image_auroc_drop: float
    evaluation_mode: str

    @property
    def synthetic_total(self) -> int:
        return len(self.defect_types) * self.synthetic_per_type


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a mapping")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _exact_float(value: object, expected: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if result != expected:
        raise ValueError(f"{name} must be exactly {expected}")
    return result


def load_p2_c0_config(path: Path) -> P2C0Config:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}") from error
    data = _mapping(raw)
    keys = {
        "schema_version",
        "category",
        "defect_types",
        "renderer_manifest",
        "synthetic_per_type",
        "generation_seed",
        "image_size",
        "downstream_image_size",
        "training_steps_per_arm",
        "batch_size",
        "learning_rate",
        "focal_gamma",
        "max_fpr",
        "promotion_pixel_ap_delta",
        "promotion_au_pro_delta",
        "maximum_image_auroc_drop",
        "evaluation_mode",
    }
    if set(data) != keys:
        raise ValueError(f"configuration keys must be exactly {sorted(keys)}")
    if data["schema_version"] != _SCHEMA:
        raise ValueError("schema_version differs from the frozen pilot")
    if data["category"] != "hazelnut":
        raise ValueError("category must be hazelnut")
    defect_types = tuple(data["defect_types"])
    if defect_types != _TYPES:
        raise ValueError(f"defect_types must be exactly {_TYPES}")
    if data["renderer_manifest"] != _RENDERER_MANIFEST:
        raise ValueError("renderer_manifest differs from the frozen renderer")
    synthetic_per_type = _positive_int(data["synthetic_per_type"], "synthetic_per_type")
    if synthetic_per_type != 64:
        raise ValueError("synthetic_per_type must be exactly 64")
    generation_seed = _positive_int(data["generation_seed"], "generation_seed")
    if generation_seed != 20260824:
        raise ValueError("generation_seed must be exactly 20260824")
    image_size = _positive_int(data["image_size"], "image_size")
    downstream_image_size = _positive_int(
        data["downstream_image_size"], "downstream_image_size"
    )
    if (image_size, downstream_image_size) != (512, 256):
        raise ValueError("image sizes must be frozen at 512 and 256")
    training_steps = _positive_int(
        data["training_steps_per_arm"], "training_steps_per_arm"
    )
    if training_steps != 600:
        raise ValueError("training_steps_per_arm must be exactly 600")
    batch_size = _positive_int(data["batch_size"], "batch_size")
    if batch_size != 8:
        raise ValueError("batch_size must be exactly 8")
    learning_rate = _exact_float(data["learning_rate"], 0.0001, "learning_rate")
    focal_gamma = _exact_float(data["focal_gamma"], 2.0, "focal_gamma")
    max_fpr = _exact_float(data["max_fpr"], 0.3, "max_fpr")
    ap_delta = _exact_float(
        data["promotion_pixel_ap_delta"], 0.01, "promotion_pixel_ap_delta"
    )
    pro_delta = _exact_float(
        data["promotion_au_pro_delta"], 0.01, "promotion_au_pro_delta"
    )
    auroc_drop = _exact_float(
        data["maximum_image_auroc_drop"], 0.02, "maximum_image_auroc_drop"
    )
    if data["evaluation_mode"] != "test_aware_pilot":
        raise ValueError("evaluation_mode must be test_aware_pilot")
    return P2C0Config(
        schema_version=_SCHEMA,
        category="hazelnut",
        defect_types=defect_types,
        renderer_manifest=_RENDERER_MANIFEST,
        synthetic_per_type=synthetic_per_type,
        generation_seed=generation_seed,
        image_size=image_size,
        downstream_image_size=downstream_image_size,
        training_steps_per_arm=training_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        focal_gamma=focal_gamma,
        max_fpr=max_fpr,
        promotion_pixel_ap_delta=ap_delta,
        promotion_au_pro_delta=pro_delta,
        maximum_image_auroc_drop=auroc_drop,
        evaluation_mode="test_aware_pilot",
    )

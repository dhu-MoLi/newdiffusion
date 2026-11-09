"""Category type-prompt maps for Module B. Cable is independent of frozen P2-B2 YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

CABLE_DEFECT_TYPES = (
    "bent_wire",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "poke_insulation",
)


@dataclass(frozen=True)
class V06BTypePrompts:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    prompts: Mapping[str, str]
    type_rank: int


def load_v06_b_type_prompts(path: Path) -> V06BTypePrompts:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("type-prompt config must be a mapping")
    expected = {
        "schema_version",
        "category",
        "defect_types",
        "type_rank",
        "prompts",
    }
    if set(loaded) != expected:
        raise ValueError("type-prompt config has an invalid exact schema")
    if str(loaded.get("schema_version")) != "v06_b_type_prompts_v1":
        raise ValueError("type-prompt config requires schema v06_b_type_prompts_v1")
    if str(loaded.get("category")) != "cable":
        raise ValueError("this type-prompt loader is Cable-only")
    defect_types = loaded.get("defect_types")
    if not isinstance(defect_types, list) or tuple(defect_types) != CABLE_DEFECT_TYPES:
        raise ValueError("Cable type prompts must use the frozen four-type vocabulary")
    if int(loaded.get("type_rank")) != 8:
        raise ValueError("Cable type rank must remain 8 to load the P2-B2 adapter")
    prompts = loaded.get("prompts")
    if not isinstance(prompts, Mapping) or tuple(prompts) != CABLE_DEFECT_TYPES:
        raise ValueError("prompts must contain the exact ordered Cable vocabulary")
    values = {str(key): str(value) for key, value in prompts.items()}
    if any(not text.strip() for text in values.values()):
        raise ValueError("Cable prompts must be non-empty")
    if any("hazelnut" in text.lower() for text in values.values()):
        raise ValueError("Cable prompts must not mention hazelnut")
    return V06BTypePrompts(
        schema_version="v06_b_type_prompts_v1",
        category="cable",
        defect_types=CABLE_DEFECT_TYPES,
        prompts=MappingProxyType(values),
        type_rank=8,
    )


__all__ = ["CABLE_DEFECT_TYPES", "V06BTypePrompts", "load_v06_b_type_prompts"]

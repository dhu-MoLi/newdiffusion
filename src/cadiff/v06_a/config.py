"""Strict Module-A RC1 contracts with explicit, user-approved run tunables."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml
from yaml.constructor import ConstructorError

_HAZELNUT_TYPES = ("crack", "cut", "hole", "print")
_CABLE_TYPES = (
    "bent_wire",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "poke_insulation",
)
_EXPECTED_KEYS = {
    "schema_version",
    "category",
    "defect_types",
    "identity_manifest",
    "output_relpath",
    "image_size",
    "proposal_size",
    "square_dilate_radius",
    "a_variants",
    "training_seed",
    "panel_seeds",
    "p0_pairs_per_category",
    "p1_normal_count",
    "base_channels",
    "type_embedding_dim",
    "timestep_embedding_dim",
    "normal_feature_channels",
    "diffusion_steps",
    "sampling_steps",
    "training_steps",
    "batch_size",
    "learning_rate",
    "gradient_clip",
    "support_border_width",
    "support_background_margin",
    "atlas_reference_count",
    "atlas_consensus",
    "teacher_outer_radius",
    "teacher_blind_gaps",
    "teacher_candidate_stride",
    "teacher_temperature",
    "teacher_relation_weight",
    "teacher_min_context_fraction",
}


@dataclass(frozen=True)
class V06AExperimentConfig:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    identity_manifest: str
    output_relpath: str
    image_size: int
    proposal_size: int
    square_dilate_radius: int
    a_variants: tuple[str, ...]
    training_seed: int
    panel_seeds: tuple[int, ...]
    p0_pairs_per_category: int
    p1_normal_count: int
    base_channels: int
    type_embedding_dim: int
    timestep_embedding_dim: int
    normal_feature_channels: int
    diffusion_steps: int
    sampling_steps: int
    training_steps: int
    batch_size: int
    learning_rate: float
    gradient_clip: float
    support_border_width: int
    support_background_margin: float
    atlas_reference_count: int
    atlas_consensus: float
    teacher_outer_radius: int
    teacher_blind_gaps: tuple[int, ...]
    teacher_candidate_stride: int
    teacher_temperature: float
    teacher_relation_weight: float
    teacher_min_context_fraction: float


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key ({key!r})",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _load_mapping(path: Path) -> Mapping[object, object]:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid v0.6-A YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("v0.6-A configuration must be a YAML mapping")
    return loaded


def _exact_keys(mapping: Mapping[object, object], expected: set[str]) -> None:
    if set(mapping) != expected:
        raise ValueError(
            "v0.6-A schema mismatch: "
            f"missing={sorted(expected - set(mapping))}, "
            f"extra={sorted(repr(key) for key in set(mapping) - expected)}"
        )


def _relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != value:
        raise ValueError(f"{name} must be a normalized relative path")
    return value


def _integer(value: object, *, name: str, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{name} must equal frozen value {expected}")
    return value


def _float(value: object, *, name: str, expected: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) != expected:
        raise ValueError(f"{name} must equal frozen value {expected}")
    return expected


def _integer_tuple(value: object, *, name: str, expected: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, list) or any(isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must equal frozen values {expected}")
    observed = tuple(value)
    if observed != expected or not all(isinstance(item, int) for item in observed):
        raise ValueError(f"{name} must equal frozen values {expected}")
    return expected


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _training_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError("training_seed must be a non-negative signed 64-bit integer")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def load_v06_a_experiment(path: Path) -> V06AExperimentConfig:
    """Load one RC1 contract without accepting legacy or unknown fields."""

    loaded = _load_mapping(path)
    _exact_keys(loaded, _EXPECTED_KEYS)
    category = loaded["category"]
    if not isinstance(category, str) or category not in {"hazelnut", "cable"}:
        raise ValueError("category must be hazelnut or cable")
    expected_types = _HAZELNUT_TYPES if category == "hazelnut" else _CABLE_TYPES
    if (
        not isinstance(loaded["defect_types"], list)
        or tuple(loaded["defect_types"]) != expected_types
    ):
        raise ValueError("defect_types must equal the frozen category vocabulary")
    if loaded["schema_version"] != "v06_a_experiment_rc1":
        raise ValueError("schema_version must equal v06_a_experiment_rc1")
    if not isinstance(loaded["a_variants"], list) or tuple(loaded["a_variants"]) != (
        "a0_v",
        "a1_gv",
        "a1_sv_plain",
    ):
        raise ValueError("a_variants must equal the frozen RC1 comparisons")

    diffusion_steps = _positive_integer(loaded["diffusion_steps"], name="diffusion_steps")
    sampling_steps = _positive_integer(loaded["sampling_steps"], name="sampling_steps")
    if sampling_steps > diffusion_steps:
        raise ValueError("sampling_steps must not exceed diffusion_steps")

    return V06AExperimentConfig(
        schema_version="v06_a_experiment_rc1",
        category=category,
        defect_types=expected_types,
        identity_manifest=_relative_path(loaded["identity_manifest"], name="identity_manifest"),
        output_relpath=_relative_path(loaded["output_relpath"], name="output_relpath"),
        image_size=_integer(loaded["image_size"], name="image_size", expected=512),
        proposal_size=_integer(loaded["proposal_size"], name="proposal_size", expected=128),
        square_dilate_radius=_integer(
            loaded["square_dilate_radius"], name="square_dilate_radius", expected=24
        ),
        a_variants=("a0_v", "a1_gv", "a1_sv_plain"),
        training_seed=_training_seed(loaded["training_seed"]),
        panel_seeds=_integer_tuple(
            loaded["panel_seeds"], name="panel_seeds", expected=(2026082801, 2026082802)
        ),
        p0_pairs_per_category=_integer(
            loaded["p0_pairs_per_category"], name="p0_pairs_per_category", expected=8
        ),
        p1_normal_count=_integer(loaded["p1_normal_count"], name="p1_normal_count", expected=4),
        base_channels=_positive_integer(loaded["base_channels"], name="base_channels"),
        type_embedding_dim=_positive_integer(
            loaded["type_embedding_dim"], name="type_embedding_dim"
        ),
        timestep_embedding_dim=_positive_integer(
            loaded["timestep_embedding_dim"], name="timestep_embedding_dim"
        ),
        normal_feature_channels=_positive_integer(
            loaded["normal_feature_channels"], name="normal_feature_channels"
        ),
        diffusion_steps=diffusion_steps,
        sampling_steps=sampling_steps,
        training_steps=_positive_integer(loaded["training_steps"], name="training_steps"),
        batch_size=_positive_integer(loaded["batch_size"], name="batch_size"),
        learning_rate=_positive_float(loaded["learning_rate"], name="learning_rate"),
        gradient_clip=_positive_float(loaded["gradient_clip"], name="gradient_clip"),
        support_border_width=_integer(
            loaded["support_border_width"], name="support_border_width", expected=16
        ),
        support_background_margin=_float(
            loaded["support_background_margin"], name="support_background_margin", expected=0.03
        ),
        atlas_reference_count=_integer(
            loaded["atlas_reference_count"], name="atlas_reference_count", expected=16
        ),
        atlas_consensus=_float(loaded["atlas_consensus"], name="atlas_consensus", expected=0.5),
        teacher_outer_radius=_integer(
            loaded["teacher_outer_radius"], name="teacher_outer_radius", expected=72
        ),
        teacher_blind_gaps=_integer_tuple(
            loaded["teacher_blind_gaps"], name="teacher_blind_gaps", expected=(32, 48)
        ),
        teacher_candidate_stride=_integer(
            loaded["teacher_candidate_stride"], name="teacher_candidate_stride", expected=16
        ),
        teacher_temperature=_float(
            loaded["teacher_temperature"], name="teacher_temperature", expected=0.25
        ),
        teacher_relation_weight=_float(
            loaded["teacher_relation_weight"], name="teacher_relation_weight", expected=2.0
        ),
        teacher_min_context_fraction=_float(
            loaded["teacher_min_context_fraction"],
            name="teacher_min_context_fraction",
            expected=0.75,
        ),
    )

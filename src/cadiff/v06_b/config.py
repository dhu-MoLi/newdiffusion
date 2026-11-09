from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import yaml
from yaml.constructor import ConstructorError

DEFECT_TYPES = ("crack", "cut", "hole", "print")
CABLE_DEFECT_TYPES = (
    "bent_wire",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "poke_insulation",
)
CABLE_SCHEME1_MODEL_CONFIG = "configs/experiments/p2_a1_hazelnut_factorized.yaml"
CABLE_SCHEME2_MODEL_CONFIG = "configs/experiments/p2_a1_cable_factorized.yaml"
CABLE_SCHEME2_RENDERER_MANIFEST = "configs/checkpoints/p2_b2_cable_renderer.yaml"
CABLE_SCHEME2_OUTPUT = "v06_b/cable/mask-first-causal-renderer/seed-20260909-scheme2"
CABLE_SCHEME2_CONTINUE_OUTPUT = (
    "v06_b/cable/mask-first-causal-renderer/seed-20260910-scheme2-r-continue"
)
CABLE_SCHEME2_A_TO_H_OUTPUT = (
    "v06_b/cable/mask-first-causal-renderer/seed-20260911-scheme2-a-to-h"
)
CABLE_SCHEME2_A2_TO_H_OUTPUT = (
    "v06_b/cable/mask-first-causal-renderer/seed-20260912-scheme2-a2-to-h"
)


def is_cable_scheme2(config: object) -> bool:
    source_model = getattr(config, "source_model_config", None)
    return getattr(config, "category", None) == "cable" and source_model == CABLE_SCHEME2_MODEL_CONFIG


def is_cable_a_to_h(config: object) -> bool:
    return (
        is_cable_scheme2(config)
        and getattr(config, "output_relpath", None) == CABLE_SCHEME2_A_TO_H_OUTPUT
    )


def is_cable_a2_to_h(config: object) -> bool:
    return (
        is_cable_scheme2(config)
        and getattr(config, "output_relpath", None) == CABLE_SCHEME2_A2_TO_H_OUTPUT
    )


V1_LOSS_NAMES = ("diffusion", "visible", "ring", "mask", "consistency")
V2_LOSS_NAMES = (
    "epsilon_gt",
    "x0_gt",
    "x0_ring",
    "epsilon_outside",
    "rgb_gt",
    "sobel_gt",
    "type_margin",
    "visible",
    "ring",
    "mask",
    "consistency",
)
LEARNING_RATE_NAMES = ("shared_renderer", "type_condition", "realization_head")
GRADIENT_CLIP_NAMES = ("renderer", "head")
MASK_FIRST_RENDERER_LEARNING_RATE_NAMES = ("shared_renderer", "type_condition")
MASK_FIRST_RENDERER_LOSS_NAMES = (
    "epsilon_gt",
    "x0_gt",
    "rgb_gt",
    "edge_gt",
    "type_margin",
)
MASK_FIRST_THRESHOLD_CANDIDATES = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)


@dataclass(frozen=True)
class V06BConfig:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    source_renderer_manifest: str
    source_experiment_config: str
    source_model_config: str
    prompts: Mapping[str, str]
    parent_experiment_config: str | None
    parent_checkpoint_relpath: str | None
    parent_checkpoint_step: int
    image_size: int
    proposal_radius: int
    support_corner_width: int
    rollout_steps: int
    backprop_steps: int
    checkpoint_tracked_steps: bool
    feature_channels: int
    head_hidden_channels: int
    realization_threshold: float
    visibility_margin: float
    appearance_min_snr: float
    ranking_target_ratio: float
    amp_initial_scale: float
    loss_weights: Mapping[str, float]
    training_steps: int
    learning_rates: Mapping[str, float]
    gradient_clips: Mapping[str, float]
    seed: int
    checkpoint_every: int
    panel_normal_count: int
    panel_seeds: tuple[int, ...]
    blind_panel_seed: int
    output_relpath: str

    @property
    def final_step(self) -> int:
        return self.parent_checkpoint_step + self.training_steps

    @property
    def learning_rate(self) -> float:
        """Legacy scalar used only by the frozen v1 experiment."""

        return self.learning_rates["shared_renderer"]

    @property
    def gradient_clip(self) -> float:
        """Legacy scalar used only by the frozen v1 experiment."""

        return self.gradient_clips["renderer"]


@dataclass(frozen=True)
class V06BMaskFirstConfig:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    source_renderer_manifest: str
    source_experiment_config: str
    source_model_config: str
    image_size: int
    proposal_radius: int
    support_corner_width: int
    planner_base_channels: int
    planner_type_embedding_dim: int
    planner_seed_noise_channels: int
    planner_training_steps: int
    planner_learning_rate: float
    planner_gradient_clip: float
    planner_dice_weight: float
    planner_calibration_normal_count: int
    planner_threshold_candidates: tuple[float, ...]
    renderer_training_steps: int
    renderer_learning_rates: Mapping[str, float]
    renderer_gradient_clip: float
    renderer_loss_weights: Mapping[str, float]
    appearance_min_snr: float
    ranking_target_ratio: float
    ddim_steps: int
    seed: int
    panel_normal_count: int
    panel_seeds: tuple[int, ...]
    blind_panel_seed: int
    output_relpath: str


V06BExperimentConfig = V06BConfig | V06BMaskFirstConfig


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key ({key!r})",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _mapping(path: Path) -> Mapping[object, object]:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid v0.6-B YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("v0.6-B configuration must be a YAML mapping")
    return loaded


def _exact_keys(loaded: Mapping[object, object], expected: set[str]) -> None:
    if set(loaded) != expected:
        missing = sorted(expected - set(loaded))
        extra = sorted(repr(value) for value in set(loaded) - expected)
        raise ValueError(f"v0.6-B schema mismatch: missing={missing}, extra={extra}")


def _relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ValueError(f"{name} must be a normalized relative path")
    return value


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _mask_first_config(loaded: Mapping[object, object]) -> V06BMaskFirstConfig:
    expected_keys = {
        "schema_version",
        "category",
        "defect_types",
        "source_renderer_manifest",
        "source_experiment_config",
        "source_model_config",
        "image_size",
        "proposal_radius",
        "support_corner_width",
        "planner_base_channels",
        "planner_type_embedding_dim",
        "planner_seed_noise_channels",
        "planner_training_steps",
        "planner_learning_rate",
        "planner_gradient_clip",
        "planner_dice_weight",
        "planner_calibration_normal_count",
        "planner_threshold_candidates",
        "renderer_training_steps",
        "renderer_learning_rates",
        "renderer_gradient_clip",
        "renderer_loss_weights",
        "appearance_min_snr",
        "ranking_target_ratio",
        "ddim_steps",
        "seed",
        "panel_normal_count",
        "panel_seeds",
        "blind_panel_seed",
        "output_relpath",
    }
    _exact_keys(loaded, expected_keys)
    category = str(loaded["category"])
    defect_types = loaded["defect_types"]
    if category == "hazelnut":
        if not isinstance(defect_types, list) or tuple(defect_types) != DEFECT_TYPES:
            raise ValueError("defect_types must equal the Hazelnut Module-B vocabulary")
    elif category == "cable":
        if not isinstance(defect_types, list) or tuple(defect_types) != CABLE_DEFECT_TYPES:
            raise ValueError("defect_types must equal the Cable Module-B vocabulary")
    else:
        raise ValueError("mask-first category must be hazelnut or cable")
    raw_threshold_candidates = loaded["planner_threshold_candidates"]
    if not isinstance(raw_threshold_candidates, list):
        raise ValueError("planner_threshold_candidates must be a list")
    threshold_candidates = tuple(
        _positive_float(value, name="planner_threshold_candidates")
        for value in raw_threshold_candidates
    )
    raw_renderer_learning_rates = loaded["renderer_learning_rates"]
    if not isinstance(raw_renderer_learning_rates, Mapping) or set(
        raw_renderer_learning_rates
    ) != set(MASK_FIRST_RENDERER_LEARNING_RATE_NAMES):
        raise ValueError("renderer_learning_rates must contain the exact optimizer groups")
    renderer_learning_rates = MappingProxyType(
        {
            name: _positive_float(
                raw_renderer_learning_rates[name],
                name=f"renderer_learning_rates.{name}",
            )
            for name in MASK_FIRST_RENDERER_LEARNING_RATE_NAMES
        }
    )
    raw_renderer_loss_weights = loaded["renderer_loss_weights"]
    if not isinstance(raw_renderer_loss_weights, Mapping) or set(raw_renderer_loss_weights) != set(
        MASK_FIRST_RENDERER_LOSS_NAMES
    ):
        raise ValueError("renderer_loss_weights must contain the exact Module-B loss names")
    renderer_loss_weights = MappingProxyType(
        {
            name: _positive_float(
                raw_renderer_loss_weights[name],
                name=f"renderer_loss_weights.{name}",
            )
            for name in MASK_FIRST_RENDERER_LOSS_NAMES
        }
    )
    panel_seeds = loaded["panel_seeds"]
    if (
        not isinstance(panel_seeds, list)
        or len(panel_seeds) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in panel_seeds
        )
        or len(set(panel_seeds)) != len(panel_seeds)
    ):
        raise ValueError("panel_seeds must contain two unique non-negative integers")
    config = V06BMaskFirstConfig(
        schema_version=str(loaded["schema_version"]),
        category=str(loaded["category"]),
        defect_types=tuple(defect_types),
        source_renderer_manifest=_relative_path(
            loaded["source_renderer_manifest"], name="source_renderer_manifest"
        ),
        source_experiment_config=_relative_path(
            loaded["source_experiment_config"], name="source_experiment_config"
        ),
        source_model_config=_relative_path(
            loaded["source_model_config"], name="source_model_config"
        ),
        image_size=_integer(loaded["image_size"], name="image_size"),
        proposal_radius=_integer(loaded["proposal_radius"], name="proposal_radius"),
        support_corner_width=_integer(loaded["support_corner_width"], name="support_corner_width"),
        planner_base_channels=_integer(
            loaded["planner_base_channels"], name="planner_base_channels"
        ),
        planner_type_embedding_dim=_integer(
            loaded["planner_type_embedding_dim"], name="planner_type_embedding_dim"
        ),
        planner_seed_noise_channels=_integer(
            loaded["planner_seed_noise_channels"],
            name="planner_seed_noise_channels",
            minimum=0,
        ),
        planner_training_steps=_integer(
            loaded["planner_training_steps"], name="planner_training_steps"
        ),
        planner_learning_rate=_positive_float(
            loaded["planner_learning_rate"], name="planner_learning_rate"
        ),
        planner_gradient_clip=_positive_float(
            loaded["planner_gradient_clip"], name="planner_gradient_clip"
        ),
        planner_dice_weight=_positive_float(
            loaded["planner_dice_weight"], name="planner_dice_weight"
        ),
        planner_calibration_normal_count=_integer(
            loaded["planner_calibration_normal_count"],
            name="planner_calibration_normal_count",
        ),
        planner_threshold_candidates=threshold_candidates,
        renderer_training_steps=_integer(
            loaded["renderer_training_steps"], name="renderer_training_steps"
        ),
        renderer_learning_rates=renderer_learning_rates,
        renderer_gradient_clip=_positive_float(
            loaded["renderer_gradient_clip"], name="renderer_gradient_clip"
        ),
        renderer_loss_weights=renderer_loss_weights,
        appearance_min_snr=_positive_float(loaded["appearance_min_snr"], name="appearance_min_snr"),
        ranking_target_ratio=_positive_float(
            loaded["ranking_target_ratio"], name="ranking_target_ratio"
        ),
        ddim_steps=_integer(loaded["ddim_steps"], name="ddim_steps"),
        seed=_integer(loaded["seed"], name="seed", minimum=0),
        panel_normal_count=_integer(loaded["panel_normal_count"], name="panel_normal_count"),
        panel_seeds=tuple(panel_seeds),
        blind_panel_seed=_integer(loaded["blind_panel_seed"], name="blind_panel_seed", minimum=0),
        output_relpath=_relative_path(loaded["output_relpath"], name="output_relpath"),
    )
    if config.category == "hazelnut":
        fixed = {
            "schema_version": "v06_b_mask_first_v1",
            "category": "hazelnut",
            "defect_types": DEFECT_TYPES,
            "source_renderer_manifest": "configs/checkpoints/p2_b2_hazelnut_renderer.yaml",
            "source_experiment_config": (
                "configs/experiments/p2_b2_hazelnut_type_prompt_training.yaml"
            ),
            "source_model_config": "configs/experiments/p2_a1_hazelnut_factorized.yaml",
            "image_size": 512,
            "planner_seed_noise_channels": 0,
            "seed": 20260826,
            "panel_normal_count": 4,
            "panel_seeds": (2026082601, 2026082602),
            "blind_panel_seed": 2026082603,
            "output_relpath": "v06_b/hazelnut/mask-first-causal-renderer/seed-20260826",
        }
    else:
        output_parts = Path(config.output_relpath).parts
        if (
            output_parts[:3] != ("v06_b", "cable", "mask-first-causal-renderer")
            or len(output_parts) != 4
        ):
            raise ValueError(
                "Cable output must be a distinct run under v06_b/cable/mask-first-causal-renderer"
            )
        fixed = {
            "schema_version": "v06_b_mask_first_v1",
            "category": "cable",
            "defect_types": CABLE_DEFECT_TYPES,
            "source_renderer_manifest": "configs/checkpoints/p2_b2_hazelnut_renderer.yaml",
            "source_experiment_config": "configs/experiments/v06_b_cable_type_prompts.yaml",
            "source_model_config": "configs/experiments/p2_a1_hazelnut_factorized.yaml",
            "image_size": 512,
            "planner_seed_noise_channels": 0,
            "seed": 20260907,
            "panel_normal_count": 4,
            "panel_seeds": (2026090701, 2026090702),
            "blind_panel_seed": 2026090703,
        }
        if config.source_model_config == "configs/experiments/p2_a1_cable_factorized.yaml":
            if config.output_relpath == CABLE_SCHEME2_CONTINUE_OUTPUT:
                scheme2_identity = {
                    "seed": 20260910,
                    "panel_seeds": (2026091001, 2026091002),
                    "blind_panel_seed": 2026091003,
                    "output_relpath": CABLE_SCHEME2_CONTINUE_OUTPUT,
                }
            elif config.output_relpath == CABLE_SCHEME2_A_TO_H_OUTPUT:
                scheme2_identity = {
                    "seed": 20260911,
                    "panel_seeds": (2026091101, 2026091102),
                    "blind_panel_seed": 2026091103,
                    "output_relpath": CABLE_SCHEME2_A_TO_H_OUTPUT,
                }
            elif config.output_relpath == CABLE_SCHEME2_A2_TO_H_OUTPUT:
                scheme2_identity = {
                    "seed": 20260912,
                    "panel_seeds": (2026091201, 2026091202),
                    "blind_panel_seed": 2026091203,
                    "output_relpath": CABLE_SCHEME2_A2_TO_H_OUTPUT,
                }
            else:
                scheme2_identity = {
                    "seed": 20260909,
                    "panel_seeds": (2026090901, 2026090902),
                    "blind_panel_seed": 2026090903,
                }
            fixed = {
                **fixed,
                "source_renderer_manifest": "configs/checkpoints/p2_b2_cable_renderer.yaml",
                "source_model_config": "configs/experiments/p2_a1_cable_factorized.yaml",
                **scheme2_identity,
            }
    for name, expected in fixed.items():
        if getattr(config, name) != expected:
            raise ValueError(f"{name} must equal {expected!r}")
    if config.planner_base_channels % 8:
        raise ValueError("planner_base_channels must be a multiple of eight")
    if (
        not config.planner_threshold_candidates
        or not all(0.0 < threshold < 1.0 for threshold in config.planner_threshold_candidates)
        or any(
            left >= right
            for left, right in zip(
                config.planner_threshold_candidates,
                config.planner_threshold_candidates[1:],
                strict=False,
            )
        )
    ):
        raise ValueError(
            "planner_threshold_candidates must be strictly increasing values in (0, 1)"
        )
    if not 0.0 < config.appearance_min_snr <= 1.0:
        raise ValueError("appearance_min_snr must be in (0, 1]")
    if not 0.0 < config.ranking_target_ratio < 1.0:
        raise ValueError("ranking_target_ratio must lie strictly between zero and one")
    return config


def load_v06_b_config(path: str | Path) -> V06BExperimentConfig:
    loaded = _mapping(Path(path))
    schema_version = loaded.get("schema_version")
    if schema_version == "v06_b_mask_first_v1":
        return _mask_first_config(loaded)
    common_keys = {
        "schema_version",
        "category",
        "defect_types",
        "source_renderer_manifest",
        "source_experiment_config",
        "source_model_config",
        "image_size",
        "proposal_radius",
        "support_corner_width",
        "rollout_steps",
        "backprop_steps",
        "checkpoint_tracked_steps",
        "feature_channels",
        "head_hidden_channels",
        "realization_threshold",
        "visibility_margin",
        "loss_weights",
        "training_steps",
        "seed",
        "checkpoint_every",
        "panel_normal_count",
        "panel_seeds",
        "output_relpath",
    }
    if schema_version == "v06_b_causal_renderer_v1":
        _exact_keys(loaded, common_keys | {"learning_rate", "gradient_clip"})
        loss_names = V1_LOSS_NAMES
        prompts: Mapping[str, str] = MappingProxyType({})
        parent_experiment_config = None
        parent_checkpoint_relpath = None
        parent_checkpoint_step = 0
        appearance_min_snr = 0.1
        ranking_target_ratio = 0.95
        amp_initial_scale = 65536.0
        scalar_learning_rate = _positive_float(loaded["learning_rate"], name="learning_rate")
        learning_rates = MappingProxyType(
            {name: scalar_learning_rate for name in LEARNING_RATE_NAMES}
        )
        scalar_gradient_clip = _positive_float(loaded["gradient_clip"], name="gradient_clip")
        gradient_clips = MappingProxyType(
            {name: scalar_gradient_clip for name in GRADIENT_CLIP_NAMES}
        )
        blind_panel_seed = _integer(loaded["seed"], name="seed", minimum=0)
    elif schema_version == "v06_b_causal_renderer_v2":
        _exact_keys(
            loaded,
            common_keys
            | {
                "prompts",
                "parent_experiment_config",
                "parent_checkpoint_relpath",
                "parent_checkpoint_step",
                "appearance_min_snr",
                "ranking_target_ratio",
                "amp_initial_scale",
                "learning_rates",
                "gradient_clips",
                "blind_panel_seed",
            },
        )
        loss_names = V2_LOSS_NAMES
        raw_prompts = loaded["prompts"]
        if not isinstance(raw_prompts, Mapping) or set(raw_prompts) != set(DEFECT_TYPES):
            raise ValueError("prompts must contain the exact Module-B defect types")
        if any(
            not isinstance(raw_prompts[name], str) or not raw_prompts[name] for name in DEFECT_TYPES
        ):
            raise ValueError("prompts must contain non-empty strings")
        prompts = MappingProxyType({name: raw_prompts[name] for name in DEFECT_TYPES})
        parent_experiment_config = _relative_path(
            loaded["parent_experiment_config"], name="parent_experiment_config"
        )
        parent_checkpoint_relpath = _relative_path(
            loaded["parent_checkpoint_relpath"], name="parent_checkpoint_relpath"
        )
        parent_checkpoint_step = _integer(
            loaded["parent_checkpoint_step"], name="parent_checkpoint_step"
        )
        appearance_min_snr = _positive_float(
            loaded["appearance_min_snr"], name="appearance_min_snr"
        )
        if appearance_min_snr > 1.0:
            raise ValueError("appearance_min_snr must be in (0, 1]")
        ranking_target_ratio = _positive_float(
            loaded["ranking_target_ratio"], name="ranking_target_ratio"
        )
        if ranking_target_ratio >= 1.0:
            raise ValueError("ranking_target_ratio must lie strictly between zero and one")
        amp_initial_scale = _positive_float(loaded["amp_initial_scale"], name="amp_initial_scale")
        raw_learning_rates = loaded["learning_rates"]
        if not isinstance(raw_learning_rates, Mapping) or set(raw_learning_rates) != set(
            LEARNING_RATE_NAMES
        ):
            raise ValueError("learning_rates must contain the exact optimizer groups")
        learning_rates = MappingProxyType(
            {
                name: _positive_float(raw_learning_rates[name], name=f"learning_rates.{name}")
                for name in LEARNING_RATE_NAMES
            }
        )
        raw_gradient_clips = loaded["gradient_clips"]
        if not isinstance(raw_gradient_clips, Mapping) or set(raw_gradient_clips) != set(
            GRADIENT_CLIP_NAMES
        ):
            raise ValueError("gradient_clips must contain renderer and head")
        gradient_clips = MappingProxyType(
            {
                name: _positive_float(raw_gradient_clips[name], name=f"gradient_clips.{name}")
                for name in GRADIENT_CLIP_NAMES
            }
        )
        blind_panel_seed = _integer(loaded["blind_panel_seed"], name="blind_panel_seed", minimum=0)
    else:
        raise ValueError("v0.6-B schema_version is unsupported")
    defect_types = loaded["defect_types"]
    if not isinstance(defect_types, list) or tuple(defect_types) != DEFECT_TYPES:
        raise ValueError("defect_types must equal the Hazelnut Module-B vocabulary")
    loss_weights = loaded["loss_weights"]
    if not isinstance(loss_weights, Mapping) or set(loss_weights) != set(loss_names):
        raise ValueError("loss_weights must contain the exact Module-B loss names")
    validated_weights = {
        name: _positive_float(loss_weights[name], name=f"loss_weights.{name}")
        for name in loss_names
    }
    panel_seeds = loaded["panel_seeds"]
    if (
        not isinstance(panel_seeds, list)
        or len(panel_seeds) != 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in panel_seeds
        )
        or len(set(panel_seeds)) != len(panel_seeds)
    ):
        raise ValueError("panel_seeds must contain two unique non-negative integers")

    config = V06BConfig(
        schema_version=str(schema_version),
        category=str(loaded["category"]),
        defect_types=tuple(defect_types),
        source_renderer_manifest=_relative_path(
            loaded["source_renderer_manifest"], name="source_renderer_manifest"
        ),
        source_experiment_config=_relative_path(
            loaded["source_experiment_config"], name="source_experiment_config"
        ),
        source_model_config=_relative_path(
            loaded["source_model_config"], name="source_model_config"
        ),
        prompts=prompts,
        parent_experiment_config=parent_experiment_config,
        parent_checkpoint_relpath=parent_checkpoint_relpath,
        parent_checkpoint_step=parent_checkpoint_step,
        image_size=_integer(loaded["image_size"], name="image_size"),
        proposal_radius=_integer(loaded["proposal_radius"], name="proposal_radius", minimum=0),
        support_corner_width=_integer(loaded["support_corner_width"], name="support_corner_width"),
        rollout_steps=_integer(loaded["rollout_steps"], name="rollout_steps"),
        backprop_steps=_integer(loaded["backprop_steps"], name="backprop_steps"),
        checkpoint_tracked_steps=loaded["checkpoint_tracked_steps"],
        feature_channels=_integer(loaded["feature_channels"], name="feature_channels"),
        head_hidden_channels=_integer(loaded["head_hidden_channels"], name="head_hidden_channels"),
        realization_threshold=_positive_float(
            loaded["realization_threshold"], name="realization_threshold"
        ),
        visibility_margin=_positive_float(loaded["visibility_margin"], name="visibility_margin"),
        appearance_min_snr=appearance_min_snr,
        ranking_target_ratio=ranking_target_ratio,
        amp_initial_scale=amp_initial_scale,
        loss_weights=MappingProxyType(validated_weights),
        training_steps=_integer(loaded["training_steps"], name="training_steps"),
        learning_rates=learning_rates,
        gradient_clips=gradient_clips,
        seed=_integer(loaded["seed"], name="seed", minimum=0),
        checkpoint_every=_integer(loaded["checkpoint_every"], name="checkpoint_every"),
        panel_normal_count=_integer(loaded["panel_normal_count"], name="panel_normal_count"),
        panel_seeds=tuple(panel_seeds),
        blind_panel_seed=blind_panel_seed,
        output_relpath=_relative_path(loaded["output_relpath"], name="output_relpath"),
    )
    fixed = {
        "category": "hazelnut",
        "image_size": 512,
        "proposal_radius": 24,
        "feature_channels": 320,
        "panel_normal_count": 4,
        "source_renderer_manifest": "configs/checkpoints/p2_b2_hazelnut_renderer.yaml",
        "source_experiment_config": (
            "configs/experiments/p2_b2_hazelnut_type_prompt_training.yaml"
        ),
        "source_model_config": "configs/experiments/p2_a1_hazelnut_factorized.yaml",
    }
    for name, expected in fixed.items():
        if getattr(config, name) != expected:
            raise ValueError(f"{name} must equal {expected!r}")
    if config.backprop_steps > config.rollout_steps:
        raise ValueError("backprop_steps must not exceed rollout_steps")
    if not isinstance(config.checkpoint_tracked_steps, bool):
        raise ValueError("checkpoint_tracked_steps must be boolean")
    if config.head_hidden_channels < 8 or config.head_hidden_channels % 8:
        raise ValueError("head_hidden_channels must be a positive multiple of eight")
    if not 0.0 < config.realization_threshold < 1.0:
        raise ValueError("realization_threshold must lie strictly between zero and one")
    if not 0.0 < config.visibility_margin <= 1.0:
        raise ValueError("visibility_margin must be in (0, 1]")
    if config.checkpoint_every > config.training_steps:
        raise ValueError("checkpoint_every must not exceed training_steps")
    return config


__all__ = [
    "CABLE_DEFECT_TYPES",
    "CABLE_SCHEME1_MODEL_CONFIG",
    "CABLE_SCHEME2_A2_TO_H_OUTPUT",
    "CABLE_SCHEME2_A_TO_H_OUTPUT",
    "CABLE_SCHEME2_CONTINUE_OUTPUT",
    "CABLE_SCHEME2_MODEL_CONFIG",
    "CABLE_SCHEME2_OUTPUT",
    "CABLE_SCHEME2_RENDERER_MANIFEST",
    "is_cable_a2_to_h",
    "is_cable_a_to_h",
    "is_cable_scheme2",
    "GRADIENT_CLIP_NAMES",
    "LEARNING_RATE_NAMES",
    "MASK_FIRST_RENDERER_LEARNING_RATE_NAMES",
    "MASK_FIRST_RENDERER_LOSS_NAMES",
    "MASK_FIRST_THRESHOLD_CANDIDATES",
    "V1_LOSS_NAMES",
    "V2_LOSS_NAMES",
    "V06BConfig",
    "V06BExperimentConfig",
    "V06BMaskFirstConfig",
    "load_v06_b_config",
]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml

_TYPES = ("crack", "cut", "hole", "print")
_PROMPTS = {
    "crack": "a photo of a hazelnut with thin cracks in the shell and fractured edges",
    "cut": "a photo of a hazelnut with a sharp cut in the shell",
    "hole": "a photo of a hazelnut with a puncture hole in the shell",
    "print": (
        "a photo of a hazelnut with dark printed characters and structured ink strokes "
        "on the shell surface"
    ),
}
_BASELINE_HASHES = {
    "adapter.safetensors": (
        "06cad68ee60e42479aa42fe2728024add0476fc3f7311c41be7904bbdd83ceb3"
    ),
    "conditioner.safetensors": (
        "0003cb5c597d8fdb35812b98e09c4c3bce26745f10424f5b636f61f3291038d8"
    ),
    "metadata.json": (
        "49d9c1c641e28e90ad3b3a28565863719be22dce3c34030985abe3c8ca8c6a67"
    ),
}
_CANDIDATE_HASHES = {
    "adapter.safetensors": (
        "520fdbd577ba6da80855ecf82b9627e53a41455bf804f0d0c713af817649af6a"
    ),
    "conditioner.safetensors": (
        "4247fed4526671e6e17da7adf78b4b15b30fd873f65be2b074c6e450e49fb073"
    ),
    "metadata.json": (
        "80f7f6e2575c827a89bbaf98a946f8668c36db9087d9bcbb9f4769ec042a5da8"
    ),
}
_REFERENCE_HASHES = {
    "metrics.jsonl": (
        "5f08113369adff8597c9cc47ece8e30ae67414ce84cd16ba477dc7633267b1af"
    ),
    "pairwise_metrics.jsonl": (
        "64207a3adfd59b38b703c80c40d1a457dbd9a686dad6adc872e66a720a3052be"
    ),
    "provenance.json": (
        "3776a5d7a6bce2931db48e4b35e01859786466f24000a24c519f646acfac5a8f"
    ),
    "report.json": (
        "9cbd22cc3380b4bca97ebb1614e832a985ea74da78440d46d9335e450ece6f93"
    ),
}
_CANDIDATE_RUN_HASHES = {
    "report.json": (
        "9c3e4badf2316cf2bf1812c149d2baa9f855462c641b893d9503f5178584e634"
    ),
    "provenance.json": (
        "e03a64c37750f6491870f6b9d8e529b374057819b9182d52bc2a71c9000b2696"
    ),
}
_TEXTURE_CANDIDATE_HASHES = {
    "adapter.safetensors": (
        "0829b0aec354bc0014114bc312b32d854799070269286e68d856ece09c3dd215"
    ),
    "conditioner.safetensors": (
        "2b2017fdbc45bd03d2a5a7adb4eb6acf5f8d0350671abaef8243b36803009e5c"
    ),
    "metadata.json": (
        "52a13170d127b92f22ed92276c0bd1fd89a7f7003c3835a07de0b64f3d8d49f9"
    ),
}
_TEXTURE_CANDIDATE_RUN_HASHES = {
    "report.json": (
        "69cc82f9577d4993f3fbe39123c9a031119013b6c8648cea5c2f0466d86b0deb"
    ),
    "provenance.json": (
        "3364f2901cad793ada341323fcf116a89fe9e2cdbb3687667b295e80f169799a"
    ),
}


@dataclass(frozen=True)
class P2B3Config:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    prompts: Mapping[str, str]
    baseline_checkpoint_relpath: str
    baseline_checkpoint_hashes: Mapping[str, str]
    candidate_checkpoint_relpath: str
    candidate_checkpoint_hashes: Mapping[str, str]
    candidate_run_hashes: Mapping[str, str]
    reference_run_relpath: str
    reference_git_head: str
    reference_hashes: Mapping[str, str]
    sample_count: int
    blind_seed: int
    denoise_steps: int
    evaluation_mode: str
    causal_difference: str


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _strict_yaml(path: Path) -> Mapping[str, Any]:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid P2-B3 YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("P2-B3 configuration must be a YAML mapping")
    return loaded


def _normalized_relpath(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty normalized relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise ValueError(f"{field} must be a normalized relative path")
    return value


def _sha_mapping(value: object, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    result = dict(value)
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in result.items()
    ):
        raise ValueError(f"{field} must map names to lowercase SHA256 values")
    return MappingProxyType(result)


def load_p2_b3_config(path: str | Path) -> P2B3Config:
    loaded = _strict_yaml(Path(path))
    keys = frozenset(P2B3Config.__dataclass_fields__)
    if set(loaded) != keys:
        missing = sorted(keys - set(loaded))
        extra = sorted(repr(key) for key in set(loaded) - keys)
        raise ValueError(f"P2-B3 schema mismatch: missing={missing}, extra={extra}")
    defect_types = loaded["defect_types"]
    prompts = loaded["prompts"]
    if not isinstance(defect_types, list) or tuple(defect_types) != _TYPES:
        raise ValueError("defect_types must equal the frozen Hazelnut vocabulary")
    if not isinstance(prompts, Mapping) or tuple(prompts) != _TYPES:
        raise ValueError("prompts must contain the exact ordered Hazelnut vocabulary")
    prompt_values = dict(prompts)
    if prompt_values != _PROMPTS:
        raise ValueError("prompts must equal the accepted P2-B2 Type-Prompt mapping")
    config = P2B3Config(
        schema_version=str(loaded["schema_version"]),
        category=str(loaded["category"]),
        defect_types=tuple(defect_types),
        prompts=MappingProxyType(prompt_values),
        baseline_checkpoint_relpath=_normalized_relpath(
            loaded["baseline_checkpoint_relpath"], "baseline_checkpoint_relpath"
        ),
        baseline_checkpoint_hashes=_sha_mapping(
            loaded["baseline_checkpoint_hashes"], "baseline_checkpoint_hashes"
        ),
        candidate_checkpoint_relpath=_normalized_relpath(
            loaded["candidate_checkpoint_relpath"], "candidate_checkpoint_relpath"
        ),
        candidate_checkpoint_hashes=_sha_mapping(
            loaded["candidate_checkpoint_hashes"], "candidate_checkpoint_hashes"
        ),
        candidate_run_hashes=_sha_mapping(
            loaded["candidate_run_hashes"], "candidate_run_hashes"
        ),
        reference_run_relpath=_normalized_relpath(
            loaded["reference_run_relpath"], "reference_run_relpath"
        ),
        reference_git_head=str(loaded["reference_git_head"]),
        reference_hashes=_sha_mapping(
            loaded["reference_hashes"], "reference_hashes"
        ),
        sample_count=loaded["sample_count"],
        blind_seed=loaded["blind_seed"],
        denoise_steps=loaded["denoise_steps"],
        evaluation_mode=str(loaded["evaluation_mode"]),
        causal_difference=str(loaded["causal_difference"]),
    )
    shared = {
        "category": "hazelnut",
        "reference_run_relpath": (
            "p2_b1/hazelnut/heldout-generation/seed-20260817-attempt2"
        ),
        "reference_git_head": "4bd6d47d83292107c9c68f99f841eb46d000789e",
        "reference_hashes": _REFERENCE_HASHES,
        "sample_count": 16,
        "denoise_steps": 50,
        "evaluation_mode": "inference_only_checkpoint_probe",
        "causal_difference": "checkpoint_weights_only",
    }
    profiles = {
        "p2_b3_checkpoint_paired_probe_v1": {
            **shared,
            "schema_version": "p2_b3_checkpoint_paired_probe_v1",
            "baseline_checkpoint_relpath": (
                "p2_a1/hazelnut/factorized/seed-20260817-attempt1/checkpoint"
            ),
            "baseline_checkpoint_hashes": _BASELINE_HASHES,
            "candidate_checkpoint_relpath": (
                "p2_b2/hazelnut/type-prompt-training/seed-20260817-attempt1/checkpoint"
            ),
            "candidate_checkpoint_hashes": _CANDIDATE_HASHES,
            "candidate_run_hashes": _CANDIDATE_RUN_HASHES,
            "blind_seed": 202608179,
        },
        "p2_b6_texture_checkpoint_paired_probe_v1": {
            **shared,
            "schema_version": "p2_b6_texture_checkpoint_paired_probe_v1",
            "baseline_checkpoint_relpath": (
                "p2_b2/hazelnut/type-prompt-training/seed-20260817-attempt1/"
                "checkpoint"
            ),
            "baseline_checkpoint_hashes": _CANDIDATE_HASHES,
            "candidate_checkpoint_relpath": (
                "p2_b5/hazelnut/type-selective-texture/"
                "seed-20260824-attempt2/checkpoint"
            ),
            "candidate_checkpoint_hashes": _TEXTURE_CANDIDATE_HASHES,
            "candidate_run_hashes": _TEXTURE_CANDIDATE_RUN_HASHES,
            "blind_seed": 202608241,
        },
    }
    expected = profiles.get(config.schema_version)
    if expected is None:
        raise ValueError("schema_version must identify a frozen P2-B3 profile")
    for field, value in expected.items():
        actual = getattr(config, field)
        if actual != value:
            raise ValueError(f"{field} must equal the frozen P2-B3 value {value!r}")
    if type(config.sample_count) is not int or type(config.blind_seed) is not int:
        raise ValueError("sample_count and blind_seed must be integers")
    if type(config.denoise_steps) is not int:
        raise ValueError("denoise_steps must be an integer")
    return config


__all__ = ["P2B3Config", "load_p2_b3_config"]

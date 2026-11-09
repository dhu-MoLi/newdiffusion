"""Strict materialisation of the frozen Module-A RC1 identity schedules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING

import yaml
from yaml.constructor import ConstructorError

if TYPE_CHECKING:
    from cadiff.data.mvtec_manifest import ManifestRecord

    from .config import V06AExperimentConfig


_EXPECTED_KEYS = {
    "schema_version",
    "category",
    "normal_train_universe",
    "a_dev_ids",
    "open_normal_universe",
    "defect_splits",
    "excluded_types",
    "atlas_normal_ids",
    "p0_pairs",
    "p1_normal_ids",
    "p2_open_normal_ids",
}
_UNIVERSE_KEYS = {"start", "end"}
_DEFECT_KEYS = {"universe", "fewshot_ids", "open_report_ids"}
_P0_KEYS = {"defect_type", "source_id", "target_normal_id", "seed"}
_MANIFEST_TOKEN = object()


def _frozen_range(start: int, end: int) -> tuple[str, ...]:
    return tuple(f"{index:03d}" for index in range(start, end + 1))


def _frozen_split(start: int, end: int, fewshot_count: int) -> Mapping[str, tuple[str, ...]]:
    universe = _frozen_range(start, end)
    return MappingProxyType(
        {
            "universe": universe,
            "fewshot_ids": universe[:fewshot_count],
            "open_report_ids": universe[fewshot_count:],
        }
    )


_EXPECTED = {
    "hazelnut": {
        "defect_types": ("crack", "cut", "hole", "print"),
        "counts": (351, 40, 40, 22, 48, 0),
        "normal_train_ids": _frozen_range(0, 390),
        "a_dev_ids": tuple(f"{index:03d}" for index in range(0, 391, 10)),
        "open_normal_ids": _frozen_range(0, 39),
        "defect_splits": MappingProxyType(
            {
                "crack": _frozen_split(0, 17, 6),
                "cut": _frozen_split(0, 16, 5),
                "hole": _frozen_split(0, 17, 6),
                "print": _frozen_split(0, 16, 5),
            }
        ),
        "p1": ("000", "130", "260", "390"),
        "p2": ("000", "013", "026", "039"),
        "p0_targets": ("000", "260"),
        "excluded": (),
        "excluded_ids": MappingProxyType({}),
        "atlas_normal_ids": (
            "001",
            "026",
            "053",
            "078",
            "104",
            "131",
            "156",
            "182",
            "208",
            "234",
            "259",
            "286",
            "312",
            "337",
            "364",
            "389",
        ),
        "p0_pairs": (
            ("crack", "000", "000", 202608280100),
            ("crack", "001", "260", 202608280101),
            ("cut", "000", "000", 202608280110),
            ("cut", "001", "260", 202608280111),
            ("hole", "000", "000", 202608280120),
            ("hole", "001", "260", 202608280121),
            ("print", "000", "000", 202608280130),
            ("print", "001", "260", 202608280131),
        ),
    },
    "cable": {
        "defect_types": (
            "bent_wire",
            "cut_inner_insulation",
            "cut_outer_insulation",
            "poke_insulation",
        ),
        "counts": (201, 23, 58, 14, 33, 45),
        "normal_train_ids": _frozen_range(0, 223),
        "a_dev_ids": (
            "000",
            "010",
            "020",
            "030",
            "041",
            "051",
            "061",
            "071",
            "081",
            "091",
            "101",
            "112",
            "122",
            "132",
            "142",
            "152",
            "162",
            "172",
            "182",
            "193",
            "203",
            "213",
            "223",
        ),
        "open_normal_ids": _frozen_range(0, 57),
        "defect_splits": MappingProxyType(
            {
                "bent_wire": _frozen_split(0, 12, 4),
                "cut_inner_insulation": _frozen_split(0, 13, 4),
                "cut_outer_insulation": _frozen_split(0, 9, 3),
                "poke_insulation": _frozen_split(0, 9, 3),
            }
        ),
        "p1": ("000", "071", "152", "223"),
        "p2": (),
        "p0_targets": ("000", "152"),
        "excluded": ("cable_swap", "combined", "missing_cable", "missing_wire"),
        "excluded_ids": MappingProxyType(
            {
                "cable_swap": _frozen_range(0, 11),
                "combined": _frozen_range(0, 10),
                "missing_cable": _frozen_range(0, 11),
                "missing_wire": _frozen_range(0, 9),
            }
        ),
        "atlas_normal_ids": (
            "001",
            "015",
            "031",
            "045",
            "059",
            "075",
            "089",
            "104",
            "119",
            "134",
            "148",
            "164",
            "178",
            "192",
            "208",
            "222",
        ),
        "p0_pairs": (
            ("bent_wire", "000", "000", 202608280200),
            ("bent_wire", "001", "152", 202608280201),
            ("cut_inner_insulation", "000", "000", 202608280210),
            ("cut_inner_insulation", "001", "152", 202608280211),
            ("cut_outer_insulation", "000", "000", 202608280220),
            ("cut_outer_insulation", "001", "152", 202608280221),
            ("poke_insulation", "000", "000", 202608280230),
            ("poke_insulation", "001", "152", 202608280231),
        ),
    },
}


@dataclass(frozen=True)
class IdentityRecord:
    category: str
    defect_type: str
    item_id: str
    image_relpath: str
    mask_relpath: str | None
    role: str
    sample_id: str


@dataclass(frozen=True)
class P0Pair:
    defect_type: str
    source_id: str
    target_normal_id: str
    seed: int
    source_role: str


@dataclass(frozen=True)
class PanelIdentity:
    defect_type: str
    normal_id: str
    seed: int
    phase: str


@dataclass(frozen=True)
class IdentityManifest:
    schema_version: str
    category: str
    defect_types: tuple[str, ...]
    records: tuple[IdentityRecord, ...]
    normal_fit: tuple[IdentityRecord, ...]
    normal_dev: tuple[IdentityRecord, ...]
    open_normals: tuple[IdentityRecord, ...]
    fewshot: tuple[IdentityRecord, ...]
    open_report: tuple[IdentityRecord, ...]
    excluded: tuple[IdentityRecord, ...]
    atlas_normal_ids: tuple[str, ...]
    fewshot_ids_by_type: Mapping[str, tuple[str, ...]]
    excluded_ids_by_type: Mapping[str, tuple[str, ...]]
    p0_pairs: tuple[P0Pair, ...]
    p1_normals: tuple[IdentityRecord, ...]
    p2_normals: tuple[IdentityRecord, ...]
    p1_identities: tuple[PanelIdentity, ...]
    p2_identities: tuple[PanelIdentity, ...]
    _provenance: object = field(default=None, repr=False, compare=False)
    _authority_digest: str = field(default="", repr=False, compare=False)

    def is_authoritative(self) -> bool:
        return self._provenance is _MANIFEST_TOKEN and self._authority_digest == _manifest_digest(
            self
        )

    @property
    def authority_digest(self) -> str:
        if not self.is_authoritative():
            raise ValueError("identity manifest is not loader-authoritative")
        return self._authority_digest


def _manifest_digest(manifest: IdentityManifest) -> str:
    payload = repr(
        (
            manifest.schema_version,
            manifest.category,
            manifest.defect_types,
            manifest.records,
            manifest.normal_fit,
            manifest.normal_dev,
            manifest.open_normals,
            manifest.fewshot,
            manifest.open_report,
            manifest.excluded,
            manifest.atlas_normal_ids,
            tuple(manifest.fewshot_ids_by_type.items()),
            tuple(manifest.excluded_ids_by_type.items()),
            manifest.p0_pairs,
            manifest.p1_normals,
            manifest.p2_normals,
            manifest.p1_identities,
            manifest.p2_identities,
        )
    )
    return sha256(payload.encode()).hexdigest()


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
        result = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid v0.6-A identity YAML: {error}") from error
    if not isinstance(result, Mapping):
        raise ValueError("v0.6-A identity manifest must be a YAML mapping")
    return result


def _exact_keys(value: Mapping[object, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(expected - set(value))}, "
            f"extra={sorted(repr(item) for item in set(value) - expected)}"
        )


def _id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 3 or not value.isdecimal():
        raise ValueError(f"{name} must be a three-digit lexical ID")
    return value


def _ids(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of IDs")
    result = tuple(_id(item, name=name) for item in value)
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique and lexicographically sorted")
    return result


def _universe(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a universe mapping")
    _exact_keys(value, _UNIVERSE_KEYS, name=name)
    start, end = _id(value["start"], name=f"{name}.start"), _id(value["end"], name=f"{name}.end")
    if int(start) > int(end):
        raise ValueError(f"{name} start must precede end")
    return tuple(f"{index:03d}" for index in range(int(start), int(end) + 1))


def _sample_id(category: str, defect_type: str, image_relpath: str) -> str:
    return sha256(f"{category}\n{defect_type}\n{image_relpath}".encode()).hexdigest()


def _path(value: str) -> str:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or pure.as_posix() != value:
        raise ValueError("materialized path must be normalized and repo-relative")
    return value


def _record(category: str, defect_type: str, item_id: str, role: str) -> IdentityRecord:
    if defect_type == "good":
        split = "train" if role in {"normal_fit", "normal_dev"} else "test"
        image_relpath = _path(f"{category}/{split}/good/{item_id}.png")
        mask_relpath = None
    else:
        image_relpath = _path(f"{category}/test/{defect_type}/{item_id}.png")
        mask_relpath = _path(f"{category}/ground_truth/{defect_type}/{item_id}_mask.png")
    return IdentityRecord(
        category=category,
        defect_type=defect_type,
        item_id=item_id,
        image_relpath=image_relpath,
        mask_relpath=mask_relpath,
        role=role,
        sample_id=_sample_id(category, defect_type, image_relpath),
    )


def load_v06_a_identity_manifest(
    path: Path, *, experiment: V06AExperimentConfig
) -> IdentityManifest:
    """Materialize all frozen roles and fixed panel identities for one category."""

    raw = _load_mapping(path)
    _exact_keys(raw, _EXPECTED_KEYS, name="identity manifest")
    if raw["schema_version"] != "v06_a_identity_manifest_rc1":
        raise ValueError("schema_version must equal v06_a_identity_manifest_rc1")
    category = raw["category"]
    if not isinstance(category, str) or category not in _EXPECTED:
        raise ValueError("category must be hazelnut or cable")
    if category != experiment.category:
        raise ValueError("identity manifest category must agree with experiment")
    expected = _EXPECTED[category]
    defect_types = expected["defect_types"]
    if tuple(experiment.defect_types) != defect_types:
        raise ValueError("identity manifest vocabulary must agree with experiment")

    train_ids = _universe(raw["normal_train_universe"], name="normal_train_universe")
    if train_ids != expected["normal_train_ids"]:
        raise ValueError("normal_train_universe must equal the frozen schedule")
    dev_ids = _ids(raw["a_dev_ids"], name="a_dev_ids")
    if dev_ids != expected["a_dev_ids"]:
        raise ValueError("a_dev_ids must equal the frozen schedule")
    fit_ids = tuple(item for item in train_ids if item not in set(dev_ids))
    open_ids = _universe(raw["open_normal_universe"], name="open_normal_universe")
    if open_ids != expected["open_normal_ids"]:
        raise ValueError("open_normal_universe must equal the frozen schedule")
    normal_fit = tuple(_record(category, "good", item, "normal_fit") for item in fit_ids)
    normal_dev = tuple(_record(category, "good", item, "normal_dev") for item in dev_ids)
    open_normals = tuple(_record(category, "good", item, "open_normal") for item in open_ids)

    defect_splits = raw["defect_splits"]
    if not isinstance(defect_splits, Mapping) or set(defect_splits) != set(defect_types):
        raise ValueError("defect_splits vocabulary must equal the frozen category vocabulary")
    fewshot: list[IdentityRecord] = []
    open_report: list[IdentityRecord] = []
    fewshot_ids_by_type: dict[str, tuple[str, ...]] = {}
    for defect_type in defect_types:
        split = defect_splits[defect_type]
        if not isinstance(split, Mapping):
            raise ValueError(f"defect_splits.{defect_type} must be a mapping")
        _exact_keys(split, _DEFECT_KEYS, name=f"defect_splits.{defect_type}")
        universe = _universe(split["universe"], name=f"defect_splits.{defect_type}.universe")
        few_ids = _ids(split["fewshot_ids"], name=f"defect_splits.{defect_type}.fewshot_ids")
        opened_ids = _ids(
            split["open_report_ids"],
            name=f"defect_splits.{defect_type}.open_report_ids",
        )
        frozen_split = expected["defect_splits"][defect_type]
        if universe != frozen_split["universe"]:
            raise ValueError(f"defect_splits.{defect_type}.universe must equal frozen schedule")
        if few_ids != frozen_split["fewshot_ids"]:
            raise ValueError(f"defect_splits.{defect_type}.fewshot_ids must equal frozen schedule")
        if opened_ids != frozen_split["open_report_ids"]:
            raise ValueError(
                f"defect_splits.{defect_type}.open_report_ids must equal frozen schedule"
            )
        fewshot_ids_by_type[defect_type] = few_ids
        fewshot.extend(_record(category, defect_type, item, "fewshot") for item in few_ids)
        open_report.extend(
            _record(category, defect_type, item, "open_report") for item in opened_ids
        )

    excluded_types = raw["excluded_types"]
    if not isinstance(excluded_types, Mapping) or set(excluded_types) != set(expected["excluded"]):
        raise ValueError("excluded_types must equal the frozen excluded vocabulary")
    excluded: list[IdentityRecord] = []
    excluded_ids_by_type: dict[str, tuple[str, ...]] = {}
    for defect_type in expected["excluded"]:
        split = excluded_types[defect_type]
        if not isinstance(split, Mapping):
            raise ValueError(f"excluded_types.{defect_type} must be a mapping")
        _exact_keys(split, {"universe"}, name=f"excluded_types.{defect_type}")
        ids = _universe(split["universe"], name=f"excluded_types.{defect_type}.universe")
        if ids != expected["excluded_ids"][defect_type]:
            raise ValueError(f"excluded_types.{defect_type}.universe must equal frozen schedule")
        excluded_ids_by_type[defect_type] = ids
        excluded.extend(_record(category, defect_type, item, "excluded") for item in ids)

    atlas_normal_ids = _ids(raw["atlas_normal_ids"], name="atlas_normal_ids")
    if atlas_normal_ids != expected["atlas_normal_ids"]:
        raise ValueError("atlas_normal_ids must equal the frozen A-fit schedule")
    p1_ids = _ids(raw["p1_normal_ids"], name="p1_normal_ids")
    if p1_ids != expected["p1"] or not set(p1_ids) <= set(dev_ids):
        raise ValueError("p1_normal_ids must equal the frozen A-dev panel IDs")
    p2_ids = _ids(raw["p2_open_normal_ids"], name="p2_open_normal_ids")
    if p2_ids != expected["p2"] or not set(p2_ids) <= set(open_ids):
        raise ValueError("p2_open_normal_ids must equal the frozen open-normal IDs")

    raw_p0 = raw["p0_pairs"]
    if not isinstance(raw_p0, list) or len(raw_p0) != experiment.p0_pairs_per_category:
        raise ValueError("p0_pairs must have the frozen per-category count")
    p0_pairs: list[P0Pair] = []
    for item in raw_p0:
        if not isinstance(item, Mapping):
            raise ValueError("p0_pairs entries must be mappings")
        _exact_keys(item, _P0_KEYS, name="p0_pairs entry")
        defect_type = item["defect_type"]
        source_id = _id(item["source_id"], name="p0_pairs.source_id")
        target_id = _id(item["target_normal_id"], name="p0_pairs.target_normal_id")
        seed = item["seed"]
        if defect_type not in defect_types or source_id not in fewshot_ids_by_type[defect_type]:
            raise ValueError("p0_pairs sources must be fewshot category-vocabulary identities")
        if target_id not in train_ids or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("p0_pairs must have train normal targets and integer seeds")
        p0_pairs.append(P0Pair(defect_type, source_id, target_id, seed, "fewshot"))
    if (
        tuple(
            (pair.defect_type, pair.source_id, pair.target_normal_id, pair.seed)
            for pair in p0_pairs
        )
        != expected["p0_pairs"]
    ):
        raise ValueError("p0_pairs must equal the frozen type/source/target/seed schedule")

    p1_normals = tuple(_record(category, "good", item, "normal_dev") for item in p1_ids)
    p2_normals = tuple(_record(category, "good", item, "open_normal") for item in p2_ids)
    p1_identities = tuple(
        PanelIdentity(defect_type, normal_id, seed, "p1")
        for defect_type in defect_types
        for normal_id in p1_ids
        for seed in experiment.panel_seeds
    )
    p2_identities = tuple(
        PanelIdentity(defect_type, normal_id, seed, "p2")
        for defect_type in defect_types
        for normal_id in p2_ids
        for seed in experiment.panel_seeds
    )
    records = tuple(
        normal_fit
        + normal_dev
        + open_normals
        + tuple(fewshot)
        + tuple(open_report)
        + tuple(excluded)
    )
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("materialized identities must have unique sample IDs")
    counts = tuple(map(len, (normal_fit, normal_dev, open_normals, fewshot, open_report, excluded)))
    if counts != expected["counts"]:
        raise ValueError(f"materialized role counts do not equal frozen values: {counts}")
    provisional = IdentityManifest(
        schema_version="v06_a_identity_manifest_rc1",
        category=category,
        defect_types=defect_types,
        records=records,
        normal_fit=normal_fit,
        normal_dev=normal_dev,
        open_normals=open_normals,
        fewshot=tuple(fewshot),
        open_report=tuple(open_report),
        excluded=tuple(excluded),
        atlas_normal_ids=atlas_normal_ids,
        fewshot_ids_by_type=MappingProxyType(fewshot_ids_by_type),
        excluded_ids_by_type=MappingProxyType(excluded_ids_by_type),
        p0_pairs=tuple(p0_pairs),
        p1_normals=p1_normals,
        p2_normals=p2_normals,
        p1_identities=p1_identities,
        p2_identities=p2_identities,
        _provenance=_MANIFEST_TOKEN,
    )
    return IdentityManifest(
        **{
            name: getattr(provisional, name)
            for name in IdentityManifest.__dataclass_fields__
            if name != "_authority_digest"
        },
        _authority_digest=_manifest_digest(provisional),
    )


def crosscheck_identity_split(
    manifest: IdentityManifest,
    dataset_root: Path,
    *,
    canonical_records: Mapping[str, ManifestRecord] | None = None,
    verify_sha256: bool = False,
) -> None:
    """Check only declared paths/masks, plus optional canonical hash equality.

    This intentionally performs no image decoding or visual inspection.
    """

    if verify_sha256 and canonical_records is None:
        raise ValueError("canonical_records is required when verify_sha256=True")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    root = dataset_root.resolve(strict=True)

    def checked_file(relpath: str, *, kind: str) -> Path:
        candidate = root / relpath
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"missing {kind}: {candidate}") from error
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{kind} escapes dataset root: {resolved}; root={root}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"missing {kind}: {candidate}")
        return resolved

    for record in manifest.records:
        image = checked_file(record.image_relpath, kind="image")
        if record.mask_relpath is not None:
            mask = checked_file(record.mask_relpath, kind="mask")
        if canonical_records is None:
            continue
        canonical = canonical_records.get(record.image_relpath) or canonical_records.get(
            record.sample_id
        )
        if canonical is None:
            raise ValueError(f"canonical manifest lacks identity: {record.image_relpath}")
        if (
            canonical.category != record.category
            or canonical.defect_type != record.defect_type
            or canonical.image_relpath != record.image_relpath
            or canonical.mask_relpath != record.mask_relpath
        ):
            raise ValueError(f"canonical manifest mismatch: {record.image_relpath}")
        if verify_sha256:
            if sha256(image.read_bytes()).hexdigest() != canonical.image_sha256:
                raise ValueError(f"image SHA256 mismatch: {record.image_relpath}")
            if record.mask_relpath is not None and (
                canonical.mask_sha256 is None
                or sha256(mask.read_bytes()).hexdigest() != canonical.mask_sha256
            ):
                raise ValueError(f"mask SHA256 mismatch: {record.mask_relpath}")

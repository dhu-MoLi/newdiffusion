"""Diagnostic only: compare object-support arms. Official A is TokenCut; this panel still shows RGB vs a2-1 vs a2-2."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from cadiff.config import load_config
from cadiff.provenance import write_json_atomic
from cadiff.v06_a.a2 import _material_proposal_maps, _upsample_map, load_a2_generator
from cadiff.v06_a.config import load_v06_a_experiment
from cadiff.v06_a.data import load_identity_record
from cadiff.v06_a.identities import IdentityRecord, load_v06_a_identity_manifest
from cadiff.v06_a.structure import (
    NormalStructure,
    _fill_binary_holes,
    _largest_connected_component,
    estimate_normal_structure,
)
from cadiff.v06_atlas_diag.cable_a import _lookup_p1_target, build_cable_a_slots
from cadiff.v06_atlas_diag.cad3_a import inventory_cad3
from cadiff.v06_atlas_diag.extra_a import (
    ExtraACategory,
    ExtraAStructureConfig,
    _even_ids,
    _fewshot_ids,
    _item_id,
    inventory_mvtec,
    inventory_visa,
    load_extra_identity,
)
from cadiff.v06_a.tokencut import (
    load_official_dinov2_encoder,
    tokencut_object_mask,
)
from cadiff.v06_atlas_diag.features import FrozenDinoEncoder
from cadiff.v06_atlas_diag.inpaint import neighbor_fill_source

SCHEMA_VERSION = "v06_a_a2_support_diag_v1"
A21_ATTN_LAYERS = 4
_FORBIDDEN_OUTPUT_PREFIXES = ("f1_", "f2_", "f3_", "f4_")
FLOW_COLUMNS = (
    "normal",
    "rgb_S",
    "a21_attn",
    "a21_S",
    "a22_ncut",
    "a22_S",
    "a2_prop",
    "a21_prop",
    "a22_prop",
)
COLUMN_MEANINGS = {
    "normal": "目标正常图 x",
    "rgb_S": "正式 a2 计算物体（边框 RGB/MAD）",
    "a21_attn": "a2-1：后 4 层 CLS→patch 注意力",
    "a21_S": "a2-1 物体 = Otsu(注意力) + 最大连通 + 填洞",
    "a22_ncut": "a2-2：TokenCut 第二小特征向量",
    "a22_S": "a2-2 物体 = 谱二分（少碰网格边的一块）",
    "a2_prop": "proposal clip 到 rgb_S（历史 RGB 臂；正式 A 已改 TokenCut）",
    "a21_prop": "同一材料匹配，clip 到 a2-1 物体",
    "a22_prop": "同一材料匹配，clip 到 a2-2 物体",
}


def otsu_threshold(values: NDArray[np.floating]) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    hist, edges = np.histogram(flat, bins=256)
    total = int(hist.sum())
    if total == 0:
        return float(flat.mean())
    center = 0.5 * (edges[:-1] + edges[1:])
    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    sum1 = np.cumsum(hist * center)
    sum2 = sum1[-1] - sum1
    mu1 = sum1 / np.maximum(weight1, 1)
    mu2 = sum2 / np.maximum(weight2, 1)
    sigma = weight1[:-1] * weight2[:-1] * (mu1[:-1] - mu2[:-1]) ** 2
    return float(center[int(np.argmax(sigma))])


def _upsample_mask(grid: NDArray[np.bool_], size: int = 512) -> NDArray[np.bool_]:
    image = Image.fromarray(grid.astype(np.uint8) * 255)
    return np.asarray(image.resize((size, size), resample=Image.Resampling.NEAREST)) > 127


def _finalize_object(raw: NDArray[np.bool_], valid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    clipped = np.asarray(raw, dtype=bool) & np.asarray(valid, dtype=bool)
    if not clipped.any():
        return np.zeros(valid.shape, dtype=bool)
    component = _largest_connected_component(clipped)
    filled = _fill_binary_holes(component) & valid
    return filled if filled.any() else component


def _less_border_partition(flags: NDArray[np.bool_]) -> NDArray[np.bool_]:
    height, width = flags.shape
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    other = ~flags
    border_a = int((flags & border).sum())
    border_b = int((other & border).sum())
    if border_a != border_b:
        return flags if border_a < border_b else other
    area_a = int(flags.sum())
    area_b = int(other.sum())
    if area_a != area_b:
        return flags if area_a < area_b else other
    return flags


def tokencut_eigenvector_map(tokens: NDArray[np.floating]) -> NDArray[np.float32]:
    grid = np.asarray(tokens, dtype=np.float32)
    height, width, channels = grid.shape
    vectors = grid.reshape(height * width, channels)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    desc = vectors / np.maximum(norms, 1e-12)
    affinity = np.clip(desc @ desc.T, 0.0, None)
    np.fill_diagonal(affinity, 0.0)
    affinity = affinity + 1e-3
    degree = affinity.sum(axis=1)
    scale = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = (scale[:, None] * affinity) * scale[None, :]
    _vals, vecs = np.linalg.eigh(np.eye(height * width, dtype=np.float64) - normalized.astype(np.float64))
    field = np.abs(vecs[:, 1]).reshape(height, width).astype(np.float32)
    return _upsample_map(field)


def cls_attention_object_mask(
    attention_grid: NDArray[np.floating],
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    similarity = _upsample_map(np.asarray(attention_grid, dtype=np.float32))
    inside = similarity[valid]
    if inside.size == 0:
        return np.zeros(valid.shape, dtype=bool)
    tau = otsu_threshold(inside)
    return _finalize_object(similarity >= tau, valid)


def proposal_with_object(
    source_gt: NDArray[np.bool_],
    source_valid: NDArray[np.bool_],
    target: NormalStructure,
    object_mask: NDArray[np.bool_],
    *,
    source_feat: Any,
    target_feat: Any,
) -> tuple[NDArray[np.bool_], float, NDArray[np.float32]]:
    support = np.ascontiguousarray(object_mask, dtype=bool) & target.valid
    alt = replace(target, support=support)
    return _material_proposal_maps(
        source_gt,
        source_valid,
        alt,
        source_feat=source_feat,
        target_feat=target_feat,
    )


def load_offline_dinov2_encoder(*, device: str | None = None) -> FrozenDinoEncoder:
    return load_official_dinov2_encoder(device=device)


def _dino_blocks(encoder: FrozenDinoEncoder) -> Any:
    model = encoder.model
    if hasattr(model, "blocks"):
        return model.blocks
    raise ValueError("a2-1 needs DINOv2 blocks; encoder fallback cannot yield CLS attention")


def _cls_to_tokens(attn_module: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    batch, length, channels = tokens.shape
    heads = int(attn_module.num_heads)
    qkv = attn_module.qkv(tokens)
    qkv = qkv.reshape(batch, length, 3, heads, channels // heads).permute(2, 0, 3, 1, 4)
    query = qkv[0] * attn_module.scale
    key = qkv[1]
    attn = (query @ key.transpose(-2, -1)).softmax(dim=-1)
    return attn[0, :, 0].mean(0)


def encode_target_with_cls_attention(
    encoder: FrozenDinoEncoder,
    rgb: NDArray[np.uint8],
    *,
    last_n: int = A21_ATTN_LAYERS,
) -> tuple[Any, NDArray[np.float32]]:
    blocks = list(_dino_blocks(encoder))
    captured: list[np.ndarray] = []
    handles = []
    for block in blocks[-last_n:]:
        attn = block.attn

        def _hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
            attn_module: torch.nn.Module = attn,
        ) -> None:
            weights = _cls_to_tokens(attn_module, inputs[0])
            captured.append(weights.detach().float().cpu().numpy())

        handles.append(attn.register_forward_hook(_hook))
    try:
        features = encoder.encode(rgb)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != last_n:
        raise ValueError("a2-1 did not capture the requested CLS attention layers")
    stacked = np.mean(np.stack(captured, axis=0), axis=0)
    grid = int(features.grid_height) * int(features.grid_width)
    if stacked.size < grid:
        raise ValueError("CLS attention is shorter than the patch grid")
    prefix = stacked.size - grid
    patch = stacked[prefix:].reshape(features.grid_height, features.grid_width)
    return features, np.ascontiguousarray(patch, dtype=np.float32)


def _contained_output(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuses to overwrite existing output: {output}")
    if any(part.startswith(_FORBIDDEN_OUTPUT_PREFIXES) for part in output.parts):
        raise ValueError("refuses to write f1-f4 outputs")
    return output


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    painted = np.array(rgb, dtype=np.uint8, copy=True)
    binary = np.asarray(mask, dtype=bool)
    painted[binary] = (
        (1.0 - alpha) * painted[binary] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return painted


def _heatmap(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float32)
    inside = np.asarray(valid, dtype=bool)
    image = np.zeros((*field.shape, 3), dtype=np.uint8)
    if not inside.any():
        return image
    lo = float(field[inside].min())
    hi = float(field[inside].max())
    norm = np.zeros_like(field, dtype=np.float32)
    if hi > lo:
        norm[inside] = (field[inside] - lo) / (hi - lo)
    image[..., 0] = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip((1.0 - np.abs(norm - 0.5) * 2.0) * 180.0, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)
    image[~inside] = 0
    return image


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=np.uint8) * 255)


def _save_sheet(
    path: Path,
    rows: list[tuple[str, dict[str, np.ndarray]]],
    columns: tuple[str, ...],
    *,
    tile: int,
    gutter: int = 280,
) -> None:
    header = 28
    canvas = Image.new(
        "RGB",
        (gutter + tile * len(columns), header + len(rows) * (tile + 4)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(columns):
        draw.text((gutter + index * tile + 4, 6), name, fill="black")
    for row_index, (label, images) in enumerate(rows):
        top = header + row_index * (tile + 4)
        draw.text((6, top + max(tile // 2 - 6, 4)), label, fill="black")
        for column, name in enumerate(columns):
            array = images[name]
            resample = Image.Resampling.NEAREST if array.ndim == 2 else Image.Resampling.LANCZOS
            tile_image = Image.fromarray(array).convert("RGB").resize((tile, tile), resample)
            canvas.paste(tile_image, (gutter + column * tile, top))
    canvas.save(path)


def run_a2_support_diagnostic(
    *,
    repository_root: Path,
    dataset_root: Path,
    output: Path,
    a_experiment: Path,
    device: str | None = None,
    tile: int = 160,
) -> dict[str, Any]:
    destination = _contained_output(output)
    experiment = load_v06_a_experiment(a_experiment)
    manifest = load_v06_a_identity_manifest(
        repository_root / experiment.identity_manifest,
        experiment=experiment,
    )
    if experiment.category != "cable":
        raise ValueError("a2 support diagnostic freezes Cable P1 identities")
    slots = build_cable_a_slots(manifest, experiment=experiment)
    generator = load_a2_generator(
        experiment_path=a_experiment,
        dataset_root=dataset_root,
        device=device,
        encoder=load_offline_dinov2_encoder(device=device),
    )
    encoder = generator._encoder
    if not isinstance(encoder, FrozenDinoEncoder):
        raise ValueError("a2-1/a2-2 require the frozen DINOv2 encoder")
    destination.mkdir(parents=True, exist_ok=False)
    identities = destination / "identities"
    identities.mkdir()
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    records: list[dict[str, Any]] = []
    target_cache: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, Any] = {}

    for slot in slots:
        identity = f"{slot.defect_type}-{slot.normal_id}-{slot.seed}"
        sample_root = identities / identity
        sample_root.mkdir()
        if slot.normal_id not in target_cache:
            target_record = _lookup_p1_target(manifest, slot.normal_id)
            loaded_target = load_identity_record(
                target_record,
                dataset_root=dataset_root,
                config=experiment,
            )
            structure = estimate_normal_structure(loaded_target, config=experiment)
            target_feat, attn_grid = encode_target_with_cls_attention(encoder, structure.normal.rgb)
            a21 = cls_attention_object_mask(attn_grid, structure.valid)
            a22 = tokencut_object_mask(target_feat.tokens, structure.valid)
            ncut = tokencut_eigenvector_map(target_feat.tokens)
            attn = _upsample_map(attn_grid)
            target_cache[slot.normal_id] = {
                "structure": structure,
                "feat": target_feat,
                "rgb_s": structure.support & structure.valid,
                "a21": a21,
                "a22": a22,
                "attn": attn,
                "ncut": ncut,
            }
        cached = target_cache[slot.normal_id]
        structure = cached["structure"]
        source = generator.select_source(slot.defect_type, slot.seed)
        assert source.anomaly_gt is not None
        source_key = source.record.item_id
        if source_key not in source_cache:
            inpainted = neighbor_fill_source(source.rgb, source.anomaly_gt, source.valid)
            source_cache[source_key] = encoder.encode(inpainted)
        source_feat = source_cache[source_key]
        official, official_tau, _official_sim = _material_proposal_maps(
            source.anomaly_gt,
            source.valid,
            structure,
            source_feat=source_feat,
            target_feat=cached["feat"],
        )
        a21_prop, a21_tau, _a21_sim = proposal_with_object(
            source.anomaly_gt,
            source.valid,
            structure,
            cached["a21"],
            source_feat=source_feat,
            target_feat=cached["feat"],
        )
        a22_prop, a22_tau, _a22_sim = proposal_with_object(
            source.anomaly_gt,
            source.valid,
            structure,
            cached["a22"],
            source_feat=source_feat,
            target_feat=cached["feat"],
        )
        failed = []
        for name, mask in (("a2", official), ("a21", a21_prop), ("a22", a22_prop)):
            if not mask.any() or np.all(mask[structure.valid]) or bool((mask & ~structure.valid).any()):
                failed.append(name)
        normal = np.ascontiguousarray(structure.normal.rgb.copy())
        arrays = {
            "normal": normal,
            "rgb_S": _overlay(normal, cached["rgb_s"], (80, 160, 255), 0.45),
            "a21_attn": _heatmap(cached["attn"], structure.valid),
            "a21_S": _overlay(normal, cached["a21"], (220, 80, 180), 0.45),
            "a22_ncut": _heatmap(cached["ncut"], structure.valid),
            "a22_S": _overlay(normal, cached["a22"], (40, 180, 90), 0.45),
            "a2_prop": _overlay(normal, official, (80, 160, 255), 0.55),
            "a21_prop": _overlay(normal, a21_prop, (220, 80, 180), 0.55),
            "a22_prop": _overlay(normal, a22_prop, (40, 180, 90), 0.55),
        }
        for name, array in arrays.items():
            Image.fromarray(array).save(sample_root / f"{name}.png")
            if name.endswith("_S") or name.endswith("_prop"):
                key = {
                    "rgb_S": cached["rgb_s"],
                    "a21_S": cached["a21"],
                    "a22_S": cached["a22"],
                    "a2_prop": official,
                    "a21_prop": a21_prop,
                    "a22_prop": a22_prop,
                }[name]
                Image.fromarray(_mask_u8(key)).save(sample_root / f"{name}_mask.png")
        flow_rows.append((identity, arrays))
        records.append(
            {
                "identity": identity,
                "defect_type": slot.defect_type,
                "target_id": slot.normal_id,
                "source_id": source.record.item_id,
                "seed": slot.seed,
                "rgb_s_pixels": int(cached["rgb_s"].sum()),
                "a21_s_pixels": int(cached["a21"].sum()),
                "a22_s_pixels": int(cached["a22"].sum()),
                "a2_prop_pixels": int(official.sum()),
                "a21_prop_pixels": int(a21_prop.sum()),
                "a22_prop_pixels": int(a22_prop.sum()),
                "a2_tau": official_tau,
                "a21_tau": a21_tau,
                "a22_tau": a22_tau,
                "failed_arms": failed,
                "failed": bool(failed),
            }
        )

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "not_official_module_a": True,
        "official_a_unchanged": True,
        "identity_count": len(records),
        "failed_rows": int(sum(1 for row in records if row["failed"])),
        "failed_a2": int(sum(1 for row in records if "a2" in row["failed_arms"])),
        "failed_a21": int(sum(1 for row in records if "a21" in row["failed_arms"])),
        "failed_a22": int(sum(1 for row in records if "a22" in row["failed_arms"])),
        "columns": list(FLOW_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "a21": "cls_attention_last4_otsu",
        "a22": "tokencut_ncut_on_patch_tokens",
        "destination": str(destination),
        "visible_defects_passed": None,
    }
    write_json_atomic(destination / "summary.json", summary)
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_a2_support_diagnostic_from_cli(
    *,
    paths: Path,
    a_experiment: Path,
    output: Path,
    cwd: Path,
    device: str | None = None,
) -> dict[str, Any]:
    project = load_config(paths if paths.is_absolute() else cwd / paths)
    return run_a2_support_diagnostic(
        repository_root=cwd,
        dataset_root=project.dataset_root,
        output=output,
        a_experiment=a_experiment if a_experiment.is_absolute() else cwd / a_experiment,
        device=device,
    )


HARD_SEED = 2026082801
HARD_NORMAL_POOL = 4
HARD_SPECS = (
    ("whole_image_texture", "mvtec", "carpet", 4),
    ("object_on_most_edges", "cad3", "Aluminum_Ipad", 4),
    ("multi_instance", "visa", "capsules", 2),
    ("holes_fill_frame", "mvtec", "transistor", 2),
    ("letterbox_wide", "can", "can", 2),
)


@dataclass(frozen=True)
class HardSupportJob:
    case: str
    dataset: str
    category: str
    target: IdentityRecord
    source: IdentityRecord
    dataset_root: Path
    seed: int

    @property
    def identity(self) -> str:
        return f"{self.dataset}/{self.category}/{self.target.item_id}"


def _jobs_from_pack(case: str, pack: ExtraACategory, count: int) -> tuple[HardSupportJob, ...]:
    if count < 1 or count > len(pack.normals):
        raise ValueError(f"{pack.dataset}/{pack.name} needs {count} frozen normals")
    first_type = pack.defect_types[0]
    source = pack.fewshot[first_type][0]
    return tuple(
        HardSupportJob(
            case=case,
            dataset=pack.dataset,
            category=pack.name,
            target=normal,
            source=source,
            dataset_root=pack.root,
            seed=HARD_SEED,
        )
        for normal in pack.normals[:count]
    )


def _can_jobs(source_root: Path, count: int) -> tuple[HardSupportJob, ...]:
    goods = tuple(sorted((source_root / "train" / "good").glob("*.png")))
    bads = tuple(sorted((source_root / "test_public" / "bad").glob("*.png")))
    if not goods or not bads:
        raise ValueError("can hard-case split requires train/good and test_public/bad")
    normal_ids = _even_ids(tuple(_item_id(path) for path in goods), HARD_NORMAL_POOL)[:count]
    fewshot_id = _fewshot_ids(tuple(_item_id(path) for path in bads))[0]
    source = IdentityRecord(
        category="can",
        defect_type="bad",
        item_id=fewshot_id,
        image_relpath=f"test_public/bad/{fewshot_id}.png",
        mask_relpath=f"test_public/ground_truth/bad/{fewshot_id}_mask.png",
        role="fewshot",
        sample_id=f"can/bad/{fewshot_id}",
    )
    jobs = []
    for item_id in normal_ids:
        target = IdentityRecord(
            category="can",
            defect_type="good",
            item_id=item_id,
            image_relpath=f"train/good/{item_id}.png",
            mask_relpath=None,
            role="normal_dev",
            sample_id=f"can/good/{item_id}",
        )
        jobs.append(
            HardSupportJob(
                case="letterbox_wide",
                dataset="can",
                category="can",
                target=target,
                source=source,
                dataset_root=source_root,
                seed=HARD_SEED,
            )
        )
    return tuple(jobs)


def build_hard_support_jobs(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cad3_root: Path,
    can_root: Path,
) -> tuple[HardSupportJob, ...]:
    jobs: list[HardSupportJob] = []
    for case, dataset, category, count in HARD_SPECS:
        if dataset == "mvtec":
            jobs.extend(_jobs_from_pack(case, inventory_mvtec(mvtec_root, category), count))
        elif dataset == "cad3":
            jobs.extend(_jobs_from_pack(case, inventory_cad3(cad3_root, category), count))
        elif dataset == "visa":
            jobs.extend(_jobs_from_pack(case, inventory_visa(visa_root, category), count))
        elif dataset == "can":
            jobs.extend(_can_jobs(can_root, count))
        else:
            raise ValueError(f"unknown hard-case dataset: {dataset}")
    expected = sum(spec[3] for spec in HARD_SPECS)
    if len(jobs) != expected:
        raise ValueError(f"hard support jobs must stay {expected} rows, got {len(jobs)}")
    return tuple(jobs)


def run_a2_support_hard_diagnostic(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cad3_root: Path,
    can_root: Path,
    output: Path,
    device: str | None = None,
    tile: int = 160,
) -> dict[str, Any]:
    destination = _contained_output(output)
    jobs = build_hard_support_jobs(
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        cad3_root=cad3_root,
        can_root=can_root,
    )
    encoder = load_offline_dinov2_encoder(device=device)
    destination.mkdir(parents=True, exist_ok=False)
    identities = destination / "identities"
    identities.mkdir()
    flow_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    records: list[dict[str, Any]] = []
    by_case: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {}
    target_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_cache: dict[tuple[str, str, str], Any] = {}

    for job in jobs:
        cache_key = (job.dataset, job.category, job.target.item_id)
        if cache_key not in target_cache:
            loaded_target = load_extra_identity(job.target, dataset_root=job.dataset_root)
            config = ExtraAStructureConfig(
                category=job.category,
                defect_types=(job.source.defect_type,),
            )
            structure = estimate_normal_structure(loaded_target, config=config)
            target_feat, attn_grid = encode_target_with_cls_attention(
                encoder, structure.normal.rgb
            )
            a21 = cls_attention_object_mask(attn_grid, structure.valid)
            a22 = tokencut_object_mask(target_feat.tokens, structure.valid)
            ncut = tokencut_eigenvector_map(target_feat.tokens)
            attn = _upsample_map(attn_grid)
            target_cache[cache_key] = {
                "structure": structure,
                "feat": target_feat,
                "rgb_s": structure.support & structure.valid,
                "a21": a21,
                "a22": a22,
                "attn": attn,
                "ncut": ncut,
            }
        cached = target_cache[cache_key]
        structure = cached["structure"]
        source_key = (job.dataset, job.category, job.source.item_id)
        if source_key not in source_cache:
            loaded_source = load_extra_identity(job.source, dataset_root=job.dataset_root)
            if loaded_source.anomaly_gt is None:
                raise ValueError(f"hard-case source needs GT: {job.identity}")
            inpainted = neighbor_fill_source(
                loaded_source.rgb, loaded_source.anomaly_gt, loaded_source.valid
            )
            source_cache[source_key] = (loaded_source, encoder.encode(inpainted))
        loaded_source, source_feat = source_cache[source_key]
        official, official_tau, _unused = _material_proposal_maps(
            loaded_source.anomaly_gt,
            loaded_source.valid,
            structure,
            source_feat=source_feat,
            target_feat=cached["feat"],
        )
        a21_prop, a21_tau, _unused_a21 = proposal_with_object(
            loaded_source.anomaly_gt,
            loaded_source.valid,
            structure,
            cached["a21"],
            source_feat=source_feat,
            target_feat=cached["feat"],
        )
        a22_prop, a22_tau, _unused_a22 = proposal_with_object(
            loaded_source.anomaly_gt,
            loaded_source.valid,
            structure,
            cached["a22"],
            source_feat=source_feat,
            target_feat=cached["feat"],
        )
        failed = []
        for name, mask in (("a2", official), ("a21", a21_prop), ("a22", a22_prop)):
            if (
                not mask.any()
                or np.all(mask[structure.valid])
                or bool((mask & ~structure.valid).any())
            ):
                failed.append(name)
        normal = np.ascontiguousarray(structure.normal.rgb.copy())
        arrays = {
            "normal": normal,
            "rgb_S": _overlay(normal, cached["rgb_s"], (80, 160, 255), 0.45),
            "a21_attn": _heatmap(cached["attn"], structure.valid),
            "a21_S": _overlay(normal, cached["a21"], (220, 80, 180), 0.45),
            "a22_ncut": _heatmap(cached["ncut"], structure.valid),
            "a22_S": _overlay(normal, cached["a22"], (40, 180, 90), 0.45),
            "a2_prop": _overlay(normal, official, (80, 160, 255), 0.55),
            "a21_prop": _overlay(normal, a21_prop, (220, 80, 180), 0.55),
            "a22_prop": _overlay(normal, a22_prop, (40, 180, 90), 0.55),
        }
        sample_root = identities / job.identity.replace("/", "_")
        sample_root.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            Image.fromarray(array).save(sample_root / f"{name}.png")
        flow_rows.append((job.identity, arrays))
        by_case.setdefault(job.case, []).append((job.identity, arrays))
        records.append(
            {
                "identity": job.identity,
                "case": job.case,
                "dataset": job.dataset,
                "category": job.category,
                "target_id": job.target.item_id,
                "source_id": job.source.item_id,
                "seed": job.seed,
                "rgb_s_pixels": int(cached["rgb_s"].sum()),
                "a21_s_pixels": int(cached["a21"].sum()),
                "a22_s_pixels": int(cached["a22"].sum()),
                "a2_prop_pixels": int(official.sum()),
                "a21_prop_pixels": int(a21_prop.sum()),
                "a22_prop_pixels": int(a22_prop.sum()),
                "a2_tau": official_tau,
                "a21_tau": a21_tau,
                "a22_tau": a22_tau,
                "valid_pixels": int(structure.valid.sum()),
                "failed_arms": failed,
                "failed": bool(failed),
            }
        )

    _save_sheet(destination / "flow_all.png", flow_rows, FLOW_COLUMNS, tile=tile, gutter=420)
    for case, rows in by_case.items():
        _save_sheet(
            destination / f"flow_{case}.png",
            rows,
            FLOW_COLUMNS,
            tile=tile,
            gutter=420,
        )
    summary = {
        "schema_version": "v06_a_a2_support_hard_v1",
        "diagnostic_only": True,
        "not_official_module_a": True,
        "official_a_unchanged": True,
        "identity_count": len(records),
        "failed_rows": int(sum(1 for row in records if row["failed"])),
        "failed_a2": int(sum(1 for row in records if "a2" in row["failed_arms"])),
        "failed_a21": int(sum(1 for row in records if "a21" in row["failed_arms"])),
        "failed_a22": int(sum(1 for row in records if "a22" in row["failed_arms"])),
        "columns": list(FLOW_COLUMNS),
        "column_meanings": COLUMN_MEANINGS,
        "hard_specs": [list(spec) for spec in HARD_SPECS],
        "selection": (
            "even_ids(train/good, 4) then first N; first lexicographic type; "
            "first floor(n/3) few-shot; seed 2026082801"
        ),
        "destination": str(destination),
        "visible_defects_passed": None,
    }
    write_json_atomic(destination / "summary.json", summary)
    write_json_atomic(
        destination / "identities.json",
        {
            "seed": HARD_SEED,
            "jobs": [
                {
                    "identity": job.identity,
                    "case": job.case,
                    "target_id": job.target.item_id,
                    "source_id": job.source.item_id,
                }
                for job in jobs
            ],
        },
    )
    (destination / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    return summary


def run_a2_support_hard_diagnostic_from_cli(
    *,
    mvtec_root: Path,
    visa_root: Path,
    cad3_root: Path,
    can_root: Path,
    output: Path,
    device: str | None = None,
) -> dict[str, Any]:
    return run_a2_support_hard_diagnostic(
        mvtec_root=mvtec_root,
        visa_root=visa_root,
        cad3_root=cad3_root,
        can_root=can_root,
        output=output,
        device=device,
    )

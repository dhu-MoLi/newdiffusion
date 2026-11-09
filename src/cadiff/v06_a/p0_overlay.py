"""Supplemental, diagnostic-only visual overlay for frozen Module-A P0."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw

from .p0 import _CATEGORY_ORDER, P0CategoryBundle, PreparedP0Pair, _validate_bundle
from .targets import construct_plain_target
from .teacher import (
    TeacherDistribution,
    TeacherGapEvidence,
    inspect_teacher_gap,
    sample_teacher_position,
)

_CELL_SIZE = 128
_GAPS = (32, 48)
_CUMULATIVE_LEVELS = (0.50, 0.80, 0.95)
_COLUMNS = (
    "source + GT edge",
    "gap32 context + GT/frame edges",
    "gap48 context + GT/frame edges",
    "target normal",
    "target + pi32 contours + sampled cross",
    "target + pi48 contours + sampled cross",
    "target + transferred-GT fill/edge",
    "target + p* outer edge + transferred-GT inner edge",
)
_ROUND4_ARTIFACT_SHA256 = {
    "p0_combined_16x8.png": ("5b9406839e10985fc56c1cffe99feea299f8a69ba6a0422330ca457e8c8e31db"),
    "p0_gate.json": "6a21d8724829e97047f802e86d76b3bc7cbcfa13c8de6eba3029e257afb34d8f",
    "pairs/00_hazelnut_crack_000_000.png": (
        "939b02ea6491f0570471996c71e2f2546fdbb2eed302dbf6a8bf1e609136f4a8"
    ),
    "pairs/01_hazelnut_crack_001_260.png": (
        "4a915f833fd7e0136fd387329d538f10187c53aa6d7a69f58c049b791c8f750a"
    ),
    "pairs/02_hazelnut_cut_000_000.png": (
        "e45a801aa57a72f66fc34503e8a5dcd0756a55c2e37fcaea414bcac0c4f3613c"
    ),
    "pairs/03_hazelnut_cut_001_260.png": (
        "f788f90e48f09fb9e341c43dd108c97850fb4f8ec6133114c2d258174e6d6193"
    ),
    "pairs/04_hazelnut_hole_000_000.png": (
        "bf2c3a75bae4862361b320935fdd61338f944f6b6f741015f2727aaf0e882b56"
    ),
    "pairs/05_hazelnut_hole_001_260.png": (
        "20d9d1814b10df24c3e168ad6b7070d2162c378e30b8d606c9c4fe64ebafd9e0"
    ),
    "pairs/06_hazelnut_print_000_000.png": (
        "b59fae4f0efd4e62cfdaf9af6dab5af4b158e9a0d53d96c68a4fffc33d326cc1"
    ),
    "pairs/07_hazelnut_print_001_260.png": (
        "6a025567b92984c21cb6b6781047211ece17571f4b6f745124a58d5dcd9c2077"
    ),
    "pairs/08_cable_bent_wire_000_000.png": (
        "fb2f62c081594ac019701ddbfc6ea71d4f833cc646756cf1a302f8b5c2df8ea0"
    ),
    "pairs/09_cable_bent_wire_001_152.png": (
        "8a12225062c32fcaf094b3204066dcb18166a056065a4a51559240d16177d23b"
    ),
    "pairs/10_cable_cut_inner_insulation_000_000.png": (
        "fae7b4a7792605cfaefd1746b67655e6c78ad3448e9cabc92ae1e0cb4737827c"
    ),
    "pairs/11_cable_cut_inner_insulation_001_152.png": (
        "811a4b350904450251d3e3f020599a6ec8a0c1f8b681867d9dfdd8cd9c150796"
    ),
    "pairs/12_cable_cut_outer_insulation_000_000.png": (
        "db872222e94dfc318dd4827d4dd61d5fa4864de6b39fa4f6afc849bff63f9426"
    ),
    "pairs/13_cable_cut_outer_insulation_001_152.png": (
        "f527ac04acf91a0d34507b7c75401af61126310c6c67b34ac7224eaefb0e1934"
    ),
    "pairs/14_cable_poke_insulation_000_000.png": (
        "a88bfe20ed833b2dee7ae91b0efd295019faeaf46f64c8397f7588516c101c03"
    ),
    "pairs/15_cable_poke_insulation_001_152.png": (
        "2a4c4bed0a971ea24e18c2c93fdc186d4fc90a3bbacb75af034c987d9b73b888"
    ),
}


@dataclass(frozen=True)
class P0SupplementalOverlayArtifact:
    destination: Path
    json_path: Path
    combined_panel_path: Path
    legend_path: Path


def snapshot_round4_artifacts(directory: Path) -> dict[str, str]:
    """Return byte hashes for the frozen JSON/PNG P0 evidence, without writing it."""

    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"round4 directory does not exist: {directory}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".png"}
    )
    if len(files) != 18:
        raise ValueError("frozen round4 must contain exactly 18 JSON/PNG artifacts")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _destination(*, output_root: Path, output_relpath: str, frozen_root: Path) -> Path:
    root = output_root.resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"overlay output root does not exist: {output_root}")
    relative = PurePosixPath(output_relpath)
    if (
        not output_relpath
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != output_relpath
    ):
        raise ValueError("overlay output must be a contained relative path")
    destination = root.joinpath(*relative.parts)
    resolved = destination.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "overlay destination must remain within the overlay output root"
        ) from error
    try:
        resolved.relative_to(frozen_root)
    except ValueError:
        pass
    else:
        raise ValueError("overlay destination must remain outside the frozen round4 directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"overlay refuses to overwrite an existing output: {destination}")
    return destination


def _edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    for y_offset in range(3):
        for x_offset in range(3):
            if (y_offset, x_offset) != (1, 1):
                interior &= padded[y_offset : y_offset + 512, x_offset : x_offset + 512]
    return mask & ~interior


def _thicken(mask: np.ndarray, *, radius: int = 2) -> np.ndarray:
    padded = np.pad(mask, radius, constant_values=False)
    thick = np.zeros_like(mask)
    for y_offset in range(2 * radius + 1):
        for x_offset in range(2 * radius + 1):
            thick |= padded[
                y_offset : y_offset + 512,
                x_offset : x_offset + 512,
            ]
    return thick


def _thick_edge(mask: np.ndarray, *, radius: int = 2) -> np.ndarray:
    return _thicken(_edge(mask), radius=radius)


def _outer_edge(mask: np.ndarray) -> np.ndarray:
    """Return only mask pixels adjacent to canvas-exterior background, excluding holes."""

    padded_mask = np.pad(mask, 1, constant_values=False)
    exterior = np.zeros_like(padded_mask)
    exterior[0, 0] = True
    queue: deque[tuple[int, int]] = deque(((0, 0),))
    height, width = padded_mask.shape
    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and not padded_mask[next_y, next_x]
                and not exterior[next_y, next_x]
            ):
                exterior[next_y, next_x] = True
                queue.append((next_y, next_x))
    adjacent = np.zeros_like(mask)
    for y_offset in range(3):
        for x_offset in range(3):
            adjacent |= exterior[
                y_offset : y_offset + 512,
                x_offset : x_offset + 512,
            ]
    return mask & adjacent


def _paint_edge(rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    result = rgb.copy()
    result[_thick_edge(mask)] = colour
    return result


def _paint_outer_edge(
    rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]
) -> np.ndarray:
    result = rgb.copy()
    result[_thicken(_outer_edge(mask))] = colour
    return result


def _context_cell(prepared: PreparedP0Pair, evidence: TeacherGapEvidence) -> np.ndarray:
    result = np.maximum(prepared.source.rgb // 5, 12)
    result[evidence.used_mask] = prepared.source.rgb[evidence.used_mask]
    result = _paint_edge(result, prepared.source.anomaly_gt, (255, 255, 255))
    left, top, right, bottom = evidence.coarse_bbox
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    outer = (left - 72, top - 72, right + 72, bottom + 72)
    inner_margin = evidence.gap + 1
    inner = (
        left - inner_margin,
        top - inner_margin,
        right + inner_margin,
        bottom + inner_margin,
    )
    draw.rectangle(outer, outline=(0, 255, 255), width=4)
    draw.rectangle(inner, outline=(255, 170, 0), width=4)
    return np.asarray(image)


def _overlay_status(cell: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, 0, _CELL_SIZE - 1, 18), fill=(150, 25, 25))
    for offset in range(-_CELL_SIZE, _CELL_SIZE, 12):
        draw.line((offset, 18, offset + 20, 0), fill=(255, 190, 70), width=2)
    draw.text((2, 3), label, fill=(255, 255, 255))
    return cell


def _cell(rgb: np.ndarray, *, status: str | None = None) -> Image.Image:
    cell = Image.fromarray(rgb).resize((_CELL_SIZE, _CELL_SIZE), resample=Image.Resampling.NEAREST)
    return cell if status is None else _overlay_status(cell, status)


def _cumulative_probability_masks(
    distribution: TeacherDistribution,
) -> dict[float, np.ndarray]:
    ranked = sorted(
        range(distribution.candidates.shape[0]),
        key=lambda index: (
            -float(distribution.probabilities[index]),
            int(distribution.candidates[index, 1]),
            int(distribution.candidates[index, 0]),
        ),
    )
    result: dict[float, np.ndarray] = {}
    for level in _CUMULATIVE_LEVELS:
        mask = np.zeros((512, 512), dtype=bool)
        mass = 0.0
        for index in ranked:
            x, y = distribution.candidates[index]
            left, top = max(int(x) - 8, 0), max(int(y) - 8, 0)
            right, bottom = min(int(x) + 8, 512), min(int(y) + 8, 512)
            mask[top:bottom, left:right] = True
            mass += float(distribution.probabilities[index])
            if mass >= level:
                break
        result[level] = mask
    return result


def _probability_cell(
    prepared: PreparedP0Pair,
    evidence: TeacherGapEvidence,
    *,
    probability_max: float,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    if evidence.distribution is None:
        return prepared.target.rgb.copy(), None
    result = prepared.target.rgb.astype(np.float64)
    distribution = evidence.distribution
    for (x, y), probability in zip(
        distribution.candidates, distribution.probabilities, strict=True
    ):
        ratio = float(np.clip(probability / probability_max, 0.0, 1.0))
        colour = np.asarray((255.0 * ratio, 70.0, 255.0 * (1.0 - ratio)))
        left, top = int(x) - 8, int(y) - 8
        right, bottom = int(x) + 8, int(y) + 8
        result[top:bottom, left:right] = 0.45 * result[top:bottom, left:right] + 0.55 * colour
    rendered = np.rint(result).astype(np.uint8)
    contour_colours = ((255, 230, 0), (0, 255, 255), (255, 255, 255))
    cumulative = _cumulative_probability_masks(distribution)
    for level, colour in reversed(tuple(zip(_CUMULATIVE_LEVELS, contour_colours, strict=True))):
        rendered[_thick_edge(cumulative[level])] = colour
    image = Image.fromarray(rendered)
    draw = ImageDraw.Draw(image)
    sampled = sample_teacher_position(distribution, seed=prepared.pair.seed)
    x, y = sampled
    draw.line((x - 14, y, x + 14, y), fill=(255, 255, 0), width=4)
    draw.line((x, y - 14, x, y + 14), fill=(255, 255, 0), width=4)
    return np.asarray(image), sampled


def _legend() -> Image.Image:
    image = Image.new("RGB", (_CELL_SIZE * 8, 72), (28, 28, 28))
    draw = ImageDraw.Draw(image)
    draw.text(
        (8, 5),
        "Shared scale: pi/p_max; contours: 50% yellow, 80% cyan, 95% white; "
        "cross: seeded sample; INVALID: hatched",
        fill=(255, 255, 255),
    )
    draw.text((8, 27), "Context: GT", fill=(255, 255, 255))
    draw.text((96, 27), "inner frame", fill=(255, 170, 0))
    draw.text((185, 27), "outer frame", fill=(0, 255, 255))
    draw.text((290, 27), "Transferred GT fill/edge", fill=(80, 255, 80))
    draw.text((465, 27), "p* boundary", fill=(0, 255, 255))
    draw.text((565, 27), "transferred-GT boundary", fill=(255, 0, 255))
    draw.text(
        (8, 49),
        "All overlays are diagnostic-only; status remains total-commander-adjudicated FAIL.",
        fill=(255, 190, 70),
    )
    return image


def write_fixed_p0_supplemental_overlay(
    bundles: tuple[P0CategoryBundle, ...],
    *,
    output_root: Path,
    output_relpath: str,
    base_round4_dir: Path,
) -> P0SupplementalOverlayArtifact:
    """Write a new P0 diagnostic sidecar while preserving frozen round4 bytes."""

    if len(bundles) != 2:
        raise ValueError("P0 overlay requires exactly Hazelnut and Cable bundles")
    for bundle, category in zip(bundles, _CATEGORY_ORDER, strict=True):
        _validate_bundle(bundle, expected_category=category)
    frozen_root = base_round4_dir.resolve(strict=True)
    baseline_hashes = snapshot_round4_artifacts(frozen_root)
    if baseline_hashes != _ROUND4_ARTIFACT_SHA256:
        raise ValueError("base P0 artifacts are not the frozen round4 adjudication")
    destination = _destination(
        output_root=output_root,
        output_relpath=output_relpath,
        frozen_root=frozen_root,
    )

    evidence_rows = []
    for bundle in bundles:
        for prepared in bundle.pairs:
            by_gap = {
                gap: inspect_teacher_gap(
                    prepared.source,
                    bundle.atlas,
                    prepared.target,
                    prepared.target_structure,
                    config=bundle.config,
                    identity_manifest=bundle.manifest,
                    gap=gap,
                )
                for gap in _GAPS
            }
            evidence_rows.append((bundle, prepared, by_gap))
    probability_max = max(
        float(evidence.distribution.probabilities.max())
        for _bundle, _prepared, by_gap in evidence_rows
        for evidence in by_gap.values()
        if evidence.distribution is not None
    )

    rows: list[Image.Image] = []
    metadata_rows: list[dict[str, object]] = []
    for index, (bundle, prepared, by_gap) in enumerate(evidence_rows):
        gap32, gap48 = by_gap[32], by_gap[48]
        source = _paint_edge(prepared.source.rgb, prepared.source.anomaly_gt, (255, 255, 255))
        pi32, sample32 = _probability_cell(prepared, gap32, probability_max=probability_max)
        pi48, sample48 = _probability_cell(prepared, gap48, probability_max=probability_max)
        transfer, proposal, transfer_status = _transfer_cells_for_bundle(bundle, prepared, gap32)
        cells = (
            _cell(source),
            _cell(_context_cell(prepared, gap32), status=None if gap32.valid else "G32 INVALID"),
            _cell(_context_cell(prepared, gap48), status=None if gap48.valid else "G48 INVALID"),
            _cell(prepared.target.rgb),
            _cell(pi32, status=None if gap32.valid else "G32 INVALID"),
            _cell(pi48, status=None if gap48.valid else "G48 INVALID"),
            _cell(transfer, status=transfer_status),
            _cell(proposal, status=transfer_status),
        )
        panel = Image.new("RGB", (_CELL_SIZE * 8, _CELL_SIZE))
        for column, cell in enumerate(cells):
            panel.paste(cell, (column * _CELL_SIZE, 0))
        rows.append(panel)
        metadata_rows.append(
            {
                "index": index,
                "category": bundle.config.category,
                "defect_type": prepared.pair.defect_type,
                "seed": prepared.pair.seed,
                "gap32": _gap_metadata(gap32, sample32),
                "gap48": _gap_metadata(gap48, sample48),
                "transfer_status": "VALID" if transfer_status is None else "UNAVAILABLE",
            }
        )

    destination.mkdir(parents=True, exist_ok=False)
    combined = Image.new("RGB", (_CELL_SIZE * 8, _CELL_SIZE * 16))
    for index, row in enumerate(rows):
        combined.paste(row, (0, index * _CELL_SIZE))
    combined_path = destination / "p0_supplemental_overlay_16x8.png"
    combined.save(combined_path, format="PNG", optimize=False, compress_level=9)
    legend_path = destination / "p0_supplemental_overlay_legend.png"
    _legend().save(legend_path, format="PNG", optimize=False, compress_level=9)
    metadata = {
        "schema_version": "v06_a_p0_supplemental_overlay_v1",
        "status": "FAIL",
        "status_source": "total_commander_round4_adjudication",
        "diagnostic_only": True,
        "a1_ctx_targets_materialized": False,
        "a1_ctx_training_authorized": False,
        "pair_count": 16,
        "category_order": list(_CATEGORY_ORDER),
        "columns": list(_COLUMNS),
        "gaps": list(_GAPS),
        "global_probability_max": probability_max,
        "cumulative_probability_levels": list(_CUMULATIVE_LEVELS),
        "round4_artifact_hashes": baseline_hashes,
        "base_round4_relpath": base_round4_dir.as_posix(),
        "pairs": metadata_rows,
    }
    json_path = destination / "p0_supplemental_overlay.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if snapshot_round4_artifacts(frozen_root) != baseline_hashes:
        raise RuntimeError("frozen round4 artifacts changed while writing overlay")
    return P0SupplementalOverlayArtifact(destination, json_path, combined_path, legend_path)


def _transfer_cells_for_bundle(
    bundle: P0CategoryBundle, prepared: PreparedP0Pair, gap32: TeacherGapEvidence
) -> tuple[np.ndarray, np.ndarray, str | None]:
    target = prepared.target.rgb.copy()
    if gap32.distribution is None:
        return target, target.copy(), "TRANSFER UNAVAILABLE"
    sampled = sample_teacher_position(gap32.distribution, seed=prepared.pair.seed)
    sampled_index = int(
        np.flatnonzero(np.all(gap32.distribution.candidates == np.asarray(sampled), axis=1))[0]
    )
    try:
        proposal = construct_plain_target(
            prepared.source,
            bundle.atlas,
            prepared.target,
            prepared.target_structure,
            bundle.config,
            bundle.manifest,
            target_center_override=tuple(gap32.distribution.object_center_overrides[sampled_index]),
        )
    except ValueError:
        return target, target.copy(), "TRANSFER UNAVAILABLE"
    transferred = proposal.transferred_gt_512
    gt_overlay = target.astype(np.float64)
    gt_overlay[transferred] = 0.55 * gt_overlay[transferred] + 0.45 * np.asarray((70, 255, 70))
    gt_overlay = _paint_edge(np.rint(gt_overlay).astype(np.uint8), transferred, (255, 255, 255))
    proposal_overlay = _paint_outer_edge(target, proposal.proposal_512, (0, 255, 255))
    proposal_overlay = _paint_edge(proposal_overlay, transferred, (255, 0, 255))
    return gt_overlay, proposal_overlay, None


def _gap_metadata(
    evidence: TeacherGapEvidence, sample: tuple[int, int] | None
) -> dict[str, object]:
    return {
        "status": "VALID" if evidence.valid else "INVALID",
        "context_fraction": evidence.context_fraction,
        "error": evidence.error,
        "sampled_position": None if sample is None else {"x": sample[0], "y": sample[1]},
    }

"""Fixed 8-column isolation panel. Not a G0/P0-R2 artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy.typing import NDArray

_CELL = 192
_COLUMNS = (
    "source",
    "source_gt",
    "inpaint",
    "target",
    "flow",
    "propagated",
    "proposal",
    "fallback_or_status",
)


def _to_image(rgb: NDArray[np.uint8]) -> Image.Image:
    return Image.fromarray(rgb, mode="RGB").resize((_CELL, _CELL), Image.Resampling.BILINEAR)


def _overlay_mask(rgb: NDArray[np.uint8], mask: NDArray[np.bool_], color: tuple[int, int, int]) -> NDArray[np.uint8]:
    painted = rgb.copy()
    painted[mask] = (0.45 * painted[mask] + 0.55 * np.array(color, dtype=np.float32)).astype(np.uint8)
    return painted


def _flow_preview(
    target_rgb: NDArray[np.uint8],
    flow_xy: NDArray[np.float32],
    valid: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    preview = target_rgb.copy()
    if not valid.any():
        return preview
    height, width = valid.shape
    cell_w = 512 / width
    cell_h = 512 / height
    step = max(1, height // 16)
    for row in range(0, height, step):
        for column in range(0, width, step):
            if not valid[row, column]:
                continue
            start_x = int((column + 0.5) * cell_w + flow_xy[row, column, 0])
            start_y = int((row + 0.5) * cell_h + flow_xy[row, column, 1])
            origin_x = int((column + 0.5) * cell_w)
            origin_y = int((row + 0.5) * cell_h)
            if 0 <= start_x < 512 and 0 <= start_y < 512:
                preview[max(0, start_y - 1) : start_y + 2, max(0, start_x - 1) : start_x + 2] = (
                    255,
                    60,
                    60,
                )
            if 0 <= origin_x < 512 and 0 <= origin_y < 512:
                preview[max(0, origin_y - 1) : origin_y + 2, max(0, origin_x - 1) : origin_x + 2] = (
                    40,
                    200,
                    255,
                )
    return preview


def render_row(
    *,
    source_rgb: NDArray[np.uint8],
    source_gt: NDArray[np.bool_],
    inpaint_rgb: NDArray[np.uint8],
    target_rgb: NDArray[np.uint8],
    flow_xy: NDArray[np.float32],
    match_valid: NDArray[np.bool_],
    propagated: NDArray[np.bool_],
    proposal: NDArray[np.bool_],
    geometry_mask: NDArray[np.bool_],
    collapsed: bool,
    title: str,
) -> Image.Image:
    cells = [
        _to_image(source_rgb),
        _to_image(_overlay_mask(source_rgb, source_gt, (255, 220, 40))),
        _to_image(inpaint_rgb),
        _to_image(target_rgb),
        _to_image(_flow_preview(target_rgb, flow_xy, match_valid)),
        _to_image(_overlay_mask(target_rgb, propagated, (80, 220, 120))),
        _to_image(_overlay_mask(target_rgb, proposal, (80, 160, 255))),
        _to_image(
            _overlay_mask(target_rgb, geometry_mask, (255, 90, 90))
            if collapsed
            else _overlay_mask(target_rgb, proposal, (80, 160, 255))
        ),
    ]
    canvas = Image.new("RGB", (_CELL * 8, _CELL + 28), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 4), title, fill=(240, 240, 240), font=font)
    for index, cell in enumerate(cells):
        canvas.paste(cell, (index * _CELL, 28))
    return canvas


def write_combined_panel(row_images: list[Image.Image], destination: Path) -> None:
    if not row_images:
        raise ValueError("combined panel requires at least one row")
    width = row_images[0].width
    height = sum(image.height for image in row_images) + 22
    canvas = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw.text((8, 4), "  ".join(_COLUMNS), fill=(200, 200, 200), font=font)
    top = 22
    for image in row_images:
        canvas.paste(image, (0, top))
        top += image.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)

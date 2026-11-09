from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class GeometryMeta:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    target_size: int

    def __post_init__(self) -> None:
        dimensions = (
            self.source_width,
            self.source_height,
            self.resized_width,
            self.resized_height,
            self.target_size,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("geometry dimensions must be positive integers")
        padding = (self.pad_left, self.pad_top, self.pad_right, self.pad_bottom)
        if any(type(value) is not int or value < 0 for value in padding):
            raise ValueError("geometry padding must be non-negative integers")
        if self.resized_width + self.pad_left + self.pad_right != self.target_size:
            raise ValueError("horizontal geometry must equal target size")
        if self.resized_height + self.pad_top + self.pad_bottom != self.target_size:
            raise ValueError("vertical geometry must equal target size")


def _validate_size(size: int) -> None:
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")


def letterbox_geometry(width: int, height: int, size: int) -> GeometryMeta:
    return _meta(width, height, size)


def _meta(width: int, height: int, size: int) -> GeometryMeta:
    _validate_size(size)
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    horizontal = size - resized_width
    vertical = size - resized_height
    pad_left = horizontal // 2
    pad_top = vertical // 2
    return GeometryMeta(
        source_width=width,
        source_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=horizontal - pad_left,
        pad_bottom=vertical - pad_top,
        target_size=size,
    )


def resize_pad_image(
    image: Image.Image, size: int
) -> tuple[Image.Image, Image.Image, GeometryMeta]:
    _validate_size(size)
    source = image.convert("RGB")
    meta = _meta(*source.size, size)
    resized = source.resize(
        (meta.resized_width, meta.resized_height), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(resized, (meta.pad_left, meta.pad_top))
    valid = Image.new("L", (size, size), 0)
    valid.paste(
        Image.new("L", (meta.resized_width, meta.resized_height), 255),
        (meta.pad_left, meta.pad_top),
    )
    return canvas, valid, meta


def resize_pad_mask(mask: Image.Image, meta: GeometryMeta) -> Image.Image:
    if mask.size != (meta.source_width, meta.source_height):
        raise ValueError("image and mask source geometry differ")
    resized = mask.convert("L").resize(
        (meta.resized_width, meta.resized_height), Image.Resampling.NEAREST
    )
    binary = resized.point(lambda value: 255 if value >= 128 else 0)
    canvas = Image.new("L", (meta.target_size, meta.target_size), 0)
    canvas.paste(binary, (meta.pad_left, meta.pad_top))
    return canvas


def unletterbox_mask(mask: Image.Image, meta: GeometryMeta) -> Image.Image:
    if mask.size != (meta.target_size, meta.target_size):
        raise ValueError("letterboxed mask must match target size")
    crop = mask.convert("L").crop(
        (
            meta.pad_left,
            meta.pad_top,
            meta.pad_left + meta.resized_width,
            meta.pad_top + meta.resized_height,
        )
    )
    native = crop.resize(
        (meta.source_width, meta.source_height), Image.Resampling.NEAREST
    )
    return native.point(lambda value: 255 if value >= 128 else 0)

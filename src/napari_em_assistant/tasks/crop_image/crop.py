"""Array cropping helpers for 2D images and 3D grayscale stacks."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math

import numpy as np


@dataclass(frozen=True)
class CropTile:
    """Description of one crop tile in ZYX coordinates."""

    index: tuple[int, ...]
    z_range: tuple[int, int] | None
    y_range: tuple[int, int]
    x_range: tuple[int, int]


SMALL_CROP_SOURCE_ELEMENTS = 1_000_000
SMALL_CROP_FRACTION = 0.01


def _validate_axis_bounds(axis: str, start: int, end: int, size: int) -> None:
    if start < 0:
        raise ValueError(f"{axis}_start must be >= 0.")
    if end > size:
        raise ValueError(f"{axis}_end must be <= {size}.")
    if end <= start:
        raise ValueError(f"{axis}_end must be greater than {axis}_start.")


def crop_image(
    image,
    y_range: Sequence[int],
    x_range: Sequence[int],
    z_range: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Crop a 2D image or 3D grayscale stack.

    Ranges are start-inclusive, end-exclusive, matching NumPy slicing.
    2D inputs use ``y_range`` and ``x_range``. 3D inputs use ``z_range``,
    ``y_range``, and ``x_range`` in ZYX order.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("Crop supports 2D images and 3D grayscale stacks.")

    if len(y_range) != 2 or len(x_range) != 2:
        raise ValueError("y_range and x_range must contain start and end values.")

    y_start, y_end = (int(y_range[0]), int(y_range[1]))
    x_start, x_end = (int(x_range[0]), int(x_range[1]))
    y_size = array.shape[-2]
    x_size = array.shape[-1]
    _validate_axis_bounds("y", y_start, y_end, y_size)
    _validate_axis_bounds("x", x_start, x_end, x_size)

    if array.ndim == 2:
        if z_range is not None:
            z_start, z_end = (int(z_range[0]), int(z_range[1]))
            if (z_start, z_end) != (0, 1):
                raise ValueError("z_range is only used for 3D stacks.")
        return array[y_start:y_end, x_start:x_end].copy()

    if z_range is None or len(z_range) != 2:
        raise ValueError("z_range must contain start and end values for 3D stacks.")

    z_start, z_end = (int(z_range[0]), int(z_range[1]))
    _validate_axis_bounds("z", z_start, z_end, array.shape[0])
    return array[z_start:z_end, y_start:y_end, x_start:x_end].copy()


def crop_element_count(
    image_shape: Sequence[int],
    y_range: Sequence[int],
    x_range: Sequence[int],
    z_range: Sequence[int] | None = None,
) -> int:
    """Return the number of pixels/voxels in a crop region."""
    y_count = int(y_range[1]) - int(y_range[0])
    x_count = int(x_range[1]) - int(x_range[0])
    if len(image_shape) == 2:
        return y_count * x_count
    if z_range is None:
        z_count = int(image_shape[0])
    else:
        z_count = int(z_range[1]) - int(z_range[0])
    return z_count * y_count * x_count


def crop_fraction(
    image_shape: Sequence[int],
    y_range: Sequence[int],
    x_range: Sequence[int],
    z_range: Sequence[int] | None = None,
) -> float:
    """Return crop size as a fraction of the source image or stack."""
    source_elements = int(np.prod(tuple(int(value) for value in image_shape)))
    if source_elements <= 0:
        return 0.0
    return crop_element_count(image_shape, y_range, x_range, z_range) / source_elements


def should_warn_small_crop(
    image_shape: Sequence[int],
    y_range: Sequence[int],
    x_range: Sequence[int],
    z_range: Sequence[int] | None = None,
    source_threshold: int = SMALL_CROP_SOURCE_ELEMENTS,
    fraction_threshold: float = SMALL_CROP_FRACTION,
) -> bool:
    """
    Return True when a crop is a very small fraction of a large source image.

    This catches likely coordinate or ROI mistakes without warning for small test
    images or intentionally modest source data.
    """
    source_elements = int(np.prod(tuple(int(value) for value in image_shape)))
    if source_elements < source_threshold:
        return False
    return crop_fraction(image_shape, y_range, x_range, z_range) < fraction_threshold


def roi_bounds_from_shape_data(
    image_shape: Sequence[int],
    shape_data,
) -> tuple[tuple[int, int] | None, tuple[int, int], tuple[int, int]]:
    """
    Convert napari Shapes coordinates to crop bounds.

    The returned bounds are ``z_range, y_range, x_range``. For 2D images,
    ``z_range`` is ``None``. For 3D stacks, 2D XY shapes keep the full Z depth.
    If a 3D shape has a nonzero Z extent, that Z extent is used.
    """
    if len(image_shape) not in (2, 3):
        raise ValueError("Crop supports 2D images and 3D grayscale stacks.")

    points = np.asarray(shape_data, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        raise ValueError("ROI shape must contain at least one 2D coordinate.")
    if not np.isfinite(points).all():
        raise ValueError("ROI shape contains non-finite coordinates.")

    y_values = points[:, -2]
    x_values = points[:, -1]
    y_range = _range_from_float_bounds(y_values.min(), y_values.max(), int(image_shape[-2]), "y")
    x_range = _range_from_float_bounds(x_values.min(), x_values.max(), int(image_shape[-1]), "x")

    if len(image_shape) == 2:
        return None, y_range, x_range

    if points.shape[1] < 3:
        return (0, int(image_shape[0])), y_range, x_range

    z_start = max(0, int(np.floor(points[:, -3].min())))
    z_end = min(int(image_shape[0]), int(np.ceil(points[:, -3].max())))
    if z_end <= z_start:
        z_range = (0, int(image_shape[0]))
    else:
        z_range = (z_start, z_end)
    return z_range, y_range, x_range


def _range_from_float_bounds(
    minimum: float,
    maximum: float,
    size: int,
    axis: str,
) -> tuple[int, int]:
    start = max(0, int(np.floor(minimum)))
    end = min(size, int(np.ceil(maximum)))
    _validate_axis_bounds(axis, start, end, size)
    return start, end


def split_axis_by_parts(size: int, parts: int) -> list[tuple[int, int]]:
    """Split an axis into near-equal start/end ranges."""
    if size <= 0:
        raise ValueError("axis size must be > 0.")
    if parts <= 0:
        raise ValueError("parts must be > 0.")
    if parts > size:
        raise ValueError("parts must be <= axis size.")

    boundaries = [round(index * size / parts) for index in range(parts + 1)]
    return [
        (boundaries[index], boundaries[index + 1])
        for index in range(parts)
        if boundaries[index + 1] > boundaries[index]
    ]


def split_axis_by_size(size: int, tile_size: int) -> list[tuple[int, int]]:
    """Split an axis into fixed-size ranges, with a smaller final edge if needed."""
    if size <= 0:
        raise ValueError("axis size must be > 0.")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0.")
    if tile_size > size:
        raise ValueError("tile_size must be <= axis size.")

    return [
        (start, min(start + tile_size, size))
        for start in range(0, size, tile_size)
    ]


def crop_tiles_by_parts(
    image_shape: Sequence[int],
    y_parts: int,
    x_parts: int,
    z_parts: int = 1,
) -> list[CropTile]:
    """Create crop tiles by splitting each axis into a requested number of parts."""
    if len(image_shape) not in (2, 3):
        raise ValueError("Crop supports 2D images and 3D grayscale stacks.")

    y_ranges = split_axis_by_parts(int(image_shape[-2]), int(y_parts))
    x_ranges = split_axis_by_parts(int(image_shape[-1]), int(x_parts))
    if len(image_shape) == 2:
        return [
            CropTile(index=(y_index, x_index), z_range=None, y_range=y_range, x_range=x_range)
            for y_index, y_range in enumerate(y_ranges)
            for x_index, x_range in enumerate(x_ranges)
        ]

    z_ranges = split_axis_by_parts(int(image_shape[0]), int(z_parts))
    return [
        CropTile(
            index=(z_index, y_index, x_index),
            z_range=z_range,
            y_range=y_range,
            x_range=x_range,
        )
        for z_index, z_range in enumerate(z_ranges)
        for y_index, y_range in enumerate(y_ranges)
        for x_index, x_range in enumerate(x_ranges)
    ]


def tile_grid_for_count(image_shape: Sequence[int], tile_count: int) -> tuple[int, int]:
    """
    Choose a Y/X grid that yields exactly ``tile_count`` tiles.

    The grid keeps tiles close to the source image aspect ratio. For 3D stacks,
    tiling is still based on the Y/X plane and each output keeps the full Z depth.
    """
    if len(image_shape) not in (2, 3):
        raise ValueError("Crop supports 2D images and 3D grayscale stacks.")
    if tile_count <= 0:
        raise ValueError("tile_count must be > 0.")

    y_size = int(image_shape[-2])
    x_size = int(image_shape[-1])
    if tile_count > y_size * x_size:
        raise ValueError("tile_count must be <= number of Y/X pixels.")

    aspect = x_size / y_size
    best_grid = (tile_count, 1)
    best_score = float("inf")
    for y_parts in range(1, int(math.sqrt(tile_count)) + 1):
        if tile_count % y_parts != 0:
            continue
        x_parts = tile_count // y_parts
        for candidate_y, candidate_x in (
            (y_parts, x_parts),
            (x_parts, y_parts),
        ):
            if candidate_y > y_size or candidate_x > x_size:
                continue
            grid_aspect = candidate_x / candidate_y
            score = abs(math.log(grid_aspect / aspect))
            if score < best_score:
                best_score = score
                best_grid = (candidate_y, candidate_x)
    return best_grid


def crop_tiles_by_count(image_shape: Sequence[int], tile_count: int) -> list[CropTile]:
    """
    Create exactly ``tile_count`` near-even Y/X tiles.

    3D stacks keep the full Z depth in every tile.
    """
    y_parts, x_parts = tile_grid_for_count(image_shape, int(tile_count))
    y_ranges = split_axis_by_parts(int(image_shape[-2]), y_parts)
    x_ranges = split_axis_by_parts(int(image_shape[-1]), x_parts)
    if len(image_shape) == 2:
        return [
            CropTile(index=(y_index, x_index), z_range=None, y_range=y_range, x_range=x_range)
            for y_index, y_range in enumerate(y_ranges)
            for x_index, x_range in enumerate(x_ranges)
        ]

    z_range = (0, int(image_shape[0]))
    return [
        CropTile(
            index=(0, y_index, x_index),
            z_range=z_range,
            y_range=y_range,
            x_range=x_range,
        )
        for y_index, y_range in enumerate(y_ranges)
        for x_index, x_range in enumerate(x_ranges)
    ]


def crop_tiles_by_size(
    image_shape: Sequence[int],
    y_size: int,
    x_size: int,
    z_size: int | None = None,
) -> list[CropTile]:
    """
    Create crop tiles by target crop size.

    For 3D stacks, ``z_size=None`` keeps the full Z depth in every crop.
    """
    if len(image_shape) not in (2, 3):
        raise ValueError("Crop supports 2D images and 3D grayscale stacks.")

    y_ranges = split_axis_by_size(int(image_shape[-2]), int(y_size))
    x_ranges = split_axis_by_size(int(image_shape[-1]), int(x_size))
    if len(image_shape) == 2:
        return [
            CropTile(index=(y_index, x_index), z_range=None, y_range=y_range, x_range=x_range)
            for y_index, y_range in enumerate(y_ranges)
            for x_index, x_range in enumerate(x_ranges)
        ]

    if z_size is None:
        z_ranges = [(0, int(image_shape[0]))]
    else:
        z_ranges = split_axis_by_size(int(image_shape[0]), int(z_size))
    return [
        CropTile(
            index=(z_index, y_index, x_index),
            z_range=z_range,
            y_range=y_range,
            x_range=x_range,
        )
        for z_index, z_range in enumerate(z_ranges)
        for y_index, y_range in enumerate(y_ranges)
        for x_index, x_range in enumerate(x_ranges)
    ]


def crop_image_tiles(image, tiles: Sequence[CropTile]) -> list[tuple[CropTile, np.ndarray]]:
    """Apply a list of crop tiles to an image."""
    array = np.asarray(image)
    return [
        (
            tile,
            crop_image(
                array,
                y_range=tile.y_range,
                x_range=tile.x_range,
                z_range=tile.z_range,
            ),
        )
        for tile in tiles
    ]

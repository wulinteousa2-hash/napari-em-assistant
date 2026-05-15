"""Batch crop export for 2D images and 3D grayscale stacks."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import tifffile

from .crop import (
    CropTile,
    crop_image_tiles,
    crop_tiles_by_parts,
    crop_tiles_by_count,
    crop_tiles_by_size,
)


def tile_output_name(stem: str, tile: CropTile) -> str:
    """Build a stable crop filename from tile bounds."""
    y0, y1 = tile.y_range
    x0, x1 = tile.x_range
    if tile.z_range is None:
        return f"{stem}_crop_y{y0}-{y1}_x{x0}-{x1}.tif"
    z0, z1 = tile.z_range
    return f"{stem}_crop_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}.tif"


def crop_tiles_for_shape(
    image_shape,
    mode: str,
    y_parts: int = 1,
    x_parts: int = 1,
    z_parts: int = 1,
    tile_count: int | None = None,
    y_size: int | None = None,
    x_size: int | None = None,
    z_size: int | None = None,
    y_range: tuple[int, int] | None = None,
    x_range: tuple[int, int] | None = None,
    z_range: tuple[int, int] | None = None,
) -> list[CropTile]:
    """Build crop tiles for an image shape and export mode."""
    if mode == "manual":
        if y_range is None or x_range is None:
            raise ValueError("y_range and x_range are required for manual crops.")
        return [
            CropTile(
                index=(0,) if len(image_shape) == 2 else (0, 0, 0),
                z_range=z_range,
                y_range=y_range,
                x_range=x_range,
            )
        ]
    if mode == "parts":
        if tile_count is not None:
            return crop_tiles_by_count(image_shape=image_shape, tile_count=tile_count)
        return crop_tiles_by_parts(
            image_shape=image_shape,
            y_parts=y_parts,
            x_parts=x_parts,
            z_parts=z_parts,
        )
    if mode == "size":
        if y_size is None or x_size is None:
            raise ValueError("y_size and x_size are required for target-size crops.")
        return crop_tiles_by_size(
            image_shape=image_shape,
            y_size=y_size,
            x_size=x_size,
            z_size=z_size,
        )
    raise ValueError("mode must be 'manual', 'parts', or 'size'.")


def _tiff_shape(path: Path):
    with tifffile.TiffFile(path) as tif:
        return tif.series[0].shape


def count_batch_crop_tiles(input_paths: Sequence[Path], **tile_kwargs) -> int:
    """Count output crop tiles for a TIFF batch without loading full image data."""
    return sum(
        len(crop_tiles_for_shape(_tiff_shape(Path(path)), **tile_kwargs))
        for path in input_paths
    )


def write_crop_tiles(
    image,
    output_dir,
    stem: str,
    tiles: Sequence[CropTile],
    progress_callback=None,
    completed: int = 0,
    total: int | None = None,
):
    """Write crop tiles as TIFF files and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written = []
    total_tiles = len(tiles) if total is None else total
    for tile, crop in crop_image_tiles(image, tiles):
        path = output_path / tile_output_name(stem, tile)
        tifffile.imwrite(path, crop)
        written.append(path)
        completed += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "completed": completed,
                    "total": total_tiles,
                    "path": str(path),
                    "tile": tile,
                }
            )
    return written


def export_active_crop_tiles(
    image,
    output_dir,
    stem: str,
    mode: str,
    y_parts: int = 1,
    x_parts: int = 1,
    z_parts: int = 1,
    tile_count: int | None = None,
    y_size: int | None = None,
    x_size: int | None = None,
    z_size: int | None = None,
    y_range: tuple[int, int] | None = None,
    x_range: tuple[int, int] | None = None,
    z_range: tuple[int, int] | None = None,
    progress_callback=None,
    completed: int = 0,
    total: int | None = None,
):
    """Export crop tiles from an in-memory image."""
    tiles = crop_tiles_for_shape(
        image.shape,
        mode=mode,
        y_parts=y_parts,
        x_parts=x_parts,
        z_parts=z_parts,
        tile_count=tile_count,
        y_size=y_size,
        x_size=x_size,
        z_size=z_size,
        y_range=y_range,
        x_range=x_range,
        z_range=z_range,
    )
    return write_crop_tiles(
        image,
        output_dir=output_dir,
        stem=stem,
        tiles=tiles,
        progress_callback=progress_callback,
        completed=completed,
        total=len(tiles) if total is None else total,
    )


def batch_crop_tiff_files(
    input_paths: Sequence[Path],
    output_dir,
    mode: str,
    y_parts: int = 1,
    x_parts: int = 1,
    z_parts: int = 1,
    tile_count: int | None = None,
    y_size: int | None = None,
    x_size: int | None = None,
    z_size: int | None = None,
    y_range: tuple[int, int] | None = None,
    x_range: tuple[int, int] | None = None,
    z_range: tuple[int, int] | None = None,
    progress_callback=None,
):
    """Crop one or more TIFF files into tile TIFF outputs."""
    written = []
    tile_kwargs = {
        "mode": mode,
        "y_parts": y_parts,
        "x_parts": x_parts,
        "z_parts": z_parts,
        "tile_count": tile_count,
        "y_size": y_size,
        "x_size": x_size,
        "z_size": z_size,
        "y_range": y_range,
        "x_range": x_range,
        "z_range": z_range,
    }
    total = count_batch_crop_tiles(input_paths, **tile_kwargs)
    completed = 0
    for input_path in input_paths:
        path = Path(input_path)
        image = tifffile.imread(path)
        paths = export_active_crop_tiles(
            image,
            output_dir=output_dir,
            stem=path.stem,
            progress_callback=progress_callback,
            completed=completed,
            total=total,
            **tile_kwargs,
        )
        written.extend(paths)
        completed += len(paths)
    return written

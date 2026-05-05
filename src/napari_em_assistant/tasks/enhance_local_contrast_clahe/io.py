"""TIFF IO helpers for CLAHE batch processing."""

from pathlib import Path

import tifffile


def read_image(path):
    """Read an image from a TIFF path."""
    return tifffile.imread(Path(path))


def write_image(path, image):
    """Write an image to a TIFF path, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, image)
    return output_path


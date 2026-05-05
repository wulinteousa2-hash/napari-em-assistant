"""Batch processing support for the CLAHE task."""

from pathlib import Path

from .gpu_clahe import apply_gpu_clahe
from .imagej_clahe import apply_imagej_clahe
from .io import read_image, write_image
from .opencv_clahe import apply_opencv_clahe


_BACKENDS = {
    "imagej_reference": apply_imagej_clahe,
    "opencv_cpu": apply_opencv_clahe,
    "gpu": apply_gpu_clahe,
}


def batch_apply_clahe(
    input_paths,
    output_dir,
    backend: str,
    block_size: int,
    histogram_bins: int,
    maximum_slope: float,
    fast: bool,
    overwrite: bool = False,
):
    """
    Batch process 2D grayscale TIFF files.

    Save outputs as <stem>_clahe.tif.
    """
    results = {"processed": [], "skipped": [], "failed": []}
    if backend not in _BACKENDS:
        raise ValueError("backend must be one of imagej_reference, opencv_cpu, or gpu.")

    output_root = Path(output_dir)
    apply_backend = _BACKENDS[backend]

    for input_path in input_paths:
        source = Path(input_path)
        destination = output_root / f"{source.stem}_clahe.tif"
        if destination.exists() and not overwrite:
            results["skipped"].append(str(source))
            continue

        try:
            image = read_image(source)
            output = apply_backend(
                image,
                block_size=block_size,
                histogram_bins=histogram_bins,
                maximum_slope=maximum_slope,
                fast=fast,
            )
            write_image(destination, output)
            results["processed"].append(str(destination))
        except Exception as exc:
            results["failed"].append({"path": str(source), "error": str(exc)})

    return results


"""Batch processing support for the CLAHE task."""

from pathlib import Path

from .gpu_clahe import apply_gpu_clahe, apply_gpu_cupy_clahe, is_gpu_cupy_available
from .imagej_clahe import apply_imagej_clahe
from .io import read_image, write_image
from .opencv_clahe import apply_opencv_clahe


_BACKENDS = {
    "imagej_reference": apply_imagej_clahe,
    "opencv_cpu": apply_opencv_clahe,
    "gpu_cupy": apply_gpu_cupy_clahe,
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
    progress_callback=None,
):
    """
    Batch process 2D grayscale TIFF files.

    Save outputs as <stem>_clahe.tif.
    """
    results = {"processed": [], "skipped": [], "failed": []}
    if backend not in _BACKENDS:
        raise ValueError(
            "backend must be one of imagej_reference, opencv_cpu, gpu_cupy, or gpu."
        )

    output_root = Path(output_dir)
    apply_backend = _BACKENDS[backend]
    effective_backend = backend
    if backend == "gpu_cupy" and not is_gpu_cupy_available():
        effective_backend = "opencv_cpu_fallback"

    for input_path in input_paths:
        source = Path(input_path)
        destination = output_root / f"{source.stem}_clahe.tif"
        if progress_callback is not None:
            progress_callback(
                {
                    "path": str(source),
                    "output": str(destination),
                    "loaded": False,
                    "processed": False,
                    "status": "queued",
                    "backend": effective_backend,
                }
            )
        if destination.exists() and not overwrite:
            results["skipped"].append(str(source))
            if progress_callback is not None:
                progress_callback(
                    {
                        "path": str(source),
                        "output": str(destination),
                        "loaded": False,
                        "processed": False,
                        "status": "skipped",
                        "backend": effective_backend,
                    }
                )
            continue

        try:
            image = read_image(source)
            if progress_callback is not None:
                progress_callback(
                    {
                        "path": str(source),
                        "output": str(destination),
                        "loaded": True,
                        "processed": False,
                        "status": "loaded",
                        "backend": effective_backend,
                    }
                )
            output = apply_backend(
                image,
                block_size=block_size,
                histogram_bins=histogram_bins,
                maximum_slope=maximum_slope,
                fast=fast,
            )
            write_image(destination, output)
            results["processed"].append(str(destination))
            if progress_callback is not None:
                progress_callback(
                    {
                        "path": str(source),
                        "output": str(destination),
                        "loaded": True,
                        "processed": True,
                        "status": "processed",
                        "backend": effective_backend,
                    }
                )
        except Exception as exc:
            results["failed"].append({"path": str(source), "error": str(exc)})
            if progress_callback is not None:
                progress_callback(
                    {
                        "path": str(source),
                        "output": str(destination),
                        "loaded": False,
                        "processed": False,
                        "status": f"failed: {exc}",
                        "backend": effective_backend,
                    }
                )

    return results

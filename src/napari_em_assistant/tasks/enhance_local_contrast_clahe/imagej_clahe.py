"""ImageJ/Fiji-targeted Enhance Local Contrast (CLAHE) API."""

import warnings

from .opencv_clahe import apply_opencv_clahe


def apply_imagej_clahe(
    image,
    block_size: int = 127,
    histogram_bins: int = 256,
    maximum_slope: float = 3.0,
    mask=None,
    fast: bool = True,
):
    """
    Target backend for ImageJ/Fiji-compatible CLAHE behavior.

    For now, this delegates to the OpenCV CPU approximation and emits a clear
    compatibility warning. Preserve shape and dtype where possible. Supports 2D
    grayscale uint8, uint16, and float32. Raises ValueError for unsupported
    image dimensions or parameters.
    """
    warnings.warn(
        "The imagej_reference CLAHE backend currently delegates to OpenCV and "
        "is an approximate ImageJ/Fiji-compatible path until validated against "
        "Fiji output.",
        RuntimeWarning,
        stacklevel=2,
    )
    return apply_opencv_clahe(
        image=image,
        block_size=block_size,
        histogram_bins=histogram_bins,
        maximum_slope=maximum_slope,
        mask=mask,
        fast=fast,
    )


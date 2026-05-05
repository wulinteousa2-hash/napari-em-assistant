"""GPU-oriented CLAHE backend.

The public ``apply_gpu_clahe`` function remains a developer stub for future
validated GPU work. The user-facing GPU backend is ``gpu_cupy``: an optional
batch-oriented approximation that uses CuPy when available and falls back to the
OpenCV CPU approximation otherwise.
"""

from __future__ import annotations

import math

import numpy as np

from .opencv_clahe import apply_opencv_clahe


def apply_gpu_clahe(*args, **kwargs):
    raise NotImplementedError(
        "GPU CLAHE backend is planned but not implemented yet."
    )


def _load_cupy():
    try:
        import cupy as cp
    except Exception:
        return None

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        cp.cuda.Device(0).use()
        cp.cuda.runtime.free(0)
    except Exception:
        return None

    return cp

def is_gpu_cupy_available() -> bool:
    """Return True when CuPy can see at least one CUDA device."""
    return _load_cupy() is not None


def has_opencv_cuda_clahe() -> bool:
    try:
        import cv2

        return (
            hasattr(cv2, "cuda")
            and cv2.cuda.getCudaEnabledDeviceCount() > 0
            and hasattr(cv2.cuda, "createCLAHE")
            and "NVIDIA CUDA" in cv2.getBuildInformation()
        )
    except Exception:
        return False


def gpu_status_summary() -> dict[str, bool]:
    """Return GPU availability flags used by the widget status indicator."""
    return {
        "cupy_cuda": is_gpu_cupy_available(),
        "opencv_cuda_clahe": has_opencv_cuda_clahe(),
    }


def _as_uint_working_array(cp, image: np.ndarray):
    if image.dtype == np.uint8:
        return cp.asarray(image), np.uint8, 0.0, 255.0
    if image.dtype == np.uint16:
        return cp.asarray(image), np.uint16, 0.0, 65535.0
    if image.dtype == np.float32:
        finite = np.isfinite(image)
        if not finite.any():
            return cp.zeros(image.shape, dtype=cp.uint16), np.float32, 0.0, 1.0
        min_value = float(np.nanmin(image[finite]))
        max_value = float(np.nanmax(image[finite]))
        if math.isclose(min_value, max_value):
            return cp.zeros(image.shape, dtype=cp.uint16), np.float32, min_value, max_value
        scaled = (np.nan_to_num(image, nan=min_value) - min_value) / (
            max_value - min_value
        )
        scaled = np.clip(scaled, 0.0, 1.0)
        return cp.asarray(np.round(scaled * 65535).astype(np.uint16)), np.float32, min_value, max_value
    raise ValueError("CLAHE supports uint8, uint16, and float32 images.")


def _restore_dtype(cp, image, original_dtype, min_value, max_value):
    if original_dtype == np.float32:
        if math.isclose(min_value, max_value):
            return np.full(image.shape, min_value, dtype=np.float32)
        restored = image.astype(cp.float32) / 65535.0
        restored = restored * (max_value - min_value) + min_value
        return cp.asnumpy(restored).astype(np.float32, copy=False)
    return cp.asnumpy(image).astype(original_dtype, copy=False)


def _apply_cupy_nearest_tile_clahe(
    image,
    block_size: int,
    histogram_bins: int,
    maximum_slope: float,
    mask=None,
):
    cp = _load_cupy()
    if cp is None:
        raise RuntimeError("CuPy with a CUDA device is not available.")

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("CuPy CLAHE processes 2D slices; pass 3D stacks through apply_gpu_cupy_clahe.")
    if block_size < 3:
        raise ValueError("block_size must be >= 3.")
    if histogram_bins not in {128, 256, 512, 1024}:
        raise ValueError("histogram_bins must be one of 128, 256, 512, or 1024.")
    if maximum_slope <= 0:
        raise ValueError("maximum_slope must be > 0.")
    if mask is not None and np.asarray(mask).shape != array.shape:
        raise ValueError("mask must match the image shape.")

    source, original_dtype, min_value, max_value = _as_uint_working_array(cp, array)
    out = cp.empty_like(source)
    height, width = source.shape
    value_max = 255 if source.dtype == cp.uint8 else 65535
    bin_scale = histogram_bins / float(value_max + 1)

    for y0 in range(0, height, block_size):
        y1 = min(y0 + block_size, height)
        for x0 in range(0, width, block_size):
            x1 = min(x0 + block_size, width)
            tile = source[y0:y1, x0:x1]
            tile_bins = cp.minimum(
                (tile.astype(cp.float32) * bin_scale).astype(cp.int32),
                histogram_bins - 1,
            )
            hist = cp.bincount(tile_bins.ravel(), minlength=histogram_bins).astype(cp.float32)
            clip_limit = max(1.0, maximum_slope * tile.size / histogram_bins)
            excess = cp.maximum(hist - clip_limit, 0.0)
            hist = cp.minimum(hist, clip_limit)
            hist = hist + cp.sum(excess) / histogram_bins
            cdf = cp.cumsum(hist)
            cdf = cdf / cdf[-1]
            lut = cp.clip(cp.round(cdf * value_max), 0, value_max).astype(source.dtype)
            out[y0:y1, x0:x1] = lut[tile_bins]

    result = _restore_dtype(cp, out, original_dtype, min_value, max_value)
    if mask is not None:
        mask_array = np.asarray(mask).astype(bool)
        result = np.where(mask_array, result, array).astype(array.dtype, copy=False)
    return result


def apply_gpu_cupy_clahe(
    image,
    block_size: int = 127,
    histogram_bins: int = 256,
    maximum_slope: float = 3.0,
    mask=None,
    fast: bool = True,
    fallback_to_cpu: bool = True,
):
    """
    Optional CuPy CLAHE approximation for batch processing.

    If CuPy or a CUDA device is unavailable, this falls back to the OpenCV CPU
    approximation by default. The CuPy path currently uses nearest-tile mapping
    and is intended as a fast batch backend, not a Fiji-validated reference.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        mask_array = None if mask is None else np.asarray(mask)
        return np.stack(
            [
                apply_gpu_cupy_clahe(
                    slice_image,
                    block_size=block_size,
                    histogram_bins=histogram_bins,
                    maximum_slope=maximum_slope,
                    mask=None if mask_array is None else mask_array[index],
                    fast=fast,
                    fallback_to_cpu=fallback_to_cpu,
                )
                for index, slice_image in enumerate(array)
            ],
            axis=0,
        ).astype(array.dtype, copy=False)

    try:
        return _apply_cupy_nearest_tile_clahe(
            image=image,
            block_size=block_size,
            histogram_bins=histogram_bins,
            maximum_slope=maximum_slope,
            mask=mask,
        )
    except RuntimeError:
        if not fallback_to_cpu:
            raise
        return apply_opencv_clahe(
            image=image,
            block_size=block_size,
            histogram_bins=histogram_bins,
            maximum_slope=maximum_slope,
            mask=mask,
            fast=fast,
        )

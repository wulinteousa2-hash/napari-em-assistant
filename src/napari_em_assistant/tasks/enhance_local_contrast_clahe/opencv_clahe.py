"""OpenCV CPU approximation of ImageJ/Fiji Enhance Local Contrast (CLAHE)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_ALLOWED_HISTOGRAM_BINS = {128, 256, 512, 1024}


def _validate_common(image, block_size, histogram_bins, maximum_slope, mask=None):
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("CLAHE supports 2D images and 3D grayscale stacks.")
    if block_size < 3:
        raise ValueError("block_size must be >= 3.")
    if histogram_bins not in _ALLOWED_HISTOGRAM_BINS:
        raise ValueError("histogram_bins must be one of 128, 256, 512, or 1024.")
    if maximum_slope <= 0:
        raise ValueError("maximum_slope must be > 0.")
    if mask is not None and np.asarray(mask).shape != array.shape:
        raise ValueError("mask must match the image shape.")
    if array.dtype not in (np.uint8, np.uint16, np.float32):
        raise ValueError("CLAHE supports uint8, uint16, and float32 images.")
    return array


def _normalize_float32_to_uint16(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint16), 0.0, 1.0

    min_value = float(np.nanmin(image[finite]))
    max_value = float(np.nanmax(image[finite]))
    if math.isclose(min_value, max_value):
        return np.zeros(image.shape, dtype=np.uint16), min_value, max_value

    scaled = (np.nan_to_num(image, nan=min_value) - min_value) / (
        max_value - min_value
    )
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * np.iinfo(np.uint16).max).astype(np.uint16), min_value, max_value


def _restore_float32(image: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    if math.isclose(min_value, max_value):
        return np.full(image.shape, min_value, dtype=np.float32)
    restored = image.astype(np.float32) / np.iinfo(np.uint16).max
    return (restored * (max_value - min_value) + min_value).astype(np.float32)


def imagej_params_to_opencv(
    block_size: int,
    histogram_bins: int,
    maximum_slope: float,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """
    Convert ImageJ-style CLAHE parameters to OpenCV parameters.

    This mapping is approximate until validated against ImageJ/Fiji output.
    ImageJ uses a block size in pixels, while OpenCV expects a tile grid count.
    """
    if block_size < 3:
        raise ValueError("block_size must be >= 3.")
    if histogram_bins not in _ALLOWED_HISTOGRAM_BINS:
        raise ValueError("histogram_bins must be one of 128, 256, 512, or 1024.")
    if maximum_slope <= 0:
        raise ValueError("maximum_slope must be > 0.")
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a 2D shape.")

    height, width = image_shape
    tiles_y = max(1, int(math.ceil(height / block_size)))
    tiles_x = max(1, int(math.ceil(width / block_size)))

    # OpenCV clipLimit is not ImageJ's maximum slope. This conservative mapping
    # keeps the user-facing parameter stable while marking the backend approximate.
    clip_limit = float(maximum_slope)
    if histogram_bins != 256:
        clip_limit *= histogram_bins / 256.0

    return {
        "clipLimit": clip_limit,
        "tileGridSize": (tiles_x, tiles_y),
        "compatibility": "approximate_imagej_parameter_mapping",
    }


def apply_opencv_clahe(
    image,
    block_size: int = 127,
    histogram_bins: int = 256,
    maximum_slope: float = 3.0,
    mask=None,
    fast: bool = True,
):
    """
    Fast CPU CLAHE approximation using cv2.createCLAHE.

    Preserve shape and dtype where possible.
    """
    array = _validate_common(image, block_size, histogram_bins, maximum_slope, mask)
    if array.ndim == 3:
        mask_array = None if mask is None else np.asarray(mask)
        return np.stack(
            [
                apply_opencv_clahe(
                    slice_image,
                    block_size=block_size,
                    histogram_bins=histogram_bins,
                    maximum_slope=maximum_slope,
                    mask=None if mask_array is None else mask_array[index],
                    fast=fast,
                )
                for index, slice_image in enumerate(array)
            ],
            axis=0,
        ).astype(array.dtype, copy=False)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for the opencv_cpu CLAHE backend."
        ) from exc

    params = imagej_params_to_opencv(
        block_size=block_size,
        histogram_bins=histogram_bins,
        maximum_slope=maximum_slope,
        image_shape=array.shape,
    )
    clahe = cv2.createCLAHE(
        clipLimit=params["clipLimit"],
        tileGridSize=params["tileGridSize"],
    )

    if array.dtype == np.float32:
        working, min_value, max_value = _normalize_float32_to_uint16(array)
        result = _restore_float32(clahe.apply(working), min_value, max_value)
    else:
        result = clahe.apply(array)
        if result.dtype != array.dtype:
            result = result.astype(array.dtype, copy=False)

    if mask is not None:
        mask_array = np.asarray(mask).astype(bool)
        result = np.where(mask_array, result, array).astype(array.dtype, copy=False)

    return result

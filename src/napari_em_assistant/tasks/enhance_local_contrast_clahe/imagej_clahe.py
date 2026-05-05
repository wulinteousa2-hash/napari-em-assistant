"""ImageJ/Fiji-targeted Enhance Local Contrast (CLAHE) API.

This module ports the 2D grayscale parts of Fiji's
``mpicbg.ij.clahe.PlugIn`` implementation from the Fiji mpicbg 1.6.0 package.
The Fiji menu command is ``Process > Enhance Local Contrast (CLAHE)`` and calls
``mpicbg.ij.clahe.PlugIn``. In that dialog, ``blocksize`` is displayed as
``2 * blockRadius + 1`` and ``histogram bins`` is displayed as ``bins + 1``.
"""

from __future__ import annotations

import math

import numpy as np


_ALLOWED_HISTOGRAM_BINS = {128, 256, 512, 1024}


def _round_pos(value):
    return np.floor(np.asarray(value, dtype=np.float64) + 0.5).astype(np.int64)


def _validate_inputs(image, block_size, histogram_bins, maximum_slope, mask):
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


def _to_fiji_byte_image(image):
    if image.dtype == np.uint8:
        return image.copy()

    finite = np.isfinite(image) if image.dtype == np.float32 else np.ones(image.shape, bool)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    min_value = float(np.nanmin(image[finite]))
    max_value = float(np.nanmax(image[finite]))
    if math.isclose(min_value, max_value):
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (np.nan_to_num(image, nan=min_value) - min_value) / (
        max_value - min_value
    )
    return np.clip(_round_pos(np.clip(scaled, 0.0, 1.0) * 255.0), 0, 255).astype(
        np.uint8
    )


def _mask_to_byte(mask, shape):
    if mask is None:
        return np.full(shape, 255, dtype=np.uint8)
    mask_array = np.asarray(mask)
    if mask_array.dtype == np.uint8:
        return mask_array.copy()
    if mask_array.dtype == bool:
        return mask_array.astype(np.uint8) * 255
    finite = np.isfinite(mask_array)
    if not finite.any():
        return np.zeros(shape, dtype=np.uint8)
    min_value = float(np.nanmin(mask_array[finite]))
    max_value = float(np.nanmax(mask_array[finite]))
    if math.isclose(min_value, max_value):
        return np.where(mask_array > 0, 255, 0).astype(np.uint8)
    scaled = (np.nan_to_num(mask_array, nan=min_value) - min_value) / (
        max_value - min_value
    )
    return np.clip(_round_pos(np.clip(scaled, 0.0, 1.0) * 255.0), 0, 255).astype(
        np.uint8
    )


def _clip_histogram(histogram, limit):
    clipped = histogram.astype(np.int64, copy=True)
    previous_excess = 0
    while True:
        excess = 0
        for index in range(clipped.size):
            delta = clipped[index] - limit
            if delta > 0:
                excess += int(delta)
                clipped[index] = limit

        clipped += excess // clipped.size
        remainder = excess % clipped.size
        if remainder:
            step = (clipped.size - 1) // remainder
            index = step // 2
            while index < clipped.size:
                clipped[index] += 1
                index += step

        if excess == previous_excess:
            return clipped
        previous_excess = excess


def _create_transfer(histogram, limit):
    clipped = _clip_histogram(histogram, limit)
    first = clipped.size - 1
    for index in range(clipped.size - 1):
        if clipped[index] != 0:
            first = index

    cumulative = 0
    for index in range(first, clipped.size):
        cumulative += int(clipped[index])
        clipped[index] = cumulative

    low = int(clipped[first])
    high = int(clipped[-1])
    if high == low:
        return np.zeros(clipped.size, dtype=np.float32)
    return ((clipped - low) / float(high - low)).astype(np.float32)


def _histogram_at(byte_image, radius, bins_internal, x, y):
    y0 = max(0, y - radius)
    y1 = min(byte_image.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(byte_image.shape[1], x + radius + 1)
    values = _round_pos(byte_image[y0:y1, x0:x1].astype(np.float32) / 255.0 * bins_internal)
    values = np.clip(values, 0, bins_internal).astype(np.int64, copy=False)
    return np.bincount(values.ravel(), minlength=bins_internal + 1).astype(np.int64)


def _grid_centers(length, block_size, radius):
    count = length // block_size
    remainder = length - count * block_size
    if remainder == 0:
        return np.array(
            [index * block_size + radius + 1 for index in range(count)],
            dtype=np.int64,
        )
    if remainder == 1:
        values = [index * block_size + radius + 1 for index in range(count)]
        values.append(length - radius - 1)
        return np.array(values, dtype=np.int64)
    values = [radius + 1]
    values.extend(
        index * block_size + radius + 1 + remainder // 2
        for index in range(count)
    )
    values.append(length - radius - 1)
    return np.array(values, dtype=np.int64)


def _transfer_at(byte_image, radius, bins_internal, limit, x, y):
    return _create_transfer(
        _histogram_at(byte_image, radius, bins_internal, x, y),
        limit,
    )


def _fiji_fast_byte_clahe(byte_image, radius, bins_internal, maximum_slope):
    height, width = byte_image.shape
    block_size = 2 * radius + 1
    limit = int(maximum_slope * block_size * block_size / bins_internal + 0.5)
    x_centers = _grid_centers(width, block_size, radius)
    y_centers = _grid_centers(height, block_size, radius)
    output = byte_image.copy()

    for y_block in range(len(y_centers) + 1):
        y0_index = max(0, y_block - 1)
        y1_index = min(len(y_centers) - 1, y_block)
        y_distance = int(y_centers[y1_index] - y_centers[y0_index])
        top_left = _transfer_at(
            byte_image, radius, bins_internal, limit, int(x_centers[0]), int(y_centers[y0_index])
        )
        bottom_left = top_left
        if y0_index != y1_index:
            bottom_left = _transfer_at(
                byte_image,
                radius,
                bins_internal,
                limit,
                int(x_centers[0]),
                int(y_centers[y1_index]),
            )

        y_start = 0 if y_block == 0 else int(y_centers[y0_index])
        y_end = int(y_centers[y1_index]) if y_block < len(y_centers) else height - 1

        for x_block in range(len(x_centers) + 1):
            x0_index = max(0, x_block - 1)
            x1_index = min(len(x_centers) - 1, x_block)
            x_distance = int(x_centers[x1_index] - x_centers[x0_index])

            top_right = top_left
            bottom_right = bottom_left
            if x0_index != x1_index:
                top_right = _transfer_at(
                    byte_image,
                    radius,
                    bins_internal,
                    limit,
                    int(x_centers[x1_index]),
                    int(y_centers[y0_index]),
                )
                if y0_index == y1_index:
                    bottom_right = top_right
                else:
                    bottom_right = _transfer_at(
                        byte_image,
                        radius,
                        bins_internal,
                        limit,
                        int(x_centers[x1_index]),
                        int(y_centers[y1_index]),
                    )

            x_start = 0 if x_block == 0 else int(x_centers[x0_index])
            x_end = int(x_centers[x1_index]) if x_block < len(x_centers) else width - 1

            for y in range(y_start, y_end):
                y_weight = (float(y_centers[y1_index] - y) / y_distance) if y_distance else 1.0
                row = byte_image[y]
                for x in range(x_start, x_end):
                    x_weight = (float(x_centers[x1_index] - x) / x_distance) if x_distance else 1.0
                    value_bin = int(_round_pos(row[x] / 255.0 * bins_internal))
                    value_bin = max(0, min(bins_internal, value_bin))

                    v00 = float(top_left[value_bin])
                    v10 = float(top_right[value_bin])
                    v01 = float(bottom_left[value_bin])
                    v11 = float(bottom_right[value_bin])
                    if x0_index == x1_index:
                        top = v00
                        bottom = v01
                    else:
                        top = x_weight * v00 + (1.0 - x_weight) * v10
                        bottom = x_weight * v01 + (1.0 - x_weight) * v11
                    if y0_index == y1_index:
                        mapped = top
                    else:
                        mapped = y_weight * top + (1.0 - y_weight) * bottom
                    output[y, x] = np.uint8(max(0, min(255, int(mapped * 255.0 + 0.5))))

            top_left = top_right
            bottom_left = bottom_right

    return output


def _transfer_value(value_bin, histogram, clipped, limit):
    clipped[:] = _clip_histogram(histogram, limit)
    first = clipped.size - 1
    for index in range(clipped.size - 1):
        if clipped[index] != 0:
            first = index

    below = 0
    for index in range(first, value_bin + 1):
        below += int(clipped[index])
    total = below
    for index in range(value_bin + 1, clipped.size):
        total += int(clipped[index])
    low = int(clipped[first])
    if total == low:
        return 0.0
    return (below - low) / float(total - low)


def _fiji_exact_byte_clahe(byte_image, radius, bins_internal, maximum_slope, mask_byte):
    height, width = byte_image.shape
    output = byte_image.copy()
    clipped = np.zeros(bins_internal + 1, dtype=np.int64)

    for y in range(height):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        box_height = y1 - y0
        x0 = max(0, -radius - 1)
        x1 = min(width - 1, radius)
        histogram = np.zeros(bins_internal + 1, dtype=np.int64)
        if x1 > x0:
            values = _round_pos(
                byte_image[y0:y1, x0:x1].astype(np.float32) / 255.0 * bins_internal
            )
            values = np.clip(values, 0, bins_internal).astype(np.int64, copy=False)
            histogram += np.bincount(values.ravel(), minlength=bins_internal + 1)

        for x in range(width):
            value_bin = int(_round_pos(byte_image[y, x] / 255.0 * bins_internal))
            value_bin = max(0, min(bins_internal, value_bin))
            x_window_min = max(0, x - radius)
            x_window_max = min(width, x + radius + 1)
            box_size = box_height * (x_window_max - x_window_min)
            if mask_byte is None:
                limit = int(maximum_slope * box_size / bins_internal + 0.5)
            else:
                local_weight = 1.0 + (
                    mask_byte[y, x] / 255.0 * (maximum_slope - 1.0)
                )
                limit = int(local_weight * box_size / bins_internal + 0.5)

            remove_x = x - radius - 1
            if remove_x >= 0:
                values = _round_pos(
                    byte_image[y0:y1, remove_x].astype(np.float32)
                    / 255.0
                    * bins_internal
                )
                values = np.clip(values, 0, bins_internal).astype(np.int64, copy=False)
                histogram -= np.bincount(values, minlength=bins_internal + 1)
            add_x = x + radius
            if add_x < width:
                values = _round_pos(
                    byte_image[y0:y1, add_x].astype(np.float32) / 255.0 * bins_internal
                )
                values = np.clip(values, 0, bins_internal).astype(np.int64, copy=False)
                histogram += np.bincount(values, minlength=bins_internal + 1)

            mapped = _transfer_value(value_bin, histogram, clipped, max(1, limit))
            output[y, x] = np.uint8(max(0, min(255, int(mapped * 255.0 + 0.5))))
    return output


def _apply_fiji_byte_result(original, source_byte, result_byte, mask_byte):
    mask_fraction = mask_byte.astype(np.float32) / 255.0
    source_float = source_byte.astype(np.float32)
    result_float = result_byte.astype(np.float32)
    ratio = np.ones(source_byte.shape, dtype=np.float32)
    nonzero = source_float != 0
    ratio[nonzero] = result_float[nonzero] / source_float[nonzero]

    if original.dtype == np.uint8:
        values = original.astype(np.float32) * (
            1.0 + mask_fraction * (ratio - 1.0)
        )
        return np.clip(_round_pos(values), 0, 255).astype(np.uint8)

    finite = np.isfinite(original) if original.dtype == np.float32 else np.ones(original.shape, bool)
    min_value = float(np.nanmin(original[finite])) if finite.any() else 0.0
    values = mask_fraction * (
        ratio * (original.astype(np.float32) - min_value)
        + min_value
        - original.astype(np.float32)
    ) + original.astype(np.float32)

    if original.dtype == np.uint16:
        return np.clip(_round_pos(values), 0, 65535).astype(np.uint16)
    return values.astype(np.float32, copy=False)


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

    Preserve shape and dtype where possible. Supports 2D grayscale images and
    3D grayscale stacks with uint8, uint16, and float32 data. 3D stacks are
    processed one Z-slice at a time. Raises ValueError for unsupported image
    dimensions or parameters.
    """
    array = _validate_inputs(image, block_size, histogram_bins, maximum_slope, mask)
    if array.ndim == 3:
        mask_array = None if mask is None else np.asarray(mask)
        return np.stack(
            [
                apply_imagej_clahe(
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

    radius = (int(block_size) - 1) // 2
    bins_internal = int(histogram_bins) - 1
    source_byte = _to_fiji_byte_image(array)
    mask_byte = _mask_to_byte(mask, array.shape)

    if fast:
        result_byte = _fiji_fast_byte_clahe(
            source_byte,
            radius=radius,
            bins_internal=bins_internal,
            maximum_slope=float(maximum_slope),
        )
    else:
        result_byte = _fiji_exact_byte_clahe(
            source_byte,
            radius=radius,
            bins_internal=bins_internal,
            maximum_slope=float(maximum_slope),
            mask_byte=None if mask is None else mask_byte,
        )

    return _apply_fiji_byte_result(array, source_byte, result_byte, mask_byte)

# Enhance Local Contrast CLAHE

This task adds an ImageJ/Fiji-style **Enhance Local Contrast (CLAHE)** workflow
for 2D grayscale EM images in napari.

## ImageJ-Style Parameters

- `block size`: local contrast block size in pixels. Default: `127`.
- `histogram bins`: local histogram resolution. Default: `256`.
- `maximum slope`: contrast limiting slope. Default: `3.00`.
- `mask`: optional mask layer. Default: `*None*`.
- `fast (less accurate)`: enabled by default.

## Backends

- `imagej_reference`: target API for ImageJ/Fiji-compatible behavior. At this
  stage it delegates to the OpenCV CPU approximation and emits a compatibility
  warning.
- `opencv_cpu`: fast CPU approximation implemented with `cv2.createCLAHE`.
- `gpu_cupy`: optional CUDA/CuPy approximation for batch processing. If CuPy or
  a CUDA device is unavailable, it falls back to `opencv_cpu`.
- `gpu`: planned backend stub. It raises `NotImplementedError` for now.

## OpenCV Compatibility Warning

OpenCV CLAHE parameters are not identical to ImageJ/Fiji CLAHE parameters.
`block size` is converted to an OpenCV tile grid size, and `maximum slope` is
used as an approximate `clipLimit`. This backend is useful for fast local
contrast enhancement, but output should not be treated as validated against Fiji
until reference comparisons are added.

## GPU Batch Processing

The `gpu_cupy` backend is intended for large TIFF batches. It processes images
one at a time, reports load/process progress to the widget table, and saves
outputs as `<stem>_clahe.tif`.

CuPy is optional because CUDA package compatibility depends on the local driver
and CUDA runtime. The package exposes a convenience extra:

```bash
pip install -e ".[gpu]"
```

If that CuPy build is not right for the workstation, install the matching CuPy
wheel manually and keep the base napari plugin install CPU-only.

## Future Plan

The module is organized as a task package so validated ImageJ-compatible CPU
behavior and more accurate GPU interpolation can be added without changing the
widget contract.

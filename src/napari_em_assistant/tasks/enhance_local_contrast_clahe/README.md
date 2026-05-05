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
- `gpu`: planned backend stub. It raises `NotImplementedError` for now.

## OpenCV Compatibility Warning

OpenCV CLAHE parameters are not identical to ImageJ/Fiji CLAHE parameters.
`block size` is converted to an OpenCV tile grid size, and `maximum slope` is
used as an approximate `clipLimit`. This backend is useful for fast local
contrast enhancement, but output should not be treated as validated against Fiji
until reference comparisons are added.

## Future Plan

The module is organized as a task package so validated ImageJ-compatible CPU
behavior, GPU acceleration, and more complete batch processing can be added
without changing the widget contract.


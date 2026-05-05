# Enhance Local Contrast CLAHE

This task adds an ImageJ/Fiji-style **Enhance Local Contrast (CLAHE)** workflow
for 2D grayscale EM images and 3D grayscale stacks in napari.

## ImageJ-Style Parameters

- `block size`: local contrast block size in pixels. Default: `127`.
- `histogram bins`: local histogram resolution. Default: `256`.
- `maximum slope`: contrast limiting slope. Default: `3.00`.
- `mask`: optional mask layer. Default: `*None*`.
- `fast (less accurate)`: enabled by default.

## Backends

- `opencv_cpu`: fast CPU approximation implemented with `cv2.createCLAHE`.
- `imagej_reference`: Python implementation ported from Fiji's
  `mpicbg.ij.clahe.PlugIn` 2D grayscale path. Fiji displays `block size` as
  `2 * blockRadius + 1` and `histogram bins` as `bins + 1`; this backend keeps
  the user-facing Fiji parameters and converts them internally the same way. 3D
  stacks are processed one Z-slice at a time.
- `gpu_cupy`: optional CUDA/CuPy approximation for batch processing. If CuPy or
  a CUDA device is unavailable, it falls back to `opencv_cpu`.

## OpenCV Compatibility Warning

OpenCV CLAHE parameters are not identical to ImageJ/Fiji CLAHE parameters.
`block size` is converted to an OpenCV tile grid size, and `maximum slope` is
used as an approximate `clipLimit`. This backend is useful for fast local
contrast enhancement. Use `imagej_reference` when Fiji-style output is the
priority.

## GPU Batch Processing

The widget includes a CUDA status bar above the parameters. It reports:

- `CuPy`: whether the GPU CuPy backend can currently see a CUDA device.
- `OpenCV CUDA CLAHE`: whether the installed OpenCV build exposes CUDA CLAHE.

The `gpu_cupy` backend is intended for large TIFF batches. It processes images
or stacks one file at a time, reports load/process progress to the widget table,
and saves outputs as `<stem>_clahe.tif`.

CuPy is optional because CUDA package compatibility depends on the local driver
and CUDA runtime. The package exposes a convenience extra:

```bash
pip install -e ".[gpu]"
```

If that CuPy build is not right for the workstation, install the matching CuPy
wheel manually and keep the base napari plugin install CPU-only.

## Future Plan

The module is organized as a task package so broader ImageJ-compatible behavior
and more accurate GPU interpolation can be added without changing the widget
contract.

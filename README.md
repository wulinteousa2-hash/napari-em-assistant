# napari-em-assistant

Task-based electron microscopy image processing tools for napari.

`napari-em-assistant` is organized around small, explicit processing tasks that
can be used interactively on an active napari layer or applied to folders of EM
images. The first task is ImageJ/Fiji-style local contrast enhancement with
CLAHE.

## Current Task

### Enhance Local Contrast CLAHE

The CLAHE task provides a napari widget for 2D grayscale EM images. It follows
the ImageJ/Fiji dialog shape and default values:

- block size: `127`
- histogram bins: `256`
- maximum slope: `3.00`
- mask: `*None*`
- fast: checked

The widget can:

- preview CLAHE on the active image layer
- apply CLAHE to the active 2D image layer
- create a new layer named `<source_layer_name>_CLAHE`
- batch process TIFF folders to an output folder
- show batch progress with load/process checks, status, and output path

## Backends

- `imagej_reference`: stable target API for ImageJ/Fiji-compatible behavior.
  It currently delegates to the OpenCV approximation and emits a compatibility
  warning until validated against Fiji output.
- `opencv_cpu`: fast CPU approximation using `cv2.createCLAHE`.
- `gpu_cupy`: optional CuPy/CUDA batch backend. If CuPy or a CUDA device is not
  available, it falls back to `opencv_cpu`.
- `gpu`: reserved stub for future validated GPU work. It raises
  `NotImplementedError`.

OpenCV CLAHE parameters are not identical to ImageJ/Fiji CLAHE parameters.
Current OpenCV and CuPy paths should be treated as practical approximations,
not validated Fiji-equivalent output.

## Install

From this repository:

```bash
pip install -e .
```

Optional CuPy support depends on the workstation CUDA stack. A convenience extra
is provided:

```bash
pip install -e ".[gpu]"
```

If that wheel does not match the local CUDA runtime, install the correct CuPy
package manually and keep the base plugin install CPU-only.

## Use In napari

After installation, open napari and choose:

```text
Plugins > Enhance Local Contrast CLAHE
```

For interactive use, select a 2D grayscale image layer and use `Preview` or
`Apply to Active Layer`.

For batch use, choose an input folder containing `.tif` or `.tiff` files, choose
an output folder, select a backend, and run the batch. Outputs are saved as:

```text
<stem>_clahe.tif
```

## Validation

The current implementation is tested for:

- widget and backend imports
- OpenCV CLAHE on synthetic `uint8` and `uint16` images
- shape preservation
- dtype preservation where possible
- 3D image rejection
- GPU stub behavior
- `gpu_cupy` CPU fallback behavior
- batch result and progress reporting

Run tests with:

```bash
python3 -m pytest tests/tasks/test_enhance_local_contrast_clahe.py
```

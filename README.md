# napari-em-assistant

Task-based electron microscopy image processing tools for napari.

`napari-em-assistant` is organized around small, explicit processing tasks that
can be used interactively on an active napari layer or applied to folders of EM
images. The first task is ImageJ/Fiji-style local contrast enhancement with
CLAHE.

## Current Task

### Enhance Local Contrast CLAHE

The CLAHE task provides a napari widget for 2D grayscale EM images and 3D
grayscale stacks. It follows the ImageJ/Fiji dialog shape and default values:

- block size: `127`
- histogram bins: `256`
- maximum slope: `3.00`
- mask: `*None*`
- fast: checked

The widget can:

- show a top acceleration bar for CPU, CUDA via CuPy, or CPU fallback
- preview CLAHE on the active image layer
- apply CLAHE to the active 2D image layer or 3D grayscale stack
- create a new layer named `<source_layer_name>_CLAHE`
- batch process TIFF folders to an output folder
- show batch progress with load/process checks, status, and output path

## Backends

- `opencv_cpu`: fast CPU approximation using `cv2.createCLAHE`.
- `imagej_reference`: Python implementation ported from Fiji's
  `mpicbg.ij.clahe.PlugIn` 2D grayscale CLAHE path. 3D stacks are processed
  slice-by-slice. The `fast` checkbox only affects this backend.
- `gpu_cupy`: experimental CuPy/CUDA batch backend. If CuPy or a CUDA device is
  not available, it falls back to `opencv_cpu`. It may be slower than
  `opencv_cpu` for many current workloads because the implementation pays GPU
  transfer overhead and is not yet a fully optimized CUDA CLAHE kernel.

The acceleration bar shows the active mode for the selected backend. Detailed
CuPy/OpenCV CUDA availability is still available in the tooltip for diagnostics.

OpenCV CLAHE parameters are not identical to ImageJ/Fiji CLAHE parameters.
Use `imagej_reference` when Fiji-style output is the priority. Current OpenCV
and CuPy paths should be treated as practical approximations.

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

For interactive use, select a 2D grayscale image layer or 3D grayscale stack and
use `Preview` or `Apply to Active Layer`.

For batch use, choose an input folder containing 2D images or 3D grayscale
stacks saved as `.tif` or `.tiff` files, choose an output folder, select a
backend, and run the batch. Outputs are saved as:

```text
<stem>_clahe.tif
```

## Validation

The current implementation is tested for:

- widget and backend imports
- Fiji-derived `imagej_reference` fast and exact paths
- OpenCV CLAHE on synthetic `uint8` and `uint16` images
- shape preservation
- dtype preservation where possible
- 3D stack support
- GPU stub behavior
- `gpu_cupy` CPU fallback behavior
- batch result and progress reporting

Run tests with:

```bash
python3 -m pytest tests/tasks/test_enhance_local_contrast_clahe.py
```

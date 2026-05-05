import numpy as np
import pytest
import tifffile


def test_import_widget_and_backends():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.gpu_clahe import (
        apply_gpu_clahe,
        apply_gpu_cupy_clahe,
    )
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.imagej_clahe import (
        apply_imagej_clahe,
    )
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.widget import (
        EnhanceLocalContrastCLAHEWidget,
    )

    assert EnhanceLocalContrastCLAHEWidget is not None
    assert apply_imagej_clahe is not None
    assert apply_opencv_clahe is not None
    assert apply_gpu_clahe is not None
    assert apply_gpu_cupy_clahe is not None


def test_opencv_clahe_uint8_shape_and_dtype():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    result = apply_opencv_clahe(image, block_size=15)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_opencv_clahe_uint16_shape_and_dtype():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.tile(np.arange(64, dtype=np.uint16), (64, 1)) * 512
    result = apply_opencv_clahe(image, block_size=15)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_imagej_reference_clahe_uint8_shape_dtype_and_no_warning(recwarn):
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.imagej_clahe import (
        apply_imagej_clahe,
    )

    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    result = apply_imagej_clahe(image, block_size=15, fast=True)

    assert len(recwarn) == 0
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_imagej_reference_exact_path_uint16_shape_dtype():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.imagej_clahe import (
        apply_imagej_clahe,
    )

    image = np.tile(np.arange(16, dtype=np.uint16), (16, 1)) * 1024
    result = apply_imagej_clahe(image, block_size=7, fast=False)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_3d_image_raises_value_error():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.zeros((4, 16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D grayscale"):
        apply_opencv_clahe(image)


def test_gpu_backend_raises_not_implemented():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.gpu_clahe import (
        apply_gpu_clahe,
    )

    with pytest.raises(NotImplementedError, match="GPU CLAHE backend"):
        apply_gpu_clahe(np.zeros((16, 16), dtype=np.uint8))


def test_gpu_cupy_backend_falls_back_to_cpu(monkeypatch):
    from napari_em_assistant.tasks.enhance_local_contrast_clahe import gpu_clahe

    monkeypatch.setattr(gpu_clahe, "_load_cupy", lambda: None)
    image = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    result = gpu_clahe.apply_gpu_cupy_clahe(image, block_size=7)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_batch_function_returns_expected_keys(tmp_path):
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.batch import (
        batch_apply_clahe,
    )

    image_path = tmp_path / "input.tif"
    output_dir = tmp_path / "out"
    tifffile.imwrite(image_path, np.zeros((16, 16), dtype=np.uint8))

    result = batch_apply_clahe(
        input_paths=[image_path],
        output_dir=output_dir,
        backend="opencv_cpu",
        block_size=7,
        histogram_bins=256,
        maximum_slope=3.0,
        fast=True,
    )

    assert set(result) == {"processed", "skipped", "failed"}
    assert len(result["processed"]) == 1
    assert result["skipped"] == []
    assert result["failed"] == []


def test_batch_gpu_cupy_reports_cpu_fallback_progress(tmp_path, monkeypatch):
    from napari_em_assistant.tasks.enhance_local_contrast_clahe import batch
    from napari_em_assistant.tasks.enhance_local_contrast_clahe import gpu_clahe

    monkeypatch.setattr(gpu_clahe, "_load_cupy", lambda: None)
    monkeypatch.setattr(batch, "is_gpu_cupy_available", lambda: False)
    image_path = tmp_path / "input.tif"
    output_dir = tmp_path / "out"
    tifffile.imwrite(image_path, np.zeros((16, 16), dtype=np.uint8))
    events = []

    result = batch.batch_apply_clahe(
        input_paths=[image_path],
        output_dir=output_dir,
        backend="gpu_cupy",
        block_size=7,
        histogram_bins=256,
        maximum_slope=3.0,
        fast=True,
        progress_callback=events.append,
    )

    assert len(result["processed"]) == 1
    assert events[-1]["processed"] is True
    assert events[-1]["backend"] == "opencv_cpu_fallback"

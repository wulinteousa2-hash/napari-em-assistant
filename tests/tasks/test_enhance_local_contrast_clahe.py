import os

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


def test_widget_backend_defaults_to_opencv_cpu():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from qtpy.QtWidgets import QApplication

    from napari_em_assistant.tasks.enhance_local_contrast_clahe.widget import (
        EnhanceLocalContrastCLAHEWidget,
    )

    app = QApplication.instance() or QApplication([])
    widget = EnhanceLocalContrastCLAHEWidget()

    assert app is not None
    assert widget.backend.currentData() == "opencv_cpu"
    assert widget.backend.count() == 3
    assert widget.fast.isEnabled() is False
    assert widget.gpu_status_bar.text() == "Acceleration: CPU"


def test_widget_acceleration_bar_uses_summary(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from qtpy.QtWidgets import QApplication

    from napari_em_assistant.tasks.enhance_local_contrast_clahe import widget as widget_module

    monkeypatch.setattr(
        widget_module,
        "gpu_status_summary",
        lambda: {"cupy_cuda": False, "opencv_cuda_clahe": True},
    )
    app = QApplication.instance() or QApplication([])
    widget = widget_module.EnhanceLocalContrastCLAHEWidget()
    widget.backend.setCurrentIndex(widget.backend.findData("gpu_cupy"))

    assert app is not None
    assert widget.gpu_status_bar.text() == "Acceleration: CPU fallback"
    assert "CuPy CUDA: not available" in widget.gpu_status_bar.toolTip()
    assert "OpenCV CUDA CLAHE: available" in widget.gpu_status_bar.toolTip()


def test_fast_checkbox_only_enabled_for_imagej_reference():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from qtpy.QtWidgets import QApplication

    from napari_em_assistant.tasks.enhance_local_contrast_clahe.widget import (
        EnhanceLocalContrastCLAHEWidget,
    )

    app = QApplication.instance() or QApplication([])
    widget = EnhanceLocalContrastCLAHEWidget()
    assert app is not None

    widget.backend.setCurrentIndex(widget.backend.findData("imagej_reference"))
    assert widget.fast.isEnabled() is True

    widget.backend.setCurrentIndex(widget.backend.findData("gpu_cupy"))
    assert widget.fast.isEnabled() is False


def test_opencv_cuda_detector_false_without_cuda(monkeypatch):
    from napari_em_assistant.tasks.enhance_local_contrast_clahe import gpu_clahe

    class FakeCuda:
        @staticmethod
        def getCudaEnabledDeviceCount():
            return 0

        @staticmethod
        def createCLAHE():
            return None

    class FakeCv2:
        cuda = FakeCuda()

        @staticmethod
        def getBuildInformation():
            return "NVIDIA CUDA"

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCv2)

    assert gpu_clahe.has_opencv_cuda_clahe() is False


def test_opencv_clahe_uint8_shape_and_dtype():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    result = apply_opencv_clahe(image, block_size=15)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_imagej_slope_to_opencv_cliplimit_mapping():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        imagej_slope_to_opencv_cliplimit,
    )

    assert imagej_slope_to_opencv_cliplimit(1.0) == 0.0
    assert imagej_slope_to_opencv_cliplimit(2.0) == 0.5
    assert imagej_slope_to_opencv_cliplimit(3.0) == 1.0
    assert imagej_slope_to_opencv_cliplimit(5.0) == 2.0


def test_opencv_clahe_maximum_slope_at_or_below_one_is_unchanged():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))

    for maximum_slope in (0.5, 1.0):
        result = apply_opencv_clahe(
            image,
            block_size=15,
            maximum_slope=maximum_slope,
        )

        assert np.array_equal(result, image)
        assert result is not image


def test_opencv_clahe_maximum_slope_above_one_changes_image():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    result = apply_opencv_clahe(image, block_size=15, maximum_slope=3.0)

    assert not np.array_equal(result, image)


def test_opencv_clahe_uint16_shape_and_dtype():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.tile(np.arange(64, dtype=np.uint16), (64, 1)) * 512
    result = apply_opencv_clahe(image, block_size=15)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_opencv_clahe_3d_stack_shape_and_dtype():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.stack(
        [np.tile(np.arange(32, dtype=np.uint8), (32, 1)) for _ in range(3)],
        axis=0,
    )
    result = apply_opencv_clahe(image, block_size=7)

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


def test_imagej_reference_clahe_3d_stack_shape_and_dtype():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.imagej_clahe import (
        apply_imagej_clahe,
    )

    image = np.stack(
        [np.tile(np.arange(32, dtype=np.uint8), (32, 1)) for _ in range(2)],
        axis=0,
    )
    result = apply_imagej_clahe(image, block_size=7, fast=True)

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


def test_4d_image_raises_value_error():
    from napari_em_assistant.tasks.enhance_local_contrast_clahe.opencv_clahe import (
        apply_opencv_clahe,
    )

    image = np.zeros((2, 3, 16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D images and 3D grayscale stacks"):
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

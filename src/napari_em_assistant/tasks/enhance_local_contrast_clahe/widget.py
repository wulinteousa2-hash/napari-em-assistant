"""Qt widget for ImageJ/Fiji-style Enhance Local Contrast (CLAHE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .batch import batch_apply_clahe
from .gpu_clahe import apply_gpu_clahe
from .imagej_clahe import apply_imagej_clahe
from .opencv_clahe import apply_opencv_clahe


_BACKENDS = {
    "imagej_reference": apply_imagej_clahe,
    "opencv_cpu": apply_opencv_clahe,
    "gpu": apply_gpu_clahe,
}


class EnhanceLocalContrastCLAHEWidget(QWidget):
    """Task widget for ImageJ/Fiji-style CLAHE."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._preview_layer = None
        self.setWindowTitle("Enhance Local Contrast CLAHE")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.block_size = QSpinBox()
        self.block_size.setRange(3, 9999)
        self.block_size.setSingleStep(2)
        self.block_size.setValue(127)
        form.addRow("block size", self.block_size)

        self.histogram_bins = QComboBox()
        for value in (128, 256, 512, 1024):
            self.histogram_bins.addItem(str(value), value)
        self.histogram_bins.setCurrentText("256")
        form.addRow("histogram bins", self.histogram_bins)

        self.maximum_slope = QDoubleSpinBox()
        self.maximum_slope.setRange(0.01, 1_000_000.0)
        self.maximum_slope.setDecimals(2)
        self.maximum_slope.setSingleStep(0.1)
        self.maximum_slope.setValue(3.00)
        form.addRow("maximum slope", self.maximum_slope)

        self.mask_layer = QComboBox()
        self.mask_layer.addItem("*None*", None)
        form.addRow("mask", self.mask_layer)

        self.fast = QCheckBox("fast (less accurate)")
        self.fast.setChecked(True)
        form.addRow("", self.fast)

        self.backend = QComboBox()
        for value in ("imagej_reference", "opencv_cpu", "gpu"):
            self.backend.addItem(value, value)
        form.addRow("backend", self.backend)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply to Active Layer")
        self.preview_button = QPushButton("Preview")
        self.batch_button = QPushButton("Batch Process Folder")
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.batch_button)
        layout.addLayout(button_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.apply_button.clicked.connect(self.apply_to_active_layer)
        self.preview_button.clicked.connect(self.preview)
        self.batch_button.clicked.connect(self.batch_process_folder)

        self._refresh_mask_layers()
        if self.viewer is not None:
            try:
                self.viewer.layers.events.inserted.connect(self._refresh_mask_layers)
                self.viewer.layers.events.removed.connect(self._refresh_mask_layers)
                self.viewer.layers.events.changed.connect(self._refresh_mask_layers)
            except Exception:
                pass

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Enhance Local Contrast CLAHE", message)

    def _refresh_mask_layers(self, event=None):
        selected_name = self.mask_layer.currentText() if hasattr(self, "mask_layer") else "*None*"
        self.mask_layer.blockSignals(True)
        self.mask_layer.clear()
        self.mask_layer.addItem("*None*", None)
        if self.viewer is not None:
            for layer in self.viewer.layers:
                if getattr(layer, "data", None) is not None:
                    self.mask_layer.addItem(layer.name, layer)
        index = self.mask_layer.findText(selected_name)
        self.mask_layer.setCurrentIndex(index if index >= 0 else 0)
        self.mask_layer.blockSignals(False)

    def _active_image_layer(self):
        if self.viewer is None:
            raise ValueError("No napari viewer is attached to this widget.")
        selected = list(self.viewer.layers.selection)
        layer = selected[-1] if selected else getattr(self.viewer.layers, "selection", None).active
        if layer is None:
            raise ValueError("Select an active 2D Image layer first.")
        try:
            from napari.layers import Image
        except Exception:
            Image = None
        if Image is not None and not isinstance(layer, Image):
            raise ValueError("The active layer must be a napari Image layer.")
        if np.asarray(layer.data).ndim != 2:
            raise ValueError("CLAHE supports 2D grayscale images only for this MVP.")
        return layer

    def _parameters(self):
        block_size = int(self.block_size.value())
        if block_size < 3:
            raise ValueError("block_size must be >= 3.")
        if block_size % 2 == 0:
            raise ValueError("Odd block size is preferred. Choose an odd value.")

        histogram_bins = int(self.histogram_bins.currentData())
        maximum_slope = float(self.maximum_slope.value())
        if maximum_slope <= 0:
            raise ValueError("maximum_slope must be > 0.")

        mask_layer = self.mask_layer.currentData()
        backend = self.backend.currentData()
        return {
            "block_size": block_size,
            "histogram_bins": histogram_bins,
            "maximum_slope": maximum_slope,
            "mask_layer": mask_layer,
            "fast": bool(self.fast.isChecked()),
            "backend": backend,
        }

    def _apply(self, layer):
        params = self._parameters()
        image = np.asarray(layer.data)
        mask = None
        if params["mask_layer"] is not None:
            mask = np.asarray(params["mask_layer"].data)
            if mask.shape != image.shape:
                raise ValueError("mask must match the image shape.")

        result = _BACKENDS[params["backend"]](
            image,
            block_size=params["block_size"],
            histogram_bins=params["histogram_bins"],
            maximum_slope=params["maximum_slope"],
            mask=mask,
            fast=params["fast"],
        )
        metadata = {
            "task": "Enhance_Local_Contrast_CLAHE",
            "source_layer": layer.name,
            "backend": params["backend"],
            "block_size": params["block_size"],
            "histogram_bins": params["histogram_bins"],
            "maximum_slope": params["maximum_slope"],
            "fast": params["fast"],
        }
        return result, metadata

    def apply_to_active_layer(self):
        try:
            layer = self._active_image_layer()
            result, metadata = self._apply(layer)
            self.viewer.add_image(
                result,
                name=f"{layer.name}_CLAHE",
                metadata=metadata,
            )
            self.status_label.setText(f"Created {layer.name}_CLAHE")
        except Exception as exc:
            self._show_error(str(exc))

    def preview(self):
        try:
            layer = self._active_image_layer()
            result, metadata = self._apply(layer)
            name = f"{layer.name}_CLAHE_preview"
            if self._preview_layer is not None and self._preview_layer in self.viewer.layers:
                self._preview_layer.data = result
                self._preview_layer.metadata = metadata
                self._preview_layer.name = name
            else:
                self._preview_layer = self.viewer.add_image(
                    result,
                    name=name,
                    metadata=metadata,
                    blending="additive",
                )
            self.status_label.setText(f"Updated {name}")
        except Exception as exc:
            self._show_error(str(exc))

    def batch_process_folder(self):
        try:
            params = self._parameters()
            input_dir = QFileDialog.getExistingDirectory(
                self,
                "Select folder with 2D grayscale TIFF files",
            )
            if not input_dir:
                return
            output_dir = QFileDialog.getExistingDirectory(
                self,
                "Select output folder",
            )
            if not output_dir:
                return
            input_paths = sorted(
                path
                for path in Path(input_dir).iterdir()
                if path.suffix.lower() in {".tif", ".tiff"}
            )
            results = batch_apply_clahe(
                input_paths=input_paths,
                output_dir=output_dir,
                backend=params["backend"],
                block_size=params["block_size"],
                histogram_bins=params["histogram_bins"],
                maximum_slope=params["maximum_slope"],
                fast=params["fast"],
                overwrite=False,
            )
            self.status_label.setText(
                "Processed {processed}, skipped {skipped}, failed {failed}".format(
                    processed=len(results["processed"]),
                    skipped=len(results["skipped"]),
                    failed=len(results["failed"]),
                )
            )
        except Exception as exc:
            self._show_error(str(exc))


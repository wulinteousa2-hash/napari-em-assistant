"""Qt widget for ImageJ/Fiji-style Enhance Local Contrast (CLAHE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .batch import batch_apply_clahe
from .gpu_clahe import apply_gpu_cupy_clahe, gpu_status_summary
from .imagej_clahe import apply_imagej_clahe
from .opencv_clahe import apply_opencv_clahe


_BACKENDS = {
    "opencv_cpu": apply_opencv_clahe,
    "imagej_reference": apply_imagej_clahe,
    "gpu_cupy": apply_gpu_cupy_clahe,
}


class EnhanceLocalContrastCLAHEWidget(QWidget):
    """Task widget for ImageJ/Fiji-style CLAHE."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._preview_layer = None
        self._batch_input_paths = []
        self._table_rows = {}
        self.setWindowTitle("Enhance Local Contrast CLAHE")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        def tooltip_label(text: str, tooltip: str) -> QLabel:
            label = QLabel(text)
            label.setToolTip(tooltip)
            return label

        self.gpu_status_bar = QLabel("")
        self.gpu_status_bar.setWordWrap(True)
        self.gpu_status_bar.setToolTip(
            "Active acceleration mode. OpenCV CPU is usually fastest today. "
            "GPU CuPy is experimental and falls back to CPU when CUDA is not available."
        )
        layout.addWidget(self.gpu_status_bar)

        form = QFormLayout()
        self.block_size = QSpinBox()
        self.block_size.setRange(3, 9999)
        self.block_size.setSingleStep(2)
        self.block_size.setValue(127)
        block_size_tip = (
            "ImageJ-style local neighborhood size in pixels. Use odd values; "
            "larger values smooth contrast over broader EM structures."
        )
        self.block_size.setToolTip(block_size_tip)
        form.addRow(tooltip_label("block size", block_size_tip), self.block_size)

        self.histogram_bins = QComboBox()
        for value in (128, 256, 512, 1024):
            self.histogram_bins.addItem(str(value), value)
        self.histogram_bins.setCurrentText("256")
        histogram_bins_tip = (
            "Number of bins used for each local histogram. 256 matches the "
            "ImageJ/Fiji default and is usually a good EM starting point."
        )
        self.histogram_bins.setToolTip(histogram_bins_tip)
        form.addRow(
            tooltip_label("histogram bins", histogram_bins_tip),
            self.histogram_bins,
        )

        self.maximum_slope = QDoubleSpinBox()
        self.maximum_slope.setRange(0.01, 1_000_000.0)
        self.maximum_slope.setDecimals(2)
        self.maximum_slope.setSingleStep(0.1)
        self.maximum_slope.setValue(3.00)
        maximum_slope_tip = (
            "ImageJ-style contrast limit. Higher values allow stronger local "
            "contrast and may amplify noise. In OpenCV CPU mode this value is "
            "rescaled for a more ImageJ-like response; maximum slope 1 applies "
            "no enhancement."
        )
        self.maximum_slope.setToolTip(maximum_slope_tip)
        form.addRow(
            tooltip_label("maximum slope", maximum_slope_tip),
            self.maximum_slope,
        )

        self.mask_layer = QComboBox()
        self.mask_layer.addItem("*None*", None)
        mask_tip = (
            "Optional mask layer. When provided, it must match the image shape; "
            "pixels outside the mask keep their original values."
        )
        self.mask_layer.setToolTip(mask_tip)
        form.addRow(tooltip_label("mask", mask_tip), self.mask_layer)

        self.fast = QCheckBox("fast (less accurate)")
        self.fast.setChecked(True)
        fast_tip = (
            "ImageJ/Fiji reference backend only. Checked uses Fiji's faster "
            "interpolated path; unchecked uses the slower exact sliding-window path."
        )
        self.fast.setToolTip(fast_tip)
        form.addRow(tooltip_label("fast", fast_tip), self.fast)

        self.backend = QComboBox()
        for label, value in (
            ("OpenCV CPU (default)", "opencv_cpu"),
            ("ImageJ/Fiji reference", "imagej_reference"),
            ("GPU CuPy experimental", "gpu_cupy"),
        ):
            self.backend.addItem(label, value)
        self.backend.setCurrentIndex(0)
        backend_tip = (
            "Processing implementation. OpenCV CPU is the default and usually "
            "fastest but is not parameter-identical to ImageJ/Fiji; it rescales "
            "the ImageJ-style maximum slope for a more similar user experience. "
            "ImageJ/Fiji reference prioritizes Fiji-style behavior. "
            "GPU CuPy is experimental and may be slower for typical image sizes."
        )
        self.backend.setToolTip(backend_tip)
        form.addRow(tooltip_label("backend", backend_tip), self.backend)
        layout.addLayout(form)

        batch_form = QFormLayout()
        self.input_folder = QLineEdit()
        self.input_folder.setReadOnly(True)
        self.output_folder = QLineEdit()
        self.output_folder.setReadOnly(True)
        self.select_input_button = QPushButton("Select Input Folder")
        self.select_output_button = QPushButton("Select Output Folder")
        input_folder_tip = "Folder containing 2D images or 3D grayscale TIFF stacks."
        output_folder_tip = "Folder where <stem>_clahe.tif outputs will be written."
        self.input_folder.setToolTip(input_folder_tip)
        self.select_input_button.setToolTip(input_folder_tip)
        self.output_folder.setToolTip(output_folder_tip)
        self.select_output_button.setToolTip(output_folder_tip)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_folder)
        input_row.addWidget(self.select_input_button)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_folder)
        output_row.addWidget(self.select_output_button)
        batch_form.addRow(tooltip_label("input folder", input_folder_tip), input_row)
        batch_form.addRow(tooltip_label("output folder", output_folder_tip), output_row)
        layout.addLayout(batch_form)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply to Active Layer")
        self.preview_button = QPushButton("Preview")
        self.batch_button = QPushButton("Run Batch")
        self.apply_button.setToolTip(
            "Apply CLAHE to the active 2D image or 3D grayscale stack and add a new *_CLAHE layer."
        )
        self.preview_button.setToolTip(
            "Create or update a temporary preview layer using the current settings."
        )
        self.batch_button.setToolTip(
            "Process all 2D images or 3D grayscale TIFF stacks from the input folder."
        )
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.batch_button)
        layout.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("Overall batch completion percentage.")
        layout.addWidget(self.progress_bar)

        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(
            ["Image", "Load", "Process", "Status", "Output"]
        )
        header_tips = {
            0: "Input TIFF filename.",
            1: "Checked after the image is successfully read from disk.",
            2: "Checked after CLAHE output is written.",
            3: "Current per-image batch state, including CPU fallback or failures.",
            4: "Destination TIFF path for the CLAHE result.",
        }
        for column, tooltip in header_tips.items():
            self.batch_table.horizontalHeaderItem(column).setToolTip(tooltip)
        self.batch_table.setToolTip(
            "Batch progress table. Load and Process columns are checked as each "
            "image moves through the pipeline."
        )
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.batch_table.setMinimumHeight(180)
        layout.addWidget(self.batch_table)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.apply_button.clicked.connect(self.apply_to_active_layer)
        self.preview_button.clicked.connect(self.preview)
        self.backend.currentIndexChanged.connect(self._on_backend_changed)
        self.select_input_button.clicked.connect(self.select_input_folder)
        self.select_output_button.clicked.connect(self.select_output_folder)
        self.batch_button.clicked.connect(self.batch_process_folder)
        self._on_backend_changed()

        self._refresh_mask_layers()
        if self.viewer is not None:
            try:
                self.viewer.layers.events.inserted.connect(self._refresh_mask_layers)
                self.viewer.layers.events.removed.connect(self._refresh_mask_layers)
                self.viewer.layers.events.changed.connect(self._refresh_mask_layers)
            except Exception:
                pass

    def _update_gpu_status_bar(self):
        status = gpu_status_summary()
        cupy_on = status["cupy_cuda"]
        backend = self.backend.currentData() if hasattr(self, "backend") else "opencv_cpu"
        if backend == "gpu_cupy" and cupy_on:
            self.gpu_status_bar.setText("Acceleration: CUDA via CuPy")
            self.gpu_status_bar.setStyleSheet(
                "QLabel { background-color: #1f7a3a; color: white; "
                "font-weight: 600; padding: 4px 8px; border-radius: 3px; }"
            )
        elif backend == "gpu_cupy":
            self.gpu_status_bar.setText("Acceleration: CPU fallback")
            self.gpu_status_bar.setStyleSheet(
                "QLabel { background-color: #8a5a00; color: white; "
                "font-weight: 600; padding: 4px 8px; border-radius: 3px; }"
            )
        else:
            self.gpu_status_bar.setText("Acceleration: CPU")
            self.gpu_status_bar.setStyleSheet(
                "QLabel { background-color: #4b5563; color: white; "
                "font-weight: 600; padding: 4px 8px; border-radius: 3px; }"
            )
        self.gpu_status_bar.setToolTip(
            "CuPy CUDA: {cupy}. OpenCV CUDA CLAHE: {opencv}. GPU CuPy is "
            "experimental; OpenCV CPU is the default because it is typically "
            "faster for current CLAHE workloads.".format(
                cupy="available" if cupy_on else "not available",
                opencv="available" if status["opencv_cuda_clahe"] else "not available",
            )
        )

    def _on_backend_changed(self):
        is_imagej = self.backend.currentData() == "imagej_reference"
        self.fast.setEnabled(is_imagej)
        if is_imagej:
            self.fast.setToolTip(
                "ImageJ/Fiji reference backend only. Checked uses Fiji's faster "
                "interpolated path; unchecked uses the slower exact sliding-window path."
            )
        else:
            self.fast.setToolTip(
                "Fast/exact only changes the ImageJ/Fiji reference backend. "
                "OpenCV CPU and GPU CuPy ignore this option."
            )
        self._update_gpu_status_bar()

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Enhance Local Contrast CLAHE", message)

    def _set_table_item(self, row: int, column: int, value: str):
        item = self.batch_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.batch_table.setItem(row, column, item)
        item.setText(value)

    def _set_check_item(self, row: int, column: int, checked: bool):
        item = self.batch_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsEnabled)
            self.batch_table.setItem(row, column, item)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _populate_batch_table(self, input_paths):
        self._table_rows = {}
        self.batch_table.setRowCount(len(input_paths))
        for row, path in enumerate(input_paths):
            output_dir = Path(self.output_folder.text()) if self.output_folder.text() else None
            output_name = ""
            if output_dir is not None:
                output_name = str(output_dir / f"{path.stem}_clahe.tif")
            self._table_rows[str(path)] = row
            self._set_table_item(row, 0, path.name)
            self._set_check_item(row, 1, False)
            self._set_check_item(row, 2, False)
            self._set_table_item(row, 3, "queued")
            self._set_table_item(row, 4, output_name)
        self.progress_bar.setValue(0)

    def select_input_folder(self):
        input_dir = QFileDialog.getExistingDirectory(
            self,
            "Select folder with 2D images or 3D grayscale TIFF stacks",
        )
        if not input_dir:
            return
        self.input_folder.setText(input_dir)
        self._batch_input_paths = sorted(
            path
            for path in Path(input_dir).iterdir()
            if path.suffix.lower() in {".tif", ".tiff"}
        )
        self._populate_batch_table(self._batch_input_paths)
        self.status_label.setText(f"Loaded {len(self._batch_input_paths)} TIFF files")

    def select_output_folder(self):
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
        )
        if not output_dir:
            return
        self.output_folder.setText(output_dir)
        if self._batch_input_paths:
            self._populate_batch_table(self._batch_input_paths)

    def _batch_progress(self, event):
        row = self._table_rows.get(event["path"])
        if row is None:
            return
        self._set_check_item(row, 1, event["loaded"])
        self._set_check_item(row, 2, event["processed"])
        status = event["status"]
        if event["backend"] == "opencv_cpu_fallback" and status in {"loaded", "processed"}:
            status = f"{status} (CPU fallback)"
        self._set_table_item(row, 3, status)
        self._set_table_item(row, 4, event["output"])
        processed_rows = sum(
            1
            for row_index in range(self.batch_table.rowCount())
            if self.batch_table.item(row_index, 3)
            and self.batch_table.item(row_index, 3).text().startswith(
                ("processed", "skipped", "failed")
            )
        )
        total_rows = max(1, self.batch_table.rowCount())
        self.progress_bar.setValue(int(processed_rows / total_rows * 100))
        QApplication.processEvents()

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
            raise ValueError("Select an active 2D image or 3D grayscale stack first.")
        try:
            from napari.layers import Image
        except Exception:
            Image = None
        if Image is not None and not isinstance(layer, Image):
            raise ValueError("The active layer must be a napari Image layer.")
        if np.asarray(layer.data).ndim not in (2, 3):
            raise ValueError("CLAHE supports 2D images and 3D grayscale stacks.")
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
            if not self.input_folder.text():
                self.select_input_folder()
            if not self.output_folder.text():
                self.select_output_folder()
            if not self.input_folder.text() or not self.output_folder.text():
                return
            input_paths = self._batch_input_paths
            if not input_paths:
                raise ValueError("No TIFF files found in the selected input folder.")
            self._populate_batch_table(input_paths)
            results = batch_apply_clahe(
                input_paths=input_paths,
                output_dir=self.output_folder.text(),
                backend=params["backend"],
                block_size=params["block_size"],
                histogram_bins=params["histogram_bins"],
                maximum_slope=params["maximum_slope"],
                fast=params["fast"],
                overwrite=False,
                progress_callback=self._batch_progress,
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

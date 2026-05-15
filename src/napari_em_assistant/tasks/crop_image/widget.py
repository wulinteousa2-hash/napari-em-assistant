"""Qt widget for frictionless 2D/3D image cropping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .batch import (
    batch_crop_tiff_files,
    count_batch_crop_tiles,
    crop_tiles_for_shape,
    export_active_crop_tiles,
)
from .crop import crop_fraction, crop_image, roi_bounds_from_shape_data, should_warn_small_crop


LARGE_OUTPUT_PIECE_WARNING = 1000


class CropImageWidget(QWidget):
    """Crop the active 2D image or 3D grayscale stack."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._preview_layer = None
        self._synced_layer = None
        self._batch_input_paths = []
        self.setWindowTitle("Crop Image")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.layer_label = QLabel("No active image layer")
        self.layer_label.setWordWrap(True)
        self.layer_label.setToolTip("Active image layer and shape used for crop bounds.")
        layout.addWidget(self.layer_label)

        active_row = QHBoxLayout()
        self.sync_button = QPushButton("Sync Active Image")
        self.full_extent_button = QPushButton("Reset Full Extent")
        self.sync_button.setToolTip("Load the selected 2D image or 3D stack shape.")
        self.full_extent_button.setToolTip("Reset typed crop bounds to the full image.")
        active_row.addWidget(self.sync_button)
        active_row.addWidget(self.full_extent_button)
        layout.addLayout(active_row)

        self.z_start, self.z_end = self._axis_spinboxes()
        self.y_start, self.y_end = self._axis_spinboxes()
        self.x_start, self.x_end = self._axis_spinboxes()

        self.tabs = QTabWidget()
        self.tabs.addTab(self._roi_tab(), "1. Crop from Drawn ROI")
        self.tabs.addTab(self._parts_tab(), "2. Tile by Total Count")
        self.tabs.addTab(self._size_tab(), "3. Tile by Pixel Size")
        self.tabs.addTab(self._manual_tab(), "4. Crop by Coordinates")
        layout.addWidget(self.tabs)

        output_box = QGroupBox("Save Crops to Folder")
        output_layout = QFormLayout(output_box)
        self.output_folder = QLineEdit()
        self.output_folder.setReadOnly(True)
        self.select_output_button = QPushButton("Choose Save Folder")
        output_tip = "Folder where cropped TIFF files will be written."
        self.output_folder.setToolTip(output_tip)
        self.select_output_button.setToolTip(output_tip)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_folder)
        output_row.addWidget(self.select_output_button)
        output_layout.addRow("save folder", output_row)

        self.input_folder = QLineEdit()
        self.input_folder.setReadOnly(True)
        self.select_input_button = QPushButton("Choose Image Folder")
        input_tip = "Optional folder of TIFF images or stacks for tiling many files."
        self.input_folder.setToolTip(input_tip)
        self.select_input_button.setToolTip(input_tip)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_folder)
        input_row.addWidget(self.select_input_button)
        output_layout.addRow("image folder", input_row)
        layout.addWidget(output_box)

        progress_box = QGroupBox("Crop Progress")
        progress_layout = QVBoxLayout(progress_box)
        self.progress_label = QLabel("No crop running")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("Shows saved or created crop pieces out of the expected total.")
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_box)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.sync_button.clicked.connect(self.sync_active_layer)
        self.full_extent_button.clicked.connect(self.reset_full_extent)
        self.select_output_button.clicked.connect(self.select_output_folder)
        self.select_input_button.clicked.connect(self.select_input_folder)
        self.tabs.currentChanged.connect(self._update_tab_state)

        self._set_axis_enabled(self.z_start, self.z_end, False)
        self._update_tab_state()
        if self.viewer is not None:
            try:
                self.viewer.layers.selection.events.active.connect(self.sync_active_layer)
                self.viewer.layers.events.inserted.connect(self._refresh_roi_layers)
                self.viewer.layers.events.removed.connect(self._refresh_roi_layers)
                self.viewer.layers.events.changed.connect(self._refresh_roi_layers)
            except Exception:
                pass
            self._refresh_roi_layers()
            self.sync_active_layer()

    def _manual_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(
            self._section_label(
                "Enter voxel coordinates for one crop from the active image or stack."
            )
        )

        form = QFormLayout()
        form.addRow("z", self._axis_row(self.z_start, self.z_end))
        form.addRow("y", self._axis_row(self.y_start, self.y_end))
        form.addRow("x", self._axis_row(self.x_start, self.x_end))
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.preview_button = QPushButton("Preview Crop")
        self.crop_button = QPushButton("Create Cropped Layer")
        self.export_manual_button = QPushButton("Save Crop TIFF")
        self.preview_button.setToolTip("Create or update a temporary crop preview layer.")
        self.crop_button.setToolTip("Add one cropped image layer from the typed bounds.")
        self.export_manual_button.setToolTip("Write the typed crop to the output folder.")
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.crop_button)
        button_row.addWidget(self.export_manual_button)
        layout.addLayout(button_row)

        self.preview_button.clicked.connect(self.preview)
        self.crop_button.clicked.connect(self.crop_active_layer)
        self.export_manual_button.clicked.connect(self.export_manual_crop)
        return tab

    def _roi_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(
            self._section_label(
                "Draw a rectangle or polygon ROI in napari, then crop the active image from that ROI."
            )
        )

        form = QFormLayout()
        self.roi_layer = QComboBox()
        self.roi_layer.setToolTip(
            "Shapes layer containing a rectangle, polygon, or other ROI. "
            "The selected shape is used; if none is selected, the newest shape is used."
        )
        form.addRow("ROI layer", self.roi_layer)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.refresh_roi_button = QPushButton("Refresh")
        self.choose_roi_button = QPushButton("Pick ROI Layer...")
        self.use_roi_button = QPushButton("Use ROI Bounds")
        self.crop_roi_button = QPushButton("Create ROI Crop")
        self.refresh_roi_button.setToolTip("Refresh the list of Shapes ROI layers.")
        self.choose_roi_button.setToolTip("Open a small chooser for available Shapes layers.")
        self.use_roi_button.setToolTip("Copy the selected ROI bounding box into typed bounds.")
        self.crop_roi_button.setToolTip("Crop the active image directly from the selected ROI.")
        button_row.addWidget(self.refresh_roi_button)
        button_row.addWidget(self.choose_roi_button)
        button_row.addWidget(self.use_roi_button)
        button_row.addWidget(self.crop_roi_button)
        layout.addLayout(button_row)

        self.refresh_roi_button.clicked.connect(self._refresh_roi_layers)
        self.choose_roi_button.clicked.connect(self.choose_roi_layer)
        self.use_roi_button.clicked.connect(self.use_roi_bounds)
        self.crop_roi_button.clicked.connect(self.crop_roi)
        return tab

    def _parts_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(
            self._section_label(
                "Enter the total number of tiles you want. The plugin chooses an even Y/X grid automatically."
            )
        )

        self.tile_count = self._positive_spinbox(1, 1_000_000, 6)
        form = QFormLayout()
        form.addRow("total tiles", self.tile_count)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.export_parts_button = QPushButton("Save Active Image Tiles")
        self.batch_parts_button = QPushButton("Save Folder Image Tiles")
        self.export_parts_button.setToolTip("Split the active image into this many near-equal crop TIFF files.")
        self.batch_parts_button.setToolTip("Split every TIFF in the image folder into this many tiles.")
        button_row.addWidget(self.export_parts_button)
        button_row.addWidget(self.batch_parts_button)
        layout.addLayout(button_row)

        self.export_parts_button.clicked.connect(lambda: self.export_active_crops("parts"))
        self.batch_parts_button.clicked.connect(lambda: self.export_batch_crops("parts"))
        return tab

    def _size_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(
            self._section_label(
                "Choose the desired output tile size in pixels/planes. The plugin saves tiles until the image is covered."
            )
        )

        self.z_size = self._positive_spinbox(0, 1_000_000, 0)
        self.y_size = self._positive_spinbox(1, 1_000_000, 512)
        self.x_size = self._positive_spinbox(1, 1_000_000, 512)
        self.z_size.setToolTip("0 keeps the full Z depth in each crop.")
        form = QFormLayout()
        form.addRow("tile size z/y/x", self._triple_row(self.z_size, self.y_size, self.x_size))
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.export_size_button = QPushButton("Save Active Image Tiles")
        self.batch_size_button = QPushButton("Save Folder Image Tiles")
        self.export_size_button.setToolTip("Write active image crops with the requested size.")
        self.batch_size_button.setToolTip("Write requested-size crops for every TIFF in the batch input folder.")
        button_row.addWidget(self.export_size_button)
        button_row.addWidget(self.batch_size_button)
        layout.addLayout(button_row)

        self.export_size_button.clicked.connect(lambda: self.export_active_crops("size"))
        self.batch_size_button.clicked.connect(lambda: self.export_batch_crops("size"))
        return tab

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("QLabel { color: #374151; }")
        return label

    def _positive_spinbox(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        return spinbox

    def _axis_spinboxes(self) -> tuple[QSpinBox, QSpinBox]:
        start = QSpinBox()
        end = QSpinBox()
        for spinbox in (start, end):
            spinbox.setRange(0, 0)
            spinbox.setToolTip("Start is inclusive; end is exclusive.")
        return start, end

    def _axis_row(self, start: QSpinBox, end: QSpinBox) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("start"))
        layout.addWidget(start)
        layout.addWidget(QLabel("end"))
        layout.addWidget(end)
        return row

    def _triple_row(self, z_value: QSpinBox, y_value: QSpinBox, x_value: QSpinBox) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("z"))
        layout.addWidget(z_value)
        layout.addWidget(QLabel("y"))
        layout.addWidget(y_value)
        layout.addWidget(QLabel("x"))
        layout.addWidget(x_value)
        return row

    def _set_axis_enabled(self, start: QSpinBox, end: QSpinBox, enabled: bool):
        start.setEnabled(enabled)
        end.setEnabled(enabled)

    def _update_tab_state(self, event=None):
        manual_active = self.tabs.currentIndex() == 3 if hasattr(self, "tabs") else True
        for spinbox in (self.y_start, self.y_end, self.x_start, self.x_end):
            spinbox.setEnabled(manual_active)
        z_enabled = (
            manual_active
            and self._synced_layer is not None
            and np.asarray(self._synced_layer.data).ndim == 3
        )
        self.z_start.setEnabled(z_enabled)
        self.z_end.setEnabled(z_enabled)

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Crop Image", message)

    def _confirm_continue(self, title: str, message: str) -> bool:
        answer = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _confirm_small_crop(self, layer) -> bool:
        image = np.asarray(layer.data)
        z_range, y_range, x_range = self._bounds(image.ndim)
        if not should_warn_small_crop(image.shape, y_range, x_range, z_range):
            return True
        fraction = crop_fraction(image.shape, y_range, x_range, z_range) * 100.0
        return self._confirm_continue(
            "Very Small Crop",
            "This crop is only {fraction:.3f}% of the source image/stack.\n\n"
            "This can happen when a coordinate or ROI was entered by mistake. "
            "Continue with this crop?".format(fraction=fraction),
        )

    def _confirm_large_output_count(self, total: int) -> bool:
        if total <= LARGE_OUTPUT_PIECE_WARNING:
            return True
        return self._confirm_continue(
            "Large Number of Output Files",
            "This operation will create {total} crop TIFF files.\n\n"
            "That is more than {limit}. Check the region count or tile size before "
            "continuing.".format(total=total, limit=LARGE_OUTPUT_PIECE_WARNING),
        )

    def _start_progress(self, total: int, label: str):
        total = max(1, int(total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"{label}: 0 / {total} pieces")
        QApplication.processEvents()

    def _set_progress(self, completed: int, total: int, label: str):
        total = max(1, int(total))
        completed = min(int(completed), total)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(completed)
        self.progress_label.setText(f"{label}: {completed} / {total} pieces")
        QApplication.processEvents()

    def _tile_progress_callback(self, event):
        self._set_progress(
            event["completed"],
            event["total"],
            "Saving crop tiles",
        )

    def _refresh_roi_layers(self, event=None):
        selected_name = self.roi_layer.currentText() if hasattr(self, "roi_layer") else ""
        self.roi_layer.blockSignals(True)
        self.roi_layer.clear()
        if self.viewer is not None:
            for layer in self.viewer.layers:
                if hasattr(layer, "data") and self._is_shapes_layer(layer):
                    self.roi_layer.addItem(layer.name, layer)
        index = self.roi_layer.findText(selected_name)
        self.roi_layer.setCurrentIndex(index if index >= 0 else 0)
        self.roi_layer.blockSignals(False)

    def _is_shapes_layer(self, layer) -> bool:
        try:
            from napari.layers import Shapes
        except Exception:
            Shapes = None
        if Shapes is not None:
            return isinstance(layer, Shapes)
        return layer.__class__.__name__ == "Shapes"

    def choose_roi_layer(self):
        self._refresh_roi_layers()
        names = [self.roi_layer.itemText(index) for index in range(self.roi_layer.count())]
        if not names:
            self._show_error("No Shapes ROI layers are available.")
            return
        current = self.roi_layer.currentText()
        current_index = names.index(current) if current in names else 0
        name, accepted = QInputDialog.getItem(
            self,
            "Choose ROI Shape Layer",
            "Shapes layer:",
            names,
            current_index,
            False,
        )
        if accepted:
            self.roi_layer.setCurrentIndex(self.roi_layer.findText(name))

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
            raise ValueError("Crop supports 2D images and 3D grayscale stacks.")
        return layer

    def _set_range(self, start: QSpinBox, end: QSpinBox, size: int):
        start.blockSignals(True)
        end.blockSignals(True)
        start.setRange(0, max(0, size - 1))
        end.setRange(1, size)
        start.setValue(0)
        end.setValue(size)
        start.blockSignals(False)
        end.blockSignals(False)

    def sync_active_layer(self, event=None):
        try:
            layer = self._active_image_layer()
            self._synced_layer = layer
            shape = np.asarray(layer.data).shape
            if len(shape) == 2:
                self._set_axis_enabled(self.z_start, self.z_end, False)
                self._set_range(self.z_start, self.z_end, 1)
                self._set_range(self.y_start, self.y_end, shape[0])
                self._set_range(self.x_start, self.x_end, shape[1])
                self.tile_count.setMaximum(shape[0] * shape[1])
                self.z_size.setValue(0)
                self.z_size.setMaximum(0)
                self.y_size.setMaximum(shape[0])
                self.x_size.setMaximum(shape[1])
                self.layer_label.setText(f"{layer.name}: 2D {shape[0]} x {shape[1]}")
            else:
                self._set_axis_enabled(self.z_start, self.z_end, True)
                self._set_range(self.z_start, self.z_end, shape[0])
                self._set_range(self.y_start, self.y_end, shape[1])
                self._set_range(self.x_start, self.x_end, shape[2])
                self.tile_count.setMaximum(shape[1] * shape[2])
                self.z_size.setMaximum(shape[0])
                self.y_size.setMaximum(shape[1])
                self.x_size.setMaximum(shape[2])
                self.layer_label.setText(
                    f"{layer.name}: 3D {shape[0]} x {shape[1]} x {shape[2]}"
                )
            self._update_tab_state()
            self.status_label.setText("")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def reset_full_extent(self):
        if self._synced_layer is None:
            self.sync_active_layer()
            return
        shape = np.asarray(self._synced_layer.data).shape
        if len(shape) == 2:
            self._set_range(self.z_start, self.z_end, 1)
            self._set_range(self.y_start, self.y_end, shape[0])
            self._set_range(self.x_start, self.x_end, shape[1])
        else:
            self._set_range(self.z_start, self.z_end, shape[0])
            self._set_range(self.y_start, self.y_end, shape[1])
            self._set_range(self.x_start, self.x_end, shape[2])
        self._update_tab_state()

    def select_output_folder(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Select crop output folder")
        if output_dir:
            self.output_folder.setText(output_dir)

    def select_input_folder(self):
        input_dir = QFileDialog.getExistingDirectory(self, "Select batch TIFF input folder")
        if not input_dir:
            return
        self.input_folder.setText(input_dir)
        self._batch_input_paths = sorted(
            path
            for path in Path(input_dir).iterdir()
            if path.suffix.lower() in {".tif", ".tiff"}
        )
        self.status_label.setText(f"Loaded {len(self._batch_input_paths)} TIFF files")

    def _bounds(self, ndim: int):
        y_range = (self.y_start.value(), self.y_end.value())
        x_range = (self.x_start.value(), self.x_end.value())
        if ndim == 2:
            return None, y_range, x_range
        return (self.z_start.value(), self.z_end.value()), y_range, x_range

    def _selected_roi_data(self):
        roi_layer = self.roi_layer.currentData()
        if roi_layer is None:
            raise ValueError("Select or create a Shapes ROI layer first.")
        data = list(getattr(roi_layer, "data", []))
        if not data:
            raise ValueError("The selected Shapes layer has no ROI shapes.")

        selected_data = getattr(roi_layer, "selected_data", None)
        if selected_data is not None and len(selected_data) > 0:
            return data[sorted(selected_data)[-1]]
        return data[-1]

    def _set_bounds(self, z_range, y_range, x_range):
        if z_range is not None:
            self.z_start.setValue(z_range[0])
            self.z_end.setValue(z_range[1])
        self.y_start.setValue(y_range[0])
        self.y_end.setValue(y_range[1])
        self.x_start.setValue(x_range[0])
        self.x_end.setValue(x_range[1])
        self.tabs.setCurrentIndex(3)
        self._update_tab_state()

    def use_roi_bounds(self):
        try:
            layer = self._active_image_layer()
            if self._synced_layer is not layer:
                self.sync_active_layer()
            self._start_progress(1, "Reading ROI bounds")
            z_range, y_range, x_range = roi_bounds_from_shape_data(
                np.asarray(layer.data).shape,
                self._selected_roi_data(),
            )
            self._set_bounds(z_range, y_range, x_range)
            self._set_progress(1, 1, "Reading ROI bounds")
            self.status_label.setText(
                "Loaded ROI bounds: z={z}, y={y}, x={x}".format(
                    z=z_range if z_range is not None else "2D",
                    y=y_range,
                    x=x_range,
                )
            )
        except Exception as exc:
            self._show_error(str(exc))

    def _crop(self, layer):
        image = np.asarray(layer.data)
        z_range, y_range, x_range = self._bounds(image.ndim)
        result = crop_image(image, y_range=y_range, x_range=x_range, z_range=z_range)
        metadata = {
            "task": "Crop_Image",
            "source_layer": layer.name,
            "z_range": z_range,
            "y_range": y_range,
            "x_range": x_range,
        }
        return result, metadata

    def _export_parameters(self, mode: str, ndim: int | None = None):
        z_size = self.z_size.value() if self.z_size.value() > 0 else None
        if ndim is None and self._synced_layer is not None:
            ndim = np.asarray(self._synced_layer.data).ndim
        if mode == "manual" and ndim is None:
            raise ValueError("Sync an active layer before exporting typed bounds.")
        z_range, y_range, x_range = self._bounds(2 if ndim is None else ndim)
        return {
            "mode": mode,
            "tile_count": self.tile_count.value(),
            "z_size": z_size,
            "y_size": self.y_size.value(),
            "x_size": self.x_size.value(),
            "z_range": z_range,
            "y_range": y_range,
            "x_range": x_range,
        }

    def crop_active_layer(self):
        try:
            layer = self._active_image_layer()
            if not self._confirm_small_crop(layer):
                return
            self._start_progress(1, "Creating cropped layer")
            result, metadata = self._crop(layer)
            self.viewer.add_image(result, name=f"{layer.name}_crop", metadata=metadata)
            self._set_progress(1, 1, "Creating cropped layer")
            self.status_label.setText(f"Created {layer.name}_crop")
        except Exception as exc:
            self._show_error(str(exc))

    def crop_roi(self):
        try:
            layer = self._active_image_layer()
            if self._synced_layer is not layer:
                self.sync_active_layer()
            self._start_progress(1, "Creating ROI crop")
            z_range, y_range, x_range = roi_bounds_from_shape_data(
                np.asarray(layer.data).shape,
                self._selected_roi_data(),
            )
            self._set_bounds(z_range, y_range, x_range)
            if not self._confirm_small_crop(layer):
                return
            result, metadata = self._crop(layer)
            metadata["roi_source"] = self.roi_layer.currentText()
            self.viewer.add_image(result, name=f"{layer.name}_ROI_crop", metadata=metadata)
            self._set_progress(1, 1, "Creating ROI crop")
            self.status_label.setText(f"Created {layer.name}_ROI_crop")
        except Exception as exc:
            self._show_error(str(exc))

    def export_manual_crop(self):
        self.export_active_crops("manual")

    def export_active_crops(self, mode: str):
        try:
            layer = self._active_image_layer()
            if not self.output_folder.text():
                self.select_output_folder()
            if not self.output_folder.text():
                return
            params = self._export_parameters(mode, np.asarray(layer.data).ndim)
            total = len(crop_tiles_for_shape(np.asarray(layer.data).shape, **params))
            if mode == "manual" and not self._confirm_small_crop(layer):
                return
            if not self._confirm_large_output_count(total):
                return
            self._start_progress(total, "Saving crop tiles")
            paths = export_active_crop_tiles(
                np.asarray(layer.data),
                output_dir=self.output_folder.text(),
                stem=layer.name,
                progress_callback=self._tile_progress_callback,
                **params,
            )
            self._set_progress(len(paths), total, "Saving crop tiles")
            self.status_label.setText(f"Wrote {len(paths)} crop TIFF files")
        except Exception as exc:
            self._show_error(str(exc))

    def export_batch_crops(self, mode: str):
        try:
            if not self.input_folder.text():
                self.select_input_folder()
            if not self.output_folder.text():
                self.select_output_folder()
            if not self.input_folder.text() or not self.output_folder.text():
                return
            if not self._batch_input_paths:
                raise ValueError("No TIFF files found in the selected input folder.")
            params = self._export_parameters(mode)
            total = count_batch_crop_tiles(self._batch_input_paths, **params)
            if not self._confirm_large_output_count(total):
                return
            self._start_progress(total, "Saving crop tiles")
            paths = batch_crop_tiff_files(
                self._batch_input_paths,
                output_dir=self.output_folder.text(),
                progress_callback=self._tile_progress_callback,
                **params,
            )
            self._set_progress(len(paths), total, "Saving crop tiles")
            self.status_label.setText(f"Wrote {len(paths)} crop TIFF files")
        except Exception as exc:
            self._show_error(str(exc))

    def preview(self):
        try:
            layer = self._active_image_layer()
            if not self._confirm_small_crop(layer):
                return
            self._start_progress(1, "Previewing crop")
            result, metadata = self._crop(layer)
            name = f"{layer.name}_crop_preview"
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
            self._set_progress(1, 1, "Previewing crop")
            self.status_label.setText(f"Updated {name}")
        except Exception as exc:
            self._show_error(str(exc))

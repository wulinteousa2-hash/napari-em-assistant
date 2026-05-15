import numpy as np
import pytest
import tifffile


def test_crop_image_2d_returns_expected_copy():
    from napari_em_assistant.tasks.crop_image.crop import crop_image

    image = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
    result = crop_image(image, y_range=(1, 4), x_range=(2, 5))

    assert np.array_equal(result, image[1:4, 2:5])
    assert result.dtype == image.dtype
    assert result.shape == (3, 3)
    assert result.base is None


def test_crop_image_3d_returns_expected_copy():
    from napari_em_assistant.tasks.crop_image.crop import crop_image

    image = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    result = crop_image(
        image,
        z_range=(1, 3),
        y_range=(1, 5),
        x_range=(0, 4),
    )

    assert np.array_equal(result, image[1:3, 1:5, 0:4])
    assert result.dtype == image.dtype
    assert result.shape == (2, 4, 4)
    assert result.base is None


def test_crop_image_rejects_invalid_bounds():
    from napari_em_assistant.tasks.crop_image.crop import crop_image

    image = np.zeros((8, 9), dtype=np.uint8)

    with pytest.raises(ValueError, match="y_end must be greater"):
        crop_image(image, y_range=(4, 4), x_range=(0, 5))

    with pytest.raises(ValueError, match="x_end must be <= 9"):
        crop_image(image, y_range=(0, 5), x_range=(0, 10))


def test_should_warn_small_crop_for_tiny_crop_from_large_image():
    from napari_em_assistant.tasks.crop_image.crop import (
        crop_fraction,
        should_warn_small_crop,
    )

    image_shape = (5000, 5000)
    y_range = (10, 50)
    x_range = (20, 60)

    assert crop_fraction(image_shape, y_range, x_range) < 0.01
    assert should_warn_small_crop(image_shape, y_range, x_range) is True


def test_should_not_warn_small_crop_for_small_source_image():
    from napari_em_assistant.tasks.crop_image.crop import should_warn_small_crop

    assert should_warn_small_crop((128, 128), (0, 8), (0, 8)) is False


def test_roi_bounds_from_2d_shape_data():
    from napari_em_assistant.tasks.crop_image.crop import roi_bounds_from_shape_data

    rectangle = np.array(
        [
            [1.2, 2.1],
            [1.2, 5.9],
            [4.8, 5.9],
            [4.8, 2.1],
        ]
    )

    z_range, y_range, x_range = roi_bounds_from_shape_data((10, 12), rectangle)

    assert z_range is None
    assert y_range == (1, 5)
    assert x_range == (2, 6)


def test_roi_bounds_from_3d_xy_shape_uses_full_z():
    from napari_em_assistant.tasks.crop_image.crop import roi_bounds_from_shape_data

    rectangle = np.array(
        [
            [7.0, 1.0, 2.0],
            [7.0, 1.0, 6.0],
            [7.0, 5.0, 6.0],
            [7.0, 5.0, 2.0],
        ]
    )

    z_range, y_range, x_range = roi_bounds_from_shape_data((20, 10, 12), rectangle)

    assert z_range == (0, 20)
    assert y_range == (1, 5)
    assert x_range == (2, 6)


def test_roi_bounds_from_3d_shape_with_z_extent():
    from napari_em_assistant.tasks.crop_image.crop import roi_bounds_from_shape_data

    box_projection = np.array(
        [
            [2.2, 1.0, 2.0],
            [5.8, 1.0, 6.0],
            [5.8, 5.0, 6.0],
            [2.2, 5.0, 2.0],
        ]
    )

    z_range, y_range, x_range = roi_bounds_from_shape_data(
        (20, 10, 12),
        box_projection,
    )

    assert z_range == (2, 6)
    assert y_range == (1, 5)
    assert x_range == (2, 6)


def test_crop_tiles_by_parts_splits_2d_into_near_equal_tiles():
    from napari_em_assistant.tasks.crop_image.crop import crop_tiles_by_parts

    tiles = crop_tiles_by_parts((10, 12), y_parts=2, x_parts=3)

    assert len(tiles) == 6
    assert [tile.y_range for tile in tiles[0::3]] == [(0, 5), (5, 10)]
    assert [tile.x_range for tile in tiles[:3]] == [(0, 4), (4, 8), (8, 12)]
    assert all(tile.z_range is None for tile in tiles)


def test_crop_tiles_by_size_keeps_full_z_by_default():
    from napari_em_assistant.tasks.crop_image.crop import crop_tiles_by_size

    tiles = crop_tiles_by_size((5, 10, 12), y_size=4, x_size=5)

    assert len(tiles) == 9
    assert tiles[0].z_range == (0, 5)
    assert tiles[0].y_range == (0, 4)
    assert tiles[0].x_range == (0, 5)
    assert tiles[-1].z_range == (0, 5)
    assert tiles[-1].y_range == (8, 10)
    assert tiles[-1].x_range == (10, 12)


def test_export_active_crop_tiles_by_size_writes_tiffs(tmp_path):
    from napari_em_assistant.tasks.crop_image.batch import export_active_crop_tiles

    image = np.arange(2 * 6 * 6, dtype=np.uint16).reshape(2, 6, 6)
    paths = export_active_crop_tiles(
        image,
        output_dir=tmp_path,
        stem="stack",
        mode="size",
        z_size=None,
        y_size=3,
        x_size=3,
    )

    assert len(paths) == 4
    assert paths[0].name == "stack_crop_z0-2_y0-3_x0-3.tif"
    assert tifffile.imread(paths[0]).shape == (2, 3, 3)
    assert np.array_equal(tifffile.imread(paths[-1]), image[:, 3:6, 3:6])


def test_export_active_crop_tiles_reports_progress(tmp_path):
    from napari_em_assistant.tasks.crop_image.batch import export_active_crop_tiles

    image = np.arange(6 * 6, dtype=np.uint8).reshape(6, 6)
    events = []

    paths = export_active_crop_tiles(
        image,
        output_dir=tmp_path,
        stem="image",
        mode="size",
        y_size=3,
        x_size=3,
        progress_callback=events.append,
    )

    assert len(paths) == 4
    assert [event["completed"] for event in events] == [1, 2, 3, 4]
    assert all(event["total"] == 4 for event in events)


def test_export_active_crop_tiles_manual_writes_one_tiff(tmp_path):
    from napari_em_assistant.tasks.crop_image.batch import export_active_crop_tiles

    image = np.arange(5 * 6, dtype=np.uint16).reshape(5, 6)
    paths = export_active_crop_tiles(
        image,
        output_dir=tmp_path,
        stem="image",
        mode="manual",
        y_range=(1, 4),
        x_range=(2, 5),
    )

    assert len(paths) == 1
    assert paths[0].name == "image_crop_y1-4_x2-5.tif"
    assert np.array_equal(tifffile.imread(paths[0]), image[1:4, 2:5])


def test_batch_crop_tiff_files_by_parts_writes_expected_count(tmp_path):
    from napari_em_assistant.tasks.crop_image.batch import batch_crop_tiff_files

    input_path = tmp_path / "input.tif"
    output_dir = tmp_path / "out"
    image = np.arange(4 * 4, dtype=np.uint8).reshape(4, 4)
    tifffile.imwrite(input_path, image)

    paths = batch_crop_tiff_files(
        [input_path],
        output_dir=output_dir,
        mode="parts",
        y_parts=2,
        x_parts=2,
    )

    assert len(paths) == 4
    assert tifffile.imread(output_dir / "input_crop_y0-2_x0-2.tif").shape == (2, 2)


def test_batch_crop_tiff_files_reports_total_progress(tmp_path):
    from napari_em_assistant.tasks.crop_image.batch import batch_crop_tiff_files

    input_paths = []
    for index in range(2):
        input_path = tmp_path / f"input_{index}.tif"
        tifffile.imwrite(input_path, np.zeros((4, 4), dtype=np.uint8))
        input_paths.append(input_path)
    events = []

    paths = batch_crop_tiff_files(
        input_paths,
        output_dir=tmp_path / "out",
        mode="parts",
        y_parts=2,
        x_parts=2,
        progress_callback=events.append,
    )

    assert len(paths) == 8
    assert events[-1]["completed"] == 8
    assert events[-1]["total"] == 8
    assert all(event["total"] == 8 for event in events)


def test_count_batch_crop_tiles_can_exceed_warning_threshold(tmp_path):
    from napari_em_assistant.tasks.crop_image.batch import count_batch_crop_tiles

    input_path = tmp_path / "large.tif"
    tifffile.imwrite(input_path, np.zeros((40, 40), dtype=np.uint8))

    total = count_batch_crop_tiles(
        [input_path],
        mode="size",
        y_size=1,
        x_size=1,
    )

    assert total == 1600


def test_crop_widget_imports():
    from napari_em_assistant.tasks.crop_image.widget import CropImageWidget

    assert CropImageWidget is not None

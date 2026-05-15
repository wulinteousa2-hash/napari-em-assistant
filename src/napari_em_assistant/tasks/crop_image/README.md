# Crop Image

This task crops 2D images and 3D grayscale stacks from napari.

## Four Crop Methods

The widget is organized into four tabs:

1. `Crop by Coordinates`: type exact `z`, `y`, and `x` start/end coordinates,
   then preview, create a cropped layer, or save one TIFF crop.
2. `Crop from Drawn ROI`: draw/select a napari Shapes ROI, choose its Shapes
   layer, then copy its bounds or crop directly.
3. `Tile by Total Count`: enter the total number of output tiles wanted, then
   save near-equal TIFF tiles. The plugin chooses the Y/X grid automatically.
4. `Tile by Pixel Size`: enter final tile dimensions, such as `512 x 512`, then
   save active-layer or batch TIFF tiles.

The progress bar reports created or saved crop pieces as `completed / total`.
The widget asks for confirmation before likely mistakes, including very tiny
crops from a large source image/stack and save operations that would create more
than `1000` crop TIFF files.

## Crop By Coordinates

Select an image layer and choose `Sync Active`. The widget loads the active
shape into start/end controls for `z`, `y`, and `x`. Start is inclusive and end
is exclusive, matching NumPy slicing. For 2D images, `z` controls are disabled.

Use `Preview Crop` to update a temporary crop layer, or `Create Cropped Layer`
to create a new `<source>_crop` layer.

## Crop From Drawn ROI

Draw a rectangle, polygon, or other ROI on a napari Shapes layer. Select that
Shapes layer in the crop widget and choose `Use ROI Bounds` to copy the ROI
bounding box into the numeric crop fields, or choose `Crop ROI` to crop the
active image immediately as a cropped layer.

For 2D images, the ROI bounds are interpreted as `y` and `x`. For 3D stacks, an
XY ROI crops the same `y/x` region through the full Z depth. If the shape data
has a real Z extent, that Z range is used.

## Batch And Automatic Cropping

Choose an output folder, then use one of the automatic tabs:

- `Tile by Total Count`: enter how many output tiles you want. The plugin
  chooses a near-even Y/X grid automatically. For example, `6` creates six tiles
  from a 2D image or six full-depth tiles from a 3D stack.
- `Tile by Pixel Size`: request the final tile size. For example, a
  `6000 x 6000 x 500` stack with `z size = 0`, `y size = 2000`, and
  `x size = 2000` writes nine `500 x 2000 x 2000` stack tiles. `z size = 0`
  keeps the full Z depth in each output.

Use the `Save Active...` buttons for the selected napari layer. Use the
`Save Folder...` buttons with an image folder to crop all `.tif` and
`.tiff` files in that folder. Output filenames include source bounds, for example
`stack_crop_z0-500_y0-2000_x0-2000.tif`.

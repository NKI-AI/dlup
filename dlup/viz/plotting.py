# coding=utf-8
# Copyright (c) DLUP Contributors

import io
import pathlib
from typing import List, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import PIL
import skimage.measure
from matplotlib.ticker import NullLocator
from PIL import Image, ImageDraw

from dlup import SlideImage

matplotlib.use("agg")


def draw_used_tiles(
    input_file_path: pathlib.Path, tile_size: int, thumbnail: np.ndarray, coords: List, original_mpp: float
) -> PIL.Image.Image:
    """
    Draw on the thumbnail image the tiles proposed by the background segmentation and foreground thresholding
    Converts the original resolution coordinates to thumbnail resolution coordinates and draws them there.

    Parameters
    ----------
    input_file_path: pathlib.Path
        path to WSI that we draw the used tiles for
    tile_size : int
        resulting tile size in px as given to the DLUP dataset class
    thumbnail : ndarray
        thumbnail of WSI to draw the tile locations on
    scaled_view_size : tuple
        size of WSI when viewed at the requested MPP. This is a (width, height) tuple for the requested level of the image.
    coords : list
        List of original mpp coordinates of the used tiles
    original_mpp: float
        the mpp of the originally requested tiles

    Examples
    --------
    Used in the wsi tiling CLI, but can also be used retrospectively by loading the coordinates.
    For details, see dlup.cli.wsi

    >>> with Pool(args.num_workers) as pool:
    >>>     for (grid_local_coordinates, local_coordinates, idx) in pool.imap(tile_iterator.iterate_tile, range(num_tiles)):
    >>>         indices[idx] = grid_local_coordinates
    >>>         coords[idx] = local_coordinates
    >>> thumbnail_with_used_tiles = draw_used_tiles(tile_size=tile_size,
    >>>                                         thumbnail=thumbnail,
    >>>                                         scaled_view_size=scaled_view_size,
    >>>                                         coords=coords)
    >>> thumbnail_with_used_tiles.save(output_directory_path / "thumbnail_with_used_tiles.png")

    Returns
    -------
    PIL Image
    """
    slide_image = SlideImage.from_file_path(input_file_path)
    scaled_view_size = slide_image.get_scaled_view(slide_image.get_scaling(original_mpp)).size
    thumbnail_with_used_tiles = Image.fromarray(thumbnail)
    draw = ImageDraw.Draw(thumbnail_with_used_tiles)

    # scaled_view_size is (width, height), whereas the thumbnail is (height, width)
    # the rounding with (int) this may cause minor rounding errors displacing boundaries by a pixel. this is not
    # detrimental for this functionality
    converted_tile_size = (
        int(tile_size / (scaled_view_size[1] / thumbnail.shape[0])),
        int(tile_size / (scaled_view_size[0] / thumbnail.shape[1])),
    )
    coordinate_scales = ((scaled_view_size[1] / thumbnail.shape[0]), (scaled_view_size[0] / thumbnail.shape[1]))

    for coord in coords:
        coord = np.array(coord)
        adjusted_coords = np.array([int(coord[0] / coordinate_scales[0]), int(coord[1] / coordinate_scales[1])])
        box = tuple(np.array((*adjusted_coords, *(adjusted_coords + converted_tile_size))).astype(int))
        draw.rectangle(box, outline="red")  # shows the tile
        draw.line(
            box, fill="red"
        )  # shows a diagonal through the tile, to discern unused tiles surrounded by tiles from actually used tiles
    return thumbnail_with_used_tiles


def plot_2d(
    image,
    mask=None,
    bboxes=None,
    contours=None,
    points=None,
    overlay=None,
    linewidth=2,
    mask_color="r",
    points_color="g",
    overlay_cmap="jet",
    overlay_threshold=0.1,
    overlay_alpha=0.1,
):

    """
    Plot image with contours and point annotations. Contours are automatically extracted from masks.

    Parameters
    ----------
    image : ndarray
        Image data of shape N x M x num channels.
    mask : ndarray
        Mask data of shape N x M.
    bboxes : tuple
        Bounding boxes to overlay. In the form of [[x, y, h, w], ...]. Can also use it with colors, in that case
        use a tuple: [([x, y, h, w], "red"), ...]. The color string has to be matplotlib supported.
    contours : list
        List of contours in form (N, 2).
    points : ndarray
        ndarray of shape (num points, 2).
    overlay : ndarray
        Heatmap of data.
    linewidth : int
        Width of plotted lines.
    mask_color : str
        Matplotlib supported color for the masks.
    points_color : str
        Matplotlib supported color for the points.
    overlay_cmap : str
    overlay_threshold : float
        Value below which overlay should not be displayed.
    overlay_alpha : float
        Overlay alpha

    Examples
    --------
    Can be used in Tensorboard, for instance as follows:

    >>> plot_overlay = torch.from_numpy(np.array(plot_2d(image_arr, mask=masks_arr)))
    >>> writer.add_image("train/overlay", plot_overlay, epoch, dataformats="HWC")

    Returns
    -------
    PIL Image
    """
    dpi = 100
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)

    fig, axis = plt.subplots(1, figsize=figsize)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    cmap = None
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = image[..., 0]
        elif image.shape[0] == 1:
            image = image[0, ...]

    if image.ndim == 2:
        cmap = "gray"

    axis.imshow(image, cmap=cmap, aspect="equal", extent=(0, width, height, 0))
    axis.set_adjustable("datalim")

    if mask is not None:
        add_2d_contours(mask, axis, linewidth, mask_color)

    if bboxes is not None:
        for bbox in bboxes:
            if isinstance(bbox, (tuple, list)) and len(bbox) == 2:  # TODO: This can use a more elaborate check.
                bbox = bbox[0]
                bbox_color = bbox[1]
            else:
                bbox_color = None
            add_2d_bbox(bbox, axis, linewidth, bbox_color)

    if contours is not None:
        for contour in contours:
            # TODO: Custom color
            axis.plot(*contour[:, [1, 0]].T, color=mask_color, linewidth=linewidth)

    if overlay is not None:
        add_2d_overlay(
            overlay,
            axis,
            threshold=overlay_threshold,
            cmap=overlay_cmap,
            alpha=overlay_alpha,
        )

    if points is not None:
        axis.plot(points[:, 1], points[:, 0], points_color + ".", markersize=2, alpha=1)

    fig.gca().set_axis_off()
    fig.gca().xaxis.set_major_locator(NullLocator())
    fig.gca().yaxis.set_major_locator(NullLocator())

    buffer = io.BytesIO()

    fig.savefig(buffer, pad_inches=0, dpi=dpi)
    buffer.seek(0)
    plt.close()

    pil_image = PIL.Image.open(buffer)

    return pil_image


def add_2d_bbox(bbox, axis, linewidth=0.5, color="b"):
    """Add bounding box to the image.

    Parameters
    ----------
    bbox : tuple
        Tuple of the form (row, col, height, width).
    axis : axis object
    linewidth : float
        thickness of the overlay lines
    color : str
        matplotlib supported color string for contour overlay.
    """
    rect = mpatches.Rectangle(
        bbox[:2][::-1],
        bbox[3],
        bbox[2],
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
    )
    axis.add_patch(rect)


def add_2d_contours(mask, axis, linewidth=0.5, color="r"):
    """Plot the contours around the `1`'s in the mask

    Parameters
    ----------
    mask : ndarray
        2D binary array.
    axis : axis object
        matplotlib axis object.
    linewidth : float
        thickness of the overlay lines.
    color : str
        matplotlib supported color string for contour overlay.
    """
    contours = skimage.measure.find_contours(mask, 0.5)

    for contour in contours:
        axis.plot(*contour[:, [1, 0]].T, color=color, linewidth=linewidth)


def add_2d_overlay(overlay, axis, threshold=0.1, cmap="jet", alpha=0.1):
    """Adds an overlay of the probability map and predicted regions

    Parameters
    ----------
    overlay : ndarray
    axis : axis object
       matplotlib axis object
    threshold : float
    cmap : str
       matplotlib supported cmap.
    alpha : float
       alpha values for the overlay.
    """
    if threshold:
        overlay = ma.masked_where(overlay < threshold, overlay)

    axis.imshow(overlay, cmap=cmap, alpha=alpha)

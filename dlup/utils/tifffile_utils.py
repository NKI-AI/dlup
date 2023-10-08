# Copyright (c) dlup contributors
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile


def _validate_inputs(page, coordinates, size):
    x0, y0 = coordinates
    w, h = size

    if not page.is_tiled:
        raise ValueError("Input page must be tiled.")

    image_width = page.imagewidth
    image_height = page.imagelength

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    if y0 < 0 or x0 < 0 or y0 + h > image_height or x0 + w > image_width:
        raise ValueError("Requested crop area is out of image bounds.")


def _compute_tile_indices(page, coordinates, size):
    x0, y0 = coordinates
    w, h = size
    tile_width, tile_height = page.tilewidth, page.tilelength
    y1, x1 = y0 + h, x0 + w

    tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width
    tile_y1, tile_x1 = np.ceil([y1 / tile_height, x1 / tile_width]).astype(int)

    return tile_y0, tile_y1, tile_x0, tile_x1


def retrieve_tile_data(page, tile_y0, tile_y1, tile_x0, tile_x1):
    tile_per_line = int(np.ceil(page.imagewidth / page.tilewidth))
    fh = page.parent.filehandle
    jpeg_tables = page.tags.get("JPEGTables", None)
    if jpeg_tables is not None:
        jpeg_tables = jpeg_tables.value

    tiles_data = []
    for idx_y in range(tile_y0, tile_y1):
        for idx_x in range(tile_x0, tile_x1):
            index = int(idx_y * tile_per_line + idx_x)
            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            if not bytecount:
                continue

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, _, _ = page.decode(data, index, jpegtables=jpeg_tables)
            tiles_data.append(((idx_y, idx_x), tile))
    return tiles_data


def _get_tile_from_data(page, tiles_data, coordinates, size) -> npt.NDArray[np.int_]:
    x0, y0 = coordinates
    w, h = size
    tile_width, tile_height = page.tilewidth, page.tilelength
    tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width

    out = np.zeros(
        (page.imagedepth, h, w, page.samplesperpixel),
        dtype=page.dtype,
    )

    if not tiles_data:
        return out

    for (idx_y, idx_x), tile in tiles_data:
        image_y = (idx_y - tile_y0) * tile_height
        image_x = (idx_x - tile_x0) * tile_width
        out[:, image_y : image_y + tile_height, image_x : image_x + tile_width, :] = tile

    image_y0 = y0 - tile_y0 * tile_height
    image_x0 = x0 - tile_x0 * tile_width
    return out[:, image_y0 : image_y0 + h, image_x0 : image_x0 + w, :]


def _retrieve_tile_data(page, tile_y0, tile_y1, tile_x0, tile_x1):
    tile_per_line = int(np.ceil(page.imagewidth / page.tilewidth))
    fh = page.parent.filehandle
    jpeg_tables = page.tags.get("JPEGTables", None)
    if jpeg_tables is not None:
        jpeg_tables = jpeg_tables.value

    tiles_data = []
    for idx_y in range(tile_y0, tile_y1):
        for idx_x in range(tile_x0, tile_x1):
            index = int(idx_y * tile_per_line + idx_x)
            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            if not bytecount:
                continue

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, _, _ = page.decode(data, index, jpegtables=jpeg_tables)
            tiles_data.append(((idx_y, idx_x), tile))
    return tiles_data


def get_tile(
    page: tifffile.TiffPage,
    coordinates: npt.NDArray[np.int_] | tuple[int, int],
    size: npt.NDArray[np.int_] | tuple[int, int],
) -> npt.NDArray[np.int_]:
    """Extract a crop from a TIFF image file directory (IFD).

    Only the tiles englobing the crop area are loaded and not the whole page.
    This is useful for large Whole slide images that can't fit into RAM.

    Code obtained from [1].

    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    coordinates: (int, int)
        Coordinates of the top left and right corner corner of the desired crop.
    size: (int, int)
        Desired crop height and width.

    References
    ----------
    .. [1] https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743

    Returns
    -------
    out : ndarray of shape (imagedepth, h, w, sampleperpixel)
        Extracted crop.

    """
    _validate_inputs(page, coordinates, size)
    tile_y0, tile_y1, tile_x0, tile_x1 = _compute_tile_indices(page, coordinates, size)
    tiles_data = _retrieve_tile_data(page, tile_y0, tile_y1, tile_x0, tile_x1)
    return _get_tile_from_data(page, tiles_data, coordinates, size)

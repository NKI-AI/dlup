# Copyright (c) dlup contributors
from typing import Any

import numpy as np
import numpy.typing as npt
import tifffile


def get_tile(page: tifffile.TiffPage, coordinates: tuple[Any, ...], size: tuple[Any, ...]) -> npt.NDArray[np.int_]:
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

    tile_width, tile_height = page.tilewidth, page.tilelength
    y1, x1 = y0 + h, x0 + w

    tile_y0, tile_x0 = y0 // tile_height, x0 // tile_width
    tile_y1, tile_x1 = np.ceil([y1 / tile_height, x1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(image_width / tile_width))

    out = np.zeros(
        (page.imagedepth, (tile_y1 - tile_y0) * tile_height, (tile_x1 - tile_x0) * tile_width, page.samplesperpixel),
        dtype=page.dtype,
    )

    fh = page.parent.filehandle

    jpeg_tables = page.tags.get("JPEGTables", None)
    if jpeg_tables is not None:
        jpeg_tables = jpeg_tables.value

    for idx_y in range(tile_y0, tile_y1):
        for idx_x in range(tile_x0, tile_x1):
            index = int(idx_y * tile_per_line + idx_x)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            # Some files written by ASAP have an empty bytecount if it is empty.
            if not bytecount:
                continue

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index, jpegtables=jpeg_tables)

            image_y = (idx_y - tile_y0) * tile_height
            image_x = (idx_x - tile_x0) * tile_width
            out[:, image_y : image_y + tile_height, image_x : image_x + tile_width, :] = tile

    image_y0 = y0 - tile_y0 * tile_height
    image_x0 = x0 - tile_x0 * tile_width

    return out[:, image_y0 : image_y0 + h, image_x0 : image_x0 + w, :]

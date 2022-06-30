# coding=utf-8
# Copyright (c) dlup contributors
import os
from typing import Dict, Tuple

import numpy as np
import PIL.Image
import tifffile

from dlup.backends.common import AbstractSlideBackend, check_mpp


def open_slide(filename: os.PathLike) -> "TifffileSlide":
    return TifffileSlide(filename)


def get_tile(page: tifffile.TiffPage, coordinates: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    # https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743

    """Extract a crop from a TIFF image file directory (IFD).

    Only the tiles englobing the crop area are loaded and not the whole page.
    This is usefull for large Whole slide images that can't fit int RAM.

    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    coordinates: (int, int)
        Coordinates of the top left and right corner corner of the desired crop.
    size: (int, int)
        Desired crop height and width.

    Returns
    -------
    out : ndarray of shape (imagedepth, h, w, sampleperpixel)
        Extracted crop.

    """
    y0, x0 = coordinates
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

    out = np.empty(
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

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index, jpegtables=jpeg_tables)

            image_y = (idx_y - tile_y0) * tile_height
            image_x = (idx_x - tile_x0) * tile_width
            out[:, image_y : image_y + tile_height, image_x : image_x + tile_width, :] = tile

    image_y0 = y0 - tile_y0 * tile_height
    image_x0 = x0 - tile_x0 * tile_width

    return out[:, image_y0 : image_y0 + h, image_x0 : image_x0 + w, :]


class TifffileSlide(AbstractSlideBackend):
    def __init__(self, path: os.PathLike) -> None:
        super().__init__(path)
        self._image = tifffile.TiffFile(path)
        self._level_count = len(self._image.pages)
        self.__parse_tifffile()

    def __parse_tifffile(self) -> None:
        unit_dict = {1: 1, 2: 254000, 3: 100000, 4: 1000000, 5: 10000000}
        self._downsamples.append(1.0)
        for idx, page in enumerate(self._image.pages):
            self._shapes.append(page.shape[::-1])

            x_res = page.tags["XResolution"].value
            x_res = x_res[0] / x_res[1]
            y_res = page.tags["YResolution"].value
            y_res = y_res[0] / y_res[1]
            unit = int(page.tags["ResolutionUnit"].value)

            mpp_x = unit_dict[unit] / x_res
            mpp_y = unit_dict[unit] / y_res
            check_mpp(mpp_x, mpp_y)
            self._spacings.append(mpp_x)

            if idx >= 1:
                downsample = mpp_x / self._spacings[0]
                self._downsamples.append(downsample)

    @property
    def properties(self) -> Dict:
        """Metadata about the image.
        This is a map: property name -> property value."""
        properties = {}
        for idx, page in enumerate(self._image.pages):
            for tag in page.tags:
                # These tags are not so relevant at this point and have a lot of output
                if tag.name in [
                    "TileOffsets",
                    "TileByteCounts",
                    "SMinSampleValue",
                    "JPEGTables",
                    "ReferenceBlackWhite",
                ]:
                    continue

                properties[f"tifffile.level[{idx}].{tag.name}"] = tag.value

        return properties

    def set_cache(self, cache):
        """Cache for tifffile."""
        raise NotImplementedError

    def read_region(self, coordinates: Tuple[int, int], level: int, size: Tuple[int, int]) -> PIL.Image:
        if level > self._level_count - 1:
            raise RuntimeError(f"Level {level} not present.")

        page = self._image.pages[level]
        tile = get_tile(page, coordinates, size)[0]
        bands = tile.shape[-1]

        if bands == 1:
            mode = "L"
            tile = tile[:, :, 0]
        elif bands == 3:
            mode = "RGB"
        elif bands == 4:
            mode = "RGBA"
        else:
            raise RuntimeError(f"Incorrect number of channels.")

        return PIL.Image.fromarray(tile, mode=mode)

    def close(self):
        self._image.close()

# coding=utf-8
# Copyright (c) dlup contributors
import os
from enum import Enum
from typing import Iterable, Optional, Union

import numpy as np
import pyvips
from tqdm import tqdm

from dlup.utils.vips import numpy_to_vips


class TiffCompression(Enum):
    NONE = "none"  # No compression
    CCITTFAX4 = "ccittfax4"  # Fax4 compression
    JPEG = "jpeg"  # Jpeg compression
    DEFLATE = "deflate"  # zip compression
    PACKBITS = "packbits"  # packbits compression
    LZW = "lzw"  # LZW compression
    WEBP = "webp"  # WEBP compression
    ZSTD = "zstd"  # ZSTD compression
    JP2K = "jp2k"  # JP2K compression


class ImageWriter:
    """Base writer class"""


class TiffImageWriter(ImageWriter):
    """Image writer that writes tile-by-tile to tiff."""

    def __init__(
        self,
        mpp,
        size,
        tile_height=256,
        tile_width=256,
        pyramid=False,
        compression: TiffCompression = TiffCompression.NONE,
        quality: Optional[int] = None,
    ):
        self._tile_height = tile_height
        self._tile_width = tile_width
        self._size = size
        self._mpp = mpp
        self._compression = compression
        self._pyramid = pyramid
        self._quality = quality

    def from_iterator(self, iterator: Iterable, save_path: Union[str, os.PathLike]):
        vips_image = None
        bit_depth = None  # This is 1, 2, 4, 8
        for tile_index, (tile_coordinates, _tile) in tqdm(enumerate(iterator)):
            _tile = np.asarray(_tile)
            if bit_depth is None:
                if _tile.dtype == np.dtype("bool"):
                    bit_depth = 1
                elif _tile.dtype == np.dtype("uint8"):
                    bit_depth = 8
                else:
                    ValueError(f"I am not so sure about this bit_depth.")

            if vips_image is None:
                vips_image = pyvips.Image.black(*self._size, bands=_tile.shape[-1])

            vips_tile = numpy_to_vips(_tile)
            vips_image = vips_image.insert(vips_tile, *tile_coordinates)

        # TODO: How to figure out the bit depth?
        # TODO: This will take significant memory resources. Can they be written tile by tile?
        self._save_tiff(vips_image, save_path, bitdepth=bit_depth)

    def _save_tiff(self, vips_image: pyvips.Image, save_path: Union[str, os.PathLike], bitdepth: int = 8):
        vips_image.tiffsave(
            str(save_path),
            compression=self._compression.value,
            tile=True,
            tile_width=self._tile_width,
            tile_height=self._tile_width,
            xres=self._mpp[0],
            yres=self._mpp[1],
            pyramid=self._pyramid,
            squash=True,
            bitdepth=bitdepth,
            bigtiff=True,
            depth="onetile" if self._pyramid else "one",
            background=[0],
            Q=self._quality,
        )

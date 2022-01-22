# coding=utf-8
# Copyright (c) dlup contributors
import os
from enum import Enum
from typing import Iterable, Optional, Union

import numpy as np
import PIL.Image

from dlup.utils.imports import PYVIPS_AVAILABLE

if PYVIPS_AVAILABLE:
    import pyvips
    from dlup.utils.vips import numpy_to_vips

from tqdm import tqdm

from dlup.tiling import Grid, TilingMode


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
        quality: int = 100,
        bit_depth: int = 8,
        silent: bool = False,
    ):
        if not PYVIPS_AVAILABLE:
            raise RuntimeError(f"pyvips needs to be installed to use TiffImageWriter.")

        self._tile_height = tile_height
        self._tile_width = tile_width
        self._size = size
        self._mpp = mpp
        self._compression = compression
        self._pyramid = pyramid
        self._quality = quality
        self._bit_depth = bit_depth
        if bit_depth not in [1, 2, 4, 8]:
            raise ValueError(f"bit_depth can only be 1, 2, 4 or 8.")

        self._silent = silent

    def from_iterator(self, iterator: Iterable, save_path: Union[str, os.PathLike], total: Optional[int] = None):
        vips_image = None
        for tile_index, (tile_coordinates, _tile) in tqdm(
            enumerate(iterator), disable=self._silent, unit="tiles", total=total
        ):
            _tile = np.asarray(_tile)
            if vips_image is None:
                # Assumes last axis is the channel!
                vips_image = pyvips.Image.black(*self._size, bands=_tile.shape[-1])

            vips_tile = numpy_to_vips(_tile)
            vips_image = vips_image.insert(vips_tile, *tile_coordinates)

        # TODO: This will take significant memory resources. Can they be written tile by tile?
        self._save_tiff(vips_image, save_path)

    def from_pil(self, pil_image: PIL.Image.Image, save_path: Union[str, os.PathLike]):
        iterator = self._grid_iterator(pil_image)
        writer = TiffImageWriter(
            mpp=self._mpp, size=pil_image.size, compression=self._compression, bit_depth=self._bit_depth
        )
        writer.from_iterator(iterator, save_path=save_path)

    def _save_tiff(self, vips_image: pyvips.Image, save_path: Union[str, os.PathLike]):
        vips_image.set_type(pyvips.GValue.gdouble_type, "dlup.mpp_x", self._mpp[0])
        vips_image.set_type(pyvips.GValue.gdouble_type, "dlup.mpp_y", self._mpp[1])
        vips_image.tiffsave(
            str(save_path),
            compression=self._compression.value,
            tile=True,
            tile_width=self._tile_width,
            tile_height=self._tile_width,
            xres=1 / self._mpp[0],
            yres=1 / self._mpp[1],
            pyramid=self._pyramid,
            squash=True,
            bitdepth=self._bit_depth,
            properties=True,
            bigtiff=True,
            depth="onetile" if self._pyramid else "one",
            background=[0],
            Q=self._quality,
        )

    def _grid_iterator(self, pil_image):
        _tile_size = (self._tile_height, self._tile_width)
        grid = Grid.from_tiling(
            (0, 0),
            size=pil_image.size[::-1],
            tile_size=_tile_size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
        )
        for tile_coordinates in grid:
            arr = np.asarray(pil_image)
            cropped_mask = arr[
                tile_coordinates[0] : tile_coordinates[0] + _tile_size[0],
                tile_coordinates[1] : tile_coordinates[1] + _tile_size[1],
            ]
            yield tile_coordinates[::-1], cropped_mask[..., np.newaxis].astype("uint8")

# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations

import pathlib
from enum import Enum
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import PIL.Image
from tifffile import tifffile

import dlup
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.types import PathLike
from dlup.utils.pyvips import numpy_to_vips, vips_to_numpy
from dlup.utils.tifffile import get_tile


class TiffCompression(str, Enum):
    NONE = "none"  # No compression
    CCITTFAX4 = "ccittfax4"  # Fax4 compression
    JPEG = "jpeg"  # Jpeg compression
    DEFLATE = "deflate"  # zip compression
    PACKBITS = "packbits"  # packbits compression
    #     LZW = "lzw"  # LZW compression, not implemented in tifffile
    WEBP = "webp"  # WEBP compression
    ZSTD = "zstd"  # ZSTD compression
    JP2K = "jp2k"  # JP2K compression


TIFFFILE_COMPRESSION = {
    "none": None,
    "ccittfax4": "CCITT_T4",
    "jpeg": "jpeg",
    "deflate": "deflate",
    "packbits": "packbits",
    "lzw": "lzw",
    "webp": "webp",
    "zstd": "zstd",
    "jp2k": "jpeg2000",
}


class ImageWriter:
    """Base writer class"""


class TifffileImageWriter(ImageWriter):
    """Image writer that writes tile-by-tile to tiff."""

    def __init__(
        self,
        size: Union[Tuple[int, int, int], Tuple[int, int]],
        mpp: Union[float, Tuple[float, float]],
        tile_height: int = 512,
        tile_width: int = 512,
        pyramid: bool = False,
        compression: Optional[TiffCompression] = TiffCompression.JPEG,
        quality: Optional[int] = 100,
    ):
        """
        Writer based on tifffile.

        Parameters
        ----------
        size : Tuple
            Size of the image to be written. This is defined as (height, width, num_channels),
            or rather (rows, columns, num_channels) and is important value to get correct.
            In case of a mask with a single channel the value is given by (rows, columns).
        mpp : int, or Tuple[int, int]
        tile_height : int
        tile_width : int
        pyramid : bool
            Whether to write a pyramidal image.
        compression
        quality
        """
        self._tile_size = (tile_height, tile_width)
        if len(size) == 2:
            size = (*size, 1)
        self._size = size

        if isinstance(mpp, float):
            mpp = (mpp, mpp)
        self._mpp: Tuple[float, float] = mpp

        if not compression:
            compression = TiffCompression.NONE

        self._compression = compression
        self._pyramid = pyramid
        self._quality = quality

    def from_pil(self, pil_image: PIL.Image, filename: PathLike):
        if not np.all(np.asarray(pil_image.size)[::-1] >= self._tile_size):
            raise RuntimeError(
                f"PIL Image must be larger than set tile size. Got {pil_image.size} and {self._tile_size}."
            )
        iterator = _pil_grid_iterator(pil_image, self._tile_size)
        self.from_iterator(iterator, filename=filename)

    def from_iterator(self, iterator: Iterator[np.ndarray | None], filename: PathLike):
        filename = pathlib.Path(filename)
        temp_filename = filename.with_suffix(f"{filename.suffix}.partial")

        native_size = self._size[:-1]
        software = f"dlup {dlup.__version__} with tifffile.py backend"
        n_subresolutions = 0
        if self._pyramid:
            n_subresolutions = int(np.ceil(np.log2(np.asarray(native_size) / np.asarray(self._tile_size))).min())
        shapes = [
            np.floor(np.asarray(native_size) / 2**n).astype(int).tolist() for n in range(0, n_subresolutions + 1)
        ]

        native_resolution = 1 / np.array(self._mpp) * 10000
        metadata = {
            # "axes": "TCYXS",
            # "SignificantBits": 10,
            "PhysicalSizeX": self._mpp[0],
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": self._mpp[1],
            "PhysicalSizeYUnit": "µm",
        }

        _compression = TIFFFILE_COMPRESSION[self._compression.value]

        is_rgb = self._size[-1] in (3, 4)
        with tifffile.TiffWriter(temp_filename, bigtiff=True) as tiff_writer:
            tiff_writer.write(
                iterator,  # noqa
                shape=(*shapes[0], self._size[-1]) if is_rgb else (*shapes[0], 1),
                dtype="uint8",
                subifds=None,
                resolution=(*native_resolution, "CENTIMETER"),  # noqa
                metadata=metadata,
                photometric="rgb" if is_rgb else "minisblack",
                compression=_compression if not self._quality else (_compression, self._quality),
                tile=self._tile_size,
                software=software,
            )

            for level in range(0, n_subresolutions):
                with tifffile.TiffReader(temp_filename) as tiff_reader:
                    page = tiff_reader.pages[level]
                    tile_iterator = _new_tile_iterator(page, self._tile_size, shapes[level], scale=2, is_rgb=is_rgb)
                    tiff_writer.write(
                        tile_iterator,  # noqa
                        shape=(*shapes[level + 1], self._size[-1]) if is_rgb else (*shapes[level + 1], 1),
                        dtype="uint8",
                        subfiletype=1,
                        resolution=(*native_resolution / 2 ** (level + 1), "CENTIMETER"),  # noqa
                        photometric="rgb" if is_rgb else "minisblack",
                        compression=_compression if not self._quality else (_compression, self._quality),
                        tile=self._tile_size,
                        software=software,
                    )

        temp_filename.rename(filename)


def _pil_grid_iterator(pil_image: PIL.Image, tile_size: Tuple[int, int]):
    grid = Grid.from_tiling(
        (0, 0),
        size=pil_image.size[::-1],
        tile_size=tile_size,
        tile_overlap=(0, 0),
        mode=TilingMode.overflow,
        order=GridOrder.F,
    )
    for tile_coordinates in grid:
        arr = np.asarray(pil_image)
        cropped_array = arr[
            tile_coordinates[0] : tile_coordinates[0] + tile_size[0],
            tile_coordinates[1] : tile_coordinates[1] + tile_size[1],
        ].astype("uint8")
        yield cropped_array


def _new_tile_iterator(
    page: tifffile.TiffPage, tile_size: Tuple[int, int], region_size: Tuple[int, int], scale: int, is_rgb: bool = True
):
    resized_tile_size = tuple(map(lambda x: x * scale, tile_size))
    grid = Grid.from_tiling(
        (0, 0),
        size=region_size,
        tile_size=resized_tile_size,
        tile_overlap=(0, 0),
        mode=TilingMode.overflow,
    )
    for coordinates in grid:
        # The tile size must be cropped to image bounds
        region_end = coordinates + resized_tile_size
        size = np.clip(region_end, 0, region_size) - coordinates

        tile = get_tile(page, coordinates[::-1], size[::-1])[0]
        vips_tile = numpy_to_vips(tile).resize(1 / scale)
        output = vips_to_numpy(vips_tile)
        if not is_rgb:
            output = output[..., 0]
        yield output

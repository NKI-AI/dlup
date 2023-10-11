# Copyright (c) dlup contributors
"""
Classes to write image and mask files
"""
from __future__ import annotations

import pathlib
import shutil
import tempfile
from enum import Enum
from typing import Any, Generator, Iterator

import numpy as np
import numpy.typing as npt
import PIL.Image
from pyvips.enums import Kernel
from tifffile import tifffile

import dlup
from dlup._image import Resampling
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.types import PathLike
from dlup.utils.pyvips_utils import numpy_to_vips, vips_to_numpy
from dlup.utils.tifffile_utils import get_tile


class TiffCompression(str, Enum):
    NONE = "none"  # No compression
    CCITTFAX4 = "ccittfax4"  # Fax4 compression
    JPEG = "jpeg"  # Jpeg compression
    DEFLATE = "deflate"  # zip compression
    PACKBITS = "packbits"  # packbits compression
    LZW = "lzw"  # LZW compression, not implemented in tifffile
    WEBP = "webp"  # WEBP compression
    ZSTD = "zstd"  # ZSTD compression
    JP2K = "jp2k"  # JP2K compression
    JP2K_LOSSY = "jp2k_lossy"
    PNG = "png"


# Mapping to map TiffCompression to their respective values in tifffile.
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
    "jp2k_lossy": "jpeg_2000_lossy",
    "png": "png",
}

# Mapping to map TiffCompression to their respective values in VIPS.
INTERPOLATOR_TO_VIPS: dict[int, Kernel] = {
    0: Kernel.NEAREST,
    2: Kernel.LINEAR,
    3: Kernel.CUBIC,
    1: Kernel.LANCZOS3,
}


class ImageWriter:
    """Base writer class"""


class TifffileImageWriter(ImageWriter):
    """Image writer that writes tile-by-tile to tiff."""

    def __init__(
        self,
        filename: PathLike,
        size: tuple[int, int] | tuple[int, int, int],
        mpp: float | tuple[float, float],
        tile_size: tuple[int, int] = (512, 512),
        pyramid: bool = False,
        compression: TiffCompression | None = TiffCompression.JPEG,
        interpolator: Resampling | None = Resampling.LANCZOS,
        anti_aliasing: bool = False,
        quality: int | None = 100,
        metadata: dict[str, str] | None = None,
    ):
        """
        Writer based on tifffile.

        Parameters
        ----------
        filename : PathLike
            Filename where to write
        size : tuple
            Size of the image to be written. This is defined as (height, width, num_channels),
            or rather (rows, columns, num_channels) and is important value to get correct.
            In case of a mask with a single channel the value is given by (rows, columns).
        mpp : int, or tuple[int, int]
        tile_size : tuple[int, int]
            Tiff tile_size, defined as (height, width).
        pyramid : bool
            Whether to write a pyramidal image.
        compression : TiffCompression
            Compressor to use.
        interpolator : Resampling, optional
            Interpolator to use when downsampling. For masks you should select nearest, as the interpolation
            could otherwise lead to unexpected results (i.e. add values which are actually not there!). By
            default the Lanczos interpolator is used.
        anti_aliasing : bool, optional
            Whether to use anti-aliasing when downsampling. By default this is set to False.
        quality : int
            Quality in case a lossy compressor is used.
        metadata : dict[str, str]
            Metadata to write to the tiff file.
        """
        self._filename = pathlib.Path(filename)
        self._tile_size = tile_size

        self._size = (*size[::-1], 1) if len(size) == 2 else (size[1], size[0], size[2])  # type: ignore
        self._mpp: tuple[float, float] = (mpp, mpp) if isinstance(mpp, (int, float)) else mpp

        if compression is None:
            compression = TiffCompression.NONE

        if interpolator is None:
            interpolator = Resampling.LANCZOS

        if interpolator.value not in INTERPOLATOR_TO_VIPS:
            raise ValueError(f"Invalid interpolator: {interpolator.name}")

        if anti_aliasing and interpolator == Resampling.NEAREST:
            raise ValueError("Anti-aliasing cannot be used with nearest neighbor interpolation.")
        elif anti_aliasing:
            raise NotImplementedError("Anti-aliasing is not yet implemented.")

        self._anti_aliasing = anti_aliasing
        self._compression = compression
        self._interpolator = interpolator
        self._pyramid = pyramid
        self._quality = quality
        self._metadata = metadata

    def from_pil(self, pil_image: PIL.Image.Image) -> None:
        """
        Create tiff image from a PIL image

        Parameters
        ----------
        pil_image : PIL.Image
        """
        if not np.all(np.asarray(pil_image.size)[::-1] >= self._tile_size):
            raise RuntimeError(
                f"PIL Image must be larger than set tile size. Got {pil_image.size} and {self._tile_size}."
            )
        iterator = _tiles_iterator_from_pil_image(pil_image, self._tile_size)
        self.from_tiles_iterator(iterator)

    def from_tiles_iterator(self, iterator: Iterator[npt.NDArray[np.int_]]) -> None:
        """
        Generate the tiff from a tiles iterator. The tiles should be in row-major (C-order) order.
        The `dlup.tiling.Grid` class has the possibility to generate such grids using `GridOrder.C`.

        Parameters
        ----------
        iterator : Iterator
            Iterator providing the tiles as numpy arrays.
            They are expected to be (tile_height, tile_width, num_channels) when RGB(A) images or
            (tile_height, tile_width) when masks. The tiles can be smaller at the border.
        """
        filename = pathlib.Path(self._filename)

        native_size = self._size[:-1]
        software = f"dlup {dlup.__version__} with tifffile.py backend"
        n_subresolutions = 0
        if self._pyramid:
            n_subresolutions = int(np.ceil(np.log2(np.asarray(native_size) / np.asarray(self._tile_size))).min())
        shapes = [
            np.floor(np.asarray(native_size) / 2**n).astype(int).tolist() for n in range(0, n_subresolutions + 1)
        ]

        # TODO: add to metadata "axes": "TCYXS", and "SignificantBits": 10,
        metadata = {
            "PhysicalSizeX": self._mpp[0],
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": self._mpp[1],
            "PhysicalSizeYUnit": "µm",
        }
        if self._metadata is not None:
            metadata = {**metadata, **self._metadata}

        # Convert the compression variable to a tifffile supported one.
        _compression = TIFFFILE_COMPRESSION[self._compression.value]

        is_rgb = self._size[-1] in (3, 4)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filename = pathlib.Path(temp_dir) / filename.name
            tiff_writer = tifffile.TiffWriter(temp_filename, bigtiff=True)
            self._write_page(
                tiff_writer,
                tile_iterator=iterator,
                level=0,
                compression=_compression,
                shapes=shapes,
                is_rgb=is_rgb,
                subifds=None,
                software=software,
                metadata=metadata,
            )

            for level in range(0, n_subresolutions):
                tiff_reader = tifffile.TiffReader(temp_filename)
                page = tiff_reader.pages[level]
                tile_iterator = _tile_iterator_from_page(
                    page,  # type: ignore
                    self._tile_size,
                    shapes[level],
                    scale=2,
                    is_rgb=is_rgb,
                    interpolator=self._interpolator,
                )
                self._write_page(
                    tiff_writer,
                    tile_iterator=tile_iterator,
                    level=level + 1,
                    compression=_compression,
                    shapes=shapes,
                    is_rgb=is_rgb,
                    subfiletype=1,
                    software=software,
                )
                tiff_reader.close()
            tiff_writer.close()
            shutil.move(str(temp_filename), str(filename))

    def _write_page(
        self,
        tiff_writer: tifffile.TiffWriter,
        tile_iterator: Iterator[npt.NDArray[np.int_]],
        level: int,
        compression: str | None,
        shapes: list[tuple[int, int]],
        is_rgb: bool,
        **options: Any,
    ) -> None:
        native_resolution = 1 / np.array(self._mpp) * 10000
        tiff_writer.write(
            tile_iterator,  # noqa
            shape=(*shapes[level], self._size[-1]) if is_rgb else (*shapes[level], 1),
            dtype="uint8",
            resolution=(*native_resolution / 2**level, "CENTIMETER"),  # type: ignore
            photometric="rgb" if is_rgb else "minisblack",
            compression=compression if not self._quality else (compression, self._quality),  # type: ignore
            tile=self._tile_size,
            **options,
        )


def _tiles_iterator_from_pil_image(
    pil_image: PIL.Image.Image, tile_size: tuple[int, int]
) -> Generator[npt.NDArray[np.int_], None, None]:
    """
    Given a PIL image return a tile-iterator.

    Parameters
    ----------
    pil_image : PIL.Image
    tile_size : tuple

    Yields
    ------
    np.ndarray
        Tile outputted in row-major format
    """
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


def _tile_iterator_from_page(
    page: tifffile.TiffPage,
    tile_size: tuple[int, int],
    region_size: tuple[int, int],
    scale: int,
    is_rgb: bool = True,
    interpolator: Resampling = Resampling.NEAREST,
) -> Generator[npt.NDArray[np.int_], None, None]:
    """
    Create an iterator from a tiff page. Useful when writing a pyramidal tiff where the previous page is read to write
    the new page. Each tile will be the downsampled version from the previous version.

    Parameters
    ----------
    page : tifffile.TiffPage
    tile_size : tuple
    region_size : tuple
    scale : int
        Scale between the two pages
    is_rgb : bool
        Whether color image or mask
    interpolator : Resampling
        Interpolation method, see `TiffImageWriter` for more details.

    Yields
    ------
    np.ndarray
        Tile outputted in row-major format
    """
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

        # For mypy
        _coordinates = coordinates[::-1]
        _size = size[::-1]

        tile = get_tile(page, (_coordinates[0], _coordinates[1]), (_size[0], _size[1]))[0]
        vips_tile = numpy_to_vips(tile).resize(1 / scale, kernel=INTERPOLATOR_TO_VIPS[interpolator.value])
        output = vips_to_numpy(vips_tile)
        if not is_rgb:
            output = output[..., 0]
        yield output

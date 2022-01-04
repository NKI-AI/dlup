# coding=utf-8
# Copyright (c) dlup contributors
import os
from enum import Enum
from typing import Iterable, Optional, Union

import numpy as np
import pyvips
from tqdm import tqdm

from dlup import SlideImage
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
#
# map np dtypes to vips
DTYPE_TO_FORMAT = {
    "uint8": "uchar",
    "int8": "char",
    "uint16": "ushort",
    "int16": "short",
    "uint32": "uint",
    "int32": "int",
    "float32": "float",
    "float64": "double",
    "complex64": "complex",
    "complex128": "dpcomplex",
}

# Will be useful when using vips to read
FORMAT_TO_DTYPE = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}


def numpy_to_vips(data):
    height, width, bands = data.shape
    vips_image = pyvips.Image.new_from_memory(data.data, width, height, bands, DTYPE_TO_FORMAT[str(data.dtype)])
    return vips_image


def vips_to_numpy(vips_image):
    return np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=FORMAT_TO_DTYPE[vips_image.format],
        shape=[vips_image.height, vips_image.width, vips_image.bands],
    )


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
        for tile_index, (tile_coordinates, _tile) in tqdm(enumerate(iterator)):
            _tile = np.asarray(_tile)
            if vips_image is None:
                vips_image = pyvips.Image.black(*self._size, bands=_tile.shape[-1])

            vips_tile = numpy_to_vips(_tile)
            vips_image = vips_image.insert(vips_tile, *tile_coordinates)

        # TODO: How to figure out the bit depth?
        # TODO: This will take significant memory resources. Can they be written tile by tile?
        self._save_tiff(vips_image, save_path, bitdepth=8)

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


def testing():
    # TODO this needs to be tested as well with PNG mask!
    #
    # mpp = slide_image.mpp * np.asarray(slide_image.size) / mask.size
    #
    # writer = TiffImageWriter("/home/a.karkala/test.tiff", mpp=mpp, size=mask.size)
    # writer.from_iterator(iterator)

    # TILE_SIZE = (1024, 1024)
    # grid = Grid.from_tiling(
    #     (0, 0),
    #     size=mask.size[::-1],
    #     tile_size=TILE_SIZE,
    #     tile_overlap=(0, 0),
    #     mode=TilingMode.overflow,
    # )

    #
    # def grid_iterator(mask):
    #     for tile_coords in grid:
    #         arr = np.asarray(mask)
    #         cropped_mask = arr[tile_coords[0]:tile_coords[0] + TILE_SIZE[0],
    #                        tile_coords[1]:tile_coords[1] + TILE_SIZE[1]]
    #         yield tile_coords, (cropped_mask*255).astype('uint8')
    #
    #
    #
    # iterator = grid_iterator(mask)

    # Level 0 ROI size
    TILE_SIZE = (512, 512)
    TARGET_MPP = 1.14
    INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"
    OUTPUT_FILE_PATH = "/processing/j.teuwen/test.tiff"
    # Generate the mask
    # from dlup.background import get_mask
    #
    # slide_image = SlideImage.from_file_path(INPUT_FILE_PATH)
    # # mask = get_mask(slide_image)
    # mask = None
    dataset = TiledROIsSlideImageDataset.from_standard_tiling(
        INPUT_FILE_PATH, TARGET_MPP, TILE_SIZE, (0, 0), mask=None, tile_mode=TilingMode.overflow
    )

    dataset2 = TiledROIsSlideImageDataset.from_standard_tiling(
        OUTPUT_FILE_PATH, TARGET_MPP, TILE_SIZE, (0, 0), mask=None, tile_mode=TilingMode.overflow
    )

    idx = 800

    for data0, data1 in zip(dataset, dataset2):
        x = np.asarray(data0["image"])
        y = np.asarray(data1["image"])
        if not x.shape == y.shape:
            pass
        # assert np.allclose(x, y)
        # z = np.abs(x - y)
    print()
    #
    # image_size = (np.asarray(dataset.grids[0][0].size) * dataset.grids[0][1]).tolist()
    #
    # def dataset_iterator(dataset):
    #     for d in dataset:
    #         yield np.array(d["coordinates"]), d["image"]
    #
    # writer = TiffImageWriter(
    #     mpp=(TARGET_MPP, TARGET_MPP),
    #     size=image_size,
    #     tile_width=TILE_SIZE[1],
    #     tile_height=TILE_SIZE[0],
    #     pyramid=False,
    #     compression=TiffCompression.JP2K,
    #     quality=90,
    # )
    #
    # writer.from_iterator(dataset_iterator(dataset), OUTPUT_FILE_PATH)

    import tifffile

    # f = tifffile("/processing/j.teuwen/test_compression_pyramid.tiff")

    # z = tifftools.read_tiff("/processing/j.teuwen/test_compression_pyramid.tiff")
    image = SlideImage.from_file_path("/processing/j.teuwen/test_compression_pyramid.tiff")
    z = pyvips.Image.new_from_file("/processing/j.teuwen/test_compression_pyramid.tiff")
    mpp = [z.get("xres"), z.get("yres")]
    h = z.get_fields()
    print()


if __name__ == "__main__":
    testing()

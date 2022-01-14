# coding=utf-8
# Copyright (c) dlup contributors
import abc
from typing import Iterable, Tuple, Union

import numpy as np

_GenericFloatArray = Union[np.ndarray, Iterable[float]]
_GenericIntArray = Union[np.ndarray, Iterable[int]]
from collections import defaultdict

from dlup import BoundaryMode, SlideImage
from dlup.tiling import Grid, TilingMode
from dlup.writers import TiffCompression, TiffImageWriter


class ImageCache(abc.ABC):
    def __init__(self, identifier, slide_image, regions):
        self._identifier = identifier
        self._slide_image = slide_image
        self._regions = regions



def test_image_cache():
    INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"
    OUTPUT_FILE_PATH = "/processing/j.teuwen/Cache.svs"
    mpp = 11.4
    tile_size = (1024, 1024)
    slide_image = SlideImage.from_file_path(INPUT_FILE_PATH)
    slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
    grid = Grid.from_tiling(
        (0, 0),
        size=slide_level_size,
        tile_size=tile_size,
        tile_overlap=(0, 0),
        mode=TilingMode.overflow,
    )
    create_tiff_cache(slide_image, grid, mpp, tile_size, output_size=slide_level_size, filename=OUTPUT_FILE_PATH)


def create_tiff_cache(slide_image, grid, mpp, tile_size, output_size, filename, tiff_tile_size):
    scaling: float = slide_image.mpp / mpp
    region_view = slide_image.get_scaled_view(scaling)
    region_view.boundary_mode = BoundaryMode.crop

    grid_offset = np.asarray([_.min() for _ in grid.coordinates])

    def _local_iterator():
        region_size: Tuple[int, int] = tile_size
        for coordinates in grid:
            yield coordinates - grid_offset, region_view.read_region(coordinates, region_size)

    # TODO: Think about if we need to pass the output_size or not?
    # Dat het grid dit ook nodig heeft zou geen punt moeten zijn. Is een ander ding, hier wil je enkel
    # over een hoop coordinaten scrollen.
    # grid_end = [_.max() + curr_tile_size for _, curr_tile_size in zip(grid.coordinates, tile_size)]
    # slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))  # Use this to compute the proper size together with the grid

    writer = TiffImageWriter(
        mpp=(mpp, mpp),
        size=output_size - grid_offset,
        tile_width=tiff_tile_size[0],
        tile_height=tiff_tile_size[1],
        pyramid=False,
        compression=TiffCompression.JPEG,
        quality=100,
        bit_depth=8,
        silent=False,
    )

    writer.from_iterator(_local_iterator(), filename, total=len(grid))


if __name__ == "__main__":
    test_image_cache()

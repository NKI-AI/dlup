# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from dlup.writers import TiffCompression, TiffImageWriter


def _dataset_iterator(dataset):
    for d in dataset:
        yield np.array(d["coordinates"]), d["image"]


class TestTiffWriter:
    def test_tiff_writer(self):
        TILE_SIZE = (512, 512)
        TARGET_MPP = 11.4
        INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"

        with tempfile.NamedTemporaryFile(suffix=".tif") as temp_file:
            dataset = TiledROIsSlideImageDataset.from_standard_tiling(
                INPUT_FILE_PATH, TARGET_MPP, TILE_SIZE, (0, 0), mask=None, tile_mode=TilingMode.skip
            )

            image_size = dataset.slide_image.get_scaled_size(dataset.slide_image.get_scaling(TARGET_MPP))

        writer = TiffImageWriter(
            mpp=(TARGET_MPP, TARGET_MPP),
            size=image_size,
            tile_width=TILE_SIZE[1],
            tile_height=TILE_SIZE[0],
            pyramid=False,
            compression=TiffCompression.NONE,
            quality=100,
        )

        writer.from_iterator(_dataset_iterator(dataset), temp_file.name)

        dataset_temp = TiledROIsSlideImageDataset.from_standard_tiling(
            temp_file.name, TARGET_MPP, TILE_SIZE, (0, 0), mask=None, tile_mode=TilingMode.skip
        )

        for data0, data1 in zip(dataset, dataset_temp):
            x = np.asarray(data0["image"])
            y = np.asarray(data1["image"])
            assert np.allclose(x, y)

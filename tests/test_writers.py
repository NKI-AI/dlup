# coding=utf-8
# Copyright (c) dlup contributors
import tempfile
import pytest
import numpy as np

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from dlup.writers import TiffCompression, TiffImageWriter


class TestTiffWriter:
    @pytest.mark.parametrize("tile_size", [[512, 512], [1024, 1024]])
    @pytest.mark.parametrize("target_mpp", [11.4])
    @pytest.mark.parametrize("tile_mode", [TilingMode.skip])
    def test_tiff_writer(self, tile_size, target_mpp, tile_mode):
        INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"

        dataset = TiledROIsSlideImageDataset.from_standard_tiling(
            INPUT_FILE_PATH, target_mpp, tile_size, (0, 0), mask=None, tile_mode=tile_mode
        )

        if tile_mode == tile_mode.skip:
            image_size = dataset.slide_image.get_scaled_size(dataset.slide_image.get_scaling(target_mpp))
        else:
            raise RuntimeError

        writer = TiffImageWriter(
            mpp=(target_mpp, target_mpp),
            size=image_size,
            tile_width=tile_size[1],
            tile_height=tile_size[0],
            pyramid=False,
            compression=TiffCompression.NONE,
            quality=100,
        )

        with tempfile.NamedTemporaryFile(suffix=".tif") as temp_file:
            writer.from_iterator(self._dataset_iterator(dataset), temp_file.name)

            dataset_temp = TiledROIsSlideImageDataset.from_standard_tiling(
                temp_file.name, target_mpp, tile_size, (0, 0), mask=None, tile_mode=TilingMode.skip
            )

            for data0, data1 in zip(dataset, dataset_temp):
                x = np.asarray(data0["image"])
                y = np.asarray(data1["image"])

                del data0["image"]
                del data1["image"]

                assert data0["path"] == INPUT_FILE_PATH
                del data0["path"]
                assert data1["path"] == temp_file.name
                del data1["path"]

                assert np.allclose(x, y)
                assert data0 == data1

    @staticmethod
    def _dataset_iterator(dataset):
        for d in dataset:
            yield np.array(d["coordinates"]), d["image"]

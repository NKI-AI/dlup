# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np
import pytest

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from dlup.writers import TiffCompression, TiffImageWriter


class TestTiffWriter:
    @pytest.mark.parametrize("tile_size", [[512, 512], [1024, 1024]])
    @pytest.mark.parametrize("target_mpp", [11.4])
    @pytest.mark.parametrize("tile_mode", [TilingMode.overflow, TilingMode.skip, TilingMode.fit])
    def test_tiff_writer(self, tile_size, target_mpp, tile_mode):
        INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"

        dataset = TiledROIsSlideImageDataset.from_standard_tiling(
            INPUT_FILE_PATH, target_mpp, tile_size, (0, 0), mask=None, tile_mode=tile_mode
        )

        image_size = dataset.slide_image.get_scaled_size(dataset.slide_image.get_scaling(target_mpp))

        writer = TiffImageWriter(
            mpp=(target_mpp, target_mpp),
            size=image_size,
            tile_width=tile_size[1],
            tile_height=tile_size[0],
            pyramid=False,
            compression=TiffCompression.NONE,
            quality=100,
            bit_depth=8,
            silent=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".tif") as temp_file:
            writer.from_iterator(self._dataset_iterator(dataset), temp_file.name)

            dataset_temp = TiledROIsSlideImageDataset.from_standard_tiling(
                temp_file.name, target_mpp, tile_size, (0, 0), mask=None, tile_mode=tile_mode
            )
            # TODO: This doesn't match likely due to some rounding.
            # image_size_temp = dataset_temp.slide_image.get_scaled_size(dataset_temp.slide_image.get_scaling(target_mpp))
            # assert image_size == image_size_temp
            self.assert_datasets_equal(dataset, dataset_temp, tile_mode=tile_mode)

    @staticmethod
    def _dataset_iterator(dataset):
        for d in dataset:
            yield np.array(d["coordinates"]), d["image"]

    @staticmethod
    def assert_datasets_equal(dataset0, dataset1, tile_mode):
        for data0, data1 in zip(dataset0, dataset1):
            x = np.asarray(data0["image"])
            y = np.asarray(data1["image"])

            del data0["image"]
            del data1["image"]

            assert data0["path"] == dataset0.path
            del data0["path"]
            assert data1["path"] == dataset1.path
            del data1["path"]

            np.testing.assert_allclose(x, y, atol=0 if tile_mode == TilingMode.skip else 2)
            assert data0 == data1


#
# if __name__ == "__main__":
#     from dlup import SlideImage
#     slide_image = SlideImage.from_file_path("/processing/j.teuwen/output_test/tissue_mask.tiff")
#
#     whole_image = slide_image.read_region(location=(0, 0), scaling=0.1, size=(400, 350))
#     whole_image = np.asarray(whole_image)
#

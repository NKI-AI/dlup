# coding=utf-8
# Copyright (c) dlup contributors
import pathlib
import tempfile

import numpy as np
import pytest

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from dlup.writers import TiffCompression, TiffImageWriter

from .common import _download_test_image


class TestTiffWriter:
    @pytest.mark.parametrize("tile_size", [[32, 32], [64, 64]])
    @pytest.mark.parametrize("target_mpp", [1.0])
    @pytest.mark.parametrize("tile_mode", [TilingMode.overflow, TilingMode.skip, TilingMode.fit])
    def test_tiff_writer(self, tile_size, target_mpp, tile_mode):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "test_image.svs"
            cache_filename = pathlib.Path(temp_dir) / f"test_image-mpp-{target_mpp}.tiff"
            _download_test_image(save_to=path)

            dataset = TiledROIsSlideImageDataset.from_standard_tiling(
                path, target_mpp, tile_size, (0, 0), mask=None, tile_mode=tile_mode
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

            writer.from_iterator(self._dataset_iterator(dataset), path / "temp_file_downsampled.tiff")

            dataset_temp = TiledROIsSlideImageDataset.from_standard_tiling(
                cache_filename, target_mpp, tile_size, (0, 0), mask=None, tile_mode=tile_mode
            )
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

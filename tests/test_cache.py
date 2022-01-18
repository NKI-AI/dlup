import numpy as np
import pytest

import dlup
dlup.IMAGE_CACHE = "TIFF"

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import Grid, TilingMode
from dlup._image import CachedSlideImage, SlideImage


# def test_dataset_equality():
#     INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"
#     DOWNSAMPLED_IMAGE_FILE_PATH = (
#         "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.mpp-11.4.tiff"
#     )
#     target_mpp = 11.4
#     tile_size = (1024, 1024)
#     tile_mode = TilingMode.skip
#     dataset = TiledROIsSlideImageDataset.from_standard_tiling(
#         INPUT_FILE_PATH, target_mpp, tile_size, (0, 0), mask=None, tile_mode=tile_mode
#     )
#
#     dataset_temp = TiledROIsSlideImageDataset.from_standard_tiling(
#         DOWNSAMPLED_IMAGE_FILE_PATH, target_mpp, tile_size, (0, 0), mask=None, tile_mode=tile_mode
#     )
#     # TODO: This doesn't match likely due to some rounding.
#     # image_size_temp = dataset_temp.slide_image.get_scaled_size(dataset_temp.slide_image.get_scaling(target_mpp))
#     # assert image_size == image_size_temp
#     assert_datasets_equal(dataset, dataset_temp, tile_mode=tile_mode)


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


class TestCache:
    @pytest.mark.parametrize("regions", [(0, 0), (512, 512)])
    def test_cache_correctness(self, regions):

        INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"
        slide_image = SlideImage.from_file_path(INPUT_FILE_PATH)
        mpp = 11.4
        tile_size = (512, 512)

        # TODO: Set hte propery on the class itself of cacher
        cached_slide_image = CachedSlideImage.from_file_path(INPUT_FILE_PATH)
        cached_slide_image.cache_directory = "/processing/j.teuwen/"

        scaling_original = slide_image.get_scaling(mpp)

        for region in regions:
            region_0 = slide_image.read_region(region, scaling_original, tile_size)
            region_1 = cached_slide_image.read_region(region, scaling_original, tile_size)

            x = np.asarray(region_0)
            y = np.asarray(region_1)

            np.testing.assert_allclose(x, y)


# def test_image_cache():
#     INPUT_FILE_PATH = "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"
#     OUTPUT_FILE_PATH = "/processing/j.teuwen/Cache.svs"
#     mpp = 512
#     tile_size = (512, 512)
#     from dlup._cache import create_tiff_cache
#     from dlup._image import CachedSlideImage, SlideImage
#
#     slide_image = SlideImage.from_file_path(INPUT_FILE_PATH)
#     slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
#     grid = Grid.from_tiling(
#         (0, 0),
#         size=slide_level_size,
#         tile_size=tile_size,
#         tile_overlap=(0, 0),
#         mode=TilingMode.overflow,
#     )
#     create_tiff_cache(
#         slide_image,
#         grid,
#         mpp,
#         tile_size,
#         output_size=slide_level_size,
#         filename=OUTPUT_FILE_PATH,
#         pyramid=False,
#         tiff_tile_size=(256, 256),
#     )


if __name__ == "__main__":
    # test_dataset_equality()
    TestCache().test_cache_correctness()

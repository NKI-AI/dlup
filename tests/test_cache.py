import numpy as np
import pytest

from dlup._cache import TiffScaleLevelCache
from dlup._image import SlideImage
from dlup.tiling import Grid, TilingMode


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
        cached_slide_image = SlideImage.from_file_path(INPUT_FILE_PATH)
        cached_slide_image.cacher = TiffScaleLevelCache(
            original_filename=INPUT_FILE_PATH,
            mpp_to_cache_map={
                11.4: "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.mpp-11.4.tiff"
            },
        )

        scaling_original = slide_image.get_scaling(mpp)

        for region in regions:
            region_0 = slide_image.read_region(region, scaling_original, tile_size)
            region_1 = cached_slide_image.read_region(region, scaling_original, tile_size)

            x = np.asarray(region_0)
            y = np.asarray(region_1)

            np.testing.assert_allclose(x, y)

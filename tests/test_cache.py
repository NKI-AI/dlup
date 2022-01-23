import numpy as np
import pytest
from .common import _download_test_image
from dlup._cache import TiffScaleLevelCache
from dlup._image import SlideImage
from dlup.tiling import Grid, TilingMode
import tempfile
from dlup._cache import create_tiff_cache
import pathlib


def _create_cache(input_file, cache_file, tile_size, mpp):
    slide_image = SlideImage.from_file_path(input_file)
    slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
    grid = Grid.from_tiling(
        (0, 0),
        size=slide_level_size,
        tile_size=tile_size,
        tile_overlap=(0, 0),
        mode=TilingMode.overflow,
    )
    create_tiff_cache(
        cache_file,
        slide_image,
        grid,
        mpp,
        tile_size,
        pyramid=False,
        tiff_tile_size=(256, 256),
        silent=False,
    )


class TestCache:
    @pytest.mark.parametrize("regions", [(0, 0), (64, 64)])
    @pytest.mark.parametrize("tile_size", [(64, 64)])
    @pytest.mark.parametrize("mpp", [1.0])
    def test_cache_correctness(self, regions, tile_size, mpp):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "test_image.svs"
            cache_filename = pathlib.Path(temp_dir) / f"test_image-mpp-{mpp}.tiff"
            _download_test_image(save_to=path)
            _create_cache(path, cache_filename, tile_size, mpp)
            slide_image = SlideImage.from_file_path(path)

            cached_slide_image = SlideImage.from_file_path(path)
            cached_slide_image.cacher = TiffScaleLevelCache(
                original_filename=path,
                mpp_to_cache_map={
                    mpp: cache_filename,
                },
            )

            scaling_original = slide_image.get_scaling(mpp)

            for region in regions:
                region_0 = slide_image.read_region(region, scaling_original, tile_size)
                region_1 = cached_slide_image.read_region(region, scaling_original, tile_size)

                x = np.asarray(region_0)
                y = np.asarray(region_1)

                np.testing.assert_allclose(x, y)

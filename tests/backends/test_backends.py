# Copyright (c) dlup contributors
"""Test for the TiffFile backend. Works by creating a tiff file and then reading it with the TiffFile backend.
The results are also compared against the openslide backend.
"""

import numpy as np
import PIL.Image
import pytest
import pyvips

from dlup.backends.openslide_backend import OpenSlideSlide
from dlup.backends.openslide_backend import open_slide as open_slide_openslide
from dlup.backends.tifffile_backend import TifffileSlide
from dlup.backends.tifffile_backend import open_slide as open_slide_tifffile
from dlup.writers import TiffCompression, TifffileImageWriter


@pytest.fixture
def file_path(tmp_path):
    # tmp_path is a pytest fixture that provides a Path object to a temporary directory.
    # Here we are appending a file name to that path.
    return tmp_path / "test_image.tif"


def write_image_to_tiff(file_path, image, mpp, size, pyramid):
    array = image.numpy()
    channels = array.shape[2] if array.ndim == 3 else 1
    writer = TifffileImageWriter(
        file_path,
        size=(*size[::-1], channels),
        mpp=mpp,
        compression=TiffCompression.NONE,
        is_mask=False,
        tile_size=(128, 128),
        pyramid=pyramid,
    )
    writer.from_pil(PIL.Image.fromarray(array))


def create_test_image(size, channels, color1, color2):
    half_size = (size[0] // 2, size[1] // 2)
    if channels == 3:
        array = np.zeros((*size, 3), dtype=np.uint8)
        array[: half_size[0], : half_size[1], :] = color1
        array[half_size[0] :, half_size[1] :, :] = color2
    else:
        array = np.zeros(size, dtype=np.uint8)
        array[: half_size[0], : half_size[1]] = color1
        array[half_size[0] :, half_size[1] :] = color2
    return pyvips.Image.new_from_array(array)


@pytest.fixture
def slides(file_path, test_image, mpp, size, pyramid):
    # Write the test image to a TIFF file
    write_image_to_tiff(str(file_path), test_image, mpp, size, pyramid)
    # Open the written TIFF file with both backends
    tiff_slide = open_slide_tifffile(str(file_path))
    openslide_slide = open_slide_openslide(str(file_path))
    yield tiff_slide, openslide_slide
    # After the test function that uses this fixture finishes, close both slides
    tiff_slide.close()
    openslide_slide.close()


@pytest.fixture
def test_image(size, channels):
    color1 = 255
    color2 = 127
    # Assuming 'create_test_image' is a static method or standalone function
    return create_test_image(size, channels, color1, color2)


@pytest.fixture
def mpp():
    return (1.0, 0.999)


class TestBackends:
    def read_region_and_properties_asserts(self, tiff_slide, openslide_slide, mode, size, mpp, pyramid, test_image):
        original_array = np.asarray(test_image)
        properties = tiff_slide.properties
        tile_size = (properties["tifffile.level[0].TileWidth"], properties["tifffile.level[0].TileLength"])
        num_levels = int(np.ceil(np.log2(np.asarray(size[::-1]) / np.asarray(tile_size))).min()) + 1

        # Let's try to read outside the slide levels. This should give a RuntimeError
        with pytest.raises(RuntimeError):
            tiff_slide.read_region((0, 0), num_levels + 1, (1, 1))

        # We need to check a few regions to make sure the backend is working correctly
        # 1. Check the whole image
        # 2. A part in the upper left
        # 3. Check a part more towards the bottom.

        # This is needed because the array swaps it with respect to the image (x,y) versus (rows, cols)
        _size = size[::-1]
        regions = [
            ((0, 0), _size),
            ((0, 0), (_size[0] // 3, _size[1] // 3)),
            ((_size[0] // 7, _size[1] // 7), (_size[0] - _size[0] // 7, _size[1] - _size[1] // 7)),
        ]
        for location, region_size in regions:
            tiff_region = PIL.Image.fromarray(np.asarray(tiff_slide.read_region(location, 0, region_size)))
            assert tiff_region.mode == mode
            openslide_region = PIL.Image.fromarray(
                np.asarray(openslide_slide.read_region(location, 0, region_size))
            ).convert(mode)

            cropped_array = original_array[
                location[1] : location[1] + region_size[1], location[0] : location[0] + region_size[0]
            ]
            tiff_array = np.asarray(tiff_region)
            openslide_array = np.asarray(openslide_region)

            # Compare the tiff_region with the array itself.
            assert (tiff_array == cropped_array).all()
            # Compare the regions read by OpenSlide and Tifffile backends
            assert (tiff_array == openslide_array).all()

    def property_asserts(self, tiff_slide, openslide_slide, size, mpp, pyramid):
        assert isinstance(tiff_slide, TifffileSlide)
        assert isinstance(openslide_slide, OpenSlideSlide)

        assert tiff_slide.vendor is None
        assert tiff_slide.magnification is None

        assert tiff_slide.dimensions == openslide_slide.dimensions
        assert np.allclose(mpp, tiff_slide.spacing)
        assert tiff_slide.spacing == openslide_slide.spacing

        assert tiff_slide.slide_bounds == openslide_slide.slide_bounds
        # Tiff does not have slide bounds defined (or at least not in our implementation)
        assert tiff_slide.slide_bounds == ((0, 0), tiff_slide.dimensions)

        assert tiff_slide.level_count == openslide_slide.level_count
        if not pyramid:
            assert tiff_slide.level_count == 1
            num_levels = 1
        else:
            # Otherwise we need to compute the number of levels required.
            # This depends on the tile size as well
            properties = tiff_slide.properties
            tile_size = (properties["tifffile.level[0].TileWidth"], properties["tifffile.level[0].TileLength"])
            num_levels = int(np.ceil(np.log2(np.asarray(size[::-1]) / np.asarray(tile_size))).min()) + 1
            assert tiff_slide.level_count == num_levels
            assert openslide_slide.level_count == num_levels

        with pytest.raises(NotImplementedError):
            tiff_slide.set_cache(None)

    @pytest.mark.parametrize("size", [(768, 512), (512, 768), (1024, 256)])
    @pytest.mark.parametrize("channels", [3, 1])
    @pytest.mark.parametrize("pyramid", [False, True])
    def test_tiff_backend(self, size, channels, pyramid, file_path, slides):
        mpp = (1.0, 0.999)
        color1 = 255
        color2 = 127
        mode = "RGB" if channels == 3 else "L"
        test_image = create_test_image(size, channels, color1, color2)
        assert isinstance(test_image, pyvips.Image)

        tiff_slide, openslide_slide = slides  # Unpack the slides from the fixture
        self.property_asserts(tiff_slide, openslide_slide, size, mpp, pyramid)
        self.read_region_and_properties_asserts(tiff_slide, openslide_slide, mode, size, mpp, pyramid, test_image)

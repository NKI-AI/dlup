import tempfile

import numpy as np
import PIL.Image
import pytest

from dlup._image import Resampling
from dlup.experimental_backends.openslide_backend import OpenSlideSlide
from dlup.experimental_backends.tifffile_backend import TifffileSlide
from dlup.writers import TiffCompression, TifffileImageWriter


class TestTiffBackend:
    def create_test_image(self, size, channels, color1, color2):
        half_size = (size[0] // 2, size[1] // 2)
        if channels == 3:
            array = np.zeros((*size, 3), dtype=np.uint8)
            array[: half_size[0], : half_size[1], :] = color1
            array[half_size[0] :, half_size[1] :, :] = color2
        else:
            array = np.zeros(size, dtype=np.uint8)
            array[: half_size[0], : half_size[1]] = color1
            array[half_size[0] :, half_size[1] :] = color2
        return PIL.Image.fromarray(array, mode="RGB" if channels == 3 else "L")

    def write_image_to_tiff(self, file_path, image, mpp, size):
        array = np.asarray(image)
        channels = array.shape[2] if array.ndim == 3 else 1
        writer = TifffileImageWriter(
            file_path,
            size=(*size[::-1], channels),
            mpp=mpp,
            compression=TiffCompression.NONE,
            interpolator=Resampling.NEAREST,
            tile_size=(128, 128),
            pyramid=False,
        )
        writer.from_pil(image)

    def read_and_assert(self, file_path, mode, size, mpp, test_image):
        # original_array = np.asarray(test_image)
        tiff_slide = TifffileSlide(file_path)
        openslide_slide = OpenSlideSlide(file_path)

        assert tiff_slide.dimensions == openslide_slide.dimensions
        assert np.allclose(mpp, tiff_slide.spacing)
        assert tiff_slide.spacing == openslide_slide.spacing

        # We need to check a few regions to make sure the backend is working correctly
        # 1. Check the whole image
        # 2. Check the top right corner
        # 3. Check the bottom left corner

        # Let's create a list of origins and sizes.
        _size = size[::-1]
        regions = [
            ((0, 0), _size),
            ((0, 0), (_size[1] // 3, _size[1] // 3)),
            ((_size[0] // 7, _size[1] // 7), (_size[0] - _size[0] // 7, _size[1] - _size[1] // 7)),
        ]
        for location, region_size in regions:
            tiff_region = tiff_slide.read_region(location, 0, region_size).convert(mode)
            openslide_region = openslide_slide.read_region(location, 0, region_size).convert(mode)

            # cropped_array = original_array[
            #     location[0] : location[0] + region_size[0], location[1] : location[1] + region_size[1]
            # ]
            tiff_array = np.asarray(tiff_region)
            openslide_array = np.asarray(openslide_region)

            # Compare the tiff_region with the array itself.
            # assert (tiff_array == cropped_array).all()
            # Compare the regions read by OpenSlide and Tifffile backends
            assert (tiff_array == openslide_array).all()

        tiff_slide.close()
        openslide_slide.close()

    @pytest.mark.parametrize("size", [(768, 512)])
    @pytest.mark.parametrize("channels", [3])
    def test_tiff_backend(self, size, channels):
        mpp = (1.0, 0.999)
        color1 = 255
        color2 = 127
        mode = "RGB" if channels == 3 else "L"
        test_image = self.create_test_image(size, channels, color1, color2)

        with tempfile.NamedTemporaryFile(suffix=".tif") as temp_tiff:
            self.write_image_to_tiff(temp_tiff.name, test_image, mpp, size)
            self.read_and_assert(temp_tiff.name, mode, size, mpp, test_image)

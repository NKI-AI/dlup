# Copyright (c) dlup contributors

import numpy as np
import pytest
import pyvips
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from dlup.backends.common import AbstractSlideBackend, numpy_to_pil
from dlup.utils.pyvips_utils import pil_to_vips


def get_sample_nonuniform_image(size: tuple[int, int] = (256, 256)) -> pyvips.Image:
    """Generate a non-uniform sample image."""
    if not (np.array(size) % 2 == 0).all():
        raise ValueError("Size should be a tuple of values divisible by two.")

    # Define data for interpolation
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])

    # Define the values at each grid point
    X, Y = np.meshgrid(x, y)
    values = X + Y

    # Use RegularGridInterpolator
    f = RegularGridInterpolator((x, y), values)

    # Sample it
    Z = np.stack([X, Y], axis=-1)
    z = f(Z).reshape(size)

    # Interpret it as HSV, so we get funny colors
    im = np.zeros((*size, 3))
    im[:, :, 0] = z
    im[:, :, 1] = 1

    # Set the value to a pixel-level checkerboard.
    im[size[1] // 2, size[0] // 2, 2] = 1
    im[:, :, 2] = np.sign(np.fft.ifft2(im[:, :, 2]).real)
    im = im * 255
    im = im.astype("uint8")
    im = Image.fromarray(im, mode="HSV")
    return pil_to_vips(im.convert(mode="RGBA"))


def test_numpy_to_pil_single_channel():
    arr = np.arange(100, dtype=np.uint8).reshape((10, 10, 1))
    pil_img = numpy_to_pil(arr)
    assert pil_img.mode == "L"


def test_numpy_to_pil_rgb():
    arr = (np.arange(300) / 300 * 255).astype(np.uint8).reshape((10, 10, 3))
    pil_img = numpy_to_pil(arr)
    assert pil_img.mode == "RGB"


def test_numpy_to_pil_rgba():
    arr = (np.arange(400) / 400 * 255).astype(np.uint8).reshape((10, 10, 4))
    pil_img = numpy_to_pil(arr)
    assert pil_img.mode == "RGBA"


def test_numpy_to_pil_invalid_channels():
    arr = (np.arange(500) / 500 * 255).astype(np.uint8).reshape((10, 10, 5))
    with pytest.raises(RuntimeError):
        numpy_to_pil(arr)


class TestAbstractBackend:
    class DummySlideBackend(AbstractSlideBackend):
        # Minimal implementation to avoid the ABC restriction.
        def __init__(self, filename: str):
            super().__init__(filename)
            # Dummy data for testing
            self._level_count = 3
            self._downsamples = [1.0, 2.0, 4.0]
            self._spacings = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
            self._shapes = ((1000, 1000), (500, 500), (250, 250))

        def read_region(self, coordinates, level, size) -> Image.Image:
            return Image.new("RGB", size, color="white")

        @property
        def properties(self):
            return {}

        @property
        def magnification(self):
            return 10.0

        @property
        def vendor(self):
            return "TestVendor"

        def close(self):
            pass

    def test_dummy_slide_backend_properties(self):
        slide = self.DummySlideBackend("test_filename.tiff")

        # Testing the level_count
        assert slide.level_count == 3

        # Testing the dimensions
        assert slide.dimensions == (1000, 1000)

        # Testing the spacing
        assert slide.spacing == (0.5, 0.5)

        # Testing the level_dimensions
        assert slide.level_dimensions == ((1000, 1000), (500, 500), (250, 250))

        # Testing the level_spacings
        assert slide.level_spacings == ((0.5, 0.5), (1.0, 1.0), (2.0, 2.0))

        # Testing the level_downsamples
        assert slide.level_downsamples == (1.0, 2.0, 4.0)

        # Testing slide bounds
        assert slide.slide_bounds == ((0, 0), (1000, 1000))

        # Testing get_best_level_for_downsample
        assert slide.get_best_level_for_downsample(0.5) == 0
        assert slide.get_best_level_for_downsample(1.0) == 0
        assert slide.get_best_level_for_downsample(2.0) == 1
        assert slide.get_best_level_for_downsample(3.0) == 1
        assert slide.get_best_level_for_downsample(4.5) == 2

    def test_repr(self):
        slide = self.DummySlideBackend("test_filename.tiff")
        assert repr(slide) == "<DummySlideBackend(test_filename.tiff)>"

    def test_spacing_without_set(self):
        slide = self.DummySlideBackend("test_filename.tiff")
        slide._spacings = None
        assert slide.spacing is None

    def test_get_thumbnail(self):
        slide = self.DummySlideBackend("test_filename.tiff")

        # Getting a 200x200 thumbnail
        thumbnail = slide.get_thumbnail(200)
        assert isinstance(thumbnail, pyvips.Image)
        assert (thumbnail.width, thumbnail.height) == (200, 200)

        # Getting a 300x150 thumbnail
        thumbnail = slide.get_thumbnail((300, 150))
        assert isinstance(thumbnail, pyvips.Image)
        # The aspect ratio should be preserved, so width might be less than 300
        assert thumbnail.height == 150

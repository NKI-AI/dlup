# coding=utf-8
# Copyright (c) DLUP Contributors
import functools
from typing import Dict, Optional, Tuple, TypeVar

import numpy as np
import openslide
import PIL

import pytest
from PIL.Image import Image
from pydantic import BaseModel, Field
from scipy import interpolate

from dlup.slide import WholeSlidePyramidalImage

_TImage = TypeVar("TImage", bound="Image")

_SAMPLE_IMAGE_SIZE = np.array((256, 256))


@functools.cache
def get_sample_nonuniform_image():
    """Generate a non-uniform sample image."""
    # Interpolate some simple function
    interp_args = ((0, 1), (0, 1), ((0, 1), (1, 0)))  # x  # y  # z
    f = interpolate.interp2d(*interp_args, kind="linear")

    # Sample it
    width, height = _SAMPLE_IMAGE_SIZE
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    z = f(x, y)

    # Interpret it as HSV, so we get funny colors
    im = np.zeros((height, width, 3))
    im[:, :, 0] = z.T
    im[:, :, 1] = 1

    # Set the value to a pixel-level checkerboard.
    # Maybe there's a faster way.
    im[height // 2, width // 2, 2] = 1
    im[:, :, 2] = np.sign(np.fft.ifft2(im[:, :, 2]).real)
    im = im * 255
    im = im.astype("uint8")
    im = PIL.Image.fromarray(im, mode="HSV")
    return im.convert(mode="RGBA")


class SlideProperties(BaseModel):
    mppx: Optional[float] = Field(1.0, alias=openslide.PROPERTY_NAME_MPP_X)
    mppy: Optional[float] = Field(1.0, alias=openslide.PROPERTY_NAME_MPP_Y)
    mag: Optional[int] = Field(40, alias=openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    vendor: str = Field("dummy", alias=openslide.PROPERTY_NAME_VENDOR)

    class Config:
        allow_population_by_field_name = True


class SlideConfig(BaseModel):
    image: _TImage = get_sample_nonuniform_image()
    properties: SlideProperties = SlideProperties()
    level_downsamples: Tuple[float, ...] = (1.0,)


class OpenSlideImage(openslide.ImageSlide):
    properties = {}
    level_downsamples = (1.0,)

    def __init__(self, image: PIL.Image, properties: Dict, level_downsamples: Tuple):
        self.properties = properties
        self.image = image
        self.level_downsamples = level_downsamples
        super().__init__(image)

    @classmethod
    def from_slide_config(cls, slide_config):
        return cls(
            slide_config.image,
            slide_config.properties.dict(by_alias=True, exclude_none=True),
            slide_config.level_downsamples,
        )


@pytest.fixture
def slide_config():
    return SlideConfig()


@pytest.fixture
def openslide_image(slide_config):
    """Create mock image."""
    return OpenSlideImage.from_slide_config(slide_config)


@pytest.fixture
def dlup_wsi(openslide_image):
    return WholeSlidePyramidalImage(openslide_image)


class TestWholeSlidePyramidalImage:
    @pytest.mark.parametrize(
        "slide_config",
        [
            SlideConfig(properties=SlideProperties(mppx=1.0, mppy=2.0)),
            SlideConfig(properties=SlideProperties(mppx=None, mppy=None)),
        ],
    )
    def test_mpp_exceptions(self, openslide_image):
        """Test that we break if the slide has no isotropic resolution."""
        with pytest.raises(RuntimeError):
            wsi = WholeSlidePyramidalImage(openslide_image)

    def test_aspect_ratio(self, dlup_wsi, openslide_image):
        """Test consistent definition of aspect ratio."""
        assert dlup_wsi.aspect_ratio == openslide_image.image.width / openslide_image.image.height

    @pytest.mark.parametrize(
        "slide_config",
        [
            SlideConfig(properties=SlideProperties(level_downsamples=(1.0,))),
        ],
    )
    @pytest.mark.parametrize('x', [0, 0.1, 7.3])
    @pytest.mark.parametrize('y', [0, 0.1, 4.2])
    @pytest.mark.parametrize('size', [(1, 1), (2, 4), (9, 13)])
    @pytest.mark.parametrize('scaling', [4, 2, 1 / 2, 1 / 5, 1 / 7])
    def test_read_region(self, dlup_wsi, openslide_image, x, y, size, scaling):
        """Test exact interpolation.

        We want to be sure that downsampling the whole image and extracting
        a region commutes with extracting a region and downsampling.
        """
        # Downsample the whole image to a certain size
        # and extract the region.
        image = openslide_image.image
        location = np.array((x, y))

        # Notice that PIL box uses upper left and lower right
        # rectangle corners coordinates.
        box = (*location, *(location + size))
        native_box = np.array(box) / scaling
        pil_extracted_region = np.array(image.resize(size, resample=PIL.Image.LANCZOS, box=native_box))

        # Use dlup read_region to extract the same location
        extracted_region = np.array(dlup_wsi.read_region(location, scaling, size))

        assert pil_extracted_region.shape == extracted_region.shape
        assert np.allclose(pil_extracted_region, extracted_region)

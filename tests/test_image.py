# coding=utf-8
# Copyright (c) dlup contributors
from typing import Any, Dict, Optional, Tuple, Sequence, Type, Union

import numpy as np
import openslide  # type: ignore
import PIL
import pytest
from PIL.Image import Image
from pydantic import BaseModel, Field
from scipy import interpolate

from dlup import DlupUnsupportedSlideError, SlideImage


def get_sample_nonuniform_image(size: Tuple[int, int] = (256, 256)):
    """Generate a non-uniform sample image."""
    # Interpolate some simple function
    interp_args = ((0, 1), (0, 1), ((0, 1), (1, 0)))  # x  # y  # z
    f = interpolate.interp2d(*interp_args, kind="linear")
    if not (np.array(size) % 2 == 0).all():
        raise ValueError("Size should be a tuple of values divisible by two.")

    # Sample it
    width, height = size
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
    return im.convert(mode="RGBA")  # type: ignore


class SlideProperties(BaseModel):
    """Mock configuration properties."""

    mpp_x: Optional[float] = Field(1.0, alias=openslide.PROPERTY_NAME_MPP_X)
    mpp_y: Optional[float] = Field(1.0, alias=openslide.PROPERTY_NAME_MPP_Y)
    mag: Optional[int] = Field(40, alias=openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    vendor: str = Field("dummy", alias=openslide.PROPERTY_NAME_VENDOR)

    class Config:
        allow_population_by_field_name = True


class SlideConfig(BaseModel):
    """Mock slide configuration."""

    image: Type[Image] = get_sample_nonuniform_image()
    properties: SlideProperties = SlideProperties()
    level_downsamples: Tuple[float, ...] = (1.0, 2.0)


class OpenSlideImageMock(openslide.ImageSlide):
    """Mock OpenSlide object also with layers.

    NOTE: read_region works a bit differently than the actual openslide.
    Openslide *does* project the float base layer values to the best layer
    and then performs different operations such as adding saturation and translations.
    https://github.com/openslide/openslide/blob/main/src/openslide.c#L488-L493
    """

    properties: Dict[Any, Any] = {}
    level_downsamples: Sequence[Union[float, int]] = (1.0,)

    def __init__(self, image: PIL.Image, properties: Dict, level_downsamples: Tuple):
        self.properties = properties
        self.image = image
        self.level_downsamples = sorted(level_downsamples)
        base_size = np.array((self.image.width, self.image.height))
        self._level_dimensions = tuple([tuple((base_size / d).astype(int)) for d in self.level_downsamples])
        super().__init__(image)

    def get_best_level_for_downsample(self, downsample):
        level_downsamples = np.array(self.level_downsamples)
        level = 0 if downsample < 1 else np.where(level_downsamples <= downsample)[0][-1]
        return level

    @property
    def level_dimensions(self):
        return self._level_dimensions

    def get_level_image(self, level):
        return self.image.resize(self.level_dimensions[level])

    def read_region(self, location, level, size):
        image = np.array(self.get_level_image(level))

        # Add a single pixel padding
        image = np.pad(image, [(0, 1), (0, 1), (0, 0)])
        image = PIL.Image.fromarray(image)
        location = np.asarray(location) / self.level_downsamples[level]
        return image.resize(size, resample=PIL.Image.LANCZOS, box=(*location, *(location + size)))

    @classmethod
    def from_slide_config(cls, slide_config):
        return cls(
            slide_config.image,
            slide_config.properties.dict(by_alias=True, exclude_none=True),
            slide_config.level_downsamples,
        )


@pytest.fixture
def slide_config():
    """Fixture returning a slide."""
    return SlideConfig()


@pytest.fixture
def openslide_image(slide_config):
    """Fixture returning a mock image."""
    return OpenSlideImageMock.from_slide_config(slide_config)


@pytest.fixture
def dlup_wsi(openslide_image):
    """Generate sample SlideImage object to test."""
    return SlideImage(openslide_image, identifier="mock")


class TestSlideImage:
    """Test the dlup.SlideImage functionality."""

    @pytest.mark.parametrize(
        "slide_config",
        [
            SlideConfig(properties=SlideProperties(mpp_x=1.0, mpp_y=2.0)),
            SlideConfig(properties=SlideProperties(mpp_x=None, mpp_y=None)),
        ],
    )
    def test_mpp_exceptions(self, openslide_image):
        """Test that we break if the slide has no isotropic resolution."""
        with pytest.raises(DlupUnsupportedSlideError):
            wsi = SlideImage(openslide_image)

    def test_properties(self, openslide_image):
        """Test properties."""
        dlup_wsi = SlideImage(openslide_image, identifier="mock")
        assert dlup_wsi.aspect_ratio == openslide_image.image.width / openslide_image.image.height
        assert dlup_wsi.mpp == openslide_image.properties[openslide.PROPERTY_NAME_MPP_X]
        assert dlup_wsi.magnification == openslide_image.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        assert isinstance(repr(dlup_wsi), str)
        assert dlup_wsi.identifier == "mock"
        assert isinstance(dlup_wsi.thumbnail, np.ndarray)

    @pytest.mark.parametrize(
        "slide_config",
        [
            SlideConfig(level_downsamples=(1.0, 2.0)),
        ],
    )
    @pytest.mark.parametrize("out_region_x", [0, 4.1, 1.5, 3.9])
    @pytest.mark.parametrize("out_region_y", [0, 0.03, 4.2])
    @pytest.mark.parametrize("out_region_size", [(4, 4), (7, 3), (1, 5)])
    @pytest.mark.parametrize(
        "scaling",
        [
            2,
            1,
            1 / 3,
            1 / 7,
        ],
    )
    def test_read_region(
        self, mocker, dlup_wsi, openslide_image, out_region_x, out_region_y, out_region_size, scaling
    ):
        """Test exact interpolation.

        We want to be sure that reading a region at some scaling level is equivalent to
        downsampling the whole image and extracting that region using PIL.
        """
        base_image = openslide_image.image
        base_image_size = np.array((base_image.width, base_image.height))

        # Compute output image global coordinates.
        out_region_location = np.array((out_region_x, out_region_y))
        out_region_size = np.array(out_region_size)

        # Get the target layer from which we downsample.
        downsampling = 1 / scaling
        expected_level = openslide_image.get_best_level_for_downsample(downsampling)
        expected_level_image = openslide_image.get_level_image(expected_level)
        relative_scaling = scaling * openslide_image.level_downsamples[expected_level]

        # Notice that PIL box uses upper left and lower right rectangle coordinates.
        box = (*out_region_location, *(out_region_location + out_region_size))
        expected_level_box = np.array(box) / relative_scaling
        pil_extracted_region = np.array(
            expected_level_image.resize(out_region_size, resample=PIL.Image.LANCZOS, box=expected_level_box)
        )

        # Spy on how our mock object was called
        mocker.spy(openslide_image, "read_region")

        # Use dlup read_region to extract the same location
        extracted_region = dlup_wsi.read_region(out_region_location, scaling, out_region_size)

        # Is a numpy array
        assert isinstance(extracted_region, np.ndarray)

        # Check that the right layer was indeed requested from our mock function.
        call_args_list = openslide_image.read_region.call_args_list
        assert len(call_args_list) == 1
        (call,) = call_args_list
        _, selected_level, _ = call.args
        assert selected_level == expected_level

        # Check that the output correspondin shape and value.
        assert pil_extracted_region.shape == extracted_region.shape
        assert np.allclose(pil_extracted_region, extracted_region)

    @pytest.mark.parametrize("shift_x", list(np.linspace(0, 2, 10)))
    def test_border_region(self, shift_x):
        """Test border region."""
        image_size = (128, 128)
        scaling = 1 / 5
        out_region_size = (7, 7)
        out_region_size = np.array(out_region_size)

        # Example image
        image = get_sample_nonuniform_image().resize(image_size)

        # Create a slide
        openslide_image = OpenSlideImageMock(
            image, {openslide.PROPERTY_NAME_MPP_X: 1.0, openslide.PROPERTY_NAME_MPP_Y: 1.0}, (1.0, 4.3)
        )

        wsi = SlideImage(openslide_image, identifier="mock")
        ssize = np.array(wsi.get_scaled_size(scaling))

        out_region_location = ssize - out_region_size - 1 + shift_x

        if (out_region_location + out_region_size > ssize).any():
            with pytest.raises(ValueError):
                extracted_region = wsi.read_region(out_region_location, scaling, out_region_size)
            return

        extracted_region = wsi.read_region(out_region_location, scaling, out_region_size)

    def test_scaled_size(self, dlup_wsi):
        """Check the scale is greater than zero."""
        size = dlup_wsi.get_scaled_size(0.5)
        assert (np.array(size) >= 0).all()

    def test_thumbnail(self, dlup_wsi):
        """Check the thumbnail is a numpy array."""
        thumbnail = dlup_wsi.thumbnail
        assert isinstance(thumbnail, np.ndarray)


@pytest.mark.parametrize("scaling", [2, 1, 1 / 3])
def test_scaled_view(dlup_wsi, scaling):
    """Check that a scaled view correctly represents a layer."""
    view = dlup_wsi.get_scaled_view(scaling)
    assert view.mpp == dlup_wsi.mpp / scaling
    location = (3.7, 0)
    size = (10, 15)
    assert (view.read_region(location, size) == dlup_wsi.read_region(location, scaling, size)).all()
    assert (dlup_wsi.get_scaled_size(scaling) == view.size).all()

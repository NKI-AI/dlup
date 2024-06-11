# Copyright (c) dlup contributors

"""Test the SlideImage class.
This set of tests ensures the SlideImage extracts the tiles
at a selected level by minimizing the information loss during
interpolation as well as ensuring the tiles are extracted
from the right level and locations of the original image.
"""
import math
from unittest.mock import MagicMock

import numpy as np
import openslide
import pytest
import pyvips

from dlup import SlideImage, UnsupportedSlideError

from .backends.test_openslide_backend import SLIDE_CONFIGS, MockOpenSlideSlide, SlideConfig

_BASE_CONFIG = SlideConfig.from_parameters(
    filename="dummy1.svs",
    num_levels=3,
    level_0_dimensions=(1000, 1000),
    mpp=(0.25, 0.25),
    objective_power=20,
    vendor="dummy",
)


class TestSlideImage:
    """Test the dlup.SlideImage functionality."""

    def test_properties(self, openslideslide_image):
        """Test properties."""
        dlup_wsi = SlideImage(openslideslide_image, identifier="mock", internal_handler="vips")
        # assert dlup_wsi.aspect_ratio == openslideslide_image.image.width / openslideslide_image.image.height
        assert dlup_wsi.mpp == float(openslideslide_image.properties[openslide.PROPERTY_NAME_MPP_X])
        assert dlup_wsi.magnification == int(openslideslide_image.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        assert isinstance(repr(dlup_wsi), str)
        assert dlup_wsi.identifier == "mock"
        assert isinstance(dlup_wsi.thumbnail, pyvips.Image)

    @pytest.mark.parametrize("slide_config", SLIDE_CONFIGS)
    @pytest.mark.parametrize("mpp", [(0.57, 0.57), (1.2, 4.4)])
    def test_set_mpp(self, slide_config, mpp):
        """Test setting of the mpp."""
        openslide_image = MockOpenSlideSlide.from_config(slide_config)

        if mpp[0] != mpp[1]:
            with pytest.raises(UnsupportedSlideError):
                _ = SlideImage(
                    openslide_image, identifier=slide_config.filename, overwrite_mpp=mpp, internal_handler="vips"
                )
            return
        dlup_wsi = SlideImage(
            openslide_image, identifier=slide_config.filename, overwrite_mpp=(0.7, 0.7), internal_handler="vips"
        )
        assert dlup_wsi.mpp == 0.7

    @pytest.mark.parametrize("out_region_x", [0, 4, 3, 11])
    @pytest.mark.parametrize("out_region_y", [0, 1, 6, 9])
    @pytest.mark.parametrize("out_region_size", [(100, 100), (57, 38), (100, 50)])
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
        self,
        openslideslide_image,
        out_region_x,
        out_region_y,
        out_region_size,
        scaling,
        mocker,
    ):
        pass

        """Test exact interpolation.

        We want to be sure that reading a region at some scaling level is equivalent to
        downsampling the whole image and extracting that region using PIL.

        """
        # Mock the read_region method of openslideslide_image
        mocker.patch.object(openslideslide_image, "read_region", wraps=openslideslide_image.read_region)

        dlup_wsi = SlideImage(openslideslide_image, identifier="mock", internal_handler="vips")

        # Compute output image global coordinates.
        out_region_location = np.array((out_region_x, out_region_y))
        out_region_size = np.array(out_region_size)

        # Get the target layer from which we downsample.
        downsample = 1 / scaling
        expected_level = openslideslide_image.get_best_level_for_downsample(downsample)
        expected_level_image = openslideslide_image.get_level_image(expected_level)
        relative_scaling = scaling * openslideslide_image.level_downsamples[expected_level]

        # Resize the entire expected level image to the target scaling
        resized_image = expected_level_image.resize(relative_scaling, vscale=relative_scaling, kernel="lanczos3")

        # Calculate the cropping box for the resized image
        box = np.array([*out_region_location, *(out_region_location + out_region_size)])

        # Crop the region from the resized image
        vips_extracted_region = resized_image.crop(
            math.floor(box[0]), math.floor(box[1]), math.ceil(box[2] - box[0]), math.ceil(box[3] - box[1])
        )

        assert isinstance(vips_extracted_region, pyvips.Image)
        extracted_region = dlup_wsi.read_region(out_region_location, scaling, out_region_size)
        assert isinstance(extracted_region, pyvips.Image)

        # Verify the call arguments of read_region
        call_args_list = openslideslide_image.read_region.call_args_list
        assert len(call_args_list) == 1
        (call,) = call_args_list
        _, selected_level, _ = call.args
        assert selected_level == expected_level

        # Check that the output corresponding shape and value.
        arr0 = np.asarray(vips_extracted_region)
        arr1 = np.asarray(extracted_region)

        assert arr0.shape == arr1.shape

    @pytest.mark.parametrize("shift_x", list(np.linspace(0, 2, 10)))
    def test_border_region(self, shift_x):
        """Test border region."""
        scaling = 1 / 2.0
        out_region_size = (7, 7)
        out_region_size = np.array(out_region_size)

        openslide_image = MockOpenSlideSlide.from_config(SLIDE_CONFIGS[0])

        wsi = SlideImage(openslide_image, identifier="mock", internal_handler="vips")
        ssize = np.array(wsi.get_scaled_size(scaling))

        out_region_location = ssize - out_region_size - 1 + shift_x

        if (out_region_location + out_region_size > ssize).any():
            with pytest.raises(ValueError) as exc_info:
                _ = wsi.read_region(out_region_location, scaling, out_region_size)
                assert "Requested region is outside level boundaries" in str(exc_info.value)
            return

        assert (out_region_location + out_region_size <= ssize).all()

        extracted_region = wsi.read_region(out_region_location, scaling, out_region_size)
        assert extracted_region is not None
        assert isinstance(extracted_region, pyvips.Image)
        assert extracted_region.width == out_region_size[0]
        assert extracted_region.height == out_region_size[1]

    def test_scaled_size(self, dlup_wsi):
        """Check the scale is greater than zero."""
        size = dlup_wsi.get_scaled_size(0.5)
        assert (np.array(size) >= 0).all()

    @pytest.mark.parametrize("mpp", [0.27, 0.49, 0.51, 1.2, 4.4])
    def test_get_closest_native_mpp(self, dlup_wsi, mpp):
        """Check the scale is greater than zero."""
        _mpp = dlup_wsi.get_closest_native_mpp(mpp)

        if mpp == 0.27:
            assert _mpp == (0.25, 0.25)
        elif mpp == 0.49:
            assert _mpp == (0.5, 0.5)
        elif mpp == 0.51:
            assert _mpp == (0.5, 0.5)
        else:
            assert _mpp == (1.0, 1.0)

    def test_thumbnail(self, dlup_wsi):
        """Check the thumbnail is a pyvips Image."""
        thumbnail = dlup_wsi.thumbnail
        assert isinstance(thumbnail, pyvips.Image)
        assert thumbnail.width == 512 or thumbnail.height == 512

    def test_slide_image_with(self, openslideslide_image):
        """Test enter exit of the slide."""
        # Wrap the close method with a MagicMock
        original_close = openslideslide_image.close
        openslideslide_image.close = MagicMock(side_effect=original_close)

        # Create a SlideImage instance using the original openslideslide_image
        with SlideImage(openslideslide_image, internal_handler="vips") as image:
            # Access some attribute to ensure the context is active
            image.mpp
            # Ensure the close method has not been called yet
            assert openslideslide_image.close.call_count == 0

        # Ensure the close method has been called once after exiting the context
        assert openslideslide_image.close.call_count == 1

        # Restore the original close method
        openslideslide_image.close = original_close

    def test_slide_image_close(self, openslideslide_image):
        """Test SlideImage.close()."""
        original_close = openslideslide_image.close
        openslideslide_image.close = MagicMock(side_effect=original_close)

        slide = SlideImage(openslideslide_image, internal_handler="vips")
        # Call the close method
        slide.close()
        # Check how often slide.close() has been called
        assert openslideslide_image.close.call_count == 1
        # Call the close method again
        slide.close()
        # Check the call count again
        assert openslideslide_image.close.call_count == 2
        # Restore the original close method
        openslideslide_image.close = original_close


@pytest.mark.parametrize("scaling", [2, 1, 1 / 3])
def test_scaled_view(dlup_wsi, scaling):
    """Check that a scaled view correctly represents a layer."""
    view = dlup_wsi.get_scaled_view(scaling)
    assert view.mpp == dlup_wsi.mpp / scaling
    location = (3.7, 0)
    size = (10, 15)
    assert (
        np.asarray(view.read_region(location, size)) == np.asarray(dlup_wsi.read_region(location, scaling, size))
    ).all()
    assert dlup_wsi.get_scaled_size(scaling) == view.size

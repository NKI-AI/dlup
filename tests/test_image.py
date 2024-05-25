# Copyright (c) dlup contributors

"""Test the SlideImage class.
This set of tests ensures the SlideImage extracts the tiles
at a selected level by minimizing the information loss during
interpolation as well as ensuring the tiles are extracted
from the right level and locations of the original image.
"""
import pytest
import numpy as np
from dlup import SlideImage, UnsupportedSlideError

from .backends.test_openslide_backend import SLIDE_CONFIGS, MockOpenSlideSlide, SlideConfig
import pyvips


@pytest.fixture
def dlup_wsi():
    config = SlideConfig.from_parameters(
        filename="dummy1.svs",
        num_levels=3,
        level_0_dimensions=(10000, 10000),
        mpp=(0.25, 0.25),
        objective_power=20,
        vendor="dummy",
    )
    openslide_slide = MockOpenSlideSlide.from_config(config)
    return SlideImage(openslide_slide)


class TestSlideImage:
    """Test the dlup.SlideImage functionality."""

    # def test_mpp_exceptions(self):
    #     """Test that we break if the slide has no isotropic resolution."""
    #
    #     config = SlideConfig.from_parameters("dummy.svs", 3, (10000, 10000), (0.25, 0.25), 20, "dummy")
    #     with pytest.raises(UnsupportedSlideError):
    #         _ = SlideImage(openslide_slide)
    #
    # # def test_properties(self):
    # #     """Test properties."""
    # #     dlup_wsi = SlideImage(openslide_image, identifier="mock")
    # #     assert dlup_wsi.aspect_ratio == openslide_image.image.width / openslide_image.image.height
    # #     assert dlup_wsi.mpp == openslide_image.properties[openslide.PROPERTY_NAME_MPP_X]
    # #     assert dlup_wsi.magnification == openslide_image.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    # #     assert isinstance(repr(dlup_wsi), str)
    # #     assert dlup_wsi.identifier == "mock"
    # #     assert isinstance(dlup_wsi.thumbnail, pyvips.Image)
    # #
    @pytest.mark.parametrize("slide_config", SLIDE_CONFIGS)
    @pytest.mark.parametrize("mpp", [(0.57, 0.57), (1.2, 4.4)])
    def test_set_mpp(self, slide_config, mpp):
        """Test setting of the mpp."""
        openslide_image = MockOpenSlideSlide.from_config(slide_config)

        if mpp[0] != mpp[1]:
            with pytest.raises(UnsupportedSlideError):
                _ = SlideImage(openslide_image, identifier=slide_config.filename, overwrite_mpp=mpp)
            return
        dlup_wsi = SlideImage(openslide_image, identifier=slide_config.filename, overwrite_mpp=(0.7, 0.7))
        assert dlup_wsi.mpp == 0.7

    # @pytest.mark.parametrize(
    #     "slide_config",
    #     [
    #         SlideConfig(level_downsamples=(1.0, 2.0)),
    #     ],
    # )
    # @pytest.mark.parametrize("out_region_x", [0, 4.1, 1.5, 3.9])
    # @pytest.mark.parametrize("out_region_y", [0, 0.03, 4.2])
    # @pytest.mark.parametrize("out_region_size", [(4, 4), (7, 3), (1, 5)])
    # @pytest.mark.parametrize(
    #     "scaling",
    #     [
    #         2,
    #         1,
    #         1 / 3,
    #         1 / 7,
    #     ],
    # )
    # def test_read_region(
    #     self,
    #     mocker,
    #     dlup_wsi,
    #     openslide_image,
    #     out_region_x,
    #     out_region_y,
    #     out_region_size,
    #     scaling,
    # ):
    #     """Test exact interpolation.
    #
    #     We want to be sure that reading a region at some scaling level is equivalent to
    #     downsampling the whole image and extracting that region using PIL.
    #     """
    #     # TODO: Use these
    #     # base_image = openslide_image.image
    #     # base_image_size = np.array((base_image.width, base_image.height))
    #
    #     # Compute output image global coordinates.
    #     out_region_location = np.array((out_region_x, out_region_y))
    #     out_region_size = np.array(out_region_size)
    #
    #     # Get the target layer from which we downsample.
    #     downsampling = 1 / scaling
    #     expected_level = openslide_image.get_best_level_for_downsample(downsampling)
    #     expected_level_image = openslide_image.get_level_image(expected_level)
    #     relative_scaling = scaling * openslide_image.level_downsamples[expected_level]
    #
    #     # Notice that PIL box uses upper left and lower right rectangle coordinates.
    #     box = (*out_region_location, *(out_region_location + out_region_size))
    #     expected_level_box = np.array(box) / relative_scaling
    #     pil_extracted_region = expected_level_image.resize(
    #         out_region_size,
    #         resample=PIL.Image.Resampling.LANCZOS,
    #         box=expected_level_box,
    #     )
    #
    #     # Spy on how our mock object was called
    #     mocker.spy(openslide_image, "read_region")
    #
    #     # Use dlup read_region to extract the same location
    #     extracted_region = dlup_wsi.read_region(out_region_location, scaling, out_region_size)
    #
    #     # Is a PIL Image
    #     assert isinstance(extracted_region, pyvips.Image)
    #
    #     # Check that the right layer was indeed requested from our mock function.
    #     call_args_list = openslide_image.read_region.call_args_list
    #     assert len(call_args_list) == 1
    #     (call,) = call_args_list
    #     _, selected_level, _ = call.args
    #     assert selected_level == expected_level
    #
    #     # Check that the output corresponding shape and value.
    #     assert np.asarray(pil_extracted_region).shape == np.asarray(extracted_region).shape
    #     assert np.allclose(pil_extracted_region, extracted_region)
    #
    # @pytest.mark.parametrize("shift_x", list(np.linspace(0, 2, 10)))
    # def test_border_region(self, shift_x):
    #     """Test border region."""
    #     image_size = (128, 128)
    #     scaling = 1 / 5
    #     out_region_size = (7, 7)
    #     out_region_size = np.array(out_region_size)
    #
    #     # Example image
    #     image = get_sample_nonuniform_image().resize(image_size)
    #
    #     # Create a slide
    #     openslide_image = OpenSlideImageMock(
    #         image,
    #         {openslide.PROPERTY_NAME_MPP_X: 1.0, openslide.PROPERTY_NAME_MPP_Y: 1.0},
    #         (1.0, 4.3),
    #     )
    #
    #     wsi = SlideImage(openslide_image, identifier="mock")
    #     ssize = np.array(wsi.get_scaled_size(scaling))
    #
    #     out_region_location = ssize - out_region_size - 1 + shift_x
    #
    #     if (out_region_location + out_region_size > ssize).any():
    #         with pytest.raises(ValueError):
    #             extracted_region = wsi.read_region(out_region_location, scaling, out_region_size)
    #         return
    #
    #     extracted_region = wsi.read_region(out_region_location, scaling, out_region_size)
    #     # TODO: Make this test smarter
    #     assert extracted_region is not None
    #
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

    # def test_slide_image_with(self, mocker, openslide_image):
    #     """Test enter exit of the slide."""
    #     mocked_close = mocker.patch("openslide.ImageSlide.close")
    #     with SlideImage(openslide_image) as image:
    #         image.mpp
    #         assert mocked_close.call_count == 0
    #     assert mocked_close.call_count == 1
    #
    # def test_slide_image_close(self, mocker, openslide_image):
    #     """Test SlideImage.close()."""
    #     mocked_close = mocker.patch("openslide.ImageSlide.close")
    #     slide = SlideImage(openslide_image)
    #     assert mocked_close.call_count == 0
    #     slide.close()
    #     assert mocked_close.call_count == 1
    #


# @pytest.mark.parametrize("scaling", [2, 1, 1 / 3])
# def test_scaled_view(dlup_wsi, scaling):
#     """Check that a scaled view correctly represents a layer."""
#     view = dlup_wsi.get_scaled_view(scaling)
#     assert view.mpp == dlup_wsi.mpp / scaling
#     location = (3.7, 0)
#     size = (10, 15)
#     assert (
#         vips_to_numpy(view.read_region(location, size)) == \
#         vips_to_numpy(dlup_wsi.read_region(location, scaling, size))
#     ).all()
#     assert dlup_wsi.get_scaled_size(scaling) == view.size
#

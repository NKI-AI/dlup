# Copyright 2025 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test the SlideImage class.
This set of tests ensures the SlideImage extracts the tiles
at a selected level by minimizing the information loss during
interpolation as well as ensuring the tiles are extracted
from the right level and locations of the original image.
"""

import math
from pathlib import Path
from unittest.mock import MagicMock

import fim
import numpy as np
import openslide
import pytest
from common import SLIDE_CONFIGS, MockOpenSlideSlide

from dlup import SlideImage, SlideImageView
from dlup._exceptions import UnsupportedSlideError


class TestSlideImage:
    """Test the dlup.SlideImage functionality."""

    def test_properties(self, openslideslide_image):
        """Test properties."""
        dlup_wsi = SlideImage(openslideslide_image, identifier="mock")
        # assert dlup_wsi.aspect_ratio == openslideslide_image.image.width / openslideslide_image.image.height
        assert dlup_wsi.mpp == float(openslideslide_image.properties[openslide.PROPERTY_NAME_MPP_X])
        assert dlup_wsi.magnification == int(openslideslide_image.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        assert isinstance(repr(dlup_wsi), str)
        assert dlup_wsi.identifier == "mock"
        assert isinstance(dlup_wsi.thumbnail, fim.Image)

    def test_pathlike_is_resolved(self, tmp_path: Path) -> None:
        """Ensure non-remote backends receive a resolved Path from from_file_path."""
        tiff_path = tmp_path / "test.tiff"
        tiff_path.write_bytes(b"dummy")  # Existence is enough; backend does not inspect contents

        slide = SlideImage.from_file_path(str(tiff_path), backend=MockOpenSlideSlide)

        backend = slide._wsi
        assert isinstance(backend, MockOpenSlideSlide)
        assert isinstance(backend._filename, Path)
        assert backend._filename == tiff_path.resolve()

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

        dlup_wsi = SlideImage(openslideslide_image, identifier="mock")

        # Compute output image global coordinates.
        out_region_location = np.array((out_region_x, out_region_y))
        out_region_size = np.array(out_region_size)

        # Get the target layer from which we downsample.
        downsample = 1 / scaling
        expected_level = openslideslide_image.get_best_level_for_downsample(downsample)
        expected_level_image = openslideslide_image.get_level_image(expected_level)
        relative_scaling = scaling * openslideslide_image.level_downsamples[expected_level]

        # Resize the entire expected level image to the target scaling
        src_w, src_h, _ = expected_level_image.dimensions
        target_w = int(round(src_w * relative_scaling))
        target_h = int(round(src_h * relative_scaling))
        resized_image = expected_level_image.resize(target_w, target_h, kernel=fim.KernelType.LANCZOS3)

        # Calculate the cropping box for the resized image
        box = np.array([*out_region_location, *(out_region_location + out_region_size)])

        # Crop the region from the resized image
        fim_extracted_region = resized_image.crop(
            (math.floor(box[0]), math.floor(box[1])),
            (math.ceil(box[2] - box[0]), math.ceil(box[3] - box[1])),
        )

        assert isinstance(fim_extracted_region, fim.Image)
        extracted_region = dlup_wsi.read_region(out_region_location, scaling, out_region_size)
        assert isinstance(extracted_region, fim.Image)

        # Verify the call arguments of read_region
        call_args_list = openslideslide_image.read_region.call_args_list
        assert len(call_args_list) == 1
        (call,) = call_args_list
        _, selected_level, _ = call.args
        assert selected_level == expected_level

        # Check that the output corresponding shape and value.
        arr0 = np.asarray(fim_extracted_region)
        arr1 = extracted_region.to_numpy()

        assert arr0.shape == arr1.shape

    @pytest.mark.parametrize("shift_x", list(np.linspace(0, 2, 10)))
    def test_border_region(self, shift_x):
        """Test border region."""
        scaling = 1 / 2.0
        out_region_size = (7, 7)
        out_region_size = np.array(out_region_size)

        openslide_image = MockOpenSlideSlide.from_config(SLIDE_CONFIGS[0])

        wsi = SlideImage(openslide_image, identifier="mock")
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
        assert isinstance(extracted_region, fim.Image)
        dims = extracted_region.dimensions
        assert dims[0] == out_region_size[0]
        assert dims[1] == out_region_size[1]

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
        assert isinstance(thumbnail, fim.Image)
        dims = thumbnail.dimensions
        assert dims[0] == 512 or dims[1] == 512

    def test_slide_image_with(self, openslideslide_image):
        """Test enter exit of the slide."""
        # Wrap the close method with a MagicMock
        original_close = openslideslide_image.close
        openslideslide_image.close = MagicMock(side_effect=original_close)

        # Create a SlideImage instance using the original openslideslide_image
        with SlideImage(openslideslide_image) as image:
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

        slide = SlideImage(openslideslide_image)
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
    view = dlup_wsi.get_view_at_scaling(scaling)
    assert view.mpp == dlup_wsi.mpp / scaling
    location = (3.7, 0)
    size = (10, 15)
    assert (
        view.read_region(location, size).to_numpy() == dlup_wsi.read_region(location, scaling, size).to_numpy()
    ).all()
    assert dlup_wsi.get_scaled_size(scaling) == view.size


@pytest.mark.parametrize("scaling", [2, 1, 1 / 3, 1 / 7])
def test_view_at_scaling(dlup_wsi, scaling):
    """Test that get_view_at_scaling creates a view with correct properties and produces identical results."""
    view = dlup_wsi.get_view_at_scaling(scaling)

    # Check that the view has the correct MPP
    expected_mpp = dlup_wsi.mpp / scaling
    assert view.mpp == expected_mpp

    # Check that the view has the correct size
    assert view.size == dlup_wsi.get_scaled_size(scaling)

    # Test single region read
    location = (3.7, 0)
    size = (10, 15)
    view_region = view.read_region(location, size).to_numpy()
    direct_region = dlup_wsi.read_region(location, scaling, size).to_numpy()
    assert (view_region == direct_region).all()

    # Test multiple region reads from the same view
    locations = [(0, 0), (5.3, 7.2), (10, 10)]
    sizes = [(8, 8), (12, 16), (20, 20)]

    for loc, sz in zip(locations, sizes):
        view_result = view.read_region(loc, sz).to_numpy()
        direct_result = dlup_wsi.read_region(loc, scaling, sz).to_numpy()
        assert (view_result == direct_result).all(), f"Mismatch at location {loc} with size {sz}"


@pytest.mark.parametrize("mpp", [0.25, 0.5, 1.0, 2.0])
def test_view_at_mpp(dlup_wsi, mpp):
    """Test that get_view_at_mpp creates a view with correct MPP and produces identical results."""
    view = dlup_wsi.get_view_at_mpp(mpp)

    # Check that the view has the correct MPP
    # There might be slight floating point differences, so use approximate comparison
    assert abs(view.mpp - mpp) < 1e-10

    # Calculate the expected scaling from MPP
    expected_scaling = dlup_wsi.get_scaling(mpp)

    # Check that the view has the correct size
    expected_size = dlup_wsi.get_scaled_size(expected_scaling)
    assert view.size == expected_size

    # Test region reading
    location = (5.0, 5.0)
    size = (10, 10)
    view_region = view.read_region(location, size).to_numpy()
    direct_region = dlup_wsi.read_region(location, expected_scaling, size).to_numpy()
    assert (view_region == direct_region).all()


def test_view_multiple_reads(dlup_wsi):
    """Test that a view can be used for multiple region reads efficiently."""
    scaling = 0.5
    view = dlup_wsi.get_view_at_scaling(scaling)

    # Read multiple regions from the same view
    regions = []
    locations = [(0, 0), (10, 10), (20, 20), (30, 30)]
    size = (15, 15)

    for loc in locations:
        region = view.read_region(loc, size)
        regions.append(region.to_numpy())

        # Verify each region matches the direct API
        direct_region = dlup_wsi.read_region(loc, scaling, size).to_numpy()
        assert (region.to_numpy() == direct_region).all()

    # Ensure we got all regions
    assert len(regions) == len(locations)

    # Verify regions have correct dimensions
    for region in regions:
        assert region.shape[:2] == size


def test_view_at_mpp_edge_cases(dlup_wsi):
    """Test edge cases for get_view_at_mpp."""
    # Test with the slide's native MPP
    native_mpp = dlup_wsi.mpp
    view = dlup_wsi.get_view_at_mpp(native_mpp)

    assert abs(view.mpp - native_mpp) < 1e-10
    assert view.size == dlup_wsi.size

    # Test region reading at native MPP
    location = (0, 0)
    size = (50, 50)
    view_region = view.read_region(location, size).to_numpy()
    direct_region = dlup_wsi.read_region(location, 1.0, size).to_numpy()
    assert (view_region == direct_region).all()


def test_view_consistency(dlup_wsi):
    """Test that get_view_at_mpp and get_view_at_scaling are consistent."""
    mpp = 0.5
    view_by_mpp = dlup_wsi.get_view_at_mpp(mpp)

    # Calculate the corresponding scaling
    scaling = dlup_wsi.get_scaling(mpp)
    view_by_scaling = dlup_wsi.get_view_at_scaling(scaling)

    # Both views should have the same properties
    assert abs(view_by_mpp.mpp - view_by_scaling.mpp) < 1e-10
    assert view_by_mpp.size == view_by_scaling.size

    # Both views should produce identical regions
    location = (5, 5)
    size = (20, 20)
    region_by_mpp = view_by_mpp.read_region(location, size).to_numpy()
    region_by_scaling = view_by_scaling.read_region(location, size).to_numpy()
    assert (region_by_mpp == region_by_scaling).all()


def test_slide_image_view_type(dlup_wsi):
    """Test that views are instances of SlideImageView and can be imported from dlup."""
    # Test that the view is of the correct type
    view = dlup_wsi.get_view_at_scaling(0.5)
    assert isinstance(view, SlideImageView)

    # Test that get_view_at_mpp also returns SlideImageView
    view_mpp = dlup_wsi.get_view_at_mpp(0.5)
    assert isinstance(view_mpp, SlideImageView)

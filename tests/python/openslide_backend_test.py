import math
from unittest.mock import MagicMock, patch

import fim
import numpy as np
import openslide
import openslide.lowlevel as openslide_lowlevel
import pytest
from dlup._exceptions import UnsupportedSlideError
from dlup.backends.openslide_backend import (
    TIFF_PROPERTY_NAME_RESOLUTION_UNIT,
    TIFF_PROPERTY_NAME_X_RESOLUTION,
    TIFF_PROPERTY_NAME_Y_RESOLUTION,
    _get_mpp_from_tiff,
)
from common import MockOpenSlideSlide, SlideConfig, SLIDE_CONFIGS, mock_lowlevel


class TestMockOpenSlideSlide:
    @pytest.mark.parametrize("config", SLIDE_CONFIGS)
    def test_properties(self, config):
        slide = MockOpenSlideSlide.from_config(config)

        # Test properties
        if openslide.PROPERTY_NAME_MPP_X in config.properties and openslide.PROPERTY_NAME_MPP_Y in config.properties:
            expected_spacing = (
                float(config.properties[openslide.PROPERTY_NAME_MPP_X]),
                float(config.properties[openslide.PROPERTY_NAME_MPP_X]),
            )
        else:
            expected_spacing = None
        assert slide.spacing == expected_spacing

        expected_magnification = (
            int(config.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            if openslide.PROPERTY_NAME_OBJECTIVE_POWER in config.properties
            else None
        )
        assert slide.magnification == expected_magnification

        expected_vendor = (
            config.properties[openslide.PROPERTY_NAME_VENDOR]
            if openslide.PROPERTY_NAME_VENDOR in config.properties
            else None
        )
        assert slide.vendor == expected_vendor

        levels = config.levels
        assert slide.level_count == len(levels)
        assert slide.level_dimensions == tuple(level.dimensions for level in levels)
        assert slide.level_downsamples == tuple(level.downsample for level in levels)

    @pytest.mark.parametrize("coordinates", [(0, 0), (500, 100)])
    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("region_size", [(0, 0), (-1, -1), (100, 150)])
    def test_read_region(self, coordinates, level, region_size):
        config = SLIDE_CONFIGS[0]  # Use the first config for read region tests
        slide = MockOpenSlideSlide.from_config(config)

        if region_size[0] <= 0 or region_size[1] <= 0:
            with pytest.raises(
                openslide_lowlevel.OpenSlideError,
                match=r"width \(%s\) or height \(%s\) must be positive" % (region_size[0], region_size[1]),
            ):
                slide.read_region(coordinates, level, region_size)
            return

        region = slide.read_region(coordinates, level, region_size)
        assert isinstance(region, fim.Image)
        dims = region.dimensions
        assert dims[0] == region_size[0]  # width
        assert dims[1] == region_size[1]  # height

    def test_broken_mpp(self):
        config = SlideConfig.from_parameters("dummy.svs", 3, (1000, 1000), (0.0, 0.25), 40, "dummy")
        with pytest.raises(UnsupportedSlideError, match=r"Unable to parse mpp."):
            _ = MockOpenSlideSlide.from_config(config)

        config = SlideConfig.from_parameters("dummy.svs", 3, (1000, 1000), (0.25, 3.0), 40, "dummy")
        with pytest.raises(UnsupportedSlideError) as exc_info:
            _ = MockOpenSlideSlide.from_config(config)
        assert "cannot deal with slides having anisotropic mpps." in str(exc_info.value)

    def test_mock_calls(self):
        """Test that a basic slide creation works with the mock."""
        # Use simple config with isotropic mpp
        config = SlideConfig.from_parameters(
            filename="test.svs",
            num_levels=2,
            level_0_dimensions=(100, 100),
            mpp=(0.5, 0.5),
            objective_power=40,
            vendor="test",
        )

        # Create a slide using from_config, which should set up all the mock correctly
        slide = MockOpenSlideSlide.from_config(config)

        # Just make some basic assertions about the slide
        assert slide.level_count == 2
        assert slide.dimensions == (100, 100)
        assert slide.spacing == (0.5, 0.5)
        assert slide.magnification == 40
        assert slide.vendor == "test"


@patch("openslide.__library_version__", "3.4.1")
def test__get_mpp_from_tiff_returns_none_lower_version():
    properties = {
        openslide.PROPERTY_NAME_VENDOR: "generic-tiff",
        TIFF_PROPERTY_NAME_RESOLUTION_UNIT: "cm",
        TIFF_PROPERTY_NAME_X_RESOLUTION: "254",
        TIFF_PROPERTY_NAME_Y_RESOLUTION: "127",
    }
    expected_mpp = (39.37007874015748, 39.37007874015748 * 2)
    result = _get_mpp_from_tiff(properties)
    assert result == expected_mpp, f"Expected {expected_mpp}, got {result}"


@patch("openslide.__library_version__", "4.0.0")
def test___get_mpp_from_tiff_returns_correct_higher_version():
    properties = {
        openslide.PROPERTY_NAME_VENDOR: "generic-tiff",
        TIFF_PROPERTY_NAME_RESOLUTION_UNIT: "cm",
        TIFF_PROPERTY_NAME_X_RESOLUTION: "254",
        TIFF_PROPERTY_NAME_Y_RESOLUTION: "254",
    }
    result = _get_mpp_from_tiff(properties)
    assert result is None, f"Expected None, got {result}"

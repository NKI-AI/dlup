import math
from unittest.mock import MagicMock, patch

import numpy as np
import openslide
import openslide.lowlevel as openslide_lowlevel
import pytest
import pyvips

from dlup import UnsupportedSlideError
from dlup.backends.openslide_backend import (
    TIFF_PROPERTY_NAME_RESOLUTION_UNIT,
    TIFF_PROPERTY_NAME_X_RESOLUTION,
    TIFF_PROPERTY_NAME_Y_RESOLUTION,
    OpenSlideSlide,
    _get_mpp_from_tiff,
)
from dlup.types import PathLike

from ..common import SlideConfig, get_sample_nonuniform_image

SLIDE_CONFIGS = [
    SlideConfig.from_parameters(
        filename="dummy1.svs",
        num_levels=3,
        level_0_dimensions=(2000, 2000),
        mpp=(0.25, 0.25),
        objective_power=40,
        vendor="dummy",
    ),
    SlideConfig.from_parameters(
        filename="dummy2.svs",
        num_levels=3,
        level_0_dimensions=(1800, 2000),
        mpp=(0.50, 0.50),
        objective_power=20,
        vendor="test_vendor",
    ),
]


class MockOpenSlideLowLevel:
    def __init__(self, config):
        self.config = config
        self.mock_open = MagicMock()
        levels = config.levels
        self.mock_get_property_names = MagicMock(return_value=list(config.properties.keys()))
        self.mock_get_property_value = MagicMock(side_effect=lambda _owsi, key: config.properties[key])
        self.mock_get_level_count = MagicMock(return_value=len(levels))
        self.mock_get_level_dimensions = MagicMock(side_effect=lambda _owsi, idx: levels[idx].dimensions)
        self.mock_get_level_downsample = MagicMock(side_effect=lambda _owsi, idx: levels[idx].downsample)
        self.mock_close = MagicMock()

        self.mock_read_region = MagicMock(side_effect=self.mock_read_region_fn)
        self.base_image = get_sample_nonuniform_image(config.levels[0].dimensions)

    def mock_read_region_fn(self, _owsi, x, y, level, w, h):
        downsample_factor = self.config.levels[level].downsample

        # Calculate coordinates and size at level 0
        w0, h0 = math.ceil(w * downsample_factor), math.ceil(h * downsample_factor)

        # Crop the base image
        cropped_image = self.base_image.crop(x, y, w0, h0)

        # Resize the cropped image to the requested level
        if downsample_factor != 1.0:
            level_image = cropped_image.resize(1.0 / downsample_factor)
        else:
            level_image = cropped_image

        np_buffer = np.ndarray(
            buffer=level_image.write_to_memory(),
            dtype=np.uint8,
            shape=(level_image.height * level_image.width * 4,),  # Flatten the shape directly
        )

        # Convert numpy buffer to pyvips image
        vips_image = pyvips.Image.new_from_memory(np_buffer, w, h, 4, "uchar")
        return vips_image


mock_lowlevel = None


class MockOpenSlideSlide(OpenSlideSlide):
    def __init__(self, filename: PathLike):
        global mock_lowlevel
        self._filename = filename
        self._owsi = "mocked_owsi"
        self._spacings = []

        mock_lowlevel.mock_open.return_value = self._owsi
        self._owsi = mock_lowlevel.mock_open(filename)

        try:
            mpp_x = float(self.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.properties[openslide.PROPERTY_NAME_MPP_Y])
            self.spacing = (mpp_x, mpp_y)
        except KeyError:
            spacing = _get_mpp_from_tiff(dict(self.properties))
            if spacing:
                self.spacing = spacing

    @property
    def properties(self) -> dict[str, str]:
        return {
            key: mock_lowlevel.mock_get_property_value(self._owsi, key)
            for key in mock_lowlevel.mock_get_property_names(self._owsi)
        }

    @property
    def level_count(self) -> int:
        return mock_lowlevel.mock_get_level_count(self._owsi)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple(mock_lowlevel.mock_get_level_dimensions(self._owsi, idx) for idx in range(self.level_count))

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return tuple(mock_lowlevel.mock_get_level_downsample(self._owsi, idx) for idx in range(self.level_count))

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
        if size[0] <= 0 or size[1] <= 0:
            raise openslide_lowlevel.OpenSlideError(f"width ({size[0]}) or height ({size[1]}) must be positive")
        return mock_lowlevel.mock_read_region(self._owsi, coordinates[0], coordinates[1], level, size[0], size[1])

    def get_level_image(self, level: int) -> pyvips.Image:
        base_image = mock_lowlevel.base_image.resize(1.0 / self.level_downsamples[level])
        return base_image

    def close(self) -> None:
        mock_lowlevel.mock_close(self._owsi)

    @classmethod
    def from_config(cls, config: SlideConfig):
        global mock_lowlevel
        mock_lowlevel = MockOpenSlideLowLevel(config)
        return cls(config.filename)


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
        assert isinstance(region, pyvips.Image)
        assert region.width == region_size[0]
        assert region.height == region_size[1]

    def test_broken_mpp(self):
        config = SlideConfig.from_parameters("dummy.svs", 3, (1000, 1000), (0.0, 0.25), 40, "dummy")
        with pytest.raises(UnsupportedSlideError, match=r"Unable to parse mpp."):
            _ = MockOpenSlideSlide.from_config(config)

        config = SlideConfig.from_parameters("dummy.svs", 3, (1000, 1000), (0.25, 3.0), 40, "dummy")
        with pytest.raises(UnsupportedSlideError) as exc_info:
            _ = MockOpenSlideSlide.from_config(config)
        assert "cannot deal with slides having anisotropic mpps." in str(exc_info.value)

    def test_mock_calls(self):
        config = SLIDE_CONFIGS[0]  # Use the first config for mock call tests
        _ = MockOpenSlideSlide.from_config(config)
        # Ensure the mocks were called
        mock_lowlevel.mock_open.assert_called_once_with(config.filename)
        mock_lowlevel.mock_get_property_names.assert_called()
        assert mock_lowlevel.mock_get_property_value.call_count == len(config.properties) * 2
        mock_lowlevel.mock_get_level_count.assert_called_once()
        assert mock_lowlevel.mock_get_level_dimensions.call_count == 0
        assert mock_lowlevel.mock_get_level_downsample.call_count == len(config.levels)
        mock_lowlevel.mock_read_region.assert_not_called()  # read_region not called in this test


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

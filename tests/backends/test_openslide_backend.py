import ctypes
from ctypes import c_uint32
from unittest.mock import MagicMock

import numpy as np
import openslide.lowlevel as openslide_lowlevel
import pytest
import pyvips

from dlup.backends.openslide_backend import OpenSlideSlide

from .test_common import get_sample_nonuniform_image


# Mock the necessary low-level functions from openslide_lowlevel
class MockOpenSlideLowLevel:
    def __init__(self):
        self.mock_open = MagicMock(name="open")
        self.mock_get_property_names = MagicMock(name="get_property_names")
        self.mock_get_property_value = MagicMock(name="get_property_value")
        self.mock_get_level_count = MagicMock(name="get_level_count")
        self.mock_get_level_dimensions = MagicMock(name="get_level_dimensions")
        self.mock_get_level_downsample = MagicMock(name="get_level_downsample")
        self.mock_read_region = MagicMock(name="_read_region")
        self.mock_close = MagicMock(name="close")

    def patch(self, mocker):
        mocker.patch("openslide.lowlevel.open", self.mock_open)
        mocker.patch("openslide.lowlevel.get_property_names", self.mock_get_property_names)
        mocker.patch("openslide.lowlevel.get_property_value", self.mock_get_property_value)
        mocker.patch("openslide.lowlevel.get_level_count", self.mock_get_level_count)
        mocker.patch("openslide.lowlevel.get_level_dimensions", self.mock_get_level_dimensions)
        mocker.patch("openslide.lowlevel.get_level_downsample", self.mock_get_level_downsample)
        mocker.patch("openslide.lowlevel._read_region", self.mock_read_region)
        mocker.patch("openslide.lowlevel.close", self.mock_close)


@pytest.fixture
def mock_lowlevel(mocker):
    mock = MockOpenSlideLowLevel()
    mock.patch(mocker)
    return mock


def mock_read_region_fn(slide, buf, x, y, level, w, h):
    downsample_factor = [1.0, 2.0, 4.0][level]
    base_image = get_sample_nonuniform_image((2000, 2000))  # Create base image at level 0

    # Calculate coordinates and size at level 0
    x0, y0 = int(x * downsample_factor), int(y * downsample_factor)
    w0, h0 = int(w * downsample_factor), int(h * downsample_factor)

    # Crop the base image
    cropped_image = base_image.crop(x0, y0, w0, h0)

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

    # Copy numpy buffer to ctypes buffer
    ctypes_array = (c_uint32 * (w * h))()
    ctypes.memmove(ctypes_array, np_buffer.ctypes.data, np_buffer.nbytes)

    for i in range(len(ctypes_array)):
        buf[i] = ctypes_array[i]


def create_slide_from_config(mock_lowlevel, filename, properties, levels):
    mock_lowlevel.mock_open.return_value = MagicMock(name="openslide_object")
    mock_lowlevel.mock_get_property_names.return_value = list(properties.keys())
    mock_lowlevel.mock_get_property_value.side_effect = lambda slide, key: properties[key]
    mock_lowlevel.mock_get_level_count.return_value = len(levels)
    mock_lowlevel.mock_get_level_dimensions.side_effect = lambda slide, idx: levels[idx]["dimensions"]
    mock_lowlevel.mock_get_level_downsample.side_effect = lambda slide, idx: levels[idx]["downsample"]
    mock_lowlevel.mock_read_region.side_effect = mock_read_region_fn

    slide = OpenSlideSlide(filename)
    slide._owsi = mock_lowlevel.mock_open.return_value  # Ensure slide uses the mocked object
    return slide


SLIDE_CONFIGS = [
    {
        "filename": "dummy1.svs",
        "properties": {
            "openslide.mpp-x": "0.25",
            "openslide.mpp-y": "0.25",
            "openslide.objective-power": "20",
            "openslide.vendor": "dummy",
        },
        "levels": [
            {"dimensions": (10000, 10000), "downsample": 1.0},
            {"dimensions": (5000, 5000), "downsample": 2.0},
            {"dimensions": (2500, 2500), "downsample": 4.0},
        ],
    },
    {
        "filename": "dummy2.svs",
        "properties": {
            "openslide.mpp-x": "0.50",
            "openslide.mpp-y": "0.50",
            "openslide.vendor": "test_vendor",
        },
        "levels": [
            {"dimensions": (10000, 10000), "downsample": 1.0},
            {"dimensions": (5000, 5000), "downsample": 2.0},
            {"dimensions": (2500, 2500), "downsample": 4.0},
        ],
    },
]


class TestOpenSlideSlide:
    @pytest.mark.parametrize("config", SLIDE_CONFIGS)
    def test_properties(self, config, mock_lowlevel):
        slide = create_slide_from_config(mock_lowlevel, config["filename"], config["properties"], config["levels"])

        # Test properties
        if "openslide.mpp-x" in config["properties"] and "openslide.mpp-y" in config["properties"]:
            expected_spacing = (
                float(config["properties"]["openslide.mpp-x"]),
                float(config["properties"]["openslide.mpp-y"]),
            )
        else:
            expected_spacing = None
        assert slide.spacing == expected_spacing

        expected_magnification = (
            int(config["properties"]["openslide.objective-power"])
            if "openslide.objective-power" in config["properties"]
            else None
        )
        assert slide.magnification == expected_magnification

        expected_vendor = (
            config["properties"]["openslide.vendor"] if "openslide.vendor" in config["properties"] else None
        )
        assert slide.vendor == expected_vendor

        assert slide.level_count == len(config["levels"])
        assert slide.level_dimensions == tuple(level["dimensions"] for level in config["levels"])
        assert slide.level_downsamples == tuple(level["downsample"] for level in config["levels"])

    @pytest.mark.parametrize("coordinates", [(0, 0), (500, 100)])
    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("region_size", [(0, 0), (-1, -1), (100, 150)])
    def test_read_region(self, coordinates, level, region_size, mock_lowlevel):
        config = SLIDE_CONFIGS[0]  # Use the first config for read region tests
        slide = create_slide_from_config(mock_lowlevel, config["filename"], config["properties"], config["levels"])

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

    def test_mock_calls(self, mock_lowlevel):
        config = SLIDE_CONFIGS[0]  # Use the first config for mock call tests
        _ = create_slide_from_config(mock_lowlevel, config["filename"], config["properties"], config["levels"])
        # Ensure the mocks were called
        mock_lowlevel.mock_open.assert_called_once_with(config["filename"])
        mock_lowlevel.mock_get_property_names.assert_called()
        assert mock_lowlevel.mock_get_property_value.call_count == len(config["properties"]) * len(SLIDE_CONFIGS)
        mock_lowlevel.mock_get_level_count.assert_called_once()
        assert mock_lowlevel.mock_get_level_dimensions.call_count == 0
        assert mock_lowlevel.mock_get_level_downsample.call_count == len(config["levels"])
        mock_lowlevel.mock_read_region.assert_not_called()  # read_region not called in this test

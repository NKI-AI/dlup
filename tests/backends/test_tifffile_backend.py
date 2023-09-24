# Copyright (c) dlup contributors
from unittest.mock import Mock, patch

import pytest

from dlup.experimental_backends.tifffile_backend import TifffileSlide


@pytest.fixture
def mock_tifffile_slide():
    with patch("tifffile.TiffFile") as MockTiffFile:
        # List to hold mock pages
        mock_pages = []

        # Starting values
        size = 4096
        res_value = (1, 1)  # Using a tuple since the code accesses the numerator and denominator

        # Create 3 mock pages (or however many you need)
        for _ in range(3):
            # Create mock tags for the current page
            x_res_mock = Mock(value=res_value)
            y_res_mock = Mock(value=res_value)
            unit_mock = Mock(value=3)

            mock_page = Mock()
            mock_page.shape = [3, size, size]
            mock_page.tags = {"XResolution": x_res_mock, "YResolution": y_res_mock, "ResolutionUnit": unit_mock}

            mock_pages.append(mock_page)

            # Halve the values for the next iteration
            size //= 2
            x_res_value = (res_value[0], res_value[1] * 2)  # To halve the resolution

        instance = MockTiffFile.return_value
        instance.pages = mock_pages
        yield TifffileSlide("path_to_image.tif")


class TestTifffileSlide:
    def test_initialization(self, mock_tifffile_slide):
        slide = mock_tifffile_slide
        assert slide._level_count == 3  # Checking the initialized _level_count

    def test_properties(self, mock_tifffile_slide):
        slide = mock_tifffile_slide

        assert slide.vendor is None
        assert slide.magnification is None

        # Check the properties

    def test_read_region_invalid_level(self, mock_tifffile_slide):
        slide = mock_tifffile_slide
        with pytest.raises(RuntimeError, match="Level 4 not present."):
            slide.read_region((0, 0), 4, (100, 100))

    def test_close(self, mock_tifffile_slide):
        slide = mock_tifffile_slide
        slide.close()
        slide._image.close.assert_called_once()

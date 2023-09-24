# Copyright (c) dlup contributors
from unittest.mock import Mock

import pytest

from dlup.utils.tifffile_utils import (
    _compute_tile_indices,
    _get_tile_from_data,
    _retrieve_tile_data,
    _validate_inputs,
    get_tile,
)


# 1. Testing input validation
def test_validate_inputs():
    mock_page = Mock()
    mock_page.is_tiled = True
    mock_page.imagewidth = 500
    mock_page.imagelength = 500

    # Test for non-tiled page
    mock_page.is_tiled = False
    with pytest.raises(ValueError, match="Input page must be tiled."):
        _validate_inputs(mock_page, (0, 0), (100, 100))

    mock_page.is_tiled = True

    # Test for negative coordinates
    with pytest.raises(ValueError, match="Requested crop area is out of image bounds."):
        _validate_inputs(mock_page, (-10, -10), (100, 100))

    # Test for out-of-bounds coordinates
    with pytest.raises(ValueError, match="Requested crop area is out of image bounds."):
        _validate_inputs(mock_page, (450, 450), (100, 100))


# 2. Testing computation of tile indices
def test_compute_tile_indices():
    mock_page = Mock()
    mock_page.tilewidth = 200
    mock_page.tilelength = 200

    tile_y0, tile_y1, tile_x0, tile_x1 = _compute_tile_indices(mock_page, (50, 50), (100, 100))
    assert (tile_y0, tile_y1, tile_x0, tile_x1) == (0, 1, 0, 1)


def create_mock_page(imagewidth, imageheight, tilewidth, tilelength, dataoffsets, databytecounts):
    mock_page = Mock()
    mock_page.imagewidth = imagewidth
    mock_page.imageheight = imageheight
    mock_page.tilewidth = tilewidth
    mock_page.tilelength = tilelength
    mock_page.dataoffsets = dataoffsets
    mock_page.databytecounts = databytecounts
    mock_page.tags = {"JPEGTables": None}
    mock_filehandle = Mock()
    mock_filehandle.read.return_value = b"some_data"
    mock_page.parent.filehandle = mock_filehandle
    return mock_page

# coding=utf-8
# Copyright (c) dlup contributors

"""Test the datasets facility classes."""

import numpy as np
import pytest
import dlup
from dlup.data.dataset import TiledROIsSlideImageDataset, TilingMode


@pytest.mark.parametrize("tile_mode", [TilingMode.skip])
def test_tiled_level_slide_image_dataset(monkeypatch, dlup_wsi, tiling_mode):
    """Test a single image dataset."""
    monkeypatch.setattr(TiledROIsSlideImageDataset, "slide_image", dlup_wsi)
    monkeypatch.setattr(dlup.SlideImage, "from_file_path", lambda x: dlup_wsi)
    dataset = TiledROIsSlideImageDataset.from_standard_tiling("dummy", 1.0, (32, 24), (0, 0), tiling_mode, None)
    tile_data = dataset[0]
    tile = tile_data["image"]
    coordinates = tile_data["coordinates"]

    # Numpy array has height, width, channels.
    # Images have width, height, channels.
    assert np.asarray(tile).shape == (24, 32, 4)
    assert len(coordinates) == 2

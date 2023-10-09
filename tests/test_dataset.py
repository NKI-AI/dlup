# coding=utf-8
# Copyright (c) dlup contributors

"""Test the datasets facility classes."""

import numpy as np

import dlup
from dlup.data.dataset import TiledWsiDataset, TilingMode


def test_tiled_level_slide_image_dataset(monkeypatch, dlup_wsi):
    """Test a single image dataset."""
    monkeypatch.setattr(TiledWsiDataset, "slide_image", dlup_wsi)
    monkeypatch.setattr(dlup.SlideImage, "from_file_path", lambda x, backend: dlup_wsi)
    dataset = TiledWsiDataset.from_standard_tiling(
        "dummy",
        mpp=1.0,
        tile_size=(32, 24),
        tile_overlap=(0, 0),
        tile_mode=TilingMode.skip,
        mask=None,
    )
    tile_data = dataset[0]
    tile = tile_data["image"]
    coordinates = tile_data["coordinates"]

    # Numpy array has height, width, channels.
    # Images have width, height, channels.
    assert np.asarray(tile).shape == (24, 32, 4)
    assert len(coordinates) == 2

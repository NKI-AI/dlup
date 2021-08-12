# coding=utf-8
# Copyright (c) dlup contributors

"""Test the datasets facility classes."""

import pathlib

from dlup.data.dataset import TiledLevelSlideImageDataset


def test_tiled_level_slide_image_dataset(monkeypatch, dlup_wsi):
    """Test a single image dataset."""
    monkeypatch.setattr(TiledLevelSlideImageDataset, "slide_image", dlup_wsi)
    dataset = TiledLevelSlideImageDataset("dummy", 1.0, (32, 24), (0, 0), "skip", None)
    tile_data = dataset[0]
    tile = tile_data["image"]
    coordinates = tile_data["coordinates"]

    # Numpy array has height, width, channels.
    # Images have width, height, channels.
    assert tile.shape == (24, 32, 4)
    assert len(coordinates) == 2

# Copyright (c) dlup contributors

"""Test the datasets facility classes."""

import numpy as np
import pytest

import dlup
from dlup.data.dataset import TiledWsiDataset, TilingMode


def _dataset_asserts(tile_data):
    tile = tile_data["image"]
    coordinates = tile_data["coordinates"]

    # Numpy array has height, width, channels.
    # Images have width, height, channels.
    assert np.asarray(tile).shape == (24, 32, 4)
    assert len(coordinates) == 2


@pytest.mark.parametrize("index", [0, slice(0, 1)])
def test_tiled_level_slide_image_dataset(monkeypatch, dlup_wsi, index):
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

    if isinstance(index, int):
        _dataset_asserts(dataset[index])
    else:
        tiles_data = dataset[index]
        for tile_data in tiles_data:
            _dataset_asserts(tile_data)

    # We can iterate through the dataset.
    assert len(dataset) == 80
    for idx, sample in enumerate(dataset):
        _dataset_asserts(sample)

    assert idx == 79

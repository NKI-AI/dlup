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
"""Test the datasets facility classes."""

from unittest.mock import patch

import dlup
import numpy as np
import pytest
from dlup.data.dataset import ConcatDataset, Dataset, MaskConfig, SlideDataset, TilingMode, TilingConfig


@patch.object(dlup.SlideImage, "from_file_path")
@patch.object(SlideDataset, "slide_image")
def test_tiled_level_slide_image_dataset(mock_slide_image, mock_from_file_path, dlup_wsi):
    """Test a single image dataset."""
    mock_from_file_path.return_value = dlup_wsi
    with patch.object(SlideDataset, "slide_image", dlup_wsi):
        tiling_config = TilingConfig(
            mpp=1.0,
            tile_size=(32, 24),
            tile_overlap=(0, 0),
            tile_mode=TilingMode.skip,
        )
        dataset = SlideDataset.from_standard_tiling(
            "dummy",
            tiling_config=tiling_config,
        )
        tile_data = dataset[0]
        tile = tile_data.image
        coordinates = tile_data.coordinates

        # Numpy array has height, width, channels.
        # Images have width, height, channels.
        assert tile.to_numpy().shape == (24, 32, 4)
        assert len(coordinates) == 2

        assert len(dataset) == 70

        # Let's grab a few samples instead of one
        tile_data = dataset[0:2]
        assert len(tile_data) == 2
        assert all([tile.image.to_numpy().shape == (24, 32, 4) for tile in tile_data])


class TestTilingConfig:
    """Test the TilingConfig pydantic model."""

    def test_valid_config(self):
        """Test creating a valid TilingConfig."""
        config = TilingConfig(
            mpp=0.5,
            tile_size=(512, 512),
            tile_overlap=(50, 50),
        )
        assert config.mpp == 0.5
        assert config.tile_size == (512, 512)
        assert config.tile_overlap == (50, 50)

    def test_invalid_tile_size(self):
        """Test that negative or zero tile_size raises ValueError."""
        with pytest.raises(ValueError, match="tile_size must be > 0"):
            TilingConfig(mpp=0.5, tile_size=(0, 512))

        with pytest.raises(ValueError, match="tile_size must be > 0"):
            TilingConfig(mpp=0.5, tile_size=(512, -1))

    def test_invalid_tile_overlap(self):
        """Test that negative tile_overlap raises ValueError."""
        with pytest.raises(ValueError, match="tile_overlap must be >= 0"):
            TilingConfig(mpp=0.5, tile_size=(512, 512), tile_overlap=(-1, 0))

        with pytest.raises(ValueError, match="tile_overlap must be >= 0"):
            TilingConfig(mpp=0.5, tile_size=(512, 512), tile_overlap=(0, -10))

    def test_invalid_mpp(self):
        """Test that zero or negative mpp raises ValueError."""
        with pytest.raises(ValueError, match="mpp must be > 0"):
            TilingConfig(mpp=0, tile_size=(512, 512))

        with pytest.raises(ValueError, match="mpp must be > 0"):
            TilingConfig(mpp=-0.5, tile_size=(512, 512))

    def test_valid_zero_overlap(self):
        """Test that zero tile_overlap is valid."""
        config = TilingConfig(
            mpp=0.5,
            tile_size=(512, 512),
            tile_overlap=(0, 0),
        )
        assert config.tile_overlap == (0, 0)

    def test_none_mpp(self):
        """Test that None mpp is valid."""
        config = TilingConfig(
            mpp=None,
            tile_size=(512, 512),
        )
        assert config.mpp is None


class TestConcatDataset:
    #  Concatenate two or more datasets and access elements via integer index
    def test_concatenate_and_access(self):
        class SimpleDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset1 = SimpleDataset([1, 2, 3])
        dataset2 = SimpleDataset([4, 5, 6])
        concat_dataset = ConcatDataset([dataset1, dataset2])

        assert concat_dataset[0] == 1
        assert concat_dataset[2] == 3
        assert concat_dataset[3] == 4
        assert concat_dataset[5] == 6
        assert concat_dataset[-6] == 1

        assert concat_dataset[0:3] == [1, 2, 3]
        assert concat_dataset[1:3] == [2, 3]
        assert concat_dataset[-1:-4:-1] == [6, 5, 4]

        with pytest.raises(ValueError) as exc_info:
            concat_dataset.index_to_dataset(-7)
        assert str(exc_info.value) == "Absolute value of index should not exceed dataset length"

    #  Initialize ConcatDataset with an empty list of datasets and catch assertion error
    def test_empty_dataset_initialization(self):
        with pytest.raises(AssertionError) as exc_info:
            _ = ConcatDataset([])
        assert str(exc_info.value) == "datasets should not be an empty iterable"

    def test_non_indexable_dataset(self):
        class SimpleDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

        dataset1 = SimpleDataset([1, 2, 3])
        with pytest.raises(ValueError) as exc_info:
            _ = ConcatDataset([dataset1])
        assert str(exc_info.value) == "ConcatDataset requires datasets to be indexable."


def test_slide_dataset_opens_its_slide_once_and_closes_it(monkeypatch):
    """A masked SlideDataset must open its slide once and release it on close().

    Regression: __init__ opened a fresh SlideImage for the mask computation (via the slide_image
    property, which reopens on every access) and never closed it, and the dataset exposed no close().
    For a remote backend each open anchors an event-loop thread, so iterating a cohort leaked file
    descriptors until it ran out. Counts opens vs closes through a mock backend.
    """
    from common import MockOpenSlideSlide, SlideConfig

    config = SlideConfig.from_parameters(
        filename="dummy.svs",
        num_levels=3,
        level_0_dimensions=(1000, 1000),
        mpp=(0.25, 0.25),
        objective_power=20,
        vendor="dummy",
    )

    opened: list[dlup.SlideImage] = []
    closed_ids: set[int] = set()
    real_close = dlup.SlideImage.close

    def fake_from_file_path(*_args, **_kwargs):
        slide = dlup.SlideImage(MockOpenSlideSlide.from_config(config))
        opened.append(slide)
        return slide

    def counting_close(self, *args, **kwargs):
        closed_ids.add(id(self))
        return real_close(self, *args, **kwargs)

    monkeypatch.setattr(dlup.SlideImage, "from_file_path", staticmethod(fake_from_file_path))
    monkeypatch.setattr(dlup.SlideImage, "close", counting_close)
    # Isolate the handle lifecycle from the mask maths — we only care that the slide gets closed.
    monkeypatch.setattr("dlup.data.dataset.compute_masked_indices", lambda *a, **k: np.array([0], dtype=np.int64))

    tiling_config = TilingConfig(mpp=1.0, tile_size=(32, 24), tile_overlap=(0, 0), tile_mode=TilingMode.skip)
    dataset = SlideDataset.from_standard_tiling(
        "dummy",
        tiling_config=tiling_config,
        mask_config=MaskConfig(mask=np.ones((8, 8), dtype=np.uint8), mask_threshold=0.5),
    )

    # Necessity: the dataset must expose close() at all.
    assert hasattr(dataset, "close"), "SlideDataset has no close(); the mask slide handle leaks"
    dataset.close()

    # Correctness: every slide opened for the dataset has been released.
    assert opened, "expected the dataset to open at least one slide"
    leaked = [s for s in opened if id(s) not in closed_ids]
    assert not leaked, f"{len(leaked)} of {len(opened)} SlideImage handle(s) left open after close()"

    dataset.close()  # idempotent

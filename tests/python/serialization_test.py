# Copyright 2024 Jonas Teuwen. All Rights Reserved.
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
import tempfile
import pathlib
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import dlup
from dlup import SlideImage
from dlup.data.dataset import SlideDataset, TilingConfig, Grid
from dlup.utils.backends import ImageBackend


class TestDatasetSerialization:
    @patch("dlup.SlideImage.from_file_path")
    def test_save_and_load(self, mock_from_file_path):
        # Setup mock slide image
        mock_slide_image = MagicMock(spec=SlideImage)
        mock_slide_image.mpp = 1.0
        mock_slide_image.get_scaling.return_value = 1.0
        mock_slide_image.get_scaled_slide_bounds.return_value = ((0, 0), (100, 100))
        mock_slide_image.size = (100, 100)

        # Mock view
        mock_view = MagicMock()
        mock_view.size = (100, 100)
        mock_slide_image.get_view_at_mpp.return_value = mock_view

        mock_from_file_path.return_value = mock_slide_image
        # Mock context manager behavior
        mock_from_file_path.return_value.__enter__.return_value = mock_slide_image

        # Create a dataset
        tiling_config = TilingConfig(mpp=1.0, tile_size=(10, 10), tile_overlap=(0, 0))
        dataset = SlideDataset.from_standard_tiling("dummy/path.svs", tiling_config=tiling_config)

        # Manually set some masked indices to simulate a mask
        original_indices = np.array([0, 2, 4], dtype=np.int64)
        dataset._masked_indices = original_indices

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = pathlib.Path(temp_dir) / "settings.dlup"
            dataset.save_settings(save_path)

            # Load the dataset
            # We provide the image path explicitly now.
            loaded_dataset = SlideDataset.from_settings_file(save_path, image_path="dummy/path.svs")

            # Verify masked indices
            np.testing.assert_array_equal(loaded_dataset._masked_indices, original_indices)

            # Verify grid properties
            assert loaded_dataset.tile_size == (10, 10)
            assert loaded_dataset.mpp == 1.0
            # Grid coordinates check
            assert len(loaded_dataset.grid) == len(dataset.grid)

            # Verify path
            assert str(loaded_dataset._path) == "dummy/path.svs"

    @patch("dlup.SlideImage.from_file_path")
    def test_filter(self, mock_from_file_path):
        # Setup mock slide image
        mock_slide_image = MagicMock(spec=SlideImage)
        mock_slide_image.mpp = 1.0
        mock_slide_image.get_scaling.return_value = 1.0
        mock_slide_image.get_scaled_slide_bounds.return_value = ((0, 0), (100, 100))
        mock_slide_image.size = (100, 100)

        # Mock view
        mock_view = MagicMock()
        mock_view.size = (100, 100)
        # Mock reading region to return a dummy image/info
        # TileSample has region_index.
        mock_slide_image.get_view_at_mpp.return_value = mock_view

        mock_from_file_path.return_value = mock_slide_image
        mock_from_file_path.return_value.__enter__.return_value = mock_slide_image

        tiling_config = TilingConfig(mpp=1.0, tile_size=(10, 10))
        dataset = SlideDataset.from_standard_tiling("dummy.svs", tiling_config=tiling_config)

        # Total tiles = 10x10 grid = 100 tiles.
        assert len(dataset) == 100

        # Filter: keep only even region indices
        filtered_dataset = dataset.filter(lambda sample: sample.region_index % 2 == 0)

        assert len(filtered_dataset) == 50
        assert filtered_dataset._masked_indices[0] == 0
        assert filtered_dataset._masked_indices[1] == 2

        # Verify original dataset is untouched
        assert dataset._masked_indices is None or len(dataset) == 100

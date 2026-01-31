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
import functools

import numpy as np
import pytest
from dlup import SlideAnnotations, SlideImage
from dlup._exceptions import DlupError
from dlup.background import compute_masked_indices
from dlup.data.dataset import _coords_to_region
from dlup.geometry import Box
from dlup.tiling import Grid
from dlup.tools import ConcatSequences, MapSequence
from common import MockOpenSlideSlide, SlideConfig


class TestComputeMaskedIndices:
    def test_threshold_none(self, dlup_wsi):
        background_mask = np.zeros((10, 10), dtype=np.int64)

        regions = np.asarray(
            [[0, 0, 100, 100, 0.5], [100, 100, 200, 200, 0.5], [200, 200, 300, 300, 0.5]], dtype=np.float64
        )
        output = compute_masked_indices(dlup_wsi, background_mask, regions, threshold=None)
        np.testing.assert_equal(output, np.array([0, 1, 2], dtype=np.int64))

    @pytest.mark.parametrize(
        "threshold",
        [
            0.0,
            0.4,
            0.5,
        ],
    )
    def test_ndarray(self, dlup_wsi, threshold):
        background_mask = np.zeros((100, 100), dtype=bool)

        background_mask[14:20, 10:20] = True
        background_mask[85:100, 50:80] = True

        regions, grid = self._compute_grid_elements(dlup_wsi)
        masked_indices = compute_masked_indices(dlup_wsi, background_mask, regions, threshold=threshold)
        grid_elems = [grid[index] for index in masked_indices]

        if threshold in [0.0, 0.4]:
            assert len(grid_elems) == 7
        else:
            assert len(grid_elems) == 4

        for elem in grid_elems:
            sliced_mask = background_mask[elem[1] // 10 : elem[1] // 10 + 10, elem[0] // 10 : elem[0] // 10 + 10]
            assert sliced_mask.mean() >= threshold

    @pytest.mark.parametrize("threshold", [0.0, 0.4, 0.5, 1.0])
    def test_wsiannotations(self, dlup_wsi, threshold):
        # TODO: Make test for different scalings

        # Let's make a shapely polygon thats equal to
        # background_mask[14:20, 10:20] = True
        # background_mask[85:100, 50:80] = True
        polygon0 = Box((10, 14), (20, 20), label="bg").as_polygon()
        polygon1 = Box((50, 85), (80, 100), label="bg").as_polygon()
        annotations = SlideAnnotations()
        annotations.add_polygon(polygon0)
        annotations.add_polygon(polygon1)
        annotations.rebuild_rtree()
        regions, grid = self._compute_grid_elements(dlup_wsi)

        masked_indices = compute_masked_indices(dlup_wsi, annotations, regions, threshold=threshold)
        grid_elems = [grid[index] for index in masked_indices]

        # if threshold in [0.0, 0.4]:
        #     assert len(grid_elems) == 7
        # elif threshold == 0.5:
        #     assert len(grid_elems) == 4
        # else:
        #     assert len(grid_elems) == 3

        for grid_elem in grid_elems:
            region = annotations.read_region(grid_elem, 1.0, (100, 100))
            assert sum(_.area for _ in region) >= threshold * 100 * 100

    @pytest.mark.parametrize("threshold", [0.0, 0.4, 0.5, 1.0])
    def test_slide_image(self, dlup_wsi, threshold):
        # TODO: Make test for different scalings
        background_mask = np.zeros((1000, 1000), dtype=np.uint8)

        background_mask[140:200, 100:200] = 1
        background_mask[850:1000, 500:800] = 1

        config = SlideConfig.from_parameters(
            filename="dummy1.svs",
            num_levels=1,
            level_0_dimensions=(1000, 1000),
            mpp=(dlup_wsi.mpp, dlup_wsi.mpp),
            objective_power=20,
            vendor="dummy",
            image=background_mask,
        )
        mock_backend = MockOpenSlideSlide.from_config(config)
        mask_image = SlideImage(mock_backend, interpolator="NEAREST")
        regions, grid = self._compute_grid_elements(dlup_wsi)
        masked_indices = compute_masked_indices(dlup_wsi, mask_image, regions, threshold=threshold)

        grid_elems = [grid[index] for index in masked_indices]

        if threshold in [0.0, 0.4]:
            assert len(grid_elems) == 7
        elif threshold == 0.5:
            assert len(grid_elems) == 4
        else:
            assert len(grid_elems) == 3

        for grid_elem in grid_elems:
            region = mask_image.read_region(grid_elem, 1.0, (100, 100)).to_numpy()
            assert region.sum() >= threshold * 100 * 100

    def test_unknown_type(self, dlup_wsi):
        with pytest.raises(DlupError, match=f"Unknown background mask type. Got {type([])}"):
            compute_masked_indices(dlup_wsi, [0, 1], [], threshold=0)

    def _compute_grid_elements(self, dlup_wsi):
        tile_size = (100, 100)
        tile_overlap = (0, 0)
        slide_level_size = dlup_wsi.get_scaled_size(1.0, limit_bounds=False)

        grid = Grid.from_tiling(
            offset=(0, 0),
            size=slide_level_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            mode="overflow",
            order="F",
        )

        regions = [MapSequence(functools.partial(_coords_to_region, tile_size, dlup_wsi.mpp), grid)]
        _regions = ConcatSequences(regions)
        return _regions, grid

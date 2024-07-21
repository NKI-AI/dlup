import functools

import numpy as np
import pytest

from dlup.background import compute_masked_indices
from dlup.data.dataset import _coords_to_region
from dlup.tiling import Grid
from dlup.tools import ConcatSequences, MapSequence


class TestComputeMaskedIndices:
    def test_threshold_none(self, dlup_wsi):
        background_mask = np.zeros((10, 10), dtype=np.int64)

        regions = np.asarray(
            [[0, 0, 100, 100, 0.5], [100, 100, 200, 200, 0.5], [200, 200, 300, 300, 0.5]], dtype=np.float64
        )
        output = compute_masked_indices(dlup_wsi, background_mask, regions, threshold=None)
        np.testing.assert_equal(output, np.array([0, 1, 2], dtype=np.int64))

    @pytest.mark.parametrize("threshold", [0.0, 0.4, 0.5])
    def test_ndarray(self, dlup_wsi, threshold):
        background_mask = np.zeros((100, 100), dtype=bool)

        background_mask[14:20, 10:20] = True
        background_mask[85:100, 50:80] = True

        tile_size = (100, 100)
        tile_overlap = (0, 0)

        slide_level_size = dlup_wsi.get_scaled_size(1.0, limit_bounds=False)

        grids = []
        grid = Grid.from_tiling(
            offset=(0, 0),
            size=slide_level_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            mode="overflow",
            order="F",
        )

        grids.append((grid, tile_size, dlup_wsi.mpp))

        regions = []
        for grid, tile_size, mpp in grids:
            regions.append(MapSequence(functools.partial(_coords_to_region, tile_size, mpp), grid))  # type: ignore
        _regions = ConcatSequences(regions)
        masked_indices = compute_masked_indices(dlup_wsi, background_mask, _regions, threshold=threshold)
        grid_elems = [grid[index] for index in masked_indices]

        if threshold in [0.0, 0.4]:
            assert len(grid_elems) == 7
        else:
            assert len(grid_elems) == 4

        for elem in grid_elems:
            sliced_mask = background_mask[elem[1] // 10 : elem[1] // 10 + 10, elem[0] // 10 : elem[0] // 10 + 10]
            assert sliced_mask.mean() >= threshold

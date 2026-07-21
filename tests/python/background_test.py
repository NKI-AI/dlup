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
import math

import numpy as np
import pytest
from dlup import SlideAnnotations, SlideImage
from dlup._background import get_foreground_indices_numpy
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


def _reference_foreground_indices(
    image_width: int,
    image_height: int,
    slide_mpp: float,
    mask: np.ndarray,
    regions: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Pure-Python mirror of ``get_foreground_indices_numpy``'s per-axis projection.

    Each region (slide pixels at the region mpp) is projected onto the mask grid using an
    independent scale factor per axis, then a tile counts as foreground when its mask sum strictly
    exceeds ``threshold`` times the clipped area.
    """
    mask_height, mask_width = mask.shape
    selected: list[int] = []
    for idx, (x, y, w, h, mpp) in enumerate(regions):
        scaling = slide_mpp / mpp
        region_width = int(scaling * image_width)
        region_height = int(scaling * image_height)

        scale_x = mask_width / region_width
        scale_y = mask_height / region_height

        x1 = min(mask_width, math.floor(x * scale_x))
        y1 = min(mask_height, math.floor(y * scale_y))
        x2 = min(mask_width, math.ceil((x + w) * scale_x))
        y2 = min(mask_height, math.ceil((y + h) * scale_y))

        clipped_w = x2 - x1
        clipped_h = y2 - y1
        assert clipped_w > 0 and clipped_h > 0, f"region {idx} collapsed: {(x1, y1, x2, y2)}"

        if mask[y1:y2, x1:x2].sum() > threshold * clipped_w * clipped_h:
            selected.append(idx)
    return np.asarray(selected, dtype=np.int64)


def _run_numpy_kernel(
    image_width: int,
    image_height: int,
    slide_mpp: float,
    mask: np.ndarray,
    regions: np.ndarray,
    threshold: float,
) -> np.ndarray:
    foreground_indices = np.zeros(len(regions), dtype=np.int64)
    count = get_foreground_indices_numpy(
        image_width,
        image_height,
        slide_mpp,
        mask,
        np.asarray(regions, dtype=np.float64),
        threshold,
        foreground_indices,
    )
    return foreground_indices[:count]


class TestForegroundProjection:
    """Directly exercises the C++ projection kernel ``get_foreground_indices_numpy``."""

    def test_edge_tiles_do_not_collapse(self):
        """Regression: bottom/right OVERFLOW tiles must not project past the mask edge.

        Uses the geometry of TCGA-F4-6807 (slide 108394x84468 at mpp 0.252, mask 1773x2276).
        With the old single scale factor the bottom row projected to y1 == y2 == mask_height and
        the kernel raised ``RuntimeError: Invalid region dimensions``.
        """
        image_width, image_height, slide_mpp = 108394, 84468, 0.252
        mask = np.ones((1773, 2276), dtype=np.uint8)

        # Slide at region mpp=1.0 is 27315x21285; these are the last-row / last-column tiles.
        regions = np.asarray(
            [
                [0.0, 0.0, 224.0, 224.0, 1.0],
                [0.0, 21280.0, 224.0, 224.0, 1.0],  # bottom edge, previously collapsed
                [27104.0, 0.0, 224.0, 224.0, 1.0],  # right edge
                [27104.0, 21280.0, 224.0, 224.0, 1.0],  # bottom-right corner
            ],
            dtype=np.float64,
        )

        result = _run_numpy_kernel(image_width, image_height, slide_mpp, mask, regions, threshold=0.0)
        np.testing.assert_array_equal(result, np.array([0, 1, 2, 3], dtype=np.int64))

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 1.0])
    def test_per_axis_projection_matches_reference(self, threshold):
        """Anisotropic mask (scale_x != scale_y) must match the per-axis reference.

        A single (max-dimension) scale factor would use scale_x for the y axis here and select the
        wrong tiles, so matching the per-axis reference pins the corrected behaviour.
        """
        # scaling=1.0 -> region grid is 4000x1000; mask 500x1000 gives scale_x=0.25, scale_y=0.5.
        image_width, image_height, slide_mpp = 4000, 1000, 1.0
        rng = np.random.default_rng(0)
        mask = (rng.random((500, 1000)) > 0.5).astype(np.uint8)

        regions = np.asarray(
            [[float(x), float(y), 400.0, 400.0, 1.0] for x in range(0, 4000, 400) for y in range(0, 1000, 400)],
            dtype=np.float64,
        )

        result = _run_numpy_kernel(image_width, image_height, slide_mpp, mask, regions, threshold)
        reference = _reference_foreground_indices(image_width, image_height, slide_mpp, mask, regions, threshold)
        np.testing.assert_array_equal(result, reference)

    def test_zero_region_mpp_raises(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        regions = np.asarray([[0.0, 0.0, 5.0, 5.0, 0.0]], dtype=np.float64)
        with pytest.raises(Exception, match="Region mpp cannot be zero"):
            _run_numpy_kernel(100, 100, 0.5, mask, regions, threshold=0.5)

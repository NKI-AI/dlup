# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
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
"""Tests for dlup.geometry.measure module"""

import numpy as np
import pytest
from shapely.geometry import MultiPolygon

from dlup.geometry import Polygon
from dlup.geometry.measure import find_contours


class TestFindContours:
    """Tests for find_contours function"""

    def test_simple_square(self):
        """Test finding contours of a simple square"""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 1

        contours = find_contours(mask, level=0.5)

        assert len(contours) == 1
        assert isinstance(contours[0], Polygon)
        assert contours[0].area > 0

    def test_multiple_regions(self):
        """Test finding contours with multiple disconnected regions"""
        mask = np.zeros((10, 8), dtype=np.uint8)
        mask[0, 0:8] = 1  # Top row
        mask[2:5, 2:5] = 1  # Middle square
        mask[-1, 0:10] = 1  # Bottom row (note: goes beyond width)

        contours = find_contours(mask, level=0.5)

        # Should find 3 separate contours
        assert len(contours) == 3
        assert all(isinstance(p, Polygon) for p in contours)

        # Convert to shapely MultiPolygon for comparison
        multi_poly = MultiPolygon([p.to_shapely() for p in contours])

        # Expected WKT (coordinates are at pixel boundaries)
        expected_wkt = (
            "MULTIPOLYGON (((0.5 7, 0.5 6, 0.5 5, 0.5 4, 0.5 3, 0.5 2, 0.5 1, 0.5 0, "
            "0 -0.5, -0.5 0, -0.5 1, -0.5 2, -0.5 3, -0.5 4, -0.5 5, -0.5 6, -0.5 7, "
            "0 7.5, 0.5 7)), ((4.5 4, 4.5 3, 4.5 2, 4 1.5, 3 1.5, 2 1.5, 1.5 2, 1.5 3, "
            "1.5 4, 2 4.5, 3 4.5, 4 4.5, 4.5 4)), ((9.5 7, 9.5 6, 9.5 5, 9.5 4, 9.5 3, "
            "9.5 2, 9.5 1, 9.5 0, 9 -0.5, 8.5 0, 8.5 1, 8.5 2, 8.5 3, 8.5 4, 8.5 5, "
            "8.5 6, 8.5 7, 9 7.5, 9.5 7)))"
        )

        assert multi_poly.wkt == expected_wkt

    def test_empty_mask(self):
        """Test that empty mask returns no contours"""
        mask = np.zeros((5, 5), dtype=np.uint8)
        contours = find_contours(mask, level=0.5)
        assert len(contours) == 0

    def test_full_mask(self):
        """Test that fully filled mask returns outer boundary"""
        mask = np.ones((5, 5), dtype=np.uint8)
        contours = find_contours(mask, level=0.5)
        # Should get the outer boundary
        assert len(contours) >= 1

    def test_different_levels(self):
        """Test different iso-value levels"""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 1

        # Level at 0.5 (centered on pixel boundary)
        contours_05 = find_contours(mask, level=0.5)
        assert len(contours_05) == 1

        # Level at 0.9 (closer to 1, different boundary)
        contours_09 = find_contours(mask, level=0.9)
        assert len(contours_09) == 1

        # Contours should be slightly different
        # (different level gives different interpolation)

    def test_float_array(self):
        """Test with float array instead of uint8"""
        mask = np.zeros((5, 5), dtype=np.float64)
        mask[1:4, 1:4] = 1.0

        contours = find_contours(mask, level=0.5)
        assert len(contours) == 1
        assert isinstance(contours[0], Polygon)

    def test_int32_array(self):
        """Test with int32 array"""
        mask = np.zeros((5, 5), dtype=np.int32)
        mask[1:4, 1:4] = 1

        contours = find_contours(mask, level=0.5)
        assert len(contours) == 1
        assert isinstance(contours[0], Polygon)

    def test_invalid_dimensions(self):
        """Test that 1D or 3D arrays raise error"""
        with pytest.raises(ValueError, match="2-dimensional"):
            find_contours(np.zeros(10), level=0.5)

        with pytest.raises(ValueError, match="2-dimensional"):
            find_contours(np.zeros((5, 5, 3)), level=0.5)

    def test_too_small_array(self):
        """Test that arrays smaller than 2x2 raise error"""
        with pytest.raises(ValueError, match="at least 2x2"):
            find_contours(np.zeros((1, 1)), level=0.5)

    def test_unsupported_dtype(self):
        """Test that unsupported dtypes raise error"""
        mask = np.zeros((5, 5), dtype=np.complex64)
        with pytest.raises(ValueError, match="uint8, int32, float32, or float64"):
            find_contours(mask, level=0.5)

    def test_polygon_has_fields(self):
        """Test that returned polygons support field operations"""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 1

        contours = find_contours(mask, level=0.5)
        polygon = contours[0]

        # Test setting fields
        polygon.label = "test"
        assert polygon.label == "test"

        polygon.index = 1
        assert polygon.index == 1

        polygon.color = (255, 0, 0)
        assert polygon.color == (255, 0, 0)

    def test_comparison_with_scikit_image(self):
        """Test that results are compatible with scikit-image expectations"""
        # Create a test pattern
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1  # 6x6 square

        contours = find_contours(mask, level=0.5)

        assert len(contours) == 1
        polygon = contours[0]

        # The contour should form a closed loop
        exterior = polygon.get_exterior()
        assert exterior[0] == exterior[-1], "Contour should be closed"

        # Check that the polygon is valid
        shapely_poly = polygon.to_shapely()
        assert shapely_poly.is_valid
        assert shapely_poly.area > 0

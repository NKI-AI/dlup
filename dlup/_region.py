# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
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
"""Defines the RegionView interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import fim
import numpy as np
from dlup._types import GenericFloatArray, GenericIntArray


class BoundaryMode(str, Enum):
    """Define the policy to sample outside the region."""

    crop = "crop"
    zero = "zero"


class RegionView(ABC):
    """A generic image object from which you can extract a region.

    A unit 'U' is assumed to be consistent across this interface.
    Could be for instance pixels.

    TODO(lromor): Add features like cyclic boundary conditions
    or zero padding, or "hard" walls.
    TODO(lromor): Add another feature to return a subregion. The logic
    could stay in the abstract class. This is especially useful to tile
    subregions instead of a whole level.
    """

    def __init__(self, boundary_mode: BoundaryMode | None = None):
        self.boundary_mode = boundary_mode

    @property
    @abstractmethod
    def size(self) -> tuple[int, ...]:
        """Returns size of the region in U units."""

    def read_region(self, location: GenericFloatArray, size: GenericIntArray) -> fim.Image:
        """Returns the requested region as a fim image."""
        location = np.asarray(location)
        size = np.asarray(size)

        # If no boundary is specified, sampling outside the region
        # is undefined behavior (result depends on the _read_region_impl).
        if self.boundary_mode is None:
            return self._read_region_impl(location, size)

        # This is slightly tricky as it can influence the mpp slightly
        offset = -np.clip(location, None, 0)

        clipped_region_size = np.clip(location + size, np.zeros_like(size), self.size) - location - offset
        clipped_region_size = clipped_region_size.astype(int)
        region: Any = self._read_region_impl(location + offset, clipped_region_size)

        if self.boundary_mode == BoundaryMode.zero:
            if np.any(size != clipped_region_size) or np.any(location < 0):
                # Create black canvas with fim
                new_region: Any = fim.Image.black(size[0], size[1])
                coordinates = tuple(np.floor(offset).astype(int))
                insert_x = max(0, coordinates[0])
                insert_y = max(0, coordinates[1])

                # Get region dimensions
                region_dims = region.dimensions
                region_width = min(region_dims[0], size[0] - insert_x)
                region_height = min(region_dims[1], size[1] - insert_y)

                region = region.crop((0, 0), (region_width, region_height))

                # Use paste to insert the region into the black canvas
                new_region = new_region.paste(region, insert_x, insert_y)

                return new_region

        return region

    @abstractmethod
    def _read_region_impl(self, location: GenericFloatArray, size: GenericIntArray) -> fim.Image:
        """Define a method to return a fim image containing the region."""

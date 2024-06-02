# Copyright (c) dlup contributors
"""Defines the RegionView interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pyvips

from dlup.types import GenericFloatArray, GenericIntArray


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

    def read_region(self, location: GenericFloatArray, size: GenericIntArray) -> pyvips.Image:
        """Returns the requested region as a vips image."""
        location = np.asarray(location)
        size = np.asarray(size)

        # If no boundary is specified, sampling outside the region
        # is undefined behavior (result depends on the _read_region_impl).
        # TODO: maybe always raise exception?
        if self.boundary_mode is None:
            return self._read_region_impl(location, size)

        # This is slightly tricky as it can influence the mpp slightly
        offset = -np.clip(location, None, 0)

        clipped_region_size = np.clip(location + size, np.zeros_like(size), self.size) - location - offset
        clipped_region_size = clipped_region_size.astype(int)
        region = self._read_region_impl(location + offset, clipped_region_size)

        if self.boundary_mode == BoundaryMode.zero:
            if np.any(size != clipped_region_size) or np.any(location < 0):
                new_region = pyvips.Image.black(size[0], size[1])
                coordinates = tuple(np.floor(offset).astype(int))
                insert_x = max(0, coordinates[0])
                insert_y = max(0, coordinates[1])
                region_width = min(region.width, size[0] - insert_x)
                region_height = min(region.height, size[1] - insert_y)

                region = region.crop(0, 0, region_width, region_height)

                new_region = new_region.insert(region, insert_x, insert_y, expand=True)

                return new_region

        return region

    @abstractmethod
    def _read_region_impl(self, location: GenericFloatArray, size: GenericIntArray) -> pyvips.Image:
        """Define a method to return a pyvips image containing the region."""

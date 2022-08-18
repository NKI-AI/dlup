# coding=utf-8
"""Defines the RegionView interface."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import PIL.Image

_GenericFloatArray = Union[np.ndarray, Iterable[float]]
_GenericIntArray = Union[np.ndarray, Iterable[int]]


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

    def __init__(self, boundary_mode: Optional[BoundaryMode] = None):
        self.boundary_mode = boundary_mode

    @property
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        """Returns size of the region in U units."""
        pass

    def read_region(self, location: _GenericFloatArray, size: _GenericIntArray) -> PIL.Image:
        """Returns the requested region as a numpy array."""
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
                new_region = PIL.Image.new(str(region.mode), tuple(size))
                # Now we need to paste the region into the new region.
                # We do some rounding to int.
                new_region.paste(region, tuple(np.floor(offset).astype(int)))
                return new_region

        return region

    @abstractmethod
    def _read_region_impl(self, location: _GenericFloatArray, size: _GenericIntArray) -> PIL.Image:
        """Define a method to return an array containing the region."""
        pass

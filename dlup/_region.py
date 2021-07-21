# coding=utf-8
"""Defines the RegionView interface."""
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np

_GenericFloatArray = Union[np.ndarray, Iterable[float]]
_GenericIntArray = Union[np.ndarray, Iterable[int]]


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

    @property
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        """Returns size of the region in U units."""
        pass

    def read_region(self, location: _GenericFloatArray, size: _GenericIntArray, crop=True) -> np.ndarray:
        """Returns the requested region as a numpy array."""
        location = np.asarray(location)
        size = np.asarray(size)

        clipped_region_size = (
            np.clip(location + size, np.zeros_like(size), self.size) - location
        )
        clipped_region_size = clipped_region_size.astype(int)
        region = self._read_region_impl(location, clipped_region_size)

        if not crop:
            padding = np.zeros((len(region.shape), 2), dtype=int)

            # This flip is justified as PIL outputs arrays with axes in reversed order
            # Extracting a box of size (width, height) results in an array
            # of shape (height, width, channels)
            padding[:-1, 1] = size - clipped_region_size
            values = np.zeros_like(padding)
            region = np.pad(region, padding, "constant", constant_values=values)

        return region

    @abstractmethod
    def _read_region_impl(self, location: _GenericFloatArray, size: _GenericIntArray) -> np.ndarray:
        pass

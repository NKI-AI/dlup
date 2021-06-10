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
    def size(self):
        """Returns size of the region in U units."""
        pass

    @abstractmethod
    def read_region(self, location: _GenericFloatArray, size: _GenericIntArray) -> np.ndarray:
        pass

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
from __future__ import annotations

import collections
import functools
from enum import Enum
from typing import Any, Iterator, Literal, Sequence, Union, overload

import numpy as np
import numpy.typing as npt

_GenericNumber = Union[int, float]
_GenericNumberArray = Union[npt.NDArray[np.int_ | np.float64], Sequence[_GenericNumber]]


class TilingMode(str, Enum):
    """Type of tiling.

    Skip will skip the border tiles if they don't fit the region.
    Overflow counts as last tile even if it's overflowing.
    """

    skip = "skip"
    overflow = "overflow"


class GridOrder(str, Enum):
    """Order of the grid.

    Fortran is column-major order, and C is in row-major order, that is, the tiles are created in a column-by-column
    fashion or in a row by row fashion.
    """

    C = "C"
    F = "F"


def _flattened_array(a: _GenericNumberArray | _GenericNumber) -> npt.NDArray[np.float64 | np.int_]:
    """Converts any generic array in a flattened numpy array."""
    return np.asarray(a).flatten()


def indexed_ndmesh(bases: Sequence[_GenericNumberArray], indexing: Literal["xy", "ij"] = "ij") -> npt.NDArray[np.int_]:
    """Converts a list of arrays into an n-dimensional indexed mesh.

    Examples
    --------

    .. code-block:: python

        import dlup
        mesh = dlup.tiling.indexed_ndmesh(((1, 2, 3), (4, 5, 6)))
        assert mesh[0, 0] == (1, 4)
        assert mesh[0, 1] == (1, 5)
    """
    return np.ascontiguousarray(np.stack(tuple(reversed(np.meshgrid(*reversed(bases), indexing=indexing)))).T)


def tiles_grid_coordinates(
    size: _GenericNumberArray,
    tile_size: _GenericNumberArray,
    tile_overlap: _GenericNumberArray | _GenericNumber = 0,
    mode: TilingMode | str = TilingMode.skip,
) -> list[npt.NDArray[np.int_ | np.float64]]:
    """Generate a list of coordinates for each dimension representing a tile location.

    The first tile has the corner located at (0, 0).
    """
    size = _flattened_array(size)
    tile_size = _flattened_array(tile_size)
    tile_overlap = _flattened_array(tile_overlap)

    if not size.shape == tile_size.shape == tile_overlap.shape:
        raise ValueError("size, tile_size and tile_overlap should have the same dimensions.")

    if (size <= 0).any():
        raise ValueError("size should always be greater than zero.")

    if (tile_size <= 0).any():
        raise ValueError("tile size should always be greater than zero.")

    # Let's force it to a valid value.
    tile_overlap = np.remainder(tile_overlap, np.minimum(tile_size, size), casting="safe")

    # Get the striding
    stride = tile_size - tile_overlap

    # Same thing as computing the output shape of a convolution with padding zero and
    # specified stride.
    num_tiles = (size - tile_size) / stride + 1

    if mode == TilingMode.skip:
        num_tiles = np.floor(num_tiles).astype(int)
        overflow = np.zeros_like(size)
    else:
        num_tiles = np.ceil(num_tiles).astype(int)
        tiled_size = (num_tiles - 1) * stride + tile_size
        overflow = np.asarray(tiled_size).astype(float) - np.asarray(size).astype(float)

    # Let's create our indices list
    coordinates: list[npt.NDArray[Any]] = []
    for n, dstride, dtile_size, doverflow, dsize in zip(num_tiles, stride, tile_size, overflow, size):
        tiles_locations = np.arange(int(n)) * dstride
        coordinates.append(tiles_locations)

    return coordinates


class Grid(collections.abc.Sequence[npt.NDArray[np.int_ | np.float64]]):
    """Facilitates the access to the coordinates of an n-dimensional grid."""

    def __init__(self, coordinates: list[npt.NDArray[np.int_ | np.float64]], order: str | GridOrder = GridOrder.C):
        """Initialize a lattice given a set of basis vectors."""
        self.coordinates = coordinates
        self._order = order if isinstance(order, GridOrder) else GridOrder(order)

    @classmethod
    def from_tiling(
        cls,
        offset: _GenericNumberArray,
        size: _GenericNumberArray,
        tile_size: _GenericNumberArray,
        tile_overlap: _GenericNumberArray | _GenericNumber = 0,
        mode: TilingMode | str = TilingMode.skip,
        order: GridOrder | str = GridOrder.C,
    ) -> "Grid":
        """Generate a grid from a set of tiling parameters."""
        coordinates = tiles_grid_coordinates(size, tile_size, tile_overlap, mode)
        coordinates = [np.asarray(c + o, dtype=c.dtype) for c, o in zip(coordinates, offset)]
        return cls(coordinates, order=order)

    @classmethod
    def from_points(cls, points: list[Sequence[_GenericNumber]], order: GridOrder | str = GridOrder.C) -> "Grid":
        """Construct a grid from a list of irregular points."""
        if not points:
            return cls([], order)

        # Treat points as a single set of coordinates
        coordinates = np.array(points)
        return cls([coordinates], order)

    @property
    def order(self) -> GridOrder:
        """Return the order of the grid."""
        return self._order

    @property
    def size(self) -> tuple[int, ...]:
        """Return the dimensions of the grid."""
        if len(self.coordinates) == 1:  # Irregular grid (from_points)
            return (len(self.coordinates[0]), len(self.coordinates[0][0]))
        return tuple(len(c) for c in self.coordinates)

    @overload
    def __getitem__(self, key: int) -> npt.NDArray[np.int_ | np.float64]: ...

    @overload
    def __getitem__(self, key: slice) -> list[npt.NDArray[np.int_ | np.float64]]: ...

    def __getitem__(
        self, key: Union[int, slice]
    ) -> npt.NDArray[np.int_ | np.float64] | list[npt.NDArray[np.int_ | np.float64]]:
        if len(self.coordinates) == 1:  # Irregular grid
            return self.coordinates[0][key]

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step or 1)]

        index = np.unravel_index(key, self.size, order="F" if self.order == "C" else "C")
        return np.array([c[i] for c, i in zip(self.coordinates, index)])

    def __len__(self) -> int:
        """Return the total number of points in the grid."""
        if len(self.coordinates) == 1:  # Irregular grid
            return len(self.coordinates[0])
        return functools.reduce(lambda value, size: value * size, self.size, 1)

    def __iter__(self) -> Iterator[npt.NDArray[np.int_ | np.float64]]:
        """Iterate through every tile or point."""
        for i in range(len(self)):
            yield self[i]

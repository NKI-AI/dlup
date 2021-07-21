# coding=utf-8
# Copyright (c) dlup contributors

import functools
from enum import Enum
from typing import List, Sequence, Tuple, Type, TypeVar, Union, Generic

import numpy as np

from ._region import RegionView

_GenericNumber = Union[int, float]
_GenericNumberArray = Union[np.ndarray, Sequence[_GenericNumber]]


class TilingMode(str, Enum):
    """Type of tiling.

    Skip will skip the border tiles if they don't fit the region.
    Overflow counts as last tile even if it's overflowing.
    Fit will change the overlapping region between tiles to make them fit the region.
    The grid will then become non-uniform in case of integral values.
    """

    skip = "skip"
    overflow = "overflow"
    fit = "fit"


def _flattened_array(a: Union[_GenericNumberArray, _GenericNumber]) -> np.ndarray:
    """Converts any generic array in a flattened numpy array."""
    return np.asarray(a).flatten()


def indexed_ndmesh(bases: Sequence[_GenericNumberArray], indexing="ij") -> np.ndarray:
    """Converts a list of arrays into an n-dimensional indexed mesh.

    Example
    -------

    .. code-block:: python

        import dlup
        mesh = dlup.tiling.indexed_ndmesh(((1, 2, 3), (4, 5, 6)))
        assert mesh[0, 0] == (1, 4)
        assert mesh[0, 1] == (1, 5)
    """
    return np.ascontiguousarray(np.stack(tuple(reversed(np.meshgrid(*reversed(bases), indexing=indexing)))).T)


def tiling_lattice_basis_vectors(
    size: _GenericNumberArray,
    tile_size: _GenericNumberArray,
    tile_overlap: Union[_GenericNumberArray, _GenericNumber] = 0,
    mode: TilingMode = TilingMode.skip,
) -> List[np.ndarray]:
    """Generate a list of coordinates for each dimension representing a tile location.

    The first tile has the corner located at (0, 0).
    """
    size = _flattened_array(size)
    tile_size = _flattened_array(tile_size)
    tile_overlap = _flattened_array(tile_overlap)

    if not (size.shape == tile_size.shape == tile_overlap.shape):
        raise ValueError("size, tile_size and tile_overlap " "should have the same dimensions.")

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
        overflow = tiled_size - size

    # Let's create our indices list
    coordinates = []
    for n, dstride, dtile_size, doverflow, dsize in zip(num_tiles, stride, tile_size, overflow, size):
        tiles_locations = np.arange(n) * dstride

        if mode == TilingMode.fit:
            if n < 2:
                coordinates.append(np.array([]))
                continue

            # The location of the last tile
            # should stay fixed at the end
            tiles_locations[-1] = dsize - dtile_size
            distribute = doverflow / (n - 1)
            tiles_locations = tiles_locations.astype(float)
            tiles_locations[1:-1] -= distribute * (np.arange(n - 2) + 1)

        coordinates.append(tiles_locations)
    return coordinates


class Lattice():
    """Facilitates the access to tiles of a region view."""

    def __init__(self, basis_vectors: List[np.ndarray]):
        self._basis_vectors = basis_vectors
        self._size = tuple(len(x) for x in self._basis_vectors)

    @property
    def size(self):
        return self._size

    @property
    def basis_vectors(self):
        return self._basis_vectors

    def get_coordinate(self, i):
        index = np.unravel_index(i, self._size)
        return np.array(list(c[i] for c, i in zip(self._tiling_bases, index)))

    def coordinates(self):
        for i in range(len(self)):
            yield self.get_coordinate(i)

    def __getitem__(self, i):
        return self.get_coordinate(i)

    def __len__(self):
        return functools.reduce(lambda value, size: value * size, self.size, 1)

    def __iter__(self):
        """Iterate through every tile."""
        return self.coordinates()

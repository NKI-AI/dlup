# coding=utf-8
# Copyright (c) DLUP Contributors

from enum import Enum
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union, List
import numpy as np

from ._region import RegionView


_GenericNumber = Union[int, float]
_GenericNumberArray = Union[np.ndarray, Iterable[_GenericNumber]]


class TilingMode(str, Enum):
    """Type of tiling.

    Skip will skip the border tiles if they
    don't fit the region. Overflow counts as last tile
    even if it's overflowing. Fit will change the
    overlapping region between tiles to make them fit
    the region. The grid will then become non-uniform
    in case of integral values.
    """
    skip = 'skip'
    overflow = 'overflow'
    fit = 'fit'


def _flattened_array(a: _GenericNumberArray) -> np.ndarray:
    """Converts any generic array in a flattened numpy array."""
    return np.asarray(a).flatten()


def indexed_ndmesh(basis: Iterable[_GenericNumberArray]):
    """Converts a list of arrays into an n-dimensional indexed mesh.

    For instance:
    ```
    mesh = indexed_ndmesh(((1, 2, 3), (4, 5, 6)))
    assert mesh[0, 0] == (1, 4)
    assert mesh[0, 1] == (1, 5)
    ```
    """
    return np.stack(tuple(reversed(np.meshgrid(*reversed(basis), indexing='ij')))).T


def span_tiling_bases(size: _GenericNumberArray, tile_size: _GenericNumberArray,
                      tile_overlap : _GenericNumberArray = 0,
                      mode: TilingMode = TilingMode.skip) -> List[np.ndarray]:
    """Generate a list of coordinates for each dimension representing a tile location.

    The first tile has the corner located at (0, 0).
    """
    size = _flattened_array(size)
    tile_size = _flattened_array(tile_size)
    tile_overlap = _flattened_array(tile_overlap)

    if not (size.shape == tile_size.shape == tile_overlap.shape):
        raise ValueError('Size, tile_size and tile_overlap '
                         'should have the same dimensions.')

    if (size <= 0).any():
        raise ValueError('Size should always be greater than zero.')

    if (tile_size <= 0).any():
        raise ValueError('Tile size should always be greater than zero.')

    # Let's force it to a valid value.
    tile_overlap = np.remainder(tile_overlap, np.minimum(tile_size, size), casting='safe')

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
    for n, dstride, dsize in zip(num_tiles, stride, size):
        tiles_locations = np.arange(n) * dstride

        if mode == TilingMode.fit and overflow > 0:
            if n < 2:
                coordinates.append(np.array([]))
                continue

            # The location of the last tile
            # should stay fixed at the end
            tiles_locations[-1] = dsize - tile_size
            distribute = overflow / (n - 2)
            tiles_locations = tiles_locations.astype(float)
            tiles_locations[1:-1] -= distribute

        coordinates.append(tiles_locations)
    return coordinates


class TilesGrid:
    """Facilitates the access to tiles of a region view."""

    def __init__(self, region_view: RegionView, tile_size: Tuple[int, int],
                 tile_overlap: Tuple[int, int], mode: TilingMode = TilingMode.skip):
        self._region_view = region_view
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        basis = span_tiling_bases(
            region_view.size, tile_size,
            tile_overlap=tile_overlap, mode=mode)
        self._coordinates_grid = indexed_ndmesh(basis)
        self._coordinates = self._coordinates_grid.view(-1, len(region_view.size))

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def tile_overlap(self):
        return self._tile_overlap

    @property
    def coordinates_grid(self):
        """Grid array containing tiles starting positions"""
        return self._coordinates_grid

    @property
    def coordinates(self):
        """A flattened view of coordinates_grid."""
        return self._coordinates

    @property
    def region_view(self):
        return self._region_view

    @property
    def num_tiles(self):
        return len(self._coordinates)

    def iterator(self, sampler, retcoords=True):
        for i in sampler:
            coordinate = self.coordinates[i]
            region = self._region_view.read_region(coordinate, self._tile_size)
            yield region, tile_size if retcoords else region

    def __iter__(self):
        """Iterate through every tile."""
        return self.iterator()

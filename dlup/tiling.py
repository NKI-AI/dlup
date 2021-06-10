# coding=utf-8
# Copyright (c) DLUP Contributors

from enum import Enum
from typing import Dict, Iterable, Sequence, List, Optional, Tuple, TypeVar, Union

import numpy as np

from ._region import RegionView

_GenericNumber = Union[int, float]
_GenericNumberArray = Union[np.ndarray, Sequence[_GenericNumber]]


class TilingMode(str, Enum):
    """Type of tiling.

    Skip will skip the border tiles if they
    don't fit the region. Overflow counts as last tile
    even if it's overflowing. Fit will change the
    overlapping region between tiles to make them fit
    the region. The grid will then become non-uniform
    in case of integral values.
    """

    skip = "skip"
    overflow = "overflow"
    fit = "fit"


def _flattened_array(a: Union[_GenericNumberArray, _GenericNumber]) -> np.ndarray:
    """Converts any generic array in a flattened numpy array."""
    return np.asarray(a).flatten()


def indexed_ndmesh(basis: Sequence[_GenericNumberArray], indexing: str = "ij"):
    """Converts a list of arrays into an n-dimensional indexed mesh.

    For instance:
    ```
    mesh = indexed_ndmesh(((1, 2, 3), (4, 5, 6)))
    assert mesh[0, 0] == (1, 4)
    assert mesh[0, 1] == (1, 5)
    ```
    """
    return np.ascontiguousarray(np.stack(tuple(reversed(np.meshgrid(*reversed(basis), indexing=indexing)))).T)


def span_tiling_bases(
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
            distribute = doverflow / (n - 2)
            tiles_locations = tiles_locations.astype(float)
            tiles_locations[1:-1] -= distribute

        coordinates.append(tiles_locations)
    return coordinates


class TiledRegionView:
    """Facilitates the access to tiles of a region view."""

    region_view_cls = RegionView

    def __init__(
        self,
        region_view: RegionView,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        mode: TilingMode = TilingMode.skip,
        crop: bool = True,
    ):
        """Initialize a Tiled Region view.

        TODO(lromor): Crop is a simplification of a bigger problem.
        We should add to RegionView different modes which define what happens when
        a region is sampled outside its boundaries. It could return a cropped sample,
        return an error, or even more complex boundary conditions, or just ignore
        and try to sample outside the region anyways.
        """
        if not isinstance(region_view, self.region_view_cls):
            ValueError("region_view is not and instance of" f" {self.region_view_cls.__class__.__name__}")

        self._region_view = region_view
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._crop = crop
        basis = span_tiling_bases(region_view.size, tile_size, tile_overlap=tile_overlap, mode=mode)

        # Image coordinates usually start from the top left corner
        # with +y pointing downwards. For this reason, we set the indexing
        # to xy.
        self._coordinates_grid = indexed_ndmesh(basis, indexing="xy")

        # Let's also flatten the coordinates for simplified access.
        self._coordinates = self._coordinates_grid.view()
        self._coordinates.shape = (-1, len(region_view.size))

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def tile_overlap(self):
        return self._tile_overlap

    @property
    def coordinates_grid(self):
        """Grid array containing tiles starting positions."""
        return self._coordinates_grid

    @property
    def coordinates(self):
        """A flattened view of coordinates_grid."""
        return self._coordinates

    @property
    def region_view(self):
        return self._region_view

    def get_tile(self, i, retcoords=False):
        coordinate = self.coordinates[i]
        tile_size = self.tile_size
        clipped_tile_size = (
            np.clip(coordinate + tile_size, np.zeros_like(self.tile_size), self.region_view.size) - coordinate
        )
        clipped_tile_size = clipped_tile_size.astype(int)
        tile = self._region_view.read_region(coordinate, clipped_tile_size)

        if not self._crop:
            padding = np.zeros((len(tile.shape), 2), dtype=int)

            # This flip is justified as PIL outputs arrays with axes in reversed order
            # Extracting a box of size (width, height) results in an array
            # of shape (height, width, channels)
            padding[:-1, 1] = np.flip(tile_size - clipped_tile_size)
            values = np.zeros_like(padding)
            tile = np.pad(tile, padding, "constant", constant_values=values)

        return (tile, coordinate) if retcoords else tile

    def __getitem__(self, i):
        return self.get_tile(i)

    def get_iterator(self, retcoords=False):
        for i in range(len(self)):
            yield self.get_tile(i, retcoords)

    def __len__(self):
        return len(self._coordinates)

    def __iter__(self):
        """Iterate through every tile."""
        return self.get_iterator()

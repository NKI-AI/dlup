# coding=utf-8
# Copyright (c) DLUP Contributors

from enum import Enum
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union, List
import numpy as np


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


# class TiledRegionView:
#     """This class takes care of creating a smart object to access a wsi tiles.

#     Features, access via slices, indexes, given the tiling properties.
#     """

#     def __init__(self, region_view: RegionView, tile_size: Tuple[int, int], tile_overlap: Tuple[int, int]):
#         self._region_view = region_view
#         self._tile_size = tile_size
#         self._tile_overlap = tile_overlap

#         # # Compute the grid.
#         # stride = np.asarray(tile_size) - tile_overlap

#         # # Same thing as computing the output shape of a convolution with padding zero and
#         # # specified stride.
#         # num_tiles = (subsampled_region_size - tile_size) / stride + 1

#         # if border_mode == "crop":
#         #     num_tiles = np.ceil(num_tiles).astype(int)
#         #     tiled_size = (num_tiles - 1) * stride + tile_size
#         #     overflow = tiled_size - subsampled_region_size
#         # elif border_mode == "skip":
#         #     num_tiles = np.floor(num_tiles).astype(int)
#         #     overflow = np.asarray((0, 0))
#         # else:
#         #     raise ValueError(f"`border_mode` has to be one of `crop` or `skip`. Got {border_mode}.")

#         # indices = [range(0, _) for _ in num_tiles]

#     def __iter__(self):
#         """Iterate through every tile."""
#         pass

#     def __getitem__(self, i: int) -> PIL.Image:
#         pass

# coding=utf-8
# Copyright (c) dlup contributors

import abc
import functools
from enum import Enum
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import PIL.Image

from ._region import RegionView

_GenericNumber = Union[int, float]
_GenericNumberArray = Union[np.ndarray, Sequence[_GenericNumber]]
_GenericIntArray = Union[np.ndarray, Iterable[int]]


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


def tiles_grid_coordinates(
    size: _GenericNumberArray,
    tile_size: _GenericNumberArray,
    tile_overlap: Union[_GenericNumberArray, _GenericNumber] = 0,
    mode: TilingMode = TilingMode.skip,
    offset: Optional[_GenericNumberArray] = None,
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
    if offset is not None:
        coordinates = [c + o for c, o in zip(coordinates, offset)]
    return coordinates


class BaseGrid(abc.ABC):
    """Facilitates the access to the coordinates of Lattice points."""

    def __init__(self, coordinates: List[np.ndarray]):
        """Initialize a lattice given a set of basis vectors."""
        self.coordinates = coordinates

    @property
    def size(self) -> Tuple[int, ...]:
        """Return the size of the generated lattice."""
        return tuple(len(x) for x in self.coordinates)

    @abc.abstractmethod
    def __getitem__(self, i) -> np.ndarray:
        "Return the specified grid coordinate"

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the total number of points in the grid."""

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate through every tile."""
        for i in range(len(self)):
            yield self[i]


class Grid(BaseGrid):
    @classmethod
    def from_tiling(
        cls,
        offset: _GenericNumberArray,
        size: _GenericNumberArray,
        tile_size: _GenericNumberArray,
        tile_overlap: Union[_GenericNumberArray, _GenericNumber] = 0,
        mode: TilingMode = TilingMode.skip,
    ):
        """Generate a grid from a set of tiling parameters."""
        coordinates = tiles_grid_coordinates(size, tile_size, tile_overlap, mode, offset)
        return cls(coordinates)

    def __getitem__(self, i) -> np.ndarray:
        "Return the specified grid coordinate"
        index = np.unravel_index(i, self.size)
        return np.array(list(c[i] for c, i in zip(self.coordinates, index)))

    def __len__(self) -> int:
        """Return the total number of points in the grid."""
        return functools.reduce(lambda value, size: value * size, self.size, 1)


class MaskedGrid(Grid):
    def __init__(self, coordinates, foreground_indices):
        super().__init__(coordinates)
        self._foreground_indices = foreground_indices

    @classmethod
    def from_tiling(
        cls,
        offset: _GenericNumberArray,
        size: _GenericNumberArray,
        tile_size: _GenericNumberArray,
        tile_overlap: Union[_GenericNumberArray, _GenericNumber] = 0,
        mode: TilingMode = TilingMode.skip,
        mask: Optional[np.ndarray] = None,
        foreground_threshold: float = 0.05,
        scaled_region_view=None,
    ):
        """Generate a grid from a set of tiling parameters."""
        unfiltered_grid = Grid.from_tiling(offset, size, tile_size, tile_overlap, mode)
        if mask is None:
            return unfiltered_grid

        boolean_mask = foreground_tiles_coordinates_mask(
            mask, scaled_region_view, unfiltered_grid, tile_size, foreground_threshold
        )
        foreground_indices = np.argwhere(boolean_mask).flatten()
        return cls(unfiltered_grid.coordinates, foreground_indices)

    def __getitem__(self, i) -> np.ndarray:
        "Return the specified grid coordinate"
        i = self._foreground_indices[i]
        index = np.unravel_index(i, self.size)
        return np.array(list(c[i] for c, i in zip(self.coordinates, index)))

    def __len__(self) -> int:
        """Return the total number of points in the grid."""
        return len(self._foreground_indices)


def foreground_tiles_coordinates_mask(
    background_mask: np.ndarray,
    region_view: RegionView,
    grid: Grid,
    tile_size: _GenericIntArray,
    threshold: float = 1.0,
):
    """Generate a numpy boolean mask that can be applied to tiles coordinates.

    A tiled region view contains the tiles coordinates as a flattened grid.
    This function returns an array of boolean values being True if
    the tile is considered foreground and False otherwise.


    Parameters
    ----------
    background_mask :
        Binary mask representing of the background generated with get_mask().
    region_view :
        Target region_view we want to generate the mask for.
    grid :
        Grid of coordinates used to define tiles top-left corner.
    tile_size :
        Size of the tiles.
    threshold :
        Threshold of amount of foreground required to classify a tile as foreground.

    Returns
    -------
    np.ndarray:
        Boolean array of the same shape as the tiled_region_view.coordinates.
    """
    mask_size = np.array(background_mask.shape[:2][::-1])

    background_mask = PIL.Image.fromarray(background_mask)

    # Type of background_mask is Any here.
    scaling = background_mask.width / region_view.size[0]  # type: ignore
    scaled_tile_size = np.array(tile_size) * scaling
    scaled_tile_size = scaled_tile_size.astype(int)

    coordinates = indexed_ndmesh(grid.coordinates).view()
    coordinates.shape = (-1, len(region_view.size))
    scaled_coordinates = coordinates * scaling

    # Generate an array of boxes.
    boxes = np.hstack([scaled_coordinates, scaled_coordinates + scaled_tile_size])

    # Let's clip values outside boundaries.
    max_a = np.tile(mask_size, 2)
    min_a = np.zeros_like(max_a)
    boxes = np.clip(boxes, min_a, max_a)  # type: ignore

    # Fill in the mask with boolean values if the mean number of pixels
    # of tissue surpasses a threshold.
    mask = np.empty(len(boxes), dtype=bool)
    for i, b in enumerate(boxes):
        mask_tile = background_mask.resize(scaled_tile_size, PIL.Image.BICUBIC, box=b)  # type: ignore
        mask_tile = np.asarray(mask_tile, dtype=float)
        mask[i] = mask_tile.mean() >= threshold

    return mask

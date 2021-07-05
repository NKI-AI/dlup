# coding=utf-8
# Copyright (c) dlup contributors

from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

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


_TLazyIndexedNDMesh = TypeVar("_TLazyIndexedNDMesh", bound="_LazyIndexedNDMesh")


def indexed_ndmesh(bases: Sequence[_GenericNumberArray], indexing="ij", lazy=False) -> Union[_TLazyIndexedNDMesh, np.ndarray]:
    """Converts a list of arrays into an n-dimensional indexed mesh.

    Example
    -------

    .. code-block:: python

        import dlup
        mesh = dlup.tiling.indexed_ndmesh(((1, 2, 3), (4, 5, 6)))
        assert mesh[0, 0] == (1, 4)
        assert mesh[0, 1] == (1, 5)
    """
    lazy_mesh = _LazyIndexedNDMesh(bases, indexing=indexing)
    return lazy_mesh if lazy else np.asarray(lazy_mesh)


class _LazyIndexedNDMesh():
    """Lazily generate an ndmesh.

    https://numpy.org/doc/stable/user/basics.dispatch.html
    """

    def __init__(self, bases: Sequence[_GenericNumberArray], indexing='ij'):
        self._bases = bases
        self._indexing = indexing

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(x) for x in self._bases) + (len(self._bases),)

    @staticmethod
    def _render_indexes(indexes, target_length):
        indexes = list(indexes)
        ellipsis_idx = None
        for i, index in enumerate(indexes):
            if isinstance(index, int) or isinstance(index, slice):
                continue
            elif isinstance(index, type(Ellipsis)):
                if ellipsis_idx is not None:
                    raise IndexError("an index can only have a single ellipsis ('...')")
                ellipsis_idx = i
            else:
                raise IndexError("only integers, slices (`:`), ellipsis (`...`),"
                                 "and integer arrays are valid indices")

        ellipsis = ellipsis_idx is not None
        if len(indexes) - ellipsis > target_length:
            raise IndexError(f"too many indices for array: array is {target_length}-dimensional, "
                             f"but {len(indexes) - ellipsis} were indexed")

        if ellipsis:
            del indexes[ellipsis_idx]
            num_slices = target_length - len(indexes)
            return tuple(indexes[:ellipsis_idx] + [slice(None, None, None)] * num_slices + indexes[ellipsis_idx:])
        return indexes

    def __getitem__(self, indexes) -> Union[np.ndarray, _TLazyIndexedNDMesh]:
        """Try to support numpy indexing.

        Slices will return a lazy numpy array copy.
        Returns a numpy array containing the values
        in that index of the grid.
        """
        indexes = self.__class__._render_indexes(indexes, len(self.shape))

        if not isinstance(indexes, tuple):
            indexes = (indexes,)

        bases = [basis[index] for index, basis in zip(indexes, self._bases)]

        if len(indexes) > len(bases):
            bases = bases[indexes[-1]]

        if np.isscalar(bases):
            return bases

        return self.__class__(tuple(map(lambda x: np.atleast_1d(x), bases)))

    def __array__(self) -> np.ndarray:
        """Renders the grid into a numpy array.

        This will transform the lazy object in an eager one by returning
        a standard np.ndarray.
        """
        return np.ascontiguousarray(np.stack(tuple(reversed(np.meshgrid(*reversed(self._bases), indexing=self._indexing)))).T)


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
            distribute = doverflow / (n - 1)
            tiles_locations = tiles_locations.astype(float)
            tiles_locations[1:-1] -= distribute * (np.arange(n - 2) + 1)

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

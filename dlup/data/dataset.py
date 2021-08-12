# coding=utf-8
# Copyright (c) dlup contributors

"""Datasets helpers to simplify the generation of a dataset made of tiles from a WSI.
Dataset and ConcatDataset are taken from pytorch 1.8.0 under BSD license.
"""

import abc
import bisect
import collections
import functools
import json
import pathlib
from typing import Callable, Generic, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import PIL

from dlup import BoundaryMode, SlideImage
from dlup.background import foreground_tiles_coordinates_mask
from dlup.tiling import Grid, TilingMode
from dlup.tools import MapSequence, IndexSequence


T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Dataset(Generic[T_co], collections.abc.Sequence):
    """An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    Notes
    -----
    Taken and adapted from pytorch 1.8.0 torch.utils.data.Dataset under BSD license.
    :class:`~torch.utils.data.DataLoader` by default constructs a index
    sampler that yields integral indices.  To make it work with a map-style
    dataset with non-integral indices/keys, a custom sampler must be provided.

    """

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])


class ConcatDataset(Dataset[T_co]):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Parameters
    ----------
    datasets : sequence
        List of datasets to be concatenated

    Notes
    -----
    Taken and adapted from pytorch 1.8.0 torch.utils.data.Dataset under BSD license.

    """

    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, "datasets should not be an empty iterable"  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            if not hasattr(d, "__getitem__"):
                raise ValueError("ConcatDataset requires datasets to be indexable.")
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


LRU_CACHE_SIZE = 32


@functools.lru_cache(LRU_CACHE_SIZE)
def _get_cached_slide_image(path: pathlib.Path):
    return SlideImage.from_file_path(path)


class SlideImageDataset(Dataset):
    """Generic :class:`Dataset` to iterate over regions of a :class:`SlideImage`class.

    This class features some logic to avoid instantiating too many slides
    which for very large datasets can cause expensive allocation due to
    openslide internal caching.
    """

    def __init__(
        self,
        path: pathlib.Path,
        regions: collections.abc.Sequence,
        crop: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        path :
            Path to the image.
        regions :
            Sequence of rectangular regions as (x, y, h, w, mpp)
        crop :
            Crop overflowing tiles.
        """
        # We need to reuse the pah in order to re-open the image if necessary.
        self._path = path
        self._crop = crop
        self.regions = regions
        self.transform = transform

    @property
    def path(self):
        """Path of whole slide image"""
        return self._path

    @property
    def crop(self):
        return self._crop

    @property
    def slide_image(self):
        return _get_cached_slide_image(self.path)

    def __getitem__(self, index):
        slide_image = self.slide_image
        x, y, w, h, mpp = self.regions[index]
        coordinates = x, y
        region_size = w, h
        scaling = slide_image.mpp / mpp
        region_view = slide_image.get_scaled_view(scaling)
        region_view.boundary_mode = BoundaryMode.crop if self.crop else BoundaryMode.zero

        region = region_view.read_region(coordinates, region_size)
        sample = {"image": region, "coordinates": coordinates, "mpp": mpp, "path": self.path}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.regions)


class TiledLevelSlideImageDataset(SlideImageDataset):
    """Typical dataset dataset use-case with fixed mpp."""

    def __init__(
        self,
        path: pathlib.Path,
        mpp: float,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        tile_mode: TilingMode = TilingMode.skip,
        mask: Optional[np.ndarray] = None,
        foreground_threshold: float = 0.1,
        transform: Optional[Callable] = None,
    ):
        super().__init__(path, [], True, transform)
        self._mpp = mpp
        self._tile_size = tile_size
        region_view = self.region_view

        grid = Grid.from_tiling(
            offset=(0, 0),
            size=region_view.size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            mode=tile_mode,
        )
        self._grid = grid

        def coords_to_region(key, coords):
            """Return the necessary tuple that represents a region."""
            return (*coords, *tile_size, mpp)

        self.regions = MapSequence(coords_to_region, grid)

        self.foreground_indices = None
        if mask is not None:
            boolean_mask = foreground_tiles_coordinates_mask(mask, region_view, grid, tile_size, foreground_threshold)
            self.foreground_indices = np.argwhere(boolean_mask).flatten()

    @property
    def region_view(self):
        slide_image = self.slide_image
        region_view = slide_image.get_scaled_view(slide_image.mpp / self.mpp)
        region_view.boundary_mode = BoundaryMode.crop if self.crop else BoundaryMode.zero
        return region_view

    @property
    def mpp(self):
        return self._mpp

    @property
    def grid(self):
        """Tiling grid (read only)"""
        return self._grid

    @property
    def tile_size(self):
        """Tile size (read only)"""
        return self._tile_size

    def __getitem__(self, index):
        has_mask = self.foreground_indices is not None
        tile_index = self.foreground_indices[index] if has_mask else index

        data = super().__getitem__(tile_index)
        grid_index = np.unravel_index(index, self.grid.size)
        data.update({"grid_index": grid_index})
        return data


class PreTiledSlideImageDataset(Dataset):
    """Dataset class to handle a pretiled WSIs. If you want to combine multiple WSIs, use :class:`ConcatDataset`.

    Examples
    --------
    >>> ds = ConcatDataset([_ for _ in self.path.glob("*.svs")]_

    """

    def __init__(self, path: pathlib.Path, transform: Optional[Callable] = None):
        """
        Parameters
        ----------
        path :
            Path to the folder containing the tiles and tiles.json.
        transform :
            Callable which should be applied after obtaining the sample, e.g. for augmentations.

        """
        self.path = pathlib.Path(path)
        self.transform = transform
        with open(self.path / "tiles.json") as json_file:
            tiles_data = json.load(json_file)

        self.original_path = pathlib.Path(tiles_data["original"]["input_file_path"])
        self.mpp = tiles_data["output"]["mpp"]
        self.size = tiles_data["output"]["size"]
        self._num_tiles = tiles_data["output"]["num_tiles"]
        self._tile_indices = tiles_data["output"]["tile_indices"]

    def __getitem__(self, index):
        grid_index = self._tile_indices[index]
        path_to_tile = self.path / "tiles" / f"{'_'.join(map(str, grid_index))}.png"
        # TODO(jt): Figure out why the mode is RGB
        tile = PIL.Image.open(path_to_tile).convert("RGB")

        # TODO(jt): do something about the coordinates
        # Perhaps, they can be inferred in the same way as the original image
        # So do not directly compute from the current grid_index
        sample = {"image": tile, "grid_index": grid_index, "path": self.original_path}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __iter__(self):
        pass

    def __len__(self):
        return self._num_tiles

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
from dlup.background import is_foreground
from dlup.tiling import Grid, TilingMode
from dlup.tools import MapSequence, ConcatSequences


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
        mask: Optional[np.ndarray] = None,
        mask_threshold: float = 0.1,
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
        transform :
            Transforming function.
        mask :
            Binary mask used to filter each region toghether with a threshold.
        mask_threshold :
            0 every region is discarded, 1 requires the whole region to be foreground.
        """
        # We need to reuse the pah in order to re-open the image if necessary.
        self._path = path
        self._crop = crop
        self.regions = regions
        self.transform = transform

        # Maps from a masked index -> regions index.
        # For instance, let's say we have three regions
        # masked according to the following boolean values: [True, False, True].
        # Then masked_indices[0] == 0, masked_indices[1] == 2.
        self.masked_indices = None
        if mask is not None:
            boolean_mask = np.zeros(len(regions))
            for i, region in enumerate(regions):
                boolean_mask[i] = is_foreground(self.slide_image, mask, region, mask_threshold)
            self.masked_indices = np.argwhere(boolean_mask).flatten()

    @property
    def path(self):
        """Path of whole slide image"""
        return self._path

    @property
    def crop(self):
        """Returns true the regions will be cropped at the boundaries."""
        return self._crop

    @property
    def slide_image(self):
        """Return the cached slide image instance associated with this dataset."""
        return _get_cached_slide_image(self.path)

    def __getitem__(self, index):
        slide_image = self.slide_image

        # If there's a mask, we consider the index as a sub-sequence index.
        # Let's map it back to the original regions index.
        has_mask = self.masked_indices is not None
        region_index = self.masked_indices[index] if has_mask else index

        x, y, w, h, mpp = self.regions[region_index]
        coordinates = x, y
        region_size = w, h
        scaling = slide_image.mpp / mpp
        region_view = slide_image.get_scaled_view(scaling)
        region_view.boundary_mode = BoundaryMode.crop if self.crop else BoundaryMode.zero

        region = region_view.read_region(coordinates, region_size)
        sample = {
            "image": region,
            "coordinates": coordinates,
            "mpp": mpp,
            "path": self.path,
            "index": index,
        }
        if has_mask:
            sample["unmasked_index"] = region_index

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        """Returns the length of the dataset.

        The length may vary depending on the provided boolean mask.
        """
        return len(self.regions) if self.masked_indices is None else len(self.masked_indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _coords_to_region(tile_size, target_mpp, key, coords):
    """Return the necessary tuple that represents a region."""
    return (*coords, *tile_size, target_mpp)


class TiledROIsSlideImageDataset(SlideImageDataset):
    """Example dataset dataset class that supports multiple ROIs."""

    def __init__(
        self,
        path: pathlib.Path,
        grids: Iterable[Tuple[Grid, Tuple[int, int], float]],
        crop: bool = True,
        mask: Optional[np.ndarray] = None,
        mask_threshold: float = 0.1,
        transform: Optional[Callable] = None,
    ):
        self._grids = grids
        regions = []
        for grid, tile_size, mpp in grids:
            regions.append(MapSequence(functools.partial(_coords_to_region, tile_size, mpp), grid))

        super().__init__(
            path, ConcatSequences(regions), crop, mask=mask, mask_threshold=mask_threshold, transform=transform
        )

    @property
    def grids(self):
        return self._grids

    @classmethod
    def from_standard_tiling(
        cls,
        path: pathlib.Path,
        mpp: float,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        tile_mode: TilingMode = TilingMode.skip,
        crop: bool = True,
        mask: Optional[np.ndarray] = None,
        mask_threshold: float = 0.1,
        transform: Optional[Callable] = None,
    ):
        with SlideImage.from_file_path(path) as slide_image:
            slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
        grid = Grid.from_tiling(
            (0, 0),
            size=slide_level_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            mode=tile_mode,
        )
        return cls(path, [(grid, tile_size, mpp)], crop, mask, mask_threshold, transform)


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

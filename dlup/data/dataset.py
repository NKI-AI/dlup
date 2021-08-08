# coding=utf-8
# Copyright (c) dlup contributors

"""Datasets helpers to simplify the generation of a dataset made of tiles from a WSI.
Dataset and ConcatDataset are taken from pytorch 1.8.0 under BSD license.
"""

import abc
import bisect
import functools
import itertools
import json
import pathlib
from typing import Callable, Generic, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import PIL

from dlup import BoundaryMode, SlideImage
from dlup.tiling import Grid, MaskedGrid, TilingMode

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class Dataset(Generic[T_co], abc.ABC):
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

    @abc.abstractmethod
    def __getitem__(self, index) -> T_co:
        """Index method for dataset."""

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


class AbstractGridSlideImageDataset(Dataset, abc.ABC):
    """
    Basic :class:`Dataset` class that represents a whole-slide image as tiles.
    """

    def __init__(
        self,
        path: pathlib.Path,
        mpp: float,
        grid_func: Callable,
        crop: bool = True,
        transform: Optional[Callable] = None,
    ):
        # We need to reuse the path in order to re-open the image if necessary.
        self._path = path
        self._mpp = mpp
        self._crop = crop
        self._grid = grid_func()
        self.transform = transform

    @staticmethod
    @functools.lru_cache(32)
    def get_slide_image(path: pathlib.Path):
        return SlideImage.from_file_path(path)

    @property
    def slide_image(self):
        return self.__class__.get_slide_image(self.path)

    @property
    def region_view(self):
        slide_image = self.slide_image
        region_view = slide_image.get_scaled_view(slide_image.mpp / self._mpp)
        region_view.boundary_mode = BoundaryMode.crop if self.crop else BoundaryMode.zero
        return region_view

    @property
    def path(self):
        """Path of whole slide image"""
        return self._path

    @property
    def mpp(self):
        return self._mpp

    @property
    def crop(self):
        return self._crop

    @property
    def grid(self):
        """Tiling grid (read only)"""
        return self._grid

    def __getitem__(self, index):
        coordinates = self._grid[index]
        tile = self.region_view.read_region(coordinates, self._tile_size)
        grid_index = np.unravel_index(index, self.grid.size, order="C")
        sample = {"image": tile, "coordinates": coordinates, "grid_index": grid_index, "path": self.path}

        if self.transform:
            sample = self.transform(sample)
        return sample

    @abc.abstractmethod
    def __len__(self):
        """Abstract method. Should return number of tiles."""


class MaskedSlideImageDataset(AbstractGridSlideImageDataset):
    """
    :class:`Dataset` class that represents a whole-slide image as tiles, possibly including a sampling mask.
    The function outputs a dictionary:
    >>> {"image": array, "coordinates": coordinates, "grid_index": grid_index, "path": path}
    Keys:
        - :code:`image`: selected tile.
        - :code:`coordinates`: coordinates in selected mpp.
        - :code:`grid_index`: index in the tiling grid.
        - :code:`path`: path of the file.
    """

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
        """
        Parameters
        ----------
        path :
            Path to the image.
        mpp :
            Requested microns per pixel.
        tile_size :
            Tile size in the requested microns per pixel.
        tile_overlap :
            Overlap of the extracted tiles.
        tile_mode :
            Which tiling mode.
        mask :
            Array denoting the sampling mask.
        foreground_threshold :
            The percentage of non-zero pixels required in the mask to include a tile.
        transform :
            Callable which should be applied after obtaining the sample, e.g. for augmentations.
        """
        self.transform = transform
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._tile_mode = tile_mode
        self._mask = mask
        self._foreground_threshold = foreground_threshold

        super().__init__(path, mpp=mpp, grid_func=self._construct_grid, transform=transform)

    def _construct_grid(self):
        slide_level_size = self.slide_image.get_scaled_size(self.slide_image.mpp / self.mpp)
        grid = MaskedGrid.from_tiling(
            size=slide_level_size,
            tile_size=self.tile_size,
            tile_overlap=self._tile_overlap,
            mode=self._tile_mode,
            mask=self._mask,
            foreground_threshold=self._foreground_threshold,
            offset=None,
        )
        return grid

    @property
    def tile_size(self):
        """Tile size (read only)"""
        return self._tile_size

    def __len__(self):
        return len(self.grid)


class ROISlideImageDataset(MaskedSlideImageDataset):
    def __init__(
        self,
        path: pathlib.Path,
        mpp: float,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        rois: Optional[list] = None,
        tile_mode: TilingMode = TilingMode.skip,
        mask: Optional[np.ndarray] = None,
        foreground_threshold: float = 0.1,
        transform: Optional[Callable] = None,
    ):
        super().__init__(path, mpp, tile_size, tile_overlap, tile_mode, mask, foreground_threshold, transform)
        self.scaling = self.slide_image.mpp / self.mpp
        if rois is not None and len(rois) > 0:
            self.rois_scaled = [np.floor(np.asarray(r) * self.scaling).astype(int) for r in rois]  # [y1, x1, h, w]
            self._grid = list(
                itertools.chain.from_iterable(
                    Grid.from_tiling(
                        size=r[2:4], tile_size=tile_size, tile_overlap=tile_overlap, mode=tile_mode, offset=r[0:2]
                    )
                    for r in self.rois_scaled
                )
            )


class TiledSlideImageDataset(Dataset):
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

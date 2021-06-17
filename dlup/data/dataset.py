# coding=utf-8
# Copyright (c) dlup contributors

"""Datasets helpers to simplify the generation of a dataset made of tiles from a WSI.
Dataset and ConcatDataset are taken from pytorch 1.8.0 under BSD license.
"""

import abc
import bisect
import pathlib
import numpy as np
from typing import Generic, Iterable, List, Tuple, TypeVar, Optional

from dlup import SlideImage, SlideImageTiledRegionView
from dlup.background import foreground_tiles_coordinates_mask
from dlup.tiling import TilingMode

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

    Note
    ----
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
    datasets : sequence:
        List of datasets to be concatenated

    Note
    ----
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


class SlideImageDataset(Dataset, SlideImageTiledRegionView):
    """Basic Slide Image dataset."""

    def __init__(
        self,
        path: pathlib.Path,
        mpp: float,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        tile_mode: TilingMode = TilingMode.skip,
        mask: Optional[np.ndarray] = None,
        foreground_threshold: float = 0.1,
    ):
        self._path = path
        self._slide_image = SlideImage.from_file_path(path)
        scaled_view = self._slide_image.get_scaled_view(self._slide_image.mpp / mpp)
        super().__init__(scaled_view, tile_size, tile_overlap, tile_mode, crop=False)

        self.foreground_indices = None
        if mask is not None:
            boolean_mask = foreground_tiles_coordinates_mask(mask, self, foreground_threshold)
            self.foreground_indices = np.argwhere(boolean_mask).flatten()

    @property
    def path(self):
        """Path of whole slide image"""
        return self._path

    @property
    def slide_image(self):
        """SlideImage instance"""
        return self._slide_image

    def __getitem__(self, i):
        index = self.foreground_indices[i] if self.foreground_indices is not None else i
        return SlideImageTiledRegionView.__getitem__(self, index)

    def __len__(self):
        return len(self.foreground_indices) if self.foreground_indices is not None else SlideImageTiledRegionView.__len__(self)

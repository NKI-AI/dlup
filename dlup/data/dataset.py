# coding=utf-8
# Copyright (c) dlup contributors

"""Datasets helpers to simplify the generation of a dataset made of tiles from a WSI.
Dataset and ConcatDataset are taken from pytorch 1.8.0 under BSD license.
"""
from __future__ import annotations

import bisect
import collections
import functools
import itertools
import pathlib
from typing import Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypedDict, TypeVar, Union, cast

import numpy as np
import PIL
from numpy.typing import NDArray
from PIL import Image

from dlup import BoundaryMode, SlideImage
from dlup.annotations import WsiAnnotations
from dlup.background import is_foreground
from dlup.experimental_backends import ImageBackend
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.tools import ConcatSequences, MapSequence

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
_BaseAnnotationTypes = Union[SlideImage, WsiAnnotations]
_AnnotationTypes = Union[List[Tuple[str, _BaseAnnotationTypes]], _BaseAnnotationTypes]
_LabelTypes = Union[str, bool, int, float]


class StandardTilingFromSlideDatasetSample(TypedDict):
    image: PIL.Image.Image
    coordinates: Tuple[int, int]
    mpp: float
    path: pathlib.Path
    region_index: int


class RegionFromSlideDatasetSample(StandardTilingFromSlideDatasetSample):
    grid_local_coordinates: Tuple
    grid_index: int


class PretiledDatasetSample(TypedDict):
    image: PIL.Image.Image
    grid_index: int
    path: pathlib.Path


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

    def __getitem__(self, index: int) -> T_co:  # type: ignore
        raise IndexError


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
    wsi_indices: Dict[str, range]

    @staticmethod
    def cumsum(sequence):
        out_sequence, total = [], 0
        for item in sequence:
            length = len(item)
            out_sequence.append(length + total)
            total += length
        return out_sequence

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
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
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


LRU_CACHE_SIZE = 32


@functools.lru_cache(LRU_CACHE_SIZE)
def _get_cached_slide_image(path: pathlib.Path, backend):
    return SlideImage.from_file_path(path, backend=backend)


class SlideImageDatasetBase(Dataset[T_co]):
    """Generic :class:`Dataset` to iterate over regions of a :class:`SlideImage`class.

    This class features some logic to avoid instantiating too many slides
    which for very large datasets can cause expensive allocation due to
    openslide internal caching.

    This class is the superclass of :class:`TiledROIsSlideImageDataset`, which has a function,
    `from_standard_tiling`, to compute all the regions for specified tiling parameters on the fly.
    """

    def __init__(
        self,
        path: pathlib.Path,
        regions: collections.abc.Sequence,
        crop: bool = False,
        mask: Optional[Union[SlideImage, np.ndarray, WsiAnnotations]] = None,
        mask_threshold: float = 0.1,
        annotations: Optional[Union[List[_AnnotationTypes], _AnnotationTypes]] = None,
        labels: Optional[List[Tuple[str, _LabelTypes]]] = None,
        transform: Optional[Callable] = None,
        backend: Callable = ImageBackend.PYVIPS,
    ):
        """
        Parameters
        ----------
        path :
            Path to the image.
        regions :
            Sequence of rectangular regions as (x, y, h, w, mpp)
        crop :
            Whether or not to crop overflowing tiles.
        mask :
            Binary mask used to filter each region toghether with a threshold.
        mask_threshold :
            0 every region is discarded, 1 requires the whole region to be foreground.
        annotations :
            Annotation classes.
        labels : list
            Image-level labels. Will be added to each individual tile.
        transform :
            Transforming function. To be used for augmentations or other model specific preprocessing.
        """
        # We need to reuse the pah in order to re-open the image if necessary.
        self._path = path
        self._crop = crop
        self.regions = regions

        self.annotations = annotations
        self.labels = labels
        self.__transform = transform
        self._backend = backend

        # Maps from a masked index -> regions index.
        # For instance, let's say we have three regions
        # masked according to the following boolean values: [True, False, True].
        # Then masked_indices[0] == 0, masked_indices[1] == 2.
        self.masked_indices: Union[NDArray[np.int_], None] = None
        if mask is not None:
            boolean_mask: NDArray[np.bool_] = np.zeros(len(regions), dtype=bool)
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
        return _get_cached_slide_image(self.path, self._backend)

    def __getitem__(self, index):
        slide_image = self.slide_image

        # If there's a mask, we consider the index as a sub-sequence index.
        # Let's map it back to the original regions index.
        region_index: int
        if self.masked_indices is not None:
            region_index = self.masked_indices[index]
        else:
            region_index = index

        x, y, w, h, mpp = self.regions[region_index]
        coordinates: Tuple[int, int] = x, y
        region_size: Tuple[int, int] = w, h
        scaling: float = slide_image.mpp / mpp
        region_view = slide_image.get_scaled_view(scaling)
        region_view.boundary_mode = BoundaryMode.crop if self.crop else BoundaryMode.zero

        region = region_view.read_region(coordinates, region_size)

        sample: StandardTilingFromSlideDatasetSample = {
            "image": region,
            "coordinates": coordinates,
            "mpp": mpp,
            "path": self.path,
            "region_index": region_index,
        }

        if self.annotations is not None:
            sample["annotations"] = self.annotations.read_region(coordinates, scaling, region_size)

        if self.labels:
            sample["labels"] = {k: v for k, v in self.labels}

        if self.__transform:
            sample = self.__transform(sample)
        return sample

    def __len__(self):
        """Returns the length of the dataset.

        The length may vary depending on the provided boolean mask.
        """
        return len(self.regions) if self.masked_indices is None else len(self.masked_indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class SlideImageDataset(SlideImageDatasetBase[StandardTilingFromSlideDatasetSample]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _coords_to_region(tile_size, target_mpp, key, coords):
    """Return the necessary tuple that represents a region."""
    return *coords, *tile_size, target_mpp


class TiledROIsSlideImageDataset(SlideImageDatasetBase[RegionFromSlideDatasetSample]):
    """Example dataset dataset class that supports multiple ROIs.

    This dataset can be used, for example, to tile your WSI on-the-fly using the `from_standard_tiling` function.

    Examples
    --------
    >>>  dlup_dataset = TiledROIsSlideImageDataset.from_standard_tiling(\
            path='/path/to/TCGA-WSI.svs',\
            mpp=0.5,\
            tile_size=(512,512),\
            tile_overlap=(0,0),\
            tile_mode='skip',\
            crop=True,\
            mask=None,\
            mask_threshold=0.5,\
            annotations=None,\
            labels=[("msi", True),]
            transform=YourTransform()\
         )
    >>> sample = dlup_dataset[5]
    >>> image = sample["image']
    """

    def __init__(
        self,
        path: pathlib.Path,
        grids: List[Tuple[Grid, Tuple[int, int], float]],
        crop: bool = False,
        mask: Optional[Union[SlideImage, np.ndarray, WsiAnnotations]] = None,
        mask_threshold: float = 0.1,
        annotations: Optional[_AnnotationTypes] = None,
        labels: Optional[List[Tuple[str, _LabelTypes]]] = None,
        transform: Optional[Callable] = None,
        backend: Callable = ImageBackend.PYVIPS,
    ):
        self._grids = grids
        regions = []
        for grid, tile_size, mpp in grids:
            regions.append(MapSequence(functools.partial(_coords_to_region, tile_size, mpp), grid))

        self._starting_indices = [0] + list(itertools.accumulate([len(s) for s in regions]))[:-1]

        super().__init__(
            path,
            ConcatSequences(regions),
            crop,
            mask=mask,
            mask_threshold=mask_threshold,
            annotations=annotations,
            labels=labels,
            transform=None,
            backend=backend,
        )
        self.__transform = transform

    @property
    def grids(self):
        return self._grids

    @classmethod
    def from_standard_tiling(
        cls,
        path: pathlib.Path,
        mpp: float | None,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        tile_mode: TilingMode = TilingMode.overflow,
        grid_order: GridOrder = GridOrder.C,
        crop: bool = False,
        mask: Optional[Union[SlideImage, np.ndarray, WsiAnnotations]] = None,
        mask_threshold: float = 0.1,
        rois: Optional[Tuple[Tuple[int, ...]]] = None,
        annotations: Optional[_AnnotationTypes] = None,
        labels: Optional[List[Tuple[str, _LabelTypes]]] = None,
        transform: Optional[Callable] = None,
        backend: Callable = ImageBackend.PYVIPS,
    ):
        """Function to be used to tile a WSI on-the-fly.
        Parameters
        ----------
        path :
            path to a single WSI
        mpp :
            float stating the microns per pixel that you wish the tiles to be.
        tile_size :
            Tuple of integers that represent the pixel size of output tiles
        tile_overlap :
            Tuple of integers that represents the overlap of tiles in the x and y direction
        tile_mode :
            "skip" or "overflow". see `dlup.tiling.TilingMode` for more information
        grid_order : GridOrder
            Run through the grid either in C order or Fortran order.
        crop :
            Whether or not to crop overflowing tiles.
        mask :
            Binary mask used to filter each region toghether with a threshold.
        mask_threshold :
            0 every region is discarded, 1 requires the whole region to be foreground.
        rois :
            Regions of interest to restrict the grids to. Coordinates should be given at level 0.
        annotations :
            Annotation class
        labels : list
            Image-level labels. Will be added to each individual tile.
        transform : ImageBackend
            Transform to be applied to the sample.
        backend :
            Backend to use to read the whole slide image.

        Examples
        --------
        See example of usage in the main class docstring

        Returns
        -------
        Initialized SlideImageDataset with all the regions as computed using the given tile size, mpp, and so on.
        Calling this dataset with an index will return a tile extracted straight from the WSI. This means tiling as
        pre-processing step is not required.
        """
        with SlideImage.from_file_path(path, backend=backend) as slide_image:
            slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
            slide_mpp = slide_image.mpp
            _rois = parse_rois(rois, slide_level_size, scaling=slide_mpp / mpp if mpp else 1.0)
        grid_mpp = mpp if mpp is not None else slide_mpp

        grids = []
        for offset, size in _rois:
            grid = Grid.from_tiling(
                offset,
                size=size,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                mode=tile_mode,
                order=grid_order,
            )
            grids.append((grid, tile_size, grid_mpp))

        return cls(
            path,
            grids=grids,
            crop=crop,
            mask=mask,
            mask_threshold=mask_threshold,
            annotations=annotations,
            labels=labels,
            transform=transform,
            backend=backend,
        )

    def __getitem__(self, index):
        data = super().__getitem__(index)
        region_data: RegionFromSlideDatasetSample = cast(RegionFromSlideDatasetSample, data)
        region_index = data["region_index"]
        starting_index = bisect.bisect_right(self._starting_indices, region_index) - 1
        grid_index = region_index - self._starting_indices[starting_index]
        grid_local_coordinates = np.unravel_index(grid_index, self.grids[starting_index][0].size)
        region_data["grid_local_coordinates"] = grid_local_coordinates
        region_data["grid_index"] = starting_index

        if self.__transform:
            region_data = self.__transform(region_data)

        return region_data


def parse_rois(rois, image_size, scaling: float) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...]:
    if rois is None:
        return (((0, 0), image_size),)
    else:
        # Do some checks whether the ROIs are within the image
        origin_positive = [np.all(np.asarray(coords) > 0) for coords, size in rois]
        image_within_borders = [np.all((np.asarray(coords) + size) <= image_size) for coords, size in rois]
        if not origin_positive or not image_within_borders:
            raise ValueError(f"ROIs should be within image boundaries. Got {rois}.")

    rois = [
        (
            np.ceil(np.asarray(coords) * scaling).astype(int).tolist(),
            np.floor(np.asarray(size) * scaling).astype(int).tolist(),
        )
        for coords, size in rois
    ]
    return rois

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
import warnings
from math import ceil, floor
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, Sequence, TypedDict, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import PIL
from numpy.typing import NDArray

from dlup import BoundaryMode, SlideImage
from dlup.annotations import WsiAnnotations
from dlup.background import is_foreground
from dlup.experimental_backends import ImageBackend  # type: ignore
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.tools import ConcatSequences, MapSequence
from dlup.types import ROIType

MaskTypes = Union["SlideImage", npt.NDArray[np.int_], "WsiAnnotations"]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
_BaseAnnotationTypes = Union[SlideImage, WsiAnnotations]
_AnnotationTypes = Union[list[tuple[str, _BaseAnnotationTypes]], _BaseAnnotationTypes]
_LabelTypes = Union[str, bool, int, float]


class TileSample(TypedDict):
    image: PIL.Image.Image
    coordinates: tuple[int | float, int | float]
    mpp: float
    path: pathlib.Path
    region_index: int
    labels: dict[str, Any] | None
    annotations: Any | None


PointType = tuple[float, float]
BoundingBoxType = tuple[tuple[int, int], tuple[int, int]]

AnnotationData = TypedDict(
    "AnnotationData",
    {
        "points": dict[str, list[PointType]],
        "boxes": dict[str, list[BoundingBoxType]],
        "mask": npt.NDArray[np.int_],
        "roi": Optional[npt.NDArray[np.int_]],
    },
)


class TileSampleWithAnnotationData(TypedDict):
    annotation_data: AnnotationData


class RegionFromWsiDatasetSample(TileSample):
    grid_local_coordinates: tuple[int, int]
    grid_index: int


class Dataset(Generic[T_co], collections.abc.Sequence[T_co]):
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

    datasets: list[Dataset[T_co]]
    cumulative_sizes: list[int]
    wsi_indices: dict[str, range]

    @staticmethod
    def cumsum(sequence: list[Dataset[T_co]]) -> list[int]:
        out_sequence, total = [], 0
        for item in sequence:
            length = len(item)
            out_sequence.append(length + total)
            total += length
        return out_sequence

    def __init__(self, datasets: Iterable[Dataset[T_co]]) -> None:
        super().__init__()
        # Cannot verify that datasets is Sized
        assert len(datasets) > 0, "datasets should not be an empty iterable"  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            if not hasattr(d, "__getitem__"):
                raise ValueError("ConcatDataset requires datasets to be indexable.")
        self.cumulative_sizes = self.cumsum(self.datasets)

    def index_to_dataset(self, idx: int) -> tuple[Dataset[T_co], int]:
        """Returns the dataset and the index of the sample in the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample in the concatenated dataset.

        Returns
        -------
        tuple[Dataset, int]
            Dataset and index of the sample in the dataset.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx], sample_idx

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    @overload
    def __getitem__(self, index: int) -> T_co:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[T_co]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> T_co | list[T_co]:
        """Returns the sample at the given index."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step or 1)]

        dataset, sample_idx = self.index_to_dataset(index)
        return dataset[sample_idx]


LRU_CACHE_SIZE = 32


@functools.lru_cache(LRU_CACHE_SIZE)
def _get_cached_slide_image(path: pathlib.Path, backend: ImageBackend, **kwargs: Any) -> "SlideImage":
    return SlideImage.from_file_path(path, backend=backend, **kwargs)


class BaseWsiDataset(Dataset[Union[TileSample, Sequence[TileSample]]]):
    """Generic :class:`Dataset` to iterate over regions of a :class:`SlideImage`class.

    This class features some logic to avoid instantiating too many slides
    which for very large datasets can cause expensive allocation due to
    internal caching of the image reading backand.

    This class is the superclass of :class:`TiledWsiDataset`, which has a function,
    `from_standard_tiling`, to compute all the regions for specified tiling parameters on the fly.
    """

    def __init__(
        self,
        path: pathlib.Path,
        regions: collections.abc.Sequence[tuple[float, float, int, int, float]],
        crop: bool = False,
        mask: MaskTypes | None = None,
        mask_threshold: float | None = 0.0,
        annotations: list[_AnnotationTypes] | _AnnotationTypes | None = None,
        labels: list[tuple[str, _LabelTypes]] | None = None,
        backend: ImageBackend = ImageBackend.PYVIPS,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        path :
            Path to the image.
        regions :
            Sequence of rectangular regions as (x, y, h, w, mpp)
        crop :
            Whether to crop overflowing tiles.
        mask :
            Binary mask used to filter each region together with a threshold.
        mask_threshold : float, optional
            Threshold to check against. The foreground percentage should be strictly larger than threshold.
            If None anything is foreground. If 1, the region must be completely foreground.
            Other values are in between, for instance if 0.5, the region must be at least 50% foreground.
        annotations :
            Annotation classes.
        labels : list
            Image-level labels. Will be added to each individual tile.
        transform :
            Transforming function. To be used for augmentations or other model specific preprocessing.
        **kwargs :
            Keyword arguments get passed to the underlying slide image.
        """
        # We need to reuse the pah in order to re-open the image if necessary.
        self._path = path
        self._crop = crop
        self.regions = regions

        self.annotations = annotations
        self.labels = labels
        self._backend = backend
        self._kwargs = kwargs

        # Maps from a masked index -> regions index.
        # For instance, let's say we have three regions
        # masked according to the following boolean values: [True, False, True].
        # Then masked_indices[0] == 0, masked_indices[1] == 2.
        self.masked_indices: NDArray[np.int_] | None = None
        if mask is not None:
            boolean_mask: NDArray[np.bool_] = np.zeros(len(regions), dtype=bool)
            for i, region in enumerate(regions):
                boolean_mask[i] = is_foreground(self.slide_image, mask, region, mask_threshold)
            self.masked_indices = np.argwhere(boolean_mask).flatten()

    @property
    def path(self) -> pathlib.Path:
        """Path of whole slide image"""
        return self._path

    @property
    def crop(self) -> bool:
        """Returns true the regions will be cropped at the boundaries."""
        return self._crop

    @property
    def slide_image(self) -> "SlideImage":
        """Return the cached slide image instance associated with this dataset."""
        return _get_cached_slide_image(self.path, self._backend, **self._kwargs)

    @overload
    def __getitem__(self, index: int) -> TileSample:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TileSample]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[TileSample, Sequence[TileSample]]:
        if isinstance(index, slice):
            # handle slicing
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step or 1)]

        slide_image = self.slide_image

        # If there's a mask, we consider the index as a sub-sequence index.
        # Let's map it back to the original regions index.
        region_index: int
        if self.masked_indices is not None:
            region_index = self.masked_indices[index]
        else:
            region_index = index

        x, y, w, h, mpp = self.regions[region_index]
        coordinates: tuple[int | float, int | float] = x, y
        region_size: tuple[int, int] = w, h
        scaling: float = slide_image.mpp / mpp
        region_view = slide_image.get_scaled_view(scaling)
        region_view.boundary_mode = BoundaryMode.crop if self.crop else BoundaryMode.zero

        region = region_view.read_region(coordinates, region_size)

        sample: TileSample = {
            "image": region,
            "coordinates": coordinates,
            "mpp": mpp,
            "path": self.path,
            "region_index": region_index,
            "labels": None,
            "annotations": None,
        }

        # TODO: This needs to move to TiledWsiDataset (v1.0)
        if self.annotations is not None:
            if not isinstance(self.annotations, WsiAnnotations):
                raise NotImplementedError("Only WsiAnnotations are supported at the moment.")
            sample["annotations"] = self.annotations.read_region(coordinates, scaling, region_size)

        if self.labels:
            sample["labels"] = {k: v for k, v in self.labels}

        return sample

    def __len__(self) -> int:
        """Returns the length of the dataset.

        The length may vary depending on the provided boolean mask.
        """
        return len(self.regions) if self.masked_indices is None else len(self.masked_indices)

    def __iter__(self) -> Iterator[TileSample]:
        for i in range(len(self)):
            yield self[i]


def _coords_to_region(
    tile_size: tuple[int, int], target_mpp: float, key: str, coords: tuple[int, int]
) -> tuple[int, int, int, int, float]:
    """Return the necessary tuple that represents a region."""
    return *coords, *tile_size, target_mpp


class TiledWsiDataset(BaseWsiDataset):
    """Example dataset class that supports multiple ROIs.

    This dataset can be used, for example, to tile your WSI on-the-fly using the `from_standard_tiling` function.

    Examples
    --------
    >>>  dlup_dataset = TiledWsiDataset.from_standard_tiling(\
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
        grids: list[tuple[Grid, tuple[int, int], float]],
        crop: bool = False,
        mask: MaskTypes | None = None,
        mask_threshold: float | None = 0.0,
        annotations: _AnnotationTypes | None = None,
        labels: list[tuple[str, _LabelTypes]] | None = None,
        transform: Callable[[RegionFromWsiDatasetSample], RegionFromWsiDatasetSample] | None = None,
        backend: ImageBackend = ImageBackend.PYVIPS,
        **kwargs: Any,
    ) -> None:
        self._grids = grids
        regions: list[Sequence[tuple[float, float, int, int, float]]] = []
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
            backend=backend,
            **kwargs,
        )
        self._transform = transform

    @property
    def grids(self) -> list[tuple[Grid, tuple[int, int], float]]:
        return self._grids

    @classmethod
    def from_standard_tiling(
        cls,
        path: pathlib.Path,
        mpp: float | None,
        tile_size: tuple[int, int],
        tile_overlap: tuple[int, int],
        tile_mode: TilingMode = TilingMode.overflow,
        grid_order: GridOrder = GridOrder.C,
        crop: bool = False,
        mask: MaskTypes | None = None,
        mask_threshold: float | None = 0.0,
        rois: list[ROIType] | None = None,
        annotations: _AnnotationTypes | None = None,
        labels: list[tuple[str, _LabelTypes]] | None = None,
        transform: Callable[[TileSample], RegionFromWsiDatasetSample] | None = None,
        backend: ImageBackend = ImageBackend.PYVIPS,
        limit_bounds: bool = True,
        **kwargs: Any,
    ) -> "TiledWsiDataset":
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
        crop : bool
             If overflowing tiles should be cropped.
        mask :
            Binary mask used to filter each region together with a threshold.
        mask_threshold : float, optional
            Threshold to check against. The foreground percentage should be strictly larger than threshold.
            If None anything is foreground. If 1, the region must be completely foreground.
            Other values are in between, for instance if 0.5, the region must be at least 50% foreground.
        rois :
            Regions of interest to restrict the grids to. Coordinates should be given at level 0.
        annotations :
            Annotation class
        labels : list
            Image-level labels. Will be added to each individual tile.
        transform : Callable
            Transform to be applied to the sample.
        backend : ImageBackend
            Backend to use to read the whole slide image.
        limit_bounds : bool
            If the bounds of the grid should be limited to the bounds of the slide given in the `slide_bounds` property
            of the `SlideImage` class.
        **kwargs :
            Gets passed to the SlideImage constructor.

        Examples
        --------
        See example of usage in the main class docstring

        Returns
        -------
        Initialized SlideImageDataset with all the regions as computed using the given tile size, mpp, and so on.
        Calling this dataset with an index will return a tile extracted straight from the WSI. This means tiling as
        pre-processing step is not required.
        """
        with SlideImage.from_file_path(path, backend=backend, **kwargs) as slide_image:
            scaling = slide_image.get_scaling(mpp)
            slide_mpp = slide_image.mpp

            if limit_bounds:
                if rois is not None:
                    raise ValueError("Cannot use both `rois` and `limit_bounds` at the same time.")
                if backend == ImageBackend.AUTODETECT or backend == "AUTODETECT":
                    raise ValueError(
                        "Cannot use AutoDetect as backend and use limit_bounds at the same time. "
                        "This is related to issue #151. See https://github.com/NKI-AI/dlup/issues/151"
                    )

                offset, bounds = slide_image.slide_bounds
                offset = (int(scaling * offset[0]), int(scaling * offset[1]))
                size = int(bounds[0] * scaling), int(bounds[1] * scaling)
                _rois = [
                    (offset, size),
                ]

            else:
                slide_level_size = slide_image.get_scaled_size(scaling)
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
            **kwargs,
        )

    @overload
    def __getitem__(self, index: int) -> RegionFromWsiDatasetSample:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[RegionFromWsiDatasetSample]:
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[RegionFromWsiDatasetSample, list[RegionFromWsiDatasetSample]]:
        region_data = super().__getitem__(index)

        if isinstance(index, slice):
            # Convince mypy.
            if not isinstance(region_data, collections.abc.Sequence):
                raise ValueError("Expected a sequence of TileSample.")
            # We know that for slices, super().__getitem__ returns a sequence of TileSample
            return [self._process_tile_sample(data) for data in region_data]

        # We know that for int, super().__getitem__ returns a TileSample. Now we need to convince mypy.
        if not isinstance(region_data, dict):
            raise ValueError("Expected a TileSample.")
        return self._process_tile_sample(region_data)

    def _process_tile_sample(self, region_data: TileSample) -> RegionFromWsiDatasetSample:
        region_index = region_data["region_index"]
        starting_index = bisect.bisect_right(self._starting_indices, region_index) - 1
        grid_index = region_index - self._starting_indices[starting_index]
        grid_local_coordinates = np.unravel_index(grid_index, self.grids[starting_index][0].size)

        output: RegionFromWsiDatasetSample = {
            "image": region_data["image"],
            "coordinates": region_data["coordinates"],
            "mpp": region_data["mpp"],
            "path": region_data["path"],
            "region_index": region_data["region_index"],
            "labels": region_data["labels"],
            "annotations": region_data["annotations"],
            "grid_local_coordinates": (int(grid_local_coordinates[0]), int(grid_local_coordinates[1])),
            "grid_index": starting_index,
        }

        if self._transform:
            output = self._transform(output)

        return output


def parse_rois(rois: list[ROIType] | None, image_size: tuple[int, int], scaling: float = 1.0) -> list[ROIType]:
    if rois is None:
        return [((0, 0), image_size)]
    else:
        # Do some checks whether the ROIs are within the image
        origin_positive = [np.all(np.asarray(coords) > 0) for coords, size in rois]
        image_within_borders = [np.all((np.asarray(coords) + size) <= image_size) for coords, size in rois]
        if not origin_positive or not image_within_borders:
            raise ValueError(f"ROIs should be within image boundaries. Got {rois}.")

    _rois = []
    for coords, size in rois:
        _coords = (ceil(coords[0] * scaling), ceil(coords[1] * scaling))
        _size = (floor(size[0] * scaling), floor(size[1] * scaling))

        _rois.append((_coords, _size))

    return _rois


class TiledROIsSlideImageDataset(TiledWsiDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "`TiledROIsSlideImageDataset` is deprecated and will be removed in dlup v1.0. "
            "Use `TiledWsiDataset` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

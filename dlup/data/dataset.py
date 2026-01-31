# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bisect
import collections.abc
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Optional, Sequence, TypeVar, Union, overload
import json
import pathlib
import zipfile
import io

import fim
import numpy as np
import numpy.typing as npt
from dlup import BoundaryMode, SlideImage
from dlup._geometry import AnnotationRegion
from dlup._types import PathLike
from dlup.annotations import SlideAnnotations
from dlup.background import compute_masked_indices
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.utils.backends import ImageBackend

# Type aliases
MaskTypes = Union[SlideImage, npt.NDArray[np.int_], SlideAnnotations]
LabelType = Union[str, bool, int, float]
AnnotationData = dict[str, Any]
PointType = tuple[float, float]
BoundingBoxType = tuple[tuple[int, int], tuple[int, int]]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


@dataclass
class TilingConfig:
    """Configuration for tiling parameters.

    Parameters
    ----------
    mpp : Optional[float]
        Target microns per pixel for the tiles.
    tile_size : tuple[int, int]
        Size of each tile (width, height).
    tile_overlap : tuple[int, int]
        Overlap between tiles (width, height). Default is (0, 0).
    tile_mode : TilingMode
        Tiling mode (overflow, skip, etc.). Default is TilingMode.overflow.
    grid_order : GridOrder
        Order in which to iterate over the grid. Default is GridOrder.C (row-major).
    limit_bounds : bool
        Whether to limit tiling to the slide bounds. Default is True.
    random_sample_in_grid : bool
        Whether to randomly sample within each grid cell. Default is False.
    """

    mpp: Optional[float]
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int] = (0, 0)
    tile_mode: TilingMode = TilingMode.overflow
    grid_order: GridOrder = GridOrder.C
    limit_bounds: bool = True
    random_sample_in_grid: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration after initialization."""
        if self.tile_size[0] <= 0 or self.tile_size[1] <= 0:
            raise ValueError(f"tile_size must be > 0, got {self.tile_size}")
        if self.tile_overlap[0] < 0 or self.tile_overlap[1] < 0:
            raise ValueError(f"tile_overlap must be >= 0, got {self.tile_overlap}")
        if self.mpp is not None and self.mpp <= 0:
            raise ValueError(f"mpp must be > 0, got {self.mpp}")


@dataclass
class MaskConfig:
    """Configuration for masking tiles.

    Parameters
    ----------
    mask : Optional[MaskTypes]
        Mask to apply (can be SlideImage, numpy array, or SlideAnnotations).
    mask_threshold : float
        Threshold for mask filtering. Default is 0.0.
    crop : bool
        Whether to crop tiles at boundaries. Default is False.
    """

    mask: Optional[MaskTypes] = None
    mask_threshold: float = 0.0
    crop: bool = False


@dataclass
class ImageConfig:
    """Configuration for image backend and processing.

    Parameters
    ----------
    backend : ImageBackend
        Image backend to use. Default is ImageBackend.OPENSLIDE.
    apply_color_profile : bool
        Whether to apply color profile to images. Default is False.
    """

    backend: ImageBackend = ImageBackend.OPENSLIDE
    apply_color_profile: bool = False


@dataclass
class AnnotationConfig:
    """Configuration for annotations and labels.

    Parameters
    ----------
    annotations : Optional[SlideAnnotations]
        Annotations to include with tiles.
    labels : Optional[dict[str, Any]]
        Labels to attach to each tile.
    """

    annotations: Optional[SlideAnnotations] = None
    labels: Optional[dict[str, Any]] = None


@dataclass
class TileSample:
    """
    A sample from a dataset, representing a tile extracted from a slide image.
    """

    image: fim.Image
    coordinates: tuple[float, float]
    mpp: float
    path: PathLike
    region_index: int
    labels: Optional[dict[str, Any]] = None
    annotations: Optional[AnnotationRegion] = None


@dataclass
class RegionFromSlideDataset(TileSample):
    """
    A tile sample with additional information about its position in the grid.
    """

    grid_local_coordinates: tuple[int, int] = field(default_factory=lambda: (0, 0))
    grid_index: int = 0


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
    """
    A dataset that concatenates multiple datasets.
    """

    def __init__(self, datasets: Iterable[Dataset[T_co]]) -> None:
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"
        for dataset in self.datasets:
            if not hasattr(dataset, "__getitem__"):
                raise ValueError("ConcatDataset requires datasets to be indexable.")
        self.cumulative_sizes = self._compute_cumulative_sizes(self.datasets)

    @staticmethod
    def _compute_cumulative_sizes(datasets: list[Dataset[T_co]]) -> list[int]:
        cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            cumulative_sizes.append(total)
        return cumulative_sizes

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def index_to_dataset(self, index: int) -> tuple[Dataset[T_co], int]:
        if index < 0:
            if -index > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            index = len(self) + index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        sample_idx = index if dataset_idx == 0 else index - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx], sample_idx

    @overload
    def __getitem__(self, index: int) -> T_co: ...

    @overload
    def __getitem__(self, index: slice) -> list[T_co]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T_co, list[T_co]]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]
        dataset, sample_idx = self.index_to_dataset(index)
        return dataset[sample_idx]


class MaskMixin:
    """
    Mixin class for applying masks to datasets.
    """

    def _apply_mask(
        self,
        regions: Sequence[tuple[float, float, int, int, float]],
        mask: MaskTypes,
        mask_threshold: Optional[float],
        slide_image: SlideImage,
    ) -> npt.NDArray[np.int64]:
        masked_indices = compute_masked_indices(slide_image, mask, regions, mask_threshold)
        return masked_indices


class AnnotationMixin:
    """
    Mixin class for handling annotations in datasets.
    """

    _annotations: Optional[SlideAnnotations] = None


class LabelMixin:
    """
    Mixin class for handling labels in datasets.
    """

    _labels: Optional[dict[str, Any]] = None

    def _assign_labels(self) -> Optional[dict[str, Any]]:
        return self._labels


class SlideDataset(Dataset[TileSample], MaskMixin, AnnotationMixin, LabelMixin):
    """Dataset class for iterating over tiles extracted from a slide image.

    This dataset provides tile-level access to whole slide images with support for
    masking, annotations, and custom transformations. Configuration is provided
    through Pydantic config models for better validation and organization.

    Examples
    --------
    Basic usage with tiling configuration:

    >>> from dlup import SlideDataset, TilingConfig
    >>> tiling_config = TilingConfig(
    ...     mpp=0.5,
    ...     tile_size=(512, 512),
    ...     tile_overlap=(0, 0),
    ... )
    >>> dataset = SlideDataset.from_standard_tiling(
    ...     "path/to/slide.svs",
    ...     tiling_config=tiling_config,
    ... )

    With masking and annotations:

    >>> from dlup import TilingConfig, MaskConfig, AnnotationConfig
    >>> from dlup.annotations import SlideAnnotations
    >>>
    >>> tiling_config = TilingConfig(mpp=0.5, tile_size=(512, 512))
    >>> mask_config = MaskConfig(mask=mask_image, mask_threshold=0.5, crop=True)
    >>> annotation_config = AnnotationConfig(
    ...     annotations=SlideAnnotations.from_geojson("annotations.json"),
    ...     labels={"diagnosis": "cancer"},
    ... )
    >>>
    >>> dataset = SlideDataset.from_standard_tiling(
    ...     "path/to/slide.svs",
    ...     tiling_config=tiling_config,
    ...     mask_config=mask_config,
    ...     annotation_config=annotation_config,
    ... )

    With custom transform:

    >>> def my_transform(sample):
    ...     # Modify the sample
    ...     return sample
    >>>
    >>> dataset = SlideDataset.from_standard_tiling(
    ...     "path/to/slide.svs",
    ...     tiling_config=TilingConfig(mpp=0.5, tile_size=(512, 512)),
    ...     transform=my_transform,
    ... )
    """

    def __init__(
        self,
        path: PathLike,
        grid: tuple[Grid, tuple[int, int], float],
        mask_config: Optional[MaskConfig] = None,
        image_config: Optional[ImageConfig] = None,
        annotation_config: Optional[AnnotationConfig] = None,
        transform: Optional[Callable[[TileSample], TileSample]] = None,
        random_sample_in_grid: bool = False,
        **kwargs: Any,
    ):
        """Initialize a SlideDataset.

        Parameters
        ----------
        path : PathLike
            Path to the slide image file.
        grid : tuple[Grid, tuple[int, int], float]
            Grid configuration (Grid object, tile size, mpp).
        mask_config : Optional[MaskConfig]
            Configuration for masking tiles.
        image_config : Optional[ImageConfig]
            Configuration for image backend and processing.
        annotation_config : Optional[AnnotationConfig]
            Configuration for annotations and labels.
        transform : Optional[Callable[[TileSample], TileSample]]
            Transform function to apply to each sample.
        random_sample_in_grid : bool
            Whether to randomly sample within each grid cell. Default is False.
        **kwargs : Any
            Additional keyword arguments passed to SlideImage.
        """
        # Use defaults if not provided
        mask_config = mask_config or MaskConfig()
        image_config = image_config or ImageConfig()
        annotation_config = annotation_config or AnnotationConfig()

        self._path = path
        self._grid = grid
        self._regions = self._compute_regions()
        self._crop = mask_config.crop
        self._mask_threshold = mask_config.mask_threshold
        self._random_sample_in_grid = random_sample_in_grid
        self._annotations = annotation_config.annotations
        self._labels = annotation_config.labels
        self._transform = transform
        self._backend = image_config.backend
        self._apply_color_profile = image_config.apply_color_profile
        self.kwargs = kwargs

        # Set the base MPP on annotations if not already set
        if self._annotations and self._annotations.mpp is None:
            self._annotations.mpp = self.slide_image.mpp

        # Create views once at the dataset's target MPP for efficiency
        dataset_mpp = self._grid[2]  # All tiles have the same MPP
        self._image_view = self.slide_image.get_view_at_mpp(dataset_mpp)
        self._image_view.boundary_mode = BoundaryMode.crop if mask_config.crop else BoundaryMode.zero

        self._annotation_view = None
        if self._annotations:
            self._annotation_view = self._annotations.get_view_at_mpp(dataset_mpp)

        self._masked_indices: Optional[npt.NDArray[np.int64]] = None
        if mask_config.mask is not None:
            self._masked_indices = self._apply_mask(
                self._regions, mask_config.mask, mask_config.mask_threshold, self.slide_image
            )

    @property
    def grid(self) -> Grid:
        """Retrieve the grid used to generate the regions."""
        return self._grid[0]

    @property
    def tile_size(self) -> tuple[int, int]:
        """Retrieve the tile size used to generate the regions."""
        return self._grid[1]

    @property
    def mpp(self) -> float:
        """Retrieve the target microns per pixel used to generate the regions."""
        return self._grid[2]

    @property
    def slide_image(self) -> SlideImage:
        """Retrieve the SlideImage instance for this dataset."""
        return SlideImage.from_file_path(
            self._path,
            backend=self._backend,
            apply_color_profile=self._apply_color_profile,
            **self.kwargs,
        )

    def __len__(self) -> int:
        if self._masked_indices is not None:
            return len(self._masked_indices)
        return len(self._regions)

    @overload
    def __getitem__(self, index: int) -> TileSample: ...

    @overload
    def __getitem__(self, index: slice) -> list[TileSample]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[TileSample, list[TileSample]]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self._get_sample(i) for i in indices]
        return self._get_sample(index)

    def _get_sample(self, index: int) -> TileSample:
        region_index = self._get_region_index(index)
        x, y, w, h, mpp = self._regions[region_index]
        coordinates = (x, y)
        region_size = (w, h)

        if self._random_sample_in_grid:
            x_rand = random.uniform(x, min(x + w, self._image_view.size[0]))
            y_rand = random.uniform(y, min(y + h, self._image_view.size[1]))
            coordinates = (int(x_rand), int(y_rand))  # TODO: A float would work here too, but not for mypy

        # Read image using the pre-created view
        image = self._image_view.read_region(coordinates, region_size)

        # Load annotations using the pre-created view if available
        annotations = None
        if self._annotation_view:
            annotations = self._annotation_view.read_region(coordinates, region_size)

        labels = self._assign_labels()

        sample = TileSample(
            image=image,
            coordinates=coordinates,
            mpp=mpp,
            path=self._path,
            region_index=region_index,
            labels=labels,
            annotations=annotations,
        )

        if self._transform:
            sample = self._transform(sample)

        return sample

    def _get_region_index(self, index: int) -> int:
        if self._masked_indices is not None:
            return int(self._masked_indices[index])
        return index

    def _compute_regions(self) -> list[tuple[int, int, int, int, float]]:
        """Compute regions from grids."""
        regions = []
        grid, tile_size, grid_mpp = self._grid
        for coords in grid:
            regions.append(_coords_to_region(tile_size, grid_mpp, key="", coords=coords.tolist()))
        return regions

    def save_settings(self, path: PathLike) -> None:
        """Save the dataset settings to a dlup file.

        Parameters
        ----------
        path : PathLike
            Path to save the dataset settings to.
        """
        path = pathlib.Path(path)
        if path.suffix != ".dlup":
            path = path.with_suffix(".dlup")

        metadata = {
            "tile_size": self.tile_size,
            "mpp": self.mpp,
            "grid_coordinates": [c.tolist() for c in self.grid.coordinates],
            "mask_config": {
                "mask_threshold": self._mask_threshold,
                "crop": self._crop,
            },
        }

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            if self._masked_indices is not None:
                with io.BytesIO() as bio:
                    np.save(bio, self._masked_indices)
                    zf.writestr("indices.npy", bio.getvalue())

    @classmethod
    def from_settings_file(
        cls,
        path: PathLike,
        image_path: PathLike,
        annotation_config: Optional[AnnotationConfig] = None,
        image_config: Optional[ImageConfig] = None,
        **kwargs: Any,
    ) -> "SlideDataset":
        """Load a dataset from a settings file.

        Parameters
        ----------
        path : PathLike
            Path to the settings ZIP file.
        image_path : PathLike
            Path to the slide image.
        annotation_config : Optional[AnnotationConfig]
            Configuration for annotations.
        image_config : Optional[ImageConfig]
            Configuration for image backend.
        **kwargs : Any
            Additional arguments passed to SlideDataset and SlideImage.

        Returns
        -------
        SlideDataset
            The configured dataset.
        """
        path = pathlib.Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            metadata = json.loads(zf.read("metadata.json"))

            # Reconstruct grid
            coordinates = [np.array(c) for c in metadata["grid_coordinates"]]
            grid = Grid(coordinates)

            # Check if indices.npy exists in the archive
            masked_indices = None
            if "indices.npy" in zf.namelist():
                with io.BytesIO(zf.read("indices.npy")) as bio:
                    masked_indices = np.load(bio)

            # Reconstruct dataset
            grid_tuple = (grid, tuple(metadata["tile_size"]), metadata["mpp"])

            # Merge configs
            mask_config = MaskConfig(
                crop=metadata["mask_config"]["crop"], mask_threshold=metadata["mask_config"]["mask_threshold"]
            )

            # kwargs can override what we just passed if user desires, but typically we construct args here
            # We must be careful not to duplicate args.
            obj = cls(
                path=image_path,
                grid=grid_tuple,
                mask_config=mask_config,
                image_config=image_config,
                annotation_config=annotation_config,
                **kwargs,
            )

            if masked_indices is not None:
                obj._masked_indices = masked_indices

            return obj

    def filter(self, func: Callable[[TileSample], bool], progress=False) -> "SlideDataset":
        """Filter the dataset based on a condition.

        Parameters
        ----------
        func : Callable[[TileSample], bool]
            Function that takes a TileSample and returns True if it should be kept.
        progress : bool
            If true, will add a tqdm progress bar.

        Returns
        -------
        SlideDataset
            A new dataset instance with the filtered indices.
        """
        valid_indices = []

        if progress:
            import tqdm

            progress_bar = tqdm.tqdm
        else:
            progress_bar = lambda x: x

        for i in progress_bar(range(len(self))):
            sample = self._get_sample(i)
            if func(sample):
                # We want to store the region index (the index into self._regions)
                valid_indices.append(sample.region_index)

        # Create a new instance (shallow copy of self)
        import copy

        new_dataset = copy.copy(self)
        new_dataset._masked_indices = np.array(valid_indices, dtype=np.int64)
        return new_dataset

    @classmethod
    def from_standard_tiling(
        cls,
        path: PathLike,
        tiling_config: TilingConfig,
        mask_config: Optional[MaskConfig] = None,
        image_config: Optional[ImageConfig] = None,
        annotation_config: Optional[AnnotationConfig] = None,
        transform: Optional[Callable[[TileSample], TileSample]] = None,
        **kwargs: Any,
    ) -> "SlideDataset":
        """Create a SlideDataset with standard tiling configuration.

        Parameters
        ----------
        path : PathLike
            Path to the slide image file.
        tiling_config : TilingConfig
            Configuration for tiling parameters.
        mask_config : Optional[MaskConfig]
            Configuration for masking tiles.
        image_config : Optional[ImageConfig]
            Configuration for image backend and processing.
        annotation_config : Optional[AnnotationConfig]
            Configuration for annotations and labels.
        transform : Optional[Callable[[TileSample], TileSample]]
            Transform function to apply to each sample.
        **kwargs : Any
            Additional keyword arguments passed to SlideImage.

        Returns
        -------
        SlideDataset
            A configured dataset instance.

        Examples
        --------
        >>> from dlup.data.dataset import SlideDataset, TilingConfig
        >>> tiling_config = TilingConfig(
        ...     mpp=0.5,
        ...     tile_size=(512, 512),
        ...     tile_overlap=(0, 0),
        ... )
        >>> dataset = SlideDataset.from_standard_tiling(
        ...     "path/to/slide.svs",
        ...     tiling_config=tiling_config,
        ... )
        """
        # Use defaults if not provided
        mask_config = mask_config or MaskConfig()
        image_config = image_config or ImageConfig()
        annotation_config = annotation_config or AnnotationConfig()

        # Extract mpp from tiling_config
        mpp = tiling_config.mpp

        with SlideImage.from_file_path(path, backend=image_config.backend, **kwargs) as slide_image:
            scaling = slide_image.get_scaling(mpp)
            slide_mpp = slide_image.mpp

            if tiling_config.limit_bounds:
                offset, size = slide_image.get_scaled_slide_bounds(scaling)
            else:
                size = slide_image.get_scaled_size(scaling, limit_bounds=False)
                offset = (0, 0)

        grid = Grid.from_tiling(
            offset=offset,
            size=size,
            tile_size=tiling_config.tile_size,
            tile_overlap=tiling_config.tile_overlap,
            mode=tiling_config.tile_mode,
            order=tiling_config.grid_order,
        )
        grid_mpp = mpp if mpp else slide_mpp

        return cls(
            path=path,
            grid=(grid, tiling_config.tile_size, grid_mpp),
            mask_config=mask_config,
            image_config=image_config,
            annotation_config=annotation_config,
            transform=transform,
            random_sample_in_grid=tiling_config.random_sample_in_grid,
            **kwargs,
        )


def _coords_to_region(
    tile_size: tuple[int, int], target_mpp: float, key: str, coords: tuple[int, int]
) -> tuple[int, int, int, int, float]:
    """Return the necessary tuple that represents a region."""
    return *coords, *tile_size, target_mpp

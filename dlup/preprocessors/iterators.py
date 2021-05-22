# coding=utf-8
# Copyright (c) dlup contributors
import itertools
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
from numpy.typing import ArrayLike

from dlup.slide import Slide, _ensure_array


@dataclass
class DataclassMapping(Mapping):
    def __iter__(self):
        return iter(asdict(self))

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return asdict(self)[item]

    def keys(self):
        return self.__dict__.keys()


@dataclass
class IteratorInfo(DataclassMapping):
    indices: List[range]
    tile_size: np.ndarray  # TODO: ArrayLike from numpy.typing
    level: int
    level_shape: np.ndarray
    resample: Union[bool, int]
    num_tiles: np.ndarray
    requested_mpp: Union[float, Tuple[float, float], None]
    effective_mpp: Union[float, Tuple[float, float], None]
    requested_mag: Optional[float]
    effective_mag: Optional[float]
    stride: np.ndarray  # TODO: ArrayLike from numpy.typing
    overflow: np.ndarray  # TODO: ArrayLike from numpy.typing
    region_scale: np.ndarray
    downsample_factor: np.ndarray


class TileIterator:
    def __init__(
        self,
        slide: Slide,
        tile_size: ArrayLike,
        tile_overlap: ArrayLike,
        start_x: int = 0,
        start_y: int = 0,
        end_x: Optional[int] = None,
        end_y: Optional[int] = None,
        magnification: Optional[float] = None,
        mpp: Union[float, ArrayLike] = None,
        border_mode: Optional[str] = None,
    ) -> None:
        tile_size = _ensure_array(tile_size)
        tile_overlap = _ensure_array(tile_overlap)

        self.slide = slide
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.magnification = magnification
        self.mpp = mpp
        self.border_mode = border_mode

        # TODO: This needs checking if it is the correct order.
        if not end_x:
            end_x = self.slide.width
        if not end_y:
            end_y = self.slide.height

        self.iter_info = self.__build_iterator_info(
            tile_size,
            tile_overlap,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            magnification=magnification,
            mpp=mpp,
            border_mode=border_mode,
        )

    @property
    def num_tiles(self):
        return sum(1 for _ in self.grid_iterator())

    def grid_iterator(self):
        for idx in itertools.product(*self.iter_info.indices):
            idx = np.array(idx)
            start_coords = idx * self.iter_info.stride
            overflow_value = self.iter_info.overflow
            overflow = idx == (self.iter_info.num_tiles - 1)
            overflow_ = np.array([overflow_value[idx] if _ else 0 for idx, _ in enumerate(overflow)])
            region = Region(
                coordinates=start_coords,
                level=self.iter_info.level,
                size=self.iter_info.tile_size,
                overflow=overflow_,
                target_tile_size=self.tile_size,
                idx=idx,
            )
            yield region

    def __iter__(self):
        region_iterator = self.grid_iterator()
        for region in region_iterator:
            yield Tile(
                slide=self.slide,
                region=region,
                resample=self.iter_info.resample,
            )

    def __build_iterator_info(
        self,
        tile_size: ArrayLike,
        tile_overlap: ArrayLike,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        magnification: Optional[float] = None,
        mpp: Union[float, ArrayLike] = None,
        border_mode: Optional[str] = "skip",
    ) -> IteratorInfo:

        if not np.all(tile_overlap < tile_size):
            raise ValueError(f"`tile_overlap` has to be smaller than `tile_size`. Got {tile_overlap} and {tile_size}.")

        if magnification is not None and mpp is not None:
            raise ValueError(
                f"Either a magnification level has to be set or a resolution has to be set. "
                f"Got {magnification} and {mpp}."
            )

        region_size_0 = np.array([end_x - start_x, end_y - start_y])
        if np.any(region_size_0 <= 0):
            raise ValueError("Size of region must be positive, as defined by start_x, start_y, end_x, end_y.")

        # If mpp or magnitude is given, we need to figure out what the downsampling factor is compared to the level 0.
        effective_mpp = None
        effective_mag = None
        downsample_factor = 1.0

        if mpp:
            shape, downsample_factor, effective_mpp = self.slide.shape_at_mpp(mpp, exact=False)
            effective_mag = self.slide.magnification / downsample_factor
        if magnification:
            shape, downsample_factor, effective_mag = self.slide.shape_at_magnification(magnification, exact=False)
            effective_mpp = self.slide.mpp / downsample_factor

        # Based on the downsampling factor, we need to find the best level to downsample from.
        level = self.slide.get_best_level_for_downsample(downsample_factor)

        # Find the level named tuple that corresponds to this label
        # and has the downsample information available.
        corresponding_level = self.slide.levels[level]

        # This is a quick check if the corresponding level is actually in there, otherwise we need to select the next
        # level. This is not implemented for now as it is unclear if this will occur, hence the check.
        if corresponding_level.level != level:
            raise NotImplementedError("Current implementation assumes levels are consecutive in the WSI.")

        # The region size is given in level 0, we need to compute what this means in the level we selected.
        region_scale = np.asarray(corresponding_level.shape) / self.slide.shape

        # This is the corresponding level region size.
        corresponding_region_size = region_size_0 * region_scale
        subsampled_region_size = corresponding_region_size * downsample_factor
        resample = False
        if downsample_factor != 1.0:
            resample = PIL.Image.LANCZOS

        # Compute the grid.
        stride = tile_size - tile_overlap

        # Same thing as computing the output shape of a convolution with padding zero and
        # specified stride.
        num_tiles = (subsampled_region_size - tile_size) / stride + 1

        if border_mode == "crop":
            num_tiles = np.ceil(num_tiles).astype(int)
            tiled_size = (num_tiles - 1) * stride + tile_size
            overflow = tiled_size - subsampled_region_size
        elif border_mode == "skip":
            num_tiles = np.floor(num_tiles).astype(int)
            overflow = np.asarray((0, 0))
        else:
            raise ValueError(f"`border_mode` has to be one of `crop` or `skip`. Got {border_mode}.")

        indices = [range(0, _) for _ in num_tiles]

        return IteratorInfo(
            indices=indices,
            tile_size=tile_size / downsample_factor,
            level=level,
            level_shape=corresponding_level.shape,
            resample=resample,
            requested_mpp=mpp,
            requested_mag=magnification,
            effective_mpp=effective_mpp,
            effective_mag=effective_mag,
            num_tiles=num_tiles,
            stride=stride / downsample_factor,
            overflow=overflow,
            region_scale=region_scale,
            downsample_factor=downsample_factor,
        )


@dataclass
class Region(DataclassMapping):
    coordinates: tuple  # The starting coordinates in level 0!
    level: int  # The level this region is represented
    size: tuple  # The size of the region in level 0
    overflow: np.ndarray  # The number of pixels the region extends beyond the slide
    target_tile_size: tuple
    idx: list

    @property
    def bbox(self):
        return np.hstack([self.coordinates, self.target_tile_size])


@dataclass
class Tile:
    slide: Slide
    region: Region
    resample: Optional[np.ndarray] = None
    exact: bool = True

    # TODO: Not yet used. Will be useful if exact mode is not being used.
    mpp: Optional[float] = None
    magnification: Optional[float] = None

    @property
    def pixel_coordinates(self):
        return np.floor(self.region.coordinates).astype(int)

    @property
    def pixel_size(self):
        pixel_size = np.ceil(self.region.coordinates + self.region.size - self.pixel_coordinates).astype(int)  # noqa
        return pixel_size

    @property
    def level_bbox(self):
        return np.hstack([self.pixel_coordinates, self.pixel_size])

    @property
    def tile(self):
        # Since we are interested in extracting a region in the real domain
        # but openslide accepts only integers we first extract
        # a slightly larger tile (max 2 pixels per dimension).
        tile = self.slide.get_tile(
            self.pixel_coordinates,
            self.pixel_size,
            self.region.level,
        )

        # If mode is exact, tiles are also interpolated in fraction pixels
        box = None
        if self.exact:
            fractional_coordinates = self.region.coordinates - self.pixel_coordinates
            box = (*fractional_coordinates, *self.region.size)
        else:
            warnings.warn("Mode is not exact. Effective mpp and effective magnification are not fully correct.")

        # Then we use PIL to extract the exact subtile
        tile = tile.resize(self.region.target_tile_size, resample=self.resample, box=box)

        if any(self.region.overflow):
            crop_bbox = [0, 0] + (tile.size - self.region.overflow).tolist()
            tile = tile.crop(crop_bbox)

        return tile

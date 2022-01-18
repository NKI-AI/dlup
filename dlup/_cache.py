# coding=utf-8
# Copyright (c) dlup contributors
import abc
import functools
import pathlib
from typing import Tuple, Union
from dlup._region import BoundaryMode
from dataclasses import dataclass
import numpy as np
import PIL
from enum import Enum
import dlup
from dlup.utils.types import GenericNumber, PathLike

from dlup.utils.imports import PYVIPS_AVAILABLE

if PYVIPS_AVAILABLE:
    from dlup.writers import TiffCompression, TiffImageWriter

IMAGE_CACHE = None


def is_cache_available():
    return IMAGE_CACHE is not None


def current_cache():
    try:
        return _CACHES_AVAILABLE[IMAGE_CACHE]
    except KeyError:
        raise RuntimeError(
            f"{IMAGE_CACHE} is not a valid value for image cache. Valid options are {_CACHES_AVAILABLE.keys()}"
        )


class AbstractImageCache(abc.ABC):
    @property
    @classmethod
    @abc.abstractmethod
    def writable(cls):
        """"""

    def __init__(self, original_filename: PathLike):
        self._original_filename = pathlib.Path(original_filename)

    @property
    @abc.abstractmethod
    def cache_lock(self):
        """Return the cache lock"""

    @abc.abstractmethod
    def read_cached_region(
        self,
        location: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        mpp: float,
        size: Union[np.ndarray, Tuple[int, int]],
    ) -> PIL.Image.Image:
        """..."""

    # @abc.abstractmethod
    # def write_cached_region(
    #     self,
    #     region: PIL.Image.Image,
    #     location: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
    #     mpp: float,
    #     size: Union[np.ndarray, Tuple[int, int]],
    # ):
    #     """Write region to cache"""


class IdentityImageCache(AbstractImageCache):
    cache_lock = None
    writable = False

    def read_cached_region(
        self,
        location: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        mpp: float,
        size: Union[np.ndarray, Tuple[int, int]],
    ) -> PIL.Image.Image:
        """..."""
        return None


class TiffImageCache(AbstractImageCache):
    writable = False

    def __init__(self, original_filename: PathLike):
        super().__init__(original_filename)
        self._cache_directory = None
        self.__image_cache = {}

    @property
    def cache_directory(self):
        return self._cache_directory

    @cache_directory.setter
    def cache_directory(self, cache_directory: PathLike):
        cache_directory = pathlib.Path(cache_directory)
        if not cache_directory.is_dir():
            raise NotADirectoryError(
                f"{cache_directory} is not a directory. It is assumed that the cached files,"
                f"reside in this directory. The format is <original_filename>.mpp.tiff"
            )
        self._cache_directory = cache_directory

    def read_cached_region(
        self,
        location: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        mpp: float,
        size: Union[np.ndarray, Tuple[int, int]],
    ) -> PIL.Image:
        """ "
        Should raise a ValueError if it does not exist, and RuntimeError if something goes wrong

        """
        slide_image = self.get_cache_for_mpp(mpp)
        # Now we need to read the same image, but our scaling is different.
        scaling = slide_image.get_scaling(mpp)
        if scaling != 1.0:
            raise RuntimeError(f"No rescaling should be required!")
        return slide_image.read_region(location, 1.0, size)

    @property
    def cache_lock(self):
        return None

    def get_cache_for_mpp(self, mpp: float):
        cache_filename = (self.cache_directory / self._original_filename.name).with_suffix(f".mpp-{mpp}.tiff")
        if not cache_filename.is_file():
            return None

        if cache_filename.name not in self.__image_cache:
            self.__image_cache[cache_filename.name] = dlup.SlideImage.from_file_path(cache_filename)
        return self.__image_cache[cache_filename.name]

    def close(self):
        """Close the underlying images."""
        [_.close() for _ in self.__image_cache.values()]


def create_tiff_cache(slide_image, grid, mpp, tile_size, filename, tiff_tile_size, pyramid):
    scaling: float = slide_image.mpp / mpp
    region_view = slide_image.get_scaled_view(scaling)
    region_view.boundary_mode = BoundaryMode.crop

    grid_offset = np.asarray([_.min() for _ in grid.coordinates])

    def _local_iterator():
        region_size: Tuple[int, int] = tile_size
        for coordinates in grid:
            yield coordinates - grid_offset, region_view.read_region(coordinates, region_size)

    slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))

    writer = TiffImageWriter(
        mpp=(mpp, mpp),
        size=slide_level_size - grid_offset,
        tile_width=tiff_tile_size[0],
        tile_height=tiff_tile_size[1],
        pyramid=pyramid,
        compression=TiffCompression.DEFLATE,
        quality=100,
        bit_depth=8,
        silent=False,
    )

    writer.from_iterator(_local_iterator(), filename, total=len(grid))


def image_cache(func):
    """
    Decorated to wrap a read_region function. Can be used for both non-writable caches such as tiff, which
    require to be pre-generated or other caches such as individual .png's or MemCached. Some of these require a
    lock, others to do not. For instance `threading.Lock()`.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cacher = self._cacher
        location, mpp, tile_size = self.region_hash(*args, **kwargs)
        lock = None if not cacher.writable else cacher.cache_lock

        if lock:
            with self.cache_lock:
                region = cacher.read_cached_region(location, mpp, tile_size)
        else:
            region = cacher.read_cached_region(location, mpp, tile_size)

        if region:
            return region

        # TODO: this needs to get it the old fashioned way!
        # We didn't manage to get a cached version to get it from the region.
        # Let's try to write it.
        v = func(self, *args, **kwargs)

        if cacher.writable:
            try:
                if lock:
                    with cacher.cache_lock:
                        cacher.write_cached_region(v, location, mpp, tile_size)
                else:
                    cacher.write_cached_region(v, location, mpp, tile_size)

            except ValueError:
                pass  # Cannot do this, just read the original region.
            except RuntimeError as exception:
                # For some reason we cannot read this region.
                raise RuntimeError(
                    f"Had a cache KeyError while trying to store location {location}, mpp {mpp} and tile_size {tile_size}."
                ) from exception
        return v

    return wrapper


_CACHES_AVAILABLE = {None: IdentityImageCache, "TIFF": TiffImageCache}
ImageCacher = current_cache()

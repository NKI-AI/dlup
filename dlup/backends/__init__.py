# coding=utf-8
# Copyright (c) dlup contributors
import os
import pathlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import PIL
import pyvips
import tifffile

import openslide
from dlup import UnsupportedSlideError

from ._openslide import OpenSlideSlide
from ._openslide import open_slide as open_slide_openslide
from ._pyvips import PyVipsSlide
from ._pyvips import open_slide as open_slide_pyvips
from ._tifffile import TifffileSlide
from ._tifffile import open_slide as open_slide_tifffile
from .common import AbstractSlideBackend


@lru_cache(maxsize=None)
def autodetect_backend(filename: os.PathLike) -> AbstractSlideBackend:
    """
    Try to read the file in consecutive order of pyvips, openslide, tifffile,
    by trying to a tile at the lowest resolution.
    The results are cached.

    Parameters
    ----------
    filename

    Returns
    -------

    """
    # Can try to be a bit more complex by first checking the path
    filename = pathlib.Path(filename)
    # OpenSlide cannot parse the mpp of tiff's
    if filename.suffix in [".tif", ".tiff"]:
        try:
            return _try_pyvips(filename)
        except UnsupportedSlideError:
            pass
        try:
            return _try_tifffile(filename)
        except UnsupportedSlideError:
            raise UnsupportedSlideError(f"Cannot read {filename} with pyvips or tifffile.")

    try:
        return _try_openslide(filename)
    except UnsupportedSlideError:
        pass

    try:
        return _try_pyvips(filename)
    except UnsupportedSlideError:
        raise UnsupportedSlideError(f"Cannot read {filename} with pyvips or openslide.")


def _try_openslide(filename: os.PathLike) -> AbstractSlideBackend:
    try:
        slide = OpenSlideSlide(filename)
        size = np.clip(0, 256, slide.level_dimensions[slide.level_count - 1]).tolist()
        slide.read_region((0, 0), slide.level_count - 1, size)
        return OpenSlideSlide(filename)
    except (openslide.OpenSlideUnsupportedFormatError, PIL.UnidentifiedImageError):
        raise UnsupportedSlideError(f"Cannot read {filename} with openslide.")


def _try_pyvips(filename: os.PathLike) -> AbstractSlideBackend:
    try:
        slide = PyVipsSlide(filename)
        size = np.clip(0, 256, slide.level_dimensions[slide.level_count - 1]).tolist()
        slide.read_region((0, 0), slide.level_count - 1, size)
        return PyVipsSlide(filename)
    except pyvips.error.Error:
        raise UnsupportedSlideError(f"Cannot read {filename} with pyvips.")


def _try_tifffile(filename: os.PathLike) -> AbstractSlideBackend:
    try:
        slide = TifffileSlide(filename)
        size = np.clip(0, 256, slide.level_dimensions[slide.level_count - 1]).tolist()
        slide.read_region((0, 0), slide.level_count - 1, size)
        return TifffileSlide(filename)
    except tifffile.tifffile.TiffFileError:
        raise UnsupportedSlideError(f"Cannot read {filename} with tifffile.")


@dataclass
class ImageBackends:
    OPENSLIDE: Callable = OpenSlideSlide
    PYVIPS: Callable = PyVipsSlide
    TIFFFILE: Callable = TifffileSlide
    AUTODETECT: Callable = autodetect_backend

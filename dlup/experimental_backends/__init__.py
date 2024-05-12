# Copyright (c) dlup contributors
from __future__ import annotations

import pathlib
from enum import Enum
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import openslide
import PIL
import pyvips
import tifffile

from dlup import UnsupportedSlideError
from dlup.backends.tifffile_backend import TifffileSlide
from dlup.types import PathLike

from ..backends.common import AbstractSlideBackend
from .openslide_backend import OpenSlideSlide
from .pyvips_backend import PyVipsSlide


@lru_cache(maxsize=None)
def autodetect_backend(filename: PathLike) -> AbstractSlideBackend:
    """
    Try to read the file in consecutive order of pyvips, openslide, tifffile,
    by trying to a tile at the lowest resolution.
    The results are cached, so the test is only performed once per filename.

    Parameters
    ----------
    filename

    Returns
    -------
    AbstractSlideBackend
        A backend which is able to read the image.
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


def _try_openslide(filename: PathLike) -> OpenSlideSlide:
    """
    Attempt to read the slide with openslide. Will open the slide and extract a region at the highest level.

    Parameters
    ----------
    filename : PathLike

    Returns
    -------
    OpenSlideSlide
    """
    try:
        slide = OpenSlideSlide(filename)
        size = np.clip(0, 256, slide.level_dimensions[slide.level_count - 1]).tolist()
        slide.read_region((0, 0), slide.level_count - 1, size)
        slide.close()
        return OpenSlideSlide(filename)
    except (openslide.OpenSlideUnsupportedFormatError, PIL.UnidentifiedImageError):
        raise UnsupportedSlideError(f"Cannot read {filename} with openslide.")


def _try_pyvips(filename: PathLike) -> PyVipsSlide:
    """
    Attempt to read the slide with pyvips. Will open the slide and extract a region at the highest level.

    Parameters
    ----------
    filename : PathLike

    Returns
    -------
    PyVipsSlide
    """
    try:
        slide = PyVipsSlide(filename)
        size = np.clip(0, 256, slide.level_dimensions[slide.level_count - 1]).tolist()
        slide.read_region((0, 0), slide.level_count - 1, size)
        slide.close()
        return PyVipsSlide(filename)
    except pyvips.error.Error:
        raise UnsupportedSlideError(f"Cannot read {filename} with pyvips.")


def _try_tifffile(filename: PathLike) -> TifffileSlide:
    """
    Attempt to read the slide with tifffile. Will open the slide and extract a region at the highest level.

    Parameters
    ----------
    filename : PathLike

    Returns
    -------
    TifffileSlide
    """
    try:
        slide = TifffileSlide(filename)
        size = np.clip(0, 256, slide.level_dimensions[slide.level_count - 1]).tolist()
        slide.read_region((0, 0), slide.level_count - 1, size)
        slide.close()
        return TifffileSlide(filename)
    except tifffile.tifffile.TiffFileError:
        raise UnsupportedSlideError(f"Cannot read {filename} with tifffile.")


class ImageBackend(Enum):
    """Available image experimental_backends."""

    OPENSLIDE: Callable[[PathLike], OpenSlideSlide] = OpenSlideSlide
    PYVIPS: Callable[[PathLike], PyVipsSlide] = PyVipsSlide
    TIFFFILE: Callable[[PathLike], TifffileSlide] = TifffileSlide
    AUTODETECT: Callable[[PathLike], AbstractSlideBackend] = autodetect_backend

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)

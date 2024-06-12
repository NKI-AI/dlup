# Copyright (c) dlup contributors
from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from dlup.backends.openslide_backend import OpenSlideSlide
from dlup.backends.tifffile_backend import TifffileSlide
from dlup.types import PathLike

from .pyvips_backend import PyVipsSlide


class ImageBackend(Enum):
    """Available image experimental_backends."""

    OPENSLIDE: Callable[[PathLike], OpenSlideSlide] = OpenSlideSlide
    PYVIPS: Callable[[PathLike], PyVipsSlide] = PyVipsSlide
    TIFFFILE: Callable[[PathLike], TifffileSlide] = TifffileSlide

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)

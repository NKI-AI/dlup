# Copyright (c) dlup contributors
"""Utilities to handle backends."""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from dlup._types import PathLike


class ImageBackend(Enum):
    """Available image experimental_backends."""

    from dlup.backends.openslide_backend import OpenSlideSlide
    from dlup.backends.pyvips_backend import PyVipsSlide
    from dlup.backends.tifffile_backend import TifffileSlide

    OPENSLIDE: Callable[[PathLike], OpenSlideSlide] = OpenSlideSlide
    PYVIPS: Callable[[PathLike], PyVipsSlide] = PyVipsSlide
    TIFFFILE: Callable[[PathLike], TifffileSlide] = TifffileSlide

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)

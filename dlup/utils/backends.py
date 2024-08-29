# Copyright (c) dlup contributors
"""Utilities to handle backends."""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from dlup._types import PathLike
from dlup.utils.imports import AIOHTTP_AVAILABLE


class ImageBackend(Enum):
    """Available image experimental_backends."""

    from dlup.backends.deepzoom_backend import DeepZoomSlide
    from dlup.backends.openslide_backend import OpenSlideSlide
    from dlup.backends.pyvips_backend import PyVipsSlide
    from dlup.backends.tifffile_backend import TifffileSlide

    if AIOHTTP_AVAILABLE:
        from dlup.backends.slidescore_backend import SlideScoreSlide

    OPENSLIDE: Callable[[PathLike], OpenSlideSlide] = OpenSlideSlide
    PYVIPS: Callable[[PathLike], PyVipsSlide] = PyVipsSlide
    TIFFFILE: Callable[[PathLike], TifffileSlide] = TifffileSlide
    DEEPZOOM: Callable[[PathLike], DeepZoomSlide] = DeepZoomSlide
    if AIOHTTP_AVAILABLE:
        SLIDESCORE: Callable[[PathLike], SlideScoreSlide] = SlideScoreSlide

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)

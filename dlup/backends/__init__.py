# Copyright (c) dlup contributors
from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from dlup._types import PathLike
from dlup.utils.imports import AIOHTTP_AVAILABLE

from .deepzoom_backend import DeepZoomSlide as DeepZoomSlide  # noqa: F401
from .openslide_backend import OpenSlideSlide as OpenSlideSlide  # noqa: F401
from .pyvips_backend import PyVipsSlide as PyVipsSlide  # noqa: F401
from .tifffile_backend import TifffileSlide as TifffileSlide  # noqa: F401

if AIOHTTP_AVAILABLE:
    from .slidescore_backend import SlideScoreSlide as SlideScoreSlide  # noqa: F401


class ImageBackend(Enum):
    """Available image experimental_backends."""

    DEEPZOOM: Callable[[PathLike], DeepZoomSlide] = DeepZoomSlide
    OPENSLIDE: Callable[[PathLike], OpenSlideSlide] = OpenSlideSlide
    PYVIPS: Callable[[PathLike], PyVipsSlide] = PyVipsSlide
    TIFFFILE: Callable[[PathLike], TifffileSlide] = TifffileSlide
    if AIOHTTP_AVAILABLE:
        SLIDESCORE: Callable[[PathLike], SlideScoreSlide] = SlideScoreSlide

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)

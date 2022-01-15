# coding=utf-8
# Copyright (c) dlup contributors
"""Top-level package for dlup."""

from ._exceptions import DlupError, DlupUnsupportedSlideError


__author__ = """dlup contributors"""
__email__ = "j.teuwen@nki.nl"
__version__ = "0.2.0"


from ._image import CachedSlideImage, SlideImage
from ._region import BoundaryMode, RegionView

# __all__ = ("SlideImage", "CachedSlideImage", "RegionView")

IMAGE_CACHE = None

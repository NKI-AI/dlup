# Copyright (c) dlup contributors"""Top-level package for dlup."""

from ._exceptions import UnsupportedSlideError
from ._image import SlideImage
from ._region import BoundaryMode, RegionView

__author__ = """dlup contributors"""
__email__ = "j.teuwen@nki.nl"
__version__ = "0.3.29"

__all__ = ("SlideImage", "RegionView", "UnsupportedSlideError", "BoundaryMode")

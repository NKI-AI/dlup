# coding=utf-8
# Copyright (c) DLUP Contributors
"""Top-level package for DLUP."""

from ._exceptions import DLUPError, DLUPUnsupportedSlideError
from ._image import SlideImage, SlideImageTiledRegionView
from ._region import RegionView

__author__ = """DLUP Contributors"""
__email__ = "j.teuwen@nki.nl"
__version__ = "0.1.0"


__all__ = ("SlideImage", "RegionView")

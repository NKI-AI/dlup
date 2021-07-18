# coding=utf-8
# Copyright (c) dlup contributors
"""Top-level package for dlup."""

from ._exceptions import DlupError, DlupUnsupportedSlideError
from ._image import SlideImage, SlideImageTiledRegionView
from ._region import RegionView

__author__ = """dlup contributors"""
__email__ = "j.teuwen@nki.nl"
__version__ = "0.2.0-dev2"

__all__ = ("SlideImage", "RegionView", "SlideImageTiledRegionView")

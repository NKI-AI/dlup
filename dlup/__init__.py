# Copyright (c) dlup contributors

import logging

from ._exceptions import UnsupportedSlideError
from ._image import SlideImage
from ._region import BoundaryMode, RegionView
from .annotations import AnnotationType, WsiAnnotations

pyvips_logger = logging.getLogger("pyvips")
pyvips_logger.setLevel(logging.CRITICAL)

__author__ = """dlup contributors"""
__email__ = "j.teuwen@nki.nl"
__version__ = "0.5.2"

__all__ = ("SlideImage", "WsiAnnotations", "AnnotationType", "RegionView", "UnsupportedSlideError", "BoundaryMode")

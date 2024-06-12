# Copyright (c) dlup contributors

import logging

from ._exceptions import UnsupportedSlideError
from ._image import SlideImage
from ._region import BoundaryMode, RegionView

pyvips_logger = logging.getLogger("pyvips")
pyvips_logger.setLevel(logging.CRITICAL)

__author__ = """dlup contributors"""
__email__ = "j.teuwen@nki.nl"
__version__ = "0.4.0"

__all__ = ("SlideImage", "RegionView", "UnsupportedSlideError", "BoundaryMode")

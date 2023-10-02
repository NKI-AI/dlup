from __future__ import annotations

import numpy as np
from numpy import typing as npt

from dlup import SlideImage
from dlup.annotations import WsiAnnotations

MaskTypes = SlideImage | npt.NDArray[np.int_] | WsiAnnotations

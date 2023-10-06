# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations

import os
from typing import Iterable, Union

import numpy as np
import numpy.typing as npt
from dlup.annotations import WsiAnnotations
from dlup import SlideImage

string_classes = (str, bytes)
PathLike = Union[str, os.PathLike]
GenericNumber = Union[int, float]
GenericNumberArray = Union[npt.NDArray[np.int_ | np.float_], Iterable[GenericNumber]]
GenericFloatArray = Union[npt.NDArray[np.float_], Iterable[float]]
GenericIntArray = Union[npt.NDArray[np.int_], Iterable[int]]
MaskTypes = Union[SlideImage, npt.NDArray[np.int_], WsiAnnotations]

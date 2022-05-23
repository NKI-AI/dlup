# coding=utf-8
# Copyright (c) dlup contributors
import os
from typing import Iterable, Union

import numpy as np

string_classes = (str, bytes)
PathLike = Union[str, os.PathLike]
GenericNumber = Union[int, float]
GenericNumberArray = Union[np.ndarray, Iterable[GenericNumber]]
GenericFloatArray = Union[np.ndarray, Iterable[float]]
GenericIntArray = Union[np.ndarray, Iterable[int]]

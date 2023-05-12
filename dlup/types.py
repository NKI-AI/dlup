# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations

import os
from typing import Iterable, Union

import numpy.typing as npt

string_classes = (str, bytes)
PathLike = Union[str, os.PathLike]
GenericNumber = Union[int, float]
GenericNumberArray = Union[npt.NDArray, Iterable[GenericNumber]]
GenericFloatArray = Union[npt.NDArray, Iterable[float]]
GenericIntArray = Union[npt.NDArray, Iterable[int]]

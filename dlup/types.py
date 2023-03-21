# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations

import os
from typing import Iterable, Union
import dlup.annotations

import numpy.typing as npt

string_classes = (str, bytes)
PathLike = str | os.PathLike
GenericNumber = int | float
GenericNumberArray = npt.NDArray | Iterable[GenericNumber]
GenericFloatArray = npt.NDArray | Iterable[float]
GenericIntArray = npt.NDArray | Iterable[int]
Size = tuple[int, int]
FloatSize = tuple[float, float]
Coordinates = tuple[GenericNumber, GenericNumber]

Box = tuple[Coordinates, Coordinates]


ROI = tuple[Coordinates, Size]

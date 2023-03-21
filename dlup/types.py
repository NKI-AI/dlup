# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations

import os
from typing import Iterable, Union

import numpy.typing as npt

string_classes = (str, bytes)
PathLike = str | os.PathLike
GenericNumber = int | float
GenericNumberArray = np.ndarray | Iterable[GenericNumber]
GenericFloatArray = np.ndarray | Iterable[float]
GenericIntArray = np.ndarray | Iterable[int]
Size = tuple[int, int]
Coordinates = tuple[GenericNumber, GenericNumber]
ROI = tuple[Coordinates, Size]
PointOrPolygon = Union["dlup.annotations.Point", "dlup.annotations.Polygon"]

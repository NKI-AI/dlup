# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

string_classes = (str, bytes)
PathLike = str | os.PathLike
GenericNumber = int | float
GenericNumberArray = np.ndarray | Iterable[GenericNumber]
GenericFloatArray = np.ndarray | Iterable[float]
GenericIntArray = np.ndarray | Iterable[int]

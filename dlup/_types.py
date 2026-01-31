# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines the types using in dlup."""

from __future__ import annotations

import pathlib
from typing import Iterable, Union

import numpy as np
import numpy.typing as npt

string_classes = (str, bytes)
PathLike = Union[str, pathlib.Path]
GenericNumber = Union[int, float]
GenericNumberArray = Union[npt.NDArray[np.int_ | np.float64], Iterable[GenericNumber]]
GenericFloatArray = Union[npt.NDArray[np.float64], Iterable[float]]
GenericIntArray = Union[npt.NDArray[np.int_], Iterable[int]]

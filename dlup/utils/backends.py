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
"""Utilities to handle backends."""

from __future__ import annotations

from enum import Enum
from typing import Any

from dlup.utils.imports import OPENSLIDE_AVAILABLE, TIFFFILE_AVAILABLE


class ImageBackend(Enum):
    """Available image experimental_backends."""

    if OPENSLIDE_AVAILABLE:
        from dlup.backends.openslide_backend import OpenSlideSlide

        OPENSLIDE = OpenSlideSlide

    if TIFFFILE_AVAILABLE:
        from dlup.backends.tifffile_backend import TifffileSlide

        TIFFFILE = TifffileSlide

    from dlup.backends.fastslide_backend import FastSlideSlide

    FASTSLIDE = FastSlideSlide

    from dlup.backends.fimage_backend import FastImageSlide

    FIMAGE = FastImageSlide

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)

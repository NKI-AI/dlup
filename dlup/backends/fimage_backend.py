# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2025 Jonas Teuwen. All Rights Reserved.
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
from __future__ import annotations

import io
from typing import Any, Optional

import fim
import pathlib
import numpy as np
from dlup._types import PathLike
from dlup.backends.common import AbstractSlideBackend
from dlup.utils.image import check_if_mpp_is_valid

FASTSLIDE_SUPPORTED_EXTENSIONS = {".svs", ".mrxs", ".qptiff"}


def open_slide(filename: PathLike) -> "FastImageSlide":
    """
    Read slide with fastslide.

    Parameters
    ----------
    filename : PathLike
        Path to image.
    """
    return FastImageSlide(filename)


class FastImageSlide(AbstractSlideBackend):
    """
    Backend for fastslide.
    """

    def __init__(self, filename: PathLike) -> None:
        """
        Read slide with fastslide.

        Parameters
        ----------
        filename : PathLike
            Path to image."""
        super().__init__(filename)
        self._filename = str(filename)
        self._image = (
            fim.open_fastslide(self._filename)
            if pathlib.Path(filename).suffix in FASTSLIDE_SUPPORTED_EXTENSIONS
            else fim.open_openslide(self._filename)
        )
        self._level_count = self._image.level_count
        self._shapes: list[tuple[int, int]] = self._image.level_dimensions
        self._downsamples: list[float] = self._image.level_downsamples
        self._slide_bounds = self._image.bounds

        mpp_x, mpp_y = self._image.mpp
        self._spacings = [(float(mpp_x * downsample), float(mpp_y * downsample)) for downsample in self._downsamples]
        check_if_mpp_is_valid(mpp_x, mpp_y)
        self._spacing = (mpp_x, mpp_y)

    @property
    def mode(self) -> Optional[str]:
        """Returns the mode of the image."""
        return "RGB"

    @property
    def slide_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide as ((x,y), (width, height)). These can be smaller than the image itself."""
        return self._slide_bounds

    @property
    def spacing(self) -> Optional[tuple[float, float]]:
        return self._spacing

    @property
    def properties(self) -> dict[str, Any]:
        """Metadata about the image as given by fimage"""
        return self._image.properties

    @property
    def color_profile(self) -> Optional[io.BytesIO]:
        raise NotImplementedError

    @property
    def magnification(self) -> Optional[float]:
        """Returns the objective power at which the WSI was sampled."""
        objective_magnification = self.properties.get("objective_magnification", None)
        if objective_magnification != None:
            return float(objective_magnification)
        return None

    @property
    def vendor(self) -> Optional[str]:
        """Returns the scanner vendor."""
        return self.properties.get("scanner_model", None)

    @property
    def associated_images(self) -> dict[str, fim.Image]:
        """Images associated with this whole-slide image."""

        return dict(self._image.associated_images)

    def set_cache(self, cache: Any) -> None:
        raise NotImplementedError

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> fim.Image:
        """
        Return the best level for displaying the given image level as a fim.Image.

        Parameters
        ----------
        coordinates : tuple[int, int]
            Coordinates of the region in level 0.
        level : int
            Level of the image pyramid.
        size : tuple[int, int]
            Size of the region to be extracted.

        Returns
        -------
        fim.Image
            The requested region.
        """
        return self._image.at_level(level).crop(coordinates, size)

    def close(self) -> None:
        """Close the underlying slide"""
        self._image.close()

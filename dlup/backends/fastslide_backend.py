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
import warnings
from typing import Any, Dict, List, Optional, Tuple

import fim
import numpy as np
from fastslide import FastSlide
from dlup._types import PathLike
from dlup.backends.common import AbstractSlideBackend
from dlup.utils.image import check_if_mpp_is_valid


def open_slide(filename: PathLike) -> "FastSlideSlide":
    """
    Read slide with fastslide.

    Parameters
    ----------
    filename : PathLike
        Path to image.
    """
    return FastSlideSlide(filename)


class FastSlideSlide(AbstractSlideBackend):
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
        self._image = FastSlide.from_file_path(self._filename)
        self._spacings: list[tuple[float, float]] = []
        self._level_count = self._image.level_count
        self._shapes: list[tuple[int, int]] = self._image.level_dimensions
        self._downsamples: list[float] = self._image.level_downsamples
        self._slide_bounds = self._image.bounds

        mpp_x = self._image.properties.get("mpp_x", 0)
        mpp_y = self._image.properties.get("mpp_y", 0)
        check_if_mpp_is_valid(mpp_x, mpp_y)
        self.spacing = (mpp_x, mpp_y)

    @property
    def mode(self) -> Optional[str]:
        """Returns the mode of the image."""
        return "RGB"

    @property
    def slide_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Returns the bounds of the slide as ((x,y), (width, height)). These can be smaller than the image itself."""
        return self._slide_bounds

    @property
    def spacing(self) -> Optional[Tuple[float, float]]:
        return self._spacings[0] if self._spacings else None

    @spacing.setter
    def spacing(self, value: Tuple[float, float]) -> None:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("`.spacing` has to be of the form (mpp_x, mpp_y).")

        mpp_x, mpp_y = value
        check_if_mpp_is_valid(mpp_x, mpp_y)
        self._spacings = [(float(mpp_x * downsample), float(mpp_y * downsample)) for downsample in self._downsamples]

    @property
    def properties(self) -> Dict[str, Any]:
        """Metadata about the image as given by fastslide"""
        return self._image.properties

    @property
    def color_profile(self) -> Optional[io.BytesIO]:
        raise NotImplementedError

    @property
    def magnification(self) -> Optional[float]:
        """Returns the objective power at which the WSI was sampled."""
        raise NotImplementedError

    @property
    def vendor(self) -> Optional[str]:
        """Returns the scanner vendor."""
        return self.properties.get("scanner_model", None)

    @property
    def associated_images(self) -> dict[str, fim.Image]:
        """Images associated with this whole-slide image."""

        return {k: fim.Image.from_numpy(self._image.associated_images[k]) for k in self._image.associated_images.keys()}

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
        # TODO(jonasteuwen): Add support for other image formats, such as multiplex.
        # TODO(jonasteuwen): Currently this assumes that the channel dimension is the last dimension.
        array = self._image.read_region(coordinates, level, size).numpy()
        return fim.Image.from_numpy(array)

    def close(self) -> None:
        """Close the underlying slide"""
        self._image.close()

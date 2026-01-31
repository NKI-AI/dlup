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
from typing import Any

import fim
import numpy as np
import tifffile
from dlup._types import PathLike
from dlup.backends.common import AbstractSlideBackend
from dlup.utils.tifffile_utils import get_tile


def open_slide(filename: PathLike) -> "TifffileSlide":
    """
    Read slide with tifffile.

    Parameters
    ----------
    filename : PathLike
        Path to image.
    """
    return TifffileSlide(filename)


class TifffileSlide(AbstractSlideBackend):
    """
    Backend for tifffile.
    """

    def __init__(self, filename: PathLike) -> None:
        """
        Parameters
        ----------
        filename : PathLike
            Path to image.
        """
        super().__init__(filename)
        self._image: tifffile.TiffFile = tifffile.TiffFile(filename)
        self._level_count = len(self._image.pages)
        self.__parse_tifffile()

    def __parse_tifffile(self) -> None:
        """Parse the file with tifffile, extract the resolution, downsample factors and sizes for each level."""
        unit_dict = {1: 10, 2: 25400, 3: 10000, 4: 100000, 5: 1000000}
        self._downsamples.append(1.0)
        for idx, page in enumerate(self._image.pages):
            self._shapes.append((page.shape[1], page.shape[0]))

            x_res = page.tags["XResolution"].value
            x_res = x_res[0] / x_res[1]
            y_res = page.tags["YResolution"].value
            y_res = y_res[0] / y_res[1]
            unit = int(page.tags["ResolutionUnit"].value)

            mpp_x = unit_dict[unit] / x_res
            mpp_y = unit_dict[unit] / y_res
            self._spacings.append((mpp_x, mpp_y))

            if idx >= 1:
                downsample = mpp_x / self._spacings[0][0]
                self._downsamples.append(downsample)

    @property
    def properties(self) -> dict[str, Any]:
        """Metadata about the image as given by tifffile."""

        properties = {}
        for idx, page in enumerate(self._image.pages):
            for tag in page.tags:
                # These tags are not so relevant at this point and have a lot of output
                if tag.name in [
                    "TileOffsets",
                    "TileByteCounts",
                    "SMinSampleValue",
                    "JPEGTables",
                    "ReferenceBlackWhite",
                ]:
                    continue

                properties[f"tifffile.level[{idx}].{tag.name}"] = tag.value

        return properties

    def set_cache(self, cache: Any) -> None:
        """Cache for tifffile."""
        raise NotImplementedError

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]):
        """
        Return the best level for displaying the given image level as a fim.Image.

        Parameters
        ----------
        coordinates : tuple
            Coordinates of the region in level 0.
        level : int
            Level of the image pyramid.
        size : tuple
            Size of the region to be extracted.

        Returns
        -------
        fim.Image
            The requested region.
        """
        if level > self._level_count - 1:
            raise RuntimeError(f"Level {level} not present.")

        page = self._image.pages[level]
        # To make mypy happy
        assert isinstance(page, tifffile.TiffPage)
        fim_tile = get_tile(page, coordinates, size)
        return fim_tile

    @property
    def vendor(self) -> None:
        """Returns the scanner vendor. For tiffs this is unknown."""
        return None

    @property
    def magnification(self) -> None:
        """Returns the objective power at which the WSI was sampled. For tiff's this is unknown."""
        return None

    def close(self) -> None:
        """Close the underlying slide"""
        self._image.close()

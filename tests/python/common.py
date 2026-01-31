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
"""Utilities to simplify the mocking of SlideImages."""

from typing import Dict, List, Optional, Tuple

import fim
import numpy as np
import openslide
import math
from pydantic import BaseModel, ConfigDict
from dlup.backends.openslide_backend import OpenSlideSlide
from dlup._types import PathLike
from unittest.mock import MagicMock
import openslide.lowlevel as openslide_lowlevel


class LevelConfig(BaseModel):
    dimensions: Tuple[int, int]
    downsample: float


class SlideConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    filename: str
    properties: Dict[str, str]
    levels: List[LevelConfig]
    image: Optional[np.ndarray]

    @classmethod
    def from_parameters(
        cls,
        filename: str,
        num_levels: int,
        level_0_dimensions: Tuple[int, int],
        mpp: Tuple[float, float],
        objective_power: Optional[int],
        vendor: str,
        image: Optional[np.ndarray] = None,
    ):
        # Calculate the dimensions and downsample for each level
        levels = []
        for level in range(num_levels):
            downsample = 2**level
            dimensions = (level_0_dimensions[0] // downsample, level_0_dimensions[1] // downsample)
            levels.append(LevelConfig(dimensions=dimensions, downsample=downsample))

        # Create the properties dictionary
        properties = {
            openslide.PROPERTY_NAME_MPP_X: str(mpp[0]),
            openslide.PROPERTY_NAME_MPP_Y: str(mpp[1]),
            openslide.PROPERTY_NAME_VENDOR: vendor,
        }
        if objective_power:
            properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER] = str(objective_power)

        # Return an instance of SlideConfig
        return cls(filename=filename, properties=properties, levels=levels, image=image)


def get_sample_nonuniform_image(size: tuple[int, int] = (256, 256), divisions: int = 10) -> fim.Image:
    """Generate a test image with a grid pattern."""
    width, height = size
    x_divisions = min(divisions, width)
    y_divisions = min(divisions, height)

    # Calculate the width and height of each grid cell
    cell_width = width // x_divisions
    cell_height = height // y_divisions

    # Create an array to store the image
    image_array = np.zeros((height, width, 4), dtype=float)

    # Define a set of distinct colors
    color_palette = [
        (255, 0, 0, 255),  # Red
        (0, 255, 0, 255),  # Green
        (0, 0, 255, 255),  # Blue
        (255, 255, 0, 255),  # Yellow
        (255, 0, 255, 255),  # Magenta
        (0, 255, 255, 255),  # Cyan
        (128, 0, 0, 255),  # Maroon
        (0, 128, 0, 255),  # Dark Green
        (0, 0, 128, 255),  # Navy
        (128, 128, 0, 255),  # Olive
    ]

    # Extend the palette to cover all cells
    num_colors = len(color_palette)
    extended_palette = [color_palette[i % num_colors] for i in range(x_divisions * y_divisions)]

    # Assign colors to each grid cell
    for i in range(x_divisions):
        for j in range(y_divisions):
            x_start = i * cell_width
            y_start = j * cell_height
            x_end = x_start + cell_width if (i != x_divisions - 1) else width
            y_end = y_start + cell_height if (j != y_divisions - 1) else height
            color = extended_palette[i * y_divisions + j]
            image_array[y_start:y_end, x_start:x_end, :] = color

    # Apply a sine wave pattern for non-uniformity
    x = np.linspace(0, np.pi * 4, width)
    y = np.linspace(0, np.pi * 4, height)
    X, Y = np.meshgrid(x, y)
    sine_wave = (np.sin(X) + np.cos(Y)) / 2 + 0.5  # Normalized to range [0, 1]

    # Multiply sine wave pattern with the image
    for k in range(3):  # Apply only to RGB channels, not alpha
        image_array[:, :, k] = image_array[:, :, k] * sine_wave

    return fim.Image.from_numpy(image_array.astype(np.uint8))


SLIDE_CONFIGS = [
    SlideConfig.from_parameters(
        filename="dummy1.svs",
        num_levels=3,
        level_0_dimensions=(2000, 2000),
        mpp=(0.25, 0.25),
        objective_power=40,
        vendor="dummy",
    ),
    SlideConfig.from_parameters(
        filename="dummy2.svs",
        num_levels=3,
        level_0_dimensions=(1800, 2000),
        mpp=(0.50, 0.50),
        objective_power=20,
        vendor="test_vendor",
    ),
]


class MockOpenSlideLowLevel:
    def __init__(self, config):
        self.config = config
        self.mock_open = MagicMock()
        levels = config.levels
        self.mock_get_property_names = MagicMock(return_value=list(config.properties.keys()))
        self.mock_get_property_value = MagicMock(side_effect=lambda _owsi, key: config.properties[key])
        self.mock_get_level_count = MagicMock(return_value=len(levels))
        self.mock_get_level_dimensions = MagicMock(side_effect=lambda _owsi, idx: levels[idx].dimensions)
        self.mock_get_level_downsample = MagicMock(side_effect=lambda _owsi, idx: levels[idx].downsample)
        self.mock_close = MagicMock()

        self.mock_read_region = MagicMock(side_effect=self.mock_read_region_fn)

        if config.image is not None:
            self.base_image = fim.Image.from_numpy(config.image)
            self.ndim = 1 if len(config.image.shape) == 2 else config.image.shape[-1]
        else:
            self.base_image = get_sample_nonuniform_image(config.levels[0].dimensions)
            self.ndim = 4

    def mock_read_region_fn(self, _owsi, x, y, level, w, h) -> fim.Image:
        downsample_factor = self.config.levels[level].downsample

        # Calculate coordinates and size at level 0
        w0, h0 = math.ceil(w * downsample_factor), math.ceil(h * downsample_factor)

        # Crop the base image
        cropped_image = self.base_image.crop((x, y), (w0, h0))

        # Resize the cropped image to the requested level
        if downsample_factor != 1.0:
            target_width = int(w)
            target_height = int(h)
            level_image = cropped_image.resize(target_width, target_height, kernel=fim.KernelType.NEAREST)
        else:
            level_image = cropped_image
        return level_image


mock_lowlevel = None


class MockOpenSlideSlide(OpenSlideSlide):
    def __init__(self, filename: PathLike):
        global mock_lowlevel
        self._filename = filename
        self._owsi = "mocked_owsi"
        self._spacings = []

        mock_lowlevel.mock_open.return_value = self._owsi
        self._owsi = mock_lowlevel.mock_open(filename)

        try:
            mpp_x = float(self.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.properties[openslide.PROPERTY_NAME_MPP_Y])
            self.spacing = (mpp_x, mpp_y)
        except KeyError:
            spacing = _get_mpp_from_tiff(dict(self.properties))
            if spacing:
                self.spacing = spacing

    @property
    def properties(self) -> dict[str, str]:
        return {
            key: mock_lowlevel.mock_get_property_value(self._owsi, key)
            for key in mock_lowlevel.mock_get_property_names(self._owsi)
        }

    @property
    def level_count(self) -> int:
        return mock_lowlevel.mock_get_level_count(self._owsi)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple(mock_lowlevel.mock_get_level_dimensions(self._owsi, idx) for idx in range(self.level_count))

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return tuple(mock_lowlevel.mock_get_level_downsample(self._owsi, idx) for idx in range(self.level_count))

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]):
        if size[0] <= 0 or size[1] <= 0:
            raise openslide_lowlevel.OpenSlideError(f"width ({size[0]}) or height ({size[1]}) must be positive")
        fim_region = mock_lowlevel.mock_read_region(self._owsi, coordinates[0], coordinates[1], level, size[0], size[1])

        return fim_region

    def get_level_image(self, level: int) -> fim.Image:
        scale = 1.0 / self.level_downsamples[level]
        base_width, base_height, _ = mock_lowlevel.base_image.dimensions
        target_width = int(round(base_width * scale))
        target_height = int(round(base_height * scale))
        return mock_lowlevel.base_image.resize(target_width, target_height, kernel=fim.KernelType.NEAREST)

    def close(self) -> None:
        mock_lowlevel.mock_close(self._owsi)

    @classmethod
    def from_config(cls, config: SlideConfig):
        global mock_lowlevel
        mock_lowlevel = MockOpenSlideLowLevel(config)
        return cls(config.filename)

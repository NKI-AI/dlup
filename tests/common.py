# Copyright (c) dlup contributors

"""Utilities to simplify the mocking of SlideImages."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import openslide
import pyvips
from pydantic import BaseModel


class LevelConfig(BaseModel):
    dimensions: Tuple[int, int]
    downsample: float


class SlideConfig(BaseModel):
    filename: str
    properties: Dict[str, str]
    levels: List[LevelConfig]

    @classmethod
    def from_parameters(
        cls,
        filename: str,
        num_levels: int,
        level_0_dimensions: Tuple[int, int],
        mpp: Tuple[float, float],
        objective_power: Optional[int],
        vendor: str,
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
        return cls(filename=filename, properties=properties, levels=levels)


def get_sample_nonuniform_image(size: tuple[int, int] = (256, 256), divisions: int = 10) -> pyvips.Image:
    """Generate a test image with a grid pattern."""
    width, height = size
    x_divisions = min(divisions, width)
    y_divisions = min(divisions, height)

    # Calculate the width and height of each grid cell
    cell_width = width // x_divisions
    cell_height = height // y_divisions

    # Create an array to store the image
    image_array = np.zeros((height, width, 4), dtype=np.uint8)

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

    return pyvips.Image.new_from_array(image_array)

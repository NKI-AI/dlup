# Copyright (c) dlup contributors

"""Utilities to simplify the mocking of SlideImages."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import openslide  # type: ignore
import PIL
import pytest  # noqa
from pydantic import BaseModel
from scipy.interpolate import RegularGridInterpolator

from dlup.utils.pyvips_utils import pil_to_vips


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


def get_sample_nonuniform_image(size: Tuple[int, int] = (256, 256)):
    """Generate a non-uniform sample image."""
    if not (np.array(size) % 2 == 0).all():
        raise ValueError("Size should be a tuple of values divisible by two.")

    # Define data for interpolation
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])

    # Define the values at each grid point
    X, Y = np.meshgrid(x, y)
    values = X + Y

    # Use RegularGridInterpolator
    f = RegularGridInterpolator((x, y), values)

    # Sample it
    Z = np.stack([X, Y], axis=-1)
    z = f(Z).reshape(size)

    # Sample it
    z = f(Z).reshape(size)

    # Interpret it as HSV, so we get funny colors
    im = np.zeros((*size, 3))
    im[:, :, 0] = z
    im[:, :, 1] = 1

    # Set the value to a pixel-level checkerboard.
    im[size[1] // 2, size[0] // 2, 2] = 1
    im[:, :, 2] = np.sign(np.fft.ifft2(im[:, :, 2]).real)
    im = im * 255
    im = im.astype("uint8")
    im = PIL.Image.fromarray(im, mode="HSV")
    return pil_to_vips(im.convert(mode="RGBA"))

# coding=utf-8
# Copyright (c) dlup contributors
import os

import numpy as np

import openslide
from dlup import UnsupportedSlideError
from dlup.experimental_backends.common import AbstractSlideBackend, check_if_mpp_is_isotropic
from typing import Optional


def open_slide(filename: os.PathLike) -> "OpenSlideSlide":
    """
    Read slide with openslide.

    Parameters
    ----------
    filename : PathLike
        Path to image.
    """
    return OpenSlideSlide(filename)


class OpenSlideSlide(openslide.OpenSlide, AbstractSlideBackend):
    """
    Backend for openslide.
    """

    def __init__(self, filename: os.PathLike):
        """
        Parameters
        ----------
        filename : PathLike
            Path to image.
        """
        super().__init__(str(filename))

        try:
            mpp_x = float(self.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.properties[openslide.PROPERTY_NAME_MPP_Y])

        except KeyError:
            raise UnsupportedSlideError(f"slide property mpp is not available.", str(filename))

        check_if_mpp_is_isotropic(mpp_x, mpp_y)
        mpp = np.array([mpp_y, mpp_x])
        self._spacings = [tuple(mpp * downsample) for downsample in self.level_downsamples]

    @property
    def magnification(self) -> float:
        """Returns the objective power at which the WSI was sampled."""
        return int(self.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    @property
    def vendor(self) -> str:
        """Returns the scanner vendor."""
        return self.properties.properties[openslide.PROPERTY_NAME_VENDOR]

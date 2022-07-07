# coding=utf-8
# Copyright (c) dlup contributors
from typing import Tuple, Union

import numpy as np
import openslide
import PIL.Image

from dlup import UnsupportedSlideError
from dlup.experimental_backends.common import AbstractSlideBackend
from dlup.types import PathLike


def open_slide(filename: PathLike) -> "OpenSlideSlide":
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

    def __init__(self, filename: PathLike):
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

    def get_thumbnail(self, size: Union[int, Tuple[int, int]]) -> PIL.Image:
        """
        Return a PIL.Image as an RGB image with the thumbnail with maximum size given by size.
        Aspect ratio is preserved.

        Parameters
        ----------
        size : int or Tuple[int, int]
            Output size of the thumbnail, will take the maximal value for the output and preserve aspect ratio.

        Returns
        -------
        PIL.Image
            The thumbnail.
        """
        if isinstance(size, int):
            size = (size, size)

        return super().get_thumbnail(size)

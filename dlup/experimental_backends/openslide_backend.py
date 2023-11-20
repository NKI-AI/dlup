# type: ignore
# Copyright (c) dlup contributors
from __future__ import annotations

import warnings
from distutils.version import LooseVersion
from typing import cast

import numpy as np
import openslide
import PIL.Image
from PIL.ImageCms import ImageCmsProfile

from dlup.backends.common import AbstractSlideBackend
from dlup.types import PathLike
from dlup.utils.image import check_if_mpp_is_valid

TIFF_PROPERTY_NAME_RESOLUTION_UNIT = "tiff.ResolutionUnit"
TIFF_PROPERTY_NAME_X_RESOLUTION = "tiff.XResolution"
TIFF_PROPERTY_NAME_Y_RESOLUTION = "tiff.YResolution"


def _get_mpp_from_tiff(properties: dict[str, str]) -> tuple[float, float] | None:
    """Get mpp values from the TIFF tags as parsed by openslide.
    This only works for openslide < 4.0.0, as newer openslide versions automatically parse this.

    Parameters
    ----------
    properties : dict[str, str]
        The properties as parsed by openslide.

    Returns
    -------
    tuple[float, float] or None
        The mpp values if they are present in the TIFF tags, otherwise None.
    """
    # It is possible we now have a TIFF file with the mpp information in the TIFF tags.
    if LooseVersion(openslide.__library_version__) < LooseVersion("4.0.0"):
        if properties[openslide.PROPERTY_NAME_VENDOR] == "generic-tiff":
            # Check if the TIFF tags are present
            resolution_unit = properties.get(TIFF_PROPERTY_NAME_RESOLUTION_UNIT, None)
            x_resolution = float(properties.get(TIFF_PROPERTY_NAME_X_RESOLUTION, 0))
            y_resolution = float(properties.get(TIFF_PROPERTY_NAME_Y_RESOLUTION, 0))

            if x_resolution > 0 and y_resolution > 0:
                unit_dict = {"cm": 10000, "centimeter": 10000}
                mpp_x = unit_dict[resolution_unit] / x_resolution
                mpp_y = unit_dict[resolution_unit] / y_resolution
                return mpp_x, mpp_y
    return None


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
        self._spacings = None

        try:
            mpp_x = float(self.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.properties[openslide.PROPERTY_NAME_MPP_Y])
            self.spacing = (mpp_x, mpp_y)

        except KeyError:
            # It is possible we now have a TIFF file with the mpp information in the TIFF tags.
            spacing = _get_mpp_from_tiff(dict(self.properties))
            if spacing:
                self.spacing = spacing

    @property
    def spacing(self) -> tuple[float, float] | None:
        if not self._spacings:
            return None
        return self._spacings[0]

    @spacing.setter
    def spacing(self, value: tuple[float, float]) -> None:
        if not isinstance(value, tuple) and len(value) != 2:
            raise ValueError("`.spacing` has to be of the form (mpp_x, mpp_y).")

        mpp_x, mpp_y = value
        check_if_mpp_is_valid(mpp_x, mpp_y)
        mpp = np.array([mpp_x, mpp_y])
        self._spacings = [cast(tuple[float, float], tuple(mpp * downsample)) for downsample in self.level_downsamples]

    @property
    def magnification(self) -> int | None:
        """Returns the objective power at which the WSI was sampled."""
        value = self.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, None)
        if value is not None:
            return int(value)
        return value

    @property
    def color_profile(self) -> ImageCmsProfile | None:
        """
        Returns the color profile of the image if available. Otherwise returns None.

        Returns
        -------
        ImageCmsProfile, optional
            The color profile of the image.
        """
        if LooseVersion(openslide.__library_version__) < LooseVersion("4.0.0"):
            warnings.warn(
                "Color profile support is only available for openslide >= 4.0.0. "
                f"You have version {openslide.__library_version__}. "
                "Please update your openslide installation if you want to use this feature (recommended)."
            )
            return None

        return super().color_profile

    @property
    def vendor(self) -> str:
        """Returns the scanner vendor."""
        return self.properties.get(openslide.PROPERTY_NAME_VENDOR, None)

    @property
    def slide_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide as ((x,y), (width, height)). These can be smaller than the image itself."""

        offset = int(self.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0)), int(
            self.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0)
        )
        size = int(self.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH, self.dimensions[0])), int(
            self.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT, self.dimensions[1])
        )

        return offset, size

    def get_thumbnail(self, size: int | tuple[int, int]) -> PIL.Image.Image:
        """
        Return a PIL.Image as an RGB image with the thumbnail with maximum size given by size.
        Aspect ratio is preserved.

        Parameters
        ----------
        size : int or tuple[int, int]
            Output size of the thumbnail, will take the maximal value for the output and preserve aspect ratio.

        Returns
        -------
        PIL.Image
            The thumbnail.
        """
        if isinstance(size, int):
            size = (size, size)

        return super().get_thumbnail(size)

# Copyright (c) dlup contributors
from __future__ import annotations

import io
import warnings
from ctypes import Array, c_uint32
from typing import Any, cast

import numpy as np
import openslide
import openslide.lowlevel as openslide_lowlevel
import pyvips
from packaging.version import Version

from dlup.backends.common import AbstractSlideBackend
from dlup.types import PathLike
from dlup.utils.image import check_if_mpp_is_valid

TIFF_PROPERTY_NAME_RESOLUTION_UNIT = "tiff.ResolutionUnit"
TIFF_PROPERTY_NAME_X_RESOLUTION = "tiff.XResolution"
TIFF_PROPERTY_NAME_Y_RESOLUTION = "tiff.YResolution"


def _load_image_vips(buffer: Array[c_uint32], size: tuple[int, int]) -> pyvips.Image:
    """Convert the raw buffer to a pyvips.Image."""
    openslide_lowlevel._convert.argb2rgba(buffer)
    mem_view = memoryview(buffer).cast("B")
    return pyvips.Image.new_from_memory(mem_view, size[0], size[1], 4, "uchar")


def read_region(slide: Any, x: int, y: int, level: int, w: int, h: int) -> pyvips.Image:
    if w <= 0 or h <= 0:
        # OpenSlide would catch this, but not before we tried to allocate
        # a negative-size buffer
        raise openslide_lowlevel.OpenSlideError(f"width ({w}) or height ({h}) must be positive")
    buf = (w * h * c_uint32)()
    openslide_lowlevel._read_region(slide, buf, x, y, level, w, h)
    return _load_image_vips(buf, (w, h))


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
    if Version(openslide.__library_version__) < Version("4.0.0"):
        if properties[openslide.PROPERTY_NAME_VENDOR] == "generic-tiff":
            # Check if the TIFF tags are present
            resolution_unit = properties.get(TIFF_PROPERTY_NAME_RESOLUTION_UNIT, None)
            if not resolution_unit:
                return None
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


class OpenSlideSlide(AbstractSlideBackend):
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
        self._filename = filename
        self._owsi = openslide_lowlevel.open(str(filename))
        self._spacings: list[tuple[float, float]] = []

        try:
            mpp_x = float(self.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.properties[openslide.PROPERTY_NAME_MPP_Y])
            self.spacing = (mpp_x, mpp_y)

        except KeyError:
            # It is possible we now have a TIFF file with the mpp information in the TIFF tags.
            spacing = _get_mpp_from_tiff(dict(self.properties))
            if spacing:
                self.spacing = spacing

        if openslide_lowlevel.read_icc_profile.available:
            self._profile = openslide_lowlevel.read_icc_profile(self._owsi)
        else:
            self._profile = None

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
    def mode(self) -> str:
        """Returns the mode of the image. For OpenSlide this is always RGBA"""
        return "RGBA"

    @property
    def magnification(self) -> float | None:
        """Returns the objective power at which the WSI was sampled."""
        value = self.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, None)
        if value is not None:
            return float(value)
        return value

    @property
    def color_profile(self) -> io.BytesIO | None:
        """
        Returns the color profile of the image if available. Otherwise returns None.

        Returns
        -------
        io.BytesIO, optional
            The color profile of the image.
        """
        if Version(openslide.__library_version__) < Version("4.0.0") or not self._profile:
            warnings.warn(
                "Color profile support is only available for openslide >= 4.0.0. "
                f"You have version {openslide.__library_version__}. "
                "Please update your openslide installation if you want to use this feature (recommended)."
            )
            return None

        if self._profile is None:
            return None

        return io.BytesIO(self._profile)

    @property
    def vendor(self) -> str | None:
        """Returns the scanner vendor."""
        vendor = self.properties.get(openslide.PROPERTY_NAME_VENDOR, None)
        if vendor is not None:
            return str(vendor)
        return None

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

    @property
    def properties(self) -> dict[str, str]:
        """Metadata about the image as given by openslide."""
        keys = openslide_lowlevel.get_property_names(self._owsi)
        return dict((key, openslide_lowlevel.get_property_value(self._owsi, key)) for key in keys)

    @property
    def level_count(self) -> int:
        """The number of levels in the image."""
        return cast(int, openslide_lowlevel.get_level_count(self._owsi))

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        """A list of (width, height) tuples, one for each level of the image."""
        return tuple(openslide_lowlevel.get_level_dimensions(self._owsi, idx) for idx in range(self.level_count))

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        """A list of downsampling factors for each level of the image."""
        return tuple(openslide_lowlevel.get_level_downsample(self._owsi, idx) for idx in range(self.level_count))

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
        region = read_region(self._owsi, coordinates[0], coordinates[1], level, size[0], size[1])
        return region

    def close(self) -> None:
        """Close the openslide object."""
        openslide_lowlevel.close(self._owsi)

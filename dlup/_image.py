# coding=utf-8
# Copyright (c) DLUP Contributors
"""Whole slide image access objects.

In this module we take care of abstracting the access to whole slide images.
The main workhorse is SlideImage which takes care of simplyfing region extraction
of discrete-levels pyramidal images in a continuous way, validating relevant
properties and offering a future aggregated api for possibly multiple different backends
other than openslide.
"""

import functools
import pathlib
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np  # type: ignore
import openslide  # type: ignore
import PIL.Image  # type: ignore
import PIL

from dlup import DLUPUnsupportedSlideError
from ._region import RegionView


_GenericNumber = Union[int, float]
_GenericNumberArray = Union[np.ndarray, Iterable[_GenericNumber]]
_GenericFloatArray = Union[np.ndarray, Iterable[float]]
_GenericIntArray = Union[np.ndarray, Iterable[int]]
_Box = Tuple[_GenericNumber, _GenericNumber, _GenericNumber, _GenericNumber]
_TSlideImage = TypeVar("TSlideImage", bound="SlideImage")


class SlideImageRegionView(RegionView):
    """Represents an image view tied to a slide image."""

    def __init__(self, wsi: _TSlideImage, scaling: _GenericNumber):
        self._wsi = wsi
        self._scaling = scaling

    @property
    def mpp(self):
        """Returns the level effective mpp."""
        return self._wsi.mpp / self._scaling

    @property
    def size(self):
        """Size"""
        return self._wsi.get_scaled_size(self._scaling)

    def read_region(self, location: _GenericFloatArray,
                    size: _GenericIntArray) -> np.ndarray:
        """Returns a region in the level."""
        return self._wsi.read_region(location, self._scaling, size)


def _clip2size(a: np.ndarray, size: Tuple[_GenericNumber, _GenericNumber]) -> np.ndarray:
    return np.clip(a, (0, 0), size)


class SlideImage:
    """Utility class to simplify whole-slide pyramidal images management.

    This helper class furtherly abstracts openslide access to WSIs
    by validating some of the properties and giving access
    to a continuous pyramid. Layer values are interpolated from
    the closest bottom layer.
    Each horizontal slices of the pyramid can be accessed using a scaling value
    z as index.

    Example usage:

    ```python
    from dlup import WholeSlidePyramidalImage
    wsi = dlup.WholeSlideImage.open('path/to/slide.svs')
    ```
    """

    def __init__(self, wsi: openslide.AbstractSlide, identifier: Union[str, None] = None):
        """Initialize a whole slide image and validate its properties."""
        self._openslide_wsi = wsi
        self._identifier = identifier

        try:
            mpp_x = float(self._openslide_wsi.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self._openslide_wsi.properties[openslide.PROPERTY_NAME_MPP_Y])
            mpp = np.array([mpp_y, mpp_x])
        except KeyError:
            raise DLUPUnsupportedSlideError(f"Slide property mpp is not available.")

        if not np.isclose(mpp[0], mpp[1]):
            raise DLUPUnsupportedSlideError("Cannot deal with slides having anisotropic mpps.")

        self._min_native_mpp = float(mpp[0])

    @classmethod
    def from_file_path(
        cls: _TSlideImage, wsi_file_path: pathlib.Path, identifier: Union[str, None] = None
    ) -> _TSlideImage:
        try:
            wsi = openslide.open_slide(str(wsi_file_path))
        except (openslide.OpenSlideUnsupportedFormatError, PIL.UnidentifiedImageError):
            raise DLUPUnsupportedSlideError('Unsupported file.')

        return cls(wsi, str(wsi_file_path) if identifier is None else identifier)

    def read_region(self, location: Tuple[_GenericNumber, _GenericNumber], scaling: float,
                    size: Tuple[int, int]) -> np.ndarray:
        """Return a pyramidal region.

        A typical slide is made of several levels at different mpps.
        In normal cirmustances, it's not possible to retrieve an image of
        intermediate mpp between these levels. This method takes care of
        sumbsampling the closest high resolution level to extract a target
        region via interpolation.

        TODO(lromor): Ideally, all the regions at higher levels could be
        also downsampled from the highest resolution level at the expenses of
        an higher computational cost. We could make an optional flag to enable
        such feature.

        Parameters
        ----------
        location :
            Location from the top left (x, y) in pixel coordinates.
        scaling :
            scaling value.
        size :
            Region size to extract in pixels.
        """
        owsi = self._openslide_wsi
        location = np.array(location)
        size = np.array(size)

        if (size < 0).any():
            raise ValueError("Size values must be greater than zero.")

        if (location < 0).any():
            raise ValueError("Location values must be greater than zero.")

        # Compute values projected onto the best layer.
        native_level = owsi.get_best_level_for_downsample(1 / scaling)
        native_level_size = owsi.level_dimensions[native_level]
        native_level_downsample = owsi.level_downsamples[native_level]
        native_scaling = scaling * owsi.level_downsamples[native_level]
        native_location = location / native_scaling
        native_size = size / native_scaling

        # Openslide doesn't feature float coordinates to extract a region.
        # We need to extract enough pixels and let PIL do the interpolation.
        # In the borders, the basis functions of other samples contribute to the final value.
        # PIL lanczos uses 3 pixels as support.
        # See pillow: https://git.io/JG0QD
        native_extra_pixels = 3 if native_scaling > 1 else np.ceil(3 / native_scaling)

        # Compute the native location while counting the extra pixels.
        native_location_adapted = np.floor(native_location - native_extra_pixels).astype(int)
        native_location_adapted = _clip2size(native_location_adapted, native_level_size)

        # Unfortunately openslide requires the location in pixels from level 0.
        level_zero_location_adapted = np.floor(native_location_adapted * native_level_downsample).astype(int)
        native_location_adapted = level_zero_location_adapted / native_level_downsample
        native_size_adapted = np.ceil(native_location + native_size + native_extra_pixels).astype(int)
        native_size_adapted = _clip2size(native_size_adapted, native_level_size) - native_location_adapted
        native_size_adapted = native_size_adapted.astype(int)

        # We extract the region via openslide with the required extra border
        region = owsi.read_region(tuple(level_zero_location_adapted), native_level, tuple(native_size_adapted))

        # Within this region, there are a bunch of extra pixels, we interpolate to sample
        # the pixel in the right position to retain the right sample weight.
        fractional_coordinates = native_location - native_location_adapted
        box = (*fractional_coordinates, *(fractional_coordinates + native_size))
        return np.asarray(region.resize(size, resample=PIL.Image.LANCZOS, box=box))

    def get_scaled_size(self, scaling: _GenericNumber):
        size = np.array(self.size) * scaling
        return size.astype(int)

    def get_scaled_view(self, scaling: _GenericNumber) -> SlideImageRegionView:
        """Return a pyramid region."""
        return SlideImageRegionView(self, scaling)

    def get_thumbnail(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Returns an RGB numpy thumbnail for the current slide.

        Parameters
        ----------
        size :
            Maximum bounding box for the thumbnail expressed as (width, height).
        """
        return np.array(self._openslide_wsi.get_thumbnail(size))

    @property
    def thumbnail(self):
        return self.get_thumbnail()

    @property
    def identifier(self) -> str:
        """Returns a user-defined identifier."""
        return self._identifier

    @property
    def properties(self) -> dict:
        """Returns any extra associated properties with the image."""
        return self._openslide_wsi.properties

    @property
    def vendor(self) -> str:
        """Returns the scanner vendor."""
        return self.properties["openslide.vendor"]

    @property
    def size(self) -> Tuple[int, int]:
        """Returns the highest resolution image size in pixels."""
        return self._openslide_wsi.dimensions

    @property
    def mpp(self) -> float:
        """Returns the microns per pixel of the high res image."""
        return self._min_native_mpp

    @property
    def magnification(self) -> int:
        """Returns the objective power at which the WSI was sampled."""
        return int(self._openslide_wsi.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    @property
    def aspect_ratio(self) -> dict:
        """Returns width / height."""
        width, height = self.size
        return width / height

    def __repr__(self) -> str:
        props = ("identifier", "vendor", "highest_resolution_mpp", "magnification")
        props_str = []
        for key in props:
            value = getattr(self, key)
            props_str.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(props_str)})"

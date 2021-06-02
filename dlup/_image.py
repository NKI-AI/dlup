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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np  # type: ignore
import openslide  # type: ignore
import PIL.Image  # type: ignore


_GenericNumber = Union[int, float]
_GenericNumberArray = Union[np.ndarray, Iterable[_GenericNumber]]
_GenericFloatArray = Union[np.ndarray, Iterable[float]]
_GenericIntArray = Union[np.ndarray, Iterable[int]]
_Box = Tuple[_GenericNumber, _GenericNumber, _GenericNumber, _GenericNumber]
_TSlideImage = TypeVar("TSlideImage", bound="SlideImage")


class RegionView(ABC):
    """A generic image object from which you can extract a region.

    A unit 'U' is assumed to be consistent across this interface.
    Could be for instance pixels.

    TODO(lromor): Add features like cyclic boundary conditions
    or zero padding, or "hard" walls.
    """

    @property
    @abstractmethod
    def width(self):
        """Returns the width of the image in unit U."""
        pass

    @property
    @abstractmethod
    def height(self):
        """Returns the height of the image in unit U."""
        pass

    @abstractmethod
    def _read_region(self, location: _GenericFloatArray, size: _GenericIntArray) -> PIL.Image:
        """Returns the region covered by the box.

        box coordinates are defined in U units. (0, 0) location
        starts from the top left.
        """
        pass


class SlideImageRegionView(RegionView):
    """Represents an image view tied to a slide image."""

    def __init__(self, wsi: _TSlideImage, scaling: float):
        self._wsi = wsi
        self._scaling = scaling
        self._width, self._height = np.array(wsi.highest_resolution_dimensions) * scaling

    @property
    def mpp(self):
        """Returns the level effective mpp."""
        return self._scaling * self._wsi.highest_resolution_mpp

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def _read_region(self, location: _GenericFloatArray, size: _GenericIntArray) -> PIL.Image:
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
            raise RuntimeError(f"Slide property mpp is not available.")

        if not np.isclose(mpp[0], mpp[1]):
            raise RuntimeError("Cannot deal with slides having anisotropic mpps.")

        self._min_native_mpp = float(mpp[0])

    @classmethod
    def from_file_path(
        cls: _TSlideImage, wsi_file_path: pathlib.Path, identifier: Union[str, None] = None
    ) -> _TSlideImage:
        wsi = openslide.open_slide(str(wsi_file_path))
        # As default identifier we use a tuple (folder, filename)
        identifier = identifier if identifier is not None else wsi_file_path.parts[-2:]
        return cls(wsi, identifier)

    def read_region(self, location: Tuple[_GenericNumber, _GenericNumber], scaling: float,
                    size: Tuple[int, int]) -> PIL.Image:
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

        # Compute the scaling value between the closest high-res layer and a target layer.
        best_level = owsi.get_best_level_for_downsample(1 / scaling)
        relative_scaling = scaling * owsi.level_downsamples[best_level]
        best_level_size = owsi.level_dimensions[best_level]

        # Openslide doesn't feature float coordinates to extract a region.
        # We need to extract enough pixels and let PIL do the interpolation.
        # In the borders, the basis functions of other samples contribute to the final value.
        # PIL lanczos uses 3 pixels as support.
        # See pillow: https://git.io/JG0QD
        extra_pixels = 3 if scaling > 1 else int(3 / relative_scaling)
        native_location = location / relative_scaling
        native_size = size / relative_scaling

        # Compute extra paddings for exact interpolation.
        native_location_adapted = np.floor(native_location - extra_pixels).astype(int)
        native_location_adapted = _clip2size(native_location_adapted, best_level_size)
        native_size_adapted = np.ceil(native_location + native_size + extra_pixels).astype(int)
        native_size_adapted = _clip2size(native_size_adapted, best_level_size) - native_location_adapted

        # We extract the region via openslide with the required extra border
        region = owsi.read_region(tuple(native_location_adapted), best_level, tuple(native_size_adapted))

        # Within this region, there are a bunch of extra pixels, we interpolate to sample
        # the pixel in the right position to retain the right sample weight.
        fractional_coordinates = native_location - native_location_adapted
        box = (*fractional_coordinates, *(fractional_coordinates + native_size))
        return region.resize(size, resample=PIL.Image.LANCZOS, box=box)

    def get_level_view(self, scaling: _GenericNumber, box: _Box = None) -> SlideImageRegionView:
        """Return a pyramid region."""
        return SlideImageRegionView(self, scaling, box)

    @property
    def thumbnail(self, size: Tuple[int, int] = (512, 512)) -> PIL.Image:
        """Returns an RGB numpy thumbnail for the current slide.

        Parameters
        ----------
        size :
            Maximum bounding box for the thumbnail expressed as (width, height).
        """
        return self._openslide_wsi.get_thumbnail(*size)

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
    def base_dimensions(self) -> Tuple[int, int]:
        """Returns the highest resolution image size in pixels."""
        return self._openslide_wsi.dimensions

    @property
    def base_mpp(self) -> float:
        """Returns the microns per pixel of the high res image."""
        return self._min_native_mpp

    @property
    def magnification(self) -> int:
        """Returns the objective power at which the WSI was sampled."""
        return int(self._openslide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    @property
    def aspect_ratio(self) -> dict:
        """Returns width / height."""
        width, height = self.base_dimensions
        return width / height

    def __repr__(self) -> str:
        props = ("identifier", "vendor", "highest_resolution_mpp", "magnification")
        props_str = []
        for key in props:
            value = getattr(self, key)
            props_str.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(props_str)})"


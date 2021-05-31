# coding=utf-8
# Copyright (c) DLUP Contributors
"""Whole slide image manipulation objects.

In this module we take care of abstracting the access to whole slide images.
The main workhorses are the WholeSlideImage and TiledView classes
which respectively take care of simplyfing the access to pyramidal images in
a continuous ways and properties validation for dlup algorithms.
"""

import functools
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, TypeVar, Union

import numpy as np  # type: ignore
import openslide  # type: ignore
import PIL.Image  # type: ignore

_GenericFloatArray = Union[np.ndarray, Iterable[float]]
_GenericIntArray = Union[np.ndarray, Iterable[int]]

_TWholeSlidePyramidalImage = TypeVar("TWholeSlidePyramidalImage", bound="WholeSlidePyramidalImage")


class RegionView(ABC):
    """A generic image object from which you can extract a region.

    A unit 'U' is assumed to be consistent across this interface.
    Could be for instance pixels.
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
    def read_region(self, location: _GenericFloatArray, size: _GenericIntArray) -> PIL.Image:
        """Returns the region covered by the box.

        box coordinates are defined in U units.
        """
        pass


class _WholeSlideImageRegionView(RegionView):
    """Represents a wsi layer view."""

    def __init__(self, wsi: _TWholeSlidePyramidalImage, scaling: float):
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

    def read_region(self, location: _GenericFloatArray, size: _GenericIntArray) -> PIL.Image:
        """Returns a region in the level."""
        return self._wsi.read_region(location, self._scaling, size)


def _clip2size(a: np.ndarray, size: Tuple[int, int]):
    return np.clip(a, (0, 0), size)


class WholeSlidePyramidalImage:
    """Utility class to simplify whole-slide pyramidal images management.

    This helper class furtherly abstracts openslide access to WSIs
    by validating some of the properties and giving access
    to a continuous pyramid.
    Each horizontal slices of the pyramid can be accessed using a scaling value
    z as index.

    Example usage:

    ```python
    from dlup import WholeSlidePyramidalImage
    wsi = dlup.WholeSlideImage.open('path/to/slide.svs')
    wsi_level = wsi[0.5]
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
            raise RuntimeError(f"Could not parse mpp.")

        if not np.isclose(mpp[0], mpp[1]):
            raise RuntimeError("Cannot deal with slides having anisotropic mpps.")

        self._min_native_mpp = float(mpp[0])

    @classmethod
    def from_file_path(
        cls: _TWholeSlidePyramidalImage, wsi_file_path: pathlib.Path, identifier: Union[str, None] = None
    ) -> _TWholeSlidePyramidalImage:
        wsi = openslide.open_slide(str(wsi_file_path))
        # As default identifier we use a tuple (folder, filename)
        identifier = identifier if identifier is not None else wsi_file_path.parts[-2:]
        return cls(wsi, identifier)

    def read_region(self, location: _GenericFloatArray, scaling: float, size: _GenericIntArray) -> PIL.Image:
        """Return a pyramidal region.

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
        relative_scaling = scaling * np.asarray(owsi.level_downsamples[best_level])
        best_level_size = owsi.level_dimensions[best_level]

        # Openslide doesn't feature float coordinates to extract a region.
        # We need to extract enough pixels and let PIL do the interpolation.
        # In the borders, the basis functions of other samples contribute to the final value.
        # PIL lanczos seems to uses 3 pixels as support.
        extra_pixels = 3 if scaling > 1 else int(3 / relative_scaling)
        native_location = location / relative_scaling
        native_size = size / relative_scaling

        # Compute extra paddings for exact interpolation.
        native_location_adapted = np.floor(native_location - extra_pixels).astype(int)
        native_location_adapted = _clip2size(native_location_adapted, best_level_size)
        native_size_adapted = np.ceil(native_location + native_size + extra_pixels).astype(int)
        native_size_adapted = _clip2size(native_size_adapted, best_level_size)

        # We extract the region via openslide with the required extra border
        region = owsi.read_region(tuple(native_location_adapted), best_level, tuple(native_size_adapted))

        # Within this region, there are a bunch of extra pixels, we interpolate to sample
        # the pixel in the right position to retain the right sample weight.
        fractional_coordinates = native_location - native_location_adapted
        box = (*fractional_coordinates, *(fractional_coordinates + native_size))
        return region.resize(size, resample=PIL.Image.LANCZOS, box=box)

    def get_level_view(self, scaling: Union[float, int]) -> _WholeSlideImageRegionView:
        if scaling < 0:
            raise ValueError(f"Scaling value should always be greater than 0.")
        return _WholeSlideImageRegionView(self, scaling)

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
        return self._identifier

    @property
    def properties(self) -> dict:
        return self._openslide_wsi.properties

    @property
    def vendor(self) -> str:
        return self.properties["openslide.vendor"]

    @property
    def highest_resolution_dimensions(self) -> Tuple[int, int]:
        """Returns the highest resolution image size in pixels."""
        return self._openslide_wsi.dimensions

    @property
    def highest_resolution_mpp(self) -> float:
        """Returns the microns per pixel of the high res image."""
        return self._min_native_mpp

    @property
    def magnification(self) -> int:
        """Returns the objective power at which the WSI was sampled."""
        return int(self._openslide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    @property
    def aspect_ratio(self) -> dict:
        """Returns width / height."""
        dims = self.highest_resolution_dimensions
        return dims[0] / dims[1]

    def __repr__(self) -> str:
        props = ("identifier", "vendor", "highest_resolution_mpp", "magnification")
        props_str = []
        for key in props:
            value = getattr(self, key)
            props_str.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(props_str)})"


class TiledView:
    """This class takes care of creating a smart object to access a wsi tiles.

    Features, access via slices, indexes, given the tiling properties.
    """

    def __init__(self, region_view: RegionView, tile_size: Tuple[int, int], tile_overlap: Tuple[int, int]):
        self._region_view = region_view
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap

        # # Compute the grid.
        # stride = np.asarray(tile_size) - tile_overlap

        # # Same thing as computing the output shape of a convolution with padding zero and
        # # specified stride.
        # num_tiles = (subsampled_region_size - tile_size) / stride + 1

        # if border_mode == "crop":
        #     num_tiles = np.ceil(num_tiles).astype(int)
        #     tiled_size = (num_tiles - 1) * stride + tile_size
        #     overflow = tiled_size - subsampled_region_size
        # elif border_mode == "skip":
        #     num_tiles = np.floor(num_tiles).astype(int)
        #     overflow = np.asarray((0, 0))
        # else:
        #     raise ValueError(f"`border_mode` has to be one of `crop` or `skip`. Got {border_mode}.")

        # indices = [range(0, _) for _ in num_tiles]

    def __iter__(self):
        """Iterate through every tile."""
        pass

    def __len__(self) -> int:
        """Returns the total number of tiles."""
        pass

    def __getitem__(self, i: int) -> PIL.Image:
        pass

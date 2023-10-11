# Copyright (c) dlup contributors

"""Whole slide image access objects.

In this module we take care of abstracting the access to whole slide images.
The main workhorse is SlideImage which takes care of simplifying region extraction
of discrete-levels pyramidal images in a continuous way, supporting multiple different backends.
"""
from __future__ import annotations

import errno
import os
import pathlib
from enum import IntEnum
from types import TracebackType
from typing import Any, Literal, Optional, Type, TypeVar, cast

import numpy as np
import numpy.typing as npt
import PIL
import PIL.Image

from dlup import UnsupportedSlideError
from dlup._region import BoundaryMode, RegionView
from dlup.backends.common import AbstractSlideBackend
from dlup.experimental_backends import ImageBackend  # type: ignore
from dlup.types import GenericFloatArray, GenericIntArray, GenericNumber, GenericNumberArray, PathLike
from dlup.utils.image import check_if_mpp_is_valid

_Box = tuple[GenericNumber, GenericNumber, GenericNumber, GenericNumber]
_TSlideImage = TypeVar("_TSlideImage", bound="SlideImage")


class Resampling(IntEnum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1


class _SlideImageRegionView(RegionView):
    """Represents an image view tied to a slide image."""

    def __init__(
        self,
        wsi: _TSlideImage,
        scaling: GenericNumber,
        boundary_mode: BoundaryMode | None = None,
    ):
        """Initialize with a slide image object and the scaling level."""
        # Always call the parent init
        super().__init__(boundary_mode=boundary_mode)
        self._wsi = wsi
        self._scaling = scaling

    @property
    def mpp(self) -> float:
        """Returns the level effective mpp."""
        return self._wsi.mpp / self._scaling

    @property
    def size(self) -> tuple[int, ...]:
        """Size"""
        return self._wsi.get_scaled_size(self._scaling)

    def _read_region_impl(self, location: GenericFloatArray, size: GenericIntArray) -> PIL.Image.Image:
        """Returns a region of the level associated to the view."""
        x, y = location
        w, h = size
        return self._wsi.read_region((x, y), self._scaling, (w, h))


def _clip2size(
    a: npt.NDArray[np.int_ | np.float_], size: tuple[GenericNumber, GenericNumber]
) -> npt.NDArray[np.int_ | np.float_]:
    """Clip values from 0 to size boundaries."""
    return np.clip(a, (0, 0), size)


class SlideImage:
    """Utility class to simplify whole-slide pyramidal images management.

    This helper class furtherly abstracts openslide access to WSIs
    by validating some of the properties and giving access
    to a continuous pyramid. Layer values are interpolated from
    the closest high resolution layer.
    Each horizontal slices of the pyramid can be accessed using a scaling value
    z as index.

    Lifetime
    --------
    SlideImage is currently initialized and holds an openslide image object.
    The openslide wsi instance is automatically closed when gargbage collected.

    Examples
    --------
    >>> import dlup
    >>> wsi = dlup.SlideImage.from_file_path('path/to/slide.svs')
    """

    def __init__(self, wsi: AbstractSlideBackend, identifier: str | None = None, **kwargs: Any) -> None:
        """Initialize a whole slide image and validate its properties."""
        self._wsi = wsi
        self._identifier = identifier

        if kwargs.get("interpolator", None) is not None:
            interpolator = kwargs["interpolator"]
            if isinstance(interpolator, Resampling):
                interpolator = interpolator.name

            self._interpolator = PIL.Image.Resampling[interpolator]
        else:
            self._interpolator = PIL.Image.Resampling.LANCZOS

        if kwargs.get("overwrite_mpp", None) is not None:
            self._wsi.spacing = kwargs["overwrite_mpp"]

        if self._wsi.spacing is None:
            raise UnsupportedSlideError(
                f"The spacing of {identifier} cannot be derived from image and is "
                "not explicitly set in the `overwrite_mpp` parameter."
            )

        check_if_mpp_is_valid(*self._wsi.spacing)
        self._avg_native_mpp = (float(self._wsi.spacing[0]) + float(self._wsi.spacing[1])) / 2

    def close(self) -> None:
        """Close the underlying image."""
        self._wsi.close()

    def __enter__(self) -> "SlideImage":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        self.close()
        return False

    @classmethod
    def from_file_path(
        cls: Type[_TSlideImage],
        wsi_file_path: PathLike,
        identifier: str | None = None,
        backend: ImageBackend = ImageBackend.PYVIPS,
        **kwargs: Any,
    ) -> _TSlideImage:
        wsi_file_path = pathlib.Path(wsi_file_path)

        if isinstance(backend, str):
            backend = ImageBackend[backend]

        if not wsi_file_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(wsi_file_path))
        try:
            wsi = backend(wsi_file_path)
        except UnsupportedSlideError:
            raise UnsupportedSlideError(f"Unsupported file: {wsi_file_path}")

        return cls(wsi, str(wsi_file_path) if identifier is None else identifier, **kwargs)

    def read_region(
        self,
        location: GenericNumberArray | tuple[GenericNumber, GenericNumber],
        scaling: float,
        size: npt.NDArray[np.int_] | tuple[int, int],
    ) -> PIL.Image.Image:
        """Return a region at a specific scaling level of the pyramid.

        A typical slide is made of several levels at different mpps.
        In normal cirmustances, it's not possible to retrieve an image of
        intermediate mpp between these levels. This method takes care of
        sumbsampling the closest high resolution level to extract a target
        region via interpolation.

        Once the best layer is selected, a native resolution region
        is extracted, with enough padding to include the samples necessary to downsample
        the final region (considering LANCZOS interpolation method basis functions).

        The steps are approximately the following:

        1. Map the region that we want to extract to the below layer.
        2. Add some extra values (left and right) to the native region we want to extract
           to take into account the interpolation samples at the border ("native_extra_pixels").
        3. Map the location to the level0 coordinates, floor it to add extra information
           on the left (level_zero_location_adapted).
        4. Re-map the integral level-0 location to the native_level.
        5. Compute the right bound of the region adding the native_size and extra pixels (native_size_adapted).
           The size is also clipped so that any extra pixel will fit within the native level.
        6. Since the native_size_adapted needs to be passed to openslide and has to be an integer, we ceil it
           to avoid problems with possible overflows of the right boundary of the target region being greater
           than the right boundary of the sample region
           (native_location + native_size > native_size_adapted + native_location_adapted).
        7. Crop the target region from within the sampled region by computing the relative
           coordinates (fractional_coordinates).

        Parameters
        ----------
        location :
            Location from the top left (x, y) in pixel coordinates given at the requested scaling.
        scaling :
            The scaling to be applied compared to level 0.
        size :
            Region size of the resulting region.

        Returns
        -------
        PIL.Image
            The extract region.

        Examples
        --------
        The locations are defined at the requested scaling (with respect to level 0), so if we want to extract at
        location ``(location_x, location_y)`` of a scaling 0.5 (with respect to level 0), and have
        resulting tile size of ``(tile_size, tile_size)`` with a scaling factor of 0.5, we can use:
        >>>  wsi.read_region(location=(coordinate_x, coordinate_y), scaling=0.5, size=(tile_size, tile_size))
        """
        owsi = self._wsi
        location = np.asarray(location)
        size = np.asarray(size)
        level_size = np.array(self.get_scaled_size(scaling))

        if (size < 0).any():
            raise ValueError("Size values must be greater than zero.")

        if ((location < 0) | ((location + size) > level_size)).any():
            raise ValueError("Requested region is outside level boundaries.")

        native_level = owsi.get_best_level_for_downsample(1 / scaling)
        native_level_size = owsi.level_dimensions[native_level]
        native_level_downsample = owsi.level_downsamples[native_level]
        native_scaling = scaling * owsi.level_downsamples[native_level]
        native_location = location / native_scaling
        native_size = size / native_scaling

        # OpenSlide doesn't feature float coordinates to extract a region.
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

        # By casting to int we introduce a small error in the right boundary leading
        # to a smaller region which might lead to the target region to overflow from the sampled
        # region.
        native_size_adapted = np.ceil(native_size_adapted).astype(int)

        # We extract the region via openslide with the required extra border
        region = owsi.read_region(
            (level_zero_location_adapted[0], level_zero_location_adapted[1]),
            native_level,
            (native_size_adapted[0], native_size_adapted[1]),
        )

        # Within this region, there are a bunch of extra pixels, we interpolate to sample
        # the pixel in the right position to retain the right sample weight.
        # We also need to clip to the border, as some readers (e.g mirax) have one pixel less at the border.
        fractional_coordinates = native_location - native_location_adapted
        # TODO: This clipping could be in an error in OpenSlide mirax reader, but it's a minor thing for now
        box = (
            *fractional_coordinates,
            *np.clip(
                (fractional_coordinates + native_size),
                a_min=0,
                a_max=np.asarray(region.size),
            ),
        )
        box = cast(tuple[float, float, float, float], box)
        size = cast(tuple[int, int], size)
        return region.resize(size, resample=self._interpolator, box=box)

    def get_scaled_size(self, scaling: GenericNumber) -> tuple[int, int]:
        """Compute slide image size at specific scaling."""
        size = np.array(self.size) * scaling
        return cast(tuple[int, int], tuple(size.astype(int)))

    def get_mpp(self, scaling: float) -> float:
        """Returns the respective mpp from the scaling."""
        return self._avg_native_mpp / scaling

    def get_scaling(self, mpp: float | None) -> float:
        """Inverse of get_mpp()."""
        if not mpp:
            return 1.0
        return self._avg_native_mpp / mpp

    def get_scaled_view(self, scaling: GenericNumber) -> _SlideImageRegionView:
        """Returns a RegionView at a specific level."""
        return _SlideImageRegionView(self, scaling)

    def get_thumbnail(self, size: tuple[int, int] = (512, 512)) -> PIL.Image.Image:
        """Returns an RGB `PIL.Image.Image` thumbnail for the current slide.

        Parameters
        ----------
        size : tuple[int, int]
            Maximum bounding box for the thumbnail expressed as (width, height).

        Returns
        -------
        PIL.Image.Image
            The thumbnail as a PIL image.
        """
        return self._wsi.get_thumbnail(size)

    @property
    def thumbnail(self) -> PIL.Image.Image:
        """Returns the thumbnail."""
        return self.get_thumbnail()

    @property
    def identifier(self) -> str | None:
        """Returns a user-defined identifier."""
        return self._identifier

    @property
    def properties(self) -> dict[str, str | int | float | bool] | None:
        """Returns any extra associated properties with the image."""
        return self._wsi.properties

    @property
    def vendor(self) -> str | None:
        """Returns the scanner vendor."""
        return self._wsi.vendor

    @property
    def size(self) -> tuple[int, int]:
        """Returns the highest resolution image size in pixels. Returns in (width, height)."""
        return self._wsi.dimensions

    @property
    def mpp(self) -> float:
        """Returns the microns per pixel of the high res image."""
        return self._avg_native_mpp

    @property
    def magnification(self) -> float | None:
        """Returns the objective power at which the WSI was sampled."""
        return self._wsi.magnification

    @property
    def aspect_ratio(self) -> float:
        """Returns width / height."""
        width, height = self.size
        return width / height

    @property
    def slide_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide. These can be smaller than the image itself.
        These bounds are in the format (x, y), (width, height), and are defined at level 0 of the image.
        """
        return self._wsi.slide_bounds

    def __repr__(self) -> str:
        """Returns the SlideImage representation and some of its properties."""
        props = ("identifier", "vendor", "mpp", "magnification", "size")
        props_str = []
        for key in props:
            value = getattr(self, key, None)
            props_str.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(props_str)})"

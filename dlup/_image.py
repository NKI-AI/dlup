# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whole slide image access objects.

In this module we take care of abstracting the access to whole slide images.
The main workhorse is SlideImage which takes care of simplifying region extraction
of discrete-levels pyramidal images in a continuous way, supporting multiple different backends.
"""

from __future__ import annotations

import errno
import io
import os
import pathlib
from enum import Enum
from types import TracebackType
from typing import Any, Literal, Optional, Type, TypeVar, cast

import fim
import numpy as np
import numpy.typing as npt
from dlup._exceptions import UnsupportedSlideError
from dlup._region import BoundaryMode, RegionView
from dlup._types import GenericFloatArray, GenericIntArray, GenericNumber, GenericNumberArray, PathLike
from dlup.backends.common import AbstractSlideBackend
from dlup.utils.backends import ImageBackend
from dlup.utils.image import check_if_mpp_is_valid

_Box = tuple[GenericNumber, GenericNumber, GenericNumber, GenericNumber]
_TSlideImage = TypeVar("_TSlideImage", bound="SlideImage")


class Resampling(str, Enum):
    """Resampling methods SlideImage (e.g. for images or masks)."""

    NEAREST = "NEAREST"
    LANCZOS = "LANCZOS"
    MAGIC2021 = "MAGIC2021"


_RESAMPLE_TO_FIM = {
    Resampling.NEAREST: fim.KernelType.NEAREST,
    Resampling.LANCZOS: fim.KernelType.LANCZOS3,
    Resampling.MAGIC2021: fim.KernelType.MAGIC2021,
}
_KERNEL_RADIUS_MAP = {
    fim.KernelType.NEAREST: 1,
    fim.KernelType.LANCZOS2: 2,
    fim.KernelType.LANCZOS3: 3,
    fim.KernelType.MAGIC2021: 5,  # ceil(4.5)
}


class SlideImageView(RegionView):
    """Represents an image view tied to a slide image at a specific scaling or MPP level.

    This class provides a convenient interface for reading regions from a slide at a fixed
    resolution without having to repeatedly specify the scaling parameter.

    Examples
    --------
    >>> import dlup
    >>> wsi = dlup.SlideImage.from_file_path('path/to/slide.svs')
    >>> # Create a view at 0.5 microns per pixel
    >>> view = wsi.get_view_at_mpp(0.5)
    >>> # Read a region at this MPP level
    >>> region = view.read_region((0, 0), (512, 512))
    >>> # Or create a view at a specific scaling
    >>> view = wsi.get_view_at_scaling(0.25)
    >>> region = view.read_region((100, 100), (256, 256))
    """

    def __init__(
        self,
        wsi: _TSlideImage,
        scaling: GenericNumber,
        boundary_mode: BoundaryMode | None = None,
    ):
        """Initialize with a slide image object and the scaling level.

        Parameters
        ----------
        wsi : SlideImage
            The parent slide image object.
        scaling : GenericNumber
            The scaling factor relative to level 0.
        boundary_mode : BoundaryMode, optional
            The boundary mode to use when reading regions that extend beyond the image bounds.
        """
        # Always call the parent init
        super().__init__(boundary_mode=boundary_mode)
        self._wsi = wsi
        self._scaling = scaling

    @property
    def mpp(self) -> float:
        """Returns the effective microns per pixel at this view's scaling level."""
        return self._wsi.mpp / self._scaling

    @property
    def size(self) -> tuple[int, int]:
        """Returns the size of the image at this view's scaling level."""
        return self._wsi.get_scaled_size(self._scaling)

    def _read_region_impl(self, location: GenericFloatArray, size: GenericIntArray):
        """Returns a region of the level associated to the view as a fim.Image."""
        x, y = location
        w, h = size
        return self._wsi.read_region((x, y), self._scaling, (w, h))


def _clip2size(
    a: npt.NDArray[np.int_ | np.float64], size: tuple[GenericNumber, GenericNumber]
) -> npt.NDArray[np.int_ | np.float64]:
    """Clip values from 0 to size boundaries."""
    return np.clip(a, (0, 0), size)


def _check_size_and_location(
    location: npt.NDArray[np.int_ | np.float64],
    size: npt.NDArray[np.int_ | np.float64],
    level_size: npt.NDArray[np.int_],
) -> None:
    """
    Check if the size and location are within bounds for the given level size.

    Parameters
    ----------
    size : npt.NDArray[np.int_ | np.float64]
        The size of the region to extract.
    location : npt.NDArray[np.int_ | np.float64]
        The location of the region to extract.
    level_size : npt.NDArray[np.int_]
        The size of the level.

    Raises
    ------
    ValueError
        If the size is negative or if the location is outside the level boundaries.

    Returns
    -------
    None
    """
    if (size < 0).any():
        raise ValueError(f"Size values must be greater than zero. Got {size}")

    if ((location < 0) | ((location + np.floor(size)) > level_size)).any():
        raise ValueError(
            f"Requested region is outside level boundaries. "
            f"{location.tolist()} + {size.tolist()} (={(location + np.floor(size)).tolist()}) > {level_size.tolist()}."
        )


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
    Basic usage:

    >>> import dlup
    >>> wsi = dlup.SlideImage.from_file_path('path/to/slide.svs')

    Direct region reading (traditional API):

    >>> region = wsi.read_region((0, 0), scaling=0.5, size=(512, 512))

    View-based API (recommended for multiple reads at same resolution):

    >>> # Create a view at a specific MPP
    >>> view = wsi.get_view_at_mpp(0.5)
    >>> region1 = view.read_region((0, 0), (512, 512))
    >>> region2 = view.read_region((512, 512), (512, 512))
    >>>
    >>> # Or create a view at a specific scaling
    >>> view = wsi.get_view_at_scaling(0.25)
    >>> region = view.read_region((100, 100), (256, 256))
    """

    def __init__(
        self,
        wsi: AbstractSlideBackend,
        identifier: str | None = None,
        interpolator: Optional[Resampling] | str = Resampling.LANCZOS,
        overwrite_mpp: Optional[tuple[float, float]] = None,
        apply_color_profile: bool = False,
    ) -> None:
        """Initialize a whole slide image and validate its properties. This class allows to read whole-slide images
        at any arbitrary resolution. This class can read images from any backend that implements the
        AbstractSlideBackend interface.

        Parameters
        ----------
        wsi : AbstractSlideBackend
            The slide object.
        identifier : str, optional
            A user-defined identifier for the slide, used in e.g. exceptions.
        interpolator : Resampling, optional
            The interpolator to use when reading regions. For images typically LANCZOS is the best choice. Masks
            can use NEAREST. By default, will use LANCZOS
        overwrite_mpp : tuple[float, float], optional
            Overwrite the mpp of the slide. For instance, if the mpp is not available, or when sourcing from
            and external database.
        apply_color_profile : bool
            Whether to apply the color profile to the output regions.

        Raises
        ------
        UnsupportedSlideError
            If the slide is not supported, or when the mpp is not valid (too anisotropic).

        Returns
        -------
        None

        """
        self._wsi = wsi
        self._identifier = identifier

        self._interpolator = interpolator if interpolator else Resampling.LANCZOS

        if overwrite_mpp is not None:
            self._wsi.spacing = overwrite_mpp

        if self._wsi.spacing is None:
            raise UnsupportedSlideError(
                f"The spacing of {identifier} cannot be derived from image and is "
                "not explicitly set in the `overwrite_mpp` parameter."
            )

        check_if_mpp_is_valid(*self._wsi.spacing)
        self._avg_native_mpp = (float(self._wsi.spacing[0]) + float(self._wsi.spacing[1])) / 2
        self._apply_color_profile = apply_color_profile

    @property
    def interpolator(self) -> Resampling:
        """Returns the interpolator used for processing the regions."""
        return self._interpolator if isinstance(self._interpolator, Resampling) else Resampling[self._interpolator]

    def close(self) -> None:
        """Close the underlying image."""
        self._wsi.close()

    @property
    def color_profile(self) -> io.BytesIO | None:
        """Returns the ICC profile of the image.
        Each image in the pyramid has the same ICC profile, but the associated images might have their own.

        # TODO: Vips can apply the color profile directly when loading the image!

        Examples
        --------
        >>> import dlup
        >>> from PIL import ImageCms
        >>> wsi = dlup.SlideImage.from_file_path("path/to/slide.svs")
        >>> region = wsi.read_region((0, 0), 1.0, (512, 512))
        >>> to_profile = ImageCms.createProfile("sRGB")
        >>> color_profile = PIL.ImageCms.getOpenProfile(wsi.color_profile)
        >>> intent = ImageCms.getDefaultIntent(color_profile)
        >>> transform = ImageCms.buildTransform(color_profile, to_profile, "RGBA", "RGBA", intent, 0)
        >>> # Now you can apply the transform to the region as a PIL Image (note that the output is fim.Image)
        >>> ImageCms.applyTransform(region, transform, True)

        Returns
        -------
        io.BytesIO
            The ICC profile of the image.
        """
        return getattr(self._wsi, "color_profile", None)

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
        backend: ImageBackend | Type[AbstractSlideBackend] | str = ImageBackend.OPENSLIDE,
        **kwargs: Any,
    ) -> _TSlideImage:
        wsi_file_path = pathlib.Path(wsi_file_path).resolve()
        if not wsi_file_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(wsi_file_path))

        if isinstance(backend, str):
            backend = ImageBackend[backend]

        # Adjust how the backend is used depending on its type
        if isinstance(backend, ImageBackend):
            backend_callable = backend.value  # Get the callable from Enum
        else:
            backend_callable = backend  # Directly use the class if it's a subclass of AbstractSlideBackend

        try:
            wsi = backend_callable(wsi_file_path)  # Instantiate the backend with the path
        except UnsupportedSlideError as exc:
            raise UnsupportedSlideError(f"Unsupported file: {wsi_file_path}") from exc

        return cls(wsi, identifier if identifier is not None else str(wsi_file_path), **kwargs)

    def read_region(
        self,
        location: GenericNumberArray | tuple[GenericNumber, GenericNumber],
        scaling: float,
        size: npt.NDArray[np.int_] | tuple[int, int],
    ) -> fim.Image:
        """Return a region at a specific scaling level of the pyramid.

        A typical slide is made of several levels at different mpps.
        In normal cirmustances, it's not possible to retrieve an image of
        intermediate mpp between these levels. This method takes care of
        subsampling the closest high resolution level to extract a target
        region via interpolation.

        Once the best layer is selected, a native resolution region
        is extracted, with enough padding to include the samples necessary to downsample
        the final region, which is subsequently cropped to the target size.

        Parameters
        ----------
        location : tuple[int, int]
            Location from the top left (x, y) in pixel coordinates given at the requested scaling.
        scaling : float
            The scaling to be applied compared to level 0.
        size : tuple[int, int]
            Region size of the resulting region.

        Returns
        -------
        fim.Image
            The extract region as a fim.Image.

        Notes
        -----
        The resampling kernel used depends on the `interpolator` setting:
        - NEAREST: Nearest neighbor (no interpolation)
        - LANCZOS: Lanczos3 kernel (radius 3)
        - MAGIC2021: Magic Kernel Sharp 2021 (radius 4.5)

        Examples
        --------
        The locations are defined at the requested scaling (with respect to level 0), so if we want to extract at
        location ``(location_x, location_y)`` of a scaling 0.5 (with respect to level 0), and have
        resulting tile size of ``(tile_size, tile_size)`` with a scaling factor of 0.5, we can use:
        >>>  wsi.read_region(location=(coordinate_x, coordinate_y), scaling=0.5, size=(tile_size, tile_size))
        """

        wsi = self._wsi
        location = np.asarray(location, dtype=float)  # (x,y) at requested scaling
        size = np.asarray(size, dtype=float)  # (w,h) at requested scaling
        level_sz = np.array(self.get_scaled_size(scaling), dtype=int)

        _check_size_and_location(location, size, level_sz)

        # Step 1: Find the closest level for downsample
        native_level = wsi.get_best_level_for_downsample(1.0 / scaling)
        native_dims = np.array(wsi.level_dimensions[native_level], dtype=float)  # (W,H) in native pixels
        downsample = float(wsi.level_downsamples[native_level])  # level0_px = native_px * downsample

        # Step 2: Scale the region to that level (keep everything in float)
        native_scaling = scaling * downsample  # requested-scaling px per native px
        native_location = location / native_scaling  # float (x, y)
        native_size = size / native_scaling  # float (w, h)

        # Step 3: Read that region plus a few pixels depending on the resize method
        # Determine padding based on kernel radius
        fim_kernel = _RESAMPLE_TO_FIM[self.interpolator]
        kernel_r = _KERNEL_RADIUS_MAP.get(fim_kernel, 3)
        # Adjust padding for upsampling case
        pad = float(kernel_r) if native_scaling >= 1.0 else np.ceil(kernel_r / native_scaling)

        # Compute read bounds in float
        read_start_float = native_location - pad  # (x, y)
        read_end_float = native_location + native_size + pad  # (x+w, y+h)

        # Convert to integers for read call - floor start, ceil end to ensure we get enough pixels
        read_start_int = np.floor(read_start_float).astype(int)
        read_end_int = np.ceil(read_end_float).astype(int)

        # Clamp to valid range
        read_start_int = np.maximum(read_start_int, 0)
        read_end_int = np.minimum(read_end_int, native_dims.astype(int))
        read_size_int = read_end_int - read_start_int

        # Read the padded region
        fim_region = wsi.read_region(
            (int(read_start_int[0]), int(read_start_int[1])),
            native_level,
            (int(read_size_int[0]), int(read_size_int[1])),
        )

        # Step 4: Crop the extra pixels
        # Compute crop box in float to capture correct location (cast to int when calling crop)
        crop_offset_float = native_location - read_start_int.astype(float)  # Where the region starts in the read
        crop_size_float = native_size  # Size we want to crop

        # Cast to int for crop call (eventually crop will support float coordinates)
        crop_offset_int = np.floor(crop_offset_float).astype(int)
        crop_size_int = np.ceil(crop_size_float).astype(int)

        # Ensure we don't exceed what we read
        crop_size_int = np.minimum(crop_size_int, read_size_int - crop_offset_int)

        crop_region = fim_region.crop(
            (int(crop_offset_int[0]), int(crop_offset_int[1])),
            (int(crop_size_int[0]), int(crop_size_int[1])),
        )

        # Final resize to the requested output size
        target_w, target_h = int(size[0]), int(size[1])
        out = crop_region.resize(target_w, target_h, kernel=fim_kernel)

        return out

    def get_scaled_size(self, scaling: GenericNumber, limit_bounds: Optional[bool] = False) -> tuple[int, int]:
        """Compute slide image size at specific scaling.

        Parameters
        -----------
        scaling: GenericNumber
            The factor by which the image needs to be scaled.

        limit_bounds: Optional[bool]
            If True, the scaled size will be calculated using the slide bounds of the whole slide image.
            This is generally the specific area within a whole slide image where we can find the tissue specimen.

        Returns
        -------
        size: tuple[int, int]
            The scaled size of the image.
        """
        if limit_bounds:
            _, bounded_size = self.slide_bounds
            size = int(bounded_size[0] * scaling), int(bounded_size[1] * scaling)
        else:
            size = int(self.size[0] * scaling), int(self.size[1] * scaling)
        return size

    def get_mpp(self, scaling: float) -> float:
        """Returns the respective mpp from the scaling."""
        return self._avg_native_mpp / scaling

    def get_closest_native_level(self, mpp: float) -> int:
        """Returns the closest native level to the given mpp.

        Returns
        -------
        int
            The closest level.
        """
        closest_index, _ = min(enumerate(self._wsi.level_spacings), key=lambda x: abs((x[1][0] + x[1][1]) / 2 - mpp))
        return closest_index

    def get_closest_native_mpp(self, mpp: float) -> tuple[float, float]:
        """Returns the closest native mpp to the given mpp.

        Returns
        -------
        tuple[float, float]
            The closest mpp in the format (mpp_x, mpp_y).
        """
        return self._wsi.level_spacings[self.get_closest_native_level(mpp)]

    def get_scaling(self, mpp: float | None) -> float:
        """Inverse of get_mpp()."""
        if not mpp:
            return 1.0
        return self._avg_native_mpp / mpp

    def get_view_at_mpp(self, mpp: float) -> SlideImageView:
        """Returns a SlideImageView at a specific MPP (microns per pixel) level.

        This provides a more ergonomic API for reading regions at a fixed resolution.
        Once you have a view, you can call `read_region(location, size)` without
        having to repeatedly specify the MPP or scaling.

        Parameters
        ----------
        mpp : float
            The microns per pixel for the view.

        Returns
        -------
        SlideImageView
            A view at the specified MPP level.

        Examples
        --------
        >>> import dlup
        >>> wsi = dlup.SlideImage.from_file_path('path/to/slide.svs')
        >>> view = wsi.get_view_at_mpp(0.5)
        >>> region = view.read_region((0, 0), (512, 512))
        """
        scaling = self.get_scaling(mpp)
        return SlideImageView(self, scaling)

    def get_view_at_scaling(self, scaling: GenericNumber) -> SlideImageView:
        """Returns a SlideImageView at a specific scaling level.

        This provides a more ergonomic API for reading regions at a fixed resolution.
        Once you have a view, you can call `read_region(location, size)` without
        having to repeatedly specify the scaling.

        Parameters
        ----------
        scaling : GenericNumber
            The scaling factor relative to level 0 (e.g., 0.5 means half the resolution).

        Returns
        -------
        SlideImageView
            A view at the specified scaling level.

        Examples
        --------
        >>> import dlup
        >>> wsi = dlup.SlideImage.from_file_path('path/to/slide.svs')
        >>> view = wsi.get_view_at_scaling(0.25)
        >>> region = view.read_region((100, 100), (256, 256))
        """
        return SlideImageView(self, scaling)

    def get_thumbnail(self, size: tuple[int, int] = (512, 512)) -> fim.Image:
        """Returns an RGB `fim.Image` thumbnail for the current slide.

        This method respects the interpolator setting of the SlideImage.

        Parameters
        ----------
        size : tuple[int, int]
            Maximum bounding box for the thumbnail expressed as (width, height).

        Returns
        -------
        fim.Image
            The thumbnail as a fim.Image image.
        """
        # Calculate the scaling needed to fit within size
        downsample = max(*(dim / thumb for dim, thumb in zip(self.size, size)))
        scaling = 1.0 / downsample

        # Get the scaled size while preserving aspect ratio
        scaled_size = self.get_scaled_size(scaling)

        # Use read_region which respects the interpolator
        thumbnail = self.read_region((0, 0), scaling, scaled_size)

        # Get dimensions
        thumb_dims = thumbnail.dimensions
        thumb_channels = thumb_dims[2]

        # If 4 channels (RGBA), flatten to RGB
        if thumb_channels == 4:
            array = thumbnail.to_numpy()
            array_rgb = array[:, :, :3].copy()
            thumbnail = fim.Image.from_numpy(array_rgb)

        return thumbnail

    @property
    def thumbnail(self) -> fim.Image:
        """Returns the thumbnail with a bounding box of (512, 512)."""
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

    def get_scaled_slide_bounds(self, scaling: float) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide at a specific scaling level. This takes the slide bounds into account
        and scales them to the appropriate scaling level.

        Parameters
        ----------
        scaling : float
            The scaling level to use.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int]]
            The slide bounds at the given scaling level.
        """
        offset, size = self.slide_bounds
        offset = (int(scaling * offset[0]), int(scaling * offset[1]))
        size = (int(scaling * size[0]), int(scaling * size[1]))
        return offset, size

    def __repr__(self) -> str:
        """Returns the SlideImage representation and some of its properties."""
        props = ("identifier", "vendor", "mpp", "magnification", "size", "interpolator", "backend")
        props_str = []
        for key in props:
            if key == "backend":
                value = self._wsi.__class__.__name__
            else:
                value = getattr(self, key, "N/A")
            props_str.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(props_str)})"

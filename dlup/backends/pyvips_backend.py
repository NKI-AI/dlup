# Copyright (c) dlup contributors
from __future__ import annotations

import io
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openslide
import pyvips
from packaging.version import Version

from dlup import UnsupportedSlideError
from dlup.backends.common import AbstractSlideBackend
from dlup.types import PathLike
from dlup.utils.image import check_if_mpp_is_valid

PYVIPS_ASSOCIATED_IMAGES = "slide-associated-images"
PYVIPS_ICC_PROFILE_DATA = "icc-profile-data"


def open_slide(filename: PathLike) -> "PyVipsSlide":
    """
    Read slide with pyvips.

    Parameters
    ----------
    filename : PathLike
        Path to image.
    """
    return PyVipsSlide(filename)


class PyVipsSlide(AbstractSlideBackend):
    """
    Backend for pyvips.
    """

    def __init__(self, filename: PathLike, load_with_openslide: bool = False) -> None:
        """
        Parameters
        ----------
        filename : PathLike
            Path to image.
        """
        super().__init__(filename)
        self._filename = str(filename)

        if not load_with_openslide:
            self._image = pyvips.Image.new_from_file(self._filename, access="sequential")
        else:
            self._image = pyvips.Image.openslideload(self._filename)

        self._images: list[pyvips.Image]
        self._loader = self._image.get("vips-loader")
        self._shapes: List[Tuple[int, int]] = []
        self._downsamples: List[float] = []
        self._spacings: List[Tuple[float, float]] = []
        self.__slide_bounds = (0, 0), (0, 0)
        self._load_metadata()

    def _load_metadata(self) -> None:
        if self._loader == "tiffload":
            self._read_as_tiff(self._filename)
        elif self._loader == "openslideload":
            self._read_as_openslide(self._filename)
        else:
            raise NotImplementedError(f"Loader {self._loader} is not implemented.")

    def _read_as_tiff(self, path: PathLike) -> None:
        self._level_count = int(self._image.get_value("n-pages"))
        self._images = [self._image] + [
            pyvips.Image.tiffload(str(path), page=level) for level in range(1, self._level_count)
        ]

        unit_dict = {"cm": 1000, "centimeter": 1000}
        self._downsamples.append(1.0)
        for idx, image in enumerate(self._images):
            mpp_x = unit_dict.get(image.get("resolution-unit"), 0) / float(image.get("xres"))
            mpp_y = unit_dict.get(image.get("resolution-unit"), 0) / float(image.get("yres"))
            check_if_mpp_is_valid(mpp_x, mpp_y)

            self._spacings.append((mpp_x, mpp_y))
            if idx >= 1:
                downsample = mpp_x / self._spacings[0][0]
                self._downsamples.append(downsample)
            self._shapes.append((image.get("width"), image.get("height")))

        self.__slide_bounds = (0, 0), self._shapes[0]

    def _read_as_openslide(self, path: PathLike) -> None:
        self._level_count = int(self._image.get("openslide.level-count"))
        self._images = [self._image] + [
            pyvips.Image.openslideload(str(path), level=level) for level in range(1, self._level_count)
        ]

        for idx, image in enumerate(self._images):
            openslide_shape = (
                int(image.get(f"openslide.level[{idx}].width")),
                int(image.get(f"openslide.level[{idx}].height")),
            )
            pyvips_shape = (image.width, image.height)
            if not openslide_shape == pyvips_shape:
                raise UnsupportedSlideError(
                    f"Reading {path} failed as openslide metadata reports different shapes than pyvips. "
                    f"Got {openslide_shape} and {pyvips_shape}."
                )

            self._shapes.append(pyvips_shape)

        for idx, image in enumerate(self._images):
            self._downsamples.append(float(image.get(f"openslide.level[{idx}].downsample")))

        mpp_x, mpp_y = None, None
        available_fields = self._images[0].get_fields()
        if openslide.PROPERTY_NAME_MPP_X in available_fields and openslide.PROPERTY_NAME_MPP_Y in available_fields:
            mpp_x = float(self._images[0].get(openslide.PROPERTY_NAME_MPP_X))
            mpp_y = float(self._images[0].get(openslide.PROPERTY_NAME_MPP_Y))

        if mpp_x is not None and mpp_y is not None:
            check_if_mpp_is_valid(mpp_x, mpp_y)
            self._spacings = [(np.array([mpp_x, mpp_y]) * downsample).tolist() for downsample in self._downsamples]
        else:
            warnings.warn(
                f"{path} does not have a parseable spacings property. You can overwrite it with `.mpp = (mpp_x, mpp_y)."
            )

        if (
            openslide.PROPERTY_NAME_BOUNDS_X in available_fields
            and openslide.PROPERTY_NAME_BOUNDS_Y in available_fields
        ):
            offset = int(self._images[0].get(openslide.PROPERTY_NAME_BOUNDS_X)), int(
                self._images[0].get(openslide.PROPERTY_NAME_BOUNDS_Y)
            )
        else:
            offset = (0, 0)

        if (
            openslide.PROPERTY_NAME_BOUNDS_WIDTH in available_fields
            and openslide.PROPERTY_NAME_BOUNDS_HEIGHT in available_fields
        ):
            size = int(self._images[0].get(openslide.PROPERTY_NAME_BOUNDS_WIDTH)), int(
                self._images[0].get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT)
            )
        else:
            size = self._shapes[0]

        self.__slide_bounds = offset, size

    @property
    def mode(self) -> Optional[str]:
        """Returns the mode of the image."""
        bands = self._image.bands
        return {1: "L", 3: "RGB", 4: "RGBA"}.get(bands, None)

    @property
    def slide_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Returns the bounds of the slide as ((x,y), (width, height)). These can be smaller than the image itself."""
        return self.__slide_bounds

    @property
    def spacing(self) -> Optional[Tuple[float, float]]:
        return self._spacings[0] if self._spacings else None

    @spacing.setter
    def spacing(self, value: Tuple[float, float]) -> None:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("`.spacing` has to be of the form (mpp_x, mpp_y).")

        mpp_x, mpp_y = value
        check_if_mpp_is_valid(mpp_x, mpp_y)
        self._spacings = [(np.array([mpp_x, mpp_y]) * downsample).tolist() for downsample in self._downsamples]

    @property
    def properties(self) -> Dict[str, Any]:
        """Metadata about the image as given by pyvips,
        which includes openslide tags in case openslide is the selected reader."""

        avoid_keys = [PYVIPS_ICC_PROFILE_DATA, PYVIPS_ASSOCIATED_IMAGES]
        return {key: self._image.get_value(key) for key in self._image.get_fields() if key not in avoid_keys}

    @property
    def color_profile(self) -> Optional[io.BytesIO]:
        """
        Returns the color profile of the image if available. Otherwise returns None.

        TODO
        ----
        The color profile can directly be applied in the read_region, so this needs to be implemented in such a way.

        Returns
        -------
        ImageCmsProfile, optional
            The color profile of the image.
        """
        if Version(openslide.__library_version__) < Version("4.0.0"):
            warnings.warn(
                "Color profile support is only available for openslide >= 4.0.0. "
                f"You have version {openslide.__library_version__}. "
                "Please update your openslide installation if you want to use this feature (recommended)."
            )
            return None

        if "icc-profile-data" not in self._image.get_fields():
            return None

        profile_data = self._image.get("icc-profile-data")
        return io.BytesIO(profile_data)

    @property
    def magnification(self) -> Optional[float]:
        """Returns the objective power at which the WSI was sampled."""
        if self._loader == "openslideload":
            if openslide.PROPERTY_NAME_OBJECTIVE_POWER in self._image.get_fields():
                return float(self._image.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER))
        return None

    @property
    def vendor(self) -> Optional[str]:
        """Returns the scanner vendor."""
        if self._loader == "openslideload":
            return str(self._image.get(openslide.PROPERTY_NAME_VENDOR))
        return None

    @property
    def associated_images(self) -> dict[str, pyvips.Image]:
        """Images associated with this whole-slide image."""
        if self._loader != "openslideload":
            return {}

        if PYVIPS_ASSOCIATED_IMAGES not in self._image.get_fields():
            return {}

        associated_image_names = self._image.get(PYVIPS_ASSOCIATED_IMAGES)
        associated_images = {k: None for k in associated_image_names.split(",")}

        for name in associated_image_names.split(","):
            associated_images[name] = pyvips.Image.openslideload(self._filename, associated=name)

        return associated_images

    def set_cache(self, cache: Any) -> None:
        raise NotImplementedError

    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
        """
        Return the best level for displaying the given image level.

        Parameters
        ----------
        coordinates : tuple[int, int]
            Coordinates of the region in level 0.
        level : int
            Level of the image pyramid.
        size : tuple[int, int]
            Size of the region to be extracted.

        Returns
        -------
        pyvips.Image
            The requested region.
        """
        x, y = coordinates
        width, height = size
        ratio = self._downsamples[level]
        region = self._images[level].crop(int(x // ratio), int(y // ratio), width, height)
        return region

    def close(self) -> None:
        """Close the underlying slide"""
        self._images.clear()

# type: ignore
# Copyright (c) dlup contributors
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import openslide
import PIL.Image
import pyvips

from dlup import UnsupportedSlideError
from dlup.backends.common import AbstractSlideBackend, numpy_to_pil
from dlup.types import PathLike
from dlup.utils.image import check_if_mpp_is_valid


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

    def __init__(self, filename: PathLike) -> None:
        """
        Parameters
        ----------
        filename : PathLike
            Path to image.
        """
        super().__init__(filename)
        # You can have pyvips figure out the reader
        self._images = []
        self._images.append(pyvips.Image.new_from_file(str(filename)))

        self._loader = self._images[0].get("vips-loader")

        self.__slide_bounds = (0, 0), (0, 0)

        if self._loader == "tiffload":
            self._read_as_tiff(filename)
        elif self._loader == "openslideload":
            self._read_as_openslide(filename)
        else:
            raise NotImplementedError(f"Loader {self._loader} is not implemented.")

        self._regions = [pyvips.Region.new(image) for image in self._images]

    def _read_as_tiff(self, path: PathLike) -> None:
        """
        Read all other pages except the first using the tiff backend of pyvips.

        Parameters
        ----------
        path : PathLike
        """
        self._level_count = int(self._images[0].get_value("n-pages"))
        for level in range(1, self._level_count):
            self._images.append(pyvips.Image.tiffload(str(path), page=level))

        # Each tiff page has a resolution
        unit_dict = {"cm": 1000, "centimeter": 1000}
        self._downsamples.append(1.0)
        for idx, image in enumerate(self._images):
            mpp_x = unit_dict.get(image.get("resolution-unit"), 0) / float(image.get("xres"))
            mpp_y = unit_dict.get(image.get("resolution-unit"), 0) / float(image.get("yres"))
            check_if_mpp_is_valid(mpp_x, mpp_y)

            self._spacings.append((mpp_y, mpp_x))
            if idx >= 1:
                downsample = mpp_x / self._spacings[0][0]
                self._downsamples.append(downsample)
            self._shapes.append((image.get("width"), image.get("height")))

        self.__slide_bounds = (0, 0), self._shapes[0]

    def _read_as_openslide(self, path: PathLike):
        """
        Read all other pages except the first using the openslide backend of pyvips.

        Parameters
        ----------
        path : PathLike
        """
        self._level_count = int(self._images[0].get("openslide.level-count"))
        for level in range(1, self._level_count):
            self._images.append(pyvips.Image.openslideload(str(path), level=level))

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
        if "openslide.mpp-x" in available_fields and "openslide.mpp-y" in available_fields:
            mpp_x = float(self._images[0].get("openslide.mpp-x"))
            mpp_y = float(self._images[0].get("openslide.mpp-y"))

        if mpp_x is not None and mpp_y is not None:
            check_if_mpp_is_valid(mpp_x, mpp_y)
            self._spacings = [(np.array([mpp_y, mpp_x]) * downsample).tolist() for downsample in self._downsamples]
        else:
            warnings.warn(
                f"{path} does not have a parseable spacings property. You can overwrite it with `.mpp = (mpp_x, mpp_y)."
            )

        if "openslide.bounds-x" in available_fields and "openslide.bounds-y" in available_fields:
            offset = int(self._images[0].get("openslide.bounds-x")), int(self._images[0].get("openslide.bounds-y"))
        else:
            offset = (0, 0)

        if "openslide.bounds-width" in available_fields and "openslide.bounds-height" in available_fields:
            size = int(self._images[0].get("openslide.bounds-width")), int(
                self._images[0].get("openslide.bounds-height")
            )
        else:
            size = self._shapes[0]

        self.__slide_bounds = offset, size

    @property
    def slide_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide as ((x,y), (width, height)). These can be smaller than the image itself."""
        return self.__slide_bounds

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
        self._spacings = [(np.array([mpp_y, mpp_x]) * downsample).tolist() for downsample in self._downsamples]

    @property
    def properties(self):
        """Metadata about the image as given by pyvips,
        which includes openslide tags in case openslide is the selected reader."""

        keys = self._images[0].get_fields()
        return {key: self._images[0].get_value(key) for key in keys}

    @property
    def magnification(self) -> float | None:
        """Returns the objective power at which the WSI was sampled."""
        if self._loader == "openslideloader":
            return float(self._images[0].properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        else:
            return None

    @property
    def vendor(self) -> str | None:
        """Returns the scanner vendor."""
        if self._loader == "openslideloader":
            return str(self._images[0].properties[openslide.PROPERTY_NAME_VENDOR])
        return None

    @property
    def associated_images(self) -> dict[str, PIL.Image.Image]:
        """Images associated with this whole-slide image."""
        if not self._loader == "openslideload":
            return {}
        # TODO: Fix this
        # associated_images = (_.strip() for _ in self.properties["slide-associated-images"].split(","))
        raise NotImplementedError

    def set_cache(self, cache) -> None:
        raise NotImplementedError

    def read_region(self, coordinates: tuple[Any, ...], level: int, size: tuple[Any, ...]) -> PIL.Image.Image:
        """
        Return the best level for displaying the given image level.

        Parameters
        ----------
        coordinates : tuple
            Coordinates of the region in level 0.
        level : int
            Level of the image pyramid.
        size : tuple
            Size of the region to be extracted.

        Returns
        -------
        PIL.Image
            The requested region.
        """
        image = self._regions[level]
        ratio = self._downsamples[level]
        x, y = coordinates
        height, width = size

        region = np.asarray(image.fetch(int(x // ratio), int(y // ratio), int(height), int(width))).reshape(
            int(width), int(height), -1
        )

        return numpy_to_pil(region)

    def close(self) -> None:
        """Close the underlying slide"""
        del self._regions
        del self._images

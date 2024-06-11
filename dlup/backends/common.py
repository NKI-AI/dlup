# Copyright (c) dlup contributors
from __future__ import annotations

import abc
import io
from types import TracebackType
from typing import Any, Literal, Optional, Type, cast

import numpy as np
import pyvips

from dlup.types import PathLike
from dlup.utils.image import check_if_mpp_is_valid


class AbstractSlideBackend(abc.ABC):
    """
    Abstract base class for slide experimental_backends
    """

    # TODO: Do something with the cache.
    def __init__(self, filename: PathLike):
        """
        Parameters
        ----------
        filename : PathLike
            Path to image.
        """
        self._filename = filename
        self._level_count = 0
        self._downsamples: list[float] = []
        self._spacings: list[tuple[float, float]] = []
        self._shapes: list[tuple[int, int]] = []

    @property
    def mode(self) -> str | None:
        """Returns the mode of the image."""
        return None

    @property
    def level_count(self) -> int:
        """The number of levels in the image."""
        return self._level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        """A list of (width, height) tuples, one for each level of the image.
        This property level_dimensions[n] contains the dimensions of the image at level n.

        Returns
        -------
        list

        """
        return tuple(self._shapes)

    @property
    def dimensions(self) -> tuple[int, int]:
        """A (width, height) tuple for the base level (level 0) of the image.

        Returns
        -------
        Tuple
        """
        return self.level_dimensions[0]

    @property
    def spacing(self) -> tuple[float, float] | None:
        """
        A (mpp_x, mpp_y) tuple for spacing of the base level

        Returns
        -------
        Tuple
        """
        if self._spacings is not None:
            return self._spacings[0]
        return

    @spacing.setter
    def spacing(self, value: tuple[float, float]) -> None:
        if not isinstance(value, tuple) and len(value) != 2:
            raise ValueError("`.spacing` has to be of the form (mpp_x, mpp_y).")

        mpp_x, mpp_y = value
        check_if_mpp_is_valid(mpp_x, mpp_y)
        mpp = np.array([mpp_x, mpp_y])
        self._spacings = [cast(tuple[float, float], tuple(mpp * downsample)) for downsample in self.level_downsamples]

    @property
    def level_spacings(self) -> tuple[tuple[float, float], ...]:
        """
        A list of (mpp_x, mpp_y) tuples, one for each level of the image.
        This property level_spacings[n] contains the spacings of the image at level n.

        Returns
        -------
        Tuple
        """

        return tuple(self._spacings)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        """A tuple of downsampling factors for each level of the image.
        level_downsample[n] contains the downsample factor of level n."""
        return tuple(self._downsamples)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """
        Compute the best level for displaying the given downsample. Returns the closest better resolution.

        Parameters
        ----------
        downsample : float

        Returns
        -------
        int
        """
        level_downsamples = np.array(self.level_downsamples)
        level = 0 if downsample < 1 else int(np.where(level_downsamples <= downsample)[0][-1])
        return level

    def get_thumbnail(self, size: int | tuple[int, int]) -> pyvips.Image:
        """
        Return a PIL.Image as an RGB image with the thumbnail with maximum size given by size.
        Aspect ratio is preserved, so the given size is the maximum size of the thumbnail respecting that aspect ratio.

        Parameters
        ----------
        size : int or tuple[int, int]
            Output size of the thumbnail, will take the maximal value for the output and preserve aspect ratio.

        Returns
        -------
        pyvips.Image
            The thumbnail.
        """
        if isinstance(size, int):
            size = (size, size)

        downsample = max(*(dim / thumb for dim, thumb in zip(self.dimensions, size)))
        level = self.get_best_level_for_downsample(downsample)

        thumbnail = self.read_region((0, 0), level, self.level_dimensions[level])

        scale_factor = min(size[0] / thumbnail.width, size[1] / thumbnail.height)
        thumbnail = thumbnail.resize(scale_factor, kernel="lanczos3")
        if thumbnail.hasalpha():
            thumbnail = thumbnail.flatten(background=(255, 255, 255))

        return thumbnail

    @property
    def slide_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide. These can be smaller than the image itself."""
        return (0, 0), self.dimensions

    @property
    def color_profile(self) -> io.BytesIO | None:
        raise NotImplementedError("Color profiles are currently not implemented in this backend.")

    @property
    @abc.abstractmethod
    def properties(self) -> dict[str, Any]:
        """Properties of slide"""

    @abc.abstractmethod
    def read_region(self, coordinates: tuple[int, int], level: int, size: tuple[int, int]) -> pyvips.Image:
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
        pyvips.Image
            The requested region as a pyvips image.
        """

    @property
    @abc.abstractmethod
    def magnification(self) -> float | None:
        """Returns the objective power at which the WSI was sampled."""

    @property
    @abc.abstractmethod
    def vendor(self) -> str | None:
        """Returns the scanner vendor."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the underlying slide"""

    def __enter__(self) -> AbstractSlideBackend:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        self.close()
        return False

    def __close__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(filename={self._filename}, "
            f"dimensions={self.dimensions}, "
            f"spacing={self.spacing}, "
            f"magnification={self.magnification}, "
            f"vendor={self.vendor}, "
            f"level_count={self.level_count}, "
            f"mode={self.mode})>"
        )

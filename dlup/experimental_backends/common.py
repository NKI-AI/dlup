# coding=utf-8
# Copyright (c) dlup contributors
import abc
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import PIL.Image

from dlup.types import PathLike


def numpy_to_pil(tile: np.ndarray) -> PIL.Image:
    """
    Convert a numpy tile to a PIL image, assuming the last axis is the channels

    Parameters
    ----------
    tile : np.ndarray

    Returns
    -------
    PIL.Image
    """
    bands = tile.shape[-1]

    if bands == 1:
        mode = "L"
        tile = tile[:, :, 0]
    elif bands == 3:
        mode = "RGB"
    elif bands == 4:
        mode = "RGBA"
    else:
        raise RuntimeError(f"Incorrect number of channels.")

    return PIL.Image.fromarray(tile, mode=mode)


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
        self._downsamples: List[float] = []
        self._spacings: List[Tuple[Any, ...]] = []  # This is to make mypy shut up
        self._shapes: List[Tuple[int, int]] = []

    @property
    def level_count(self) -> int:
        """The number of levels in the image."""
        return self._level_count

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """A list of (width, height) tuples, one for each level of the image.
        This property level_dimensions[n] contains the dimensions of the image at level n.

        Returns
        -------
        List

        """
        return self._shapes

    @property
    def dimensions(self) -> Tuple[int, int]:
        """A (width, height) tuple for the base level (level 0) of the image.

        Returns
        -------
        Tuple
        """
        return self.level_dimensions[0]

    @property
    def spacing(self) -> Tuple[Any, ...]:
        """
        A (mpp_x, mpp_y) tuple for spacing of the base level

        Returns
        -------
        Tuple
        """
        return self._spacings[0]

    @property
    def level_spacings(self) -> Tuple[Tuple[Any, ...], ...]:
        """
        A list of (mpp_x, mpp_y) tuples, one for each level of the image.
        This property level_spacings[n] contains the spacings of the image at level n.

        Returns
        -------
        Tuple
        """

        return tuple(self._spacings)

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
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
        sorted_downsamples = sorted(self._downsamples, reverse=True)

        def difference(sorted_list):
            return np.clip(0, None, downsample - sorted_list)

        number = max(sorted_downsamples, key=difference)
        return self._downsamples.index(number)

    def get_thumbnail(self, size: Union[int, Tuple[int, int]]) -> PIL.Image:
        """
        Return a PIL.Image as an RGB image with the thumbnail with maximum size given by size.
        Aspect ratio is preserved.

        Parameters
        ----------
        size : int or Tuple[int, int]
            Output size of the thumbnail, will take the maximal value for the output and preserve aspect ratio.

        Returns
        -------
        PIL.Image
            The thumbnail.
        """
        if isinstance(size, int):
            size = (size, size)

        downsample = max(*(dim / thumb for dim, thumb in zip(self.dimensions, size)))
        level = self.get_best_level_for_downsample(downsample)
        thumbnail = (
            self.read_region((0, 0), level, self.level_dimensions[level])
            .convert("RGB")
            .resize(
                np.floor(np.asarray(self.dimensions) / downsample).astype(int).tolist(),
                resample=PIL.Image.Resampling.LANCZOS,
            )
        )
        return thumbnail

    @property
    @abc.abstractmethod
    def properties(self):
        """Properties of slide"""

    @abc.abstractmethod
    def read_region(self, coordinates: Tuple[Any, ...], level: int, size: Tuple[Any, ...]) -> PIL.Image:
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

    @property
    @abc.abstractmethod
    def magnification(self) -> Optional[float]:
        """Returns the objective power at which the WSI was sampled."""

    @property
    @abc.abstractmethod
    def vendor(self) -> Optional[str]:
        """Returns the scanner vendor."""

    @abc.abstractmethod
    def close(self):
        """Close the underlying slide"""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self._filename})>"

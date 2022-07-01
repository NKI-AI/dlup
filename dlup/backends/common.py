# coding=utf-8
# Copyright (c) dlup contributors
import abc
from typing import Any, List, Tuple, Union

import numpy as np
import PIL.Image

from dlup._exceptions import UnsupportedSlideError
from dlup.types import PathLike


def numpy_to_pil(tile: np.ndarray) -> PIL.Image:
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


def check_mpp(mpp_x, mpp_y):
    if not np.isclose(mpp_x, mpp_y, rtol=1.0e-2):
        raise UnsupportedSlideError(f"cannot deal with slides having anisotropic mpps. Got {mpp_x} and {mpp_y}.")


class AbstractSlideBackend(abc.ABC):
    # TODO: Do something with the cache.
    def __init__(self, filename: PathLike):
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
        level_dimensions[n] contains the dimensions of level n."""
        return self._shapes

    @property
    def dimensions(self) -> Tuple[int, int]:
        """A (width, height) tuple for level 0 of the image."""
        return self.level_dimensions[0]

    @property
    def spacing(self) -> Tuple[Any, ...]:
        return self._spacings[0]

    @property
    def level_spacings(self) -> Tuple[Tuple[Any, ...], ...]:
        return tuple(self._spacings)

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        """A tuple of downsampling factors for each level of the image.
        level_downsample[n] contains the downsample factor of level n."""
        return tuple(self._downsamples)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the best level for displaying the given downsample."""
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
                np.floor(np.asarray(self.dimensions) / downsample).astype(int).tolist(), resample=PIL.Image.LANCZOS
            )
        )
        return thumbnail

    @property
    @abc.abstractmethod
    def properties(self):
        """Properties of slide"""

    @abc.abstractmethod
    def read_region(self, coordinates: Tuple[Any, ...], level: int, size: Tuple[Any, ...]) -> PIL.Image:
        """Read region of multiresolution image"""

    @abc.abstractmethod
    def close(self):
        """Close the underlying slide"""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self._filename})>"

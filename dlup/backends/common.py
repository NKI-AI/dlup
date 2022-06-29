# coding=utf-8
# Copyright (c) dlup contributors
import abc

import numpy as np
import PIL.Image


def check_mpp(mpp_x, mpp_y):
    if not np.allclose(mpp_x, mpp_y):
        raise RuntimeError(f"mpp_x and mpp_y are assumed to be close. Got {mpp_x} and {mpp_y}")


class AbstractSlideBackend(abc.ABC):
    def __init__(self, path):
        self._level_count = 0
        self._downsamples = []
        self._spacings = []
        self._shapes = []

    @property
    def level_count(self):
        """The number of levels in the image."""
        return self._level_count

    @property
    def level_dimensions(self):
        """A list of (width, height) tuples, one for each level of the image.
        level_dimensions[n] contains the dimensions of level n."""
        return self._shapes

    @property
    def dimensions(self):
        """A (width, height) tuple for level 0 of the image."""
        return self.level_dimensions[0]

    @property
    def level_downsamples(self):
        """A list of downsampling factors for each level of the image.
        level_downsample[n] contains the downsample factor of level n."""
        return tuple(self._downsamples)

    def get_best_level_for_downsample(self, downsample) -> int:
        """Return the best level for displaying the given downsample."""
        sorted_downsamples = sorted(self._downsamples, reverse=True)
        difference = lambda sorted_list: np.clip(0, None, downsample - sorted_list)
        number = max(sorted_downsamples, key=difference)
        return self._downsamples.index(number)

    def get_thumbnail(self, size):
        """Return a PIL.Image containing an RGB thumbnail of the image.
        size:     the maximum size of the thumbnail."""
        downsample = max(*(dim / thumb for dim, thumb in zip(self.dimensions, size)))
        level = self.get_best_level_for_downsample(downsample)
        tile = self.read_region((0, 0), level, self.level_dimensions[level])
        # Apply on solid background
        # bg_color = "#" + self.properties.get(PROPERTY_NAME_BACKGROUND_COLOR, "ffffff")
        bg_color = "#ffffff"
        thumb = PIL.Image.new("RGB", tile.size, bg_color)
        thumb.paste(tile, None, tile)
        # Image.Resampling added in Pillow 9.1.0
        # Image.LANCZOS removed in Pillow 10
        thumb.thumbnail(size, getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS)
        return thumb

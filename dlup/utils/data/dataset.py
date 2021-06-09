# coding=utf-8
# Copyright (c) DLUP Contributors

"""Datasets helpers to simplify the generation of a dataset made of tiles from a WSI."""

import pathlib
from typing import Tuple

from dlup.tiling import TilingMode
from dlup import SlideImageTiledRegionView
from dlup import SlideImage

from torch.utils.data import Dataset


class SlideImageDataset(Dataset, SlideImageTiledRegionView):
    """Basic Slide Image dataset."""

    def __init__(self, path: pathlib.Path, mpp: float, tile_size: Tuple[int, int],
                 tile_overlap: Tuple[int, int], background_threshold: float = 0.0):
        self._path = path
        self._slide_image = SlideImage.from_file_path(self.path)
        scaled_view = self._slide_image.get_scaled_view(self._slide_image.mpp / mpp)
        super().__init__(scaled_view, tile_size, tile_overlap, TilingMode.overflow, crop=True)

    @property
    def path(self):
        return self._path

    @property
    def slide_image(self):
        return self._slide_image

    def __getitem__(self, i):
        return SlideImageTiledRegionView.__getitem__(self, i)

    def __len__(self):
        return SlideImageTiledRegionView.__len__(self)

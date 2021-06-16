# coding=utf-8
# Copyright (c) dlup contributors

"""Datasets helpers to simplify the generation of a dataset made of tiles from a WSI."""

import pathlib
from typing import Tuple

from torch.utils.data import Dataset

from dlup import SlideImage, SlideImageTiledRegionView
from dlup.tiling import TilingMode


class SlideImageDataset(Dataset, SlideImageTiledRegionView):
    """Basic Slide Image dataset."""

    def __init__(
        self,
        path: pathlib.Path,
        mpp: float,
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        tile_mode: TilingMode = TilingMode.skip,
    ):
        self._path = path
        self._slide_image = SlideImage.from_file_path(path)
        scaled_view = self._slide_image.get_scaled_view(self._slide_image.mpp / mpp)
        super().__init__(scaled_view, tile_size, tile_overlap, tile_mode, crop=False)

    @property
    def path(self):
        """Path of whole slide image"""
        return self._path

    @property
    def slide_image(self):
        """SlideImage instance"""
        return self._slide_image

    def __getitem__(self, i):
        return SlideImageTiledRegionView.__getitem__(self, i)

    def __len__(self):
        return SlideImageTiledRegionView.__len__(self)

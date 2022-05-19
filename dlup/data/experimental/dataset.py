# coding=utf-8
# Copyright (c) dlup contributors
"""Experimental dataset functions, might e.g. lack tests, or requires input from users"""

import pathlib
from typing import Iterable, Tuple, Optional, Callable, List

import numpy as np

from dlup import SlideImage
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import Grid, TilingMode


class MultiScaleTiledROIsSlideImageDataset(TiledROIsSlideImageDataset):
    """Dataset class that supports multiscale output, and can have multiple ROIs.

    This dataset can be used, for example, to tile your WSI on-the-fly using the `multiscale_from_tiling` function.
    The output of the dataset will be provided as a list of dictionaries as outputs of `TiledROIsSlideImageDataset`

    Example
    -------
    >>>  dlup_dataset = MultiScaleTiledROIsSlideImageDataset.multiscale_from_tiling(\
            path="/path/to/TCGA-WSI.svs",\
            mpps=[0.0625, 0.125],\
            tile_size=(1024, 1024),\
            tile_overlap=(512, 512),\
            tile_mode=TilingMode.skip,\
            crop=True,\
            mask=None,\
            mask_threshold=0.5,\
            transform=YourTransform()\
         )
    >>> sample = dlup_dataset[5]
    >>> images = (sample[0]["image"], sample[1]["image"])
    """
    def __init__(
        self,
        path: pathlib.Path,
        grids: Iterable[Tuple[Grid, Tuple[int, int], float]],
        num_scales: int,
        crop: bool = True,
        mask: Optional[np.ndarray] = None,
        mask_threshold: float = 0.1,
        transform: Optional[Callable] = None,
    ):
        self._grids = grids
        self._num_scales = num_scales
        if len(list(grids)) % num_scales != 0:
            raise ValueError(f"In a multiscale dataset the grids needs to be divisible by the number of scales.")

        self._step_size = len(grids[0][0])
        self._index_ranges = [
            range(idx * self._step_size, (idx + 1) * self._step_size) for idx in range(0, num_scales)
        ]
        super().__init__(path, grids, crop, mask=mask, mask_threshold=mask_threshold, transform=None)
        self.__transform = transform

    def __len__(self):
        return self._step_size

    @classmethod
    def multiscale_from_tiling(
        cls,
        path: pathlib.Path,
        mpps: List[float],
        tile_size: Tuple[int, int],
        tile_overlap: Tuple[int, int],
        tile_mode: TilingMode = TilingMode.skip,
        crop: bool = True,
        mask: Optional[np.ndarray] = None,
        rois: Optional = None,
        mask_threshold: float = 0.1,
        transform: Optional[Callable] = None,
    ):

        if mpps != sorted(mpps):
            raise ValueError(f"The mpp values should be in increasing order.")

        with SlideImage.from_file_path(path) as slide_image:
            original_mpp = slide_image.mpp
            original_size = slide_image.size
            if rois is None:
                rois = [[0, 0, *original_size]]
            else:
                # Do some checks whether the ROIs are within the image
                origin_positive = [np.all(np.asarray(_[:2]) > 0) for _ in rois]
                image_within_borders = [np.all((np.asarray(_[:2]) + _[2:]) <= original_size) for _ in rois]
                if not origin_positive or not image_within_borders:
                    raise ValueError(f"ROIs should be within image boundaries. Got {rois}.")

        view_scalings = [mpp / original_mpp for mpp in mpps]
        grids = []
        for scaling in view_scalings:
            for roi in rois:
                offset = roi[:2]
                size = roi[2:]

                # We CEIL the offset and FLOOR the size, so that we are always in a fully annotated area.
                offset = np.ceil(offset).astype(int)
                size = np.floor(size).astype(int)

                curr_tile_overlap = (np.asarray(tile_size)) * (scaling - 1) / scaling
                curr_offset = (offset - np.asarray(tile_size) * (scaling - 1) / 2) / scaling
                curr_grid = Grid.from_tiling(
                    curr_offset,
                    size=size / scaling + curr_tile_overlap,
                    tile_size=tile_size,
                    tile_overlap=curr_tile_overlap + np.asarray(tile_overlap) / scaling,
                    mode=tile_mode,
                )

                grids.append((curr_grid, tile_size, original_mpp * scaling))

        return cls(
            path,
            grids,
            num_scales=len(view_scalings),
            crop=crop,
            mask=mask,
            mask_threshold=mask_threshold,
            transform=transform,
        )

    def __getitem__(self, index):
        indices = [_[index] for _ in self._index_ranges]
        sample = [TiledROIsSlideImageDataset.__getitem__(self, _) for _ in indices]
        if self.__transform:
            sample = self.__transform(sample)

        return sample

# coding=utf-8
# Copyright (c) dlup contributors
"""Experimental dataset functions, might e.g. lack tests, or requires input from users"""

import pathlib
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledROIsSlideImageDataset, parse_rois
from dlup.experimental_backends import ImageBackend
from dlup.tiling import Grid, GridOrder, TilingMode

_BaseAnnotationTypes = Union[SlideImage, WsiAnnotations]
_AnnotationTypes = Union[List[Tuple[str, _BaseAnnotationTypes]], _BaseAnnotationTypes]
_LabelTypes = Union[str, bool, int, float]


class MultiScaleSlideImageDataset(TiledROIsSlideImageDataset):
    """Dataset class that supports multiscale output, and can have multiple ROIs.

    This dataset can be used, for example, to tile your WSI on-the-fly using the `multiscale_from_tiling` function.
    The output of the dataset will be provided as a list of dictionaries as outputs of `TiledROIsSlideImageDataset`

    Examples
    --------
    >>>  dlup_dataset = MultiScaleSlideImageDataset.multiscale_from_tiling(\
            path="/path/to/TCGA-WSI.svs",\
            mpps=[0.0625, 0.125],\
            tile_size=(1024, 1024),\
            tile_overlap=(512, 512),\
            tile_mode=TilingMode.skip,\
            crop=False,\
            mask=None,\
            mask_threshold=0.5,\
            annotations=None,\
            labels=[("msi", True),],\
            transform=YourTransform()\
         )
    >>> sample = dlup_dataset[5]
    >>> images = (sample[0]["image"], sample[1]["image"])

    Setting `crop` to False will pad the image with zeros in the lower resolutions at the borders.
    """

    def __init__(
        self,
        path: pathlib.Path,
        grids: List[Tuple[Grid, Tuple[int, int], float]],
        num_scales: int,
        crop: bool = True,
        mask: Optional[Union[SlideImage, np.ndarray, WsiAnnotations]] = None,
        mask_threshold: float = 0.1,
        annotations: Optional[_AnnotationTypes] = None,
        labels: Optional[List[Tuple[str, _LabelTypes]]] = None,
        transform: Optional[Callable] = None,
        backend: Callable = ImageBackend.PYVIPS,
    ):
        self._grids = grids
        self._num_scales = num_scales
        if len(list(grids)) % num_scales != 0:
            raise ValueError(f"In a multiscale dataset the grids needs to be divisible by the number of scales.")

        self._step_size = len(list(grids)[0][0])
        self._index_ranges = [
            range(idx * self._step_size, (idx + 1) * self._step_size) for idx in range(0, num_scales)
        ]
        super().__init__(
            path,
            grids,
            crop,
            mask=mask,
            mask_threshold=mask_threshold,
            annotations=annotations,
            labels=labels,
            transform=None,
            backend=backend,
        )
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
        tile_mode: TilingMode = TilingMode.overflow,
        grid_order: GridOrder = GridOrder.C,
        crop: bool = False,
        mask: Optional[np.ndarray] = None,
        mask_threshold: float = 0.1,
        rois: Optional[Tuple[Tuple[int, ...]]] = None,
        transform: Optional[Callable] = None,
        backend: Callable = ImageBackend.PYVIPS,
    ):

        if mpps != sorted(mpps):
            raise ValueError(f"The mpp values should be in increasing order.")

        with SlideImage.from_file_path(path, backend=backend) as slide_image:
            original_mpp = slide_image.mpp
            _rois = parse_rois(rois, tuple(slide_image.size), scaling=1)

        view_scalings = [mpp / original_mpp for mpp in mpps]
        grids = []
        for scaling in view_scalings:
            for offset, size in _rois:
                # We CEIL the offset and FLOOR the size, so that we are always in a fully annotated area.
                _offset = np.ceil(offset).astype(int)
                _size = np.floor(size).astype(int)

                _tile_size = np.asarray(tile_size)
                _tile_overlap = np.asarray(tile_overlap)
                curr_tile_overlap = _tile_size * (scaling - 1) / scaling
                curr_offset = (_offset - _tile_size * (scaling - 1) / 2) / scaling
                curr_grid: Grid = Grid.from_tiling(
                    curr_offset,
                    size=_size / scaling + curr_tile_overlap,
                    tile_size=_tile_size,
                    tile_overlap=curr_tile_overlap + _tile_overlap / scaling,
                    mode=tile_mode,
                    order=grid_order,
                )

                grids.append((curr_grid, tile_size, float(original_mpp * scaling)))

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

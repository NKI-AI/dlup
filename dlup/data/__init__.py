# coding=utf-8
# Copyright (c) dlup contributors
"""Data module for dlup."""
from __future__ import annotations

import pathlib
from typing import Any

import PIL
from typing_extensions import NotRequired, TypedDict

from dlup.types import Coordinates


class _StandardTilingFromSlideDatasetSample(TypedDict):
    image: PIL.Image.Image
    coordinates: tuple[int | float, int | float]
    mpp: float
    path: pathlib.Path
    region_index: int
    # FIXME: better typing
    annotations: NotRequired[Any]
    labels: NotRequired[Any]


class RegionFromSlideDatasetSample(_StandardTilingFromSlideDatasetSample):
    """A sample from a :class:`RegionFromSlideDataset`."""

    grid_local_coordinates: NotRequired[Coordinates]
    grid_index: NotRequired[int]
    annotation_data: NotRequired[Any]

# coding=utf-8
# Copyright (c) DLUP Contributors

import pathlib
from typing import List, Union

from paquo.java import BufferedImage, File, PathIO
from paquo.pathobjects import _PathROIObject
from shapely.geometry import Polygon


def read_qpdata(filename: Union[str, pathlib.Path]) -> List[Polygon]:
    """Extract List of Shapely polygon(s) from QuPath .qpdata file

    Based on https://github.com/bayer-science-for-a-better-life/paquo/blob/f863af53ff9ce2239763029c889a8c73f07a5460/paquo/images.py#L395

    Parameters
    ----------
    filename : PathLike
        Filename to .qpdata file

    Returns
    -------
    List
        List of Shapely Polygon(s)
    """

    if not str(filename).endswith(".qpdata"):
        raise ValueError(f"Expected filename pointing to .qpdata file. Got {filename}.")

    total_annotations = []

    image_data = PathIO.readImageData(File(str(filename)), None, None, BufferedImage)

    for item in image_data.getHierarchy().getAnnotationObjects():
        total_annotations.append(_PathROIObject(item).roi)

    return total_annotations

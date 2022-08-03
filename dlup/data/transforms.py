# coding=utf-8
# Copyright (c) dlup contributors
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np

import dlup.experimental_annotations

_AnnotationsTypes = Union[dlup.experimental_annotations.Point, dlup.experimental_annotations.Polygon]


def convert_annotations(
    annotations: Iterable[_AnnotationsTypes],
    region_size: Tuple[int, int],
    index_map: Dict[str, int],
    roi_name: Optional[str] = None,
) -> (Dict, np.ndarray, np.ndarray | None):
    """
    Convert the polygon and point annotations as output of a dlup dataset class, where:
    - In case of points the output is dictionary mapping the annotation name to a list of locations.
    - In case of polygons these are converted into a mask according to segmentation_index_map.

    *BE AWARE*: the polygon annotations are processed sequentially and later annotations can overwrite earlier ones.
    This is for instance useful when you would annotate "tumor associated stroma" on top of "stroma".
    The dlup Annotation classes return the polygons with area from large to small.


    TODO
    ----
    - Convert segmentation index map to an Enum
    - Replace opencv with pyvips (convert shapely to svg) or anything else available - and perhaps a cython function.
    - Do we need to return PIL images here? If we load a tif mask the mask will be returned as a PIL image, so
      for consistency it might be relevant to do the same here.

    Parameters
    ----------
    annotations
    region_size : Tuple[int, int]
    index_map : Dict[str, int]
    roi_name : Name of the region-of-interest key.

    Returns
    -------
    dict, np.ndarray, np.ndarray or None
        Dictionary of points, mask and roi_mask.

    """
    mask = np.zeros(region_size, dtype=np.int32)
    points = defaultdict(list)

    roi_mask = np.empty(region_size, dtype=np.int32)

    for curr_annotation in annotations:
        if isinstance(curr_annotation, dlup.experimental_annotations.Point):
            points[curr_annotation.label] += tuple(curr_annotation.coords)

        if roi_name and curr_annotation.label == roi_name:
            cv2.fillPoly(
                roi_mask,
                [np.asarray(curr_annotation.exterior.coords).round().astype(np.int32)],
                1,
            )
            continue

        if not (curr_annotation.label in index_map):
            continue

        cv2.fillPoly(
            mask,
            [np.asarray(curr_annotation.exterior.coords).round().astype(np.int32)],
            index_map[curr_annotation.label],
        )

    return dict(points), mask, roi_mask if roi_name else None


class ConvertAnnotationsToMask:
    """Transform which converts polygons to masks. Will overwrite the annotations key"""

    def __init__(self, roi_name: str, index_map: Dict[str, int]):
        """
        Parameters
        ----------
        roi_name : str
            Name of the ROI key.
        index_map : dict
            Dictionary mapping the label to the integer in the output.
        """
        self._roi_name = roi_name
        self._index_map = index_map

    def __call__(self, sample):
        if "annotations" not in sample:
            return sample

        annotations = sample["annotations"]
        points, mask, roi = convert_annotations(annotations, sample["image"].size, roi_name="roi", index_map=self._index_map)
        sample["annotations"] = {
            "points": points,
            "mask": mask,
        }
        if roi is not None:
            sample["annotations"]["roi"] = roi

        return sample

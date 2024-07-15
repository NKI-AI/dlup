# Copyright (c) dlup contributors
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, cast

import cv2
import numpy as np
import numpy.typing as npt

import dlup.annotations
from dlup._exceptions import AnnotationError
from dlup.annotations import AnnotationClass, AnnotationType
from dlup.data.dataset import PointType, TileSample, TileSampleWithAnnotationData

_AnnotationsTypes = dlup.annotations.Point | dlup.annotations.Polygon


def convert_annotations(
    annotations: Iterable[_AnnotationsTypes],
    region_size: tuple[int, int],
    index_map: dict[str, int],
    roi_name: str | None = None,
    default_value: int = 0,
    multiclass: bool = False,
) -> tuple[dict[str, list[PointType]], npt.NDArray[np.int_], npt.NDArray[np.int_] | None]:
    """
    Convert the polygon and point annotations as output of a dlup dataset class, where:
    - In case of points the output is dictionary mapping the annotation name to a list of locations.
    - In case of bounding boxes the output is a dictionary mapping the annotation name to a list of bounding boxes.
      Note that the internal representation of a bounding box is a polygon (`AnnotationType is AnnotationType.BOX`),
      so the bounding box of that polygon is computed to convert.
    - In case of polygons these are converted into a mask according to `index_map`, or the z_index + 1.

    *BE AWARE*: the polygon annotations are processed sequentially and later annotations can overwrite earlier ones.
    This is for instance useful when you would annotate "tumor associated stroma" on top of "stroma".
    The dlup Annotation classes return the polygons with area from large to small.

    When the polygon has holes, the previous written annotation is used to fill the holes.

    *BE AWARE*: This function will silently ignore annotations which are written out of bounds.

    TODO
    ----
    - Convert segmentation index map to an Enum

    Parameters
    ----------
    annotations : Iterable[_AnnotationsTypes]
        The annotations as a list, e.g., as output from `dlup.annotations.WsiAnnotations.read_region()`.
    region_size : tuple[int, int]
    index_map : dict[str, int]
        Map mapping annotation name to index number in the output.
    roi_name : str, optional
        Name of the region-of-interest key.
    default_value : int
        The mask will be initialized with this value.
    multiclass : bool
        Output a multiclass mask, the first axis will be the index. This requires that an index_map is available.

    Returns
    -------
    dict, dict, np.ndarray, np.ndarray or None
        Dictionary of points, dictionary of boxes, mask and roi_mask.

    """
    if multiclass and index_map is None:
        raise ValueError("index_map must be provided when multiclass is True.")

    # Initialize the base mask, which may be multi-channel if output_class_axis is True
    if multiclass:
        assert isinstance(index_map, dict)
        num_classes = max(index_map.values()) + 1  # Assuming index_map starts from 0
        mask = np.zeros((num_classes, *region_size), dtype=np.int32)
    else:
        mask = np.empty(region_size, dtype=np.int32)
    mask[:] = default_value

    points: dict[str, list[PointType]] = defaultdict(list)
    roi_mask = np.zeros(region_size, dtype=np.int32)

    has_roi = False
    for curr_annotation in annotations:
        holes_mask = None
        if isinstance(curr_annotation, dlup.annotations.Point):
            coords = tuple(curr_annotation.coords)
            points[curr_annotation.label] += tuple(coords)
            continue

        index_value: int
        if curr_annotation.label != roi_name:
            if curr_annotation.label not in index_map:
                raise ValueError(f"Label {curr_annotation.label} is not in the index map {index_map}")
        elif roi_name is not None:
            cv2.fillPoly(
                roi_mask,
                [np.asarray(curr_annotation.exterior.coords).round().astype(np.int32)],
                [1],
            )
            has_roi = True
            continue

        index_value = index_map[curr_annotation.label]

        original_values = None
        interiors = [np.asarray(pi.coords).round().astype(np.int32) for pi in curr_annotation.interiors]
        if interiors is not []:
            original_values = mask.copy()
            holes_mask = np.zeros(region_size, dtype=np.int32)
            # Get a mask where the holes are
            cv2.fillPoly(holes_mask, interiors, [1])

        if not multiclass:
            cv2.fillPoly(
                mask,
                [np.asarray(curr_annotation.exterior.coords).round().astype(np.int32)],
                [index_value],
            )
        else:
            cv2.fillPoly(
                mask[index_value],
                [np.asarray(curr_annotation.exterior.coords).round().astype(np.int32)],
                [1],
            )
        if interiors != []:
            # TODO: This is a bit hacky to ignore mypy here, but I don't know how to fix it.
            mask = np.where(holes_mask == 1, original_values, mask)  # type: ignore

    # This is a hard to find bug, so better give an explicit error.
    if not has_roi and roi_name is not None:
        raise AnnotationError(f"ROI mask {roi_name} not found, please add a ROI mask to the annotations.")

    return dict(points), mask, roi_mask if roi_name else None


class ConvertAnnotationsToMask:
    """Transform which converts polygons to masks. Will overwrite the annotations key"""

    def __init__(
        self, *, roi_name: str | None, index_map: dict[str, int], multiclass: bool = False, default_value: int = 0
    ):
        """
        Converts annotations given my `dlup.annotations.Polygon` or `dlup.annotations.Point` to a mask and a dictionary
        of points. The mask is initialized with `default_value`, (i.e., background). The values in the mask are
        subsequently determined by `index_map`, where each value is written to the mask according to this map, in the
        order of the elements in the annotations. This means that if you have overlapping polygons, the last polygon
        will overwrite the previous one. The sorting can be handled in the `dlup.annotations.WsiAnnotation` class.

        In case there are no annotations present (i.e. the "annotations" key is None) a `ValueError` is
        raised.

        Parameters
        ----------
        roi_name : str, optional
            Name of the ROI key.
        index_map : dict
            Dictionary mapping the label to the integer in the output.
        multiclass: bool
            Output a multiclass mask, the first axis will be the index. This requires that an index_map is available.
        default_value : int
            The mask will be initialized with this value.
        """
        self._roi_name = roi_name
        self._index_map = index_map
        self._default_value = default_value
        self._multiclass = multiclass

    def __call__(self, sample: TileSample) -> TileSampleWithAnnotationData:
        """
        Convert the annotations to a mask.

        Parameters
        ----------
        sample : TileSample
            The input sample.

        Raises
        ------
        ValueError
            If no annotations are found.

        Returns
        -------
        TileSampleWithAnnotationData
            The input sample with the annotation data added.

        """

        _annotations = sample["annotations"]
        if _annotations is None:
            raise ValueError("No annotations found to convert to mask.")

        image = sample["image"]
        points, mask, roi = convert_annotations(
            _annotations,
            (image.height, image.width),
            roi_name=self._roi_name,
            index_map=self._index_map,
            default_value=self._default_value,
        )

        output: TileSampleWithAnnotationData = cast(TileSampleWithAnnotationData, sample)
        output["annotation_data"] = {
            "points": points,
            "mask": mask,
            "roi": roi,
        }

        return output


def rename_labels(annotations: Iterable[_AnnotationsTypes], remap_labels: dict[str, str]) -> list[_AnnotationsTypes]:
    """
    Rename the labels in the annotations.

    Parameters
    ----------
    annotations: Iterable[_AnnotationsTypes]
        The annotations
    remap_labels: dict[str, str]
        The renaming table

    Returns
    -------
    list[_AnnotationsTypes]
    """
    output_annotations = []
    for annotation in annotations:
        label = annotation.label
        if label not in remap_labels:
            output_annotations.append(annotation)
            continue

        if annotation.a_cls.annotation_type == AnnotationType.BOX:
            a_cls = AnnotationClass(label=remap_labels[label], annotation_type=AnnotationType.BOX)
            output_annotations.append(dlup.annotations.Polygon(annotation, a_cls=a_cls))
        elif annotation.a_cls.annotation_type == AnnotationType.POLYGON:
            a_cls = AnnotationClass(label=remap_labels[label], annotation_type=AnnotationType.POLYGON)
            output_annotations.append(dlup.annotations.Polygon(annotation, a_cls=a_cls))
        elif annotation.a_cls.annotation_type == AnnotationType.POINT:
            a_cls = AnnotationClass(label=remap_labels[label], annotation_type=AnnotationType.POINT)
            output_annotations.append(dlup.annotations.Point(annotation, a_cls=a_cls))
        else:
            raise AnnotationError(f"Unsupported annotation type {annotation.a_cls.a_cls}")

    return output_annotations


class RenameLabels:
    """Remap the label names"""

    def __init__(self, remap_labels: dict[str, str]):
        """

        Parameters
        ----------
        remap_labels : dict
            Dictionary mapping old name to new name.
        """
        self._remap_labels = remap_labels

    def __call__(self, sample: TileSample) -> TileSample:
        _annotations = sample["annotations"]
        if _annotations is None:
            raise ValueError("No annotations found to rename.")

        sample["annotations"] = rename_labels(_annotations, self._remap_labels)
        return sample

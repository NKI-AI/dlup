# coding=utf-8
# Copyright (c) dlup contributors
"""
Annotation module for dlup.

There are three types of annotations:
- points
- boxes (which are internally polygons)
- polygons

When working with annotations it is assumed that for each label, the type is always the same. So a label e.g., "lymphocyte"
would always be a box and not suddenly a polygon. In the latter case you better have labels such as `lymphocyte_point`,
`lymphocyte_box` or so.

Assumed:
- The type of object (point, box, polygon) is fixed per label.
- The mpp is fixed per label.

shapely json is assumed to contain all the objects belonging to one label

{
    "label": <label_name>,
    "type": points, polygon, box
    "data": [list of objects]
}

"""
import errno
import json
import os
import pathlib
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import shapely
from shapely import geometry
from shapely.geometry import shape
from shapely.strtree import STRtree

# from dlup.utils.types import GenericNumber, PathLike

_TAnnotationParser = TypeVar("_TAnnotationParser", bound="AnnotationParser")
ShapelyTypes = Union[shapely.geometry.Point, shapely.geometry.MultiPolygon, shapely.geometry.Polygon]


class AnnotationType(Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"


_POSTPROCESSORS = {
    AnnotationType.POINT: lambda x, region: x,
    AnnotationType.BOX: lambda x, region: x.intersection(region),
    AnnotationType.POLYGON: lambda x, region: x.intersection(region),
}

_ASAP_TYPES = {
    "polygon": AnnotationType.POLYGON,
    "rectangle": AnnotationType.BOX,
    "dot": AnnotationType.POINT,
    "spline": AnnotationType.POLYGON,
    "pointset": AnnotationType.POINT,
}


class WholeSlideAnnotation:
    """Class to hold the annotations of one specific label for a label"""

    def __init__(self, data: List[ShapelyTypes], metadata: Dict[str, Union[AnnotationType, float, str]]):
        self.type = metadata["type"]
        self._annotations = data
        self.mpp = metadata["mpp"]
        self.label = metadata["label"]

    def as_strtree(self) -> STRtree:
        return STRtree(self._annotations)

    def __str__(self) -> str:
        return f"WholeSlideAnnotation(label={self.label})"


class SlideAnnotations:
    def __init__(self, annotations: List[WholeSlideAnnotation], labels: List[str] = None):
        self._annotations = annotations
        self._label_dict = {
            annotation.label: (idx, annotation.mpp, annotation.type) for idx, annotation in enumerate(annotations)
        }
        self._annotation_trees = {label: self[label].as_strtree() for label in self.available_labels}

    @classmethod
    def from_geojson(
        cls: Type[_TAnnotationParser],
        geo_jsons,  # : Iterable[PathLike],
        labels: List[str],
        mpp: float = None,
        label_map=None,  #: Optional[Dict[float, PathLike]] = None,
    ) -> _TAnnotationParser:
        annotations = []
        if isinstance(mpp, float):
            mpp = [mpp] * len(labels)

        for idx, (curr_mpp, label, path) in enumerate(zip(mpp, labels, geo_jsons)):
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # There can be multiple polygons in one file, but they should all be constant
                # TODO: Verify this assumption
                annotation_type = _infer_shapely_type(data[0]["type"], None if label_map is None else label_map[idx])
                metadata = {"type": annotation_type, "mpp": curr_mpp, "label": label}

            annotations.append(WholeSlideAnnotation([shape(x) for x in data], metadata))

        return cls(annotations)

    @classmethod
    def from_asap_xml(cls, asap_xml, labels=None, mpp=None):
        tree = ET.parse(asap_xml)
        opened_annotation = tree.getroot()
        annotations = []
        for parent in opened_annotation:
            for child in parent:
                if child.tag != "Annotation":
                    continue
                annotation_type = _ASAP_TYPES[child.attrib.get("Type").lower()]
                label = child.attrib.get("PartOfGroup").lower().strip()
                coordinates = _parse_asap_coordinates(child, annotation_type)

                metadata = {"type": annotation_type, "mpp": mpp, "label": label}
                annotations.append(WholeSlideAnnotation([coordinates], metadata))
        return cls(annotations)

    @property
    def available_labels(self) -> List[str]:
        return list(self._label_dict.keys())

    def get_annotations_for_labels(self, labels: Iterable[str]) -> Dict[str, WholeSlideAnnotation]:
        return {z.label: z for z in (x for x in self._annotations) if z.label in labels}

    def label_to_mpp(self, label: str) -> Optional[float]:
        return self._label_dict[label][1]

    def label_to_type(self, label: str) -> AnnotationType:
        return self._label_dict[label][2]

    @staticmethod
    def filter_annotations(
        annotations: STRtree,
        coordinates,  # Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        region_size,  #: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        crop_func: Optional[Callable] = None,
    ) -> List[ShapelyTypes]:
        box = coordinates.tolist() + (coordinates + region_size).tolist()
        # region can be made into a box class
        query_box = geometry.box(*box)
        annotations = annotations.query(query_box)

        if crop_func is not None:
            annotations = [x for x in (crop_func(_, query_box) for _ in annotations) if x]

        return annotations

    def __getitem__(self, label: str) -> WholeSlideAnnotation:
        index = self._label_dict[label][0]
        return self._annotations[index]

    def __str__(self):
        return f"AnnotationParser(labels={self.available_labels})"

    def read_region(
        self,
        coordinates,  #: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        region_size,  #: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        mpp: float,
    ) -> Dict[str, List[ShapelyTypes]]:
        scaling = {
            k: self.label_to_mpp(k) / mpp if self.label_to_mpp(k) else 1.0 / mpp for k in self.available_labels
        }
        filtered_annotations = {
            k: self.filter_annotations(
                self._annotation_trees[k],
                np.asarray(coordinates) / scaling[k],
                np.asarray(region_size) / scaling[k],
                _POSTPROCESSORS[self.label_to_type(k)],
            )
            for k in self.available_labels
        }

        transformed_annotations = {
            k: [
                shapely.affinity.affine_transform(
                    annotation, [scaling[k], 0, 0, scaling[k], -coordinates[0], -coordinates[1]]
                )
                for annotation in filtered_annotations[k]
            ]
            for k in self.available_labels
        }

        return transformed_annotations


def _infer_shapely_type(shapely_type: str, label: Optional[str] = None) -> AnnotationType:
    if label:
        return label

    if shapely_type in ["Polygon", "MultiPolygon"]:
        return AnnotationType.POLYGON
    elif shapely_type == "Point":
        return AnnotationType.POINT
    else:  # LineString, MultiPoint, MultiLineString
        raise RuntimeError(f"Not a supported shapely type: {shapely_type}")


def _infer_asap_type():
    pass


def _parse_asap_coordinates(annotation_structure, annotation_type):
    coordinates = []
    coordinate_structure = annotation_structure[0]
    for coordinate in coordinate_structure:
        coordinates.append(
            (
                float(coordinate.get("X").replace(",", ".")),
                float(coordinate.get("Y").replace(",", ".")),
            )
        )

    if annotation_type == AnnotationType.POLYGON:
        coordinates = shapely.geometry.Polygon(coordinates)
    elif annotation_type == AnnotationType.BOX:
        raise NotImplementedError
    elif annotation_type == AnnotationType.POINT:
        coordinates = shapely.geometry.MultiPoint(coordinates)
    else:
        raise RuntimeError

    return coordinates





#
# if __name__ == "__main__":
#     data = AnnotationParser.from_asap_xml("/Users/jteuwen/103S.xml", mpp=None)

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

# TODO: We need to return our own class of annotations, not just shapely objects.
# TODO: We have POLYGON, POINT, BOX. There is already the internal use of ANNOTATIONTYPE

"""
import errno
import json
import os
import pathlib
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import shapely
import shapely.validation
from shapely import geometry
from shapely.geometry import shape
from shapely.strtree import STRtree

from dlup.utils.types import GenericNumber, PathLike

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
    """Class to hold the annotations of one specific label for a whole slide image"""

    def __init__(self, data):
        self.type = data["type"]
        self._annotations = data["coordinates"]
        self.mpp = data["mpp"]
        self.label = data["label"]

    def as_strtree(self) -> STRtree:
        return STRtree(self._annotations)

    def as_list(self) -> List:
        return self._annotations

    def bounding_boxes(self, scaling=1):
        def _get_bbox(z):
            return z.min(axis=0).tolist() + (z.max(axis=0) - z.min(axis=0)).tolist()

        data = [np.asarray(_.envelope.exterior.coords) * scaling for _ in self.as_list()]
        return [_get_bbox(_) for _ in data]

    def __len__(self) -> int:
        return len(self._annotations)

    def __str__(self) -> str:
        return f"WholeSlideAnnotation(label={self.label}, length={self.__len__()})"


class SlideAnnotations:
    def __init__(self, annotations: Dict[str, WholeSlideAnnotation], labels: List[str] = None):
        self._annotations = {k: WholeSlideAnnotation(v) for k, v in annotations.items()}
        # Now we have a dict of label: annotations.
        self._label_dict = {k: v.type for k, v in self._annotations.items()}
        self.available_labels = sorted(list(self._annotations.keys()))
        self._annotation_trees = {label: self[label].as_strtree() for label in self.available_labels}

    @classmethod
    def from_geojson(
        cls: Type[_TAnnotationParser],
        geo_jsons,  # : Iterable[PathLike],
        labels: List[str],
        mpp: float = None,
        label_map: Optional[Dict[float, PathLike]] = None,
    ) -> _TAnnotationParser:
        annotations = defaultdict(dict)
        if isinstance(mpp, float):
            mpp = [mpp] * len(labels)

        for idx, (curr_mpp, label, path) in enumerate(zip(mpp, labels, geo_jsons)):
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as annotation_file:
                data = [shape(x) for x in json.load(annotation_file)]
                # There can be multiple polygons in one file, but they should all have to be constant(same mpp value)
                # TODO: Verify this assumption
                annotation_types = [
                    _infer_shapely_type(polygon.type, None if label_map is None else label_map[idx])
                    for polygon in data
                ]
                # In the above line, we make a list of shapely types since there maybe multiple annotations for a particular label.
                # However, it is assumed that all annotations corresponding to a particular label are of the same type.
                # Therefore, in the next line, we assign "type" to the first member of this list
                annotations[label].update(
                    {"type": annotation_types[0], "label": label, "mpp": curr_mpp, "coordinates": data}
                )

        return cls(annotations)

    @classmethod
    def from_asap_xml(cls, asap_xml, label_map=None, mpp=None):
        tree = ET.parse(asap_xml)
        opened_annotation = tree.getroot()
        annotations = dict()
        opened_annotations = 0
        for parent in opened_annotation:
            for child in parent:
                if child.tag != "Annotation":
                    continue
                label = child.attrib.get("PartOfGroup").lower().strip()

                # If we have a label map and there is nothing defined, then continue.
                if label_map is not None and label not in label_map:
                    continue

                annotation_type = _ASAP_TYPES[child.attrib.get("Type").lower()]
                coordinates = _parse_asap_coordinates(child, annotation_type)

                if not coordinates.is_valid:
                    coordinates = shapely.validation.make_valid(coordinates)

                # It is possible there have been linestrings or so added.
                if isinstance(coordinates, shapely.geometry.collection.GeometryCollection):
                    split_up = [_ for _ in coordinates.geoms if _.area > 0]
                    if len(split_up) != 1:
                        raise RuntimeError(f"Got unexpected object.")
                    coordinates = split_up[0]

                if coordinates.area == 0:
                    continue

                # Sometimes we have two adjecent polygons which can be split
                if isinstance(coordinates, shapely.geometry.multipolygon.MultiPolygon):
                    coordinates_list = coordinates.geoms
                else:
                    # Explicitly turn into a list
                    coordinates_list = [coordinates]

                for coordinates in coordinates_list:
                    # If we have a label map function, we apply it to the coordinates.
                    if label_map is not None and label_map[label] is not None:
                        coordinates, annotation_type = label_map[label](coordinates, mpp)

                    metadata = {"type": annotation_type, "mpp": mpp, "label": label}

                    if label not in annotations:
                        annotations[label] = metadata
                        annotations[label]["coordinates"] = [coordinates]
                    else:
                        annotations[label]["coordinates"].append(coordinates)

                    opened_annotations += 1

        return cls(annotations)

    def label_to_type(self, label: str) -> AnnotationType:
        return self._label_dict[label]

    def __getitem__(self, label: str) -> WholeSlideAnnotation:
        return self._annotations[label]

    def __str__(self):
        return f"SlideAnnotations(labels={self.available_labels})"

    def read_region(
        self,
        coordinates,  #: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        region_size,  #: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        scaling: float,
    ) -> List[Tuple[str, ShapelyTypes]]:

        box = list(coordinates) + list(np.asarray(coordinates) + np.asarray(region_size))
        box = (np.asarray(box) / scaling).tolist()
        query_box = geometry.box(*box)

        filtered_annotations = []
        for k in self.available_labels:
            curr_annotations = self._annotation_trees[k].query(query_box)
            for v in curr_annotations:
                filtered_annotations.append((k, v))

        # Sort on name
        filtered_annotations = sorted(filtered_annotations, key=lambda x: x[0])
        # Sort on area (largest to smallest)
        filtered_annotations = sorted(filtered_annotations, key=lambda x: x[1].area, reverse=True)

        cropped_annotations = []
        for annotation_name, annotation in filtered_annotations:
            crop_func = _POSTPROCESSORS[self.label_to_type(annotation_name)]
            if crop_func is not None:
                annotation = crop_func(annotation, query_box)
            if annotation:
                cropped_annotations.append((annotation_name, annotation))

        transformation_matrix = [scaling, 0, 0, scaling, -coordinates[0], -coordinates[1]]

        output = []
        for annotation_name, annotation in cropped_annotations:
            annotation = shapely.affinity.affine_transform(annotation, transformation_matrix)
            if isinstance(annotation, (geometry.MultiPolygon, geometry.GeometryCollection)):
                output += [(annotation_name, _) for _ in annotation.geoms if _.area > 0]
            else:
                output.append((annotation_name, annotation))

        # TODO: This can be an annotation container.
        # Should we split it here?
        return output


def _infer_shapely_type(shapely_type: Union[list, str], label: Optional[str] = None) -> AnnotationType:
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


def _parse_asap_coordinates(annotation_structure: List, annotation_type: AnnotationType) -> ShapelyTypes:
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
        raise RuntimeError(f"Annotation type not supported. Got {annotation_type}.")

    return coordinates

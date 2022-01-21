from typing import List, Optional, Union, Dict

from shapely import geometry
from shapely.strtree import STRtree
import shapely
import pathlib
from enum import Enum
from typing import NamedTuple
from collections.abc import Sequence
import numpy as np
import errno
import json
from dlup import SlideImage
from dlup import BoundaryMode, SlideImage
from dlup.tiling import Grid
from dlup.utils.types import PathLike
from shapely.geometry import shape
from typing import Tuple
import os

"""
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

This can be a named tuple or so so it can be evaluated lazily.

"""


# How to
class AnnotationType(Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"


_POSTPROCESSORS = {
    AnnotationType.POINT: lambda x, region: x,
    AnnotationType.BOX: lambda x, region: x.intersection(region),
    AnnotationType.POLYGON: lambda x, region: x.intersection(region),
}


# class PolygonAnnotation:
#     pass
#
#
# class BoxAnnotation:
#     pass
#
#
# class PointAnnotation:
#     pass
#
#
# class Annotation_(NamedTuple):
#     label: str
#     type: AnnotationType
#     data: List = []
#     annotation_res: Optional[float] = None
#
#


class SlideAnnotation:  # (Sequence):
    def __init__(self, data):
        # Convert to datatype
        self.type = AnnotationType[data["type"].upper()]
        self._annotations = data["data"]
        self.mpp = data["mpp"]
        self.label = data["label"]

    def as_strtree(self):
        return STRtree(self._annotations)


class AnnotationParser:
    def __init__(self, annotations):
        self._annotations = annotations

        self._label_to_mpp = {annotation.label: annotation.mpp for annotation in annotations}
        self._label_to_type = {annotation.label: annotation.type for annotation in annotations}

    @classmethod
    def from_geojson(
        cls,
        geo_jsons: List[PathLike],
        labels: List[str],
        mpp=None,
        label_map=None,
    ) -> object:
        annotations = []
        if isinstance(float, mpp):
            mpp = [mpp] * len(labels)

        for curr_mpp, label, annotation_label, path in zip(mpp, labels, label_map, geo_jsons):
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                annotation_type = _infer_shapely_type(data, annotation_label)
                data += [shape(x) for x in data]
                data["type"] = annotation_type
                data["mpp"] = curr_mpp
                data["label"] = label

            annotations.append(SlideAnnotation(data))

        return cls(annotations)

    @property
    def available_labels(self) -> List[str]:
        return list(self._label_to_type.keys())

    def get_annotations_for_labels(self, labels) -> Dict[SlideAnnotation]:
        return {z.label: z for z in (x for x in self._annotations) if z.label in labels}

    def label_to_mpp(self, label) -> float:
        return self._label_to_mpp[label]

    def label_to_type(self, label) -> AnnotationType:
        return self._label_to_type[label]

    @staticmethod
    def filter_annotations(annotations, coordinates, region_size, crop_func=None):
        box = coordinates.tolist() + (coordinates + region_size).tolist()
        # region can be made into a box class
        query_box = geometry.box(*box)
        annotations = annotations.query(query_box)

        if crop_func is not None:
            annotations = [x for x in (crop_func(_, query_box) for _ in annotations) if x]

        return annotations

    def __getitem__(self, label) -> SlideAnnotation:
        annotation = self.get_annotations_for_labels([label])
        z = SlideAnnotation(annotation[label])
        return z


def _infer_shapely_type(shapely_type, label=None) -> AnnotationType:
    if label:
        return label

    if shapely_type in ["Polygon", "MultiPolygon"]:
        return AnnotationType.POLYGON
    elif shapely_type == "Point":
        return AnnotationType.POINT
    else:  # LineString, MultiPoint, MultiLineString
        raise RuntimeError(f"Not a supported shapely type: {shapely_type}")

class SlideScoreParser:
    def __init__(self, data):
        pass


class ShapelyAnnotations:
    pass


def get_parser(parser_name):
    pass


class SlideAnnotations:  # Handle all annotations for one slide
    STREE = {}
    TYPES = ("points", "boxes", "polygon")

    def __init__(self, parser, labels=None):
        self._labels = labels  # T

        self._parser = parser
        self._labels = labels if labels else parser.available_labels

        # Create the trees, and load in memory.
        # TODO: How to ensure memory is not being used to keep the annotations themselves? This is enough.
        self._annotation_trees = {label: parser[label].as_strtree() for label in self._labels}
        # Can we do parser.close()?

    def get_region(self, coordinates, region_size, mpp):  # coordinates at which mpp
        scaling = {k: self._parser.label_to_mpp(k) / mpp for k in self._labels}
        filtered_annotations = {
            k: self._parser.filter_annotations(
                self._annotation_trees[k],
                np.asarray(coordinates) / scaling[k],
                np.asarray(region_size) / scaling[k],
                _POSTPROCESSORS[self._parser.label_to_type(k)],
            )
            for k in self._labels
        }

        transformed_annotations = {
            k: [
                shapely.affinity.affine_transform(
                    annotation, [scaling[k], 0, 0, scaling[k], -coordinates[0], -coordinates[1]]
                )
                for annotation in filtered_annotations[k]
            ]
            for k in self._labels
        }

        return transformed_annotations

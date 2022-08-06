# coding=utf-8
# Copyright (c) dlup contributors
"""
Annotation module for dlup.

There are three types of annotations, in the `AnnotationType` variable:
- points
- boxes (which are internally polygons)
- polygons

When working with annotations it is assumed that for each label, the type is always the same.
So a label e.g., "lymphocyte" would always be a box and not suddenly a polygon.
In the latter case you better have labels such as `lymphocyte_point`,
'lymphocyte_box` or so.

Assumed:
- The type of object (point, box, polygon) is fixed per label.
- The mpp is fixed per label.

Also the ASAP XML data format is supported.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypedDict, TypeVar, Union

import numpy as np
import shapely
import shapely.validation
from shapely import geometry
from shapely.strtree import STRtree
from shapely.validation import make_valid

from dlup.types import GenericNumber, PathLike

_TWsiAnnotations = TypeVar("_TWsiAnnotations", bound="WsiAnnotations")
ShapelyTypes = Union[shapely.geometry.Point, shapely.geometry.MultiPolygon, shapely.geometry.Polygon]


class GeoJsonDict(TypedDict):
    """
    TypedDict for standard GeoJSON output
    """

    type: str
    features: List[Dict[str, Union[str, Dict[str, str]]]]


class AnnotationType(Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"
    IMAGELEVEL = "imagelevel"


class Point(shapely.geometry.Point):
    # https://github.com/shapely/shapely/issues/1233#issuecomment-1034324441
    _id_to_attrs: ClassVar[Dict[str, Any]] = {}
    __slots__ = (
        shapely.geometry.Point.__slots__
    )  # slots must be the same for assigning __class__ - https://stackoverflow.com/a/52140968
    name: str  # For documentation generation and static type checking

    def __init__(self, coord: Union[shapely.geometry.Point, Tuple[float, float]], label: str) -> None:
        self._id_to_attrs[str(id(self))] = dict(label=label)

    @property
    def type(self):
        return AnnotationType.POINT

    def __new__(cls, coord: Tuple[float, float], *args, **kwargs) -> "Point":
        point = super().__new__(cls, coord)
        point.__class__ = cls
        return point

    def __del__(self) -> None:
        del self._id_to_attrs[str(id(self))]

    def __getattr__(self, name: str) -> Any:
        try:
            return Point._id_to_attrs[str(id(self))][name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __str__(self) -> str:
        return f"{self.label}, {self.wkt}"


class Polygon(shapely.geometry.Polygon):
    # https://github.com/shapely/shapely/issues/1233#issuecomment-1034324441
    _id_to_attrs: ClassVar[Dict[str, Any]] = {}
    __slots__ = (
        shapely.geometry.Polygon.__slots__
    )  # slots must be the same for assigning __class__ - https://stackoverflow.com/a/52140968
    name: str  # For documentation generation and static type checking

    def __init__(self, coord: Union[shapely.geometry.Polygon, Tuple[float, float]], label: str) -> None:
        self._id_to_attrs[str(id(self))] = dict(label=label)

    @property
    def type(self):
        return AnnotationType.POLYGON

    def __new__(cls, coord: Tuple[float, float], *args, **kwargs) -> "Point":
        point = super().__new__(cls, coord)
        point.__class__ = cls
        return point

    def __del__(self) -> None:
        del self._id_to_attrs[str(id(self))]

    def __getattr__(self, name: str) -> Any:
        try:
            return Polygon._id_to_attrs[str(id(self))][name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __str__(self) -> str:
        return f"{self.label}, {self.wkt}"


def shape(coordinates, label):
    geom_type = coordinates.get("type").lower()
    if geom_type == "point":
        return Point(coordinates["coordinates"], label=label)
    elif geom_type == "polygon":
        return Polygon(coordinates["coordinates"][0], label=label)
    else:
        raise NotImplementedError


_POSTPROCESSORS = {
    AnnotationType.POINT: lambda x, region: x,
    AnnotationType.BOX: lambda x, region: x.intersection(region),
    AnnotationType.POLYGON: lambda x, region: x.intersection(region),
}


def _to_geojson_format(list_of_points: list, answers: dict, label: str) -> GeoJsonDict:
    """
    Convert a given list of annotations into the GeoJSON standard.
    Parameters
    ----------
    list_of_points: list
        A list containing annotation shapes or coordinates.
    answers: Dict
        slidescore answers per annotation
    label: str
        The string identifying the annotation class.
    """

    feature_collection: GeoJsonDict = {
        "type": "FeatureCollection",
        "features": [],
    }

    features: List[Any] = []
    properties: Dict[str, Union[str, Dict[str, str]]] = {
        "object_type": "annotation",
        "classification": {
            "name": label,
        },
    }
    idx = 0
    for index, data in enumerate(list_of_points):
        if data.type != "Point":
            modified_on = answers[index]["modifiedOn"]
        geometry = shapely.geometry.mapping(data)
        features.append(
            {
                "id": str(index),
                "type": "Feature",
                "properties": properties,
                "geometry": geometry,
            }
        )
        idx += 1
    feature_collection["features"] = features
    return feature_collection


class WsiSingleLabelAnnotation:
    """Class to hold the annotations of one specific label for a whole slide image"""

    def __init__(self, label: str, type: AnnotationType, coordinates):
        self.__type = type
        self._annotations = coordinates
        self.__label = label

    @property
    def type(self):
        """The type of annotation, e.g. box, polygon or points."""
        return self.__type

    @property
    def label(self):
        """The label name for this annotation."""
        return self.__label

    def append(self, sample):
        self._annotations.append(sample)

    def as_strtree(self) -> STRtree:
        return STRtree(self._annotations)

    def as_list(self) -> List:
        return self._annotations

    def as_json(self) -> List[Dict[str, Any]]:
        """
                Return the annotation as json format.
        .
                ReturnsP
                -------
                Dict
        """
        data = [
            {
                "type": "Feature",
                "properties": {"classification": {"name": _.label}},
                "geometry": shapely.geometry.mapping(_),
            }
            for _ in self._annotations
        ]
        return data

    @property
    def bounding_boxes(self):
        def _get_bbox(z):
            return tuple(z.min(axis=0).tolist()), tuple((z.max(axis=0) - z.min(axis=0)).tolist())

        data = [np.asarray(annotation.envelope.exterior.coords) for annotation in self.as_list()]
        return [_get_bbox(_) for _ in data]

    def __len__(self) -> int:
        return len(self._annotations)

    def __str__(self) -> str:
        return f"{type(self).__name__}(label={self.label}, length={self.__len__()})"


class WsiAnnotations:
    """Class to hold the annotations of all labels specific label for a whole slide image."""

    def __init__(self, annotations: List[WsiSingleLabelAnnotation]):
        self.available_labels = sorted([annotation.label for annotation in annotations])
        if len(set(self.available_labels)) != len(self.available_labels):
            raise ValueError(
                f"annotations should be a list of `WsiSingleLabelAnnotation` with unique labels. "
                f"Got {self.available_labels}."
            )

        # We convert the list internally into a dictionary so we have an easy way to access the data.
        self._annotations = {annotation.label: annotation for annotation in annotations}
        # Now we have a dict of label: annotations.
        self._annotation_trees = {label: self[label].as_strtree() for label in self.available_labels}

    @classmethod
    def from_geojson(
        cls: Type[_TWsiAnnotations],
        geojsons: Iterable[PathLike],
        scaling: float | None = None,
    ) -> _TWsiAnnotations:
        """
        Constructs an WsiAnnotations object from geojson.

        Parameters
        ----------
        geojsons : Iterable
            List of geojsons representing objects. The properties object must have the name which is the label of this
            object.
        scaling : float, optional
            The scaling to apply to the annotations.

        Returns
        -------
        WsiAnnotations

        """
        data = defaultdict(list)
        _scaling = 1.0 if not scaling else scaling
        for idx, path in enumerate(geojsons):
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as annotation_file:
                geojson_dict = json.load(annotation_file)["features"]
                for x in geojson_dict:
                    coordinates = np.asarray(x["geometry"]["coordinates"]) * _scaling
                    x["geometry"]["coordinates"] = coordinates.tolist()
                    _label = x["properties"]["classification"]["name"]
                    data[_label].append(shape(x["geometry"], label=_label))

        # It is assume that a specific label can only be one type (point or polygon)
        annotations: List[WsiSingleLabelAnnotation] = [
            WsiSingleLabelAnnotation(label=k, type=data[k][0].type, coordinates=data[k]) for k in data.keys()
        ]

        return cls(annotations)

    @classmethod
    def from_asap_xml(cls, asap_xml, label_map=None, scaling=None):
        # ASAP is WSI viewer/annotator of https://github.com/computationalpathologygroup/ASAP
        _ASAP_TYPES = {
            "polygon": AnnotationType.POLYGON,
            "rectangle": AnnotationType.BOX,
            "dot": AnnotationType.POINT,
            "spline": AnnotationType.POLYGON,
            "pointset": AnnotationType.POINT,
        }

        tree = ET.parse(asap_xml)
        opened_annotation = tree.getroot()
        annotations: Dict[str, WsiSingleLabelAnnotation] = dict()
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
                coordinates = _parse_asap_coordinates(child, annotation_type, scaling=scaling)

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
                        coordinates, annotation_type = label_map[label](coordinates)

                    if isinstance(coordinates, shapely.geometry.Point):
                        coordinates = Point(coordinates, label=label)
                    elif isinstance(coordinates, shapely.geometry.Polygon):
                        coordinates = Polygon(coordinates, label=label)
                    else:
                        raise NotImplementedError

                    if label not in annotations:
                        annotations[label] = WsiSingleLabelAnnotation(
                            label=label,
                            type=annotation_type,
                            coordinates=[coordinates],
                        )
                    else:
                        annotations[label].append(coordinates)

                    opened_annotations += 1

        return cls(list(annotations.values()))

    def __getitem__(self, label: str) -> WsiSingleLabelAnnotation:
        return self._annotations[label]

    def as_geojson(self):
        data = {"type": "FeatureCollection", "features": []}
        jsons = [self[label].as_json() for label in self.available_labels]

        index = 0
        for json_list in jsons:
            for json_dict in json_list:
                json_dict["id"] = str(index)
                data["features"].append(json_dict)
                index += 1

        return data

    def read_region(
        self,
        coordinates: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
        scaling: float,
        region_size: Union[np.ndarray, Tuple[GenericNumber, GenericNumber]],
    ) -> List[Union[Polygon, Point]]:
        """
        Reads the region of the annotations. API is the same as `dlup.SlideImage` so they can be used in conjunction.

        The process is as follows:
        1. All the annotations which overlap with the requested region of interested are filtered with an STRTree.
        2. The annotations are filtered by name, and subsequently by area from large to small. The reason this is
         implemented this way is because sometimes one can annotate a larger region, and the smaller regions should
         overwrite the previous part. A function `dlup.data.transforms.shapely_to_mask` can be used to convert such
         outputs to a mask.
         3. The annotations are cropped to the region-of-interest, or filtered in case of points. Polygons which
          convert into points after intersection are removed. If it's a image-level label, nothing happens.
         4. The annotation is rescaled and shifted to the origin to match the local patch coordinate system.

         The final returned data is a list of tuples with `(annotation_name, annotation)`.

        Parameters
        ----------
        coordinates: np.ndarray or tuple
        region_size : np.ndarray or tuple
        scaling : float

        Returns
        -------
        List[Tuple[str, ShapelyTypes]]
            List of tuples denoting the name of the annotation and a shapely object.

        Examples
        --------
        1. To read geojson annotations and convert them into masks:

        >>> from pathlib import Path
        >>> from dlup import SlideImage
        >>> import numpy as np
        >>> from rasterio.features import rasterize
        >>> wsi = SlideImage.from_file_path(Path("path/to/image.svs"))
        >>> wsi = wsi.get_scaled_view(scaling=0.5)
        >>> wsi = wsi.read_region(location=(0,0), size=wsi.size)
        >>> annotations = WsiAnnotations.from_geojson([Path("path/to/geojson.json")], labels=["class_name"])
        >>> polygons: list[Polygons] = annotations.read_region(coordinates=(0,0), region_size=wsi.size, scaling=0.01)
        >>> mask = np.zeros(wsi.size, dtype=np.uint8)
        >>> mask = rasterize(polygons, out_shape=(wsi.size[1], wsi.size[0]))
        """

        box = list(coordinates) + list(np.asarray(coordinates) + np.asarray(region_size))
        box = (np.asarray(box) / scaling).tolist()
        query_box = geometry.box(*box)

        filtered_annotations = []
        for k in self.available_labels:
            curr_indices = self._annotation_trees[k].query(query_box)
            curr_annotations = self._annotation_trees[k].geometries[curr_indices]
            for v in curr_annotations:
                filtered_annotations.append((k, v))

        # Sort on name
        filtered_annotations = sorted(filtered_annotations, key=lambda x: x[0])
        # Sort on area (largest to smallest)
        filtered_annotations = sorted(filtered_annotations, key=lambda x: x[1].area, reverse=True)

        cropped_annotations = []
        for annotation_name, annotation in filtered_annotations:
            if annotation.is_valid is False:
                annotation = make_valid(annotation)
            crop_func = _POSTPROCESSORS[self[annotation_name].type]
            if crop_func is not None:
                curr_area = annotation.area
                annotation = crop_func(annotation, query_box)
                post_area = annotation.area
                # Remove annotations which had area before (e.g. polygons) but after cropping are a point.
                if curr_area > 0 and post_area == 0:
                    continue
            if annotation:
                cropped_annotations.append((annotation_name, annotation))

        transformation_matrix = [scaling, 0, 0, scaling, -coordinates[0], -coordinates[1]]

        output: List[Union[Polygon, Point]] = []
        for annotation_name, annotation in cropped_annotations:
            annotation = shapely.affinity.affine_transform(annotation, transformation_matrix)
            if isinstance(
                annotation,
                (geometry.MultiPolygon, geometry.GeometryCollection),
            ):
                output += [self.__cast(annotation_name, _) for _ in annotation.geoms if _.area > 0]

            # TODO: Double check
            elif isinstance(annotation, (geometry.LineString, geometry.multilinestring.MultiLineString)):
                continue

            else:
                # The conversion to an internal format is only done here, because we only support Points and Polygons.
                output.append(self.__cast(annotation_name, annotation))
        return output

    def __cast(self, annotation_name: str, annotation: ShapelyTypes) -> Union[Point, Polygon]:
        """
        Cast the shapely object with annotation_name to internal format.

        Parameters
        ----------
        annotation_name : str
        annotation : ShapelyTypes

        Returns
        -------
        Union[Point, Polygon]

        """
        if self[annotation_name].type == AnnotationType.POINT:
            return Point(annotation, label=annotation_name)
        elif self[annotation_name].type == AnnotationType.POLYGON:
            return Polygon(annotation, label=annotation_name)
        else:
            raise RuntimeError(f"Unexpected type. Got {self[annotation_name].type}.")

    def __str__(self):
        # Create a string for the labels
        output = ""
        for annotation_name in self._annotations:
            output += f"{annotation_name} ({len(self._annotations[annotation_name])}, "

        return f"{type(self).__name__}(labels={output[:-2]})"


def _parse_asap_coordinates(
    annotation_structure: List, annotation_type: AnnotationType, scaling: float | None
) -> ShapelyTypes:
    """
    Parse ASAP XML coordinates into Shapely objects.

    Parameters
    ----------
    annotation_structure : list of strings
    annotation_type : AnnotationType
        The annotation type this structure is representing.
    scaling : float
        Scaling to apply to the coordinates

    Returns
    -------
    Shapely object

    """
    coordinates = []
    coordinate_structure = annotation_structure[0]

    _scaling = 1.0 if not scaling else scaling
    for coordinate in coordinate_structure:
        coordinates.append(
            (
                float(coordinate.get("X").replace(",", ".")) * _scaling,
                float(coordinate.get("Y").replace(",", ".")) * _scaling,
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

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

import copy
import errno
import json
import os
import pathlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable, Type, TypedDict, TypeVar, Union

import numpy as np
import shapely
import shapely.affinity
import shapely.validation
from shapely import geometry
from shapely.strtree import STRtree
from shapely.validation import make_valid

from dlup._exceptions import AnnotationError
from dlup.types import GenericNumber, PathLike
from dlup.utils.imports import DARWIN_SDK_AVAILABLE, PYHALOXML_AVAILABLE

if PYHALOXML_AVAILABLE:
    import pyhaloxml
    import pyhaloxml.shapely

if DARWIN_SDK_AVAILABLE:
    from darwin.utils import parse_darwin_json

_TWsiAnnotations = TypeVar("_TWsiAnnotations", bound="WsiAnnotations")
ShapelyTypes = Union[shapely.geometry.Point, shapely.geometry.MultiPolygon, shapely.geometry.Polygon]


class AnnotationType(Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"


@dataclass(frozen=True)  # Frozen makes the class hashable
class AnnotationClass:
    label: str
    annotation_type: AnnotationType


class GeoJsonDict(TypedDict):
    """
    TypedDict for standard GeoJSON output
    """

    id: str | None
    type: str
    features: list[dict[str, str | dict[str, str]]]


class Point(shapely.geometry.Point):
    # https://github.com/shapely/shapely/issues/1233#issuecomment-1034324441
    _id_to_attrs: ClassVar[dict[str, Any]] = {}
    __slots__ = (
        shapely.geometry.Point.__slots__
    )  # slots must be the same for assigning __class__ - https://stackoverflow.com/a/52140968
    name: str  # For documentation generation and static type checking

    def __init__(self, coord: shapely.geometry.Point | tuple[float, float], label: str | None = None) -> None:
        self._id_to_attrs[str(id(self))] = dict(label=label)

    @property
    def type(self):
        return AnnotationType.POINT

    def __new__(cls, coord: tuple[float, float], *args, **kwargs) -> "Point":
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
    _id_to_attrs: ClassVar[dict[str, Any]] = {}
    __slots__ = (
        shapely.geometry.Polygon.__slots__
    )  # slots must be the same for assigning __class__ - https://stackoverflow.com/a/52140968
    name: str  # For documentation generation and static type checking

    def __init__(self, coord: shapely.geometry.Polygon | tuple[float, float], label: str | None = None) -> None:
        self._id_to_attrs[str(id(self))] = dict(label=label)

    @property
    def type(self):
        return AnnotationType.POLYGON

    def __new__(cls, coord: tuple[float, float], *args, **kwargs) -> "Point":
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


def rescale_geometry(geometry: Union[Point, Polygon], scaling: float | None = None):
    if scaling is None:
        return geometry
    if scaling == 1.0:
        return geometry

    scaled_geometry = shapely.affinity.scale(geometry, scaling, scaling)
    if isinstance(geometry, Polygon):
        return Polygon(scaled_geometry, label=geometry.label)
    elif isinstance(geometry, Point):
        return Point(scaled_geometry, label=geometry.label)
    else:
        raise ValueError(f"geometry type {type(geometry)} is not a valid dlup type.")


def shape(coordinates, label, multiplier: float = 1.0):
    geom_type = coordinates.get("type").lower()
    if geom_type == "point":
        return [Point(np.asarray(coordinates["coordinates"]) * multiplier, label=label)]
    elif geom_type == "multipoint":
        return [Point(np.asarray(c) * multiplier, label=label) for c in coordinates["coordinates"]]
    elif geom_type == "polygon":
        return [Polygon(np.asarray(coordinates["coordinates"][0]) * multiplier, label=label)]
    elif geom_type == "multipolygon":
        multi_polygon = shapely.geometry.MultiPolygon(
            [[np.asarray(c[0]) * multiplier, np.asarray(c[1:]) * multiplier] for c in coordinates["coordinates"]]
        )
        return [Polygon(_, label=label) for _ in multi_polygon.geoms]
    else:
        raise NotImplementedError(f"Not support geom_type {geom_type}")


_POSTPROCESSORS = {
    AnnotationType.POINT: lambda x, region: x,
    AnnotationType.BOX: lambda x, region: x.intersection(region),
    AnnotationType.POLYGON: lambda x, region: x.intersection(region),
}


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

    @property
    def annotation_class(self):
        return AnnotationClass(self.__label, self.__type)

    @label.setter
    def label(self, label):
        self.__label = label

        # TODO: We also need to rewrite all the polygons. This cannot yet be set in-place
        _annotations = []
        for geometry in self._annotations:
            if isinstance(geometry, shapely.geometry.Polygon):
                _annotations.append(Polygon(geometry, label=label))
            elif isinstance(geometry, shapely.geometry.Point):
                _annotations.append(Point(geometry, label=label))
            else:
                raise AnnotationError(f"Unknown annotation type {type(geometry)}.")

        self._annotations = _annotations

    def append(self, sample):
        self._annotations.append(sample)

    def as_strtree(self) -> STRtree:
        return STRtree(self._annotations)

    def as_list(self) -> list:
        return self._annotations

    def as_json(self) -> list[dict[str, Any]]:
        """
        Return the annotation as json format.

        Returns
        -------
        dict
        """
        data = [
            {
                "type": "Feature",
                "properties": {
                    "classification": {
                        "name": _.label,
                        "color": None,
                    },
                },
                "geometry": shapely.geometry.mapping(_),
            }
            for _ in self._annotations
        ]
        return data

    @staticmethod
    def __get_bbox(z):
        return tuple(z.min(axis=0).tolist()), tuple((z.max(axis=0) - z.min(axis=0)).tolist())

    @property
    def bounding_boxes(self):
        data = []
        for annotation in self.as_list():
            if isinstance(annotation, Polygon):
                data.append(np.asarray(annotation.envelope.exterior.coords))
            elif isinstance(annotation, Point):
                # Create a 2D numpy array to represent the point
                point_coords = np.asarray([annotation.x, annotation.y])
                data.append(np.array([point_coords, point_coords]))
        return [self.__get_bbox(_) for _ in data]

    def simplify(self, tolerance: float, *, preserve_topology: bool = True):
        if self.__type != AnnotationType.POLYGON:
            return
        self._annotations = [
            Polygon(annotation.simplify(tolerance, preserve_topology=preserve_topology), label=self.__label)
            for annotation in self._annotations
        ]

    def __len__(self) -> int:
        return len(self._annotations)

    def __str__(self) -> str:
        return f"{type(self).__name__}(label={self.label}, length={self.__len__()})"


class WsiAnnotations:
    """Class to hold the annotations of all labels specific label for a whole slide image."""

    def __init__(self, annotations: list[WsiSingleLabelAnnotation]):
        self.available_labels = sorted(
            [_.annotation_class for _ in annotations],
            key=lambda annotation_class: (annotation_class.label, annotation_class.annotation_type),
        )

        # We convert the list internally into a dictionary so we have an easy way to access the data.
        self._annotations = {annotation.annotation_class: annotation for annotation in annotations}
        # Now we have a dict of label: annotations.
        self._annotation_trees = {label: self[label].as_strtree() for label in self.available_labels}

    def filter(self, labels: str | list[str] | tuple[str]) -> None:
        """
        Filter annotations based on the given label list. If annotations with the same name but a different type are
        present, they will all be added irrespective of the type.

        Parameters
        ----------
        labels : tuple or list
            The list or tuple of labels

        Returns
        -------
        None
        """

        _labels = [labels] if isinstance(labels, str) else labels
        self.available_labels = [_ for _ in self.available_labels if _.label in _labels]
        self._annotations = {k: v for k, v in self._annotations.items() if k.label in _labels}
        self._annotation_trees = {k: v for k, v in self._annotation_trees.items() if k.label in _labels}

    def relabel(self, labels: tuple[tuple[AnnotationClass, AnnotationClass], ...]) -> None:
        """
        Rename labels in the class in-place.

        Parameters
        ----------
        labels : tuple
            Tuple of tuples of the form (original_annotation_class, new_annotation_class).
            Labels which are not present will be kept the same.

        Returns
        -------
        None
        """
        # Create a dictionary with the mapping
        mapping = {k: k for k in self.available_labels}

        for old_annotation_class, new_annotation_class in labels:
            if old_annotation_class.annotation_type != new_annotation_class.annotation_type:
                raise AnnotationError(
                    f"Relabel error. Annotation types do not match for {old_annotation_class.label}."
                )

            if old_annotation_class not in self:
                raise AnnotationError(f"Relabel error. Label {old_annotation_class.label} not currently present.")
            mapping[old_annotation_class] = new_annotation_class

        self.available_labels = sorted([mapping[label] for label in self.available_labels], key=lambda x: x.label)

        _annotations = {}
        for annotation_class in self._annotations:
            _geometry = self._annotations[annotation_class]
            _geometry.annotation_type = mapping[annotation_class].annotation_type
            _geometry.label = mapping[annotation_class].label
            _annotations[mapping[annotation_class]] = _geometry
        self._annotations = _annotations
        self._annotation_trees = {
            annotation_class: self[annotation_class].as_strtree() for annotation_class in self.available_labels
        }

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Return the bounding box of all annotations.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            Bounding box of the form ((x, y), (w, h)).
        """
        all_boxes = []
        for label in self.available_labels:
            curr_bboxes = self[label].bounding_boxes
            for box_start, box_size in curr_bboxes:
                max_x, max_y = box_start[0] + box_size[0], box_start[1] + box_size[1]
                all_boxes.append(shapely.geometry.box(*box_start, max_x, max_y))

        boxes_as_multipolygon = shapely.geometry.MultiPolygon(all_boxes)
        min_x, min_y, max_x, max_y = boxes_as_multipolygon.bounds
        return (min_x, min_y), (max_x - min_x, max_y - min_y)

    def copy(self) -> WsiAnnotations:
        """Make a copy of the object."""
        return copy.deepcopy(self)

    @classmethod
    def from_geojson(
        cls: Type[_TWsiAnnotations],
        geojsons: PathLike | Iterable[PathLike],
        scaling: float | None = None,
    ) -> _TWsiAnnotations:
        """
        Constructs an WsiAnnotations object from geojson.

        Parameters
        ----------
        geojsons : Iterable, or PathLike
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
        if isinstance(geojsons, str):
            _geojsons: Iterable[Any] = [pathlib.Path(geojsons)]

        _geojsons = [geojsons] if not isinstance(geojsons, (tuple, list)) else geojsons
        for idx, path in enumerate(_geojsons):
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as annotation_file:
                geojson_dict = json.load(annotation_file)["features"]
                for x in geojson_dict:
                    _label = x["properties"]["classification"]["name"]
                    _geometry = shape(x["geometry"], label=_label, multiplier=_scaling)
                    for _ in _geometry:
                        data[_label].append(_)

        # It is assumed that a specific label can only be one type (point or polygon)
        annotations: list[WsiSingleLabelAnnotation] = [
            WsiSingleLabelAnnotation(label=k, type=data[k][0].type, coordinates=data[k]) for k in data.keys()
        ]

        return cls(annotations)

    @classmethod
    def from_asap_xml(
        cls,
        asap_xml: PathLike,
        scaling: float | None = None,
    ):
        """
        Read annotations as an ASAP [1] XML file. ASAP is a tool for viewing and annotating whole slide images.

        Parameters
        ----------
        asap_xml : PathLike
            Path to ASAP XML annotation file.
        scaling : float, optional

        References
        ----------
        .. [1] https://github.com/computationalpathologygroup/ASAP

        Returns
        -------
        WsiAnnotations
        """
        _ASAP_TYPES = {
            "polygon": AnnotationType.POLYGON,
            "rectangle": AnnotationType.BOX,
            "dot": AnnotationType.POINT,
            "spline": AnnotationType.POLYGON,
            "pointset": AnnotationType.POINT,
        }

        tree = ET.parse(asap_xml)
        opened_annotation = tree.getroot()
        annotations: dict[str, WsiSingleLabelAnnotation] = dict()
        opened_annotations = 0
        for parent in opened_annotation:
            for child in parent:
                if child.tag != "Annotation":
                    continue
                label = child.attrib.get("PartOfGroup").lower().strip()  # type: ignore

                _type = child.attrib.get("Type").lower()  # type: ignore
                annotation_type = _ASAP_TYPES[_type]
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
                    # TODO: There is a cast function
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

    @classmethod
    def from_halo_xml(cls, halo_xml: PathLike, scaling: float | None = None) -> WsiAnnotations:
        """
        Read annotations as a Halo [1] XML file.
        This function requires `pyhaloxml` [2] to be installed.

        Parameters
        ----------
        halo_xml : PathLike
            Path to the Halo XML file.
        scaling : float, optional
            The scaling to apply to the annotations.

        References
        ----------
        .. [1] https://indicalab.com/halo/
        .. [2] https://github.com/rharkes/pyhaloxml

        Returns
        -------
        WsiAnnotations
        """
        if not PYHALOXML_AVAILABLE:
            raise RuntimeError("`pyhaloxml` is not available. Install using `python -m pip install pyhaloxml`.")

        output = defaultdict(list)
        with pyhaloxml.HaloXMLFile(halo_xml) as hx:
            for layer in hx.layers:
                shapely_multipolygon = pyhaloxml.shapely.layer_to_shapely(layer)
                for shapely_polygon in shapely_multipolygon.geoms:
                    curr_polygon = rescale_geometry(Polygon(shapely_polygon, label=layer.name), scaling=scaling)
                    output[layer.name].append(Polygon(curr_polygon, label=layer.name))

        annotations: list[WsiSingleLabelAnnotation] = []
        for label in output:
            annotations.append(
                WsiSingleLabelAnnotation(label=label, type=AnnotationType.POLYGON, coordinates=output[label])
            )

        return cls(annotations)

    @classmethod
    def from_darwin_json(cls, darwin_json: PathLike, scaling: float | None = None) -> WsiAnnotations:
        if not DARWIN_SDK_AVAILABLE:
            raise RuntimeError("`darwin` is not available. Install using `python -m pip install darwin-py`.")

        annotations = defaultdict(list)
        _scaling = 1.0 if not scaling else scaling

        darwin_an = parse_darwin_json(darwin_json, None).annotations

        for curr_annotation in darwin_an:
            name = curr_annotation.annotation_class.name
            annotation_type = _v7_annotation_type_to_dlup_annotation_type(
                curr_annotation.annotation_class.annotation_type
            )
            key = (name, annotation_type)
            curr_data = curr_annotation.data

            if annotation_type == AnnotationType.POINT:
                curr_point = Point((curr_data["x"], curr_data["y"]), label=name)
                curr_point = rescale_geometry(curr_point, scaling=scaling)

                annotations[key].append(curr_point)
            elif annotation_type == AnnotationType.POLYGON:
                curr_polygon = Polygon([(_["x"], _["y"]) for _ in curr_data["path"]], label=name)
                curr_polygon = rescale_geometry(curr_polygon, scaling=scaling)
                annotations[key].append(Polygon(curr_polygon, label=name))
            elif annotation_type == AnnotationType.BOX:
                x, y, h, w = curr_data.values()
                curr_polygon = shapely.geometry.box(x, y, x + w, y + h)
                curr_polygon = rescale_geometry(curr_polygon, scaling=scaling)
                annotations[key].append(Polygon(curr_polygon, label=name))
            else:
                ValueError(f"Annotation type {annotation_type} is not supported.")

        # Now we can make WsiSingleLabel annotations
        output = []
        for label, annotation_class in annotations:
            output.append(
                WsiSingleLabelAnnotation(
                    label=label, type=annotation_class, coordinates=annotations[(label, annotation_class)]
                )
            )
        return cls(output)

    def __getitem__(self, label: AnnotationClass) -> WsiSingleLabelAnnotation:
        return self._annotations[label]

    def as_geojson(self, split_per_label=False) -> GeoJsonDict | list[tuple[str, GeoJsonDict]]:
        """
        Output the annotations as proper geojson.

        Parameters
        ----------
        split_per_label : bool
            If set will return a list of a tuple with str, GeoJSON dict for this specific label.

        Returns
        -------
        list of (str, GeoJsonDict)
        """
        jsons = [(label, self[label].as_json()) for label in self.available_labels]
        if split_per_label:
            per_label_jsons = []
            for label, json_per_label in jsons:
                per_label_data: GeoJsonDict = {"type": "FeatureCollection", "features": [], "id": None}
                for idx, json_dict in enumerate(json_per_label):
                    per_label_data["features"].append(json_dict)
                    per_label_data["id"] = str(idx)
                per_label_jsons.append((label, per_label_data))
            return per_label_jsons

        data: GeoJsonDict = {"type": "FeatureCollection", "features": [], "id": None}
        index = 0
        for label, json_per_label in jsons:
            for json_dict in json_per_label:
                json_dict["id"] = str(index)
                data["features"].append(json_dict)
                index += 1
        return data

    def simplify(self, tolerance: float, *, preserve_topology: bool = True):
        """Simplify the polygons in the annotation (i.e. reduce points). Other annotations will remain unchanged.
        All points in the resulting polygons object will be in the tolerance distance of the original polygon.

        Parameters
        ----------
        tolerance : float
        preserve_topology : bool
            Preserve the topology, if false, this function will be much faster. Internally the `shapely` simplify
            algorithm is used.

        Returns
        -------
        None

        """
        for k in self._annotations:
            self._annotations[k].simplify(tolerance, preserve_topology=preserve_topology)

    def read_region(
        self,
        coordinates: np.ndarray | tuple[GenericNumber, GenericNumber],
        scaling: float,
        region_size: np.ndarray | tuple[GenericNumber, GenericNumber],
    ) -> list[Polygon | Point]:
        """Reads the region of the annotations. API is the same as `dlup.SlideImage` so they can be used in conjunction.

        The process is as follows:

        1.  All the annotations which overlap with the requested region of interested are filtered with an STRTree.
        2.  The annotations are filtered by name (if the names are the same they are in order of POINT, BOX and POLYGON)
            , and subsequently by area from large to small. The reason this is implemented this way is because sometimes
            one can annotate a larger region, and the smaller regions should overwrite the previous part.
            A function `dlup.data.transforms.convert_annotations` can be used to convert such outputs to a mask.
        3.  The annotations are cropped to the region-of-interest, or filtered in case of points. Polygons which
            convert into points after intersection are removed. If it's a image-level label, nothing happens.
        4.  The annotation is rescaled and shifted to the origin to match the local patch coordinate system.

        The final returned data is a list of `dlup.annotations.Polygon` or `dlup.annotations.Point`.

        Parameters
        ----------
        coordinates: np.ndarray or tuple
        region_size : np.ndarray or tuple
        scaling : float

        Returns
        -------
        list[tuple[str, ShapelyTypes]]
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

        The polygons can be converted to masks using `dlup.data.transforms.convert_annotations` or
        `dlup.data.transforms.ConvertAnnotationsToMask`.
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
        filtered_annotations = sorted(filtered_annotations, key=lambda x: x[0].label)
        # Sort on area (largest to smallest)
        filtered_annotations = sorted(filtered_annotations, key=lambda x: x[1].area, reverse=True)

        cropped_annotations = []
        for annotation_class, annotation in filtered_annotations:
            if annotation.is_valid is False:
                annotation = make_valid(annotation)

            crop_func = _POSTPROCESSORS[annotation_class.annotation_type]
            if crop_func is not None:
                curr_area = annotation.area
                # The following function casts this again as a shapely Polygon, so we will need to convert
                # further down the road back to a dlup Polygon.
                annotation = crop_func(annotation, query_box)
                post_area = annotation.area
                # Remove annotations which had area before (e.g. polygons) but after cropping are a point.
                if curr_area > 0 and post_area == 0:
                    continue

            if annotation:
                cropped_annotations.append((annotation_class, annotation))

        transformation_matrix = [scaling, 0, 0, scaling, -coordinates[0], -coordinates[1]]

        output: list[Polygon | Point] = []
        for annotation_class, annotation in cropped_annotations:
            annotation = shapely.affinity.affine_transform(annotation, transformation_matrix)
            # It can occur that single polygon annotations result in being points after being intersected.
            # This part is required because shapely operations on the edited polygons lose the label and type.
            if self[annotation_class].type == AnnotationType.POLYGON and annotation.area == 0:
                continue

            if isinstance(
                annotation,
                (geometry.MultiPolygon, geometry.GeometryCollection),
            ):
                output += [self.__cast(annotation_class, _) for _ in annotation.geoms if _.area > 0]

            # TODO: Double check
            elif isinstance(annotation, (geometry.LineString, geometry.multilinestring.MultiLineString)):
                continue

            else:
                # The conversion to an internal format is only done here, because we only support Points and Polygons.
                output.append(self.__cast(annotation_class, annotation))
        return output

    def __cast(self, annotation_class: AnnotationClass, annotation: ShapelyTypes) -> Point | Polygon:
        """
        Cast the shapely object with annotation_name to internal format.

        Parameters
        ----------
        annotation_class : AnnotationClass
        annotation : ShapelyTypes

        Returns
        -------
        Point or Polygon

        """
        # TODO: There are weird things now with the annotation class and the type. Fix this.
        if self[annotation_class].type == AnnotationType.POINT:
            return Point(annotation, label=annotation_class)
        elif self[annotation_class].type == AnnotationType.POLYGON:
            return Polygon(annotation, label=annotation_class)
        elif self[annotation_class].type == AnnotationType.BOX:
            return Polygon(annotation, label=annotation_class)
        else:
            raise RuntimeError(f"Unexpected type. Got {self[annotation_class].type}.")

    def __contains__(self, item: Union[str, AnnotationClass]) -> bool:
        if isinstance(item, str):
            return item in [_.label for _ in self.available_labels]
        else:
            return item in self.available_labels

    def __add__(self, other: WsiAnnotations) -> WsiAnnotations:
        if set(self.available_labels).intersection(other.available_labels) != set():
            raise AnnotationError(
                f"Can only add annotations with different labels. "
                f"Use `.relabel` or relabel during construction of the object."
            )

        curr_annotations = list(self._annotations.values())
        curr_annotations += list(other._annotations.values())
        return WsiAnnotations(curr_annotations)

    def __str__(self):
        # Create a string for the labels
        output = ""
        for annotation_name in self._annotations:
            output += f"{annotation_name} ({len(self._annotations[annotation_name])}, "

        return f"{type(self).__name__}(labels={output[:-2]})"


def _parse_asap_coordinates(
    annotation_structure: ET.Element, annotation_type: AnnotationType, scaling: float | None
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
                float(coordinate.get("X").replace(",", ".")) * _scaling,  # type: ignore
                float(coordinate.get("Y").replace(",", ".")) * _scaling,  # type: ignore
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


def _v7_annotation_type_to_dlup_annotation_type(annotation_type: str) -> AnnotationType:
    """
    Convert a v7 annotation type to a dlup annotation type.

    Parameters
    ----------
    annotation_type : str
        The annotation type as defined in the v7 annotation format.

    Returns
    -------
    AnnotationType
    """
    if annotation_type == "bounding_box":
        return AnnotationType.BOX
    elif annotation_type == "polygon":
        return AnnotationType.POLYGON
    elif annotation_type == "keypoint":
        return AnnotationType.POINT
    else:
        raise NotImplementedError(f"annotation_type {annotation_type} is not implemented or not a valid dlup type.")

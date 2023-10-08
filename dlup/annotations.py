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
from typing import Any, Callable, ClassVar, Iterable, Type, TypedDict, TypeVar, Union

import numpy as np
import numpy.typing as npt
import shapely
import shapely.affinity
import shapely.validation
from shapely import geometry
from shapely.strtree import STRtree
from shapely.validation import make_valid

from dlup._exceptions import AnnotationError
from dlup.types import GenericNumber, PathLike, ROIType
from dlup.utils.imports import DARWIN_SDK_AVAILABLE, PYHALOXML_AVAILABLE

_TWsiAnnotations = TypeVar("_TWsiAnnotations", bound="WsiAnnotations")
ShapelyTypes = Union[shapely.geometry.Point, shapely.geometry.MultiPolygon, shapely.geometry.Polygon]


class AnnotationType(Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"


class AnnotationSorting(Enum):
    REVERSE = "reverse"
    BY_AREA = "by_area"
    NONE = "none"


@dataclass(frozen=True)  # Frozen makes the class hashable
class AnnotationClass:
    label: str
    a_cls: AnnotationType


class GeoJsonDict(TypedDict):
    """
    TypedDict for standard GeoJSON output
    """

    id: str | None
    type: str
    features: list[dict[str, str | dict[str, str]]]


class Point(shapely.geometry.Point):  # type: ignore
    # https://github.com/shapely/shapely/issues/1233#issuecomment-1034324441
    _id_to_attrs: ClassVar[dict[str, Any]] = {}
    __slots__ = (
        shapely.geometry.Point.__slots__
    )  # slots must be the same for assigning __class__ - https://stackoverflow.com/a/52140968
    name: str  # For documentation generation and static type checking

    def __init__(
        self,
        coord: shapely.geometry.Point | tuple[float, float],
        a_cls: AnnotationClass | None = None,
    ) -> None:
        self._id_to_attrs[str(id(self))] = dict(a_cls=a_cls)

    @property
    def annotation_class(self) -> AnnotationClass:
        return self._id_to_attrs[str(id(self))]["a_cls"]  # type: ignore

    @property
    def type(self) -> AnnotationType:
        return self.annotation_class.a_cls

    @property
    def label(self) -> str:
        return self.annotation_class.label

    def __new__(cls, coord: tuple[float, float], *args: Any, **kwargs: Any) -> "Point":
        point = super().__new__(cls, coord)
        point.__class__ = cls
        return point  # type: ignore

    def __del__(self) -> None:
        del self._id_to_attrs[str(id(self))]

    def __getattr__(self, name: str) -> Any:
        try:
            return Point._id_to_attrs[str(id(self))][name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __str__(self) -> str:
        return f"{self.annotation_class}, {self.wkt}"


class Polygon(shapely.geometry.Polygon):  # type: ignore
    # https://github.com/shapely/shapely/issues/1233#issuecomment-1034324441
    _id_to_attrs: ClassVar[dict[str, Any]] = {}
    __slots__ = (
        shapely.geometry.Polygon.__slots__
    )  # slots must be the same for assigning __class__ - https://stackoverflow.com/a/52140968
    name: str  # For documentation generation and static type checking

    def __init__(
        self,
        coord: shapely.geometry.Polygon | tuple[float, float],
        a_cls: AnnotationClass | None = None,
    ) -> None:
        self._id_to_attrs[str(id(self))] = dict(a_cls=a_cls)

    @property
    def annotation_class(self) -> AnnotationClass:
        return self._id_to_attrs[str(id(self))]["a_cls"]  # type: ignore

    @property
    def type(self) -> AnnotationType:
        return self.annotation_class.a_cls

    @property
    def label(self) -> str:
        return self.annotation_class.label

    def __new__(cls, coord: tuple[float, float], *args: Any, **kwargs: Any) -> "Point":
        point = super().__new__(cls, coord)
        point.__class__ = cls
        return point  # type: ignore

    def __del__(self) -> None:
        del self._id_to_attrs[str(id(self))]

    def __getattr__(self, name: str) -> Any:
        try:
            return Polygon._id_to_attrs[str(id(self))][name]
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __str__(self) -> str:
        return f"{self.annotation_class}, {self.wkt}"


def rescale_geometry(geometry: Union[Point, Polygon], scaling: float | None = None) -> Union[Point, Polygon]:
    if scaling is None:
        return geometry
    if scaling == 1.0:
        return geometry

    scaled_geometry = shapely.affinity.scale(geometry, scaling, scaling)
    if isinstance(geometry, Polygon):
        return Polygon(scaled_geometry, a_cls=geometry.annotation_class)
    elif isinstance(geometry, Point):
        return Point(scaled_geometry, a_cls=geometry.annotation_class)
    else:
        raise ValueError(f"geometry type {type(geometry)} is not a valid dlup type.")


class CoordinatesDict(TypedDict):
    type: str
    coordinates: list[list[list[float]]]


def shape(coordinates: CoordinatesDict, label: str, multiplier: float = 1.0) -> list[Polygon | Point]:
    geom_type = coordinates.get("type", None)
    if geom_type is None:
        raise ValueError("No type found in coordinates.")
    geom_type = geom_type.lower()
    if geom_type == "point":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POINT)
        return [
            Point(
                np.asarray(coordinates["coordinates"]) * multiplier,
                a_cls=annotation_class,
            )
        ]
    elif geom_type == "multipoint":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POINT)
        return [Point(np.asarray(c) * multiplier, a_cls=annotation_class) for c in coordinates["coordinates"]]
    elif geom_type == "polygon":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POLYGON)
        return [
            Polygon(
                np.asarray(coordinates["coordinates"][0]) * multiplier,
                a_cls=annotation_class,
            )
        ]
    elif geom_type == "multipolygon":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POLYGON)
        multi_polygon = shapely.geometry.MultiPolygon(
            [[np.asarray(c[0]) * multiplier, np.asarray(c[1:]) * multiplier] for c in coordinates["coordinates"]]
        )
        return [Polygon(_, a_cls=annotation_class) for _ in multi_polygon.geoms]
    else:
        raise NotImplementedError(f"Not support geom_type {geom_type}")


_POSTPROCESSORS = {
    AnnotationType.POINT: lambda x, region: x,
    AnnotationType.BOX: lambda x, region: x.intersection(region),
    AnnotationType.POLYGON: lambda x, region: x.intersection(region),
}


class SingleAnnotationWrapper:
    """Class to hold the annotations of one specific label (class) for a whole slide image"""

    def __init__(self, a_cls: AnnotationClass, annotation: list[Polygon | Point]):
        self._annotation_class = a_cls
        self._label = a_cls.label
        self._type = a_cls.a_cls
        self._annotation = annotation

    @property
    def type(self) -> AnnotationType:
        """The type of annotation, e.g. box, polygon or points."""
        return self._type

    @property
    def label(self) -> str:
        """The label name for this annotation."""
        return self._label

    @property
    def annotation_class(self) -> AnnotationClass:
        return self._annotation_class

    @annotation_class.setter
    def annotation_class(self, a_cls: AnnotationClass) -> None:
        self._annotation_class = a_cls
        self._label = a_cls.label
        self._type = a_cls.a_cls
        # TODO: We also need to rewrite all the polygons. This cannot yet be set in-place
        _annotations = []
        for _geometry in self._annotation:
            if isinstance(_geometry, shapely.geometry.Polygon):
                _annotations.append(Polygon(_geometry, a_cls=a_cls))
            elif isinstance(_geometry, shapely.geometry.Point):
                _annotations.append(Point(_geometry, a_cls=a_cls))
            else:
                raise AnnotationError(f"Unknown annotation type {type(_geometry)}.")

        self._annotation = _annotations

    def append(self, sample: Polygon | Point) -> None:
        self._annotation.append(sample)

    def as_strtree(self) -> STRtree:
        return STRtree(self._annotation)

    def as_list(self) -> list[Polygon | Point]:
        return self._annotation

    def as_json(self) -> list[Any]:
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
            for _ in self._annotation
        ]
        return data

    @staticmethod
    def _get_bbox(z: npt.NDArray[np.int_ | np.float_]) -> ROIType:
        coords = tuple(z.min(axis=0).tolist())
        size = tuple((z.max(axis=0) - z.min(axis=0)).tolist())
        return (coords[0], coords[1]), (size[0], size[1])

    @property
    def bounding_boxes(self) -> tuple[ROIType, ...]:
        data = []
        for annotation in self.as_list():
            if isinstance(annotation, Polygon):
                data.append(np.asarray(annotation.envelope.exterior.coords))
            elif isinstance(annotation, Point):
                # Create a 2D numpy array to represent the point
                point_coords = np.asarray([annotation.x, annotation.y])
                data.append(np.array([point_coords, point_coords]))
        return tuple([self._get_bbox(_) for _ in data])

    def simplify(self, tolerance: float, *, preserve_topology: bool = True) -> None:
        if self.type != AnnotationType.POLYGON:
            return
        self._annotation = [
            Polygon(
                annotation.simplify(tolerance, preserve_topology=preserve_topology),
                a_cls=self.annotation_class,
            )
            for annotation in self._annotation
        ]

    def __len__(self) -> int:
        return len(self._annotation)

    def __str__(self) -> str:
        return f"{type(self).__name__}(label={self.label}, length={self.__len__()})"


class WsiAnnotations:
    """Class to hold the annotations of all labels specific label for a whole slide image."""

    def __init__(
        self,
        annotations: list[SingleAnnotationWrapper],
        sorting: AnnotationSorting = AnnotationSorting.NONE,
    ):
        self.available_labels = sorted(
            [_.annotation_class for _ in annotations],
            key=lambda annotation_class: (
                annotation_class.label,
                annotation_class.a_cls,
            ),
        )

        # We convert the list internally into a dictionary, so we have an easy way to access the data.
        self._annotations = {annotation.annotation_class: annotation for annotation in annotations}
        # Now we have a dict of label: annotations.
        self._annotation_trees = {a_cls: self[a_cls].as_strtree() for a_cls in self.available_labels}

        self._sorting = sorting

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
        mapping: dict[AnnotationClass, AnnotationClass] = {k: k for k in self.available_labels}

        for old_annotation_class, new_annotation_class in labels:
            if old_annotation_class.a_cls != new_annotation_class.a_cls:
                raise AnnotationError(
                    f"Relabel error. Annotation types do not match for {old_annotation_class.label}."
                )

            if old_annotation_class not in self:
                raise AnnotationError(f"Relabel error. Label {old_annotation_class.label} not currently present.")
            mapping[old_annotation_class] = new_annotation_class

        # TODO: Is thie correct?
        self.available_labels = sorted([mapping[label] for label in self.available_labels], key=lambda x: x.label)

        _annotations = {}
        for annotation_class, single_label_annotation in self._annotations.items():
            single_label_annotation.annotation_class = mapping[annotation_class]
            _annotations[mapping[annotation_class]] = single_label_annotation
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
        for annotation_class in self.available_labels:
            curr_bboxes = self[annotation_class].bounding_boxes
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
        _annotations: list[SingleAnnotationWrapper] = [
            SingleAnnotationWrapper(a_cls=data[k][0].annotation_class, annotation=data[k]) for k in data.keys()
        ]

        return cls(_annotations)

    @classmethod
    def from_asap_xml(
        cls,
        asap_xml: PathLike,
        scaling: float | None = None,
    ) -> WsiAnnotations:
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
        annotations: dict[str, SingleAnnotationWrapper] = dict()
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
                        raise RuntimeError("Got unexpected object.")
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
                    _cls = AnnotationClass(label=label, a_cls=annotation_type)
                    if isinstance(coordinates, shapely.geometry.Point):
                        coordinates = Point(coordinates, a_cls=_cls)
                    elif isinstance(coordinates, shapely.geometry.Polygon):
                        coordinates = Polygon(coordinates, a_cls=_cls)
                    else:
                        raise NotImplementedError

                    if label not in annotations:
                        annotations[label] = SingleAnnotationWrapper(
                            a_cls=_cls,
                            annotation=[coordinates],
                        )
                    else:
                        annotations[label].append(coordinates)

                    opened_annotations += 1

        return cls(list(annotations.values()), sorting=AnnotationSorting.BY_AREA)

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
        import pyhaloxml
        import pyhaloxml.shapely

        output = defaultdict(list)
        with pyhaloxml.HaloXMLFile(halo_xml) as hx:
            for layer in hx.layers:
                shapely_multipolygon = pyhaloxml.shapely.layer_to_shapely(layer)
                _cls = AnnotationClass(label=layer.name, a_cls=AnnotationType.POLYGON)
                for shapely_polygon in shapely_multipolygon.geoms:
                    curr_polygon = rescale_geometry(Polygon(shapely_polygon, a_cls=_cls), scaling=scaling)
                    output[layer.name].append(Polygon(curr_polygon, a_cls=_cls))

        annotations: list[SingleAnnotationWrapper] = []
        for label in output:
            annotations.append(
                SingleAnnotationWrapper(
                    a_cls=AnnotationClass(label=label, a_cls=AnnotationType.POLYGON),
                    annotation=output[label],
                )
            )

        return cls(annotations, sorting=AnnotationSorting.NONE)

    @classmethod
    def from_darwin_json(cls, darwin_json: PathLike, scaling: float | None = None) -> WsiAnnotations:
        if not DARWIN_SDK_AVAILABLE:
            raise RuntimeError("`darwin` is not available. Install using `python -m pip install darwin-py`.")
        import darwin

        all_annotations = defaultdict(list)
        _scaling = 1.0 if not scaling else scaling

        darwin_an = darwin.utils.parse_darwin_json(pathlib.Path(darwin_json), None)

        for curr_annotation in darwin_an.annotations:
            name = curr_annotation.annotation_class.name
            annotation_type = _v7_annotation_type_to_dlup_annotation_type(
                curr_annotation.annotation_class.annotation_type
            )
            key = AnnotationClass(label=name, a_cls=annotation_type)
            curr_data = curr_annotation.data

            _cls = AnnotationClass(label=name, a_cls=annotation_type)
            if annotation_type == AnnotationType.POINT:
                curr_point = Point((curr_data["x"], curr_data["y"]), a_cls=_cls)
                curr_point = rescale_geometry(curr_point, scaling=_scaling)
                all_annotations[key].append(curr_point)
            elif annotation_type == AnnotationType.POLYGON:
                if "path" in curr_data:  # This is a regular polygon
                    curr_polygon = Polygon([(_["x"], _["y"]) for _ in curr_data["path"]], a_cls=_cls)
                    curr_polygon = rescale_geometry(curr_polygon, scaling=_scaling)
                    all_annotations[key].append(Polygon(curr_polygon, a_cls=_cls))
                elif "paths" in curr_data:  # This is a complex polygon which needs to be parsed with the even-odd rule
                    curr_complex_polygon = _parse_darwin_complex_polygon(curr_data)
                    for curr_polygon in curr_complex_polygon.geoms:
                        curr_polygon = rescale_geometry(curr_polygon, scaling=_scaling)
                        all_annotations[key].append(Polygon(curr_polygon, a_cls=_cls))
                else:
                    raise ValueError(f"Got unexpected data keys: {curr_data.keys()}")

            elif annotation_type == AnnotationType.BOX:
                x, y, h, w = curr_data.values()
                curr_polygon = shapely.geometry.box(x, y, x + w, y + h)
                curr_polygon = rescale_geometry(curr_polygon, scaling=_scaling)
                all_annotations[key].append(Polygon(curr_polygon, a_cls=_cls))
            else:
                ValueError(f"Annotation type {annotation_type} is not supported.")

        # Now we can make SingleAnnotationWrapper annotations
        output = []
        for an_cls, annotations in all_annotations.items():
            output.append(SingleAnnotationWrapper(a_cls=an_cls, annotation=annotations))
        return cls(output, sorting=AnnotationSorting.NONE)

    def __getitem__(self, a_cls: AnnotationClass) -> SingleAnnotationWrapper:
        return self._annotations[a_cls]

    def as_geojson(self, split_per_label: bool = False) -> GeoJsonDict | list[tuple[str, GeoJsonDict]]:
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
                per_label_data: GeoJsonDict = {
                    "type": "FeatureCollection",
                    "features": [],
                    "id": None,
                }
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

    def simplify(self, tolerance: float, *, preserve_topology: bool = True) -> None:
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
        coordinates: npt.NDArray[np.int_ | np.float_] | tuple[GenericNumber, GenericNumber],
        scaling: float,
        region_size: npt.NDArray[np.int_ | np.float_] | tuple[GenericNumber, GenericNumber],
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

        if self._sorting == AnnotationSorting.BY_AREA:
            # Sort on name
            filtered_annotations = sorted(filtered_annotations, key=lambda x: x[0].label)
            # Sort on area (largest to smallest)
            filtered_annotations = sorted(filtered_annotations, key=lambda x: x[1].area, reverse=True)
        elif self._sorting == AnnotationSorting.REVERSE:
            filtered_annotations = list(reversed(filtered_annotations))
        else:  # AnnotationSorting.NONE
            pass

        cropped_annotations = []
        for annotation_class, annotation in filtered_annotations:
            if annotation.is_valid is False:
                annotation = make_valid(annotation)

            crop_func = _POSTPROCESSORS[annotation_class.a_cls]
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

        transformation_matrix = [
            scaling,
            0,
            0,
            scaling,
            -coordinates[0],
            -coordinates[1],
        ]

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
                output += [self._cast(annotation_class, _) for _ in annotation.geoms if _.area > 0]

            # TODO: Double check
            elif isinstance(
                annotation,
                (geometry.LineString, geometry.multilinestring.MultiLineString),
            ):
                continue
            else:
                # The conversion to an internal format is only done here, because we only support Points and Polygons.
                output.append(self._cast(annotation_class, annotation))
        return output

    def _cast(self, annotation_class: AnnotationClass, annotation: ShapelyTypes) -> Point | Polygon:
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
            return Point(annotation, a_cls=annotation_class)
        elif self[annotation_class].type == AnnotationType.POLYGON:
            return Polygon(annotation, a_cls=annotation_class)
        elif self[annotation_class].type == AnnotationType.BOX:
            return Polygon(annotation, a_cls=annotation_class)
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
                "Can only add annotations with different labels. "
                "Use `.relabel` or relabel during construction of the object."
            )

        curr_annotations = list(self._annotations.values())
        curr_annotations += list(other._annotations.values())
        return WsiAnnotations(curr_annotations)

    def __str__(self) -> str:
        # Create a string for the labels
        output = ""
        for annotation_name in self._annotations:
            output += f"{annotation_name} ({len(self._annotations[annotation_name])}, "

        return f"{type(self).__name__}(labels={output[:-2]})"


class _ComplexDarwinPolygonWrapper:
    """Wrapper class for a complex polygon (i.e. polygon with holes) from a Darwin annotation."""

    def __init__(self, polygon: shapely.geometry.Polygon):
        self.geom = polygon
        self.hole = False
        self.holes: list[float] = []


def _parse_darwin_complex_polygon(annotation) -> shapely.geometry.MultiPolygon:
    """
    Parse a complex polygon (i.e. polygon with holes) from a Darwin annotation.

    Parameters
    ----------
    annotation : dict

    Returns
    -------
    shapely.geometry.MultiPolygon
    """
    polygons = [
        _ComplexDarwinPolygonWrapper(shapely.geometry.Polygon([(p["x"], p["y"]) for p in path]))
        for path in annotation["paths"]
    ]

    # Naive even-odd rule, but seems to work
    sorted_polygons = sorted(polygons, key=lambda x: x.geom.area, reverse=True)
    for idx, my_polygon in enumerate(sorted_polygons):
        for outer_polygon in reversed(sorted_polygons[:idx]):
            contains = outer_polygon.geom.contains(my_polygon.geom)
            if contains and outer_polygon.hole:
                break
            if outer_polygon.hole:
                continue
            if contains:
                my_polygon.hole = True
                outer_polygon.holes.append(my_polygon.geom.exterior.coords)

    # create complex polygon with MultiPolygon
    complex_polygon = [
        shapely.geometry.Polygon(my_polygon.geom.exterior.coords, my_polygon.holes)
        for my_polygon in sorted_polygons
        if not my_polygon.hole
    ]
    return shapely.geometry.MultiPolygon(complex_polygon)


def _parse_asap_coordinates(
    annotation_structure: ET.Element,
    annotation_type: AnnotationType,
    scaling: float | None,
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
    elif annotation_type in ["polygon", "complex_polygon"]:
        return AnnotationType.POLYGON
    elif annotation_type == "keypoint":
        return AnnotationType.POINT
    else:
        raise NotImplementedError(f"annotation_type {annotation_type} is not implemented or not a valid dlup type.")

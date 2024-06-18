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

Supported file formats:
- ASAP XML
- Darwin V7 JSON
- GeoJSON
- HaloXML
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
from typing import Any, Callable, ClassVar, Iterable, Optional, Type, TypedDict, TypeVar, Union, cast, NamedTuple
import functools
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

class DarwinV7Metadata(NamedTuple):
    label: str
    color: tuple[int, int, int]
    type: AnnotationType


@functools.lru_cache(maxsize=None)
def _get_v7_metadata(filename: pathlib.Path) -> Optional[dict[str, DarwinV7Metadata]]:
    if not DARWIN_SDK_AVAILABLE:
        raise RuntimeError("`darwin` is not available. Install using `python -m pip install darwin-py`.")
    import darwin.path_utils
    if not filename.is_dir():
        raise RuntimeError(f"Provide the path to the root folder of the Darwin V7 annotations")

    v7_metadata_fn = filename / ".v7" / "metadata.json"
    if not v7_metadata_fn.exists():
        return None
    v7_metadata = darwin.path_utils.parse_metadata(v7_metadata_fn)
    output = {}
    for sample in v7_metadata["classes"]:
        annotation_type = _v7_annotation_type_to_dlup_annotation_type(sample["type"])

        label = sample["name"]
        color = sample["color"][5:-1].split(",")
        if color[-1] != "1.0":
            raise RuntimeError("Expected A-channel of color to be 1.0")
        rgb_colors = (int(color[0]), int(color[1]), int(color[2]))

        output[label] = DarwinV7Metadata(label=label, color=rgb_colors, type=annotation_type)

    return output

def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

class AnnotationType(Enum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"
    TAG = "tag"

_ASAP_TYPES = {
    "polygon": AnnotationType.POLYGON,
    "rectangle": AnnotationType.BOX,
    "dot": AnnotationType.POINT,
    "spline": AnnotationType.POLYGON,
    "pointset": AnnotationType.POINT,
}


def _get_geojson_color(properties: dict[str, str | list[int]]) -> Optional[tuple[int, int, int]]:
    """Parse the properties dictionary of a GeoJSON object to get the color.

    Arguments
    ---------
    properties : dict
        The properties dictionary of a GeoJSON object.

    Returns
    -------
    Optional[tuple[int, int, int]]
        The color of the object as a tuple of RGB values.
    """
    color = properties.get("color", None)
    if color is None:
        return None

    return cast(tuple[int, int, int], tuple(color))


class AnnotationSorting(Enum):
    """The ways to sort the annotations. This is used in the constructors of the `WsiAnnotations` class, and applied
    to the output of `WsiAnnotations.read_region()`.

    - REVERSE: Sort the output in reverse order.
    - AREA: Often when the annotation tools do not properly support hierarchical order, one would annotate in a way
        that the smaller objects are on top of the larger objects. This option sorts the output by area, so that the
        larger objects appear first in the output and then the smaller objects.
    - Z_INDEX: Sort the output by the z-index of the annotations. This is useful when the annotations have a z-index
    - NONE: Do not apply any sorting and output as is presented in the input file.
    """

    REVERSE = "reverse"
    AREA = "area"
    Z_INDEX = "z_index"
    NONE = "none"


@dataclass(frozen=True)  # Frozen makes the class hashable
class AnnotationClass:
    """An annotation class. An annotation has two required properties:
    - label: The name of the annotation, e.g., "lymphocyte".
    - a_cls: The type of annotation, e.g., AnnotationType.POINT.

    And two optional properties:
    - color: The color of the annotation as a tuple of RGB values.
    - z_index: The z-index of the annotation. This is useful when the annotations have a z-index.

    Parameters
    ----------
    label : str
        The name of the annotation.
    a_cls : AnnotationType
        The type of annotation.
    color : Optional[tuple[int, int, int]]
        The color of the annotation as a tuple of RGB values.
    z_index : Optional[int]
        The z-index of the annotation.
    """

    label: str
    a_cls: AnnotationType
    color: Optional[tuple[int, int, int]] = None
    z_index: Optional[int] = None

    def __post_init__(self):
        if self.a_cls in (AnnotationType.POINT, AnnotationType.TAG) and self.z_index is not None:
            raise AnnotationError("z_index is not supported for point annotations or tags.")


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

    @property
    def color(self) -> Optional[tuple[int, int, int]]:
        return self.annotation_class.color

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

    @property
    def color(self) -> tuple[int, int, int]:
        return self.annotation_class.color

    @property
    def z_index(self) -> int:
        return self.annotation_class.z_index

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




class CoordinatesDict(TypedDict):
    type: str
    coordinates: list[list[list[float]]]


def shape(
    coordinates: CoordinatesDict,
    label: str,
    color: Optional[tuple[int, int, int]] = None,
    z_index: Optional[int] = None,
) -> list[Polygon | Point]:
    geom_type = coordinates.get("type", None)
    if geom_type is None:
        raise ValueError("No type found in coordinates.")
    geom_type = geom_type.lower()

    if geom_type in ["point", "multipoint"] and z_index is not None:
        raise AnnotationError("z_index is not supported for point annotations.")

    if geom_type == "point":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POINT, color=color, z_index=None)
        return [
            Point(
                np.asarray(coordinates["coordinates"]),
                a_cls=annotation_class,
            )
        ]

    if geom_type == "multipoint":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POINT, color=color, z_index=None)
        return [Point(np.asarray(c), a_cls=annotation_class) for c in coordinates["coordinates"]]

    if geom_type == "polygon":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POLYGON, color=color, z_index=z_index)
        return [
            Polygon(
                np.asarray(coordinates["coordinates"][0]),
                a_cls=annotation_class,
            )
        ]

    if geom_type == "multipolygon":
        annotation_class = AnnotationClass(label=label, a_cls=AnnotationType.POLYGON)
        # the first element is the outer polygon, the rest are holes.
        multi_polygon = shapely.geometry.MultiPolygon(
            [
                [
                    np.asarray(c[0]),
                    [np.asarray(hole) for hole in c[1:]],
                ]
                for c in coordinates["coordinates"]
            ]
        )
        return [Polygon(_, a_cls=annotation_class) for _ in multi_polygon.geoms]

    raise AnnotationError(f"Unsupported geom_type {geom_type}")


_POSTPROCESSORS: dict[AnnotationType, Callable[[Polygon | Point, Polygon], Polygon | Point]] = {
    AnnotationType.POINT: lambda x, region: x,
    AnnotationType.BOX: lambda x, region: x.intersection(region),
    AnnotationType.POLYGON: lambda x, region: x.intersection(region),
}


def _geometry_to_geojson(geometry: Polygon | Point, label: str, color: tuple[int, int, int]) -> dict[str, Any]:
    """Function to convert a geometry to a GeoJSON object.

    Parameters
    ----------
    geometry : Polygon | Point
        A polygon or point object
    label : str
        The label name
    color : tuple[int, int, int]
        The color of the object in RGB values

    Returns
    -------
    dict[str, Any]
        Output dictionary representing the data in GeoJSON

    """
    data = {
        "type": "Feature",
        "properties": {
            "classification": {
                "name": label,
                "color": color,
            },
        },
        "geometry": shapely.geometry.mapping(geometry),
    }
    return data


class AnnotationLayer:
    """Class to hold the annotations of one specific label (class) for a whole slide image"""

    def __init__(self, a_cls: AnnotationClass, data: list[Polygon | Point]):
        self._annotation_class = a_cls
        self._annotation = data

    @property
    def type(self) -> AnnotationType:
        """The type of annotation, e.g. box, polygon or points."""
        return self._annotation_class.a_cls

    @property
    def label(self) -> str:
        """The label name for this annotation."""
        return self._annotation_class.label

    @property
    def color(self) -> Optional[tuple[int, int, int]]:
        """The color of the annotation."""
        return self._annotation_class.color

    @property
    def z_index(self) -> Optional[int]:
        """The z-index of the annotation."""
        return self._annotation_class.z_index

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
        data = [_geometry_to_geojson(_, label=_.label) for _ in self._annotation]
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
        """Simplify the polygons in the annotation in-place using the Douglas-Peucker algorithm.

        Parameters
        ----------
        tolerance : float
            The tolerance parameter for the Douglas-Peucker algorithm.
        preserve_topology : bool
            If True, the algorithm will preserve the topology of the polygons.
        """
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
        layers: list[AnnotationLayer],
        tags: Optional[list[AnnotationClass]] = None,
        sorting: AnnotationSorting = AnnotationSorting.NONE,
        offset_to_slide_bounds: bool = False,
    ):
        """
        Parameters
        ----------
        layers : list[AnnotationLayer]
            A list of layers for a single label.
        tags: Optional[list[AnnotationClass]]
            A list of tags for the annotations. These have to be of type `AnnotationType.TAG`.
        sorting : AnnotationSorting
            How to sort the annotations returned from the `read_region()` function.
        offset_to_slide_bounds : bool
            If true, will set the property `offset_to_slide_bounds` to True. This means that the annotations need
            to be offset to the slide bounds. This is useful when the annotations are read from a file format which
            requires this, for instance HaloXML.
        """
        self._sorting = sorting
        self._offset_to_slide_bounds = offset_to_slide_bounds

        self.available_labels = [_.annotation_class for _ in layers]
        if self._sorting != AnnotationSorting.NONE:
            self.available_labels = sorted(self.available_labels, key=lambda x: (x.label, x.a_cls))

        # We convert the list internally into a dictionary, so we have an easy way to access the data.
        self._layers = {annotation.annotation_class: annotation for annotation in layers}
        # Now we have a dict of label: annotations.
        self._annotation_trees = {a_cls: self[a_cls].as_strtree() for a_cls in self.available_labels}
        self._tags = tags

    @property
    def tags(self):
        return self._tags

    @property
    def offset_to_slide_bounds(self) -> bool:
        """
        If True, the annotations need to be offset to the slide bounds. This is useful when the annotations are read
        from a file format which requires this, for instance HaloXML.

        Returns
        -------
        bool
        """
        return self._offset_to_slide_bounds

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
        self._layers = {k: v for k, v in self._layers.items() if k.label in _labels}
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
        self.available_labels = [mapping[label] for label in self.available_labels]
        if self._sorting != AnnotationSorting.NONE:
            self.available_labels = sorted(self.available_labels, key=lambda x: x.label)

        _annotations = {}
        for annotation_class, single_label_annotation in self._layers.items():
            single_label_annotation.annotation_class = mapping[annotation_class]
            _annotations[mapping[annotation_class]] = single_label_annotation
        self._layers = _annotations
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
        sorting: AnnotationSorting | str = AnnotationSorting.AREA,
    ) -> _TWsiAnnotations:
        """
        Constructs an WsiAnnotations object from geojson.

        Parameters
        ----------
        geojsons : Iterable, or PathLike
            List of geojsons representing objects. The properties object must have the name which is the label of this
            object.
        sorting: AnnotationSorting
            The sorting to apply to the annotations. Check the `AnnotationSorting` enum for more information.
            By default, the annotations are sorted by area.

        Returns
        -------
        WsiAnnotations

        """
        data = defaultdict(list)
        if isinstance(geojsons, str):
            _geojsons: Iterable[Any] = [pathlib.Path(geojsons)]

        _geojsons = [geojsons] if not isinstance(geojsons, (tuple, list)) else geojsons
        for path in _geojsons:
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as annotation_file:
                geojson_dict = json.load(annotation_file)["features"]
                for x in geojson_dict:
                    properties = x["properties"]
                    if "classification" in properties:
                        _label = properties["classification"]["name"]
                        _color = _get_geojson_color(properties["classification"])
                    elif properties.get("objectType", None) == "annotation":
                        _label = properties["name"]
                        _color = _get_geojson_color(properties)

                    _geometry = shape(x["geometry"], label=_label)
                    for _ in _geometry:
                        data[_label].append(_)

        # It is assumed that a specific label can only be one type (point or polygon)
        _annotations: list[AnnotationLayer] = [
            AnnotationLayer(a_cls=data[k][0].annotation_class, data=data[k]) for k in data.keys()
        ]

        return cls(_annotations, sorting=sorting)

    @classmethod
    def from_asap_xml(
        cls,
        asap_xml: PathLike,
        scaling: float | None = None,
        sorting: AnnotationSorting = AnnotationSorting.AREA,
    ) -> WsiAnnotations:
        """
        Read annotations as an ASAP [1] XML file. ASAP is a tool for viewing and annotating whole slide images.

        Parameters
        ----------
        asap_xml : PathLike
            Path to ASAP XML annotation file.
        scaling : float, optional
            Scaling factor. Sometimes required when ASAP annotations are stored in a different resolution than the
            original image.
        sorting: AnnotationSorting
            The sorting to apply to the annotations. Check the `AnnotationSorting` enum for more information.
            By default, the annotations are sorted by area.

        References
        ----------
        .. [1] https://github.com/computationalpathologygroup/ASAP

        Returns
        -------
        WsiAnnotations
        """
        tree = ET.parse(asap_xml)
        opened_annotation = tree.getroot()
        annotations: dict[str, AnnotationLayer] = dict()
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
                        annotations[label] = AnnotationLayer(
                            a_cls=_cls,
                            data=[coordinates],
                        )
                    else:
                        annotations[label].append(coordinates)

                    opened_annotations += 1

        return cls(list(annotations.values()), sorting=sorting)

    @classmethod
    def from_halo_xml(
        cls, halo_xml: PathLike, scaling: float | None = None, sorting: AnnotationSorting = AnnotationSorting.NONE
    ) -> WsiAnnotations:
        """
        Read annotations as a Halo [1] XML file.
        This function requires `pyhaloxml` [2] to be installed.

        Parameters
        ----------
        halo_xml : PathLike
            Path to the Halo XML file.
        scaling : float, optional
            The scaling to apply to the annotations.
        sorting: AnnotationSorting
            The sorting to apply to the annotations. Check the `AnnotationSorting` enum for more information. By default
            the annotations are not sorted as HALO supports hierarchical annotations.

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
        import pyhaloxml.shapely

        output = defaultdict(list)
        with pyhaloxml.HaloXMLFile(halo_xml) as hx:
            hx.matchnegative()
            for layer in hx.layers:
                for region in layer.regions:
                    curr_polygon = pyhaloxml.shapely.region_to_shapely(region)
                    if region.type == pyhaloxml.RegionType.Rectangle:
                        _cls = AnnotationClass(label=layer.name, a_cls=AnnotationType.BOX)
                    if region.type in [pyhaloxml.RegionType.Ellipse, pyhaloxml.RegionType.Polygon]:
                        _cls = AnnotationClass(label=layer.name, a_cls=AnnotationType.POLYGON)
                    if region.type == pyhaloxml.RegionType.Pin:
                        _cls = AnnotationClass(label=layer.name, a_cls=AnnotationType.POINT)
                    else:
                        raise NotImplementedError(f"Regiontype {region.type} is not implemented in DLUP")
                    output[layer.name].append(Polygon(curr_polygon, a_cls=_cls))

        annotations: list[AnnotationLayer] = []
        for label in output:
            annotations.append(
                AnnotationLayer(
                    a_cls=AnnotationClass(label=label, a_cls=AnnotationType.POLYGON),
                    data=output[label],
                )
            )

        return cls(annotations, sorting=sorting, offset_to_slide_bounds=True)

    @classmethod
    def from_darwin_json(
        cls, darwin_json: PathLike, sorting: AnnotationSorting | str = AnnotationSorting.NONE, z_indices: Optional[dict[str, int]] = None) -> WsiAnnotations:
        """
        Read annotations as a V7 Darwin [1] JSON file.

        Parameters
        ----------
        darwin_json : PathLike
            Path to the Darwin JSON file.
        sorting: AnnotationSorting
            The sorting to apply to the annotations. Check the `AnnotationSorting` enum for more information.
            By default, the annotations are sorted by the z-index which is generated by the order of the saved
            annotations.
        z_indices: dict[str, int], optional
            If set, these z_indices will be used rather than the default order.

        References
        ----------
        .. [1] https://darwin.v7labs.com/

        Returns
        -------
        WsiAnnotations

        """
        if not DARWIN_SDK_AVAILABLE:
            raise RuntimeError("`darwin` is not available. Install using `python -m pip install darwin-py`.")
        import darwin

        all_annotations = defaultdict(list)

        darwin_json_fn = pathlib.Path(darwin_json)
        darwin_an = darwin.utils.parse_darwin_json(darwin_json_fn, None)

        # Let's see if we actually have a metadata file, this can be useful to get the color
        v7_metadata = _get_v7_metadata(darwin_json_fn.parent)

        tags = []
        for curr_annotation in darwin_an.annotations:
            name = curr_annotation.annotation_class.name

            annotation_type = _v7_annotation_type_to_dlup_annotation_type(
                curr_annotation.annotation_class.annotation_type
            )
            annotation_color = v7_metadata[name].color if v7_metadata else None

            if annotation_type == AnnotationType.TAG:
                tags.append(AnnotationClass(label=name, a_cls=AnnotationType.TAG, color=annotation_color, z_index=None))
                continue

            z_index = None if annotation_type == AnnotationType.POINT or z_indices is None else z_indices[name]

            curr_data = curr_annotation.data

            _cls = AnnotationClass(label=name, a_cls=annotation_type, color=annotation_color, z_index=z_index)
            if annotation_type == AnnotationType.POINT:
                curr_point = Point((curr_data["x"], curr_data["y"]), a_cls=_cls)
                all_annotations[_cls].append(curr_point)
            elif annotation_type == AnnotationType.POLYGON:
                if "path" in curr_data:  # This is a regular polygon
                    curr_polygon = Polygon([(_["x"], _["y"]) for _ in curr_data["path"]], a_cls=_cls)
                    all_annotations[_cls].append(Polygon(curr_polygon, a_cls=_cls))
                elif "paths" in curr_data:  # This is a complex polygon which needs to be parsed with the even-odd rule
                    curr_complex_polygon = _parse_darwin_complex_polygon(curr_data)
                    for curr_polygon in curr_complex_polygon.geoms:
                        all_annotations[_cls].append(Polygon(curr_polygon, a_cls=_cls))
                else:
                    raise ValueError(f"Got unexpected data keys: {curr_data.keys()}")

            elif annotation_type == AnnotationType.BOX:
                x, y, w, h = curr_data.values()
                curr_polygon = shapely.geometry.box(x, y, x + w, y + h)
                all_annotations[_cls].append(Polygon(curr_polygon, a_cls=_cls))
            else:
                ValueError(f"Annotation type {annotation_type} is not supported.")

        # Now we can make AnnotationLayer annotations
        output = []

        for an_cls, _annotation in all_annotations.items():
            output.append(AnnotationLayer(a_cls=an_cls, data=_annotation))
        return cls(layers=output, tags=tags, sorting=sorting)

    def __getitem__(self, a_cls: AnnotationClass) -> AnnotationLayer:
        return self._layers[a_cls]

    def as_geojson(self) -> GeoJsonDict:
        """
        Output the annotations as proper geojson. These outputs are sorted according to the `AnnotationSorting` selected
        for the annotations. This ensures the annotations are correctly sorted in the output.

        Returns
        -------
        list of (str, GeoJsonDict)
        """
        coordinates, size = self.bounding_box
        region_size = (coordinates[0] + size[0], coordinates[1] + size[1])
        all_annotations = self.read_region((0, 0), 1.0, region_size)

        # We should group annotations that belong to the same class
        grouped_annotations = []
        previous_label = None
        group = []
        for annotation in all_annotations:
            label = annotation.label
            if not previous_label:
                previous_label = label

            if previous_label == label:
                group.append(annotation)
            else:
                grouped_annotations.append(group)
                group = [annotation]
                previous_label = label
        # After the loop, add the last group if it's not empty
        if group:
            grouped_annotations.append(group)

        data: GeoJsonDict = {"type": "FeatureCollection", "features": [], "id": None}
        for idx, annotation_list in enumerate(grouped_annotations):
            label = annotation_list[0].label
            color = annotation_list[0].color
            if len(annotation_list) == 1:
                json_dict = _geometry_to_geojson(annotation_list[0], label=label, color=color)
            else:
                if annotation_list[0].type in [AnnotationType.BOX, AnnotationType.POLYGON]:
                    annotation = shapely.geometry.MultiPolygon(annotation_list)
                else:
                    annotation = shapely.geometry.MultiPoint(annotation_list)
                json_dict = _geometry_to_geojson(annotation, label=label, color=color)

            json_dict["id"] = str(idx)
            data["features"].append(json_dict)

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
        for k in self._layers:
            self._layers[k].simplify(tolerance, preserve_topology=preserve_topology)

    def read_region(
        self,
        location: npt.NDArray[np.int_ | np.float_] | tuple[GenericNumber, GenericNumber],
        scaling: float,
        size: npt.NDArray[np.int_ | np.float_] | tuple[GenericNumber, GenericNumber],
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
        location: np.ndarray or tuple
        size : np.ndarray or tuple
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
        >>> polygons: list[Polygons] = annotations.read_region(location=(0,0), size=wsi.size, scaling=0.01)

        The polygons can be converted to masks using `dlup.data.transforms.convert_annotations` or
        `dlup.data.transforms.ConvertAnnotationsToMask`.
        """
        box = list(location) + list(np.asarray(location) + np.asarray(size))
        box = (np.asarray(box) / scaling).tolist()
        query_box = geometry.box(*box)

        filtered_annotations = []
        for k in self.available_labels:
            curr_indices = self._annotation_trees[k].query(query_box)
            curr_annotations = self._annotation_trees[k].geometries[curr_indices]
            for v in curr_annotations:
                filtered_annotations.append((k, v))

        if self._sorting == AnnotationSorting.AREA:
            # Sort on name
            filtered_annotations = sorted(filtered_annotations, key=lambda x: x[0].label)
            # Sort on area (largest to smallest)
            filtered_annotations = sorted(filtered_annotations, key=lambda x: x[1].area, reverse=True)
        elif self._sorting == AnnotationSorting.REVERSE:
            filtered_annotations = list(reversed(filtered_annotations))
        elif self._sorting == AnnotationSorting.Z_INDEX:
            filtered_annotations = sorted(filtered_annotations, key=lambda x: x[0].z_index)
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
            -location[0],
            -location[1],
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

        if self[annotation_class].type == AnnotationType.POLYGON:
            return Polygon(annotation, a_cls=annotation_class)

        if self[annotation_class].type == AnnotationType.BOX:
            return Polygon(annotation, a_cls=annotation_class)

        raise AnnotationError(f"Unexpected type. Got {self[annotation_class].type}.")

    def __contains__(self, item: Union[str, AnnotationClass]) -> bool:
        if isinstance(item, str):
            return item in [_.label for _ in self.available_labels]
        return item in self.available_labels

    def __add__(self, other: WsiAnnotations) -> WsiAnnotations:
        if set(self.available_labels).intersection(other.available_labels) != set():
            raise AnnotationError(
                "Can only add annotations with different labels. "
                "Use `.relabel` or relabel during construction of the object."
            )

        curr_annotations = list(self._layers.values())
        curr_annotations += list(other._layers.values())
        return WsiAnnotations(curr_annotations)

    def __str__(self) -> str:
        # Create a string for the labels
        output = ""
        for annotation_name in self._layers:
            output += f"{annotation_name} ({len(self._layers[annotation_name])}, "

        return f"{type(self).__name__}(labels={output[:-2]})"


class _ComplexDarwinPolygonWrapper:
    """Wrapper class for a complex polygon (i.e. polygon with holes) from a Darwin annotation."""

    def __init__(self, polygon: shapely.geometry.Polygon):
        self.geom = polygon
        self.hole = False
        self.holes: list[float] = []


def _parse_darwin_complex_polygon(annotation: dict[str, Any]) -> shapely.geometry.MultiPolygon:
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
        raise AnnotationError(f"Annotation type not supported. Got {annotation_type}.")

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

    if annotation_type in ["polygon", "complex_polygon"]:
        return AnnotationType.POLYGON

    if annotation_type == "keypoint":
        return AnnotationType.POINT

    if annotation_type == "tag":
        return AnnotationType.TAG

    raise NotImplementedError(f"annotation_type {annotation_type} is not implemented or not a valid dlup type.")

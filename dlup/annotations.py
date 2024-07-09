# Copyright (c) dlup contributors
"""
Annotation module for dlup.

There are three types of annotations, in the `AnnotationType` variable:
- points
- boxes (which are internally polygons)
- polygons

Supported file formats:
- ASAP XML
- Darwin V7 JSON
- GeoJSON
- HaloXML
"""
from __future__ import annotations

import copy
import errno
import functools
import json
import os
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, ClassVar, Iterable, NamedTuple, Optional, Type, TypedDict, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import shapely
import shapely.affinity
import shapely.geometry
import shapely.validation
from shapely import geometry
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.strtree import STRtree
from shapely.validation import make_valid

from dlup._exceptions import AnnotationError
from dlup.types import GenericNumber, PathLike
from dlup.utils.imports import DARWIN_SDK_AVAILABLE, PYHALOXML_AVAILABLE

# TODO:
# Group when exporting to GeoJSON
# Make GeoJSON work, use colors
# Verify ASAP


_TWsiAnnotations = TypeVar("_TWsiAnnotations", bound="WsiAnnotations")
ShapelyTypes = Union[ShapelyPoint, ShapelyMultiPolygon, ShapelyPolygon]


class DarwinV7Metadata(NamedTuple):
    label: str
    color: tuple[int, int, int]
    type: AnnotationType


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    if "#" not in hex_color:
        if hex_color == "black":
            return 0, 0, 0
    hex_color = hex_color.lstrip("#")

    # Convert the string from hex to an integer and extract each color component
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


class AnnotationType(str, Enum):
    POINT = "POINT"
    BOX = "BOX"
    POLYGON = "POLYGON"
    TAG = "TAG"


class AnnotationSorting(str, Enum):
    """The ways to sort the annotations. This is used in the constructors of the `WsiAnnotations` class, and applied
    to the output of `WsiAnnotations.read_region()`.

    - REVERSE: Sort the output in reverse order.
    - AREA: Often when the annotation tools do not properly support hierarchical order, one would annotate in a way
        that the smaller objects are on top of the larger objects. This option sorts the output by area, so that the
        larger objects appear first in the output and then the smaller objects.
    - Z_INDEX: Sort the output by the z-index of the annotations. This is useful when the annotations have a z-index
    - NONE: Do not apply any sorting and output as is presented in the input file.
    """

    REVERSE = "REVERSE"
    AREA = "AREA"
    Z_INDEX = "Z_INDEX"
    NONE = "NONE"


@dataclass(frozen=True)  # Frozen makes the class hashable
class AnnotationClass:
    """An annotation class. An annotation has two required properties:
    - label: The name of the annotation, e.g., "lymphocyte".
    - annotation_type: The type of annotation, e.g., AnnotationType.POINT.

    And two optional properties:
    - color: The color of the annotation as a tuple of RGB values.
    - z_index: The z-index of the annotation. This is useful when the annotations have a z-index.

    Parameters
    ----------
    label : str
        The name of the annotation.
    annotation_type : AnnotationType
        The type of annotation.
    color : Optional[tuple[int, int, int]]
        The color of the annotation as a tuple of RGB values.
    z_index : Optional[int]
        The z-index of the annotation.
    """

    label: str
    annotation_type: AnnotationType | str
    color: Optional[tuple[int, int, int]] = None
    z_index: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.annotation_type, str):
            if self.annotation_type in AnnotationType.__members__:
                object.__setattr__(self, "annotation_type", AnnotationType[self.annotation_type])
            else:
                raise ValueError(f"Unsupported annotation type {self.annotation_type}")

        if self.annotation_type in (AnnotationType.POINT, AnnotationType.TAG) and self.z_index is not None:
            raise ValueError("z_index is not supported for point annotations or tags.")


@functools.lru_cache(maxsize=None)
def _get_v7_metadata(filename: pathlib.Path) -> Optional[dict[str, DarwinV7Metadata]]:
    if not DARWIN_SDK_AVAILABLE:
        raise RuntimeError("`darwin` is not available. Install using `python -m pip install darwin-py`.")
    import darwin.path_utils

    if not filename.is_dir():
        raise RuntimeError("Provide the path to the root folder of the Darwin V7 annotations")

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

def _get_geojson_z_index(properties: dict[str, str | list[int]]) -> Optional[int]:
    """Parse the properties dictionary of a GeoJSON object to get the z_index`.

    Arguments
    ---------
    properties : dict
        The properties dictionary of a GeoJSON object.

    Returns
    -------
    Optional[tuple[int, int, int]]
        The color of the object as a tuple of RGB values.
    """
    z_index = properties.get("z_index", None)
    if z_index is None:
        return None

    return cast(int, z_index)


def _is_rectangle(polygon: Polygon | ShapelyPolygon) -> tuple[bool, bool]:
    if not polygon.is_valid or len(polygon.exterior.coords) != 5 or len(polygon.interiors) != 0:
        return False, False
    min_rotated_rect = polygon.minimum_rotated_rectangle
    aligned_rect = min_rotated_rect.minimum_rotated_rectangle
    return (np.isclose(polygon.area, min_rotated_rect.area), min_rotated_rect == aligned_rect)


class GeoJsonDict(TypedDict):
    """
    TypedDict for standard GeoJSON output
    """

    id: str | None
    type: str
    features: list[dict[str, str | dict[str, str]]]
    metadata: Optional[dict[str, str | list[str]]]


class AnnotatedGeometry(geometry.base.BaseGeometry):
    __slots__ = geometry.base.BaseGeometry.__slots__
    _a_cls: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Get annotation_class from args and kwargs. We do this because the __new__ method takes different (kw)args
        a_cls = next((arg for arg in args if isinstance(arg, AnnotationClass)), kwargs.get("a_cls", None))
        self._a_cls[id(self)] = a_cls

    @property
    def annotation_class(self) -> AnnotationClass:
        return cast(AnnotationClass, self._a_cls[id(self)])

    @property
    def annotation_type(self) -> AnnotationType | str:
        return self.annotation_class.annotation_type

    @property
    def label(self) -> str:
        return self.annotation_class.label

    @property
    def color(self) -> Optional[tuple[int, int, int]]:
        return self.annotation_class.color

    @property
    def z_index(self) -> Optional[int]:
        return self.annotation_class.z_index

    def __del__(self) -> None:
        if id(self) in self._a_cls:
            del self._a_cls[id(self)]

    def __str__(self) -> str:
        return f"{self.annotation_class}, {self.wkt}"

    def __eq__(self, other: object) -> bool:
        geometry_equal = self.equals(other)
        if not geometry_equal:
            return False

        # TODO: Check evaluation with ShapelyPolygon / ShapelyPoint. Does it evaluate to True?
        if not isinstance(other, type(self)):
            return False

        if not other.annotation_class == self.annotation_class:
            return False
        return True


class Point(ShapelyPoint, AnnotatedGeometry):
    __slots__ = ShapelyPoint.__slots__

    def __new__(
        cls, coord: ShapelyPoint | tuple[float, float], a_cls: AnnotationClass = None
    ) -> "Point":
        point = super().__new__(cls, coord)
        point.__class__ = cls
        return cast("Point", point)

    def __reduce__(self):  # type: ignore
        return (self.__class__, (ShapelyPoint(self.xy), self.annotation_class))


class Polygon(ShapelyPolygon, AnnotatedGeometry):
    __slots__ = ShapelyPolygon.__slots__

    def __new__(
        cls,
        shell: Union[tuple[float, float], ShapelyPolygon],
        holes: Optional[list[list[float, float]]] = None,
        a_cls: Optional[AnnotationClass] = None,
    ) -> "Polygon":
        instance = super().__new__(cls, shell, holes)
        instance.__class__ = cls
        return cast("Polygon", instance)

    def intersect_with_box(
        self, other: ShapelyPolygon, affine_transform_matrix: list[float]
    ) -> Optional[list["Polygon"]]:
        result = make_valid(self).intersection(other)
        if self.area > 0 and result.area == 0:
            return None

        # Verify if this box is still a box. Create annotation_type to polygon if that is not the case
        if self.annotation_type == AnnotationType.BOX and not _is_rectangle(result)[0]:
            annotation_class = replace(self.annotation_class, annotation_type=AnnotationType.POLYGON)
        else:
            annotation_class = self.annotation_class

        transformed_results = shapely.affinity.affine_transform(result, affine_transform_matrix)
        # FIXME: Can we even have MultiPolygons at this stage?
        if isinstance(transformed_results, ShapelyPolygon):
            return [Polygon(transformed_results, a_cls=annotation_class)]
        elif isinstance(transformed_results, (ShapelyMultiPolygon, shapely.geometry.collection.GeometryCollection)):
            return [Polygon(geom, a_cls=annotation_class) for geom in transformed_results.geoms if geom.area > 0]
        else:
            raise NotImplementedError(f"{type(result)}")

    def __reduce__(self):  # type: ignore
        return (
            self.__class__,
            (self.exterior.coords[:], [ring.coords[:] for ring in self.interiors], self.annotation_class),
        )

class CoordinatesDict(TypedDict):
    type: str
    coordinates: list[list[list[float]]]

def shape(
    coordinates: CoordinatesDict,
    label: str,
    color: Optional[tuple[int, int, int]] = None,
    z_index: Optional[int] = None,
) -> list[Polygon | Point]:
    geom_type = coordinates.get("type", "not_found").lower()
    if geom_type == "not_found":
        raise ValueError("No type found in coordinates.")
    elif geom_type in ["point", "multipoint"]:
        if z_index is not None:
            raise AnnotationError("z_index is not supported for point annotations.")

        annotation_class = AnnotationClass(
            label=label, annotation_type=AnnotationType.POINT, color=color, z_index=None
        )
        _coordinates = coordinates["coordinates"]
        return [
            Point(np.asarray(c), a_cls=annotation_class)
            for c in (_coordinates if geom_type == "multipoint" else [_coordinates])
        ]

    if geom_type == "polygon":
        _coordinates = coordinates["coordinates"]
        annotation_type = AnnotationType.BOX if _is_rectangle(Polygon(_coordinates[0]))[0] else AnnotationType.POLYGON
        annotation_class = AnnotationClass(label=label, annotation_type=annotation_type, color=color, z_index=z_index)
        return [Polygon(shell=np.asarray(_coordinates[0]), holes=[np.asarray(hole) for hole in _coordinates[1:]], a_cls=annotation_class)]
    if geom_type == "multipolygon":
        annotation_class = AnnotationClass(
            label=label, annotation_type=AnnotationType.POLYGON, color=color, z_index=z_index
        )
        # the first element is the outer polygon, the rest are holes.
        # TODO: This needs to work by directly constructing it with an a_cls
        multi_polygon = ShapelyMultiPolygon(
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


def _geometry_to_geojson(
    geometry: Polygon | Point, label: str, color: tuple[int, int, int] | None, z_index: int | None
) -> dict[str, Any]:
    """Function to convert a geometry to a GeoJSON object.

    Parameters
    ----------
    geometry : Polygon | Point
        A polygon or point object
    label : str
        The label name
    color : tuple[int, int, int]
        The color of the object in RGB values
    z_index : int
        The z-index of the object

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
            },
        },
        "geometry": shapely.geometry.mapping(geometry),
    }
    if color is not None:
        data["properties"]["classification"]["color"] = color

    if z_index is not None:
        data["properties"]["classification"]["z_index"] = z_index

    return data


def _sort_layers_in_place(layers: list[Polygon | Point], sorting: AnnotationSorting | str) -> None:
    """
    Sorts a list of layers. Check AnnotationSorting for more information of the sorting types.


    Parameters
    ----------
    layers : list[Polygon | Point]
        All annotations belonging to a single image
    sorting : AnnotationSorting
        How the layers should be sorted

    Returns
    -------
    None
    """
    if sorting == AnnotationSorting.Z_INDEX:
        layers.sort(key=lambda x: (x.z_index is None, x.z_index))
    elif sorting == AnnotationSorting.REVERSE:
        layers.reverse()
    elif sorting == AnnotationSorting.AREA:
        layers.sort(key=lambda x: x.area, reverse=True)
    # the other case is NONE, and nothing is done in that case


class WsiAnnotations:
    """Class that holds all annotations for a specific image"""

    def __init__(
        self,
        layers: list[Point | Polygon],
        tags: Optional[list[AnnotationClass]] = None,
        offset_to_slide_bounds: bool = False,
    ):
        """
        Parameters
        ----------
        layers : list[Point | Polygon]
            A list of layers for a single label.
        tags: Optional[list[AnnotationClass]]
            A list of tags for the annotations. These have to be of type `AnnotationType.TAG`.
        offset_to_slide_bounds : bool
            If true, will set the property `offset_to_slide_bounds` to True. This means that the annotations need
            to be offset to the slide bounds. This is useful when the annotations are read from a file format which
            requires this, for instance HaloXML.
        """

        self._offset_to_slide_bounds = offset_to_slide_bounds
        self._available_classes: list[AnnotationClass] = []
        for layer in layers:
            if layer.annotation_class in self._available_classes:
                continue
            self._available_classes.append(layer.annotation_class)

        self._layers = layers
        self._str_tree = STRtree(self._layers)
        self._tags = tags

    @property
    def available_classes(self) -> list[AnnotationClass]:
        return self._available_classes

    @property
    def tags(self) -> Optional[list[AnnotationClass]]:
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
        self._available_classes = []
        self._layers = []
        _labels = [labels] if isinstance(labels, str) else labels
        for layer in self._layers:
            if layer.label in _labels:
                self._available_classes += [layer.annotation_class]
                self._layers += [layer]

        self._str_tree = STRtree(self._layers)

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Return the bounding box of all annotations.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            Bounding box of the form ((x, y), (w, h)).
        """
        if not self._layers:
            return ((0.0, 0.0), (0.0, 0.0))

        # Extract the bounds for each annotation
        bounds = np.array(
            [
                (
                    annotation.bounds
                    if isinstance(annotation, Polygon)
                    else (annotation.x, annotation.y, annotation.x, annotation.y)
                )
                for annotation in self._layers
            ]
        )

        # Calculate the min and max coordinates
        min_coords = bounds[:, [0, 1]].min(axis=0)
        max_coords = bounds[:, [2, 3]].max(axis=0)

        # Calculate width and height
        width, height = max_coords - min_coords

        return (tuple(min_coords), (width, height))

    def copy(self) -> WsiAnnotations:
        """Make a copy of the object."""
        return copy.deepcopy(self)

    @classmethod
    def from_geojson(
        cls: Type[_TWsiAnnotations],
        geojsons: PathLike | Iterable[PathLike],
        sorting: AnnotationSorting | str = AnnotationSorting.NONE,
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
        if isinstance(geojsons, str):
            _geojsons: Iterable[Any] = [pathlib.Path(geojsons)]

        _geojsons = [geojsons] if not isinstance(geojsons, (tuple, list)) else geojsons
        layers: list[Polygon | Point] = []
        tags = None
        for path in _geojsons:
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as annotation_file:
                geojson_dict = json.load(annotation_file)
                if "metadata" in geojson_dict:
                    if geojson_dict["metadata"] and geojson_dict["metadata"].get("tags", None) is not None:
                        _tags = geojson_dict["metadata"]["tags"]
                        tags = [
                            AnnotationClass(label=tag, annotation_type=AnnotationType.TAG, color=None, z_index=None)
                            for tag in _tags
                        ]
                features = geojson_dict["features"]
                for x in features:
                    properties = x["properties"]
                    if "classification" in properties:
                        _label = properties["classification"]["name"]
                        _color = _get_geojson_color(properties["classification"])
                        _z_index = _get_geojson_z_index(properties["classification"])
                    elif properties.get("objectType", None) == "annotation":
                        _label = properties["name"]
                        _color = _get_geojson_color(properties)
                        _z_index = _get_geojson_z_index(properties)
                    else:
                        raise ValueError("Could not find label in the GeoJSON properties.")

                    _geometry = shape(x["geometry"], label=_label, color=_color, z_index=_z_index)
                    layers += _geometry

        _sort_layers_in_place(layers, sorting)
        return cls(layers=layers, tags=tags)

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
        layers: list[Polygon | Point] = []
        opened_annotations = 0
        for parent in opened_annotation:
            for child in parent:
                if child.tag != "Annotation":
                    continue
                label = child.attrib.get("PartOfGroup").strip()  # type: ignore
                color = _hex_to_rgb(child.attrib.get("Color").strip())  # type: ignore

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

                # Sometimes we have two adjacent polygons which can be split
                if isinstance(coordinates, ShapelyMultiPolygon):
                    coordinates_list = coordinates.geoms
                else:
                    # Explicitly turn into a list
                    coordinates_list = [coordinates]

                for coordinates in coordinates_list:
                    _cls = AnnotationClass(label=label, annotation_type=annotation_type, color=color)
                    if isinstance(coordinates, ShapelyPoint):
                        layers.append(Point(coordinates, a_cls=_cls))
                    elif isinstance(coordinates, ShapelyPolygon):
                        layers.append(Polygon(coordinates, a_cls=_cls))
                    else:
                        raise NotImplementedError

                    opened_annotations += 1

        _sort_layers_in_place(layers, sorting)
        return cls(layers=layers)

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

        output_layers = []
        with pyhaloxml.HaloXMLFile(halo_xml) as hx:
            hx.matchnegative()
            for layer in hx.layers:
                for region in layer.regions:
                    curr_geometry = pyhaloxml.shapely.region_to_shapely(region)
                    if region.type == pyhaloxml.RegionType.Rectangle:
                        _cls = AnnotationClass(label=layer.name, annotation_type=AnnotationType.BOX)
                        output_layers.append(Polygon(curr_geometry, a_cls=_cls))
                    if region.type in [pyhaloxml.RegionType.Ellipse, pyhaloxml.RegionType.Polygon]:
                        _cls = AnnotationClass(label=layer.name, annotation_type=AnnotationType.POLYGON)
                        output_layers.append(Polygon(curr_geometry, a_cls=_cls))
                    if region.type == pyhaloxml.RegionType.Pin:
                        _cls = AnnotationClass(label=layer.name, annotation_type=AnnotationType.POINT)
                        output_layers.append(Point(curr_geometry, a_cls=_cls))
                    else:
                        raise NotImplementedError(f"Regiontype {region.type} is not implemented in DLUP")

        _sort_layers_in_place(output_layers, sorting)
        return cls(output_layers, offset_to_slide_bounds=True)

    @classmethod
    def from_darwin_json(
        cls,
        darwin_json: PathLike,
        sorting: AnnotationSorting | str = AnnotationSorting.NONE,
        z_indices: Optional[dict[str, int]] = None,
    ) -> WsiAnnotations:
        """
        Read annotations as a V7 Darwin [1] JSON file. If available will read the `.v7/metadata.json` file to extract
        colors from the annotations.

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

        darwin_json_fn = pathlib.Path(darwin_json)
        darwin_an = darwin.utils.parse_darwin_json(darwin_json_fn, None)
        v7_metadata = _get_v7_metadata(darwin_json_fn.parent)

        tags = []
        layers = []
        for curr_annotation in darwin_an.annotations:
            name = curr_annotation.annotation_class.name

            annotation_type = _v7_annotation_type_to_dlup_annotation_type(
                curr_annotation.annotation_class.annotation_type
            )
            annotation_color = v7_metadata[name].color if v7_metadata else None

            if annotation_type == AnnotationType.TAG:
                tags.append(
                    AnnotationClass(
                        label=name, annotation_type=AnnotationType.TAG, color=annotation_color, z_index=None
                    )
                )
                continue

            z_index = None if annotation_type == AnnotationType.POINT or z_indices is None else z_indices[name]
            curr_data = curr_annotation.data

            _cls = AnnotationClass(
                label=name, annotation_type=annotation_type, color=annotation_color, z_index=z_index
            )
            if annotation_type == AnnotationType.POINT:
                curr_point = Point((curr_data["x"], curr_data["y"]), a_cls=_cls)
                layers.append(curr_point)
                continue

            elif annotation_type == AnnotationType.POLYGON:
                if "path" in curr_data:  # This is a regular polygon
                    curr_polygon = Polygon([(_["x"], _["y"]) for _ in curr_data["path"]])
                    layers.append(Polygon(curr_polygon, a_cls=_cls))

                elif "paths" in curr_data:  # This is a complex polygon which needs to be parsed with the even-odd rule
                    curr_complex_polygon = _parse_darwin_complex_polygon(curr_data)
                    for polygon in curr_complex_polygon.geoms:
                        layers.append(Polygon(polygon, a_cls=_cls))
                else:
                    raise ValueError(f"Got unexpected data keys: {curr_data.keys()}")

            elif annotation_type == AnnotationType.BOX:
                x, y, w, h = list(map(curr_data.get, ["x", "y", "w", "h"]))
                curr_polygon = shapely.geometry.box(x, y, x + w, y + h)
                layers.append(Polygon(curr_polygon, a_cls=_cls))
            else:
                ValueError(f"Annotation type {annotation_type} is not supported.")

        _sort_layers_in_place(layers, sorting)

        return cls(layers=layers, tags=tags)

    def __getitem__(self, idx: int) -> Point | Polygon:
        return self._layers[idx]

    def __iter__(self) -> Iterable[Point | Polygon]:
        for layer in self._layers:
            yield layer

    def as_geojson(self) -> GeoJsonDict:
        """
        Output the annotations as proper geojson. These outputs are sorted according to the `AnnotationSorting` selected
        for the annotations. This ensures the annotations are correctly sorted in the output.

        The output is not completely GeoJSON compliant as some parts such as the metadata and properties are not part
        of the standard. However, these are implemented to ensure the output is compatible with QuPath.

        Returns
        -------
        GeoJsonDict
            The output as a GeoJSON dictionary.
        """
        data: GeoJsonDict = {"type": "FeatureCollection", "metadata": None, "features": [], "id": None}
        if self.tags:
            data["metadata"] = {"tags": [_.label for _ in self.tags]}

        # # This used to be it.
        for idx, curr_annotation in enumerate(self._layers):
            json_dict = _geometry_to_geojson(
                curr_annotation,
                label=curr_annotation.label,
                color=curr_annotation.color,
                z_index=curr_annotation.z_index if isinstance(curr_annotation, Polygon) else None,
            )
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
        # TODO: Implement simplify on Polygon
        for idx, layer in enumerate(self._layers):
            a_cls = layer.annotation_class
            if a_cls.annotation_type == AnnotationType.POINT:
                continue
            layer.simplify(tolerance, preserve_topology=preserve_topology)
            self._layers[idx] = Polygon(self._layers[idx], a_cls=a_cls)

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

        curr_indices = self._str_tree.query(query_box)
        # This is needed because the STRTree returns (seemingly) arbitrary order, and this would destroy the order
        curr_indices.sort()
        filtered_annotations: list[Point | Polygon] = self._str_tree.geometries.take(curr_indices).tolist()

        cropped_annotations = []
        affine_transform_matrix = [scaling, 0, 0, scaling, -location[0], -location[1]]
        for annotation in filtered_annotations:
            if annotation.annotation_type in (AnnotationType.BOX, AnnotationType.POLYGON):
                _annotations = annotation.intersect_with_box(query_box, affine_transform_matrix=affine_transform_matrix)
                if _annotations is not None:
                    cropped_annotations += _annotations
            elif annotation.annotation_type == AnnotationType.POINT:
                _annotation = shapely.affinity.affine_transform(annotation, affine_transform_matrix)
                cropped_annotations.append(Point(_annotation, a_cls=annotation.annotation_class))
            # Tags could end up here in theory?
            else:
                cropped_annotations.append(annotation)
        return cropped_annotations

    def __add__(self, other: WsiAnnotations | Point | Polygon) -> WsiAnnotations:
        raise NotImplementedError

    def __iadd__(self, other: WsiAnnotations | Point | Polygon) -> WsiAnnotations:
        raise NotImplementedError

    def __radd__(self, other: WsiAnnotations) -> WsiAnnotations:
        raise NotImplementedError
    
    def __sub__(self, other: WsiAnnotations | Point | Polygon) -> WsiAnnotations:
        raise NotImplementedError

    def __isub__(self, other: WsiAnnotations | Point | Polygon) -> WsiAnnotations:
        raise NotImplementedError

    def __rsub__(self, other: WsiAnnotations) -> WsiAnnotations:
        raise NotImplementedError

    def __contains__(self, item: Union[str, AnnotationClass]) -> bool:
        if isinstance(item, str):
            return item in [_.label for _ in self.available_classes]
        return item in self.available_classes

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}(n_layers={len(self._layers)}, "
            f"tags={[tag.label for tag in self.tags] if self.tags else None})"
        )


class _ComplexDarwinPolygonWrapper:
    """Wrapper class for a complex polygon (i.e. polygon with holes) from a Darwin annotation."""

    def __init__(self, polygon: ShapelyPolygon):
        self.geom = polygon
        self.hole = False
        self.holes: list[float] = []


def _parse_darwin_complex_polygon(annotation: dict[str, Any]) -> ShapelyMultiPolygon:
    """
    Parse a complex polygon (i.e. polygon with holes) from a Darwin annotation.

    Parameters
    ----------
    annotation : dict

    Returns
    -------
    ShapelyMultiPolygon
    """
    polygons = [
        _ComplexDarwinPolygonWrapper(ShapelyPolygon([(p["x"], p["y"]) for p in path])) for path in annotation["paths"]
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
        ShapelyPolygon(my_polygon.geom.exterior.coords, my_polygon.holes)
        for my_polygon in sorted_polygons
        if not my_polygon.hole
    ]
    return ShapelyMultiPolygon(complex_polygon)


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
        coordinates = ShapelyPolygon(coordinates)
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

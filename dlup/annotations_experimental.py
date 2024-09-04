# Copyright (c) dlup contributors
"""
Experimental annotations module for dlup.

"""
from __future__ import annotations

import copy
import errno
import functools
import json
import os
import pathlib
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterable, NamedTuple, Optional, Type, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig
from xsdata.models.datatype import XmlDate

from dlup import SlideImage, __version__
from dlup._exceptions import AnnotationError
from dlup._geometry import AnnotationRegion  # pylint: disable=no-name-in-module
from dlup._types import GenericNumber, PathLike
from dlup.geometry import Box, GeometryCollection, Point, Polygon
from dlup.utils.annotations_utils import get_geojson_color, hex_to_rgb, rgb_to_hex
from dlup.utils.geometry_xml import (
    create_xml_geometries,
    create_xml_rois,
    parse_dlup_xml_polygon,
    parse_dlup_xml_roi_box,
)
from dlup.utils.imports import DARWIN_SDK_AVAILABLE, PYHALOXML_AVAILABLE
from dlup.utils.schemas.generated import DlupAnnotations as XMLDlupAnnotations
from dlup.utils.schemas.generated import Metadata as XMLMetadata
from dlup.utils.schemas.generated import Tag as XMLTag
from dlup.utils.schemas.generated import Tags as XMLTags

_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="SlideAnnotations")


class AnnotationType(str, Enum):
    POINT = "POINT"
    BOX = "BOX"
    POLYGON = "POLYGON"
    TAG = "TAG"
    RASTER = "RASTER"


class GeoJsonDict(TypedDict):
    """
    TypedDict for standard GeoJSON output
    """

    id: str | None
    type: str
    features: list[dict[str, str | dict[str, str]]]
    metadata: Optional[dict[str, str | list[str]]]


class CoordinatesDict(TypedDict):
    type: str
    coordinates: list[list[list[float]]]


class DarwinV7Metadata(NamedTuple):
    label: str
    color: Optional[tuple[int, int, int]]
    annotation_type: AnnotationType


@functools.lru_cache(maxsize=None)
def get_v7_metadata(filename: pathlib.Path) -> Optional[dict[tuple[str, str], DarwinV7Metadata]]:
    if not DARWIN_SDK_AVAILABLE:
        raise ImportError("`darwin` is not available. Install using `python -m pip install darwin-py`.")
    import darwin.path_utils

    if not filename.is_dir():
        raise ValueError("Provide the path to the root folder of the Darwin V7 annotations")

    v7_metadata_fn = filename / ".v7" / "metadata.json"
    if not v7_metadata_fn.exists():
        return None
    v7_metadata = darwin.path_utils.parse_metadata(v7_metadata_fn)
    output = {}
    for sample in v7_metadata["classes"]:
        annotation_type = sample["type"]
        # This is not implemented and can be skipped. The main function will raise a NonImplementedError
        if annotation_type == "raster_layer":
            continue

        label = sample["name"]
        color = sample["color"][5:-1].split(",")
        if color[-1] != "1.0":
            raise RuntimeError("Expected A-channel of color to be 1.0")
        rgb_colors = (int(color[0]), int(color[1]), int(color[2]))

        output[(label, annotation_type)] = DarwinV7Metadata(
            label=label, color=rgb_colors, annotation_type=annotation_type
        )
    return output


class AnnotationSorting(str, Enum):
    """The ways to sort the annotations. This is used in the constructors of the `SlideAnnotations` class, and applied
    to the output of `SlideAnnotations.read_region()`.

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

    def to_sorting_params(self) -> Any:
        """Get the sorting parameters for the annotation sorting."""
        if self == AnnotationSorting.REVERSE:
            return lambda x: None, True

        if self == AnnotationSorting.AREA:
            return lambda x: x.area, False

        if self == AnnotationSorting.Z_INDEX:
            return lambda x: x.get_field("z_index"), False


def _geometry_to_geojson(
    geometry: Polygon | Point, label: str | None, color: tuple[int, int, int] | None
) -> dict[str, Any]:
    """Function to convert a geometry to a GeoJSON object.

    Parameters
    ----------
    geometry : DlupPolygon | DlupPoint
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
    geojson: dict[str, Any] = {
        "type": "Feature",
        "properties": {
            "classification": {
                "name": label,
            },
        },
        "geometry": {},
    }

    if isinstance(geometry, Polygon):
        # Construct the coordinates for the polygon
        exterior = geometry.get_exterior()  # Get exterior coordinates
        interiors = geometry.get_interiors()  # Get interior coordinates (holes)

        # GeoJSON requires [ [x1, y1], [x2, y2], ... ] format
        geojson["geometry"] = {
            "type": "Polygon",
            "coordinates": [[list(coord) for coord in exterior]]  # Exterior ring
            + [[list(coord) for coord in interior] for interior in interiors],  # Interior rings (holes)
        }

    elif isinstance(geometry, Point):
        # Construct the coordinates for the point
        geojson["geometry"] = {
            "type": "Point",
            "coordinates": [geometry.x, geometry.y],
        }
    else:
        raise ValueError(f"Unsupported geometry type {type(geometry)}")

    if color is not None:
        classification: dict[str, Any] = geojson["properties"]["classification"]
        classification["color"] = color

    return geojson


def geojson_to_dlup(
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
        x, y = np.asarray(coordinates["coordinates"])
        return [Point(*(x, y), label=label, color=color)]

    if geom_type == "multipoint":
        return [Point(*np.asarray(c).tolist(), label=label, color=color) for c in coordinates["coordinates"]]

    if geom_type == "polygon":
        _coordinates = coordinates["coordinates"]
        polygon = Polygon(
            np.asarray(_coordinates[0]), [np.asarray(hole) for hole in _coordinates[1:]], label=label, color=color
        )
        return [polygon]
    if geom_type == "multipolygon":
        output: list[Polygon | Point] = []
        for polygon_coords in coordinates["coordinates"]:
            exterior = np.asarray(polygon_coords[0])
            interiors = [np.asarray(hole) for hole in polygon_coords[1:]]
            output.append(Polygon(exterior, interiors, label=label, color=color))
        return output

    raise AnnotationError(f"Unsupported geom_type {geom_type}")


class TagAttribute(NamedTuple):
    label: str
    color: Optional[tuple[int, int, int]]


class SlideTag(NamedTuple):
    attributes: Optional[list[TagAttribute]]
    label: str
    color: Optional[tuple[int, int, int]]


class SlideAnnotations:
    """Class that holds all annotations for a specific image"""

    def __init__(
        self,
        layers: GeometryCollection,
        tags: Optional[tuple[SlideTag, ...]] = None,
        sorting: Optional[AnnotationSorting | str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        layers : GeometryCollection
            Geometry collection containing the polygons, boxes and points
        tags: Optional[tuple[SlideTag, ...]]
            A tuple of image-level tags such as staining quality
        sorting: AnnotationSorting
            Sorting method, see `AnnotationSorting`. This value is typically passed to the constructor
            because of operations layer on (such as `__add__`). Typically the classmethod already sorts the data
        **kwargs: Any
            Additional keyword arguments. In this class they are used for additional metadata or offset functions.
            Currently only HaloXML requires offsets. See `.from_halo_xml` for an example
        """
        self._layers = layers
        self._tags = tags
        self._sorting = sorting
        self._offset_function: bool = bool(kwargs.get("offset_function", False))
        self._metadata: Optional[dict[str, list[str] | str | int | float | bool]] = kwargs.get("metadata", None)

    @property
    def sorting(self) -> Optional[AnnotationSorting | str]:
        return self._sorting

    @property
    def tags(self) -> Optional[tuple[SlideTag, ...]]:
        return self._tags

    @property
    def num_polygons(self) -> int:
        return len(self.layers.polygons)

    @property
    def num_points(self) -> int:
        return len(self.layers.points)

    @property
    def num_boxes(self) -> int:
        return len(self.layers.boxes)

    @property
    def metadata(self) -> Optional[dict[str, list[str] | str | int | float | bool]]:
        """Additional metadata for the annotations"""
        return self._metadata

    @property
    def offset_function(self) -> Any:
        """
        In some cases a function needs to be applied to the coordinates which cannot be handled in this class as
        it might require additional information. This function will be applied to the coordinates of all annotations.
        This is useful from a file format which requires this, for instance HaloXML.

        Example
        -------
        For HaloXML this is `offset = slide.slide_bounds[0] - slide.slide_bounds[0] % 256`.
        >>> slide = Slide.from_file_path("image.svs")
        >>> ann = SlideAnnotations.from_halo_xml("annotations.xml")
        >>> assert ann.offset_function == lambda slide: slide.slide_bounds[0] - slide.slide_bounds[0] % 256
        >>> ann.set_offset(annotation.offset_function(slide))

        Returns
        -------
        Callable

        """
        return self._offset_function

    @offset_function.setter
    def offset_function(self, func: Any) -> None:
        self._offset_function = func

    @property
    def layers(self) -> GeometryCollection:
        """
        Get the layers of the annotations.
        This is a GeometryCollection object which contains all the polygons and points
        """
        return self._layers

    @classmethod
    def from_geojson(
        cls: Type[_TSlideAnnotations],
        geojsons: PathLike,
        scaling: float | None = None,
        sorting: AnnotationSorting | str = AnnotationSorting.NONE,
        roi_names: Optional[list[str]] = None,
    ) -> _TSlideAnnotations:
        """
        Read annotations from a GeoJSON file.

        Parameters
        ----------
        geojsons : PathLike
            GeoJSON file. In the properties key there must be a name which is the label of this
            object.
        scaling : float, optional
            Scaling factor. Sometimes required when GeoJSON annotations are stored in a different resolution than the
            original image.
        sorting : AnnotationSorting
            The sorting to apply to the annotations. Check the `AnnotationSorting` enum for more information.
            By default, the annotations are sorted by area.
        roi_names : list[str], optional
            List of names that should be considered as regions of interest. If set, these will be added as ROIs rather
            than polygons.

        Returns
        -------
        SlideAnnotations
        """
        roi_names = [] if roi_names is None else roi_names
        collection = GeometryCollection()
        path = pathlib.Path(geojsons)
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

        with open(path, "r", encoding="utf-8") as annotation_file:
            geojson_dict = json.load(annotation_file)
            features = geojson_dict["features"]
            for x in features:
                properties = x["properties"]
                if "classification" in properties:
                    _label = properties["classification"]["name"]
                    _color = get_geojson_color(properties["classification"])
                elif properties.get("objectType", None) == "annotation":
                    _label = properties["name"]
                    _color = get_geojson_color(properties)
                else:
                    raise ValueError("Could not find label in the GeoJSON properties.")

                _geometries = geojson_to_dlup(x["geometry"], label=_label, color=_color)
                for geometry in _geometries:
                    if isinstance(geometry, Polygon):
                        if geometry.label in roi_names:
                            collection.add_roi(geometry)
                        else:
                            collection.add_polygon(geometry)
                    elif isinstance(geometry, Point):
                        collection.add_point(geometry)
                    else:
                        raise ValueError(f"Unsupported geometry type {geometry}")

        SlideAnnotations._in_place_sort_and_scale(collection, scaling, sorting)
        return cls(layers=collection)

    @classmethod
    def from_asap_xml(
        cls: Type[_TSlideAnnotations],
        asap_xml: PathLike,
        scaling: float | None = None,
        sorting: AnnotationSorting | str = AnnotationSorting.AREA,
        roi_names: Optional[list[str]] = None,
    ) -> _TSlideAnnotations:
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
        roi_names : list[str], optional
            List of names that should be considered as regions of interest. If set, these will be added as ROIs rather
            than polygons.

        References
        ----------
        .. [1] https://github.com/computationalpathologygroup/ASAP

        Returns
        -------
        SlideAnnotations
        """
        path = pathlib.Path(asap_xml)
        roi_names = [] if roi_names is None else roi_names
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

        tree = ET.parse(asap_xml)
        opened_annotation = tree.getroot()
        collection: GeometryCollection = GeometryCollection()
        for parent in opened_annotation:
            for child in parent:
                if child.tag != "Annotation":
                    continue
                label = child.attrib.get("PartOfGroup").strip()  # type: ignore
                color = hex_to_rgb(child.attrib.get("Color").strip())  # type: ignore

                annotation_type = child.attrib.get("Type").lower()  # type: ignore
                coordinates = _parse_asap_coordinates(child)

                if annotation_type == "pointset":
                    for point in coordinates:
                        collection.add_point(Point(point, label=label, color=color))

                elif annotation_type == "polygon":
                    if label in roi_names:
                        collection.add_roi(Polygon(coordinates, [], label=label, color=color))
                    else:
                        collection.add_polygon(Polygon(coordinates, [], label=label, color=color))

        SlideAnnotations._in_place_sort_and_scale(collection, scaling, sorting)
        return cls(layers=collection)

    @classmethod
    def from_darwin_json(
        cls: Type[_TSlideAnnotations],
        darwin_json: PathLike,
        scaling: float | None = None,
        sorting: AnnotationSorting | str = AnnotationSorting.NONE,
        z_indices: Optional[dict[str, int]] = None,
        roi_names: Optional[list[str]] = None,
    ) -> _TSlideAnnotations:
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
        scaling : float, optional
            Scaling factor. Sometimes required when Darwin annotations are stored in a different resolution
            than the original image.
        z_indices: dict[str, int], optional
            If set, these z_indices will be used rather than the default order.
        roi_names : list[str], optional
            List of names that should be considered as regions of interest. If set, these will be added as ROIs rather
            than polygons.

        References
        ----------
        .. [1] https://darwin.v7labs.com/

        Returns
        -------
        SlideAnnotations

        """
        if not DARWIN_SDK_AVAILABLE:
            raise RuntimeError("`darwin` is not available. Install using `python -m pip install darwin-py`.")
        import darwin

        roi_names = [] if roi_names is None else roi_names

        darwin_json_fn = pathlib.Path(darwin_json)
        if not darwin_json_fn.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(darwin_json_fn))

        darwin_an = darwin.utils.parse_darwin_json(darwin_json_fn, None)
        v7_metadata = get_v7_metadata(darwin_json_fn.parent)

        tags = []
        polygons: list[tuple[Polygon, int]] = []

        collection = GeometryCollection()
        for curr_annotation in darwin_an.annotations:
            name = curr_annotation.annotation_class.name
            annotation_type = curr_annotation.annotation_class.annotation_type
            if annotation_type == "raster_layer":
                raise NotImplementedError("Raster annotations are not supported.")

            annotation_color = v7_metadata[(name, annotation_type)].color if v7_metadata else None

            if annotation_type == "tag":
                attributes = []
                if curr_annotation.subs:
                    for subannotation in curr_annotation.subs:
                        if subannotation.annotation_type == "attributes":
                            attributes.append(TagAttribute(label=subannotation.data, color=None))

                tags.append(
                    SlideTag(
                        attributes=attributes if attributes != [] else None,
                        label=name,
                        color=annotation_color if annotation_color else None,
                    )
                )
                continue

            z_index = 0 if annotation_type == "keypoint" or z_indices is None else z_indices[name]
            curr_data = curr_annotation.data

            if annotation_type == "keypoint":
                x, y = curr_data["x"], curr_data["y"]
                curr_point = Point(curr_data["x"], curr_data["y"])
                curr_point.label = name
                curr_point.color = annotation_color
                collection.add_point(curr_point)

            elif annotation_type in ("polygon", "complex_polygon"):
                if "path" in curr_data:  # This is a regular polygon
                    curr_polygon = Polygon(
                        [(_["x"], _["y"]) for _ in curr_data["path"]], [], label=name, color=annotation_color
                    )
                    polygons.append((curr_polygon, z_index))

                elif "paths" in curr_data:  # This is a complex polygon which needs to be parsed with the even-odd rule
                    for curr_polygon in _parse_darwin_complex_polygon(curr_data, label=name, color=annotation_color):
                        polygons.append((curr_polygon, z_index))
                else:
                    raise ValueError(f"Got unexpected data keys: {curr_data.keys()}")
            elif annotation_type == "bounding_box":
                warnings.warn(
                    "Bounding box annotations are not fully supported and will be converted to Polygons.", UserWarning
                )
                x, y, w, h = curr_data["x"], curr_data["y"], curr_data["w"], curr_data["h"]
                curr_polygon = Polygon(
                    [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], [], label=name, color=annotation_color
                )
                polygons.append((curr_polygon, z_index))

            else:
                raise ValueError(f"Annotation type {annotation_type} is not supported.")

        if sorting == "Z_INDEX":
            for polygon, _ in sorted(polygons, key=lambda x: x[1]):
                if polygon.label in roi_names:
                    collection.add_roi(polygon)
                else:
                    collection.add_polygon(polygon)
        else:
            for polygon, _ in polygons:
                if polygon.label in roi_names:
                    collection.add_roi(polygon)
                else:
                    collection.add_polygon(polygon)

        SlideAnnotations._in_place_sort_and_scale(
            collection, scaling, sorting="NONE" if sorting == "Z_INDEX" else sorting
        )
        return cls(layers=collection, tags=tuple(tags), sorting=sorting)

    @classmethod
    def from_dlup_xml(cls: Type[_TSlideAnnotations], dlup_xml: PathLike) -> _TSlideAnnotations:
        """
        Read annotations as a DLUP XML file.

        Parameters
        ----------
        dlup_xml : PathLike
            Path to the DLUP XML file.

        Returns
        -------
        SlideAnnotations
        """
        path = pathlib.Path(dlup_xml)
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

        parser = XmlParser()
        with open(dlup_xml, "rb") as f:
            dlup_annotations = parser.from_bytes(f.read(), XMLDlupAnnotations)

        metadata = None if not dlup_annotations.metadata else asdict(dlup_annotations.metadata)
        tags: list[SlideTag] = []
        if dlup_annotations.tags:
            for tag in dlup_annotations.tags.tag:
                if not tag.label:
                    raise ValueError("Tag does not have a label.")
                curr_tag = SlideTag(attributes=[], label=tag.label, color=hex_to_rgb(tag.color) if tag.color else None)
                tags.append(curr_tag)

        collection = GeometryCollection()
        polygons: list[tuple[Polygon, int]] = []
        if not dlup_annotations.geometries:
            return cls(layers=collection, tags=tuple(tags))

        if dlup_annotations.geometries.polygon:
            polygons += parse_dlup_xml_polygon(dlup_annotations.geometries.polygon)

        # Complain if there are multipolygons
        if dlup_annotations.geometries.multi_polygon:
            for curr_polygons in dlup_annotations.geometries.multi_polygon:
                polygons += parse_dlup_xml_polygon(
                    curr_polygons.polygon,
                    order=curr_polygons.order,
                    label=curr_polygons.label,
                    index=curr_polygons.index,
                )

        # Now we sort the polygons on order
        for polygon, _ in sorted(polygons, key=lambda x: x[1]):
            collection.add_polygon(polygon)

        for curr_point in dlup_annotations.geometries.point:
            point = Point(
                curr_point.x,
                curr_point.y,
                label=curr_point.label,
                color=hex_to_rgb(curr_point.color) if curr_point.color else None,
            )
            collection.add_point(point)

        # Complain if there are multipoints
        if dlup_annotations.geometries.multi_point:
            raise NotImplementedError("Multipoints are not supported.")

        for curr_box in dlup_annotations.geometries.box:
            # mypy struggles
            assert isinstance(curr_box.x_min, float)
            assert isinstance(curr_box.y_min, float)
            assert isinstance(curr_box.x_max, float)
            assert isinstance(curr_box.y_max, float)
            box = Box(
                (curr_box.x_min, curr_box.y_min),
                (curr_box.x_max - curr_box.x_min, curr_box.y_max - curr_box.y_min),
                label=curr_box.label,
                color=hex_to_rgb(curr_box.color) if curr_box.color else None,
            )
            collection.add_box(box)

        rois: list[tuple[Polygon, int]] = []
        if dlup_annotations.regions_of_interest:
            for region_of_interest in dlup_annotations.regions_of_interest.multi_polygon:
                raise NotImplementedError(
                    "MultiPolygon regions of interest are not yet supported. "
                    "If you have a use case for this, "
                    "please open an issue at https://github.com/NKI-AI/dlup/issues."
                )

            if dlup_annotations.regions_of_interest.polygon:
                rois += parse_dlup_xml_polygon(dlup_annotations.regions_of_interest.polygon)

            if dlup_annotations.regions_of_interest.box:
                for _curr_box in dlup_annotations.regions_of_interest.box:
                    box, curr_order = parse_dlup_xml_roi_box(_curr_box)
                    rois.append((box.as_polygon(), curr_order))
            for roi, _ in sorted(rois, key=lambda x: x[1]):
                collection.add_roi(roi)

        return cls(layers=collection, tags=tuple(tags), metadata=metadata)

    @classmethod
    def from_halo_xml(
        cls: Type[_TSlideAnnotations],
        halo_xml: PathLike,
        scaling: float | None = None,
        sorting: AnnotationSorting | str = AnnotationSorting.NONE,
        box_as_polygon: bool = False,
        roi_names: Optional[list[str]] = None,
    ) -> _TSlideAnnotations:
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
        box_as_polygon : bool
            If True, rectangles are converted to polygons, and added as such.
            This is useful when the rectangles are actually implicitly bounding boxes.
        roi_names : list[str], optional
            List of names that should be considered as regions of interest. If set, these will be added as ROIs rather
            than polygons.

        References
        ----------
        .. [1] https://indicalab.com/halo/
        .. [2] https://github.com/rharkes/pyhaloxml

        Returns
        -------
        SlideAnnotations
        """
        path = pathlib.Path(halo_xml)
        roi_names = [] if roi_names is None else roi_names

        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

        if not PYHALOXML_AVAILABLE:
            raise RuntimeError("`pyhaloxml` is not available. Install using `python -m pip install pyhaloxml`.")
        import pyhaloxml.shapely

        collection = GeometryCollection()
        with pyhaloxml.HaloXMLFile(halo_xml) as hx:
            hx.matchnegative()
            for layer in hx.layers:
                _color = layer.linecolor.rgb
                color = (_color[0], _color[1], _color[2])
                for region in layer.regions:
                    if region.type == pyhaloxml.RegionType.Rectangle:
                        # The data is a CCW polygon, so the first and one to last coordinates are the coordinates
                        vertices = region.getvertices()
                        min_x = min(v[0] for v in vertices)
                        max_x = max(v[0] for v in vertices)
                        min_y = min(v[1] for v in vertices)
                        max_y = max(v[1] for v in vertices)
                        curr_box = Box((min_x, min_y), (max_x - min_x, max_y - min_y))

                        if box_as_polygon:
                            polygon = curr_box.as_polygon()
                            if polygon.label in roi_names:
                                collection.add_roi(polygon)
                            else:
                                collection.add_polygon(polygon)
                        else:
                            collection.add_box(curr_box)
                        continue

                    elif region.type in [pyhaloxml.RegionType.Ellipse, pyhaloxml.RegionType.Polygon]:
                        polygon = Polygon(
                            region.getvertices(), [x.getvertices() for x in region.holes], label=layer.name, color=color
                        )
                        if polygon.label in roi_names:
                            collection.add_roi(polygon)
                        else:
                            collection.add_polygon(polygon)
                    elif region.type == pyhaloxml.RegionType.Pin:
                        point = Point(*(region.getvertices()[0]), label=layer.name, color=color)
                        collection.add_point(point)
                    elif region.type == pyhaloxml.RegionType.Ruler:
                        warnings.warn(
                            f"Ruler annotations are not supported. Annotation {layer.name} will be skipped",
                            UserWarning,
                        )
                        continue
                    else:
                        raise NotImplementedError(f"Regiontype {region.type} is not implemented in dlup")

        SlideAnnotations._in_place_sort_and_scale(collection, scaling, sorting)

        def offset_function(slide: "SlideImage") -> tuple[int, int]:
            return (
                slide.slide_bounds[0][0] - slide.slide_bounds[0][0] % 256,
                slide.slide_bounds[0][1] - slide.slide_bounds[0][1] % 256,
            )

        return cls(collection, tags=None, sorting=sorting, offset_function=offset_function)

    @staticmethod
    def _in_place_sort_and_scale(
        collection: GeometryCollection, scaling: Optional[float], sorting: Optional[AnnotationSorting | str]
    ) -> None:
        if sorting == "REVERSE":
            raise NotImplementedError("This doesn't work for now.")

        if scaling != 1.0 and scaling is not None:
            collection.scale(scaling)
        if sorting == AnnotationSorting.NONE or sorting is None:
            return
        if isinstance(sorting, str):
            key, reverse = AnnotationSorting[sorting].to_sorting_params()
        else:
            key, reverse = sorting.to_sorting_params()
        collection.sort_polygons(key, reverse)

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

        if self._layers.boxes:
            warnings.warn("Bounding boxes are not supported in GeoJSON and will be skipped.", UserWarning)

        all_layers = self.layers.polygons + self.layers.points
        for idx, curr_annotation in enumerate(all_layers):
            json_dict = _geometry_to_geojson(curr_annotation, label=curr_annotation.label, color=curr_annotation.color)
            json_dict["id"] = str(idx)
            data["features"].append(json_dict)

        return data

    def as_dlup_xml(
        self,
        image_id: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        authors: Optional[list[str]] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """
        Output the annotations as DLUP XML.
        This format supports the complete serialization of a SlideAnnotations object.

        Parameters
        ----------
        image_id : str, optional
            The image ID corresponding to this annotation.
        description : str, optional
            Description of the annotations.
        version : str, optional
            Version of the annotations.
        authors : list[str], optional
            Authors of the annotations.
        indent : int, optional
            Indent for pretty printing.

        Returns
        -------
        str
            The output as a DLUP XML string.
        """

        metadata = XMLMetadata(
            image_id=image_id if image_id is not None else "",
            description=description if description is not None else "",
            version=version if version is not None else "",
            authors=XMLMetadata.Authors(authors) if authors is not None else None,
            date_created=XmlDate.from_string(datetime.now().strftime("%Y-%m-%d")),
            software=f"dlup {__version__}",
        )
        xml_tags: list[XMLTag] = []
        if self.tags:
            for tag in self.tags:
                if tag.attributes:
                    attrs = [
                        XMLTag.Attribute(value=_.label, color=rgb_to_hex(*_.color) if _.color else None)
                        for _ in tag.attributes
                    ]
                xml_tag = XMLTag(
                    attribute=attrs if tag.attributes else [],
                    label=tag.label,
                    color=rgb_to_hex(*tag.color) if tag.color else None,
                )
                xml_tags.append(xml_tag)

        tags = XMLTags(tag=xml_tags) if xml_tags else None

        geometries = create_xml_geometries(self._layers)
        rois = create_xml_rois(self._layers)

        extra_annotation_params: dict[str, XMLTags] = {}
        if tags:
            extra_annotation_params["tags"] = tags

        dlup_annotations = XMLDlupAnnotations(
            metadata=metadata, geometries=geometries, regions_of_interest=rois, **extra_annotation_params
        )
        config = SerializerConfig(pretty_print=True)
        serializer = XmlSerializer(config=config)
        return serializer.render(dlup_annotations)

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get the bounding box of the annotations combining points and polygons.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            The bounding box of the annotations.

        """
        return self._layers.bounding_box

    def bounding_box_at_scaling(self, scaling: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get the bounding box of the annotations at a specific scaling factor.

        Parameters
        ----------
        scaling : float
            The scaling factor to apply to the annotations.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            The bounding box of the annotations at the specific scaling factor.

        """
        bbox = self.bounding_box
        return ((bbox[0][0] * scaling, bbox[0][1] * scaling), (bbox[1][0] * scaling, bbox[1][1] * scaling))

    def simplify(self, tolerance: float) -> None:
        """Simplify the polygons in the annotation (i.e. reduce points). Other annotations will remain unchanged.
        All points in the resulting polygons object will be in the tolerance distance of the original polygon.

        Parameters
        ----------
        tolerance : float
            The tolerance to simplify the polygons with.
        Returns
        -------
        None
        """
        self._layers.simplify(tolerance)

    def __contains__(self, item: str | Point | Polygon) -> bool:
        if isinstance(item, str):
            return item in self.available_classes
        if isinstance(item, Point):
            return item in self.layers.points
        if isinstance(item, Polygon):
            return item in self.layers.polygons

        return False

    def __len__(self) -> int:
        return self._layers.size()

    @property
    def available_classes(self) -> set[str]:
        """Get the available classes in the annotations.

        Returns
        -------
        set[str]
            The available classes in the annotations.

        """
        available_classes = set()
        for polygon in self.layers.polygons:
            if polygon.label is not None:
                available_classes.add(polygon.label)
        for point in self.layers.points:
            if point.label is not None:
                available_classes.add(point.label)

        return available_classes

    def __iter__(self) -> Iterable[Polygon | Point]:
        # First returns all the polygons then all points
        for polygon in self.layers.polygons:
            yield polygon

        for point in self.layers.points:
            yield point

    def __add__(self, other: Any) -> "SlideAnnotations":
        """
        Add two annotations together. This will return a new `SlideAnnotations` object with the annotations combined.
        The polygons will be added from left to right followed the points from left to right.

        Notes
        -----
        - The polygons and points are shared between the objects. This means that if you modify the polygons or points
          in the new object, the original objects will also be modified. If you wish to avoid this, you must add two
          copies together.
        - Note that the sorting is not applied to this object. You can apply this by calling `sort_polygons()` on
        the resulting object.

        Parameters
        ----------
        other : SlideAnnotations
            The other annotations to add.

        """
        if not isinstance(other, (SlideAnnotations, Point, Polygon, list)):
            raise TypeError(f"Unsupported type {type(other)}")

        if isinstance(other, SlideAnnotations):
            if not self.sorting == other.sorting:
                raise TypeError("Cannot add annotations with different sorting.")
            if self.offset_function != other.offset_function:
                raise TypeError(
                    "Cannot add annotations with different requirements for offsetting to slide bounds "
                    "(`offset_function`)."
                )

            tags: tuple[SlideTag, ...] = ()
            if self.tags is None and other.tags is not None:
                tags = other.tags

            if other.tags is None and self.tags is not None:
                tags = self.tags

            if self.tags is not None and other.tags is not None:
                tags = tuple(set(self.tags + other.tags))

            # Let's add the annotations
            collection = GeometryCollection()
            for polygon in self.layers.polygons:
                collection.add_polygon(copy.deepcopy(polygon))
            for point in self.layers.points:
                collection.add_point(copy.deepcopy(point))

            for polygon in other.layers.polygons:
                collection.add_polygon(copy.deepcopy(polygon))
            for point in other.layers.points:
                collection.add_point(copy.deepcopy(point))

            SlideAnnotations._in_place_sort_and_scale(collection, None, self.sorting)
            return self.__class__(layers=collection, tags=tuple(tags) if tags else None, sorting=self.sorting)

        if isinstance(other, (Point, Polygon)):
            other = [other]

        if isinstance(other, list):
            if not all(isinstance(item, (Point, Polygon)) for item in other):
                raise TypeError(
                    f"can only add list purely containing Point and Polygon objects to {self.__class__.__name__}"
                )

            collection = copy.copy(self._layers)
            for item in other:
                if isinstance(item, Polygon):
                    collection.add_polygon(item)
                elif isinstance(item, Point):
                    collection.add_point(item)
            SlideAnnotations._in_place_sort_and_scale(collection, None, self.sorting)
            return self.__class__(layers=collection, tags=copy.copy(self._tags), sorting=self.sorting)

        raise ValueError(f"Unsupported type {type(other)}")

    def __iadd__(self, other: Any) -> "SlideAnnotations":
        if isinstance(other, (Point, Polygon)):
            other = [other]

        if isinstance(other, list):
            if not all(isinstance(item, (Point, Polygon)) for item in other):
                raise TypeError(
                    f"can only add list purely containing Point and Polygon objects {self.__class__.__name__}"
                )

            for item in other:
                if isinstance(item, Polygon):
                    self._layers.add_polygon(copy.deepcopy(item))
                elif isinstance(item, Point):
                    self._layers.add_point(copy.deepcopy(item))

        elif isinstance(other, SlideAnnotations):
            if self.sorting != other.sorting or self.offset_function != other.offset_function:
                raise ValueError(
                    f"Both sorting and offset_function must be the same to " f"add {self.__class__.__name__}s together."
                )

            if self._tags is None:
                self._tags = other._tags
            elif other._tags is not None:
                assert self
                self._tags += other._tags

            for polygon in other.layers.polygons:
                self._layers.add_polygon(copy.deepcopy(polygon))
            for point in other.layers.points:
                self._layers.add_point(copy.deepcopy(point))
        else:
            return NotImplemented
        SlideAnnotations._in_place_sort_and_scale(self._layers, None, self.sorting)

        return self

    def __radd__(self, other: Any) -> "SlideAnnotations":
        # in-place addition (+=) of Point and Polygon will raise a TypeError
        if not isinstance(other, (SlideAnnotations, Point, Polygon, list)):
            raise TypeError(f"Unsupported type {type(other)}")
        if isinstance(other, list):
            if not all(isinstance(item, (Polygon, Point)) for item in other):
                raise TypeError(
                    f"can only add list purely containing Point and Polygon objects to {self.__class__.__name__}"
                )
            raise TypeError(
                "use the __add__ or __iadd__ operator instead of __radd__ when working with lists to avoid \
                            unexpected behavior."
            )
        return self + other

    def __sub__(self, other: Any) -> "SlideAnnotations":
        return NotImplemented

    def __isub__(self, other: Any) -> "SlideAnnotations":
        return NotImplemented

    def __rsub__(self, other: Any) -> "SlideAnnotations":
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SlideAnnotations):
            return False

        our_sorting = self._sorting if self._sorting != AnnotationSorting.NONE else None
        other_sorting = other._sorting if other._sorting != AnnotationSorting.NONE else None

        if our_sorting != other_sorting:
            return False

        if self._tags != other._tags:
            return False

        if self._layers != other._layers:
            return False

        if self._offset_function != other._offset_function:
            return False

        return True

    def read_region(
        self,
        coordinates: tuple[GenericNumber, GenericNumber],
        scaling: float,
        size: tuple[GenericNumber, GenericNumber],
    ) -> AnnotationRegion:
        """Reads the region of the annotations. Function signature is the same as `dlup.SlideImage`
        so they can be used in conjunction.

        The process is as follows:

        1.  All the annotations which overlap with the requested region of interest are filtered
        2.  The polygons in the GeometryContainer in `.layers` are cropped.
            The boxes and points are only filtered, so it's possible the boxes have negative (x, y) values
        3.  The annotation is rescaled and shifted to the origin to match the local patch coordinate system.

        The final returned data is a `dlup.geometry.AnnotationRegion`.

        Parameters
        ----------
        location: tuple[GenericNumber, GenericNumber]
            Top-left coordinates of the region in the requested scaling
        size : tuple[GenericNumber, GenericNumber]
            Output size of the region
        scaling : float
            Scaling to apply compared to the base level

        Returns
        -------
        AnnotationRegion

        Examples
        --------
        1. To read geojson annotations and convert them into masks:

        >>> from pathlib import Path
        >>> from dlup import SlideImage
        >>> import numpy as np
        >>> wsi = SlideImage.from_file_path(Path("path/to/image.svs"))
        >>> wsi = wsi.get_scaled_view(scaling=0.5)
        >>> wsi = wsi.read_region(location=(0,0), size=wsi.size)
        >>> annotations = SlideAnnotations.from_geojson("path/to/geojson.json")
        >>> region = annotations.read_region((0,0), 0.01, wsi.size)
        >>> mask = region.to_mask()
        >>> color_mask = annotations.color_lut[mask]
        >>> polygons = region.polygons.get_geometries()  # This is a list of `dlup.geometry.Polygon` objects
        """
        region = self._layers.read_region(coordinates, scaling, size)
        return region

    def scale(self, scaling: float) -> None:
        """
        Scale the annotations by a multiplication factor.
        This operation will be performed in-place.

        Parameters
        ----------
        scaling : float
            The scaling factor to apply to the annotations.

        Notes
        -----
        This invalidates the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or have the function
        `read_region()` do it for you on-demand.

        Returns
        -------
        None
        """
        self._layers.scale(scaling)

    def set_offset(self, offset: tuple[float, float]) -> None:
        """Set the offset for the annotations. This operation will be performed in-place.

        For example, if the offset is 1, 1, the annotations will be moved by 1 unit in the x and y direction.

        Parameters
        ----------
        offset : tuple[float, float]
            The offset to apply to the annotations.

        Notes
        -----
        This invalidates the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or have the function
        `read_region()` do it for you on-demand.

        Returns
        -------
        None
        """
        self._layers.set_offset(offset)

    def rebuild_rtree(self) -> None:
        """
        Rebuild the R-tree for the annotations. This operation will be performed in-place.
        The R-tree is used for fast spatial queries on the annotations and is invalidated when the annotations are
        modified. This function will rebuild the R-tree. Strictly speaking, this is not required as the R-tree will be
        rebuilt on-demand when you invoke a `read_region()`. You could however do this if you want to avoid
        the `read_region()` to do it for you the first time it runs.
        """

        self._layers.rebuild_rtree()

    def reindex_polygons(self, index_map: dict[str, int]) -> None:
        """
        Reindex the polygons in the annotations. This operation will be performed in-place.
        This is useful if you want to change the index of the polygons in the annotations.

        This requires that the `.label` property on the polygons is set.

        Parameters
        ----------
        index_map : dict[str, int]
            A dictionary that maps the label to the new index.

        Returns
        -------
        None
        """
        self._layers.reindex_polygons(index_map)

    def relabel_polygons(self, relabel_map: dict[str, str]) -> None:
        """
        Relabel the polygons in the annotations. This operation will be performed in-place.

        Parameters
        ----------
        relabel_map : dict[str, str]
            A dictionary that maps the label to the new label.

        Returns
        -------
        None
        """
        # TODO: Implement in C++
        for polygon in self._layers.polygons:
            if not polygon.label:
                continue
            if polygon.label in relabel_map:
                polygon.label = relabel_map[polygon.label]

    def filter_polygons(self, label: str) -> None:
        """Filter polygons in-place.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Parameters
        ----------
        label : str
            The label to filter.

        """
        for polygon in self._layers.polygons:
            if polygon.label == label:
                self._layers.remove_polygon(polygon)

    def filter_points(self, label: str) -> None:
        """Filter points in-place.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Parameters
        ----------
        label : str
            The label to filter.

        """
        for point in self._layers.points:
            if point.label == label:
                self._layers.remove_point(point)

    def filter(self, label: str) -> None:
        """Filter annotations in-place.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Parameters
        ----------
        label : str
            The label to filter.

        """
        self.filter_polygons(label)
        self.filter_points(label)

    def sort_polygons(self, key: Callable[[Polygon], int | float | str], reverse: bool = False) -> None:
        """Sort the polygons in-place.

        Parameters
        ----------
        key : callable
            The key to sort the polygons on, this has to be a lambda function or similar.
            For instance `lambda polygon: polygon.area` will sort the polygons on the area, or
            `lambda polygon: polygon.get_field(field_name)` will sort the polygons on that field.
        reverse : bool
            Whether to sort in reverse order.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Returns
        -------
        None

        """
        self._layers.sort_polygons(key, reverse)

    @property
    def color_lut(self) -> npt.NDArray[np.uint8]:
        """Get the color lookup table for the annotations.

        Requires that the polygons have an index and color set. Be aware that for the background always
        the value 0 is assumed.
        So if you are using the `to_mask(default_value=0)` with a default value other than 0,
        the LUT will still have this as index 0.

        Example
        -------
        >>> color_lut = annotations.color_lut
        >>> region = annotations.read_region(region_start, 0.02, region_size).to_mask()
        >>> colored_mask = PIL.Image.fromarray(color_lut[mask])

        Returns
        -------
        np.ndarray
            The color lookup table.

        """
        return self._layers.color_lut

    def __copy__(self) -> "SlideAnnotations":
        return self.__class__(layers=copy.copy(self._layers), tags=copy.copy(self._tags))

    def __deepcopy__(self, memo: dict[int, Any]) -> "SlideAnnotations":
        return self.__class__(layers=copy.deepcopy(self._layers, memo), tags=copy.deepcopy(self._tags, memo))

    def copy(self) -> "SlideAnnotations":
        return self.__copy__()


def _parse_asap_coordinates(
    annotation_structure: ET.Element,
) -> list[tuple[float, float]]:
    """
    Parse ASAP XML coordinates into list.

    Parameters
    ----------
    annotation_structure : list of strings

    Returns
    -------
    list[tuple[float, float]]

    """
    coordinates = []
    coordinate_structure = annotation_structure[0]

    for coordinate in coordinate_structure:
        coordinates.append(
            (
                float(coordinate.get("X").replace(",", ".")),  # type: ignore
                float(coordinate.get("Y").replace(",", ".")),  # type: ignore
            )
        )

    return coordinates


def _parse_darwin_complex_polygon(
    annotation: dict[str, Any], label: str, color: Optional[tuple[int, int, int]]
) -> Iterable[Polygon]:
    """
    Parse a complex polygon (i.e. polygon with holes) from a Darwin annotation.

    Parameters
    ----------
    annotation : dict
        The annotation dictionary
    label : str
        The label of the annotation
    color : tuple[int, int, int]
        The color of the annotation

    Returns
    -------
    Iterable[DlupPolygon]
    """
    # Create Polygons and sort by area in descending order
    polygons = [Polygon([(p["x"], p["y"]) for p in path], []) for path in annotation["paths"]]
    polygons.sort(key=lambda x: x.area, reverse=True)

    outer_polygons: list[tuple[Polygon, list[Any], bool]] = []
    for polygon in polygons:
        is_hole = False
        # Check if the polygon can be a hole in any of the previously processed polygons
        for outer_poly, holes, outer_poly_is_hole in reversed(outer_polygons):
            contains = outer_poly.contains(polygon)
            # If polygon is contained by a hole, it should be added as new polygon
            if contains and outer_poly_is_hole:
                break
            # Polygon is added as hole if outer polygon is not a hole
            elif contains:
                holes.append(polygon.get_exterior())
                is_hole = True
                break
        outer_polygons.append((polygon, [], is_hole))

    for outer_poly, holes, _is_hole in outer_polygons:
        if not _is_hole:
            polygon = Polygon(outer_poly.get_exterior(), holes)
            polygon.label = label
            polygon.color = color
            yield polygon

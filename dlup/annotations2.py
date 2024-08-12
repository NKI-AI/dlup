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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Iterable, NamedTuple, Optional, Type, TypedDict, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import shapely
import shapely.affinity
import shapely.geometry
import shapely.validation
from shapely import geometry
from shapely import lib as shapely_lib
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon

from dlup._exceptions import AnnotationError
from dlup._types import GenericNumber, PathLike
from dlup.annotations import (
    _ASAP_TYPES,
    AnnotationType,
    CoordinatesDict,
    GeoJsonDict,
    _geometry_to_geojson,
    _get_geojson_color,
)
from dlup.geometry import DlupGeometryContainer, DlupPoint, DlupPolygon
from dlup.utils.imports import DARWIN_SDK_AVAILABLE, PYHALOXML_AVAILABLE


class CoordinatesDict(TypedDict):
    type: str
    coordinates: list[list[list[float]]]


def shape(
    coordinates: CoordinatesDict,
    label: str,
    color: Optional[tuple[int, int, int]] = None,
    z_index: Optional[int] = None,
) -> list[DlupPolygon | DlupPoint]:
    geom_type = coordinates.get("type", None)
    if geom_type is None:
        raise ValueError("No type found in coordinates.")
    geom_type = geom_type.lower()

    if geom_type in ["point", "multipoint"] and z_index is not None:
        raise AnnotationError("z_index is not supported for point annotations.")

    if geom_type == "point":
        x, y = np.asarray(coordinates["coordinates"])
        return [DlupPoint((x, y), label=label, color=color)]

    if geom_type == "multipoint":
        return [DlupPoint(np.asarray(c), label=label, color=color) for c in coordinates["coordinates"]]

    if geom_type == "polygon":
        _coordinates = coordinates["coordinates"]
        polygon = DlupPolygon(
            np.asarray(_coordinates[0]), [np.asarray(hole) for hole in _coordinates[1:]], label=label, color=color
        )
        return [polygon]
    if geom_type == "multipolygon":
        multi_polygon = ShapelyMultiPolygon(
            [
                [
                    np.asarray(c[0]),
                    [np.asarray(hole) for hole in c[1:]],
                ]
                for c in coordinates["coordinates"]
            ]
        )

        output = []
        for polygon in multi_polygon.geoms:
            shell = polygon.exterior.coords
            holes = [hole.coords for hole in polygon.interiors]
            output.append(DlupPolygon(shell, holes, label=label, color=color))

    raise AnnotationError(f"Unsupported geom_type {geom_type}")


class WsiAnnotations2:
    """Class that holds all annotations for a specific image"""

    def __init__(self, layers):
        self._layers = layers

    @classmethod
    def from_geojson(
        cls: Type[_TWsiAnnotations],
        geojsons: PathLike | Iterable[PathLike],
    ) -> _TWsiAnnotations:

        if isinstance(geojsons, str):
            _geojsons: Iterable[Any] = [pathlib.Path(geojsons)]

        _geojsons = [geojsons] if not isinstance(geojsons, (tuple, list)) else geojsons
        layers: list[DlupPolygon | DlupPoint] = []
        for path in _geojsons:
            path = pathlib.Path(path)
            if not path.exists():
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

            with open(path, "r", encoding="utf-8") as annotation_file:
                geojson_dict = json.load(annotation_file)
                features = geojson_dict["features"]
                for x in features:
                    properties = x["properties"]
                    if "classification" in properties:
                        _label = properties["classification"]["name"]
                        _color = _get_geojson_color(properties["classification"])
                    elif properties.get("objectType", None) == "annotation":
                        _label = properties["name"]
                        _color = _get_geojson_color(properties)
                    else:
                        raise ValueError("Could not find label in the GeoJSON properties.")

                    _geometry = shape(x["geometry"], label=_label, color=_color)
                    layers += _geometry
        # We need to add the layers to the GeometryContainer
        container = DlupGeometryContainer()
        print("Number of layers in container 2: ", len(layers))
        for layer in layers:
            if isinstance(layer, DlupPolygon):
                container.add_polygon(layer)
            elif isinstance(layer, DlupPoint):
                container.add_point(layer)
            else:
                raise ValueError(f"Unsupported layer type {type(layer)}")

        return cls(layers=container)

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
        layers: list[DlupPolygon | DlupPoint] = []
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
                        layers.append(DlupPoint(coordinates, a_cls=_cls))
                    elif isinstance(coordinates, ShapelyPolygon):
                        layers.append(DlupPolygon(coordinates, a_cls=_cls))
                    else:
                        raise NotImplementedError

                    opened_annotations += 1

        return cls(layers=layers)

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
            json_dict = _geometry_to_geojson(curr_annotation, label=curr_annotation.label, color=curr_annotation.color)
            json_dict["id"] = str(idx)
            data["features"].append(json_dict)

        return data

    def read_region(self, coordinates, scaling, size):
        return self._layers.read_region(coordinates, scaling, size)


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

    if annotation_type == "raster_layer":
        return AnnotationType.RASTER

    raise NotImplementedError(f"annotation_type {annotation_type} is not implemented or not a valid dlup type.")

# Copyright (c) dlup contributors
"""
Experimental annotations module for dlup.

"""
from __future__ import annotations
import cv2

import time
import errno
import json
import os
import pathlib
from typing import Any, Iterable, Optional, Type, TypedDict

import numpy as np
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon

from dlup._exceptions import AnnotationError
from dlup._types import PathLike
from dlup.annotations import (
    CoordinatesDict,
    GeoJsonDict,
    _geometry_to_geojson,
    _get_geojson_color,
)
from dlup.geometry import DlupGeometryContainer, DlupPoint, DlupPolygon


class CoordinatesDict(TypedDict):
    type: str
    coordinates: list[list[list[float]]]


def _geometry_to_geojson(geometry: Polygon | Point, label: str, color: tuple[int, int, int] | None) -> dict[str, Any]:
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
    geojson = {
        "type": "Feature",
        "properties": {
            "classification": {
                "name": label,
            },
        },
        "geometry": {},
    }

    if isinstance(geometry, DlupPolygon):
        # Construct the coordinates for the polygon
        exterior = geometry.get_exterior()  # Get exterior coordinates
        interiors = geometry.get_interiors()  # Get interior coordinates (holes)

        # GeoJSON requires [ [x1, y1], [x2, y2], ... ] format
        geojson["geometry"] = {
            "type": "Polygon",
            "coordinates": [[list(coord) for coord in exterior]]  # Exterior ring
            + [[list(coord) for coord in interior] for interior in interiors],  # Interior rings (holes)
        }

    elif isinstance(geometry, DlupPoint):
        # Construct the coordinates for the point
        geojson["geometry"] = {
            "type": "Point",
            "coordinates": [geometry.x, geometry.y],
        }

    if color is not None:
        geojson["properties"]["classification"]["color"] = color

    return geojson


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


class WsiAnnotationsExperimental:
    """Class that holds all annotations for a specific image"""

    def __init__(self, layers: DlupGeometryContainer):
        self._layers = layers
        self._tags = []

    @property
    def tags(self) -> list[str]:
        return self._tags

    @classmethod
    def from_geojson(
        cls: Type[_TWsiAnnotations],
        geojsons: PathLike | Iterable[PathLike],
    ) -> _TWsiAnnotations:

        if isinstance(geojsons, str):
            _geojsons: Iterable[Any] = [pathlib.Path(geojsons)]

        _geojsons = [geojsons] if not isinstance(geojsons, (tuple, list)) else geojsons
        geometries: list[DlupPolygon | DlupPoint] = []
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
                    geometries += _geometry

        container = DlupGeometryContainer()
        for layer in geometries:
            if isinstance(layer, DlupPolygon):
                container.add_polygon(layer)
            elif isinstance(layer, DlupPoint):
                container.add_point(layer)
            else:
                raise ValueError(f"Unsupported layer type {type(layer)}")

        return cls(layers=container)

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

        all_layers = self._layers.polygons + self._layers.points
        for idx, curr_annotation in enumerate(all_layers):
            json_dict = _geometry_to_geojson(curr_annotation, label=curr_annotation.label, color=curr_annotation.color)
            json_dict["id"] = str(idx)
            data["features"].append(json_dict)

        return data

    def read_region(self, coordinates: tuple[int, int], scaling: float, size: tuple[int, int]):
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

    def rebuild_rtree(self):
        """
        Rebuild the R-tree for the annotations. This operation will be performed in-place.
        The R-tree is used for fast spatial queries on the annotations and is invalidated when the annotations are
        modified. This function will rebuild the R-tree. Strictly speaking, this is not required as the R-tree will be
        rebuilt on-demand when you invoke a `read_region()`. You could however do this if you want to avoid the `read_region()`
        to do it for you the first time it runs.
        """

        self._layers.rebuild_rtree()

    def reindex_polygons(self, index_map: dict[str, int]):
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


    def sort_polygons(self, key: callable, reverse: bool = False) -> None:
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
# Copyright (c) dlup contributors
"""Utilities to convert GeometryCollection objects into XML-like objects"""

from pathlib import Path

from dlup.geometry import GeometryCollection, Point, Polygon
from dlup.utils.annotations_utils import rgb_to_hex
from dlup.utils.schemas.generated import (
    BasePolygonType,
    DlupAnnotations,
    Geometries,
    Metadata,
    MultiPolygonType,
    StandalonePolygonType,
    Tag,
    Tags,
)


def create_xml_polygon(polygon: Polygon) -> StandalonePolygonType:
    """
    Convert a Polygon object to a Polygon XML object.

    Parameters
    ----------
    polygon : Polygon
        The Polygon object to convert.

    Returns
    -------
    StandalonePolygonType
        The converted Polygon XML object.

    """
    exterior_coords = [BasePolygonType.Exterior.Point(x=coord[0], y=coord[1]) for coord in polygon.get_exterior()]
    exterior = BasePolygonType.Exterior(point=exterior_coords)

    interiors_list = []
    for interior in polygon.get_interiors():
        interior_coords = [BasePolygonType.Interiors.Interior.Point(x=coord[0], y=coord[1]) for coord in interior]
        interiors_list.append(BasePolygonType.Interiors.Interior(point=interior_coords))
    interiors = BasePolygonType.Interiors(interior=interiors_list) if interiors_list else None

    return StandalonePolygonType(
        exterior=exterior,
        interiors=interiors,
        label=polygon.label,
        color=rgb_to_hex(*polygon.color) if polygon.color else None,
        index=polygon.index,
    )


def create_xml_point(point: Point) -> Geometries.Point:
    """
    Convert a Point object to a Point XML object.

    Parameters
    ----------
    point : Point
        The Point object to convert.

    Returns
    -------
    Geometries.Point
        The converted Point XML object.
    """
    return Geometries.Point(
        x=point.x, y=point.y, label=point.label, color=rgb_to_hex(*point.color) if point.color else None
    )


def create_xml_geometries(collection: GeometryCollection) -> Geometries:
    polygons = [create_xml_polygon(polygon) for polygon in collection.polygons]
    points = [create_xml_point(point) for point in collection.points]

    return Geometries(polygon=polygons, multi_polygon=[], point=points)

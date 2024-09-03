# Copyright (c) dlup contributors
"""Utilities to convert GeometryCollection objects into XML-like objects"""

from dlup.geometry import GeometryCollection, Point, Polygon
from dlup.utils.annotations_utils import rgb_to_hex
from dlup.utils.schemas.generated import BasePolygonType, Geometries, RegionsOfInterest, StandalonePolygonType


def create_xml_polygon(polygon: Polygon, order: int) -> StandalonePolygonType:
    """
    Convert a Polygon object to a Polygon XML object.

    Parameters
    ----------
    polygon : Polygon
        The Polygon object to convert.
    order : int
        The order of the polygon.

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
        order=order,
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
    polygons = [create_xml_polygon(polygon, order=idx) for idx, polygon in enumerate(collection.polygons)]
    points = [create_xml_point(point) for point in collection.points]

    return Geometries(polygon=polygons, multi_polygon=[], point=points)


def create_xml_rois(collection: GeometryCollection) -> RegionsOfInterest:
    raise NotImplementedError("This function is not implemented yet.")
    # polygons = [create_xml_polygon(polygon, order=idx) for idx, polygon in enumerate(collection.rois)]
    # return RegionsOfInterest(polygon=polygons, multi_polygon=[])

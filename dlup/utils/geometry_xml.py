# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to convert GeometryCollection objects into XML-like objects"""

from typing import Any, Optional

from dlup.geometry import Box, GeometryCollection, Point, Polygon
from dlup.utils.annotations_utils import hex_to_rgb, rgb_to_hex
from dlup.utils.schemas.generated import (
    BasePolygonType,
    BoxType,
    Geometries,
    RegionBoxType,
    RegionPolygonType,
    RegionsOfInterest,
    StandalonePolygonType,
)


def _create_base_polygon_type(polygon: Polygon) -> tuple[BasePolygonType.Exterior, BasePolygonType.Interiors]:
    """
    Helper function to create the exterior and interiors of a polygon.

    Parameters
    ----------
    polygon : Polygon
        The Polygon object to convert.

    Returns
    -------
    tuple
        A tuple containing exterior and interiors.
    """
    exterior_coords = [BasePolygonType.Exterior.Point(x=coord[0], y=coord[1]) for coord in polygon.get_exterior()]
    exterior = BasePolygonType.Exterior(point=exterior_coords)

    interiors_list = []
    for interior in polygon.get_interiors():
        interior_coords = [BasePolygonType.Interiors.Interior.Point(x=coord[0], y=coord[1]) for coord in interior]
        interiors_list.append(BasePolygonType.Interiors.Interior(point=interior_coords))
    interiors = BasePolygonType.Interiors(interior=interiors_list) if interiors_list else BasePolygonType.Interiors()

    return exterior, interiors


def create_xml_polygon(polygon: Polygon, order: int) -> StandalonePolygonType:
    """
    Convert a Polygon object to a StandalonePolygonType XML object.

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
    exterior, interiors = _create_base_polygon_type(polygon)
    return StandalonePolygonType(
        exterior=exterior,
        interiors=interiors,
        label=polygon.label,
        color=rgb_to_hex(*polygon.color) if polygon.color else None,
        index=polygon.index,
        order=order,
    )


def create_xml_roi_polygon(polygon: Polygon, order: int) -> RegionPolygonType:
    """
    Convert a Polygon object to a RegionPolygonType XML object.

    Parameters
    ----------
    polygon : Polygon
        The Polygon object to convert.
    order : int
        The order of the polygon.

    Returns
    -------
    RegionPolygonType
        The converted Polygon XML object.
    """
    exterior, interiors = _create_base_polygon_type(polygon)
    return RegionPolygonType(
        exterior=exterior,
        interiors=interiors,
        label=polygon.label,
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


def create_xml_box(box: Box) -> BoxType:
    """
    Convert a Box object to a BoxType XML object.

    Parameters
    ----------
    box : Box
        The Box object to convert.

    Returns
    -------
    BoxType
        The converted Box XML object.
    """
    x_min, y_min = box.coordinates
    width, height = box.size
    x_max = x_min + width
    y_max = y_min + height
    return BoxType(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        label=box.label,
        color=rgb_to_hex(*box.color) if box.color else None,
    )


def create_xml_geometries(collection: GeometryCollection) -> Geometries:
    """
    Convert a GeometryCollection to Geometries XML object.

    Parameters
    ----------
    collection : GeometryCollection
        The GeometryCollection to convert.

    Returns
    -------
    Geometries
        The converted Geometries XML object.
    """
    polygons = [create_xml_polygon(polygon, order=idx) for idx, polygon in enumerate(collection.polygons)]
    points = [create_xml_point(point) for point in collection.points]
    boxes = [create_xml_box(box) for box in collection.boxes]

    return Geometries(polygon=polygons, multi_polygon=[], box=boxes, point=points)


def create_xml_rois(collection: GeometryCollection) -> RegionsOfInterest:
    """
    Convert a GeometryCollection to RegionsOfInterest XML object, distinguishing between polygons and boxes.

    Parameters
    ----------
    collection : GeometryCollection
        The GeometryCollection to convert.

    Returns
    -------
    RegionsOfInterest
        The converted RegionsOfInterest XML object, including both polygons and boxes.
    """
    polygons = []
    boxes: list[RegionBoxType] = []

    for idx, polygon in enumerate(collection.rois):
        polygons.append(create_xml_roi_polygon(polygon, order=idx))

    return RegionsOfInterest(
        polygon=polygons,
        box=boxes,
        multi_polygon=[],  # Ensure that RegionsOfInterest schema includes 'box'
    )


def parse_dlup_xml_polygon(
    polygons: list[Any], order: Optional[int] = None, label: Optional[str] = None, index: Optional[int] = None
) -> list[tuple[Polygon, int]]:
    """
    Parse a list of polygons from a DLUP XML file.

    Parameters
    ----------
    polygons : list
        The list of polygons to parse.
    order : int, optional
        The order of the polygons, by default None.
    label : str, optional
        The label of the polygons, by default None.
    index : int, optional
        The index of the polygons, by default None.

    Returns
    -------
    list
        The list of parsed polygons as a list of tuples containing the polygon and its order.

    """
    output: list[tuple[Polygon, int]] = []
    for curr_polygon in polygons:
        if not curr_polygon.exterior:
            raise ValueError("Polygon does not have an exterior.")
        exterior = [(point.x, point.y) for point in curr_polygon.exterior.point]
        if curr_polygon.interiors:
            interiors = [
                [(point.x, point.y) for point in interior.point] for interior in curr_polygon.interiors.interior
            ]
        else:
            interiors = []

        # Use the label from curr_polygon if not passed in the function arguments
        polygon_label = label if label is not None else curr_polygon.label

        polygon = Polygon(
            exterior,
            interiors,
        )

        order = 0
        if hasattr(curr_polygon, "order"):
            if curr_polygon.order is not None:
                order = curr_polygon.order
        assert isinstance(order, int)

        # We use the given values if they are set, otherwise we take them from the properties
        if index is not None:
            polygon.index = index
        elif hasattr(curr_polygon, "index") and curr_polygon.index is not None:
            polygon.index = curr_polygon.index

        if polygon_label is not None:
            polygon.label = polygon_label
        elif hasattr(curr_polygon, "label"):
            polygon.label = curr_polygon.label

        if hasattr(curr_polygon, "color") and curr_polygon.color is not None:
            polygon.color = hex_to_rgb(curr_polygon.color)

        output.append((polygon, order))
    return output


def parse_dlup_xml_roi_box(box: RegionBoxType) -> tuple[Box, int]:
    """
    Parse a box from a DLUP XML file.

    Parameters
    ----------
    box : RegionBoxType
        The box to parse.

    Returns
    -------
    tuple
        The parsed box and its order.
    """
    # mypy struggles here
    assert isinstance(box.x_min, float)
    assert isinstance(box.y_min, float)
    assert isinstance(box.x_max, float)
    assert isinstance(box.y_max, float)

    output_box = Box((box.x_min, box.y_min), (box.x_max - box.x_min, box.y_max - box.y_min))

    if box.label:
        output_box.label = box.label
    if box.order:
        order = box.order
    else:
        order = 0

    return (output_box, order)

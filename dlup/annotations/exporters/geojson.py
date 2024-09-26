# Copyright (c) dlup contributors
import warnings
from typing import Any, Optional, TypedDict

import dlup
from dlup.geometry import Point, Polygon


class GeoJsonDict(TypedDict):
    """
    TypedDict for standard GeoJSON output
    """

    id: str | None
    type: str
    features: list[dict[str, str | dict[str, str]]]
    metadata: Optional[dict[str, str | list[str]]]


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


def geojson_exporter(cls: "dlup.annotations.SlideAnnotations") -> GeoJsonDict:
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
    if cls.tags:
        data["metadata"] = {"tags": [_.label for _ in cls.tags]}

    if cls.layers.boxes:
        warnings.warn("Bounding boxes are not supported in GeoJSON and will be skipped.", UserWarning)

    all_layers = cls.layers.polygons + cls.layers.points
    for idx, curr_annotation in enumerate(all_layers):
        json_dict = _geometry_to_geojson(curr_annotation, label=curr_annotation.label, color=curr_annotation.color)
        json_dict["id"] = str(idx)
        data["features"].append(json_dict)

    return data

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
"""
GeoJSON importer for annotations.
"""

from __future__ import annotations

import errno
import json
import os
import pathlib
from typing import Optional, Type, TypedDict, TypeVar

import dlup
import numpy as np
from dlup._exceptions import AnnotationError
from dlup._types import PathLike
from dlup.annotations import AnnotationSorting
from dlup.geometry import Point, Polygon
from dlup.utils.annotations_utils import get_geojson_color

_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="dlup.annotations.SlideAnnotations")


class CoordinatesDict(TypedDict):
    type: str
    coordinates: list[list[list[float]]]


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


def geojson_importer(
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
    path = pathlib.Path(geojsons)
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

    instance = cls(sorting=sorting)

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
                        instance.add_roi(geometry)
                    else:
                        instance.add_polygon(geometry)
                elif isinstance(geometry, Point):
                    instance.add_point(geometry)
                else:
                    raise ValueError(f"Unsupported geometry type {geometry}")

    instance._in_place_sort_and_scale(scaling, sorting)
    instance.rebuild_rtree()
    return instance

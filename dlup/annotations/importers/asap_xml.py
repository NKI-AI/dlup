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
Experimental annotations module for dlup.

"""

from __future__ import annotations

import errno
import os
import pathlib
import xml.etree.ElementTree as ET
from typing import Optional, Type, TypeVar

from dlup._types import PathLike
from dlup.annotations import AnnotationSorting, SlideAnnotations
from dlup.geometry import Point, Polygon
from dlup.utils.annotations_utils import hex_to_rgb

_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="SlideAnnotations")


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


def asap_xml_importer(
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
    instance = cls(tags=None, sorting=sorting)
    for parent in opened_annotation:
        for child in parent:
            if child.tag != "Annotation":
                continue
            label = child.attrib.get("PartOfGroup").strip()  # type: ignore
            color = hex_to_rgb(child.attrib.get("Color").strip())  # type: ignore

            annotation_type = child.attrib.get("Type").lower()  # type: ignore
            coordinates = _parse_asap_coordinates(child)

            if annotation_type in ["dot", "pointset"]:
                for point in coordinates:
                    instance.add_point(Point(*point, label=label, color=color))

            elif annotation_type == "polygon":
                if label in roi_names:
                    instance.add_roi(Polygon(coordinates, [], label=label, color=color))
                else:
                    instance.add_polygon(Polygon(coordinates, [], label=label, color=color))

            else:
                raise ValueError(f"Unknown annotation type: {annotation_type}")

    instance._in_place_sort_and_scale(scaling, sorting)
    instance.rebuild_rtree()
    return instance

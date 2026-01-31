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
import functools
import os
import pathlib
import warnings
from enum import Enum
from typing import Any, Iterable, NamedTuple, Optional, Type, TypeVar

from dlup._types import PathLike
from dlup.annotations import AnnotationSorting, SlideAnnotations
from dlup.annotations.tags import SlideTag, TagAttribute
from dlup.geometry import Box, Point, Polygon
from dlup.utils.imports import DARWIN_SDK_AVAILABLE

_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="SlideAnnotations")


class AnnotationType(str, Enum):
    POINT = "POINT"
    BOX = "BOX"
    POLYGON = "POLYGON"
    TAG = "TAG"
    RASTER = "RASTER"


class DarwinV7Metadata(NamedTuple):
    label: str
    color: Optional[tuple[int, int, int]]
    annotation_type: AnnotationType


@functools.lru_cache(maxsize=None)
def get_v7_metadata(filename: pathlib.Path) -> Optional[dict[tuple[str, str], DarwinV7Metadata]]:
    """Get V7 metadata from a filename.

    Parameters
    ----------
    filename : pathlib.Path
        Path to the metadata file.

    Returns
    -------
    Optional[dict[tuple[str, str], DarwinV7Metadata]]
        Dictionary with metadata.
    """
    if not DARWIN_SDK_AVAILABLE:
        warnings.warn("`darwin` is not available. Install using `python -m pip install darwin-py`.")
        return None
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


def darwin_json_importer(
    cls: Type[_TSlideAnnotations],
    darwin_json: PathLike,
    scaling: float | None = None,
    sorting: AnnotationSorting | str = AnnotationSorting.NONE,
    z_indices: Optional[dict[str, int]] = None,
    roi_names: Optional[list[str]] = None,
) -> _TSlideAnnotations:
    """Import annotations from Darwin V7 [1]_.

    Parameters
    ----------
    darwin_json : PathLike
        Path to Darwin annotations in json format.
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
        warnings.warn("`darwin` is not available. Install using `python -m pip install darwin-py`.")
        # Return an empty instance instead of raising an exception
        return cls(sorting=sorting)
    import darwin

    roi_names = [] if roi_names is None else roi_names

    darwin_json_fn = pathlib.Path(darwin_json)
    if not darwin_json_fn.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(darwin_json_fn))

    darwin_an = darwin.utils.parse_darwin_json(darwin_json_fn, None)
    v7_metadata = get_v7_metadata(darwin_json_fn.parent)

    tags: list[SlideTag] = []
    polygons: list[tuple[Polygon, int]] = []
    boxes: list[Box] = []

    instance = cls(tags=tuple(tags), sorting=sorting)

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
            curr_point = Point(curr_data["x"], curr_data["y"])
            curr_point.label = name
            curr_point.color = annotation_color
            instance.add_point(curr_point)

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
            x, y, w, h = curr_data["x"], curr_data["y"], curr_data["w"], curr_data["h"]
            box = Box((x, y), (w, h))
            box.label = name
            boxes.append(box)

        else:
            raise ValueError(f"Annotation type {annotation_type} is not supported.")

    if sorting == "Z_INDEX":
        for polygon, _ in sorted(polygons, key=lambda x: x[1]):
            if polygon.label in roi_names:
                instance.add_roi(polygon)
            else:
                instance.add_polygon(polygon)
    else:
        for polygon, _ in polygons:
            if polygon.label in roi_names:
                instance.add_roi(polygon)
            else:
                instance.add_polygon(polygon)
        for box in boxes:
            if box.label in roi_names:
                instance.add_roi(box.as_polygon())
            else:
                instance.add_box(box)

    instance._in_place_sort_and_scale(scaling, sorting="NONE" if sorting == "Z_INDEX" else sorting)
    instance._tags = tuple(tags)
    instance.rebuild_rtree()
    return instance

import errno
import os
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Type, TypeVar

from xsdata.formats.dataclass.parsers import XmlParser

import dlup
from dlup._types import PathLike
from dlup.annotations.tags import SlideTag
from dlup.geometry import Box, GeometryCollection, Point, Polygon
from dlup.utils.annotations_utils import hex_to_rgb
from dlup.utils.geometry_xml import parse_dlup_xml_polygon, parse_dlup_xml_roi_box
from dlup.utils.schemas.generated import DlupAnnotations as XMLDlupAnnotations

_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="dlup.annotations.SlideAnnotations")


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


def dlup_xml_importer(cls: Type[_TSlideAnnotations], dlup_xml: PathLike) -> _TSlideAnnotations:
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

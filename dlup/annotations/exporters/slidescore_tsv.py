import csv
import io
import json
import warnings
from collections import defaultdict
from typing import Any, List, Optional, Union, cast

from dlup.annotations import SlideAnnotations
from dlup.annotations.importers.slidescore_tsv import (
    CoordDict,
    SlideScoreBox,
    SlideScoreBrush,
    SlideScoreEllipse,
    SlideScorePolgyon,
    _SlideScorePolygonTypes,
)
from dlup.geometry import Box, Point, Polygon


def process_point_annotation(point: Point) -> CoordDict:
    """Convert Point annotation to Slidescore Point annotation."""
    return {"x": point.x, "y": point.y}


def process_box_annotation(box: Box) -> SlideScoreBox:
    """Process Box annotation into Slidescore Rectangle annotation"""
    coord_x, coord_y = box.coordinates
    size_x, size_y = box.size
    return {"type": "rect", "corner": {"x": coord_x, "y": coord_y}, "size": {"x": size_x, "y": size_y}}


def process_polygon_annotation(polygon: Polygon) -> Union[SlideScoreEllipse, SlideScorePolgyon, SlideScoreBrush]:
    """Process Polygon into Slidescore Ellipse, Polygon or Brush annotation depending on attributes and interior
    holes."""
    if "_ellipse_approximation" in polygon.fields:
        return convert_polygon_to_slidescore_ellipse(polygon)
    exterior: List[CoordDict] = [{"x": p[0], "y": p[1]} for p in polygon.get_exterior()]
    interiors: List[List[CoordDict]] = [[{"x": p[0], "y": p[1]} for p in hole] for hole in polygon.get_interiors()]

    if interiors:
        return {"type": "brush", "positivePolygons": [exterior], "negativePolygons": interiors}
    else:
        return {"type": "polygon", "points": exterior}


def convert_polygon_to_slidescore_ellipse(polygon: Polygon) -> SlideScoreEllipse:
    """Converts a Polygon with ellipse approximation attributes back to Slidescore Ellipse."""
    if "_ellipse_approximation" not in polygon.fields:
        raise ValueError("Polygon should have _is_ellipse_appriximation field to be converted to ellipse")

    center_x, center_y = polygon.get_field("_ellipse_center")
    width, height = polygon.get_field("_ellipse_size")
    return {"type": "ellipse", "center": {"x": center_x, "y": center_y}, "size": {"x": width, "y": height}}


def process_annotation(annotation: Union[Point, Polygon, Box]) -> Union[CoordDict, _SlideScorePolygonTypes]:
    """Process a single DLUP annotation to Slidescore format

    Parameters
    ----------
    annotation : Union[Point, Polygon, Box]
        DLUP annotation

    Returns
    -------
    Union[CoordDict, _SlideScorePolygonTypes]
        Annotation in Slidescore answer format.
    """
    if isinstance(annotation, Point):
        return process_point_annotation(annotation)
    elif isinstance(annotation, Box):
        return process_box_annotation(annotation)
    return process_polygon_annotation(annotation)


def slidescore_tsv_exporter(
    cls: "SlideAnnotations",
    image_id: Optional[int] = None,
    image_name: Optional[str] = None,
    user_email: Optional[str] = None,
) -> str:
    """Export the annotations as tab seperated Slidescore results answer.

    Parameters
    ----------
    image_id : Optional[int], optional
        Slidescore image ID associated with annotations, by default None
    image_name : Optional[str], optional
        Slidescore image name associated with annotations, by default None
    user_email : Optional[str], optional
        User email associated with annotations, by default None

    Returns
    -------
    str
        Slidescore annotations as a tab seperated string. Each line consist of a single question and corresponding
        answer. First row are the standard Slidescore headers.

    Raises
    ------
    ValueError
        Raises if multiple tags have the same label.
    """
    tag_annotations: dict[str, list[Union[str, bool]] | Union[str, bool]] = {}
    polygon_annotations: defaultdict[str, list[_SlideScorePolygonTypes]] = defaultdict(list)
    point_annotations: defaultdict[str, list[CoordDict]] = defaultdict(list)

    # TODO: Get image_id, image_name and user_email from metadata or tags if possible
    if cls.tags:
        for tag in cls.tags:
            if tag.label in list(tag_annotations.keys()):
                raise ValueError("Tags should have unique names.")

            if tag.attributes is not None:
                tag_answer: list[str | bool] = [attr.label for attr in tag.attributes]
            else:
                tag_answer = [True]
            tag_annotations[tag.label] = tag_answer if len(tag_answer) > 1 else tag_answer[0]

    all_layers = cls.polygons + cls.boxes + cls.points + cls.rois
    for curr_annotation in all_layers:
        label = curr_annotation.label
        annotation = process_annotation(curr_annotation)

        if isinstance(curr_annotation, (Polygon, Box)):
            if label is None:
                label = "Unknown AnnoShapes"
            polygon_annotations[label].append(cast(_SlideScorePolygonTypes, annotation))
        elif isinstance(curr_annotation, Point):
            if label is None:
                label = "Unknown AnnoPoints"
            point_annotations[label].append(cast(CoordDict, annotation))

    point_labels = set(point_annotations.keys())
    polygon_labels = set(polygon_annotations.keys())
    if point_labels.intersection(polygon_labels):
        warnings.warn(
            f"Found overlapping labels for points and polygon like annotation:"
            f"{point_labels.intersection(polygon_labels)}. Be aware that the same question cannot hold both"
            "AnnoPoints and AnnoShapes."
        )

    tag_labels = set(tag_annotations.keys())
    if tag_labels.intersection(point_labels.union(polygon_labels)):
        warnings.warn(
            f"Found overlapping labels for tags and annotations: {point_labels.intersection(polygon_labels)}."
            "Be aware that the questions should hold labels or AnnoPoints and AnnoShapes."
        )

    annotations_list: list[dict[str, Any]] = [
        tag_annotations,
        point_annotations,
        polygon_annotations,
    ]
    with io.StringIO() as output:
        csv_writer = csv.writer(output, delimiter="\t", quotechar=None)
        csv_writer.writerow(["ImageID", "Image Name", "By", "Question", "Answer"])
        for annotations_dict in annotations_list:
            for label, annotations in annotations_dict.items():
                answer = json.dumps(annotations) if isinstance(annotations, (dict, list)) else annotations
                csv_writer.writerow([image_id, image_name, user_email, label, answer])
        output_str = output.getvalue()
    return output_str

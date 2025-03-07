import csv
import errno
import json
import os
import pathlib
from typing import List, Optional, Type, TypedDict, TypeVar, Union

import numpy as np

import dlup
from dlup._types import PathLike
from dlup.annotations.tags import SlideTag, TagAttribute
from dlup.geometry import Box, GeometryCollection, Point, Polygon


class CoordDict(TypedDict):
    x: float
    y: float


class SlideScoreAnnotation(TypedDict):
    type: str


class SlideScoreBox(SlideScoreAnnotation):
    corner: CoordDict
    size: CoordDict


class SlideScoreBrush(SlideScoreAnnotation):
    positivePolygons: List[List[CoordDict]]
    negativePolygons: List[List[CoordDict]]


class SlideScoreEllipse(SlideScoreAnnotation):
    center: CoordDict
    size: CoordDict


class SlideScorePolgyon(SlideScoreAnnotation):
    points: List[CoordDict]


_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="dlup.annotations.SlideAnnotations")
_SlideScorePolygonTypes = Union[SlideScoreBox, SlideScoreEllipse, SlideScorePolgyon, SlideScoreBrush]


def parse_points(data: list[CoordDict], label: str) -> List[Point]:
    """Parse Point answers as list of Points"""
    if not isinstance(data, list) or not all("x" in item and "y" in item for item in data):
        raise ValueError("Invalid point data format")
    return [Point(item["x"], item["y"], label=label) for item in data]


def parse_polygon(data: SlideScorePolgyon, label: str) -> Polygon:
    """Parse Polygon answers as Polgyon"""
    if not isinstance(data, dict) or "type" not in data or "points" not in data:
        raise ValueError("Invalid polygon data format")
    return Polygon([(p["x"], p["y"]) for p in data["points"]], label=label)


def parse_brush(data: SlideScoreBrush, label: str) -> List[Polygon]:
    """Parse Brush answers as Polygon (with holes)"""
    if not isinstance(data, dict) or "positivePolygons" not in data or "negativePolygons" not in data:
        raise ValueError("Invalid brush data format")

    positive_polygons = [Polygon([(p["x"], p["y"]) for p in poly], label=label) for poly in data["positivePolygons"]]
    negative_polygons = [Polygon([(p["x"], p["y"]) for p in poly]) for poly in data["negativePolygons"]]

    polygons: list[Polygon] = []
    for pos_poly in positive_polygons:
        holes = []
        for neg_poly in negative_polygons:
            if pos_poly.contains(neg_poly):
                holes.append(neg_poly.get_exterior())  # Store exterior as a hole

        # If holes exist, attach them
        if holes:
            state = pos_poly.__getstate__()  # Get current state
            state["_object"]["interiors"] = holes  # Update interiors (holes)
            # Setting state does not work for some reason
            polygon_w_holes = Polygon(state["_object"]["exterior"], state["_object"]["interiors"], **state["_fields"])
            polygons.append(polygon_w_holes)
        else:
            polygons.append(pos_poly)

    return polygons


def parse_ellipse(data: SlideScoreEllipse, label: str, num_approximation_points: int = 100) -> Polygon:
    """Parse Ellipse answers as Polgyon by approximating it's boundaries using equations"""
    if not isinstance(data, dict) or "center" not in data or "size" not in data:
        raise ValueError("Invalid ellipse data format")

    center_x, center_y = data["center"]["x"], data["center"]["y"]
    width, height = data["size"]["x"], data["size"]["y"]

    # Generate ellipse points using parametric equations
    theta = np.linspace(0, 2 * np.pi, num_approximation_points)
    ellipse_points = [(center_x + width * np.cos(t), center_y + height * np.sin(t)) for t in theta]
    ellipse_polygon = Polygon(ellipse_points, label=label)
    # Set ellipse information as fields to be able to retrieve them later
    ellipse_polygon.set_field("_ellipse_approximation", True)
    ellipse_polygon.set_field("_ellipse_center", (center_x, center_y))
    ellipse_polygon.set_field("_ellipse_size", (width, height))
    return ellipse_polygon


def parse_rectangle(data: SlideScoreBox, label: str, as_box: bool = False) -> Union[Box, Polygon]:
    """Parse Rect answers as Polygon or Box"""
    if not isinstance(data, dict) or "corner" not in data or "size" not in data:
        raise ValueError("Invalid rectangle data format")
    min_x, min_y = data["corner"]["x"], data["corner"]["y"]
    width, height = data["size"]["x"], data["size"]["y"]

    # GeometryCollection does not iterate over boxes so we return a polygon by default
    if as_box:
        return Box((min_x, min_y), (width, height), label=label)
    max_x, max_y = min_x + width, min_y + height
    return Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)], label=label)


def parse_tag(data: Union[str, int, float], label: str) -> SlideTag:
    """Parse numberic or text answers as SlideTag"""
    if not isinstance(data, (str, float, int)):
        raise ValueError("Tags answers must be of type int, str or float")
    return SlideTag(attributes=[TagAttribute(str(data), color=None)], label=label, color=None)


def parse_annotations(annotation: str, label: str) -> Union[List[Union[Box, Polygon]], List[Point], List[SlideTag]]:
    """Parse Slidescore annotation answer to list of DLUP annotations

    Parameters
    ----------
    annotation : str
        Annotation in Slidescore json format from Slidescore `Answer` column
    label : str
        Label to assign to output annotation(s). Slidescore `Question` column can be used for this.

    Returns
    -------
    Union[List[Union[Box, Polygon]], List[Point], List[SlideTag]]
        Parsed annotations as a list of DLUP annotations.

    Raises
    ------
    ValueError
        ValueError gets raised if json data cannot be parsed or if unexpected data format is encountered.
    """
    stripped = annotation.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        try:
            data = json.loads(stripped)
            assert isinstance(data, list)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in annotations")
    else:
        data = annotation

    if isinstance(data, list):
        if all("x" in item and "y" in item for item in data):
            return parse_points(data, label=label)
        elif all("type" in item for item in data):
            parsed_data: list[Union[Polygon, Box]] = []
            for item in data:
                if item["type"] == "polygon":
                    parsed_data.append(parse_polygon(item, label=label))
                elif item["type"] == "brush":
                    parsed_data.extend(parse_brush(item, label=label))
                elif item["type"] == "ellipse":
                    parsed_data.append(parse_ellipse(item, label=label))
                elif item["type"] == "rect":
                    parsed_data.append(parse_rectangle(item, label=label))
            return parsed_data
    elif isinstance(data, (str, int, float)):
        return [parse_tag(data, label=label)]
    raise ValueError("Unexpected annotation format")


def slidescore_tsv_importer(
    cls: Type[_TSlideAnnotations],
    slidescore_tsv: PathLike,
    image_id: Optional[List[int] | int] = None,
    image_name: Optional[List[str] | str] = None,
    user_email: Optional[List[str] | str] = None,
    question: Optional[List[str] | str] = None,
) -> _TSlideAnnotations:
    """Read annotations as a Slidescore answer TSV file.

    Parameters
    ----------
    slidescore_tsv : PathLike
        Path to the Slidescore TSV file.
    image_id : Optional[List[int]  |  int], optional
        image id(s) to include in imported annotations, by default None
    image_name : Optional[List[str]  |  str], optional
        image name(s) to include in imported annotations, by default None
    user_email : Optional[List[str]  |  str], optional
        user email(s) to include in imported annotations, by default None
    question : Optional[List[str]  |  str], optional
        question(s) to include in imported annotations, by default None

    Returns
    -------
    _TSlideAnnotations
        Imported annotations as SlideAnnotations object

    Raises
    ------
    FileNotFoundError
        FileNotFoundError will be raise if file does not exists.
    RuntimeError
        RuntimeError will be raised when Slidescore answer cannot be parsed correctly
    """
    path = pathlib.Path(slidescore_tsv)
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

    with open(slidescore_tsv, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows = list(reader)

    filters = {"ImageID": image_id, "Image Name": image_name, "By": user_email, "Question": question}
    if all(value is None for value in filters.values()):
        filtered_rows = rows
    else:
        filtered_rows = []
        for row in rows:
            if all(
                filter_value is None
                or (isinstance(filter_value, (str, int)) and row.get(filter_key) == filter_value)
                or (isinstance(filter_value, list) and row.get(filter_key) in filter_value)
                for filter_key, filter_value in filters.items()
            ):
                filtered_rows.append(row)
    metadata = filters

    collection = GeometryCollection()
    tags = []
    for row in filtered_rows:
        parsed_annotations = parse_annotations(annotation=row["Answer"], label=row["Question"])
        for parsed_annotation in parsed_annotations:
            if isinstance(parsed_annotation, SlideTag):
                tags.append(parsed_annotation)
            elif isinstance(parsed_annotation, Point):
                collection.add_point(parsed_annotation)
            elif isinstance(parsed_annotation, Polygon):
                collection.add_polygon(parsed_annotation)
            elif isinstance(parsed_annotation, Box):
                collection.add_box(parsed_annotation)
            else:
                raise RuntimeError("Could not parse slidescore annotation.")
    return cls(layers=collection, tags=tuple(tags), metadata=metadata)

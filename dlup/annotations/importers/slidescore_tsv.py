import csv
import errno
import json
import os
import pathlib
from typing import List, Optional, Type, TypedDict, TypeVar, Union

import numpy as np

import dlup
from dlup._types import PathLike
from dlup.annotations import AnnotationSorting
from dlup.annotations.tags import SlideTag, TagAttribute
from dlup.geometry import Box, Point, Polygon


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


def parse_points(data: list[CoordDict], label: str, color: Optional[tuple[int, int, int]] = None) -> List[Point]:
    """Parse Point answers as list of Points"""
    if not isinstance(data, list) or not all("x" in item and "y" in item for item in data):
        raise ValueError("Invalid point data format")
    kwargs = {"label": label, "color": color}
    return [Point(item["x"], item["y"], **kwargs) for item in data]


def parse_polygon(
    data: SlideScorePolgyon, label: str, index: Optional[int] = None, color: Optional[tuple[int, int, int]] = None
) -> Polygon:
    """Parse Polygon answers as Polgyon"""
    if not isinstance(data, dict) or "type" not in data or "points" not in data:
        raise ValueError("Invalid polygon data format")
    kwargs = {"label": label}
    if index is not None:
        kwargs["index"] = index
    if color is not None:
        kwargs["color"] = color
    return Polygon([(p["x"], p["y"]) for p in data["points"]], **kwargs)


def parse_brush(
    data: SlideScoreBrush, label: str, index: Optional[int] = None, color: Optional[tuple[int, int, int]] = None
) -> List[Polygon]:
    """Parse Brush answers as Polygon (with holes)"""
    if not isinstance(data, dict) or "positivePolygons" not in data or "negativePolygons" not in data:
        raise ValueError("Invalid brush data format")

    kwargs = {"label": label}
    if index is not None:
        kwargs["index"] = index
    if color is not None:
        kwargs["color"] = color
    positive_polygons = [Polygon([(p["x"], p["y"]) for p in poly], **kwargs) for poly in data["positivePolygons"]]
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


def parse_ellipse(
    data: SlideScoreEllipse,
    label: str,
    index: Optional[int] = None,
    color: Optional[tuple[int, int, int]] = None,
    num_approximation_points: int = 100,
) -> Polygon:
    """Parse Ellipse answers as Polgyon by approximating it's boundaries using equation"""
    if not isinstance(data, dict) or "center" not in data or "size" not in data:
        raise ValueError("Invalid ellipse data format")

    center_x, center_y = data["center"]["x"], data["center"]["y"]
    width, height = data["size"]["x"], data["size"]["y"]

    # Generate ellipse points using parametric equations
    # NOTE: Slidescore only allows unrotated ellipses
    theta = np.linspace(0, 2 * np.pi, num_approximation_points)
    ellipse_points = [(center_x + width * np.cos(t), center_y + height * np.sin(t)) for t in theta]
    kwargs = {"label": label}
    if index is not None:
        kwargs["index"] = index
    if color is not None:
        kwargs["color"] = color
    ellipse_polygon = Polygon(ellipse_points, **kwargs)
    # Set ellipse information as fields to be able to retrieve them later
    ellipse_polygon.set_field("_ellipse_approximation", True)
    ellipse_polygon.set_field("_ellipse_center", (center_x, center_y))
    ellipse_polygon.set_field("_ellipse_size", (width, height))
    return ellipse_polygon


def parse_rectangle(
    data: SlideScoreBox,
    label: str,
    index: Optional[int] = None,
    color: Optional[tuple[int, int, int]] = None,
    box_as_polygon: bool = True,
) -> Union[Box, Polygon]:
    """Parse Rect answers as Polygon or Box"""
    if not isinstance(data, dict) or "corner" not in data or "size" not in data:
        raise ValueError("Invalid rectangle data format")
    min_x, min_y = data["corner"]["x"], data["corner"]["y"]
    width, height = data["size"]["x"], data["size"]["y"]

    # GeometryCollection does not iterate over boxes so we return a polygon by default
    kwargs = {"label": label}
    if index is not None:
        kwargs["index"] = index
    if color is not None:
        kwargs["color"] = color
    box_annotation = Box((min_x, min_y), (width, height), **kwargs)
    return box_annotation.as_polygon() if box_as_polygon else box_annotation


def parse_tag(data: Union[str, int, float, list], label: str) -> SlideTag:
    """Parse numberic or text answers as SlideTag"""
    if not isinstance(data, (str, float, int, list)):
        raise ValueError("Tags answers must be of type int, str or float")
    attributes_list = (
        [TagAttribute(data, color=None)]  # type: ignore
        if not isinstance(data, list)
        else [TagAttribute(item, color=None) for item in data]
    )
    return SlideTag(attributes=attributes_list, label=label, color=None)


def parse_annotations(
    annotation: str,
    label: str,
    box_as_polygon: bool = False,
    index: Optional[int] = None,
    color: Optional[tuple[int, int, int]] = None,
) -> Union[List[Union[Box, Polygon]], List[Point], List[SlideTag]]:
    """Parse Slidescore annotation answer to list of DLUP annotations

    Parameters
    ----------
    annotation : str
        Annotation in Slidescore json format from Slidescore `Answer` column
    label : str
        Label to assign to output annotation(s). Slidescore `Question` column can be used for this.
    box_as_polygon : bool, optional
        If True, rectangles are converted to polygons, and added as such.
        This is useful because GeoJSON does not support boxes.
    index : Optional[int], optional
        If set, this index will be used as a z_index for the annotation.
    color : Optional[tuple[int, int, int]], optional
        If set, this color will be assigned to the annotation as an RGB tuple.

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

    if isinstance(data, list) and all("x" in item and "y" in item for item in data):
        return parse_points(data, label=label, color=color)
    elif isinstance(data, list) and all("type" in item for item in data):
        parsed_data: list[Union[Polygon, Box]] = []
        for item in data:
            if item["type"] == "polygon":
                parsed_data.append(parse_polygon(item, label=label, index=index, color=color))
            elif item["type"] == "brush":
                parsed_data.extend(parse_brush(item, label=label, index=index, color=color))
            elif item["type"] == "ellipse":
                parsed_data.append(parse_ellipse(item, label=label, index=index, color=color))
            elif item["type"] == "rect":
                parsed_data.append(
                    parse_rectangle(item, label=label, box_as_polygon=box_as_polygon, index=index, color=color)
                )
        return parsed_data
    elif isinstance(data, (str, int, float, list)):
        return [parse_tag(data, label=label)]
    raise ValueError("Unexpected annotation format")


def slidescore_tsv_importer(
    cls: Type[_TSlideAnnotations],
    slidescore_tsv: PathLike,
    image_id: Optional[list[int] | int] = None,
    image_name: Optional[list[str] | str] = None,
    user_email: Optional[list[str] | str] = None,
    question: Optional[list[str] | str] = None,
    sorting: AnnotationSorting | str = AnnotationSorting.NONE,
    box_as_polygon: bool = False,
    z_indices: Optional[dict[str, int]] = None,
    color_map: Optional[dict[str, tuple[int, int, int]]] = None,
    roi_names: Optional[list[str]] = None,
) -> _TSlideAnnotations:
    """Read annotations as a Slidescore answer TSV file.

    Parameters
    ----------
    slidescore_tsv : PathLike
        Path to the Slidescore TSV file.
    image_id : Optional[list[int]  |  int], optional
        image id(s) to include in imported annotations, by default None
    image_name : Optional[list[str]  |  str], optional
        image name(s) to include in imported annotations, by default None
    user_email : Optional[list[str]  |  str], optional
        user email(s) to include in imported annotations, by default None
    question : Optional[list[str]  |  str], optional
        question(s) to include in imported annotations, by default None
    sorting : AnnotationSorting
        The sorting to apply to the annotations. Check the `AnnotationSorting` enum for more information.
        By default, the annotations are sorted by area.
    z_indices: dict[str, int], optional
        If set, these z_indices will be used rather than the default order.
    color_map: dict[str, tuple[int, int, int]], optional
        If set, these colors will be assigned to the annotations as RGB tuples. SlideScore does not export colors, so they must be
        provided manually.
    roi_names : list[str], optional
        List of names that should be considered as regions of interest. If set, these will be added as ROIs rather
        than polygons.
    box_as_polygon : bool, optional
        If True, rectangles are converted to polygons, and added as such.
        This is useful because GeoJSON does not support boxes.

    Returns
    -------
    SlideAnnotations
        Imported annotations as SlideAnnotations object

    Raises
    ------
    FileNotFoundError
        FileNotFoundError will be raise if file does not exists.
    RuntimeError
        RuntimeError will be raised when Slidescore answer cannot be parsed correctly
    """
    path = pathlib.Path(slidescore_tsv)
    roi_names = [] if roi_names is None else roi_names
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

    tags: list[SlideTag] = []
    instance = cls(sorting=sorting, tags=tuple(tags), metadata=metadata)
    for row in filtered_rows:
        annotation: str = row["Answer"]
        label: str = row["Question"]
        z_index = z_indices.get(label, None) if z_indices else None
        color = color_map.get(label, None) if color_map else None

        parsed_annotations = parse_annotations(
            annotation=annotation, label=label, box_as_polygon=box_as_polygon, index=z_index, color=color
        )
        for parsed_annotation in parsed_annotations:
            if isinstance(parsed_annotation, SlideTag):
                tags.append(parsed_annotation)
            elif isinstance(parsed_annotation, Point):
                instance.add_point(parsed_annotation)
            # We check the label for rois before adding boxes / polygons but after points and slidetags
            elif parsed_annotation.label in roi_names:
                instance.add_roi(parsed_annotation)
            elif isinstance(parsed_annotation, Polygon):
                instance.add_polygon(parsed_annotation)
            elif isinstance(parsed_annotation, Box):
                instance.add_box(parsed_annotation)
            else:
                raise RuntimeError("Could not parse slidescore annotation.")

    instance._tags = tuple(tags)
    return instance

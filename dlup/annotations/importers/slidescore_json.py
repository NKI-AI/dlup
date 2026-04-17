"""Slidescore JSON importer for dlup.

This importer supports Slidescore exports of the form:

{
  "image_id": 123,
  "study_id": 456,
  "image_name": "Some slide",
  "annotations": [
    {"user": "a@b", "question": "ROI", "data": [{"type": "rect", ...}, ...]},
    {"user": "a@b", "question": "Tumor+", "data": [{"x": 1, "y": 2}, ...]},
  ]
}

The per-question payload is routed through the existing Slidescore TSV parsing logic
(`parse_annotations`) by serializing `data` to JSON when appropriate.
"""

from __future__ import annotations

import errno
import json
import os
import pathlib
from typing import Any, Optional, Type

from dlup._types import PathLike
from dlup.annotations import AnnotationSorting
from dlup.annotations.importers.slidescore_tsv import parse_annotations
from dlup.annotations.tags import SlideTag
from dlup.geometry import Box, Point, Polygon


def _matches_filter(value: Any, filter_value: Any) -> bool:
    """Return True if `value` passes the Slidescore filter semantics used by the TSV importer."""
    if filter_value is None:
        return True
    if isinstance(filter_value, list):
        return value in filter_value
    return value == filter_value


def slidescore_json_importer(
    cls: Type[Any],
    slidescore_json: PathLike,
    image_id: Optional[list[int] | int] = None,
    image_name: Optional[list[str] | str] = None,
    user_email: Optional[list[str] | str] = None,
    question: Optional[list[str] | str] = None,
    sorting: AnnotationSorting | str = AnnotationSorting.NONE,
    box_as_polygon: bool = False,
    roi_names: Optional[list[str]] = None,
) -> Any:
    """Read annotations from a Slidescore JSON export.

    Parameters
    ----------
    slidescore_json : PathLike
        Path to the Slidescore JSON export.
    image_id : Optional[list[int] | int], optional
        Image id(s) to include in imported annotations, by default None.
    image_name : Optional[list[str] | str], optional
        Image name(s) to include in imported annotations, by default None.
    user_email : Optional[list[str] | str], optional
        User email(s) to include in imported annotations, by default None.
    question : Optional[list[str] | str], optional
        Question(s) to include in imported annotations, by default None.
    sorting : AnnotationSorting | str
        Sorting applied to the annotations, by default AnnotationSorting.NONE.
    box_as_polygon : bool, optional
        If True, rectangles are converted to polygons (useful for GeoJSON), by default False.
    roi_names : Optional[list[str]], optional
        Labels (questions) that should be interpreted as ROIs and added via `add_roi()`.

    Returns
    -------
    SlideAnnotations
        Imported annotations.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If the JSON does not match the expected Slidescore export structure.
    """
    path = pathlib.Path(slidescore_json)
    roi_names = [] if roi_names is None else roi_names
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Slidescore JSON must be a JSON object at the top level.")

    top_image_id = payload.get("image_id")
    top_image_name = payload.get("image_name")

    # If a filter is specified and doesn't match the file-level metadata, return an empty collection.
    if not _matches_filter(top_image_id, image_id) or not _matches_filter(top_image_name, image_name):
        return cls(
            sorting=sorting,
            tags=tuple(),
            metadata={
                "slidescore_filtered_out": True,
            },
        )

    annotations_groups = payload.get("annotations", [])
    if not isinstance(annotations_groups, list):
        raise ValueError("Slidescore JSON must contain an 'annotations' list.")

    study_id = payload.get("study_id")
    metadata: dict[str, list[str] | str | int | float | bool] = {
        "slidescore_image_id": int(top_image_id) if isinstance(top_image_id, int) else 0,
        "slidescore_study_id": int(study_id) if isinstance(study_id, int) else 0,
        "slidescore_image_name": str(top_image_name) if top_image_name is not None else "",
    }

    tags: list[SlideTag] = []
    instance = cls(sorting=sorting, tags=tuple(tags), metadata=metadata)

    for group in annotations_groups:
        if not isinstance(group, dict):
            continue

        group_user = group.get("user")
        group_question = group.get("question")
        group_data = group.get("data")

        if not _matches_filter(group_user, user_email):
            continue
        if not _matches_filter(group_question, question):
            continue

        if not isinstance(group_question, str) or not group_question:
            # Skip invalid questions; they are required as DLUP labels.
            continue

        if group_data is None:
            continue

        # `parse_annotations` expects a string. We serialize non-string payloads as JSON,
        # but keep plain strings intact (json.dumps("x") would yield '"x"').
        if isinstance(group_data, str):
            annotation_str = group_data
        else:
            annotation_str = json.dumps(group_data)

        parsed_annotations = parse_annotations(
            annotation=annotation_str,
            label=group_question,
            box_as_polygon=box_as_polygon,
        )
        for parsed_annotation in parsed_annotations:
            if isinstance(parsed_annotation, SlideTag):
                tags.append(parsed_annotation)
            elif isinstance(parsed_annotation, Point):
                instance.add_point(parsed_annotation)
            elif parsed_annotation.label in roi_names:
                instance.add_roi(parsed_annotation)
            elif isinstance(parsed_annotation, Polygon):
                instance.add_polygon(parsed_annotation)
            elif isinstance(parsed_annotation, Box):
                instance.add_box(parsed_annotation)
            else:
                raise RuntimeError("Could not parse Slidescore JSON annotation.")

    instance._tags = tuple(tags)
    return instance

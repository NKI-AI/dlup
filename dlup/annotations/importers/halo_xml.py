# Copyright (c) dlup contributors
"""
Experimental annotations module for dlup.

"""
from __future__ import annotations

import errno
import os
import pathlib
import warnings
from typing import Optional, Type, TypeVar

from dlup import SlideImage
from dlup._types import PathLike
from dlup.annotations import AnnotationSorting, SlideAnnotations
from dlup.geometry import Box, GeometryCollection, Point, Polygon
from dlup.utils.imports import PYHALOXML_AVAILABLE

_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="SlideAnnotations")


def halo_xml_importer(
    cls: Type[_TSlideAnnotations],
    halo_xml: PathLike,
    scaling: float | None = None,
    sorting: AnnotationSorting | str = AnnotationSorting.NONE,
    box_as_polygon: bool = False,
    roi_names: Optional[list[str]] = None,
) -> _TSlideAnnotations:
    """
    Read annotations as a Halo [1] XML file.
    This function requires `pyhaloxml` [2] to be installed.

    Parameters
    ----------
    halo_xml : PathLike
        Path to the Halo XML file.
    scaling : float, optional
        The scaling to apply to the annotations.
    sorting: AnnotationSorting
        The sorting to apply to the annotations. Check the `AnnotationSorting` enum for more information. By default
        the annotations are not sorted as HALO supports hierarchical annotations.
    box_as_polygon : bool
        If True, rectangles are converted to polygons, and added as such.
        This is useful when the rectangles are actually implicitly bounding boxes.
    roi_names : list[str], optional
        List of names that should be considered as regions of interest. If set, these will be added as ROIs rather
        than polygons.

    References
    ----------
    .. [1] https://indicalab.com/halo/
    .. [2] https://github.com/rharkes/pyhaloxml

    Returns
    -------
    SlideAnnotations
    """
    path = pathlib.Path(halo_xml)
    roi_names = [] if roi_names is None else roi_names

    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))

    if not PYHALOXML_AVAILABLE:
        raise RuntimeError("`pyhaloxml` is not available. Install using `python -m pip install pyhaloxml`.")
    import pyhaloxml.shapely

    collection = GeometryCollection()
    with pyhaloxml.HaloXMLFile(halo_xml) as hx:
        hx.matchnegative()
        for layer in hx.layers:
            _color = layer.linecolor.rgb
            color = (_color[0], _color[1], _color[2])
            for region in layer.regions:
                if region.type == pyhaloxml.RegionType.Rectangle:
                    # The data is a CCW polygon, so the first and one to last coordinates are the coordinates
                    vertices = region.getvertices()
                    min_x = min(v[0] for v in vertices)
                    max_x = max(v[0] for v in vertices)
                    min_y = min(v[1] for v in vertices)
                    max_y = max(v[1] for v in vertices)
                    curr_box = Box((min_x, min_y), (max_x - min_x, max_y - min_y))

                    if box_as_polygon:
                        polygon = curr_box.as_polygon()
                        if polygon.label in roi_names:
                            collection.add_roi(polygon)
                        else:
                            collection.add_polygon(polygon)
                    else:
                        collection.add_box(curr_box)
                    continue

                elif region.type in [pyhaloxml.RegionType.Ellipse, pyhaloxml.RegionType.Polygon]:
                    polygon = Polygon(
                        region.getvertices(), [x.getvertices() for x in region.holes], label=layer.name, color=color
                    )
                    if polygon.label in roi_names:
                        collection.add_roi(polygon)
                    else:
                        collection.add_polygon(polygon)
                elif region.type == pyhaloxml.RegionType.Pin:
                    point = Point(*(region.getvertices()[0]), label=layer.name, color=color)
                    collection.add_point(point)
                elif region.type == pyhaloxml.RegionType.Ruler:
                    warnings.warn(
                        f"Ruler annotations are not supported. Annotation {layer.name} will be skipped",
                        UserWarning,
                    )
                    continue
                else:
                    raise NotImplementedError(f"Regiontype {region.type} is not implemented in dlup")

    SlideAnnotations._in_place_sort_and_scale(collection, scaling, sorting)

    def offset_function(slide: "SlideImage") -> tuple[int, int]:
        return (
            slide.slide_bounds[0][0] - slide.slide_bounds[0][0] % 256,
            slide.slide_bounds[0][1] - slide.slide_bounds[0][1] % 256,
        )

    return cls(collection, tags=None, sorting=sorting, offset_function=offset_function)

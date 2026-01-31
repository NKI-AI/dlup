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
Annotations module for dlup.

"""

from __future__ import annotations

import copy
import importlib
import warnings
from enum import Enum
from typing import Any, Callable, Iterable, Optional, cast

import numpy as np
import numpy.typing as npt
from dlup._geometry import AnnotationRegion  # pylint: disable=no-name-in-module
from dlup._types import GenericNumber
from dlup.annotations.tags import SlideTag
from dlup.geometry import GeometryCollection, Point, Polygon


class SlideAnnotationsView:
    """Represents an annotations view at a specific scaling or MPP level.

    This class provides a convenient interface for reading annotation regions at a fixed
    resolution without having to repeatedly specify the scaling parameter.

    Examples
    --------
    >>> from dlup.annotations import SlideAnnotations
    >>> annotations = SlideAnnotations.from_geojson("annotations.json")
    >>> # Create a view at 0.5 microns per pixel
    >>> view = annotations.get_view_at_mpp(0.5)
    >>> # Read a region at this MPP level
    >>> region = view.read_region((0, 0), (512, 512))
    >>> # Or create a view at a specific scaling
    >>> view = annotations.get_view_at_scaling(0.25)
    >>> region = view.read_region((100, 100), (256, 256))
    """

    def __init__(
        self,
        annotations: "SlideAnnotations",
        scaling: float,
    ):
        """Initialize with a SlideAnnotations object and the scaling level.

        Parameters
        ----------
        annotations : SlideAnnotations
            The parent annotations object.
        scaling : float
            The scaling factor relative to the base level.
        """
        self._annotations = annotations
        self._scaling = scaling

    @property
    def mpp(self) -> Optional[float]:
        """Returns the effective microns per pixel at this view's scaling level."""
        if self._annotations.mpp is None:
            return None
        return self._annotations.mpp / self._scaling

    @property
    def size(self) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        """Returns the bounding box at this view's scaling level."""
        return self._annotations.bounding_box_at_scaling(self._scaling)

    def read_region(
        self,
        coordinates: tuple[GenericNumber, GenericNumber],
        size: tuple[GenericNumber, GenericNumber],
    ) -> AnnotationRegion:
        """Read an annotation region at this view's scaling level.

        Parameters
        ----------
        coordinates : tuple[GenericNumber, GenericNumber]
            Top-left coordinates of the region in the view's coordinate system.
        size : tuple[GenericNumber, GenericNumber]
            Size of the region.

        Returns
        -------
        AnnotationRegion
            The annotation region at the specified location and size.
        """
        return self._annotations.read_region(coordinates, self._scaling, size)


class AnnotationSorting(str, Enum):
    """The ways to sort the annotations. This is used in the constructors of the `SlideAnnotations` class, and applied
    to the output of `SlideAnnotations.read_region()`.

    - REVERSE: Sort the output in reverse order.
    - AREA: Often when the annotation tools do not properly support hierarchical order, one would annotate in a way
        that the smaller objects are on top of the larger objects. This option sorts the output by area, so that the
        larger objects appear first in the output and then the smaller objects.
    - Z_INDEX: Sort the output by the z-index of the annotations. This is useful when the annotations have a z-index
    - NONE: Do not apply any sorting and output as is presented in the input file.
    """

    REVERSE = "REVERSE"
    AREA = "AREA"
    Z_INDEX = "Z_INDEX"
    NONE = "NONE"

    def to_sorting_params(self) -> Any:
        """Get the sorting parameters for the annotation sorting."""
        if self == AnnotationSorting.REVERSE:
            return lambda x: None, True

        if self == AnnotationSorting.AREA:
            return lambda x: x.area, True

        if self == AnnotationSorting.Z_INDEX:
            return lambda x: x.get_field("z_index"), False


class SlideAnnotations(GeometryCollection):
    """Class that holds all annotations for a specific image"""

    _dynamic_methods: set[str] = set()  # Track dynamically registered methods

    def __init__(
        self,
        tags: Optional[tuple[SlideTag, ...]] = None,
        sorting: Optional[AnnotationSorting | str] = None,
        mpp: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        tags: Optional[tuple[SlideTag, ...]]
            A tuple of image-level tags such as staining quality
        sorting: AnnotationSorting
            Sorting method, see `AnnotationSorting`. This value is typically passed to the constructor
            because of operations layer on (such as `__add__`). Typically the classmethod already sorts the data
        mpp: Optional[float]
            The microns per pixel for the annotations at the base level
        **kwargs: Any
            Additional keyword arguments. In this class they are used for additional metadata or offset functions.
            Currently only HaloXML requires offsets. See `.from_halo_xml` for an example
        """
        super().__init__()
        self._tags = tags
        self._sorting = sorting
        self._mpp = mpp
        self._offset_function: bool = bool(kwargs.get("offset_function", False))
        self._metadata: Optional[dict[str, list[str] | str | int | float | bool]] = kwargs.get("metadata", None)

    @classmethod
    def register_importer(cls, module_name: str, name: str) -> None:
        """Register an importer function dynamically as a class method."""
        method_name = f"from_{name}"

        if hasattr(cls, method_name):
            raise ValueError(f"Method `{method_name}` already registered.")

        try:
            module = importlib.import_module(module_name, package=__name__)
            func = getattr(module, f"{name}_importer")

            # Register the importer as a class method
            setattr(cls, method_name, classmethod(func))
            func.__doc__ = f"Import annotations from {name} format."
            cls._dynamic_methods.add(method_name)
        except ModuleNotFoundError:
            warnings.warn(
                f"Module {module_name} not found. Cannot register importer. The function `SlideAnnotations.from_{name}` will not be available."
            )

    @classmethod
    def register_exporter(cls, module_name: str, name: str) -> None:
        """Register an exporter function dynamically as an instance method."""
        method_name = f"as_{name}"

        if hasattr(cls, method_name):
            raise ValueError(f"Method `{method_name}` already registered.")

        if not module_name:
            raise ValueError(f"No internal exporter found for '{name}'")

        try:
            module = importlib.import_module(module_name, package=__name__)
            func = getattr(module, f"{name}_exporter")
            # Register the exporter as an instance method
            setattr(cls, method_name, func)
            func.__doc__ = f"Export annotations from {name} format."
            cls._dynamic_methods.add(method_name)
        except ModuleNotFoundError:
            warnings.warn(
                f"Module {module_name} not found. Cannot register exporter. The function `SlideAnnotations.as_{name}` will not be available."
            )

    @classmethod
    def from_file_path(cls, file_path: str, reader: str, *args: Any, **kwargs: Any) -> SlideAnnotations:
        """
        Load annotations from a file using the specified reader.

        Parameters
        ----------
        file_path : str
            The path to the file from which annotations should be loaded.
        reader : str
            The name of the reader to use, such as 'geojson', 'halo_xml', etc.
        *args, **kwargs:
            Additional arguments passed to the specific reader method.

        Returns
        -------
        SlideAnnotations
            The loaded annotations.

        Raises
        ------
        ValueError
            If the specified reader is not registered.
        """
        method_name = f"from_{reader}"

        if not hasattr(cls, method_name):
            raise ValueError(f"Reader '{reader}' is not registered.")

        reader_method = getattr(cls, method_name)
        return cast("SlideAnnotations", reader_method(file_path, *args, **kwargs))

    @property
    def sorting(self) -> Optional[AnnotationSorting | str]:
        return self._sorting

    @property
    def tags(self) -> Optional[tuple[SlideTag, ...]]:
        return self._tags

    @property
    def mpp(self) -> Optional[float]:
        """Returns the microns per pixel for the annotations if set."""
        return self._mpp

    @mpp.setter
    def mpp(self, value: Optional[float]) -> None:
        """Set the microns per pixel for the annotations."""
        self._mpp = value

    @property
    def num_polygons(self) -> int:
        return len(self.polygons)

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    @property
    def metadata(self) -> dict[str, list[str] | str | int | float | bool]:
        """Additional metadata for the annotations"""
        return {} if self._metadata is None else self._metadata

    @metadata.setter
    def metadata(self, metadata: dict[str, list[str] | str | int | float | bool]) -> None:
        self._metadata = metadata

    @property
    def offset_function(self) -> Any:
        """
        In some cases a function needs to be applied to the coordinates which cannot be handled in this class as
        it might require additional information. This function will be applied to the coordinates of all annotations.
        This is useful from a file format which requires this, for instance HaloXML.

        Example
        -------
        For HaloXML this is `offset = slide.slide_bounds[0] - slide.slide_bounds[0] % 256`.
        >>> slide = Slide.from_file_path("image.svs")
        >>> ann = SlideAnnotations.from_halo_xml("annotations.xml")
        >>> assert ann.offset_function == lambda slide: slide.slide_bounds[0] - slide.slide_bounds[0] % 256
        >>> ann.set_offset(annotation.offset_function(slide))

        Returns
        -------
        Callable

        """
        return self._offset_function

    @offset_function.setter
    def offset_function(self, func: Any) -> None:
        self._offset_function = func

    def _in_place_sort_and_scale(self, scaling: Optional[float], sorting: Optional[AnnotationSorting | str]) -> None:
        if sorting == "REVERSE":
            raise NotImplementedError("This doesn't work for now.")

        if scaling != 1.0 and scaling is not None:
            self.scale(scaling)
        if sorting == AnnotationSorting.NONE or sorting is None:
            return
        if isinstance(sorting, str):
            key, reverse = AnnotationSorting[sorting].to_sorting_params()
        else:
            key, reverse = sorting.to_sorting_params()
        self.sort_polygons(key, reverse)

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get the bounding box of the annotations combining points and polygons.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            The bounding box of the annotations.

        """
        return super().bounding_box

    def bounding_box_at_scaling(self, scaling: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get the bounding box of the annotations at a specific scaling factor.

        Parameters
        ----------
        scaling : float
            The scaling factor to apply to the annotations.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            The bounding box of the annotations at the specific scaling factor.

        """
        bbox = self.bounding_box
        return ((bbox[0][0] * scaling, bbox[0][1] * scaling), (bbox[1][0] * scaling, bbox[1][1] * scaling))

    def rois_at_scaling(self, scaling: float) -> Optional[list[Polygon]]:
        """Get the regions of interest (ROIs) for the annotations at a specific scaling factor.

        Parameters
        ----------
        scaling : float
            The scaling factor to apply to the annotations.

        Returns
        -------
        list[Polygon], optional
            The regions of interest (ROIs) for the annotations at the specific scaling factor.

        """
        if not self.has_rois:
            return None

        out_polygons = []
        for roi in self.rois:
            out_roi = roi.__copy__()
            out_roi.scale(scaling)
            out_polygons.append(out_roi)

        return out_polygons

    def simplify(self, tolerance: float) -> None:
        """Simplify the polygons in the annotation (i.e. reduce points). Other annotations will remain unchanged.
        All points in the resulting polygons object will be in the tolerance distance of the original polygon.

        Parameters
        ----------
        tolerance : float
            The tolerance to simplify the polygons with.
        Returns
        -------
        None
        """
        self.simplify(tolerance)

    def __contains__(self, item: str | Point | Polygon) -> bool:
        if isinstance(item, str):
            return item in self.available_classes
        if isinstance(item, Point):
            return item in self.points
        if isinstance(item, Polygon):
            return item in self.polygons

        return False

    @property
    def available_classes(self) -> set[str]:
        """Get the available classes in the annotations.

        Returns
        -------
        set[str]
            The available classes in the annotations.

        """
        available_classes = set()
        for polygon in self.polygons:
            if polygon.label is not None:
                available_classes.add(polygon.label)
        for point in self.points:
            if point.label is not None:
                available_classes.add(point.label)
        for box in self.boxes:
            if box.label is not None:
                available_classes.add(box.label)

        return available_classes

    def __iter__(self) -> Iterable[Polygon | Point]:
        # First returns all the polygons then all points
        for polygon in self.polygons:
            yield polygon

        for point in self.points:
            yield point

    def __add__(self, other: Any) -> "SlideAnnotations":
        """
        Add two annotations together. This will return a new `SlideAnnotations` object with the annotations combined.

        Notes
        -----
        - The polygons and points are shared between the objects. This means that if you modify the polygons or points
        in the new object, the original objects will also be modified. If you wish to avoid this, you must add two
        copies together.
        - Note that the sorting is not applied to this object. You can apply this by calling `sort_polygons()` on
        the resulting object.

        Parameters
        ----------
        other : SlideAnnotations, Point, Polygon, or list
            The other annotations or geometric objects to add.

        Returns
        -------
        SlideAnnotations
            A new `SlideAnnotations` object with the combined annotations.

        Raises
        ------
        TypeError
            If the other object is not of a compatible type.
        """
        if not isinstance(other, (SlideAnnotations, Point, Polygon, list)):
            raise TypeError(f"Unsupported type {type(other)}")

        # Start with a deep copy of the current instance
        result = copy.deepcopy(self)

        if isinstance(other, SlideAnnotations):
            if self.sorting != other.sorting:
                raise TypeError("Cannot add annotations with different sorting.")
            if self.offset_function != other.offset_function:
                raise TypeError(
                    "Cannot add annotations with different requirements for offsetting to slide bounds "
                    "(`offset_function`)."
                )

            # Combine tags
            if other.tags:
                result._tags = tuple(set(result.tags or ()) | set(other.tags))

            # Add polygons, points, boxes, and ROIs from the other instance
            for polygon in other.polygons:
                result.add_polygon(polygon)
            for point in other.points:
                result.add_point(point)
            for box in other.boxes:
                result.add_box(box)
            for roi in other.rois:
                result.add_roi(roi)

        elif isinstance(other, (Point, Polygon)):
            result.add_polygon(other) if isinstance(other, Polygon) else result.add_point(other)

        elif isinstance(other, list):
            if not all(isinstance(item, (Point, Polygon)) for item in other):
                raise TypeError(
                    f"Can only add a list containing Point and Polygon objects to {self.__class__.__name__}"
                )
            for item in other:
                result.add_polygon(item) if isinstance(item, Polygon) else result.add_point(item)

        return result

    def __iadd__(self, other: Any) -> "SlideAnnotations":
        if isinstance(other, (Point, Polygon)):
            other = [other]

        if isinstance(other, list):
            if not all(isinstance(item, (Point, Polygon)) for item in other):
                raise TypeError(
                    f"can only add list purely containing Point and Polygon objects {self.__class__.__name__}"
                )

            for item in other:
                if isinstance(item, Polygon):
                    self.add_polygon(copy.deepcopy(item))
                elif isinstance(item, Point):
                    self.add_point(copy.deepcopy(item))

        elif isinstance(other, SlideAnnotations):
            if self.sorting != other.sorting or self.offset_function != other.offset_function:
                raise ValueError(
                    f"Both sorting and offset_function must be the same to add {self.__class__.__name__}s together."
                )

            if self._tags is None:
                self._tags = other._tags
            elif other._tags is not None:
                assert self
                self._tags += other._tags

            for polygon in other.polygons:
                self.add_polygon(copy.deepcopy(polygon))
            for point in other.points:
                self.add_point(copy.deepcopy(point))
            for box in other.boxes:
                self.add_box(copy.deepcopy(box))
            for roi in other.rois:
                self.add_roi(copy.deepcopy(roi))
        else:
            return NotImplemented
        self._in_place_sort_and_scale(None, self.sorting)

        return self

    def __radd__(self, other: Any) -> "SlideAnnotations":
        # in-place addition (+=) of Point and Polygon will raise a TypeError
        if not isinstance(other, (SlideAnnotations, Point, Polygon, list)):
            raise TypeError(f"Unsupported type {type(other)}")
        if isinstance(other, list):
            if not all(isinstance(item, (Polygon, Point)) for item in other):
                raise TypeError(
                    f"can only add list purely containing Point and Polygon objects to {self.__class__.__name__}"
                )
            raise TypeError(
                "use the __add__ or __iadd__ operator instead of __radd__ when working with lists to avoid \
                            unexpected behavior."
            )
        return self + other

    def __sub__(self, other: Any) -> "SlideAnnotations":
        return NotImplemented

    def __isub__(self, other: Any) -> "SlideAnnotations":
        return NotImplemented

    def __rsub__(self, other: Any) -> "SlideAnnotations":
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SlideAnnotations):
            return False

        our_sorting = self._sorting if self._sorting != AnnotationSorting.NONE else None
        other_sorting = other._sorting if other._sorting != AnnotationSorting.NONE else None

        if our_sorting != other_sorting:
            return False

        if self._tags != other._tags:
            return False

        if self.rtree_invalidated != other.rtree_invalidated:
            return False

        if not super().__eq__(other):
            return False

        if self._offset_function != other._offset_function:
            return False

        return True

    def __getstate__(self) -> dict[str, Any]:
        """Return the state for pickling."""
        # Get the C++ GeometryCollection state
        base_state: dict[str, Any] = super().__getstate__()  # type: ignore

        # Create our complete state dictionary in one go
        state = {
            # Include base class state fields
            **base_state,
            # Add our class-specific state
            "_dynamic_methods": sorted(list(self._dynamic_methods)),
            "_tags": self._tags,
            "_sorting": self._sorting,
            "_mpp": self._mpp,
            "_offset_function": self._offset_function,
            "_metadata": self._metadata,
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the state from pickling."""
        # Extract C++ GeometryCollection state fields
        cpp_state = {
            "polygons": state.get("polygons", []),
            "points": state.get("points", []),
            "boxes": state.get("boxes", []),
            "rois": state.get("rois", []),
            "rtree_invalidated": state.get("rtree_invalidated", True),
        }

        # Restore the C++ GeometryCollection state
        super().__setstate__(cpp_state)  # type: ignore

        # Restore Python-specific fields from SlideAnnotations
        self._dynamic_methods = set(state.get("_dynamic_methods", []))
        self._tags = state.get("_tags")
        self._sorting = state.get("_sorting")
        self._mpp = state.get("_mpp")
        self._offset_function = bool(state.get("_offset_function", False))
        self._metadata = state.get("_metadata")

    def read_region(
        self,
        coordinates: tuple[GenericNumber, GenericNumber],
        scaling: float,
        size: tuple[GenericNumber, GenericNumber],
    ) -> AnnotationRegion:
        """Reads the region of the annotations. Function signature is the same as `dlup.SlideImage`
        so they can be used in conjunction.

        The process is as follows:

        1.  All the annotations which overlap with the requested region of interest are filtered
        2.  The polygons in the annotations in are cropped.
            The boxes and points are only filtered, so it's possible the boxes have negative (x, y) values
        3.  The annotation is rescaled and shifted to the origin to match the local patch coordinate system.

        The final returned data is a `dlup.geometry.AnnotationRegion`.

        Parameters
        ----------
        location: tuple[GenericNumber, GenericNumber]
            Top-left coordinates of the region in the requested scaling
        size : tuple[GenericNumber, GenericNumber]
            Output size of the region
        scaling : float
            Scaling to apply compared to the base level

        Returns
        -------
        AnnotationRegion

        Examples
        --------
        1. To read geojson annotations and convert them into masks:

        >>> from pathlib import Path
        >>> from dlup import SlideImage
        >>> import numpy as np
        >>> wsi = SlideImage.from_file_path(Path("path/to/image.svs"))
        >>> view = wsi.get_view_at_scaling(scaling=0.5)
        >>> wsi = view.read_region(location=(0,0), size=view.size)
        >>> annotations = SlideAnnotations.from_geojson("path/to/geojson.json")
        >>> region = annotations.read_region((0,0), 0.01, wsi.size)
        >>> mask = region.to_mask()
        >>> color_mask = annotations.color_lut[mask]
        >>> polygons = region.polygons.get_geometries()  # This is a list of `dlup.geometry.Polygon` objects
        """
        region = super().read_region(coordinates, scaling, size)
        return region

    def get_view_at_mpp(self, mpp: float) -> SlideAnnotationsView:
        """Returns a SlideAnnotationsView at a specific MPP (microns per pixel) level.

        This provides a more ergonomic API for reading annotation regions at a fixed resolution.
        Once you have a view, you can call `read_region(location, size)` without
        having to repeatedly specify the MPP or scaling.

        Parameters
        ----------
        mpp : float
            The microns per pixel for the view.

        Returns
        -------
        SlideAnnotationsView
            A view at the specified MPP level.

        Raises
        ------
        ValueError
            If the annotations don't have an MPP set.

        Examples
        --------
        >>> annotations = SlideAnnotations.from_geojson("annotations.json")
        >>> annotations.mpp = 0.25  # Set the base MPP
        >>> view = annotations.get_view_at_mpp(0.5)
        >>> region = view.read_region((0, 0), (512, 512))
        """
        if self.mpp is None:
            raise ValueError(
                "Cannot create MPP-based view: annotations have no MPP set. Use get_view_at_scaling() instead."
            )
        scaling = self.mpp / mpp
        return SlideAnnotationsView(self, scaling)

    def get_view_at_scaling(self, scaling: float) -> SlideAnnotationsView:
        """Returns a SlideAnnotationsView at a specific scaling level.

        This provides a more ergonomic API for reading annotation regions at a fixed resolution.
        Once you have a view, you can call `read_region(location, size)` without
        having to repeatedly specify the scaling.

        Parameters
        ----------
        scaling : float
            The scaling factor relative to the base level (e.g., 0.5 means half the resolution).

        Returns
        -------
        SlideAnnotationsView
            A view at the specified scaling level.

        Examples
        --------
        >>> annotations = SlideAnnotations.from_geojson("annotations.json")
        >>> view = annotations.get_view_at_scaling(0.25)
        >>> region = view.read_region((100, 100), (256, 256))
        """
        return SlideAnnotationsView(self, scaling)

    def scale(self, scaling: float) -> None:
        """
        Scale the annotations by a multiplication factor.
        This operation will be performed in-place.

        Parameters
        ----------
        scaling : float
            The scaling factor to apply to the annotations.

        Notes
        -----
        This invalidates the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or have the function
        `read_region()` do it for you on-demand.

        Returns
        -------
        None
        """
        super().scale(scaling)

    def set_offset(self, offset: tuple[float, float]) -> None:
        """Set the offset for the annotations. This operation will be performed in-place.

        For example, if the offset is 1, 1, the annotations will be moved by 1 unit in the x and y direction.

        Parameters
        ----------
        offset : tuple[float, float]
            The offset to apply to the annotations.

        Notes
        -----
        This invalidates the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or have the function
        `read_region()` do it for you on-demand.

        Returns
        -------
        None
        """
        super().set_offset(offset)

    def reindex_polygons(self, index_map: dict[str, int]) -> None:
        """
        Reindex the polygons in the annotations. This operation will be performed in-place.
        This is useful if you want to change the index of the polygons in the annotations.

        This requires that the `.label` property on the polygons is set.

        Parameters
        ----------
        index_map : dict[str, int]
            A dictionary that maps the label to the new index.

        Returns
        -------
        None
        """
        super().reindex_polygons(index_map)

    @property
    def index_map(self) -> dict[str, int]:
        """Get the index map for the annotations.

        Returns
        -------
        dict[str, int]
            The index map for the annotations.

        """
        return super().index_map

    @index_map.setter
    def index_map(self, index_map: dict[str, int]) -> None:
        """Set the index map for the annotations.

        Parameters
        ----------
        index_map : dict[str, int]
            The index map for the annotations.

        Returns
        -------
        None

        """
        self.reindex_polygons(index_map)

    def relabel_polygons(self, relabel_map: dict[str, str]) -> None:
        """
        Relabel the polygons in the annotations. This operation will be performed in-place.

        Parameters
        ----------
        relabel_map : dict[str, str]
            A dictionary that maps the label to the new label.

        Returns
        -------
        None
        """
        # TODO: Implement in C++
        for polygon in self.polygons:
            if not polygon.label:
                continue
            if polygon.label in relabel_map:
                polygon.label = relabel_map[polygon.label]

    def filter_polygons(self, label: str) -> None:
        """Filter polygons in-place.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Parameters
        ----------
        label : str
            The label to filter.

        """
        for polygon in self.polygons:
            if polygon.label == label:
                self.remove_polygon(polygon)

    def filter_points(self, label: str) -> None:
        """Filter points in-place.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Parameters
        ----------
        label : str
            The label to filter.

        """
        for point in self.points:
            if point.label == label:
                self.remove_point(point)

    def filter_boxes(self, label: str) -> None:
        """Filter boxes in-place.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Parameters
        ----------
        label : str
            The label to filter.

        """
        for box in self.boxes:
            if box.label == label:
                self.remove_box(box)

    def filter(self, label: str) -> None:
        """Filter annotations in-place.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Parameters
        ----------
        label : str
            The label to filter.

        """
        self.filter_polygons(label)
        self.filter_points(label)
        self.filter_boxes(label)

    def sort_polygons(self, key: Callable[[Polygon], int | float | str | None], reverse: bool = False) -> None:
        """Sort the polygons in-place.

        Parameters
        ----------
        key : callable
            The key to sort the polygons on, this has to be a lambda function or similar.
            For instance `lambda polygon: polygon.area` will sort the polygons on the area, or
            `lambda polygon: polygon.get_field(field_name)` will sort the polygons on that field.
        reverse : bool
            Whether to sort in reverse order.

        Note
        ----
        This will internally invalidate the R-tree. You could rebuild this manually using `.rebuild_rtree()`, or
        have the function itself do this on-demand (typically when you invoke a `.read_region()`)

        Returns
        -------
        None

        """
        super().sort_polygons(key, reverse)

    @property
    def color_lut(self) -> npt.NDArray[np.uint8]:
        """Get the color lookup table for the annotations.

        Requires that the polygons have an index and color set. Be aware that for the background always
        the value 0 is assumed.
        So if you are using the `to_mask(default_value=0)` with a default value other than 0,
        the LUT will still have this as index 0.

        Example
        -------
        >>> color_lut = annotations.color_lut
        >>> region = annotations.read_region(region_start, 0.02, region_size).to_mask()
        >>> colored_mask = PIL.Image.fromarray(color_lut[mask])

        Returns
        -------
        np.ndarray
            The color lookup table.

        """
        return super().color_lut

    def __copy__(self) -> "SlideAnnotations":
        """Create a shallow copy of SlideAnnotations."""
        # Create a new SlideAnnotations instance with copied attributes
        new_instance = self.__class__(
            tags=copy.copy(self._tags),
            sorting=self._sorting,
            mpp=self._mpp,
            offset_function=self._offset_function,
            metadata=copy.copy(self._metadata),
        )

        for polygon in self.polygons:
            new_instance.add_polygon(polygon)
        for point in self.points:
            new_instance.add_point(point)
        for box in self.boxes:
            new_instance.add_box(box)

        return new_instance

    def __deepcopy__(self, memo: dict[int, Any]) -> "SlideAnnotations":
        """Create a deep copy of SlideAnnotations."""
        # Create a new SlideAnnotations instance with copied attributes
        new_instance = self.__class__(
            tags=copy.deepcopy(self._tags, memo),
            sorting=self._sorting,
            mpp=self._mpp,
            offset_function=self._offset_function,
            metadata=copy.deepcopy(self._metadata, memo),
        )

        for polygon in self.polygons:
            new_instance.add_polygon(polygon.clone())
        for point in self.points:
            new_instance.add_point(point.clone())
        for box in self.boxes:
            new_instance.add_box(box.clone())

        return new_instance

    def copy(self) -> "SlideAnnotations":
        """Return a shallow copy of the SlideAnnotations."""
        return self.__copy__()

    def __dir__(self) -> Any:
        """Override `__dir__` to include dynamically registered methods."""
        yield from self._dynamic_methods
        yield super().__dir__()


# TODO: These exporters/importers can be class based, would make registration more convenient.
SlideAnnotations.register_importer(".importers.geojson", "geojson")
SlideAnnotations.register_importer(".importers.halo_xml", "halo_xml")
SlideAnnotations.register_importer(".importers.asap_xml", "asap_xml")
SlideAnnotations.register_importer(".importers.dlup_xml", "dlup_xml")
SlideAnnotations.register_importer(".importers.darwin_json", "darwin_json")
SlideAnnotations.register_importer(".importers.slidescore_tsv", "slidescore_tsv")

SlideAnnotations.register_exporter(".exporters.geojson", "geojson")
SlideAnnotations.register_exporter(".exporters.dlup_xml", "dlup_xml")
SlideAnnotations.register_exporter(".exporters.slidescore_tsv", "slidescore_tsv")

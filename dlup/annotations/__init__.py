# Copyright (c) dlup contributors
"""
Experimental annotations module for dlup.

"""
from __future__ import annotations

import copy
import importlib
from enum import Enum
from typing import Any, Callable, Iterable, Optional, TypeVar

import numpy as np
import numpy.typing as npt

from dlup._geometry import AnnotationRegion  # pylint: disable=no-name-in-module
from dlup._types import GenericNumber
from dlup.annotations.tags import SlideTag
from dlup.geometry import GeometryCollection, Point, Polygon

_TSlideAnnotations = TypeVar("_TSlideAnnotations", bound="SlideAnnotations")


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
            return lambda x: x.area, False

        if self == AnnotationSorting.Z_INDEX:
            return lambda x: x.get_field("z_index"), False


class SlideAnnotations:
    """Class that holds all annotations for a specific image"""

    def __init__(
        self,
        layers: GeometryCollection,
        tags: Optional[tuple[SlideTag, ...]] = None,
        sorting: Optional[AnnotationSorting | str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        layers : GeometryCollection
            Geometry collection containing the polygons, boxes and points
        tags: Optional[tuple[SlideTag, ...]]
            A tuple of image-level tags such as staining quality
        sorting: AnnotationSorting
            Sorting method, see `AnnotationSorting`. This value is typically passed to the constructor
            because of operations layer on (such as `__add__`). Typically the classmethod already sorts the data
        **kwargs: Any
            Additional keyword arguments. In this class they are used for additional metadata or offset functions.
            Currently only HaloXML requires offsets. See `.from_halo_xml` for an example
        """
        self._layers = layers
        self._tags = tags
        self._sorting = sorting
        self._offset_function: bool = bool(kwargs.get("offset_function", False))
        self._metadata: Optional[dict[str, list[str] | str | int | float | bool]] = kwargs.get("metadata", None)

    @classmethod
    def register_importer(cls, func: Optional[Callable[..., Any]], name: str) -> None:
        """Register an importer function dynamically as a class method."""
        method_name = f"from_{name}"

        if hasattr(cls, method_name):
            raise ValueError(f"Method `{method_name}` already registered.")

        # If no function is provided, assume it's an internal importer and load it dynamically
        if func is None:
            module_name = {
                "geojson": ".importers.geojson",
                "halo_xml": ".importers.halo_xml",
                "asap_xml": ".importers.asap_xml",
                "dlup_xml": ".importers.dlup_xml",
                "darwin_json": ".importers.darwin_json",
            }.get(name)

            if not module_name:
                raise ValueError(f"No internal importer found for '{name}'")

            module = importlib.import_module(module_name, package=__name__)
            func = getattr(module, f"{name}_importer")

        # Register the importer as a class method
        setattr(cls, method_name, classmethod(func))

    @classmethod
    def register_exporter(cls, func: Optional[Callable[..., Any]], name: str) -> None:
        """Register an exporter function dynamically as an instance method."""
        method_name = f"as_{name}"

        if hasattr(cls, method_name):
            raise ValueError(f"Method `{method_name}` already registered.")

        # If no function is provided, assume it's an internal exporter and load it dynamically
        if func is None:
            module_name = {
                "geojson": ".exporters.geojson",
                "dlup_xml": ".exporters.dlup_xml",
            }.get(name)

            if not module_name:
                raise ValueError(f"No internal exporter found for '{name}'")

            module = importlib.import_module(module_name, package=__name__)
            func = getattr(module, f"{name}_exporter")

        # Register the exporter as an instance method
        setattr(cls, method_name, func)

    @property
    def sorting(self) -> Optional[AnnotationSorting | str]:
        return self._sorting

    @property
    def tags(self) -> Optional[tuple[SlideTag, ...]]:
        return self._tags

    @property
    def num_polygons(self) -> int:
        return len(self.layers.polygons)

    @property
    def num_points(self) -> int:
        return len(self.layers.points)

    @property
    def num_boxes(self) -> int:
        return len(self.layers.boxes)

    @property
    def metadata(self) -> Optional[dict[str, list[str] | str | int | float | bool]]:
        """Additional metadata for the annotations"""
        return self._metadata

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

    @property
    def layers(self) -> GeometryCollection:
        """
        Get the layers of the annotations.
        This is a GeometryCollection object which contains all the polygons and points
        """
        return self._layers

    @staticmethod
    def _in_place_sort_and_scale(
        collection: GeometryCollection, scaling: Optional[float], sorting: Optional[AnnotationSorting | str]
    ) -> None:
        if sorting == "REVERSE":
            raise NotImplementedError("This doesn't work for now.")

        if scaling != 1.0 and scaling is not None:
            collection.scale(scaling)
        if sorting == AnnotationSorting.NONE or sorting is None:
            return
        if isinstance(sorting, str):
            key, reverse = AnnotationSorting[sorting].to_sorting_params()
        else:
            key, reverse = sorting.to_sorting_params()
        collection.sort_polygons(key, reverse)

    @property
    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get the bounding box of the annotations combining points and polygons.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            The bounding box of the annotations.

        """
        return self._layers.bounding_box

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
        self._layers.simplify(tolerance)

    def __contains__(self, item: str | Point | Polygon) -> bool:
        if isinstance(item, str):
            return item in self.available_classes
        if isinstance(item, Point):
            return item in self.layers.points
        if isinstance(item, Polygon):
            return item in self.layers.polygons

        return False

    def __len__(self) -> int:
        return self._layers.size()

    @property
    def available_classes(self) -> set[str]:
        """Get the available classes in the annotations.

        Returns
        -------
        set[str]
            The available classes in the annotations.

        """
        available_classes = set()
        for polygon in self.layers.polygons:
            if polygon.label is not None:
                available_classes.add(polygon.label)
        for point in self.layers.points:
            if point.label is not None:
                available_classes.add(point.label)

        return available_classes

    def __iter__(self) -> Iterable[Polygon | Point]:
        # First returns all the polygons then all points
        for polygon in self.layers.polygons:
            yield polygon

        for point in self.layers.points:
            yield point

    def __add__(self, other: Any) -> "SlideAnnotations":
        """
        Add two annotations together. This will return a new `SlideAnnotations` object with the annotations combined.
        The polygons will be added from left to right followed the points from left to right.

        Notes
        -----
        - The polygons and points are shared between the objects. This means that if you modify the polygons or points
          in the new object, the original objects will also be modified. If you wish to avoid this, you must add two
          copies together.
        - Note that the sorting is not applied to this object. You can apply this by calling `sort_polygons()` on
        the resulting object.

        Parameters
        ----------
        other : SlideAnnotations
            The other annotations to add.

        """
        if not isinstance(other, (SlideAnnotations, Point, Polygon, list)):
            raise TypeError(f"Unsupported type {type(other)}")

        if isinstance(other, SlideAnnotations):
            if not self.sorting == other.sorting:
                raise TypeError("Cannot add annotations with different sorting.")
            if self.offset_function != other.offset_function:
                raise TypeError(
                    "Cannot add annotations with different requirements for offsetting to slide bounds "
                    "(`offset_function`)."
                )

            tags: tuple[SlideTag, ...] = ()
            if self.tags is None and other.tags is not None:
                tags = other.tags

            if other.tags is None and self.tags is not None:
                tags = self.tags

            if self.tags is not None and other.tags is not None:
                tags = tuple(set(self.tags + other.tags))

            # Let's add the annotations
            collection = GeometryCollection()
            for polygon in self.layers.polygons:
                collection.add_polygon(copy.deepcopy(polygon))
            for point in self.layers.points:
                collection.add_point(copy.deepcopy(point))

            for polygon in other.layers.polygons:
                collection.add_polygon(copy.deepcopy(polygon))
            for point in other.layers.points:
                collection.add_point(copy.deepcopy(point))

            SlideAnnotations._in_place_sort_and_scale(collection, None, self.sorting)
            return self.__class__(layers=collection, tags=tuple(tags) if tags else None, sorting=self.sorting)

        if isinstance(other, (Point, Polygon)):
            other = [other]

        if isinstance(other, list):
            if not all(isinstance(item, (Point, Polygon)) for item in other):
                raise TypeError(
                    f"can only add list purely containing Point and Polygon objects to {self.__class__.__name__}"
                )

            collection = copy.copy(self._layers)
            for item in other:
                if isinstance(item, Polygon):
                    collection.add_polygon(item)
                elif isinstance(item, Point):
                    collection.add_point(item)
            SlideAnnotations._in_place_sort_and_scale(collection, None, self.sorting)
            return self.__class__(layers=collection, tags=copy.copy(self._tags), sorting=self.sorting)

        raise ValueError(f"Unsupported type {type(other)}")

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
                    self._layers.add_polygon(copy.deepcopy(item))
                elif isinstance(item, Point):
                    self._layers.add_point(copy.deepcopy(item))

        elif isinstance(other, SlideAnnotations):
            if self.sorting != other.sorting or self.offset_function != other.offset_function:
                raise ValueError(
                    f"Both sorting and offset_function must be the same to " f"add {self.__class__.__name__}s together."
                )

            if self._tags is None:
                self._tags = other._tags
            elif other._tags is not None:
                assert self
                self._tags += other._tags

            for polygon in other.layers.polygons:
                self._layers.add_polygon(copy.deepcopy(polygon))
            for point in other.layers.points:
                self._layers.add_point(copy.deepcopy(point))
        else:
            return NotImplemented
        SlideAnnotations._in_place_sort_and_scale(self._layers, None, self.sorting)

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

        if self._layers != other._layers:
            return False

        if self._offset_function != other._offset_function:
            return False

        return True

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
        2.  The polygons in the GeometryContainer in `.layers` are cropped.
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
        >>> wsi = wsi.get_scaled_view(scaling=0.5)
        >>> wsi = wsi.read_region(location=(0,0), size=wsi.size)
        >>> annotations = SlideAnnotations.from_geojson("path/to/geojson.json")
        >>> region = annotations.read_region((0,0), 0.01, wsi.size)
        >>> mask = region.to_mask()
        >>> color_mask = annotations.color_lut[mask]
        >>> polygons = region.polygons.get_geometries()  # This is a list of `dlup.geometry.Polygon` objects
        """
        region = self._layers.read_region(coordinates, scaling, size)
        return region

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
        self._layers.scale(scaling)

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
        self._layers.set_offset(offset)

    def rebuild_rtree(self) -> None:
        """
        Rebuild the R-tree for the annotations. This operation will be performed in-place.
        The R-tree is used for fast spatial queries on the annotations and is invalidated when the annotations are
        modified. This function will rebuild the R-tree. Strictly speaking, this is not required as the R-tree will be
        rebuilt on-demand when you invoke a `read_region()`. You could however do this if you want to avoid
        the `read_region()` to do it for you the first time it runs.
        """

        self._layers.rebuild_rtree()

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
        self._layers.reindex_polygons(index_map)

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
        for polygon in self._layers.polygons:
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
        for polygon in self._layers.polygons:
            if polygon.label == label:
                self._layers.remove_polygon(polygon)

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
        for point in self._layers.points:
            if point.label == label:
                self._layers.remove_point(point)

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

    def sort_polygons(self, key: Callable[[Polygon], int | float | str], reverse: bool = False) -> None:
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
        self._layers.sort_polygons(key, reverse)

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
        return self._layers.color_lut

    def __copy__(self) -> "SlideAnnotations":
        return self.__class__(layers=copy.copy(self._layers), tags=copy.copy(self._tags))

    def __deepcopy__(self, memo: dict[int, Any]) -> "SlideAnnotations":
        return self.__class__(layers=copy.deepcopy(self._layers, memo), tags=copy.deepcopy(self._tags, memo))

    def copy(self) -> "SlideAnnotations":
        return self.__copy__()


SlideAnnotations.register_importer(None, "geojson")
SlideAnnotations.register_importer(None, "halo_xml")
SlideAnnotations.register_importer(None, "asap_xml")
SlideAnnotations.register_importer(None, "dlup_xml")
SlideAnnotations.register_importer(None, "darwin_json")

SlideAnnotations.register_exporter(None, "geojson")
SlideAnnotations.register_exporter(None, "dlup_xml")

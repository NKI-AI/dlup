# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2025 Jonas Teuwen. All Rights Reserved.
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
"""Module for geometric objects"""

import copy
import warnings
from typing import Any, Optional

import dlup._geometry as _dg
import numpy as np
import numpy.typing as npt

# This line needs to stay so you can import AnnotationRegion from here
from dlup._geometry import AnnotationRegion  # noqa
from dlup.utils.imports import SHAPELY_AVAILABLE

if SHAPELY_AVAILABLE:
    from shapely.geometry import Point as ShapelyPoint
    from shapely.geometry import Polygon as ShapelyPolygon

__all__ = [
    "AnnotationRegion",
    "Polygon",
    "Point",
    "Box",
    "GeometryCollection",
]


class _BaseGeometry:
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def set_field(self, name: str, value: Any) -> None:
        """Set a field on the geometry. This can be in arbitrary python object.

        Parameters
        ----------
        name : str
            The name of the field to set
        value : Any
            The value of the field

        Returns
        -------
        None
        """
        raise NotImplementedError

    if SHAPELY_AVAILABLE:

        @classmethod
        def from_shapely(self, shapely_geometry: ShapelyPoint | ShapelyPolygon) -> "_BaseGeometry":
            """Create a new instance of the geometry from a Shapely geometry

            Parameters
            ----------
            shapely_geometry : ShapelyPoint | ShapelyPolygon
                The Shapely geometry to convert

            Returns
            -------
            _BaseGeometry
                The new instance of the geometry
            """
            raise NotImplementedError

    else:
        warnings.warn("Shapely is not available, and the `from_shapely` method will not be available.")

    def get_field(self, name: str) -> Any:
        """Get a field from the geometry. This can be in arbitrary python object.

        Parameters
        ----------
        name : str
            The name of the field to get

        Returns
        -------
        Any
            The value of the field
        """
        raise NotImplementedError

    @property
    def fields(self) -> list[str]:
        raise NotImplementedError

    @property
    def wkt(self) -> str:
        raise NotImplementedError

    @property
    def label(self) -> Optional[str]:
        field = self.get_field("label")
        if field is None:
            return None
        assert isinstance(field, str)
        return field

    @label.setter
    def label(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError(f"Label must be a string, got {type(value)}")
        self.set_field("label", value)

    @property
    def index(self) -> Optional[int]:
        field = self.get_field("index")
        if field is None:
            return None
        if not isinstance(field, int):
            raise ValueError(f"Index must be an integer, got {type(field)}")
        assert isinstance(field, int)
        return field

    @index.setter
    def index(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError(f"Index must be an integer, got {type(value)}")
        self.set_field("index", value)

    @property
    def color(self) -> Optional[tuple[int, int, int]]:
        field = self.get_field("color")
        if field is None:
            return None
        if not isinstance(field, tuple) or len(field) != 3:
            raise ValueError(f"Color must be an RGB tuple, got {type(field)}")
        assert isinstance(field, tuple)
        return field

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError(f"Color must be an RGB tuple, got {type(value)}")
        self.set_field("color", value)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False

        fields = self.fields
        other_fields = other.fields

        if sorted(fields) != sorted(other_fields):
            return False

        for field in fields:
            if self.get_field(field) != other.get_field(field):
                return False

        if self.wkt != other.wkt:
            return False

        return True

    def __iadd__(self, other: Any) -> None:
        raise TypeError(f"Unsupported operand type(s) for +=: {type(self)} and {type(other)}")

    def __isub__(self, other: Any) -> None:
        raise TypeError(f"Unsupported operand type(s) for -=: {type(self)} and {type(other)}")

    def __repr__(self) -> str:
        repr_string = f"<{self.__class__.__name__}("
        parts = []
        for field in sorted(self.fields):
            value = self.get_field(field)
            parts.append(f"{field}={value}")

        repr_string += ", ".join(parts)

        if len(self.wkt) > 30:
            repr_string += f") WKT='{self.wkt[:30]}...'>"
        else:
            repr_string += f") WKT='{self.wkt}'>"
        return repr_string


class Polygon(_dg.Polygon, _BaseGeometry):
    def __init__(self, *args: Any, **kwargs: Any):
        _BaseGeometry.__init__(self)
        if SHAPELY_AVAILABLE:
            if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ShapelyPolygon):
                warnings.warn(
                    "Creating a Polygon from a Shapely Polygon is deprecated and will be removed dlup v1.0.0. "
                    "Please use the `from_shapely` method instead.",
                    UserWarning,
                )
                shapely_polygon = args[0]
                exterior = list(shapely_polygon.exterior.coords)
                interiors = [list(interior.coords) for interior in shapely_polygon.interiors]
                args = (exterior, interiors)

        # Ensure no new Polygon is created; just wrap the existing one
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], _dg.Polygon):
            super().__init__(args[0])
        else:
            fields = {}
            if "label" in kwargs:
                fields["label"] = kwargs.pop("label")
            if "index" in kwargs:
                fields["index"] = kwargs.pop("index")
            if "color" in kwargs:
                fields["color"] = kwargs.pop("color")

            if len(args) == 1:
                # No interior
                args = (args[0], [])

            super().__init__(*args, **kwargs)
            for key, value in fields.items():
                self.set_field(key, value)

    @classmethod
    def from_shapely(cls, shapely_polygon: "ShapelyPolygon") -> "Polygon":
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "Shapely is not available, and this functionality requires it. "
                "Install it using `pip install shapely`, "
                "or consult the documentation https://shapely.readthedocs.io/en/stable/installation.html "
                "for more information."
            )

        if not isinstance(shapely_polygon, ShapelyPolygon):
            raise ValueError(f"Expected a shapely.geometry.Polygon, but got {type(shapely_polygon)}")

        exterior = list(shapely_polygon.exterior.coords)
        interiors = [list(interior.coords) for interior in shapely_polygon.interiors]
        return cls(exterior, interiors)

    def __getstate__(self) -> dict[str, dict[str, Any]]:
        state = {
            "_fields": {field: self.get_field(field) for field in self.fields},
            "_object": {"interiors": self.get_interiors(), "exterior": self.get_exterior()},
        }
        return state

    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None:
        exterior = state["_object"]["exterior"]
        interiors = state["_object"]["interiors"]

        Polygon.__init__(self, exterior, interiors)

        for key, value in state["_fields"].items():
            self.set_field(key, value)

    def __copy__(self) -> "Polygon":
        warnings.warn(
            "Copying a Polygon currently creates a complete new object, without reference to the previous one, "
            "and is essentially the same as a deepcopy."
        )
        new_copy = Polygon(self)
        return new_copy

    def to_shapely(self) -> "ShapelyPolygon":
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "Shapely is not available, and this functionality requires it. "
                "Install it using `pip install shapely`, "
                "or consult the documentation https://shapely.readthedocs.io/en/stable/installation.html "
                "for more information."
            )

        import shapely.geometry

        exterior = self.get_exterior()
        interiors = self.get_interiors()
        return shapely.geometry.Polygon(exterior, interiors)


def _polygon_factory(polygon: _dg.Polygon) -> "Polygon":
    return Polygon(polygon)


# This is required to ensure that the polygons created in the C++ code are converted to the correct Python class
_dg.set_polygon_factory(_polygon_factory)


class Point(_dg.Point, _BaseGeometry):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _BaseGeometry.__init__(self)
        if SHAPELY_AVAILABLE:
            if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], ShapelyPoint):
                warnings.warn(
                    "Creating a Polygon from a Shapely Point is deprecated and will be removed dlup v1.0.0. "
                    "Please use the `from_shapely` method instead.",
                    UserWarning,
                )
                shapely_point = args[0]
                args = (shapely_point.x, shapely_point.y)

        # Ensure no new Point is created; just wrap the existing one
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], _dg.Point):
            super().__init__(args[0])
        else:
            fields = {}
            if "label" in kwargs:
                fields["label"] = kwargs.pop("label")
            if "index" in kwargs:
                fields["index"] = kwargs.pop("index")
            if "color" in kwargs:
                fields["color"] = kwargs.pop("color")

            super().__init__(*args, **kwargs)
            for key, value in fields.items():
                self.set_field(key, value)

    @classmethod
    def from_shapely(cls, shapely_point: "ShapelyPoint") -> "Point":
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "Shapely is not available, and this functionality requires it. "
                "Install it using `pip install shapely`, "
                "or consult the documentation https://shapely.readthedocs.io/en/stable/installation.html "
                "for more information."
            )

        if not isinstance(shapely_point, ShapelyPoint):
            raise ValueError(f"Expected a shapely.geometry.Point, but got {type(shapely_point)}")

        return cls(shapely_point.x, shapely_point.y)

    def to_shapely(self) -> "ShapelyPoint":
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "Shapely is not available, and this functionality requires it. "
                "Install it using `pip install shapely`, "
                "or consult the documentation https://shapely.readthedocs.io/en/stable/installation.html "
                "for more information."
            )

        return ShapelyPoint(self.x, self.y)

    def __copy__(self) -> "Point":
        new_copy = Point(self.x, self.y)

        for field in self.fields:
            new_copy.set_field(field, self.get_field(field))

        return new_copy

    def __deepcopy__(self, memo: Any) -> "Point":
        new_copy = Point(copy.deepcopy(self.x), copy.deepcopy(self.y))

        for field in self.fields:
            new_copy.set_field(field, copy.deepcopy(self.get_field(field), memo))

        return new_copy

    def __getstate__(self) -> dict[str, dict[str, Any]]:
        state = {
            "_fields": {field: self.get_field(field) for field in self.fields},
            "_object": {"coordinates": (self.x, self.y)},
        }
        return state

    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None:
        coordinates = state["_object"]["coordinates"]
        Point.__init__(self, coordinates[0], coordinates[1])
        for key, value in state["_fields"].items():
            self.set_field(key, value)


def _point_factory(point: _dg.Point) -> Point:
    return Point(point)


# Register the point factory
_dg.set_point_factory(_point_factory)


class Box(_dg.Box, _BaseGeometry):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _BaseGeometry.__init__(self)
        # Ensure no new Point is created; just wrap the existing one
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], _dg.Box):
            super().__init__(args[0])
        else:
            fields = {}
            if "label" in kwargs:
                fields["label"] = kwargs.pop("label")
            if "index" in kwargs:
                fields["index"] = kwargs.pop("index")
            if "color" in kwargs:
                fields["color"] = kwargs.pop("color")

            super().__init__(*args, **kwargs)
            for key, value in fields.items():
                self.set_field(key, value)

    def __copy__(self) -> "Box":
        new_copy = Box(self.coordinates, self.size)

        for field in self.fields:
            new_copy.set_field(field, self.get_field(field))

        return new_copy

    def __deepcopy__(self, memo: Any) -> "Box":
        new_copy = Box(copy.deepcopy(self.coordinates), copy.deepcopy(self.size))

        for field in self.fields:
            new_copy.set_field(field, copy.deepcopy(self.get_field(field), memo))

        return new_copy

    def __getstate__(self) -> dict[str, dict[str, Any]]:
        state = {
            "_fields": {field: self.get_field(field) for field in self.fields},
            "_object": {"coordinates": self.coordinates, "size": self.size},
        }
        return state

    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None:
        coordinates = state["_object"]["coordinates"]
        size = state["_object"]["size"]
        Box.__init__(self, coordinates, size)
        for key, value in state["_fields"].items():
            self.set_field(key, value)


def _box_factory(box: _dg.Box) -> Box:
    return Box(box)


# Register the box factory
_dg.set_box_factory(_box_factory)


class GeometryCollection(_dg.GeometryCollection):
    def __init__(self) -> None:
        super().__init__()

    @property
    def color_lut(self) -> npt.NDArray[np.uint8]:
        color_map = {}
        for r in self.polygons:
            color = r.color
            index = r.index
            if not index:
                raise ValueError("Index needs to be set on Polygon to create a color lookup table")
            if not color:
                raise ValueError("Color needs to be set on Polygon to create a color lookup table")

            color_map[index] = color

        max_index = max(color_map.keys())
        LUT = np.zeros((max_index + 1, 3), dtype=np.uint8)
        for key, color in color_map.items():
            LUT[key] = color

        return LUT

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False

        if len(self) != len(other):
            return False

        if self.boxes != other.boxes:
            return False

        if self.rois != other.rois:
            return False

        if self.polygons != other.polygons:
            return False

        if self.points != other.points:
            return False

        return True

    def __len__(self) -> int:
        # Also self.size()
        return len(self.polygons) + len(self.points) + len(self.boxes) + len(self.rois)

    def __getstate__(self) -> dict[str, Any]:
        """Return the state for pickling, ensuring all geometries are Python-wrapped.

        This method wraps any raw C++ _dg.* objects into their Python equivalents
        (Point, Polygon, Box) before returning the state, which fixes pickling issues
        when geometries are created inline.
        """
        # Get the base C++ state
        base_state: dict[str, Any] = super().__getstate__()  # type: ignore

        # Wrap raw C++ objects into Python wrappers
        def wrap_if_needed(obj: Any) -> Any:
            """Wrap raw C++ geometry objects into Python wrappers."""
            if isinstance(obj, _dg.Point) and not isinstance(obj, Point):
                return Point(obj)
            elif isinstance(obj, _dg.Polygon) and not isinstance(obj, Polygon):
                return Polygon(obj)
            elif isinstance(obj, _dg.Box) and not isinstance(obj, Box):
                return Box(obj)
            return obj

        # Wrap all geometry lists
        state = {
            "polygons": [wrap_if_needed(p) for p in base_state.get("polygons", [])],
            "points": [wrap_if_needed(p) for p in base_state.get("points", [])],
            "boxes": [wrap_if_needed(b) for b in base_state.get("boxes", [])],
            "rois": [wrap_if_needed(r) for r in base_state.get("rois", [])],
            "rtree_invalidated": base_state.get("rtree_invalidated", True),
        }

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the state from pickling."""
        # The C++ parent class should handle the state restoration
        super().__setstate__(state)  # type: ignore

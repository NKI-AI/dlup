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
"""Module for geometric objects.

Nanobind no longer supports holders or multiple inheritance, so we cannot
mix the C++ types with a Python ``_BaseGeometry`` parent the way pybind11
allowed. Instead, we install all helper properties, dunder methods, and
pickle hooks directly on the bound C++ classes (``_dg.Polygon`` etc.).

The public ``Polygon``/``Point``/``Box`` symbols are *factory classes*
implemented with a custom metaclass:

* ``Polygon(...)`` constructs a fresh ``_dg.Polygon`` and applies any
  ``label``/``index``/``color`` keyword arguments and Shapely conversion
  sugar.
* ``isinstance(obj, Polygon)`` returns ``True`` for any instance of
  ``_dg.Polygon``, so polygons returned from C++ (e.g.
  ``GeometryCollection.polygons``) still pass ``isinstance`` checks
  throughout the codebase.

This avoids nanobind's restrictions on multiple inheritance and on
subclassing the C++-bound type while preserving the user-facing API.
"""

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


if not SHAPELY_AVAILABLE:
    warnings.warn("Shapely is not available, and the `from_shapely` method will not be available.")


# ---------------------------------------------------------------------------
# Shared helpers monkey-patched onto the bound C++ types.
# ---------------------------------------------------------------------------


def _label_getter(self: Any) -> Optional[str]:
    field = self.get_field("label")
    if field is None:
        return None
    assert isinstance(field, str)
    return field


def _label_setter(self: Any, value: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"Label must be a string, got {type(value)}")
    self.set_field("label", value)


def _index_getter(self: Any) -> Optional[int]:
    field = self.get_field("index")
    if field is None:
        return None
    if not isinstance(field, int):
        raise ValueError(f"Index must be an integer, got {type(field)}")
    return field


def _index_setter(self: Any, value: int) -> None:
    if not isinstance(value, int):
        raise ValueError(f"Index must be an integer, got {type(value)}")
    self.set_field("index", value)


def _color_getter(self: Any) -> Optional[tuple[int, int, int]]:
    field = self.get_field("color")
    if field is None:
        return None
    if not isinstance(field, tuple) or len(field) != 3:
        raise ValueError(f"Color must be an RGB tuple, got {type(field)}")
    return field


def _color_setter(self: Any, value: tuple[int, int, int]) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"Color must be an RGB tuple, got {type(value)}")
    self.set_field("color", value)


def _geometry_eq(self: Any, other: Any) -> bool:
    if type(self) is not type(other):
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


def _geometry_iadd(self: Any, other: Any) -> None:
    raise TypeError(f"Unsupported operand type(s) for +=: {type(self)} and {type(other)}")


def _geometry_isub(self: Any, other: Any) -> None:
    raise TypeError(f"Unsupported operand type(s) for -=: {type(self)} and {type(other)}")


def _geometry_repr(self: Any) -> str:
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


def _install_shared_helpers(cls: type) -> None:
    cls.label = property(_label_getter, _label_setter)
    cls.index = property(_index_getter, _index_setter)
    cls.color = property(_color_getter, _color_setter)
    cls.__eq__ = _geometry_eq
    cls.__iadd__ = _geometry_iadd
    cls.__isub__ = _geometry_isub
    cls.__repr__ = _geometry_repr


_install_shared_helpers(_dg.Polygon)
_install_shared_helpers(_dg.Point)
_install_shared_helpers(_dg.Box)


def _pop_field_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("label", "index", "color"):
        if key in kwargs:
            fields[key] = kwargs.pop(key)
    return fields


def _apply_fields(obj: Any, fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        if value is None:
            continue
        obj.set_field(key, value)


# ---------------------------------------------------------------------------
# Pickle and Shapely helpers monkey-patched onto each bound C++ type.
# ---------------------------------------------------------------------------


def _polygon_getstate(self: Any) -> dict[str, dict[str, Any]]:
    return {
        "_fields": {field: self.get_field(field) for field in self.fields},
        "_object": {"interiors": self.get_interiors(), "exterior": self.get_exterior()},
    }


def _polygon_setstate(self: Any, state: dict[str, dict[str, Any]]) -> None:
    exterior = state["_object"]["exterior"]
    interiors = state["_object"]["interiors"]
    _dg.Polygon.__init__(self, exterior, interiors)
    for key, value in state["_fields"].items():
        self.set_field(key, value)


def _polygon_to_shapely(self: Any) -> "ShapelyPolygon":
    if not SHAPELY_AVAILABLE:
        raise ImportError(
            "Shapely is not available, and this functionality requires it. "
            "Install it using `pip install shapely`, "
            "or consult the documentation https://shapely.readthedocs.io/en/stable/installation.html "
            "for more information."
        )

    import shapely.geometry

    return shapely.geometry.Polygon(self.get_exterior(), self.get_interiors())


def _polygon_copy(self: Any) -> "_dg.Polygon":
    warnings.warn(
        "Copying a Polygon currently creates a complete new object, without reference to the previous one, "
        "and is essentially the same as a deepcopy."
    )
    new_copy = Polygon(self.get_exterior(), self.get_interiors())
    for field in self.fields:
        new_copy.set_field(field, self.get_field(field))
    return new_copy


def _polygon_deepcopy(self: Any, memo: Any) -> "_dg.Polygon":
    new_copy = Polygon(
        copy.deepcopy(self.get_exterior(), memo),
        copy.deepcopy(self.get_interiors(), memo),
    )
    for field in self.fields:
        new_copy.set_field(field, copy.deepcopy(self.get_field(field), memo))
    return new_copy


_dg.Polygon.__getstate__ = _polygon_getstate
_dg.Polygon.__setstate__ = _polygon_setstate
_dg.Polygon.to_shapely = _polygon_to_shapely
_dg.Polygon.__copy__ = _polygon_copy
_dg.Polygon.__deepcopy__ = _polygon_deepcopy


def _point_getstate(self: Any) -> dict[str, dict[str, Any]]:
    return {
        "_fields": {field: self.get_field(field) for field in self.fields},
        "_object": {"coordinates": (self.x, self.y)},
    }


def _point_setstate(self: Any, state: dict[str, dict[str, Any]]) -> None:
    coordinates = state["_object"]["coordinates"]
    _dg.Point.__init__(self, coordinates[0], coordinates[1])
    for key, value in state["_fields"].items():
        self.set_field(key, value)


def _point_to_shapely(self: Any) -> "ShapelyPoint":
    if not SHAPELY_AVAILABLE:
        raise ImportError(
            "Shapely is not available, and this functionality requires it. "
            "Install it using `pip install shapely`, "
            "or consult the documentation https://shapely.readthedocs.io/en/stable/installation.html "
            "for more information."
        )

    return ShapelyPoint(self.x, self.y)


def _point_copy(self: Any) -> "_dg.Point":
    new_copy = Point(self.x, self.y)
    for field in self.fields:
        new_copy.set_field(field, self.get_field(field))
    return new_copy


def _point_deepcopy(self: Any, memo: Any) -> "_dg.Point":
    new_copy = Point(copy.deepcopy(self.x), copy.deepcopy(self.y))
    for field in self.fields:
        new_copy.set_field(field, copy.deepcopy(self.get_field(field), memo))
    return new_copy


_dg.Point.__getstate__ = _point_getstate
_dg.Point.__setstate__ = _point_setstate
_dg.Point.to_shapely = _point_to_shapely
_dg.Point.__copy__ = _point_copy
_dg.Point.__deepcopy__ = _point_deepcopy


def _box_getstate(self: Any) -> dict[str, dict[str, Any]]:
    return {
        "_fields": {field: self.get_field(field) for field in self.fields},
        "_object": {"coordinates": self.coordinates, "size": self.size},
    }


def _box_setstate(self: Any, state: dict[str, dict[str, Any]]) -> None:
    coordinates = state["_object"]["coordinates"]
    size = state["_object"]["size"]
    _dg.Box.__init__(self, coordinates, size)
    for key, value in state["_fields"].items():
        self.set_field(key, value)


def _box_copy(self: Any) -> "_dg.Box":
    new_copy = Box(self.coordinates, self.size)
    for field in self.fields:
        new_copy.set_field(field, self.get_field(field))
    return new_copy


def _box_deepcopy(self: Any, memo: Any) -> "_dg.Box":
    new_copy = Box(copy.deepcopy(self.coordinates), copy.deepcopy(self.size))
    for field in self.fields:
        new_copy.set_field(field, copy.deepcopy(self.get_field(field), memo))
    return new_copy


_dg.Box.__getstate__ = _box_getstate
_dg.Box.__setstate__ = _box_setstate
_dg.Box.__copy__ = _box_copy
_dg.Box.__deepcopy__ = _box_deepcopy


# ---------------------------------------------------------------------------
# Factory-class metaclass: ``Polygon(...)`` constructs a ``_dg.Polygon``,
# and ``isinstance(obj, Polygon)`` is true for any ``_dg.Polygon`` instance.
# ---------------------------------------------------------------------------


class _GeometryFactoryMeta(type):
    """Metaclass that turns the host class into a thin factory for a
    nanobind-bound C++ type while keeping ``isinstance`` checks accurate.
    """

    bound_type: type
    construct: Any

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.construct(*args, **kwargs)

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, cls.bound_type)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, cls.bound_type)


def _polygon_construct(*args: Any, **kwargs: Any) -> "_dg.Polygon":
    if SHAPELY_AVAILABLE and len(args) == 1 and not kwargs and isinstance(args[0], ShapelyPolygon):
        warnings.warn(
            "Creating a Polygon from a Shapely Polygon is deprecated and will be removed dlup v1.0.0. "
            "Please use the `from_shapely` method instead.",
            UserWarning,
        )
        shapely_polygon = args[0]
        exterior = list(shapely_polygon.exterior.coords)
        interiors = [list(interior.coords) for interior in shapely_polygon.interiors]
        args = (exterior, interiors)

    fields = _pop_field_kwargs(kwargs)

    if len(args) == 1 and not kwargs and not isinstance(args[0], _dg.Polygon):
        args = (args[0], [])

    polygon = _dg.Polygon(*args, **kwargs)
    _apply_fields(polygon, fields)
    return polygon


def _point_construct(*args: Any, **kwargs: Any) -> "_dg.Point":
    if SHAPELY_AVAILABLE and len(args) == 1 and not kwargs and isinstance(args[0], ShapelyPoint):
        warnings.warn(
            "Creating a Polygon from a Shapely Point is deprecated and will be removed dlup v1.0.0. "
            "Please use the `from_shapely` method instead.",
            UserWarning,
        )
        shapely_point = args[0]
        args = (shapely_point.x, shapely_point.y)

    fields = _pop_field_kwargs(kwargs)
    point = _dg.Point(*args, **kwargs)
    _apply_fields(point, fields)
    return point


def _box_construct(*args: Any, **kwargs: Any) -> "_dg.Box":
    fields = _pop_field_kwargs(kwargs)
    box = _dg.Box(*args, **kwargs)
    _apply_fields(box, fields)
    return box


class Polygon(metaclass=_GeometryFactoryMeta):
    """Factory class for :class:`_dg.Polygon`.

    Construction goes through :func:`_polygon_construct`, and
    ``isinstance(obj, Polygon)`` matches any ``_dg.Polygon`` instance.
    """

    bound_type = _dg.Polygon
    construct = staticmethod(_polygon_construct)

    @classmethod
    def from_shapely(cls, shapely_polygon: "ShapelyPolygon") -> "_dg.Polygon":
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


class Point(metaclass=_GeometryFactoryMeta):
    """Factory class for :class:`_dg.Point`."""

    bound_type = _dg.Point
    construct = staticmethod(_point_construct)

    @classmethod
    def from_shapely(cls, shapely_point: "ShapelyPoint") -> "_dg.Point":
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


class Box(metaclass=_GeometryFactoryMeta):
    """Factory class for :class:`_dg.Box`."""

    bound_type = _dg.Box
    construct = staticmethod(_box_construct)


def _polygon_factory(polygon: "_dg.Polygon") -> "_dg.Polygon":
    return polygon


def _point_factory(point: "_dg.Point") -> "_dg.Point":
    return point


def _box_factory(box: "_dg.Box") -> "_dg.Box":
    return box


_dg.set_polygon_factory(_polygon_factory)
_dg.set_point_factory(_point_factory)
_dg.set_box_factory(_box_factory)


# ---------------------------------------------------------------------------
# GeometryCollection
# ---------------------------------------------------------------------------


class GeometryCollection(_dg.GeometryCollection):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    @property
    def color_lut(self) -> npt.NDArray[np.uint8]:
        color_map: dict[int, tuple[int, int, int]] = {}
        for r in self.polygons:
            color = r.color
            index = r.index
            if not index:
                raise ValueError("Index needs to be set on Polygon to create a color lookup table")
            if not color:
                raise ValueError("Color needs to be set on Polygon to create a color lookup table")
            color_map[index] = color

        max_index = max(color_map.keys())
        lookup = np.zeros((max_index + 1, 3), dtype=np.uint8)
        for key, color in color_map.items():
            lookup[key] = color

        return lookup

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _dg.GeometryCollection):
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
        return len(self.polygons) + len(self.points) + len(self.boxes) + len(self.rois)

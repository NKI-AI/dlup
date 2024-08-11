# Copyright (c) dlup contributors
"""Module for geometric objects"""
import dlup._geometry as _dg
from dlup.utils.imports import SHAPELY_AVAILABLE


class Polygon(_dg.Polygon):
    def __init__(self, *args, label=None, index=None, color=None, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], _dg.Polygon):
            super().__init__(args[0])
        else:
            super().__init__(*args, **kwargs)

        if label is not None:
            self.set_field("label", label)
        if index is not None:
            self.set_field("index", index)
        if color is not None:
            self.set_field("color", color)

    @classmethod
    def from_wkt(cls, wkt):
        # TODO: Maybe this can also be done in the C++ code
        return cls(_dg.BoostPolygon.from_wkt(wkt))

    @property
    def label(self):
        return self.get_field("label")

    @property
    def index(self):
        return self.get_field("index")

    @property
    def color(self):
        return self.get_field("color")

    @property
    def area(self):
        return self.get_area()

    @property
    def wkt(self):
        return self.to_wkt()

    def to_shapely(self):
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "Shapely is not available, and this functionality requires it. Install it using `pip install shapely`, or consult the documentation https://shapely.readthedocs.io/en/stable/installation.html for more information."
            )
        import shapely.geometry

        exterior = self.get_exterior()
        interiors = self.get_interiors()
        return shapely.geometry.Polygon(exterior, interiors)

    def __repr__(self):
        repr_string = f"<{self.__class__.__name__}("

        parts = []
        if self.label:
            parts.append(f"label='{self.label}'")
        if self.color:
            parts.append(f"color='{self.color}'")
        if self.index is not None:
            parts.append(f"index={self.index}")

        repr_string += ", ".join(parts)

        if len(self.wkt) > 30:
            repr_string += f") WKT='{self.wkt[:30]}...'>"
        else:
            repr_string += f") WKT='{self.wkt}'>"
        return repr_string


def dlup_polygon_factory(polygon):
    try:
        return Polygon(polygon)
    except Exception as e:
        raise ValueError(f"Could not create Polygon from {polygon}") from e


# This is required to ensure that the polygons created in the C++ code are converted to the correct Python class
_dg.set_polygon_factory(dlup_polygon_factory)


class Point(_dg.Point):
    def __init__(self, x, y, label=None, index=None, color=None):
        super().__init__(x, y)
        # This also needs a factory to support transforms on the point, unless we change it in place. Is that a good idea?
        if label is not None:
            self.set_field("label", label)
        if index is not None:
            self.set_field("index", index)
        if color is not None:
            self.set_field("color", color)

    @property
    def label(self):
        return self.get_field("label")

    @property
    def index(self):
        return self.get_field("index")

    @property
    def color(self):
        return self.get_field("color")

    def scale(self, scaling, origin=None):
        if origin is None:
            origin = Point(0, 0)
        return super().scale(scaling, origin)


class GeometryContainer(_dg.GeometryContainer):
    def __init__(self):
        super().__init__()

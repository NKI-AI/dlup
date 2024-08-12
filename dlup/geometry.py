# Copyright (c) dlup contributors
"""Module for geometric objects"""
import dlup._geometry as _dg
from dlup.utils.imports import SHAPELY_AVAILABLE


class DlupPolygon(_dg.Polygon):
    def __init__(self, *args, **kwargs):
        # Ensure no new Polygon is created; just wrap the existing one
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], _dg.Polygon):
            super().__init__(args[0])  # This should keep the original parameters intact
        else: # This needs to be way more elaborate
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

    @property
    def label(self):
        return self.get_field("label")

    @property
    def index(self):
        return self.get_field("index")

    @property
    def color(self):
        return self.get_field("color")

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
        dlup_polygon = DlupPolygon(polygon)
        return dlup_polygon
    except _dg.GeometryFactoryFunctionError as e:
        raise RuntimeError(f"Could not create Polygon from C++ backend {polygon}") from e
    except _dg.GeometryError as e:
        raise RuntimeError(f"Generic exception raised trying to create Polygon from C++ backend {polygon}") from e

# This is required to ensure that the polygons created in the C++ code are converted to the correct Python class
_dg.set_polygon_factory(dlup_polygon_factory)


class DlupPoint(_dg.Point):
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
            origin = DlupPoint(0, 0)
        return super().scale(scaling, origin)


class DlupGeometryContainer(_dg.GeometryContainer):
    def __init__(self):
        super().__init__()


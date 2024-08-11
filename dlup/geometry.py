import dlup._geometry as _geometry
import shapely.geometry

class PolygonZ(_geometry.Polygon):
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], _geometry.BoostPolygon):
            super().__init__(args[0])
        else:
            super().__init__(*args, **kwargs)
        self._lazy_properties = {}

    @classmethod
    def from_wkt(cls, wkt):
        return cls(_geometry.BoostPolygon.from_wkt(wkt))
        
    @property
    def area(self):
        return self.get_area()

    @property
    def wkt(self):
        return self.to_wkt()

    def add_property(self, name, func):
        self._lazy_properties[name] = func

    def to_shapely(self):
        exterior = self.get_exterior()
        interiors = self.get_interiors()
        return shapely.geometry.Polygon(exterior, interiors)

class Point(_geometry.Point):
    def __init__(self, x, y):
        super().__init__(x, y)

class LazyGeometryContainer:
    def __init__(self):
        self._container = _geometry.GeometryContainer()
        self._pipeline = []

    def add_polygon(self, polygon):
        self._container.add_polygon(polygon)

    def add_point(self, point):
        self._container.add_point(point)

    def read_region(self, coordinates, scaling, size):
        if isinstance(coordinates, tuple) and len(coordinates) == 2:
            coordinates = Point(*coordinates)
        self._pipeline.append(("read_region", coordinates, scaling, size))
        return self

    @property
    def polygons(self):
        return self._execute_pipeline().polygons

    @property
    def points(self):
        return self._execute_pipeline().points

    def _execute_pipeline(self):
        result = self._container
        for operation, *args in self._pipeline:
            if operation == "read_region":
                result = result.read_region(*args)
        return result

# Create a polygon with exterior and interior coordinates
exterior = [(0, 0), (0, 1), (1, 1), (1, 0)]
interiors = [[(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)]]
poly = PolygonZ(exterior, interiors)
poly.set_parameter("color", "red")
poly.add_property("area", lambda: poly.get_parameter("calculated_area"))

print(poly.to_shapely().area, poly.area)


# Let's create a point
point = Point(0.5, 0.5)
print(Point.to_wkt())
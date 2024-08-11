import dlup._geometry as _geometry
import shapely.geometry



class PolygonZ(_geometry.Polygon):
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], _geometry.Polygon):
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

def polygonz_factory(polygon):
    try:
        return PolygonZ(polygon)
    except Exception as e:
        print(f"Error in polygonz_factory: {e}")
        return polygon

_geometry.set_polygonz_factory(polygonz_factory)


class Point(_geometry.Point):
    def __init__(self, x, y):
        super().__init__(x, y)

    def scale(self, scaling, origin=None):
        if origin is None:
            origin = Point(0, 0)
        return super().scale(scaling, origin)
    
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

# Create multiple polygons and points
polygons = [
    PolygonZ([(0, 0), (0, 3), (3, 3), (3, 0)], []),
    PolygonZ([(2, 2), (2, 5), (5, 5), (5, 2)], []),
    PolygonZ([(4, 4), (4, 7), (7, 7), (7, 4)], []),
    PolygonZ([(6, 6), (6, 9), (9, 9), (9, 6)], [])
]

points = [
    Point(1, 1),
    Point(4, 4),
    Point(6, 6),
    Point(8, 8)
]

# Initialize the LazyGeometryContainer
container = LazyGeometryContainer()

# Add polygons and points to the container
for polygon in polygons:
    container.add_polygon(polygon)

for point in points:
    container.add_point(point)

# Query the region
scaling = 1.0
size = (3, 3)

print(container.polygons)


region = container._container.read_region((2, 2), scaling, size)
# print(region)
# Output the WKT of the intersecting polygons and points
# print("Intersecting Polygons WKT:")
# for poly in region.polygons:
#     print(poly.to_wkt())

for sample in region:
    # if isinstance(sample, _geometry.Polygon):
    #     sample = PolygonZ(sample)

    if isinstance(sample, PolygonZ):
        print(sample.to_wkt())
    elif isinstance(sample, Point):
        print(f"Point at ({sample.get_x()}, {sample.get_y()})")
    else:
        print("Unknown geometry type", sample.to_wkt())

# print("Intersecting Poin
# ts:")
# for point in region.points:
#     print(f"Point at ({point.get_x()}, {point.get_y()})")

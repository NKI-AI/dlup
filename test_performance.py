from pathlib import Path

import shapely.geometry

import dlup._geometry as dg
from dlup.annotations import WsiAnnotations
from dlup.annotations2 import WsiAnnotations2
from dlup.geometry import DlupGeometryContainer, DlupPoint, DlupPolygon

# Let's test conversion

dgPolygon = dg.Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])
dgPolygon.set_field("label", "X")

new_polygon = DlupPolygon(dgPolygon)
assert dgPolygon.fields == new_polygon.fields != []


exterior = [(0, 0), (0, 3), (3, 3), (3, 0)]
interior = [(1, 1), (1, 2), (2, 2), (2, 1)]
# More holes
interior2 = [(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)]
shapely_polygon = shapely.geometry.Polygon(exterior, [interior, interior2])
print(shapely_polygon.area)

# Let's create a polygon with a hole in dlup
dlup_polygon = DlupPolygon(exterior, [interior, interior2])

assert dlup_polygon.area == dlup_polygon.to_shapely().area == shapely_polygon.area


# Create multiple polygons and points
polygons = [
    DlupPolygon(dg.Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])),
    DlupPolygon(dg.Polygon([(2, 2), (2, 5), (5, 5), (5, 2)], [])),
    DlupPolygon(dg.Polygon([(4, 4), (4, 7), (7, 7), (7, 4)], [])),
    DlupPolygon(dg.Polygon([(6, 6), (6, 9), (9, 9), (9, 6)], [])),
]

points = [DlupPoint(1, 1, label="taart"), DlupPoint(4, 4, index=1), DlupPoint(6, 6), DlupPoint(8, 8)]

pointers = []
point_pointers = []

print("Looping over the polygons")
for poly in polygons:
    poly.set_field("label", "test")
    pointers.append(poly.pointer_id)

for point in points:
    point_pointers.append(point.pointer_id)
    print(point, point.pointer_id)

# Initialize the LazyGeometryContainer
container = DlupGeometryContainer()

second_pointers = []
# Add polygons and points to the container
for polygon in polygons:
    # print(polygon, polygon.get_pointer_id())
    second_pointers.append(polygon.pointer_id)
    container.add_polygon(polygon)

second_point_pointers = []
for point in points:
    container.add_point(point)
    second_point_pointers.append(point.pointer_id)

assert pointers == second_pointers
assert point_pointers == second_point_pointers


third_pointers = []
third_point_pointers = []
print(container.polygons)
for sample in container.polygons:
    third_pointers.append(sample.pointer_id)
    assert sample.get_field("label") == "test"
    # print(sample, sample.get_fields(), sample.get_pointer_id())
#

for sample in container.points:
    third_point_pointers.append(sample.pointer_id)
    print(sample, sample.fields, sample.pointer_id)

assert pointers == third_pointers

print("Getting regions\n====================")

regions = container.read_region((2, 2), 1.0, (10, 10))
polygon_shift = 0
point_counter = 0
for region in regions:
    if isinstance(region, DlupPolygon):
        polygon_shift += 1
        assert region.get_field("label") == "test"
    else:
        assert isinstance(region, DlupPoint)
        print(region, points[point_counter])
        point_counter += 1

# Let is try to get a non-existing field
assert polygons[0].get_field("non_existing") is None


import dlup

print(dlup.geometry.__file__)

import time

fn = Path("TCGA-E9-A1R4-01Z-00-DX1.B04D5A22-8CE5-49FD-8510-14444F46894D.geojson")

start_time = time.time()
annotations = WsiAnnotations.from_geojson(fn, sorting="NONE")

print(f"Time to load annotations (dlup v0.7.0): {(time.time() - start_time):.5f}s")


# Bounding box:
bbox = annotations.bounding_box
print(f"Bounding box: {bbox}")

# Let's get the region
start_time = time.time()
region = annotations.read_region((0, 0), 1.0, bbox[1])
dlup_reg = time.time() - start_time
print(f"Time to read region (dlup v0.7.0): {dlup_reg:.5f}s")
# print(f"Number of polygons in region (dlup v0.7.0): {len(region)}")
print()

start_time = time.time()
annotations2 = WsiAnnotations2.from_geojson(fn)
print(f"Time to load annotations (dlup v0.8.0.beta): {(time.time() - start_time):.5f}s")

start_time = time.time()
region2 = annotations2.read_region((0, 0), 1.0, bbox[1])
# Let's get all label names
labels0 = set([_.label for _ in region2])

new_reg = time.time() - start_time
print(f"Time to read (dlup v0.8.0.beta): {new_reg:.5f}s")
print(f"Number of polygons in region (dlup v0.8.0.beta): {len(region2)}")

print(f"Factor faster: {((dlup_reg / new_reg)):.3f}")
labels1 = set([_.label for _ in region2])

assert labels0 == labels1 != []


# print(annotations2._layers.polygons[:2])
# print(annotations2._layers.polygons[0].label)

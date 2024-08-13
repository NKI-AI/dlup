from pathlib import Path

import shapely.geometry

import dlup._geometry as dg
from dlup.annotations import WsiAnnotations
from dlup.annotations_experimental import WsiAnnotationsExperimental
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
    DlupPolygon(dg.Polygon([(4, 2), (4, 7), (7, 7), (7, 4)], [])),
    DlupPolygon(dg.Polygon([(6, 6), (6, 9), (9, 9), (9, 6)], [])),
]

for idx, polygon in enumerate(polygons):
    polygon.label = str(idx)


print("Areas: ", [poly.area for poly in polygons])

points = [DlupPoint(1, 1, label="taart"), DlupPoint(4, 4, index=1), DlupPoint(6, 6), DlupPoint(8, 8)]

pointers = []
point_pointers = []

print("Looping over the polygons")
for idx, poly in enumerate(polygons):
    if idx == 0:
        poly.set_field("label", "sample0")
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

# So my polygons are now this:
print("Polygons:")
print(container.polygons)

# # Remove the one with label 'taart'
# container.filter_polygons({"label": "taart"})
# print(container.polygons)

second_point_pointers = []
for point in points:
    container.add_point(point)
    second_point_pointers.append(point.pointer_id)



print(f"Points: {container.points}: {len(container.points)}")
# Let's remove a point
assert container.rtree_invalidated == False

print(f"Rtree valid: {not container.rtree_invalidated}")
container.remove_point(points[0])
assert container.rtree_invalidated == True
container.rebuild_rtree()
assert container.rtree_invalidated == False
print(f"Rtree valid: {not container.rtree_invalidated}")
container.remove_point(0)
print(f"Rtree valid: {not container.rtree_invalidated}")
assert container.rtree_invalidated == True


print(container.points)
print(f"Points after deletion: {container.points}: {len(container.points)}")


assert pointers == second_pointers
assert point_pointers == second_point_pointers


third_pointers = []
third_point_pointers = []
print(container.polygons)
for idx, sample in enumerate(container.polygons):
    third_pointers.append(sample.pointer_id)
    if idx == 0:
        assert sample.get_field("label") == "sample0"
    # print(sample, sample.get_fields(), sample.get_pointer_id())
#

for sample in container.points:
    third_point_pointers.append(sample.pointer_id)
    print(sample, sample.fields, sample.pointer_id)

assert pointers == third_pointers


regions = container.read_region((2, 2), 1.0, (10, 10))
polygon_shift = 0
point_counter = 0
for region in regions.polygons:
    assert isinstance(region, DlupPolygon)
    assert region.get_field("label") == "test"

for region in regions.points:
    assert isinstance(region, DlupPoint)
    print(region, points[point_counter])

# Let is try to get a non-existing field
assert polygons[0].get_field("non_existing") is None

print("Before sorting\n")
# Let's sort the polygons, but lets first get the typeS:
for sample in container.polygons:
    print(sample.area, sample.pointer_id)

print("After sorting\n")
container.sort_polygons(lambda x: x.area, False)
for sample in container.polygons:
    print(sample.area, sample.pointer_id)

container.sort_polygons(lambda x: x.area, True)
for sample in container.polygons:
    print(sample.area, sample.pointer_id)

container.rebuild_rtree()

print(container.polygons)
container.sort_polygons(lambda x: x.get_field("label"), False)
print(container.polygons)
for sample in container.polygons:
    print(sample.label, sample.pointer_id)


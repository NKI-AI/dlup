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
"""Test the geometry classes."""

import copy
import pickle
import tempfile

import dlup._geometry as dg
import pytest
import shapely.geometry
from dlup.geometry import (
    Box,
    GeometryCollection,
    Point,
    Polygon,
    _BaseGeometry,
    _box_factory,
    _point_factory,
    _polygon_factory,
)
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

polygons = [
    Polygon(dg.Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])),
    Polygon(dg.Polygon([(2, 2), (2, 5), (5, 5), (5, 2)], [])),
    Polygon(dg.Polygon([(4, 2), (4, 7), (7, 7), (7, 4)], [])),
    Polygon(dg.Polygon([(6, 6), (6, 9), (9, 9), (9, 6)], [])),
]
points = [Point(1, 1, label="label0"), Point(4, 4, index=1), Point(6, 6), Point(8, 8)]
boxes = [Box((1, 1), (2, 2)), Box((3, 3), (4, 4)), Box((5, 5), (6, 6)), Box((7, 7), (8, 8))]
rois = polygons


class TestGeometry:
    def test_base_geometry(self):
        _BaseGeometry()
        with pytest.raises(NotImplementedError):
            _BaseGeometry().from_shapely(None)

        with pytest.raises(NotImplementedError):
            _BaseGeometry().set_field("name", "test")

        with pytest.raises(NotImplementedError):
            _BaseGeometry().get_field("name")

    def test_try_to_set_incorrect_field_type(self):
        base = Polygon()
        with pytest.raises(ValueError):
            base.label = True
        with pytest.raises(ValueError):
            base.color = "red"
        with pytest.raises(ValueError):
            base.index = "1"

    def test_point_factory(self):
        c_point = dg.Point(1, 1)
        point = _point_factory(c_point)
        assert point == Point(1, 1)

    def test_polygon_factory(self):
        c_polygon = dg.Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])
        polygon = _polygon_factory(c_polygon)
        assert polygon == Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])

    def test_box_factory(self):
        c_box = dg.Box((1, 1), (2, 2))
        box = _box_factory(c_box)
        assert box == Box((1, 1), (2, 2))

    def test_box_area(self):
        box = Box((1, 1), (2, 2))
        box.area == 4
        box.as_polygon().area == box.area

    def box_to_polygon(self):
        box = Box((1, 1), (2, 2))
        polygon = box.as_polygon()
        assert isinstance(box, Box)
        assert isinstance(polygon, Polygon)
        assert polygon == Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])

    @pytest.mark.parametrize(
        "exterior,interiors,expected_area",
        [
            (
                [(0, 0), (0, 3), (3, 3), (3, 0)],
                [[(1, 1), (1, 2), (2, 2), (2, 1)], [(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)]],
                7.0,
            )
        ],
    )
    def test_if_area_is_correct(self, exterior, interiors, expected_area):
        shapely_polygon = shapely.geometry.Polygon(exterior, interiors)
        dlup_polygon = Polygon(exterior, interiors)
        assert dlup_polygon.area == dlup_polygon.to_shapely().area == shapely_polygon.area == expected_area

    @pytest.mark.parametrize(
        "field_name,field_value",
        [
            ("arbitrary field", [1, 2, 3, 4]),
            ("random", object),
        ],
    )
    def test_set_incompatible_field(self, field_name, field_value):
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])
        with pytest.raises(TypeError):
            polygon.set_field(field_name, field_value)

    def test_clone(self):
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], label=1)
        polygon_clone = polygon.clone()
        assert polygon == polygon_clone
        assert polygon is not polygon_clone
        assert polygon.pointer_id != polygon_clone.pointer_id

        point = Point(1, 1)
        point.set_field("label", 1)
        point_clone = point.clone()
        assert point == point_clone
        assert point is not point_clone
        assert point_clone.get_field("label") == 1
        assert point.pointer_id != point_clone.pointer_id

        box = Box((1, 1), (2, 2))
        box_clone = box.clone()
        assert box == box_clone
        assert box is not box_clone
        assert box.pointer_id != box_clone.pointer_id

    @pytest.mark.parametrize("object_to_pickle", [polygons[0], points[0]])
    def test_pickle_objects(self, object_to_pickle):
        object_to_pickle = copy.deepcopy(object_to_pickle)
        object_to_pickle.set_field("random", True)
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(object_to_pickle, f)
            f.seek(0)
            new_object = pickle.load(f)
            assert new_object == object_to_pickle

    def test_pickle_geometry_collection(self):
        import shapely

        roi = Polygon.from_shapely(shapely.geometry.Point((500, 500)).buffer(200))
        collection = GeometryCollection()
        collection.add_roi(roi)

        x = pickle.dumps(collection)
        collection2 = pickle.loads(x)
        assert collection == collection2
        assert collection.rtree_invalidated

        collection.rebuild_rtree()
        x = pickle.dumps(collection)
        collection2 = pickle.loads(x)
        assert collection == collection2
        assert not collection.rtree_invalidated

    def test_repr(self):
        polygon = Polygon([(1, 1), (2, 3), (3, 4), (0, 0)], label="label", index=1, color=(1, 1, 1))
        polygon.set_field("random", 1)
        assert (
            repr(polygon)
            == "<Polygon(color=(1, 1, 1), index=1, label=label, random=1) WKT='POLYGON((1 1,2 3,3 4,0 0,1 1))'>"
        )

        point = Point(1, 1, label="label", index=1, color=(1, 1, 1))
        assert repr(point) == "<Point(color=(1, 1, 1), index=1, label=label) WKT='POINT(1 1)'>"

        polygon = Polygon([(1, 1) for _ in range(100)])
        assert repr(polygon) == "<Polygon() WKT='POLYGON((1 1,1 1,1 1,1 1,1 1,1...'>"

    @pytest.mark.parametrize("original_object", polygons + points)
    def test_deep_copy(self, original_object):
        copied_object = copy.deepcopy(original_object)
        assert original_object is not copied_object
        assert original_object == copied_object

    def test_copy_polygon(self):
        polygon = polygons[0]
        polygon_copy = copy.copy(polygon)

        # TODO: Figure out way to not make a copy. Creating an InteriorRing object might be an option.
        assert polygon.get_interiors() == polygon_copy.get_interiors()
        assert polygon.get_exterior() == polygon_copy.get_exterior()

    def test_copy_point(self):
        point = points[0]
        point_copy = copy.copy(point)

        assert point == point_copy

    def test_collection_add_object(self):
        collection = GeometryCollection()
        collection.add_polygon(polygons[0])
        collection.add_polygon(polygons[1])
        assert collection.polygons == polygons[:2]

        collection.add_point(points[0])
        collection.add_point(points[1])
        assert collection.points == points[:2]

        collection.add_box(boxes[0])
        collection.add_box(boxes[1])
        assert collection.boxes == boxes[:2]

    def test_if_keeps_reference(self):
        collection = GeometryCollection()
        for polygon in polygons:
            collection.add_polygon(polygon)

        for point in points:
            collection.add_point(point)

        for box in boxes:
            collection.add_box(box)

        for idx, polygon in enumerate(collection.polygons):
            assert polygon == polygons[idx]
            assert polygon.pointer_id == polygons[idx].pointer_id

        for idx, point in enumerate(collection.points):
            assert point == points[idx]
            assert point.pointer_id == points[idx].pointer_id

        for idx, box in enumerate(collection.boxes):
            assert box == boxes[idx]
            assert box.pointer_id == boxes[idx].pointer_id

    def test_pointers(self):
        pointers = [poly.pointer_id for poly in polygons]
        point_pointers = [point.pointer_id for point in points]

        collection = GeometryCollection()
        for poly in polygons:
            collection.add_polygon(poly)

        for point in points:
            collection.add_point(point)

        for box in boxes:
            collection.add_box(box)

        for idx, poly in enumerate(collection.polygons):
            assert poly.pointer_id == pointers[idx]

        for idx, point in enumerate(collection.points):
            assert point.pointer_id == point_pointers[idx]

        for idx, box in enumerate(collection.boxes):
            assert box.pointer_id == boxes[idx].pointer_id

    def test_remove_geometry_from_collection(self):
        collection = GeometryCollection()
        for poly in polygons:
            collection.add_polygon(poly)

        for box in boxes:
            collection.add_box(box)

        for point in points:
            collection.add_point(point)

        assert collection.rtree_invalidated

        assert len(collection.polygons) == 4
        assert len(collection.points) == 4
        assert len(collection.points) == 4
        assert len(collection.boxes) == 4

        collection.remove_polygon(polygons[0])
        assert collection.polygons == polygons[1:]
        assert len(collection.polygons) == 3

        collection.remove_box(boxes[0])
        assert collection.boxes == boxes[1:]
        assert len(collection.boxes) == 3

        collection.remove_point(points[0])
        assert len(collection.points) == 3

        collection.remove_polygon(0)
        assert len(collection.polygons) == 2

        collection.remove_point(0)
        assert len(collection.points) == 2

        assert collection.rtree_invalidated
        collection.rebuild_rtree()
        assert not collection.rtree_invalidated

    def test_wkt(self):
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [[(1, 1), (1, 2), (2, 2), (2, 1)]])
        assert polygon.wkt == "POLYGON((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"

        point = Point(1, 1)
        assert point.wkt == "POINT(1 1)"

        box = Box((1, 1), (2, 2))
        assert box.wkt == "POLYGON((1 1,1 3,3 3,3 1,1 1))"

    @pytest.mark.parametrize("object_type", [Polygon, Point])
    def test_setting_properties(self, object_type):
        obj = object_type()
        obj.label = "test"
        obj.color = (1, 1, 1)

        if isinstance(obj, Polygon):
            obj.index = 1
            assert obj.index == 1

        assert obj.label == "test"
        assert obj.color == (1, 1, 1)

    def test_color_lut(self):
        collection = GeometryCollection()
        for idx, polygon in enumerate(polygons):
            collection.add_polygon(polygon)
            polygon.set_field("label", f"label {idx}")
            polygon.set_field("color", (idx + 1, idx + 1, idx + 1))
            polygon.set_field("index", idx + 1)

        # Add expected color LUT test here

    def test_close_loop(self):
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [[(1, 1), (1, 2), (2, 2), (2, 1)]])

        assert polygon.get_exterior() == [(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)]
        assert polygon.get_interiors() == [[(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]]

    def test_from_shapely_polygon(self):
        exterior = [(0, 0), (0, 3), (3, 3), (3, 0)]
        interiors = [[(1, 1), (1, 2), (2, 2), (2, 1)]]

        shapely_polygon = ShapelyPolygon(exterior, interiors)
        polygon_converted = Polygon.from_shapely(shapely_polygon)
        polygon_direct = Polygon(exterior, interiors)

        polygon_shapely_2 = Polygon(shapely_polygon)
        assert polygon_shapely_2 == polygon_converted

        assert (
            polygon_converted.get_exterior() == list(shapely_polygon.exterior.coords) == polygon_direct.get_exterior()
        )
        assert (
            polygon_converted.get_interiors()
            == [list(_.coords) for _ in shapely_polygon.interiors]
            == polygon_direct.get_interiors()
        )
        assert shapely_polygon == polygon_direct.to_shapely()

    def test_from_shapely_point(self):
        dlup_point = Point(1, 1)
        shapely_point = ShapelyPoint(1, 1)
        dlup_point2 = Point(shapely_point)

        assert dlup_point2 == dlup_point

        assert dlup_point == Point.from_shapely(shapely_point)
        assert dlup_point.to_shapely() == shapely_point

    @pytest.mark.parametrize("object_type", [Point, Polygon])
    def test_shapely_wrong_type(self, object_type):
        with pytest.raises(ValueError):
            object_type.from_shapely([])

    def test_sort_polygon(self):
        collection = GeometryCollection()
        for poly in polygons:
            collection.add_polygon(poly)

        assert [_.area for _ in collection.polygons] == [9.0, 9.0, 12.0, 9.0]

        collection.sort_polygons(lambda x: x.area, True)

        assert [_.area for _ in collection.polygons] == [12.0, 9.0, 9.0, 9.0]
        assert collection.polygons[0] == polygons[2]

        collection.sort_polygons(lambda x: x.area, False)
        assert [_.area for _ in collection.polygons] == [9.0, 9.0, 9.0, 12.0]
        assert collection.polygons[0] == polygons[0]

    @pytest.mark.parametrize("object_type", [Point, Polygon])
    def test_to_shapely_missing(self, object_type, monkeypatch):
        monkeypatch.setattr("dlup.geometry.SHAPELY_AVAILABLE", False)
        with pytest.raises(ImportError):
            object_type().to_shapely()

    @pytest.mark.parametrize("object_type", [ShapelyPoint, ShapelyPolygon])
    def test_from_shapely_missing(self, object_type, monkeypatch):
        monkeypatch.setattr("dlup.geometry.SHAPELY_AVAILABLE", False)
        with pytest.raises(ImportError):
            if object_type == ShapelyPoint:
                Point.from_shapely(object_type())
            else:
                Polygon.from_shapely(object_type())

    def test_point_scaling(self):
        point = Point(1, 1)
        pointer_id = point.pointer_id
        point.scale(2)

        assert point == Point(2, 2)
        assert point.pointer_id == pointer_id

    def test_polygon_scaling(self):
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [[(1, 1), (1, 2), (2, 2), (2, 1)]])
        pointer_id = polygon.pointer_id
        polygon.scale(2)

        assert polygon == Polygon([(0, 0), (0, 6), (6, 6), (6, 0)], [[(2, 2), (2, 4), (4, 4), (4, 2)]])
        assert polygon.pointer_id == pointer_id

    def test_box_scaling(self):
        box = Box((1, 1), (3.5, 2))
        pointer_id = box.pointer_id
        box.scale(2)
        assert box == Box((2, 2), (7, 4))
        assert box.pointer_id == pointer_id

    @pytest.mark.parametrize("scaling", [1.0, 2.0])
    def test_read_region(self, scaling):
        collection = GeometryCollection()
        for poly in polygons:
            collection.add_polygon(poly)

        for idx, poly in enumerate(polygons):
            poly.set_field("label", f"label {idx}")

        for point in points:
            collection.add_point(point)

        for idx, point in enumerate(points):
            point.set_field("index", idx + 1)

        assert collection.rtree_invalidated
        collection.read_region((2, 2), scaling, (10, 10))
        # It's still invalid because of the lazy evaluation!
        assert collection.rtree_invalidated
        collection.rebuild_rtree()
        assert not collection.rtree_invalidated
        region = collection.read_region((2, 2), scaling, (10, 10))
        region.polygons  # Call to ensure that the polygons are obtained
        region.points  # Call to ensure that the points are obtained
        assert not collection.rtree_invalidated

        # TODO: Add more elaborate tests for regions

    @pytest.mark.parametrize("shapely_available", [True, False])
    def test_importerror_for_from_shapely(self, shapely_available, monkeypatch):
        def mock_shapely_available():
            return shapely_available

        monkeypatch.setattr("dlup.geometry.SHAPELY_AVAILABLE", shapely_available)

        if shapely_available:
            shapely_polygon = ShapelyPolygon([(0, 0), (0, 3), (3, 3), (3, 0)])
            Polygon.from_shapely(shapely_polygon)
            Polygon(shapely_polygon)
        else:
            with pytest.raises(ImportError):
                Polygon.from_shapely(None)

    def test_compare_mismatch(self):
        point = Point(1, 1)
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [[(1, 1), (1, 2), (2, 2), (2, 1)]])

        assert point != polygon

    def test_compare_incorrect_fields(self):
        point0 = Point(1, 1, label="label0")
        point1 = Point(1, 1, label="label1")

        assert point0 != point1

        polygon0 = copy.deepcopy(polygons[0])
        polygon1 = copy.deepcopy(polygons[1])
        polygon1.set_field("random", False)

        assert polygon0 != polygon1

    def test_cannot_add_geometries(self):
        with pytest.raises(TypeError):
            polygons[0] += points[0]

        with pytest.raises(TypeError):
            polygons[0] += polygons[0]

    def test_cannot_subtract_geometries(self):
        with pytest.raises(TypeError):
            polygons[0] -= points[0]

        with pytest.raises(TypeError):
            polygons[0] -= polygons[0]

    def test_inequality(self):
        polygon0 = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])
        polygon1 = Polygon([(0, 0), (1, 3), (3, 3), (3, 0)], [])

        polygon0.label = "test"
        polygon1.label = "test"

        assert polygon0 != polygon1

        box0 = Box((1, 1), (2, 2))
        box1 = Box((1, 1), (3, 2))

        box0.label = "test"
        box1.label = "test"

        point0 = Point(0, 1)
        point1 = Point(1, 1)

        point0.color = (1, 2, 3)
        point1.color = (1, 2, 3)

        assert polygon0 != point1
        assert point0 != point1
        assert box0 != box1

    def test_equality(self):
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])
        polygon.label = "test"
        assert polygon is polygon

    def test_geometry_collection_lut(self):
        collection = GeometryCollection()
        for idx, polygon in enumerate(polygons[:3]):
            collection.add_polygon(polygon)
            polygon.set_field("label", f"label {idx}")
            polygon.set_field("color", (idx + 1, idx + 1, idx + 1))
            polygon.set_field("index", idx + 1)

        assert collection.color_lut.tolist() == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]

    def test_geometry_collection_lut_exceptions(self):
        collection = GeometryCollection()
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])
        collection.add_polygon(polygon)
        with pytest.raises(ValueError):
            collection.color_lut

        polygon.index = 1
        with pytest.raises(ValueError):
            collection.color_lut

    def test_geometry_collection_pickle(self):
        collection = GeometryCollection()
        for polygon in polygons:
            collection.add_polygon(polygon)

        for box in boxes:
            collection.add_box(box)

        for point in points:
            collection.add_point(point)

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(collection, f)
            f.seek(0)
            new_collection = pickle.load(f)
            assert new_collection == collection

    def test_geometry_collection_length(self):
        collection = GeometryCollection()
        for polygon in polygons:
            collection.add_polygon(polygon)

        assert len(collection) == 4

        for point in points:
            collection.add_point(point)

        assert len(collection) == 8

    def test_geometry_equality_different_type_and_length(self):
        collection0 = GeometryCollection()
        assert collection0 is not None
        collection1 = GeometryCollection()

        assert collection0 == collection1
        collection0.add_polygon(polygons[0])
        collection1.add_polygon(polygons[1])
        assert collection0 != collection1

        collection0.remove_polygon(polygons[0])
        collection1.remove_polygon(polygons[1])

        collection0.add_point(points[0])
        collection1.add_point(points[1])
        assert collection0 != collection1

    def test_geometry_read_region(self):
        collection = GeometryCollection()

        # Let's make a nice polygon that's a square
        polygon = Polygon([(0, 0), (0, 8), (8, 8), (8, 0)], [])

        collection.add_polygon(polygon)

        collection.add_roi(polygon)
        collection.add_roi(polygon)

        point0 = Point(1, 1)
        point1 = Point(4, 4, index=1)
        point2 = Point(4, 4)
        collection.add_point(point0)
        collection.add_point(point1)
        collection.add_point(point2)
        collection.rebuild_rtree()

        regions = collection.read_region((2, 2), 1.0, (5, 5))

        assert len(regions.points) == 2
        assert len(regions.polygons.get_geometries()) == 1
        assert regions.points == [Point(2, 2, index=1), Point(2, 2)]
        assert regions.polygons.get_geometries() == [Polygon([(0, 0), (0, 5), (5, 5), (5, 0)], [])]
        assert regions.rois.get_geometries() == [Polygon([(0, 0), (0, 5), (5, 5), (5, 0)], [])] * 2

    def test_geometry_scaling(self):
        collection = GeometryCollection()

        collection.add_polygon(polygons[0].__copy__())
        collection.add_point(points[0].__copy__())

        collection.scale(2)

        polygon0 = collection.polygons[0]

        assert polygon0.get_exterior() == [(0, 0), (0, 6), (6, 6), (6, 0), (0, 0)]
        assert polygon0.get_interiors() == []

        assert polygon0 == Polygon(
            [(0, 0), (0, 6), (6, 6), (6, 0), (0, 0)], [], color=(1, 1, 1), index=1, label="label 0"
        )

    def test_has_rois_property(self):
        """Test that the has_rois property works correctly in GeometryCollection and AnnotationRegion."""
        # Create a new collection and check that has_rois is initially False
        collection = GeometryCollection()
        assert not collection.has_rois

        # Add a regular polygon and check that has_rois is still False
        collection.add_polygon(polygons[0].__copy__())
        assert not collection.has_rois

        # Now add an ROI polygon and check that has_rois is True
        collection.add_roi(polygons[1].__copy__())
        assert collection.has_rois

        # Test that the has_rois property is properly transferred to AnnotationRegion
        collection.rebuild_rtree()
        region = collection.read_region((0, 0), 1.0, (10, 10))

        # The region should have the has_rois property set to True
        assert region.has_rois

        # Create a new collection without ROIs
        collection_no_rois = GeometryCollection()
        collection_no_rois.add_polygon(polygons[0].__copy__())

        # Test that has_rois is False
        assert not collection_no_rois.has_rois

        # Test that the region created from this collection has has_rois=False
        collection_no_rois.rebuild_rtree()
        region_no_rois = collection_no_rois.read_region((0, 0), 1.0, (10, 10))

        # The region should have the has_rois property set to False
        assert not region_no_rois.has_rois

    def test_mask(self):
        collection = GeometryCollection()
        polygon = Box((1, 1), (4, 4)).as_polygon()
        polygon.index = 2
        collection.add_polygon(polygon)
        collection.rebuild_rtree()

        region = collection.read_region((0, 0), 1.0, (5, 5))
        mask = region.polygons.to_mask().numpy()
        assert mask.sum() == 16 * 2

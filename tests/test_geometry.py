# Copyright (c) dlup contributors

"""Test the geometry classes."""

import copy
import pickle
import tempfile

import pytest
import shapely.geometry
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

import dlup._geometry as dg
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

polygons = [
    Polygon(dg.Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])),
    Polygon(dg.Polygon([(2, 2), (2, 5), (5, 5), (5, 2)], [])),
    Polygon(dg.Polygon([(4, 2), (4, 7), (7, 7), (7, 4)], [])),
    Polygon(dg.Polygon([(6, 6), (6, 9), (9, 9), (9, 6)], [])),
]

points = [Point(1, 1, label="label0"), Point(4, 4, index=1), Point(6, 6), Point(8, 8)]


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
            ("random", None),
        ],
    )
    def test_set_arbitrary_field(self, field_name, field_value):
        polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])
        polygon.set_field(field_name, field_value)
        assert polygon.get_field(field_name) == field_value

    @pytest.mark.parametrize("object_to_pickle", [polygons[0], points[0]])
    def test_pickle_objects(self, object_to_pickle):
        object_to_pickle = copy.deepcopy(object_to_pickle)
        object_to_pickle.set_field("random", True)
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(object_to_pickle, f)
            f.seek(0)
            new_object = pickle.load(f)
            assert new_object == object_to_pickle

    def test_repr(self):
        polygon = Polygon([(1, 1), (2, 3), (3, 4), (0, 0)], label="label", index=1, color=(1, 1, 1))
        polygon.set_field("random", True)
        assert (
            repr(polygon)
            == "<Polygon(color=(1, 1, 1), index=1, label=label, random=True) WKT='POLYGON((1 1,2 3,3 4,0 0,1 1))'>"
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

    def test_if_keeps_reference(self):
        collection = GeometryCollection()
        for polygon in polygons:
            collection.add_polygon(polygon)

        for point in points:
            collection.add_point(point)

        for idx, polygon in enumerate(collection.polygons):
            assert polygon == polygons[idx]
            assert polygon.pointer_id == polygons[idx].pointer_id

        for idx, point in enumerate(collection.points):
            assert point == points[idx]
            assert point.pointer_id == points[idx].pointer_id

    def test_pointers(self):
        pointers = [poly.pointer_id for poly in polygons]
        point_pointers = [point.pointer_id for point in points]

        collection = GeometryCollection()
        for poly in polygons:
            collection.add_polygon(poly)

        for point in points:
            collection.add_point(point)

        for idx, poly in enumerate(collection.polygons):
            assert poly.pointer_id == pointers[idx]

        for idx, point in enumerate(collection.points):
            assert point.pointer_id == point_pointers[idx]

    def test_remove_geometry_from_collection(self):
        collection = GeometryCollection()
        for poly in polygons:
            collection.add_polygon(poly)

        for point in points:
            collection.add_point(point)

        assert collection.rtree_invalidated

        assert len(collection.polygons) == 4
        assert len(collection.points) == 4

        collection.remove_polygon(polygons[0])
        assert collection.polygons == polygons[1:]
        assert len(collection.polygons) == 3

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

    @pytest.mark.parametrize("scaling", [1.0, 2.0])
    def test_read_region(self, scaling):
        collection = GeometryCollection()
        for poly in polygons:
            collection.add_polygon(poly)

        for idx, poly in enumerate(polygons):
            poly.set_field("label", f"label {idx}")

        assert collection.rtree_invalidated
        region = collection.read_region((2, 2), scaling, (10, 10))
        # It's still invalid because of the lazy evaluation!
        assert collection.rtree_invalidated
        region.polygons  # Call to ensure that the polygons are obtained
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

        point0 = Point(0, 1)
        point1 = Point(1, 1)

        point0.color = (1, 2, 3)
        point1.color = (1, 2, 3)

        assert polygon0 != point1
        assert point0 != point1

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

        for point in points:
            collection.add_point(point)

        regions = collection.read_region((2, 2), 1.0, (5, 5))

        assert len(regions.points) == 2
        assert len(regions.polygons.get_geometries()) == 1
        assert regions.points == [Point(2, 2, index=1), Point(4, 4)]
        assert regions.polygons.get_geometries() == [Polygon([(0, 0), (0, 5), (5, 5), (5, 0)], [])]

    def test_geometry_scaling(self):
        collection = GeometryCollection()
        for polygon in polygons:
            collection.add_polygon(polygon)

        for point in points:
            collection.add_point(point)

        collection.scale(2)

        polygon0 = collection.polygons[0]
        points0 = collection.points[0]

        assert points0 == Point(2, 2, label="label0")
        assert polygon0.get_exterior() == [(0, 0), (0, 6), (6, 6), (6, 0), (0, 0)]
        assert polygon0.get_interiors() == []

        assert polygon0 == Polygon(
            [(0, 0), (0, 6), (6, 6), (6, 0), (0, 0)], [], color=(1, 1, 1), index=1, label="label 0"
        )

    def test_mask(self):
        collection = GeometryCollection()
        polygon = Box((1, 1), (4, 4)).as_polygon()
        polygon.index = 2
        collection.add_polygon(polygon)

        region = collection.read_region((0, 0), 1.0, (5, 5))
        mask = region.polygons.to_mask().numpy()
        assert mask.sum() == 16 * 2

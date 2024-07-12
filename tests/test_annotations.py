# Copyright (c) dlup contributors

"""Test the annotation facilities."""
import json
import pathlib
import pickle
import tempfile

import pytest
import shapely.geometry
from shapely import Point as ShapelyPoint
from shapely import Polygon as ShapelyPolygon

from dlup.annotations import AnnotationClass, AnnotationType, Point, Polygon, WsiAnnotations, shape
from dlup.utils.imports import DARWIN_SDK_AVAILABLE

ASAP_XML_EXAMPLE = b"""<?xml version="1.0"?>
<ASAP_Annotations>
    <Annotations>
        <Annotation Name="Annotation 0" Type="Polygon" PartOfGroup="healthy glands" Color="#f96400">
            <Coordinates>
                <Coordinate Order="0" X="11826" Y="12804"/>
                <Coordinate Order="1" X="11818" Y="12808"/>
                <Coordinate Order="2" X="11804" Y="12826"/>
                <Coordinate Order="3" X="11788" Y="12860"/>
                <Coordinate Order="4" X="11778" Y="12874"/>
                <Coordinate Order="5" X="11858" Y="12874"/>
                <Coordinate Order="6" X="11862" Y="12858"/>
                <Coordinate Order="7" X="11844" Y="12814"/>
                <Coordinate Order="8" X="11842" Y="12810"/>
            </Coordinates>
        </Annotation>
    </Annotations>
    <AnnotationGroups>
        <Group Name="healthy glands" PartOfGroup="None" Color="#f96400">
            <Attributes/>
        </Group>
    </AnnotationGroups>
</ASAP_Annotations>"""


class TestAnnotations:
    with tempfile.NamedTemporaryFile(suffix=".xml") as asap_file:
        asap_file.write(ASAP_XML_EXAMPLE)
        asap_file.flush()
        asap_annotations = WsiAnnotations.from_asap_xml(pathlib.Path(asap_file.name))

    with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
        asap_geojson = asap_annotations.as_geojson()
        geojson_out.write(json.dumps(asap_geojson).encode("utf-8"))
        geojson_out.flush()

        geojson_annotations = WsiAnnotations.from_geojson([pathlib.Path(geojson_out.name)])

    _v7_annotations = None

    @property
    def v7_annotations(self):
        if self._v7_annotations is None:
            assert pathlib.Path(pathlib.Path(__file__).parent / "files/103S.json").exists()
            self._v7_annotations = WsiAnnotations.from_darwin_json(pathlib.Path(__file__).parent / "files/103S.json")
        return self._v7_annotations

    def test_conversion_geojson(self):
        # We need to read the asap annotations and compare them to the geojson annotations
        v7_region = self.v7_annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))
        with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
            geojson_out.write(json.dumps(self.v7_annotations.as_geojson()).encode("utf-8"))
            geojson_out.flush()
            annotations = WsiAnnotations.from_geojson([pathlib.Path(geojson_out.name)], sorting="NONE")

        geojson_region = annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))
        assert len(v7_region) == len(geojson_region)

        for elem0, elem1 in zip(v7_region, geojson_region):
            assert elem0 == elem1

    def test_reading_qupath05_geojson_export(self):
        annotations = WsiAnnotations.from_geojson([pathlib.Path("tests/files/qupath05.geojson")])
        assert len(annotations.available_classes) == 2

    def test_asap_to_geojson(self):
        # TODO: Make sure that the annotations hit the border of the region.
        asap_geojson = self.asap_annotations.as_geojson()
        geojson_geojson = self.geojson_annotations.as_geojson()
        assert len(asap_geojson) == len(geojson_geojson)

        # TODO: Collect the geometries together per name and compare
        for elem0, elem1 in zip(asap_geojson["features"], geojson_geojson["features"]):
            assert elem0["type"] == elem1["type"]
            assert elem0["properties"] == elem1["properties"]
            assert elem0["id"] == elem1["id"]

            # Now we need to compare the geometries, given the sorting they could become different
            shape0 = shape(elem0["geometry"], label="")
            shape1 = shape(elem1["geometry"], label="")
            assert len(set([_.label for _ in shape0])) == 1
            assert len(set([_.label for _ in shape1])) == 1
            if isinstance(shape0[0], Polygon):
                complete_shape0 = shapely.geometry.MultiPolygon(shape0)
                complete_shape1 = shapely.geometry.MultiPolygon(shape1)
            else:
                raise NotImplementedError("Different shape types not implemented yet.")

            # Check if the polygons are equal, as they can have a different parametrization
            assert complete_shape0.equals(complete_shape1)

    @pytest.mark.parametrize("region", [((10000, 10000), (5000, 5000), 3756.0), ((0, 0), (5000, 5000), None)])
    def test_read_region(self, region):
        coordinates, size, area = region
        region = self.asap_annotations.read_region(coordinates, 1.0, size)
        if area and area > 0:
            assert len(region) == 1
            assert region[0].area == area
            assert region[0].label == "healthy glands"
            assert isinstance(region[0], Polygon)

        if not area:
            assert region == []

    def test_copy(self):
        copied_annotations = self.asap_annotations.copy()
        # Now we can change a parameter
        copied_annotations.filter([""])
        assert copied_annotations.available_classes != self.asap_annotations.available_classes

    def test_read_darwin_v7(self):
        if not DARWIN_SDK_AVAILABLE:
            return None

        assert len(self.v7_annotations.available_classes) == 5
        assert self.v7_annotations.available_classes[0].label == "ROI (segmentation)"
        assert self.v7_annotations.available_classes[0].annotation_type == AnnotationType.BOX
        assert self.v7_annotations.available_classes[1].label == "stroma (area)"
        assert self.v7_annotations.available_classes[1].annotation_type == AnnotationType.POLYGON
        assert self.v7_annotations.available_classes[2].label == "lymphocyte (cell)"
        assert self.v7_annotations.available_classes[2].annotation_type == AnnotationType.POINT
        assert self.v7_annotations.available_classes[3].label == "tumor (cell)"
        assert self.v7_annotations.available_classes[3].annotation_type == AnnotationType.BOX
        assert self.v7_annotations.available_classes[4].label == "tumor (area)"
        assert self.v7_annotations.available_classes[4].annotation_type == AnnotationType.POLYGON

        assert self.v7_annotations.bounding_box == (
            (15291.49, 18094.48),
            (5122.9400000000005, 4597.509999999998),
        )

        region = self.v7_annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))
        expected_output = [
            (23552767.879399993, "BOX", "ROI (segmentation)"),
            (2417436.551849999, "POLYGON", "stroma (area)"),
            (398284.54274999996, "POLYGON", "stroma (area)"),
            (5124.669950000004, "POLYGON", "stroma (area)"),
            (3516247.3012999967, "POLYGON", "stroma (area)"),
            (0.0, "POINT", "lymphocyte (cell)"),
            (0.0, "POINT", "lymphocyte (cell)"),
            (0.0, "POINT", "lymphocyte (cell)"),
            (141.48809999997553, "BOX", "tumor (cell)"),
            (171.6099999999857, "BOX", "tumor (cell)"),
            (181.86480000002024, "BOX", "tumor (cell)"),
            (100.99830000001499, "BOX", "tumor (cell)"),
            (132.57199999999577, "BOX", "tumor (cell)"),
            (171.38699999998718, "BOX", "tumor (cell)"),
            (7705.718799999956, "POLYGON", "tumor (area)"),
            (10985.104649999945, "POLYGON", "tumor (area)"),
            (585.8433000000017, "BOX", "tumor (cell)"),
        ]

        assert [(_.area, _.annotation_type.value, _.label) for _ in region] == expected_output

    def test_polygon_pickling(self):
        annotation_class = AnnotationClass(
            label="example", annotation_type=AnnotationType.POLYGON, color=(255, 0, 0), z_index=1
        )
        exterior = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole1 = [(1, 1), (2, 1), (2, 2), (1, 2)]
        hole2 = [(3, 3), (3, 3.5), (3.5, 3.5), (3.5, 3)]
        shapely_polygon_with_holes = ShapelyPolygon(exterior, [hole1, hole2])
        dlup_polygon_with_holes = Polygon(shapely_polygon_with_holes, a_cls=annotation_class)
        dlup_solid_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], a_cls=annotation_class)
        with tempfile.NamedTemporaryFile(suffix=".pkl", mode="w+b") as pickled_polygon_file:
            pickle.dump(dlup_solid_polygon, pickled_polygon_file)
            pickled_polygon_file.flush()
            pickled_polygon_file.seek(0)
            loaded_solid_polygon = pickle.load(pickled_polygon_file)
        assert dlup_solid_polygon == loaded_solid_polygon

        with tempfile.NamedTemporaryFile(suffix=".pkl", mode="w+b") as pickled_polygon_file:
            pickle.dump(dlup_polygon_with_holes, pickled_polygon_file)
            pickled_polygon_file.flush()
            pickled_polygon_file.seek(0)
            loaded_polygon_with_holes = pickle.load(pickled_polygon_file)
        assert dlup_polygon_with_holes == loaded_polygon_with_holes

    def test_point_pickling(self):
        annotation_class = AnnotationClass(
            label="example", annotation_type=AnnotationType.POINT, color=(255, 0, 0), z_index=None
        )
        coordinates = [(1, 2)]
        shapely_point = ShapelyPoint(coordinates)
        dlup_point = Point(shapely_point, a_cls=annotation_class)
        with tempfile.NamedTemporaryFile(suffix=".pkl", mode="w+b") as pickled_point_file:
            pickle.dump(shapely_point, pickled_point_file)
            pickled_point_file.flush()
            pickled_point_file.seek(0)
            loaded_point = pickle.load(pickled_point_file)
        assert dlup_point == loaded_point

# Copyright (c) dlup contributors

"""Test the annotation facilities."""
import copy
import json
import pathlib
import pickle
import tempfile

import numpy as np
import pytest

from dlup.annotations_experimental import SlideAnnotations, geojson_to_dlup
from dlup.geometry import Point as Point
from dlup.geometry import Polygon as Polygon
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
        asap_annotations = SlideAnnotations.from_asap_xml(pathlib.Path(asap_file.name))
        asap_annotations.rebuild_rtree()

    with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
        asap_geojson = asap_annotations.as_geojson()
        geojson_out.write(json.dumps(asap_geojson).encode("utf-8"))
        geojson_out.flush()

        geojson_annotations = SlideAnnotations.from_geojson([pathlib.Path(geojson_out.name)])

    _v7_annotations = None
    _v7_raster_annotations = None

    additional_point = Point(*(1, 2), label="example", color=(255, 0, 0))
    additional_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], label="example", color=(255, 0, 0))
    additional_polygon.set_field("z_index", 1)

    @property
    def v7_annotations(self):
        if self._v7_annotations is None:
            assert pathlib.Path(pathlib.Path(__file__).parent / "files/103S.json").exists()
            self._v7_annotations = SlideAnnotations.from_darwin_json(pathlib.Path(__file__).parent / "files/103S.json")
        return self._v7_annotations

    def test_raster_annotations(self):
        if self._v7_raster_annotations is None:
            assert pathlib.Path(pathlib.Path(__file__).parent / "files/raster.json").exists()
            with pytest.raises(NotImplementedError):
                SlideAnnotations.from_darwin_json(pathlib.Path(__file__).parent / "files/raster.json")

    def test_conversion_geojson(self):
        # We need to read the asap annotations and compare them to the geojson annotations
        with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
            geojson_out.write(json.dumps(self.v7_annotations.as_geojson()).encode("utf-8"))
            geojson_out.flush()
            annotations = SlideAnnotations.from_geojson([pathlib.Path(geojson_out.name)], sorting="NONE")

        assert self.v7_annotations.num_points == annotations.num_points
        assert self.v7_annotations.num_polygons == annotations.num_polygons

        assert self.v7_annotations._layers.polygons == annotations._layers.polygons
        assert self.v7_annotations._layers.points == annotations._layers.points

        self.v7_annotations.rebuild_rtree()
        annotations.rebuild_rtree()

        v7_region = self.v7_annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))
        geojson_region = annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))

        assert len(v7_region.polygons) == len(geojson_region.polygons)

        for elem0, elem1 in zip(v7_region.polygons, geojson_region.polygons):
            assert elem0.wkt == elem1.wkt
            assert elem0.label == elem1.label

        for elem0, elem1 in zip(v7_region.points, geojson_region.points):
            assert elem0.wkt == elem1.wkt
            assert elem0.label == elem1.label

    def test_reading_qupath05_geojson_export(self):
        annotations = SlideAnnotations.from_geojson([pathlib.Path("tests/files/qupath05.geojson")])
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
            shape0 = geojson_to_dlup(elem0["geometry"], label="")
            shape1 = geojson_to_dlup(elem1["geometry"], label="")
            assert len(set([_.label for _ in shape0])) == 1
            assert len(set([_.label for _ in shape1])) == 1
            if isinstance(shape0[0], Polygon):
                pass
            else:
                raise NotImplementedError("Different shape types not implemented yet.")

            for p0, p1 in zip(shape0, shape1):
                # The shapes should be equal
                assert p0 == p1

    @pytest.mark.parametrize("region", [((10000, 10000), (5000, 5000), 3756.0), ((0, 0), (5000, 5000), None)])
    def test_read_region(self, region):
        coordinates, size, area = region
        region = self.asap_annotations.read_region(coordinates, 1.0, size)

        polygons = region.polygons

        if area and area > 0:
            assert len(polygons) == 1
            assert polygons[0].area == area
            assert polygons[0].label == "healthy glands"
            assert isinstance(polygons[0], Polygon)

        if not area:
            assert region.polygons == []
            assert region.points == []

    def test_copy(self):
        copied_annotations = copy.copy(self.asap_annotations)

        copied_annotations.tags == self.asap_annotations.tags
        copied_annotations._layers = self.asap_annotations._layers

    def test_pickle(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl") as pkl_file:
            pickle.dump(self.asap_annotations, pkl_file)
            pkl_file.flush()

            with open(pkl_file.name, "rb") as pkl_file:
                annotations = pickle.load(pkl_file)

        assert annotations.tags == self.asap_annotations.tags
        assert annotations._layers == self.asap_annotations._layers

    def test_read_darwin_v7(self):
        if not DARWIN_SDK_AVAILABLE:
            return None

        assert len(self.v7_annotations.available_classes) == 5

        assert "lymphocyte (cell)" in self.v7_annotations
        assert "ROI (segmentation)" in self.v7_annotations
        assert "stroma (area)" in self.v7_annotations
        assert "tumor (cell)" in self.v7_annotations
        assert "tumor (area)" in self.v7_annotations

        assert self.v7_annotations.bounding_box == (
            (15291.49, 18094.48),
            (5122.9400000000005, 4597.509999999998),
        )
        region = self.v7_annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))

        expected_output_polygon = [
            (6250000.0, "ROI (segmentation)"),
            (1616768.0657540853, "stroma (area)"),
            (398284.54274999996, "stroma (area)"),
            (5124.669949999994, "stroma (area)"),
            (103262.97951705182, "stroma (area)"),
            (141.48809999997553, "tumor (cell)"),
            (171.60999999998563, "tumor (cell)"),
            (181.86480000002044, "tumor (cell)"),
            (100.99830000001506, "tumor (cell)"),
            (132.57199999999582, "tumor (cell)"),
            (0.5479999999621504, "tumor (cell)"),
            (7705.718799999958, "tumor (area)"),
            (10985.104649999948, "tumor (area)"),
            (585.8433000000018, "tumor (cell)"),
        ]
        for x, y in zip(region.polygons, expected_output_polygon):
            if x.area <= 1:
                assert np.allclose(x.area, y[0], atol=1e-3)
            else:
                assert np.allclose(x.area, y[0])
            assert x.label == y[1]

        assert [(_.area, _.label) for _ in region.polygons] == expected_output_polygon
        assert len(region.points) == 3

    def test_annotation_filter(self):
        annotations = self.asap_annotations.copy()
        annotations.filter(["healthy glands"])
        assert "healthy glands" in annotations

        annotations.filter_polygons(["non-existing"])
        assert len(annotations._layers.polygons) == 1

    def test_length(self):
        annotations = self.geojson_annotations
        assert len(annotations._layers) == len(annotations) == 1

    def test_dunder_add_methods_with_point(self):
        annotations = self.geojson_annotations.copy()
        initial_annotations_id = id(annotations)
        initial_length = len(annotations)

        # __add__
        new_annotations = annotations + self.additional_point
        assert initial_annotations_id != id(new_annotations)
        assert initial_length + 1 == len(new_annotations)
        assert self.additional_point in new_annotations

        # __radd__
        new_annotations = self.additional_point + annotations
        assert initial_annotations_id != id(new_annotations)
        assert initial_length + 1 == len(new_annotations)
        assert self.additional_point in new_annotations
        with pytest.raises(TypeError):
            self.additional_point += annotations

        # __iadd__
        annotations += self.additional_point
        assert initial_annotations_id == id(annotations)
        assert initial_length + 1 == len(annotations)
        assert self.additional_point in annotations

    def test_add_with_polygon(self):
        annotations = self.geojson_annotations.copy()
        initial_annotations_id = id(annotations)
        initial_length = len(annotations)

        # __add__
        new_annotations = annotations + self.additional_polygon
        assert initial_annotations_id != id(new_annotations)
        assert initial_length + 1 == len(new_annotations)
        assert self.additional_polygon in new_annotations

        # __radd__
        new_annotations = self.additional_polygon + annotations
        assert initial_annotations_id != id(new_annotations)
        assert initial_length + 1 == len(new_annotations)
        assert self.additional_polygon in new_annotations
        with pytest.raises(TypeError):
            self.additional_polygon += annotations

        # __iadd__
        annotations += self.additional_polygon
        assert initial_annotations_id == id(annotations)
        assert initial_length + 1 == len(annotations)
        assert self.additional_polygon in annotations

    def test_add_with_list(self):
        annotations = self.geojson_annotations.copy()
        initial_annotations_id = id(annotations)
        initial_length = len(annotations)

        # __add__
        new_annotations = annotations + [self.additional_point, self.additional_polygon]
        assert initial_annotations_id != id(new_annotations)
        assert initial_length + 2 == len(new_annotations)
        assert self.additional_polygon in new_annotations
        assert self.additional_point in new_annotations

        # __radd__
        _annotations_list = [self.additional_point, self.additional_polygon]
        with pytest.raises(TypeError):
            new_annotations = _annotations_list + annotations

        _annotations_list = [self.additional_point, self.additional_polygon]
        with pytest.raises(TypeError):
            _annotations_list += annotations

        # __iadd__
        annotations += [self.additional_point, self.additional_polygon]
        assert initial_annotations_id == id(annotations)
        assert initial_length + 2 == len(annotations)
        assert all(ann in new_annotations for ann in annotations)

    def test_add_with_wsi_annotations(self):
        annotations = self.geojson_annotations.copy()
        other_annotations = self.geojson_annotations.copy()
        initial_annotations_id = id(annotations)
        initial_length = len(annotations)

        # __add__
        new_annotations = annotations + other_annotations
        assert initial_annotations_id != id(new_annotations)
        assert len(annotations) + len(other_annotations) == len(new_annotations)
        assert all(ann in new_annotations for ann in annotations)

        # __iadd__
        annotations += other_annotations
        assert initial_annotations_id == id(annotations)
        assert initial_length + len(other_annotations) == len(annotations)
        assert all(ann in annotations for ann in other_annotations)

    def test_add_with_invalid_type(self):
        annotations = self.geojson_annotations.copy()
        with pytest.raises(TypeError):
            _ = annotations + "invalid type"
        with pytest.raises(TypeError):
            annotations += "invalid type"
        with pytest.raises(TypeError):
            _ = "invalid type" + annotations

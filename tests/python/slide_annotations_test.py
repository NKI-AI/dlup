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
"""Test the annotation facilities."""

import json
import pathlib
import pickle
import tempfile
import warnings

import os
import numpy as np
import pytest
from dlup.annotations import SlideAnnotations
from dlup.annotations.importers.geojson import geojson_to_dlup
from dlup.geometry import Box, Point, Polygon
from dlup.utils.imports import DARWIN_SDK_AVAILABLE

# Define test files path
TEST_FILES_PATH = pathlib.Path(os.getenv("TEST_TARGET").lstrip("/").split(":")[0]) / "files"

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


DLUP_XML_EXAMPLE = b"""<DlupAnnotations version="1.0">
    <Metadata>
        <ImageID>IMG_12345</ImageID>
        <Description>Sample annotations with polygons, multipolygons, points, and boxes.</Description>
        <Version>1.0</Version>
        <Authors>
            <Author>Jane Doe</Author>
            <Author>John Smith</Author>
        </Authors>
        <DateCreated>2024-08-19</DateCreated>
        <Software>dlup v0.9.0</Software>
    </Metadata>

    <Tags>
        <Tag label="test" color="#FF5733">
            <Attribute color="#336699">Attribute 1</Attribute>
            <Attribute>Attribute 2</Attribute>
            <Text>This is the single text field for this tag.</Text>
        </Tag>
    </Tags>

    <Geometries>
        <!-- Polygon, MultiPolygon and Box can appear in any arbitrary order-->
        <Polygon label="Polygon1" color="#FF5733" order="0">
            <Exterior>
                <Point x="0.0" y="0.0"/>
                <Point x="4.0" y="0.0"/>
                <Point x="4.0" y="4.0"/>
                <Point x="0.0" y="4.0"/>
                <Point x="0.0" y="0.0"/>
            </Exterior>
            <Interiors>
                <Interior>
                    <Point x="0.5" y="0.5"/>
                    <Point x="3.5" y="0.5"/>
                    <Point x="3.5" y="3.5"/>
                    <Point x="0.5" y="3.5"/>
                    <Point x="0.5" y="0.5"/>
                </Interior>
            </Interiors>
        </Polygon>

        <Box xMin="5.0" yMin="5.0" xMax="10.0" yMax="10.0" label="Box1" color="#33FF57" order="1" />

        <!-- All Points are at the bottom -->
        <Point x="1.0" y="2.0" label="Point1" color="#FF5733"/>
        <Point x="3.0" y="4.0" label="Point2" color="#33FF57"/>

    </Geometries>
</DlupAnnotations>
"""


polygons = [
    Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], []),
    Polygon([(2, 2), (2, 5), (5, 5), (5, 2)], []),
    Polygon([(4, 2), (4, 7), (7, 7), (7, 4)], []),
    Polygon([(6, 6), (6, 9), (9, 9), (9, 6)], []),
]


class TestAnnotations:
    with tempfile.NamedTemporaryFile(suffix=".xml") as asap_file:
        asap_file.write(ASAP_XML_EXAMPLE)
        asap_file.flush()
        asap_annotations = SlideAnnotations.from_asap_xml(pathlib.Path(asap_file.name))
        asap_annotations.rebuild_rtree()

    with tempfile.NamedTemporaryFile(suffix=".xml") as dlup_file:
        dlup_file.write(DLUP_XML_EXAMPLE)
        dlup_file.flush()
        dlup_annotations = SlideAnnotations.from_dlup_xml(pathlib.Path(dlup_file.name))

    with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
        asap_geojson = asap_annotations.as_geojson()
        geojson_out.write(json.dumps(asap_geojson).encode("utf-8"))
        geojson_out.flush()

        geojson_annotations = SlideAnnotations.from_geojson(pathlib.Path(geojson_out.name))

    _v7_annotations = None
    _v7_raster_annotations = None

    _halo_annotations = None

    additional_point = Point(*(1, 2), label="example", color=(255, 0, 0))
    additional_polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], label="example", color=(255, 0, 0))
    additional_polygon.set_field("z_index", 1)

    @property
    def v7_annotations(self):
        if self._v7_annotations is None:
            file_path = TEST_FILES_PATH / "103S.json"
            assert file_path.exists()
            self._v7_annotations = SlideAnnotations.from_darwin_json(file_path, roi_names=["ROI (segmentation)"])
        return self._v7_annotations

    @property
    def halo_annotations(self):
        if self._halo_annotations is None:
            file_path = TEST_FILES_PATH / "halo_holes.annotations"
            assert file_path.exists()
            self._halo_annotations = SlideAnnotations.from_halo_xml(
                file_path,
                box_as_polygon=False,
            )
        return self._halo_annotations

    def test_raster_annotations(self):
        if not DARWIN_SDK_AVAILABLE:
            pytest.skip("Darwin SDK is not available, skipping test.")

        if self._v7_raster_annotations is None:
            file_path = TEST_FILES_PATH / "raster.json"
            assert file_path.exists()
            with pytest.raises(NotImplementedError):
                SlideAnnotations.from_darwin_json(file_path)

    def test_conversion_geojson_v7(self):
        # We need to read the asap annotations and compare them to the geojson annotations
        with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
            geojson_out.write(json.dumps(self.v7_annotations.as_geojson()).encode("utf-8"))
            geojson_out.flush()
            annotations = SlideAnnotations.from_geojson(pathlib.Path(geojson_out.name), sorting="NONE")

        assert self.v7_annotations.num_points == annotations.num_points
        assert self.v7_annotations.num_polygons == annotations.num_polygons

        assert self.v7_annotations.polygons == annotations.polygons
        assert self.v7_annotations.points == annotations.points

        self.v7_annotations.rebuild_rtree()
        annotations.rebuild_rtree()

        v7_region = self.v7_annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))
        geojson_region = annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))

        assert len(v7_region.polygons.get_geometries()) == len(geojson_region.polygons.get_geometries())

        for elem0, elem1 in zip(v7_region.polygons.get_geometries(), geojson_region.polygons.get_geometries()):
            assert elem0.wkt == elem1.wkt
            assert elem0.label == elem1.label
            elem0.index = 1
            elem1.index = 1

        assert np.allclose(v7_region.polygons.to_mask().numpy(), geojson_region.polygons.to_mask().numpy())

        for elem0, elem1 in zip(v7_region.points, geojson_region.points):
            assert elem0.wkt == elem1.wkt
            assert elem0.label == elem1.label

    def test_conversion_halo_geojson(self):
        # We read the halo annotations and compare them to the geojson annotations
        with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
            geojson_out.write(json.dumps(self.halo_annotations.as_geojson()).encode("utf-8"))
            geojson_out.flush()
            geojson_annotations = SlideAnnotations.from_geojson(pathlib.Path(geojson_out.name), sorting="NONE")
            geojson_annotations.offset_function = self.halo_annotations.offset_function

        assert self.halo_annotations.num_points == geojson_annotations.num_points
        assert self.halo_annotations.num_polygons == geojson_annotations.num_polygons
        assert self.halo_annotations.polygons == geojson_annotations.polygons
        assert self.halo_annotations.points == geojson_annotations.points
        # This won't work because we have no boxes in GeoJSON
        assert self.halo_annotations.boxes != geojson_annotations.boxes
        # assert self.halo_annotations.__eq__(geojson_annotations)

    def test_halo_annotations(self):
        halo_annotations = self.halo_annotations.copy()
        bounding_box = halo_annotations.bounding_box
        assert bounding_box[0] == (-29349.0, 49265.0)
        halo_annotations.set_offset((29349.0, -49265.0))
        assert halo_annotations.bounding_box[0] == (0, 0)

        for polygon in halo_annotations.polygons:
            polygon.index = 1

        new_bbox = halo_annotations.bounding_box_at_scaling(0.01)
        halo_annotations.rebuild_rtree()
        region = halo_annotations.read_region(new_bbox[0], 0.01, new_bbox[1])
        output_color_mask = halo_annotations.color_lut[region.polygons.to_mask().numpy()]
        assert output_color_mask.sum() == 51327280

    def test_slidescore_annotations(self):
        file_path = TEST_FILES_PATH / "slidescore_annotation_test.txt"
        assert file_path.exists()
        color_map = {
            "Polygon Annotations": (255, 0, 0),
            "Ellipse Annotations": (0, 255, 0),
            "Point Annotations": (0, 0, 255),
        }
        z_indices = {
            "Polygon Annotations": 1,
            "Ellipse Annotations": 2,
        }
        annotations = SlideAnnotations.from_file_path(
            file_path, reader="slidescore_tsv", box_as_polygon=True, color_map=color_map, z_indices=z_indices
        )

        for polygon in annotations.polygons:
            assert polygon.color == color_map[polygon.label]
            assert polygon.index == z_indices[polygon.label]

        for point in annotations.points:
            assert point.color == color_map[point.label]

        assert annotations.num_points == 4
        # 5 polygons + 2 ellipses + 1 brush (as polygon with holes)
        assert annotations.num_polygons == 8

        # Check labels using available_labels property
        available_labels = annotations.available_classes
        assert "Polygon Annotations" in available_labels
        assert "Ellipse Annotations" in available_labels
        assert "Point Annotations" in available_labels

        # Check that at least one polygon is an ellipse approximation
        found_ellipse = any(p.get_field("_ellipse_approximation") for p in annotations.polygons)
        assert found_ellipse

        # Export to slidescore TSV and re-import
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_tsv:
            temp_tsv.write(
                annotations.as_slidescore_tsv(
                    image_id=1234, image_name="Test Image", user_email="j.doe@example.com"
                ).encode("utf-8")
            )
            temp_tsv.flush()
            reimported = SlideAnnotations.from_file_path(
                temp_tsv.name, reader="slidescore_tsv", box_as_polygon=True, color_map=color_map, z_indices=z_indices
            )

        assert reimported.num_points == annotations.num_points
        assert reimported.num_polygons == annotations.num_polygons
        assert reimported.available_classes == annotations.available_classes
        assert reimported.polygons == annotations.polygons
        assert reimported.points == annotations.points
        assert reimported.boxes == annotations.boxes

    def test_slidescore_json_annotations(self):
        file_path = TEST_FILES_PATH / "slidescore_annotation_test.json"
        assert file_path.exists()

        annotations = SlideAnnotations.from_file_path(file_path, reader="slidescore_json", box_as_polygon=True)

        assert annotations.num_points == 2
        # 1 rect (as polygon) + 1 brush (as polygon with a hole)
        assert annotations.num_polygons == 2
        assert "Point Annotations" in annotations.available_classes
        assert "Rect Annotations" in annotations.available_classes
        assert "Brush Annotations" in annotations.available_classes

        assert annotations.metadata["slidescore_image_id"] == 1234
        assert annotations.metadata["slidescore_study_id"] == 1
        assert annotations.metadata["slidescore_image_name"] == "Test Image"

        assert annotations.tags is not None
        assert any(tag.label == "Stain quality" for tag in annotations.tags)

    def test_conversion_slidescore_geojson(self):
        file_path = TEST_FILES_PATH / "slidescore_annotation_test.txt"
        assert file_path.exists()
        color_map = {
            "Polygon Annotations": (255, 0, 0),
            "Ellipse Annotations": (0, 255, 0),
            "Point Annotations": (0, 0, 255),
        }
        z_indices = {
            "Polygon Annotations": 1,
            "Ellipse Annotations": 2,
        }
        slidescore_annotations = SlideAnnotations.from_file_path(
            file_path, reader="slidescore_tsv", box_as_polygon=True, color_map=color_map, z_indices=z_indices
        )

        with tempfile.NamedTemporaryFile(suffix=".xml") as dlup_xml_out:
            with open(dlup_xml_out.name, "w") as f:
                f.write(slidescore_annotations.as_dlup_xml())
            dlup_annotations = SlideAnnotations.from_dlup_xml(pathlib.Path(dlup_xml_out.name))

        # Number of points and polygons should be preserved
        assert slidescore_annotations.num_points == dlup_annotations.num_points
        assert slidescore_annotations.num_polygons == dlup_annotations.num_polygons

        # Polygons and points should be equal (ellipses are polygons after round-trip)
        # Remove ellipse-specific attributes from both slidescore_annotations and dlup_annotations before comparison
        ellipse_attrs = ["_ellipse_approximation", "_ellipse_center", "_ellipse_size"]
        for polygons in (slidescore_annotations.polygons, dlup_annotations.polygons):
            for poly in polygons:
                for attr in ellipse_attrs:
                    poly.set_field(attr, None)

        assert slidescore_annotations.polygons == dlup_annotations.polygons
        assert slidescore_annotations.points == dlup_annotations.points
        assert slidescore_annotations.boxes == dlup_annotations.boxes

    def test_reexpert_dlup_xml(self):
        with tempfile.NamedTemporaryFile(suffix=".xml") as dlup_file:
            with open(dlup_file.name, "w") as f:
                f.write(self.dlup_annotations.as_dlup_xml())

            annotations = SlideAnnotations.from_dlup_xml(dlup_file.name)
            assert self.dlup_annotations == annotations
            assert self.dlup_annotations.tags == annotations.tags
            assert self.dlup_annotations.sorting == annotations.sorting
            assert self.dlup_annotations.offset_function == annotations.offset_function

    @pytest.mark.parametrize("scaling", [0.5, 0.3, 1.0])
    def test_read_region_point_scaling(self, scaling):
        coordinates, size = self.dlup_annotations.bounding_box_at_scaling(scaling)
        base_region = self.dlup_annotations.read_region(
            self.dlup_annotations.bounding_box[0], 1.0, self.dlup_annotations.bounding_box[1]
        )
        region = self.dlup_annotations.read_region(coordinates, scaling, size)

        for base_point, point in zip(base_region.points, region.points):
            assert np.allclose(np.asarray(base_point.coordinates) * scaling, np.asarray(point.coordinates))

    def test_reading_qupath05_geojson_export(self):
        annotations = SlideAnnotations.from_geojson(TEST_FILES_PATH / "qupath05.geojson")
        assert len(annotations.available_classes) == 2

    @pytest.mark.parametrize(
        "class_method",
        [
            "from_geojson",
            "from_halo_xml",
            "from_dlup_xml",
            "from_asap_xml",
            "from_slidescore_tsv",
            "from_slidescore_json",
        ],
    )
    def test_missing_file_constructor(self, class_method):
        constructor = getattr(SlideAnnotations, class_method)
        with pytest.raises(FileNotFoundError):
            constructor("doesnotexist.xml.json")

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

        polygons = region.polygons.get_geometries()

        if area and area > 0:
            assert len(polygons) == 1
            assert polygons[0].area == area
            assert polygons[0].label == "healthy glands"
            assert isinstance(polygons[0], Polygon)

        if not area:
            assert region.polygons.get_geometries() == []
            assert region.points == []

    def test_pickle_slideannotations(self):
        roi = Box((0, 0), (10, 10), label="ROI (segmentation)").as_polygon()
        collection = SlideAnnotations()
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

    def test_pickle_inline_created_point(self):
        """Test that inline-created points can be pickled correctly.

        This test verifies the fix for the bug where inline-created geometries
        (created directly in add_point/add_polygon calls) would fail to pickle
        because they were stored as raw C++ _dg.* objects instead of Python wrappers.
        """
        ann = SlideAnnotations()
        ann.add_point(Point(100, 200, label="inline"))

        # Should not raise TypeError
        pickled = pickle.dumps(ann)
        unpickled = pickle.loads(pickled)

        assert len(unpickled.points) == 1
        assert unpickled.points[0].x == 100
        assert unpickled.points[0].y == 200
        assert unpickled.points[0].label == "inline"

    def test_pickle_inline_created_polygon(self):
        """Test that inline-created polygons can be pickled correctly."""
        ann = SlideAnnotations()
        coords = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        ann.add_polygon(Polygon(coords, label="inline_poly"))

        # Should not raise TypeError
        pickled = pickle.dumps(ann)
        unpickled = pickle.loads(pickled)

        assert len(unpickled.polygons) == 1
        assert unpickled.polygons[0].label == "inline_poly"
        assert unpickled.polygons[0].area == 10000.0

    def test_pickle_inline_created_box(self):
        """Test that inline-created boxes can be pickled correctly."""
        ann = SlideAnnotations()
        ann.add_box(Box((50, 50), (100, 100), label="inline_box"))

        # Should not raise TypeError
        pickled = pickle.dumps(ann)
        unpickled = pickle.loads(pickled)

        assert len(unpickled.boxes) == 1
        assert unpickled.boxes[0].label == "inline_box"
        assert unpickled.boxes[0].coordinates == (50, 50)
        assert unpickled.boxes[0].size == (100, 100)

    def test_pickle_mixed_creation_methods(self):
        """Test pickling with a mix of pre-created and inline-created geometries."""
        ann = SlideAnnotations()

        # Add inline point
        ann.add_point(Point(50, 50, label="point1"))

        # Add pre-created point
        p = Point(150, 150, label="point2")
        ann.add_point(p)

        # Add inline polygon
        coords = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        ann.add_polygon(Polygon(coords, label="poly1"))

        # Add pre-created polygon
        poly2 = Polygon([(200, 200), (300, 200), (300, 300), (200, 300), (200, 200)], label="poly2")
        ann.add_polygon(poly2)

        # Should not raise TypeError
        pickled = pickle.dumps(ann)
        unpickled = pickle.loads(pickled)

        assert len(unpickled.points) == 2
        assert len(unpickled.polygons) == 2
        assert set(p.label for p in unpickled.points) == {"point1", "point2"}
        assert set(p.label for p in unpickled.polygons) == {"poly1", "poly2"}

    def test_pickle_with_mpp(self):
        """Test that the mpp attribute is preserved during pickling."""
        ann = SlideAnnotations(mpp=0.5)
        ann.add_point(Point(100, 200, label="test"))

        pickled = pickle.dumps(ann)
        unpickled = pickle.loads(pickled)

        assert unpickled.mpp == 0.5
        assert len(unpickled.points) == 1

    def test_reindex_polygons(self):
        ann = self.dlup_annotations.copy()
        ann.reindex_polygons({"Polygon1": 7})
        for polygon in ann.polygons:
            assert polygon.index == 7

        assert ann.index_map == {"Polygon1": 7}

        ann.reindex_polygons({"Polygon1": 8})
        assert ann.index_map == {"Polygon1": 8}

    def test_relabel_polygons(self):
        ann = self.dlup_annotations.copy()
        ann.relabel_polygons({"Polygon1": "Polygon2"})
        for polygon in ann.polygons:
            assert polygon.label == "Polygon2"

    @pytest.mark.parametrize("scaling", [0.5, 0.3, 1.0])
    def test_bounding_box(self, scaling):
        if not DARWIN_SDK_AVAILABLE:
            pytest.skip("Darwin SDK is not available, skipping test.")

        assert self.v7_annotations.bounding_box == (
            (15291.49, 18094.48),
            (5122.9400000000005, 4597.509999999998),
        )

        assert self.v7_annotations.bounding_box_at_scaling(scaling) == (
            (15291.49 * scaling, 18094.48 * scaling),
            (5122.9400000000005 * scaling, 4597.509999999998 * scaling),
        )

    def test_read_darwin_v7(self):
        if not DARWIN_SDK_AVAILABLE:
            pytest.skip("Darwin SDK is not available, skipping test.")

        assert len(self.v7_annotations.available_classes) == 4

        assert "lymphocyte (cell)" in self.v7_annotations
        assert "stroma (area)" in self.v7_annotations
        assert "tumor (cell)" in self.v7_annotations
        assert "tumor (area)" in self.v7_annotations

        assert self.v7_annotations.bounding_box == (
            (15291.49, 18094.48),
            (5122.9400000000005, 4597.509999999998),
        )

        region = self.v7_annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))

        expected_output_polygon = [
            (1616768.0657540853, "stroma (area)"),
            (398284.54274999996, "stroma (area)"),
            (5124.669949999994, "stroma (area)"),
            (103262.97951705182, "stroma (area)"),
            (7705.718799999958, "tumor (area)"),
            (10985.104649999948, "tumor (area)"),
        ]

        expected_output_boxes = [
            (141.48809999997553, "tumor (cell)"),
            (171.60999999998563, "tumor (cell)"),
            (181.86480000002044, "tumor (cell)"),
            (100.99830000001506, "tumor (cell)"),
            (132.57199999999582, "tumor (cell)"),
            (171.38699999998718, "tumor (cell)"),
            (585.8433000000018, "tumor (cell)"),
        ]

        assert len(region.rois.get_geometries()) == 1
        assert region.rois.get_geometries()[0].area == 6250000.0
        assert region.rois.get_geometries()[0].label == "ROI (segmentation)"

        for x, y in zip(region.polygons.get_geometries(), expected_output_polygon):
            assert np.allclose(x.area, y[0], atol=1e-3)
            assert x.label == y[1]
        assert len(region.points) == 3

        for x, y in zip(region.boxes, expected_output_boxes):
            assert np.allclose(x.as_polygon().area, y[0], atol=1e-3)
            assert x.label == y[1]

    def test_annotation_filter(self):
        annotations = self.asap_annotations.copy()
        annotations.filter(["healthy glands"])
        assert "healthy glands" in annotations

        annotations.filter_polygons(["non-existing"])
        assert len(annotations.polygons) == 1

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

    def test_add_with_slide_annotations(self):
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

    def test_v7_metadata(self, monkeypatch):
        if DARWIN_SDK_AVAILABLE:
            from dlup.annotations.importers.darwin_json import get_v7_metadata

            with pytest.raises(ValueError):
                get_v7_metadata(pathlib.Path("../tests"))

            # Test the case where DARWIN_SDK_AVAILABLE is False
            monkeypatch.setattr("dlup.annotations.importers.darwin_json.DARWIN_SDK_AVAILABLE", False)
            with pytest.warns(UserWarning):
                result = get_v7_metadata(pathlib.Path("."))
            assert result is None
        else:
            # Skip this test if DARWIN_SDK_AVAILABLE is False
            pytest.skip("Darwin SDK not available")

    @pytest.mark.parametrize("sorting_type", ["NONE", "REVERSE", "AREA", "Z_INDEX", "NON_EXISTENT"])
    # TODO: We need to update
    def test_sorting(self, sorting_type):
        collection = SlideAnnotations()
        for polygon in polygons:
            collection.add_polygon(polygon)

        if sorting_type == "NONE":
            curr_collection = collection.__deepcopy__(None)
            collection._in_place_sort_and_scale(scaling=1.0, sorting=sorting_type)
            assert curr_collection == collection

        if sorting_type == "REVERSE":
            with pytest.raises(NotImplementedError):
                curr_collection = collection.__deepcopy__(None)
                collection._in_place_sort_and_scale(scaling=1.0, sorting=sorting_type)

        if sorting_type == "Z_INDEX":
            curr_collection = collection.__copy__()
            for idx, polygon in enumerate(curr_collection.polygons):
                polygon.set_field("z_index", len(curr_collection.polygons) - idx)
            collection._in_place_sort_and_scale(scaling=1.0, sorting=sorting_type)
            assert curr_collection.polygons == collection.polygons[::-1]

        if sorting_type == "AREA":
            new_collection = SlideAnnotations()
            polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)], [])
            polygon2 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)], [])
            polygon3 = Polygon([(0, 0), (0, 6), (6, 6), (6, 0)], [])
            new_collection.add_polygon(polygon)
            new_collection.add_polygon(polygon2)
            new_collection.add_polygon(polygon3)
            curr_areas = [_.area for _ in new_collection.polygons]
            assert curr_areas == [9.0, 4.0, 36.0]
            new_collection._in_place_sort_and_scale(scaling=1.0, sorting="AREA")
            assert [_.area for _ in new_collection.polygons] == [36.0, 9.0, 4.0]

        if sorting_type == "NON_EXISTENT":
            with pytest.raises(KeyError):
                collection._in_place_sort_and_scale(scaling=1.0, sorting=sorting_type)

    def test_halo_annotations_with_pins(self):
        file_path = TEST_FILES_PATH / "test_different_types_halo.annotations"
        assert file_path.exists()
        annotations = SlideAnnotations.from_halo_xml(file_path)
        assert len(annotations.polygons) == 5  # 2 ellipses, 3 polygons
        assert len(annotations.points) == 1

    def test_has_rois_property(self):
        """Test that the has_rois property is correctly implemented in SlideAnnotations."""
        # Create a new SlideAnnotations object
        annotations = SlideAnnotations()

        # Initially, it should have no ROIs
        assert not annotations.has_rois

        # Add a regular polygon (not an ROI)
        annotations.add_polygon(Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]))

        # It should still have no ROIs
        assert not annotations.has_rois

        # Add an ROI
        roi = Polygon([(0, 0), (0, 100), (100, 100), (100, 0)], label="ROI")
        annotations.add_roi(roi)

        # Now it should have ROIs
        assert annotations.has_rois

        # Test that read_region uses the has_rois property correctly
        annotations.rebuild_rtree()
        region = annotations.read_region((0, 0), 1.0, (100, 100))

        # The AnnotationRegion should have has_rois=True
        assert region.has_rois

        # Create another collection without ROIs
        annotations_no_rois = SlideAnnotations()
        annotations_no_rois.add_polygon(Polygon([(0, 0), (0, 100), (100, 100), (100, 0)]))

        # Test that has_rois is False
        assert not annotations_no_rois.has_rois

        # Test that the region has has_rois set correctly
        annotations_no_rois.rebuild_rtree()
        region_no_rois = annotations_no_rois.read_region((0, 0), 1.0, (100, 100))

        # The AnnotationRegion should have has_rois=False
        assert not region_no_rois.has_rois

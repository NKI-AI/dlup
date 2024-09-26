# Copyright (c) dlup contributors
# pylint: disable=no-member
"""Test the annotation facilities."""
import copy
import json
import os
import pathlib
import pickle
import tempfile

import numpy as np
import pytest

from dlup.annotations import GeometryCollection, SlideAnnotations
from dlup.annotations.importers.darwin_json import get_v7_metadata
from dlup.annotations.importers.geojson import geojson_to_dlup
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

# <MultiPolygon label="MultiPolygon1" color="#3357FF" order="1">
#     <Polygon>
#         <Exterior>
#             <Point x="10.0" y="10.0"/>
#             <Point x="14.0" y="10.0"/>
#             <Point x="14.0" y="14.0"/>
#             <Point x="10.0" y="14.0"/>
#             <Point x="10.0" y="10.0"/>
#         </Exterior>
#     </Polygon>
#     <Polygon>
#         <Exterior>
#             <Point x="20.0" y="20.0"/>
#             <Point x="24.0" y="20.0"/>
#             <Point x="24.0" y="24.0"/>
#             <Point x="20.0" y="24.0"/>
#             <Point x="20.0" y="20.0"/>
#         </Exterior>
#     </Polygon>
# </MultiPolygon>


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
        <Software>dlup v0.8.0</Software>
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
            assert pathlib.Path(pathlib.Path(__file__).parent / "files/103S.json").exists()
            self._v7_annotations = SlideAnnotations.from_darwin_json(pathlib.Path(__file__).parent / "files/103S.json")
        return self._v7_annotations

    @property
    def halo_annotations(self):
        if self._halo_annotations is None:
            assert pathlib.Path(pathlib.Path(__file__).parent / "files/halo_holes.annotations").exists()
            self._halo_annotations = SlideAnnotations.from_halo_xml(
                pathlib.Path(__file__).parent / "files/halo_holes.annotations",
                box_as_polygon=False,
            )
        return self._halo_annotations

    def test_raster_annotations(self):
        if self._v7_raster_annotations is None:
            assert pathlib.Path(pathlib.Path(__file__).parent / "files/raster.json").exists()
            with pytest.raises(NotImplementedError):
                SlideAnnotations.from_darwin_json(pathlib.Path(__file__).parent / "files/raster.json")

    def test_conversion_geojson_v7(self):
        # We need to read the asap annotations and compare them to the geojson annotations
        with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
            geojson_out.write(json.dumps(self.v7_annotations.as_geojson()).encode("utf-8"))
            geojson_out.flush()
            annotations = SlideAnnotations.from_geojson(pathlib.Path(geojson_out.name), sorting="NONE")

        assert self.v7_annotations.num_points == annotations.num_points
        assert self.v7_annotations.num_polygons == annotations.num_polygons

        assert self.v7_annotations.layers.polygons == annotations.layers.polygons
        assert self.v7_annotations.layers.points == annotations.layers.points

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

        assert np.allclose(v7_region.polygons.to_mask(), geojson_region.polygons.to_mask())

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
        assert self.halo_annotations.layers.polygons == geojson_annotations.layers.polygons
        assert self.halo_annotations.layers.points == geojson_annotations.layers.points
        # This won't work because we have no boxes in GeoJSON
        assert self.halo_annotations.layers.boxes != geojson_annotations.layers.boxes
        # assert self.halo_annotations.__eq__(geojson_annotations)

    def test_halo_annotations(self):
        halo_annotations = self.halo_annotations.copy()
        bounding_box = halo_annotations.bounding_box
        assert bounding_box[0] == (-29349.0, 50000.55808864343)
        halo_annotations.set_offset((29349.0, -50000.55808864343))
        assert halo_annotations.bounding_box[0] == (0, 0)

        for polygon in halo_annotations.layers.polygons:
            polygon.index = 1

        new_bbox = halo_annotations.bounding_box_at_scaling(0.01)
        region = halo_annotations.read_region(new_bbox[0], 0.01, new_bbox[1])
        output_color_mask = halo_annotations.color_lut[region.polygons.to_mask().numpy()]
        assert output_color_mask.sum() == 51485183

    def test_reexpert_dlup_xml(self):
        with tempfile.NamedTemporaryFile(suffix=".xml") as dlup_file:
            with open(dlup_file.name, "w") as f:
                f.write(self.dlup_annotations.as_dlup_xml())

            annotations = SlideAnnotations.from_dlup_xml(dlup_file.name)
            assert self.dlup_annotations._layers == annotations._layers
            assert self.dlup_annotations.tags == annotations.tags
            assert self.dlup_annotations.sorting == annotations.sorting
            assert self.dlup_annotations.offset_function == annotations.offset_function
            assert self.dlup_annotations == annotations

    def test_reading_qupath05_geojson_export(self):
        annotations = SlideAnnotations.from_geojson(pathlib.Path("tests/files/qupath05.geojson"))
        assert len(annotations.available_classes) == 2

    @pytest.mark.parametrize("class_method", ["from_geojson", "from_halo_xml", "from_dlup_xml", "from_asap_xml"])
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

    def test_reindex_polygons(self):
        ann = self.dlup_annotations.copy()
        ann.reindex_polygons({"Polygon1": 7})
        for polygon in ann._layers.polygons:
            assert polygon.index == 7

    def test_relabel_polygons(self):
        ann = self.dlup_annotations.copy()
        ann.relabel_polygons({"Polygon1": "Polygon2"})
        for polygon in ann._layers.polygons:
            assert polygon.label == "Polygon2"

    @pytest.mark.parametrize("scaling", [0.5, 0.3, 1.0])
    def test_bounding_box(self, scaling):
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
        for x, y in zip(region.polygons.get_geometries(), expected_output_polygon):
            if os.environ.get("GITHUB_ACTIONS", False):
                if x.area <= 1:
                    assert np.allclose(x.area, y[0], atol=1e-3)
                else:
                    assert np.allclose(x.area, y[0])
            else:
                assert [(_.area, _.label) for _ in region.polygons.get_geometries()] == expected_output_polygon
            assert x.label == y[1]
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
        with pytest.raises(ValueError):
            get_v7_metadata(pathlib.Path("../tests"))

        monkeypatch.setattr("dlup.annotations.importers.darwin_json.DARWIN_SDK_AVAILABLE", False)
        with pytest.raises(ImportError):
            get_v7_metadata(pathlib.Path("."))

    @pytest.mark.parametrize("sorting_type", ["NONE", "REVERSE", "AREA", "Z_INDEX", "NON_EXISTENT"])
    def test_sorting(self, sorting_type):
        collection = GeometryCollection()
        for polygon in polygons:
            collection.add_polygon(polygon)

        if sorting_type == "NONE":
            curr_collection = collection.__copy__()
            SlideAnnotations._in_place_sort_and_scale(curr_collection, scaling=1.0, sorting=sorting_type)
            assert curr_collection == collection

        if sorting_type == "REVERSE":
            with pytest.raises(NotImplementedError):
                curr_collection = collection.__copy__()
                SlideAnnotations._in_place_sort_and_scale(curr_collection, scaling=1.0, sorting=sorting_type)
            # Needs fixing
            # assert curr_collection.polygons == collection.polygons[::-1]

        if sorting_type == "Z_INDEX":
            curr_collection = collection.__copy__()
            for idx, polygon in enumerate(curr_collection.polygons):
                polygon.set_field("z_index", len(curr_collection.polygons) - idx)
            SlideAnnotations._in_place_sort_and_scale(curr_collection, scaling=1.0, sorting=sorting_type)
            assert curr_collection.polygons == collection.polygons[::-1]

        if sorting_type == "NON_EXISTENT":
            with pytest.raises(KeyError):
                SlideAnnotations._in_place_sort_and_scale(collection, scaling=1.0, sorting=sorting_type)

    def test_halo_annotations_with_pins(self):
        assert pathlib.Path(pathlib.Path(__file__).parent / "files/test_different_types_halo.annotations").exists()
        annotations = SlideAnnotations.from_halo_xml(
            pathlib.Path(__file__).parent / "files/test_different_types_halo.annotations"
        )
        assert len(annotations.layers.polygons) == 5  # 2 ellipses, 3 polygons
        assert len(annotations.layers.points) == 1

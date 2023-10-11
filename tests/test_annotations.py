# coding=utf-8
# Copyright (c) dlup contributors

"""Test the annotation facilities."""

import json
import pathlib
import tempfile

import numpy as np
import pytest

from dlup.annotations import AnnotationClass, AnnotationType, Polygon, WsiAnnotations
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
        asap_geojson = asap_annotations.as_geojson(split_per_label=False)
        geojson_out.write(json.dumps(asap_geojson).encode("utf-8"))
        geojson_out.flush()

        geojson_annotations = WsiAnnotations.from_geojson([pathlib.Path(geojson_out.name)], scaling=1)

    _v7_annotations = None

    @property
    def v7_annotations(self):
        if self._v7_annotations is None:
            assert pathlib.Path("tests/files/103S.json").exists()
            self._v7_annotations = WsiAnnotations.from_darwin_json("tests/files/103S.json")
        return self._v7_annotations

    def test_asap_to_geojson(self, split_per_label=False):
        asap_geojson = self.asap_annotations.as_geojson(split_per_label=split_per_label)
        geojson_geojson = self.geojson_annotations.as_geojson(split_per_label=split_per_label)
        assert asap_geojson == geojson_geojson

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

    @pytest.mark.parametrize("scaling", [1, 5.0])
    def test_and_scaling_label(self, scaling):
        coordinates, size, area = ((10000, 10000), (5000, 5000), 3756.0)
        coordinates = (np.asarray(coordinates) * scaling).tolist()
        size = (np.asarray(size) * scaling).tolist()
        with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
            geojson_out.write(json.dumps(self.asap_annotations.as_geojson()).encode("utf-8"))
            geojson_out.flush()
            annotations = WsiAnnotations.from_geojson([pathlib.Path(geojson_out.name)], scaling=scaling)

        region = annotations.read_region(coordinates, 1.0, size)
        assert len(region) == 1
        assert region[0].area == scaling**2 * area
        assert isinstance(region[0], Polygon)

    def test_relabel(self):
        coordinates, size = (10000, 10000), (5000, 5000)
        _annotations = self.asap_annotations.copy()

        original_class = AnnotationClass(label="healthy glands", a_cls=AnnotationType.POLYGON)
        assert _annotations.available_labels == [original_class]
        target_class = AnnotationClass(label="healthy glands 2", a_cls=AnnotationType.POLYGON)
        _annotations.relabel(((original_class, target_class),))
        assert _annotations.available_labels == [target_class]

        region = _annotations.read_region(coordinates, 1.0, size)
        for polygon in region:
            assert polygon.label == "healthy glands 2"

    def test_copy(self):
        copied_annotations = self.asap_annotations.copy()
        # Now we can change a parameter
        copied_annotations.filter([""])
        assert copied_annotations.available_labels != self.asap_annotations.available_labels

    def test_add(self):
        copied_annotations = self.asap_annotations.copy()
        a = AnnotationClass(label="healthy glands", a_cls=AnnotationType.POLYGON)
        b = AnnotationClass(label="healthy glands 2", a_cls=AnnotationType.POLYGON)
        copied_annotations.relabel(((a, b),))
        assert copied_annotations.available_labels == [b]
        new_annotations = self.asap_annotations + copied_annotations
        assert new_annotations.available_labels == [a, b]

    def test_read_darwin_v7(self):
        if not DARWIN_SDK_AVAILABLE:
            return None
        assert len(self.v7_annotations.available_labels) == 5
        assert self.v7_annotations.available_labels[0].label == "ROI (segmentation)"
        assert self.v7_annotations.available_labels[0].a_cls == AnnotationType.BOX
        assert self.v7_annotations.available_labels[1].label == "lymphocyte (cell)"
        assert self.v7_annotations.available_labels[1].a_cls == AnnotationType.POINT
        assert self.v7_annotations.available_labels[2].label == "stroma (area)"
        assert self.v7_annotations.available_labels[2].a_cls == AnnotationType.POLYGON
        assert self.v7_annotations.available_labels[3].label == "tumor (area)"
        assert self.v7_annotations.available_labels[3].a_cls == AnnotationType.POLYGON
        assert self.v7_annotations.available_labels[4].label == "tumor (cell)"
        assert self.v7_annotations.available_labels[4].a_cls == AnnotationType.BOX

        assert self.v7_annotations.bounding_box == (
            (15291.49, 18094.48),
            (5010.769999999999, 5122.939999999999),
        )

        region = self.v7_annotations.read_region((15300, 19000), 1.0, (2500.0, 2500.0))

        areas = [
            6250000.0,
            0.0,
            0.0,
            0.0,
            103262.97951705178,
            1616768.0657540846,
            5124.669950000004,
            398284.54274999996,
            10985.104649999945,
            7705.718799999957,
            15.387299999994807,
            141.48810000001885,
            181.86480000001157,
            171.6099999999857,
            132.57199999999034,
            100.99829999999875,
            585.8432999999652,
        ]
        assert [_.area for _ in region] == areas

        annotation_types = [
            AnnotationType.BOX,
            AnnotationType.POINT,
            AnnotationType.POINT,
            AnnotationType.POINT,
            AnnotationType.POLYGON,
            AnnotationType.POLYGON,
            AnnotationType.POLYGON,
            AnnotationType.POLYGON,
            AnnotationType.POLYGON,
            AnnotationType.POLYGON,
            AnnotationType.BOX,
            AnnotationType.BOX,
            AnnotationType.BOX,
            AnnotationType.BOX,
            AnnotationType.BOX,
            AnnotationType.BOX,
            AnnotationType.BOX,
        ]
        assert [_.type for _ in region] == annotation_types

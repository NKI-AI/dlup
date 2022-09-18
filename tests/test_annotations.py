# coding=utf-8
# Copyright (c) dlup contributors

"""Test the annotation facilities."""

import json
import pathlib
import tempfile

import numpy as np
import pytest

from dlup.annotations import Polygon, WsiAnnotations

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
        annotations = WsiAnnotations.from_asap_xml(pathlib.Path(asap_file.name))

    with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
        asap_geojson = annotations.as_geojson(split_per_label=False)
        geojson_out.write(json.dumps(asap_geojson).encode("utf-8"))
        geojson_out.flush()

        geojson_annotations = WsiAnnotations.from_geojson([pathlib.Path(geojson_out.name)], scaling=1)

    def test_asap_to_geojson(self, split_per_label=False):
        asap_geojson = self.annotations.as_geojson(split_per_label=split_per_label)
        geojson_geojson = self.geojson_annotations.as_geojson(split_per_label=split_per_label)
        assert asap_geojson == geojson_geojson

    @pytest.mark.parametrize("region", [((10000, 10000), (5000, 5000), 3756.0), ((0, 0), (5000, 5000), None)])
    def test_read_region(self, region):
        coordinates, size, area = region
        region = self.annotations.read_region(coordinates, 1.0, size)
        if area and area > 0:
            assert len(region) == 1
            assert region[0].area == area
            assert region[0].label == "healthy glands"
            assert isinstance(region[0], Polygon)

        if not area:
            assert region == []

    @pytest.mark.parametrize("scaling", [1, 5.0])
    def test_remap_and_scaling_label(self, scaling):
        remap_labels = {"healthy glands": "new_label"}
        coordinates, size, area = ((10000, 10000), (5000, 5000), 3756.0)
        coordinates = (np.asarray(coordinates) * scaling).tolist()
        size = (np.asarray(size) * scaling).tolist()
        with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
            geojson_out.write(json.dumps(self.annotations.as_geojson()).encode("utf-8"))
            geojson_out.flush()
            annotations = WsiAnnotations.from_geojson(
                [pathlib.Path(geojson_out.name)], remap_labels=remap_labels, scaling=scaling
            )

        region = annotations.read_region(coordinates, 1.0, size)
        assert len(region) == 1
        assert region[0].area == scaling**2 * area
        assert region[0].label == "new_label"
        assert isinstance(region[0], Polygon)

    def test_relabel(self):
        coordinates, size = (10000, 10000), (5000, 5000)
        _annotations = self.annotations.copy()

        assert _annotations.available_labels == ["healthy glands"]
        _annotations.relabel((("healthy glands", "healthy glands 2"),))
        assert _annotations.available_labels == ["healthy glands 2"]

        region = _annotations.read_region(coordinates, 1.0, size)
        for polygon in region:
            assert polygon.label == "healthy glands 2"

    def test_copy(self):
        copied_annotations = self.annotations.copy()
        # Now we can change a parameter
        copied_annotations.filter([""])
        assert copied_annotations.available_labels != self.annotations.available_labels

    def test_add(self):
        copied_annotations = self.annotations.copy()
        copied_annotations.relabel((("healthy glands", "healthy glands 2"),))

        new_annotations = copied_annotations + self.annotations
        assert new_annotations.available_labels == ["healthy glands", "healthy glands 2"]

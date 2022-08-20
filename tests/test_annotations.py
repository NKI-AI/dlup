# coding=utf-8
# Copyright (c) dlup contributors

"""Test the annotation facilities."""

import json
import pathlib
import tempfile

import pytest

from dlup.experimental_annotations import WsiAnnotations

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


@pytest.mark.parametrize("scaling", [1, 5.0])
# @pytest.mark.parametrize("split_per_label", [True, False])
def test_annotations(scaling, split_per_label=False):
    with tempfile.NamedTemporaryFile(suffix=".xml") as asap_file:
        asap_file.write(ASAP_XML_EXAMPLE)
        asap_file.flush()
        asap_annotations = WsiAnnotations.from_asap_xml(pathlib.Path(asap_file.name), scaling=scaling)

    with tempfile.NamedTemporaryFile(suffix=".json") as geojson_out:
        asap_geojson = asap_annotations.as_geojson(split_per_label=split_per_label)
        geojson_out.write(json.dumps(asap_geojson).encode("utf-8"))
        geojson_out.flush()

        geojson_annotations = WsiAnnotations.from_geojson([pathlib.Path(geojson_out.name)], scaling=1)
        geojson_geojson = geojson_annotations.as_geojson(split_per_label=split_per_label)

        assert asap_geojson == geojson_geojson

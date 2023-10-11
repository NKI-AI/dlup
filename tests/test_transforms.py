# Copyright (c) dlup contributors
import numpy as np
import pytest

from dlup.annotations import Point, Polygon
from dlup.data.transforms import AnnotationClass, AnnotationType, RenameLabels, convert_annotations


def test_convert_annotations_points_only():
    point = Point((5, 5), AnnotationClass(label="point1", a_cls=AnnotationType.POINT))
    points, boxes, mask, roi_mask = convert_annotations([point], (10, 10), {"point1": 1})

    assert mask.sum() == 0
    assert roi_mask is None
    assert boxes == {}

    assert points["point1"] == [(5.0, 5.0)]


def test_convert_annotations_polygons_only():
    polygon = Polygon(
        [(2, 2), (2, 8), (8, 8), (8, 2)], AnnotationClass(label="polygon1", a_cls=AnnotationType.POLYGON)
    )
    points, boxes, mask, roi_mask = convert_annotations([polygon], (10, 10), {"polygon1": 2})

    assert points == {}
    assert boxes == {}
    assert roi_mask is None

    assert np.all(mask[2:8, 2:8] == 2)


class MockBoxAnnotation:
    def __init__(self, label):
        self.a_cls = AnnotationClass(label=label, a_cls=AnnotationType.BOX)
        self.label = label


class MockPolygonAnnotation:
    def __init__(self, label):
        self.a_cls = AnnotationClass(label=label, a_cls=AnnotationType.POLYGON)
        self.label = label


class TestRenameLabels:
    @pytest.fixture
    def transformer(self):
        return RenameLabels(remap_labels={"old_name": "new_name"})

    def test_no_remap(self, transformer):
        old_annotation = Polygon(
            [(2, 2), (2, 8), (8, 8), (8, 2)], AnnotationClass(label="unchanged_name", a_cls=AnnotationType.POLYGON)
        )
        sample = {"annotations": [old_annotation]}
        transformed_sample = transformer(sample)
        assert transformed_sample["annotations"][0].label == "unchanged_name"

    def test_remap_polygon(self, transformer):
        old_annotation = Polygon(
            [(2, 2), (2, 8), (8, 8), (8, 2)], AnnotationClass(label="old_name", a_cls=AnnotationType.POLYGON)
        )
        sample = {"annotations": [old_annotation]}
        transformed_sample = transformer(sample)
        assert transformed_sample["annotations"][0].label == "new_name"
        assert isinstance(transformed_sample["annotations"][0], Polygon)

    def test_unsupported_annotation(self, transformer):
        class UnsupportedAnnotation:
            def __init__(self):
                self.a_cls = AnnotationClass(label="old_name", a_cls="UNSUPPORTED")
                self.label = "old_name"

        old_annotation = UnsupportedAnnotation()
        sample = {"annotations": [old_annotation]}
        with pytest.raises(Exception, match="Unsupported annotation type UNSUPPORTED"):
            transformer(sample)

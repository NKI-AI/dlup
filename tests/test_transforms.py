# coding=utf-8
# Copyright (c) dlup contributors
import pytest

from dlup.data.transforms import AnnotationClass, AnnotationType, RenameLabels


class Polygon:
    def __init__(self, coordinates, a_cls):
        self.coordinates = coordinates
        self.label = a_cls.label
        self.a_cls = a_cls


class Point:
    def __init__(self, coordinates, a_cls):
        self.coordinates = coordinates
        self.label = a_cls.label
        self.a_cls = a_cls


@pytest.fixture
def remap_labels():
    return {"old_label1": "new_label1", "old_label2": "new_label2"}


@pytest.fixture
def renamer(remap_labels):
    return RenameLabels(remap_labels)

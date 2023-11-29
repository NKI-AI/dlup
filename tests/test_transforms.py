# Copyright (c) dlup contributors
import numpy as np
import PIL.Image
import pytest
from shapely.geometry import Polygon as ShapelyPolygon  # TODO: Our Polygon should support holes as well

from dlup._exceptions import AnnotationError
from dlup.annotations import Point, Polygon
from dlup.data.transforms import (
    AnnotationClass,
    AnnotationType,
    ConvertAnnotationsToMask,
    RenameLabels,
    convert_annotations,
)


def test_convert_annotations_points_only():
    point = Point((5, 5), AnnotationClass(label="point1", a_cls=AnnotationType.POINT))
    points, boxes, mask, roi_mask = convert_annotations([point], (10, 10), {"point1": 1})

    assert mask.sum() == 0
    assert roi_mask is None
    assert boxes == {}

    assert points["point1"] == [(5.0, 5.0)]


def test_convert_annotations_default_value():
    points, boxes, mask, roi_mask = convert_annotations([], (10, 10), {}, default_value=-1)

    assert (mask == -1).all()
    assert roi_mask is None
    assert boxes == {}
    assert points == {}


def test_convert_annotations_polygons_only():
    polygon = Polygon(
        [(2, 2), (2, 8), (8, 8), (8, 2)], AnnotationClass(label="polygon1", a_cls=AnnotationType.POLYGON)
    )
    points, boxes, mask, roi_mask = convert_annotations([polygon], (10, 10), {"polygon1": 2})

    assert points == {}
    assert boxes == {}
    assert roi_mask is None

    assert np.all(mask[2:8, 2:8] == 2)


@pytest.mark.parametrize("top_add", [0.0, 0.1, 0.49, 0.51, 0.9])
@pytest.mark.parametrize("bottom_add", [0.0, 0.1, 0.49, 0.51, 0.9])
def test_convert_annotations_polygons_with_floats(top_add, bottom_add):
    polygon = Polygon(
        [
            (2 + top_add, 2 + top_add),
            (2 + top_add, 8 + bottom_add),
            (8 + bottom_add, 8 + bottom_add),
            (8 + bottom_add, 2 + top_add),
        ],
        AnnotationClass(label="polygon1", a_cls=AnnotationType.POLYGON),
    )
    points, boxes, mask, roi_mask = convert_annotations([polygon], (10, 10), {"polygon1": 2})

    assert points == {}
    assert boxes == {}
    assert roi_mask is None

    if top_add < 0.5 and bottom_add < 0.5:
        assert (mask[2:8, 2:8] == 2).all()

    if top_add > 0.5 and bottom_add < 0.5:
        assert (mask[3:8, 3:8] == 2).all()

    if top_add < 0.5 and bottom_add > 0.5:
        assert (mask[2:9, 2:9] == 2).all()

    if top_add > 0.5 and bottom_add > 0.5:
        assert (mask[3:9, 3:9] == 2).all()


def test_convert_annotations_label_not_present():
    polygon = Polygon([(1, 1), (1, 7), (7, 7), (7, 1)], AnnotationClass(label="polygon", a_cls=AnnotationType.POLYGON))
    with pytest.raises(ValueError, match="Label polygon is not in the index map {}"):
        convert_annotations([polygon], (10, 10), {})


def test_convert_annotations_box():
    box = Polygon([(1, 1), (1, 7), (7, 7), (7, 1)], AnnotationClass(label="polygon", a_cls=AnnotationType.BOX))

    points, boxes, mask, roi_mask = convert_annotations([box], (10, 10), {})

    assert points == {}
    assert boxes == {"polygon": [((1, 1), (6, 6))]}  # coords, size
    assert (mask == 0).all()
    assert roi_mask is None


def test_roi_exception():
    box = Polygon([(1, 1), (1, 7), (7, 7), (7, 1)], AnnotationClass(label="polygon", a_cls=AnnotationType.BOX))

    with pytest.raises(AnnotationError, match="ROI mask roi not found, please add a ROI mask to the annotations."):
        _ = convert_annotations(annotations=[box], region_size=(10, 10), index_map={"polygon": 1}, roi_name="roi")


def _create_complex_polygons():
    spolygon = ShapelyPolygon([(1, 1), (1, 7), (7, 7), (7, 1)], holes=[[(2, 2), (2, 4), (4, 4), (4, 1)]])
    polygon0 = Polygon(spolygon, AnnotationClass(label="polygon1", a_cls=AnnotationType.POLYGON))
    spolygon = ShapelyPolygon([(4, 4), (4, 9), (9, 9), (9, 4)], holes=[[(6, 6), (6, 8), (8, 8), (8, 6)]])
    polygon1 = Polygon(spolygon, a_cls=AnnotationClass(label="polygon2", a_cls=AnnotationType.POLYGON))
    roi = Polygon([(3, 3), (3, 6), (6, 6), (6, 3)], AnnotationClass(label="roi", a_cls=AnnotationType.POLYGON))

    target = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 2, 0, 0],
            [0, 2, 0, 0, 0, 2, 2, 2, 0, 0],
            [0, 2, 0, 0, 0, 2, 2, 2, 0, 0],
            [0, 2, 0, 0, 3, 3, 3, 3, 3, 3],
            [0, 2, 2, 2, 3, 3, 3, 3, 3, 3],
            [0, 2, 2, 2, 3, 3, 2, 2, 0, 3],
            [0, 2, 2, 2, 3, 3, 2, 2, 0, 3],
            [0, 0, 0, 0, 3, 3, 0, 0, 0, 3],
            [0, 0, 0, 0, 3, 3, 3, 3, 3, 3],
        ]
    )

    return [polygon0, polygon1, roi], target


def test_convert_annotations_multiple_polygons_and_holes():
    [polygon0, polygon1, roi], target = _create_complex_polygons()
    points, boxes, mask, roi_mask = convert_annotations(
        [polygon0, polygon1, roi], (10, 10), {"polygon1": 2, "polygon2": 3}, roi_name="roi"
    )

    # This checks if the square with value 3 overwrites the underlying one

    assert (mask == target).all()
    assert points == {}
    assert boxes == {}
    assert roi_mask is not None
    assert (roi_mask[3:7, 3:7] == 1).all()  # pylint: disable=unsubscriptable-object


def test_convert_annotations_out_of_bounds():
    polygon = Polygon(
        [(2, 2), (2, 11), (11, 11), (11, 2)], AnnotationClass(label="polygon1", a_cls=AnnotationType.POLYGON)
    )
    points, boxes, mask, roi_mask = convert_annotations([polygon], (10, 10), {"polygon1": 2})

    assert np.all(mask[2:10, 2:10] == 2)


def test_ConvertAnnotationsToMask():
    index_map = {"polygon1": 2, "polygon2": 3}
    transform = ConvertAnnotationsToMask(roi_name="roi", index_map=index_map)
    [polygon0, polygon1, roi], target = _create_complex_polygons()
    image = np.zeros((10, 10, 3))
    sample = {"annotations": None, "image": PIL.Image.fromarray(image, mode="RGB")}

    with pytest.raises(ValueError, match="No annotations found to convert to mask."):
        transform(sample)

    sample["annotations"] = [polygon0, polygon1, roi]

    output = transform(sample)["annotation_data"]

    assert (output["mask"] == target).all()
    roi_mask = output["roi"]
    assert output["points"] == {}
    assert output["boxes"] == {}
    assert roi_mask is not None
    assert (roi_mask[3:7, 3:7] == 1).all()  # pylint: disable=unsubscriptable-object


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
    def transformer0(self):
        return RenameLabels(remap_labels={"old_name": "new_name"})

    @pytest.fixture
    def transformer1(self):
        return RenameLabels(remap_labels={"old_name": "new_name", "some_point": "some_point2", "some_box": "some_box"})

    def test_no_remap(self, transformer0):
        old_annotation = Polygon(
            [(2, 2), (2, 8), (8, 8), (8, 2)], AnnotationClass(label="unchanged_name", a_cls=AnnotationType.POLYGON)
        )
        sample = {"annotations": [old_annotation]}
        transformed_sample = transformer0(sample)
        assert transformed_sample["annotations"][0].label == "unchanged_name"

    def test_remap_polygon(self, transformer1):
        old_annotation = Polygon(
            [(2, 2), (2, 8), (8, 8), (8, 2)], AnnotationClass(label="old_name", a_cls=AnnotationType.POLYGON)
        )

        random_box = Polygon(
            [(2, 2), (2, 8), (8, 8), (8, 2)], AnnotationClass(label="some_box", a_cls=AnnotationType.BOX)
        )

        random_point = Point((1, 1), AnnotationClass(label="some_point", a_cls=AnnotationType.POINT))

        sample = {"annotations": [old_annotation, random_box, random_point]}
        transformed_sample = transformer1(sample)
        assert transformed_sample["annotations"][0].label == "new_name"
        assert transformed_sample["annotations"][1].label == "some_box"
        assert transformed_sample["annotations"][2].label == "some_point2"
        assert isinstance(transformed_sample["annotations"][0], Polygon)
        assert transformed_sample["annotations"][0].type == AnnotationType.POLYGON
        assert transformed_sample["annotations"][1].type == AnnotationType.BOX
        assert transformed_sample["annotations"][2].type == AnnotationType.POINT

    def test_unsupported_annotation(self, transformer0):
        class UnsupportedAnnotation:
            def __init__(self):
                self.a_cls = AnnotationClass(label="old_name", a_cls="UNSUPPORTED")
                self.label = "old_name"

        old_annotation = UnsupportedAnnotation()
        sample = {"annotations": [old_annotation]}
        with pytest.raises(Exception, match="Unsupported annotation type UNSUPPORTED"):
            transformer0(sample)

    def test_missing_annotations(self, transformer0):
        sample = {"annotations": None}
        with pytest.raises(ValueError, match="No annotations found to rename."):
            transformer0(sample)

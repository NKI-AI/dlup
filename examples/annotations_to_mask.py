# Copyright (c) dlup contributors
# mypy: disable-error-code="attr-defined"
"""This code provides an example of how to convert annotations to a mask."""
from pathlib import Path

import PIL.Image

from dlup.annotations import SlideAnnotations


def convert_annotations_to_mask() -> None:
    scaling = 0.02
    annotations = SlideAnnotations.from_dlup_xml(Path(__file__).parent / "files" / "dlup_annotation_test.xml")
    bbox = annotations.bounding_box_at_scaling(scaling)

    region = annotations.read_region((0, 0), scaling, bbox[1])
    LUT = annotations.color_lut

    bbox = annotations.bounding_box_at_scaling(scaling)

    curr_mask = region.polygons.to_mask().numpy()
    print(curr_mask.shape)
    PIL.Image.fromarray(LUT[curr_mask]).save("output.png")


convert_annotations_to_mask()

# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
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
"""This code provides an example of how to convert annotations to a mask."""

from pathlib import Path

import PIL.Image
from dlup.annotations import SlideAnnotations


def convert_annotations_to_mask() -> None:
    scaling = 0.02
    annotations = SlideAnnotations.from_file_path(
        Path(__file__).parent / "files" / "dlup_annotation_test.xml", reader="dlup_xml"
    )
    bbox = annotations.bounding_box_at_scaling(scaling)

    region = annotations.read_region((0, 0), scaling, bbox[1])
    LUT = annotations.color_lut

    bbox = annotations.bounding_box_at_scaling(scaling)

    curr_mask = region.polygons.to_mask().numpy()
    print(curr_mask.shape)
    output_path = Path(__file__).parent / "output.png"
    PIL.Image.fromarray(LUT[curr_mask]).save(output_path)
    print(f"Saved mask to {output_path}")


convert_annotations_to_mask()

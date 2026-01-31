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
import shapely
from dlup import SlideAnnotations
from dlup.geometry import Polygon

annotations = SlideAnnotations()
circle_center = shapely.geometry.Point(150, 250)  # The center of the point
circle = circle_center.buffer(100)  # The radius of the circle

roi = Polygon.from_shapely(circle)
roi.set_field("label", "most invasive region")
annotations.add_roi(roi)

print(annotations.as_dlup_xml())


# Some checks
original_roi = annotations.rois[0]
assert not roi.index
assert not roi.color
assert original_roi.to_shapely() == circle

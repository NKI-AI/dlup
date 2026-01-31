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

from ._image import SlideImage, SlideImageView, Resampling
from ._region import BoundaryMode, RegionView

from .annotations import SlideAnnotations, SlideAnnotationsView
from .data.dataset import SlideDataset, TilingConfig, MaskConfig, ImageConfig, AnnotationConfig

__author__ = """dlup contributors"""
__email__ = "j.teuwen@nki.nl"
__version__ = "0.8.0"
__all__ = (
    "SlideImage",
    "SlideImageView",
    "SlideDataset",
    "SlideAnnotations",
    "SlideAnnotationsView",
    "RegionView",
    "BoundaryMode",
    "Resampling",
    "TilingConfig",
    "MaskConfig",
    "ImageConfig",
    "AnnotationConfig",
)

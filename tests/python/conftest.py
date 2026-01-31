# Copyright 2025 Jonas Teuwen. All Rights Reserved.
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
"""Fixtures, hooks and plugins."""

import pytest
from dlup import SlideImage

from common import MockOpenSlideSlide, SlideConfig

_BASE_CONFIG = SlideConfig.from_parameters(
    filename="dummy1.svs",
    num_levels=3,
    level_0_dimensions=(1000, 1000),
    mpp=(0.25, 0.25),
    objective_power=20,
    vendor="dummy",
)


@pytest.fixture
def dlup_wsi():
    openslide_slide = MockOpenSlideSlide.from_config(_BASE_CONFIG)
    return SlideImage(openslide_slide)


@pytest.fixture
def openslideslide_image():
    return MockOpenSlideSlide.from_config(_BASE_CONFIG)

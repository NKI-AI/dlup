# Copyright 2025 AI for Oncology Research Group. All Rights Reserved.
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
"""Measurement functions for geometry.

This module provides functions for measuring and extracting geometric features
from images, similar to scikit-image's measure module.

The marching squares implementation is based on scikit-image's algorithm,
which is licensed under the BSD-3-Clause license.
See: https://github.com/scikit-image/scikit-image
"""

import dlup._geometry as _dg

__all__ = ["find_contours"]

# Import from C++ module - factory pattern is already set up
# This wraps the C++ function and returns dlup.geometry.Polygon objects
find_contours = _dg.find_contours


def __dir__():
    return __all__

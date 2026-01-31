// Copyright 2024 Jonas Teuwen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef AIFO_DLUP_SRC_BACKENDS_VIPS_DOCSTRINGS_H_
#define AIFO_DLUP_SRC_BACKENDS_VIPS_DOCSTRINGS_H_

namespace dlup::backends::python::vips {

constexpr const char* get_best_level_for_downsample_doc = R"doc(
Returns the best level for the given downsample.

Parameters
----------
downsample : float
    Desired downsample factor.

Returns
-------
int
    The best-matching level index.
)doc";

constexpr const char* level_count_doc = R"doc(
The number of levels in the image.
)doc";

constexpr const char* level_dimensions_doc = R"doc(
A list of (width, height) tuples, one for each level of the image.
This property `level_dimensions[n]` contains the dimensions of the image at level n.

Returns
-------
tuple[tuple[int, int], ...]
    The dimensions for each level as a list of tuples.
)doc";

constexpr const char* dimensions_doc = R"doc(
A (width, height) tuple for the base level (level 0) of the image.

Returns
-------
tuple[int, int]
    Dimensions of the image at the level 0.
)doc";

// VipsSlide docstrings
constexpr const char* read_region_doc = R"doc(
Reads a region as pyvips.Image from the slide at the specified coordinates and level.

Parameters
----------
coordinates : tuple of int
    The x, y coordinates of the region's top-left corner.
level : int
    The level of the region to read.
size : tuple of int
    The width and height of the region to read.

Returns
-------
pyvips.Image
    Image as pyvips.Image
)doc";

}  // namespace dlup::backends::python::vips

#endif  // AIFO_DLUP_SRC_BACKENDS_VIPS_DOCSTRINGS_H_

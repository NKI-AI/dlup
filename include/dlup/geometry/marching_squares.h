// Copyright 2025 Jonas Teuwen. All Rights Reserved.
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
//
// This implementation is based on the marching squares algorithm from
// scikit-image, which is licensed under the BSD-3-Clause license:
// https://github.com/scikit-image/scikit-image
// Copyright (C) 2019, the scikit-image team
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_MARCHING_SQUARES_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_MARCHING_SQUARES_H_

#include <memory>
#include <utility>
#include <vector>

#include "aifocore/math/ndarray.h"

namespace dlup::geometry {

class Polygon;  // Forward declaration

/**
 * @brief Finds iso-valued contours in a binary 2D array using marching squares.
 *
 * Uses the marching squares algorithm to compute contours at the specified
 * level. Array values are linearly interpolated to provide better precision.
 * Contours that touch the image border are automatically closed by adding
 * corner points along the border.
 *
 * @param image Input binary image in which to find contours (2D NDArrayView).
 * @param level The iso-value level at which to extract contours. Default is
 * 0.5. For binary masks, use 0.5 for centered contours, or values closer to 0
 * or 1 for pixel-aligned boundaries.
 * @return Vector of Polygon objects, where each Polygon represents a contour
 *         with its exterior coordinates.
 * @throws std::invalid_argument If image dimensions are invalid (< 2x2).
 */
std::vector<std::shared_ptr<Polygon>> FindContours(
    const aifocore::math::NDArrayView<double, 2>& image, double level = 0.5);

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_MARCHING_SQUARES_H_

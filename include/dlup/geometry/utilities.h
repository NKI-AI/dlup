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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_UTILITIES_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_UTILITIES_H_

#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/is_valid.hpp>
#include <boost/geometry/algorithms/simplify.hpp>
#include <boost/geometry/algorithms/transform.hpp>
#include <boost/geometry/geometries/geometries.hpp>

namespace dlup::geometry::utilities {

namespace bg = boost::geometry;

// Aliases for common types
using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;

// Function to make a polygon valid
BoostPolygon MakeValid(const BoostPolygon& polygon);

void AffineTransform(BoostPolygon& polygon,
                     const std::pair<double, double>& origin, double scaling);

void AffineTransform(BoostPoint& point, const std::pair<double, double>& origin,
                     double scaling);

void AffineTransform(BoostBox& box, const std::pair<double, double>& origin,
                     double scaling);
}  // namespace dlup::geometry::utilities

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_UTILITIES_H_

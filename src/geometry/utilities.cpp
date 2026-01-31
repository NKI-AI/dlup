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
#include <utility>

#include "dlup/geometry/utilities.h"

namespace dlup::geometry::utilities {

namespace bg = boost::geometry;

// Function to make a polygon valid
BoostPolygon MakeValid(const BoostPolygon& polygon) {
  BoostPolygon valid_polygon = polygon;

  // Check if the polygon is valid
  if (!bg::is_valid(valid_polygon)) {
    // Correct the polygon (removing self-intersections and duplicate points)
    bg::correct(valid_polygon);

    // If still not valid, simplify it
    if (!bg::is_valid(valid_polygon)) {
      BoostPolygon simplifiedPolygon;
      // TODO(jonasteuwen): emit a warning
      bg::simplify(valid_polygon, simplifiedPolygon,
                   0.01);  // TODO(jonasteuwen): Adjust tolerance
      valid_polygon = simplifiedPolygon;
    }
  }

  return valid_polygon;
}

void AffineTransform(BoostPolygon& polygon,
                     const std::pair<double, double>& origin, double scaling) {
  bg::strategy::transform::matrix_transformer<double, 2, 2> transform(
      scaling, 0, -origin.first, 0, scaling, -origin.second, 0, 0, 1);

  // TODO(jonasteuwen): This is a bit weird that we can't just
  // immediately apply this to the polygon
  // Apply the transformation to each point of the exterior ring
  for (auto& point : bg::exterior_ring(polygon)) {
    bg::transform(point, point, transform);
  }

  // Apply the transformation to each point of each interior ring
  for (auto& ring : bg::interior_rings(polygon)) {
    for (auto& point : ring) {
      bg::transform(point, point, transform);
    }
  }
}

// Function to apply an affine transformation to a point
void AffineTransform(BoostPoint& point, const std::pair<double, double>& origin,
                     double scaling) {
  double x = bg::get<0>(point) * scaling - origin.first;
  double y = bg::get<1>(point) * scaling - origin.second;
  bg::set<0>(point, x);
  bg::set<1>(point, y);
}

void AffineTransform(BoostBox& box, const std::pair<double, double>& origin,
                     double scaling) {
  bg::strategy::transform::matrix_transformer<double, 2, 2> transform(
      scaling, 0, -origin.first, 0, scaling, -origin.second, 0, 0, 1);

  // Apply the transformation to the min corner
  bg::transform(bg::return_envelope<BoostBox>(box), box, transform);
}
}  // namespace dlup::geometry::utilities

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
#include "dlup/geometry/polygon.h"

#include <memory>
#include <vector>

#include "dlup/geometry/box.h"

namespace dlup::geometry {

void Polygon::Scale(double scaling) {
  dlup::geometry::utilities::AffineTransform(*polygon_, {0.0, 0.0}, scaling);
}

double Polygon::GetArea() const {
  // Shapely reorients the polygon in memory if it is
  // not oriented correctly, but keeps the coordinates
  // So we need to make a copy here to avoid modifying the original polygon
  if (!is_corrected_) {
    // Make a copy of the current polygon
    BoostPolygon new_polygon = *polygon_;
    bg::correct(new_polygon);  // Correct the copied polygon
    return bg::area(new_polygon);
  }

  return bg::area(*polygon_);
}

void Polygon::SetInteriors(
    const std::vector<std::vector<std::pair<double, double>>>& interiors) {
  bg::interior_rings(*polygon_).clear();
  polygon_->inners().resize(interiors.size());

  for (size_t i = 0; i < interiors.size(); ++i) {
    const auto& interior_coords = interiors[i];
    auto& inner = polygon_->inners()[i];
    inner.clear();

    for (const auto& coord : interior_coords) {
      bg::append(inner, BoostPoint(coord.first, coord.second));
    }

    // Close the ring if it's not already closed
    if (interior_coords.front() != interior_coords.back()) {
      bg::append(inner, BoostPoint(interior_coords.front().first,
                                   interior_coords.front().second));
    }
  }

  is_corrected_ =
      false;  // Mark as not corrected. Correction reorients and closes
}

std::vector<std::shared_ptr<Polygon>> Polygon::Intersection(
    const BoostPolygon& otherPolygon) const {
  // CorrectIfNeeded();
  // Make the polygon valid if needed before performing the intersection
  // TODO(jonasteuwen): This simplifies the polygon!!
  BoostPolygon validPolygon = dlup::geometry::utilities::MakeValid(*polygon_);

  std::vector<BoostPolygon> intersectionResult;
  bg::intersection(validPolygon, otherPolygon, intersectionResult);

  std::vector<std::shared_ptr<Polygon>> result;
  for (const auto& intersectedBoostPolygon : intersectionResult) {
    auto intersectedPolygon =
        std::make_shared<Polygon>(intersectedBoostPolygon);
    // Copy the parameters from this polygon to the new one

    for (const auto& param : parameters_) {
      intersectedPolygon->SetField(param.first, param.second);
    }

    result.emplace_back(intersectedPolygon);
  }

  return result;
}

void Polygon::SimplifyPolygon(double tolerance) {
  bg::simplify(*polygon_, *polygon_, tolerance);
}

void Polygon::CorrectIfNeeded() const {
  if (!is_corrected_) {
    bg::correct(
        *polygon_);  // Dereference the shared pointer to apply the correction
    is_corrected_ = true;
  }
}

std::vector<std::pair<double, double>> Polygon::GetExterior() const {
  std::vector<std::pair<double, double>> result;
  result.reserve(bg::exterior_ring(*polygon_).size());
  for (const auto& point : bg::exterior_ring(*polygon_)) {
    result.emplace_back(bg::get<0>(point), bg::get<1>(point));
  }
  return result;
}

std::vector<std::vector<std::pair<double, double>>> Polygon::GetInteriors()
    const {
  std::vector<std::vector<std::pair<double, double>>> result;
  result.reserve(polygon_->inners().size());
  for (const auto& inner : polygon_->inners()) {
    std::vector<std::pair<double, double>> inner_result;
    for (const auto& point : inner) {
      inner_result.emplace_back(bg::get<0>(point), bg::get<1>(point));
    }
    result.emplace_back(inner_result);
  }
  return result;
}

void Polygon::SetExterior(
    const std::vector<std::pair<double, double>>& coordinates) {
  bg::exterior_ring(*polygon_).clear();
  bg::exterior_ring(*polygon_).reserve(coordinates.size());
  for (const auto& coord : coordinates) {
    bg::append(*polygon_, BoostPoint(coord.first, coord.second));
  }

  // Close the ring if it's not already closed
  // Shapely does this, so we want to keep compatibility.
  if (coordinates.front() != coordinates.back()) {
    bg::append(*polygon_, BoostPoint(coordinates.front().first,
                                     coordinates.front().second));
  }

  is_corrected_ =
      false;  // Mark as not corrected. Correction reorients and closes
}

Box Polygon::GetBoundingBox() const {
  auto envelope = bg::return_envelope<BoostBox>(*polygon_);
  return Box(envelope);
}

}  // namespace dlup::geometry

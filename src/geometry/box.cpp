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
#include "dlup/geometry/box.h"

#include <memory>
#include <vector>

#include "dlup/geometry/base.h"
#include "dlup/geometry/polygon.h"

namespace dlup::geometry {

Box::Box() : box_(std::make_shared<BoostBox>()) {}

Box::Box(const BoostBox& p) : box_(std::make_shared<BoostBox>(p)) {}

Box::Box(std::shared_ptr<BoostBox> p) : box_(p) {}

Box::Box(const std::array<double, 2>& coordinates,
         const std::array<double, 2>& size)
    : box_(std::make_shared<BoostBox>()) {
  SetBoxParameters(coordinates, size);
}

void Box::SetBoxParameters(const std::array<double, 2>& coordinates,
                           const std::array<double, 2>& size) {
  bg::set<bg::min_corner, 0>(*box_, coordinates[0]);
  bg::set<bg::min_corner, 1>(*box_, coordinates[1]);
  bg::set<bg::max_corner, 0>(*box_, coordinates[0] + size[0]);
  bg::set<bg::max_corner, 1>(*box_, coordinates[1] + size[1]);
}

std::array<double, 2> Box::GetCoordinates() const {
  return {bg::get<bg::min_corner, 0>(*box_), bg::get<bg::min_corner, 1>(*box_)};
}

std::array<double, 2> Box::GetSize() const {
  auto x1 = bg::get<bg::min_corner, 0>(*box_);
  auto y1 = bg::get<bg::min_corner, 1>(*box_);
  auto x2 = bg::get<bg::max_corner, 0>(*box_);
  auto y2 = bg::get<bg::max_corner, 1>(*box_);

  return {x2 - x1, y2 - y1};
}

void Box::Scale(double scaling) {
  dlup::geometry::utilities::AffineTransform(*box_, {0.0, 0.0}, scaling);
}

double Box::GetArea() const {
  std::array<double, 2> size = GetSize();
  return size[0] * size[1];
}

std::shared_ptr<BaseGeometry> Box::Clone() const {
  return std::make_shared<Box>(*this);
}

std::shared_ptr<Polygon> Box::AsPolygon() const {
  BoostPolygon poly;
  bg::convert(*box_, poly);
  std::shared_ptr<Polygon> polygon = std::make_shared<Polygon>(poly);
  for (const auto& param : parameters_) {
    polygon->SetField(param.first, param.second);
  }
  return polygon;
}

std::vector<std::pair<double, double>> Box::GetExterior() const {
  return AsPolygon()->GetExterior();
}

}  // namespace dlup::geometry

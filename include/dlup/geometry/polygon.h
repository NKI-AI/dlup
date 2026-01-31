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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POLYGON_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POLYGON_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/geometry.hpp>

#include "dlup/geometry/base.h"
#include "dlup/geometry/utilities.h"

namespace dlup::geometry {

class Box;  // Forward declaration

namespace bg = boost::geometry;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;

class Polygon : public BaseGeometry {
 public:
  using ExteriorRing = std::vector<BoostPoint>&;
  using InteriorRings = std::vector<BoostRing>&;

  ~Polygon() override = default;
  std::shared_ptr<BoostPolygon> polygon_;

  Polygon() : polygon_(std::make_shared<BoostPolygon>()) {}

  explicit Polygon(const BoostPolygon& p)
      : polygon_(std::make_shared<BoostPolygon>(p)) {}

  explicit Polygon(std::shared_ptr<BoostPolygon> p) : polygon_(std::move(p)) {}

  Polygon(
      const std::vector<std::pair<double, double>>& exterior,
      const std::vector<std::vector<std::pair<double, double>>>& interiors = {})
      : polygon_(std::make_shared<BoostPolygon>()) {
    SetExterior(std::move(exterior));
    SetInteriors(std::move(interiors));
  }

  std::shared_ptr<BaseGeometry> Clone() const override {
    return std::make_shared<Polygon>(*this);  // Use the copy constructor
  }

  bool Equals(const Polygon& other) const {
    bool polygon_is_equal = bg::equals(*polygon_, *(other.polygon_));
    return parameters_ == other.parameters_ && polygon_is_equal;
  }

  // TODO(jonasteuwen): Box is probably sufficient.
  std::vector<std::shared_ptr<Polygon>> Intersection(
      const BoostPolygon& otherPolygon) const;

  std::string ToWkt() const override { return ConvertToWkt(*polygon_); }

  std::vector<std::pair<double, double>> GetExterior() const;
  std::vector<std::vector<std::pair<double, double>>> GetInteriors() const;

  bool Contains(const Polygon& other) const {
    return bg::within(*(other.polygon_), *polygon_);
  }

  bool IsValid() const { return bg::is_valid(*polygon_); }

  void MakeValid() {
    *polygon_ = dlup::geometry::utilities::MakeValid(*polygon_);
  }

  Box GetBoundingBox() const;

  ExteriorRing GetExteriorAsIterator() { return bg::exterior_ring(*polygon_); }

  InteriorRings GetInteriorAsIterator() { return polygon_->inners(); }

  double GetArea() const;

  void SetExterior(const std::vector<std::pair<double, double>>& coordinates);
  void SetInteriors(
      const std::vector<std::vector<std::pair<double, double>>>& interiors);
  void CorrectIfNeeded() const;
  void Scale(double scaling);
  void SimplifyPolygon(double tolerance);

 private:
  mutable bool is_corrected_ =
      false;  // mutable allows modification in const methods
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POLYGON_H_

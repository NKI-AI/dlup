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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POINT_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POINT_H_

#include <memory>
#include <string>
#include <utility>

#include <boost/geometry.hpp>

#include "dlup/geometry/base.h"
#include "dlup/geometry/polygon.h"

namespace bg = boost::geometry;

using BoostPoint = bg::model::d2::point_xy<double>;

namespace dlup::geometry {

class Point : public BaseGeometry {
 public:
  ~Point() override = default;
  std::shared_ptr<BoostPoint> point_;

  Point() : point_(std::make_shared<BoostPoint>()) {}

  explicit Point(const BoostPoint& p)
      : point_(std::make_shared<BoostPoint>(p)) {}

  explicit Point(std::shared_ptr<BoostPoint> p) : point_(p) {}

  Point(double x, double y) : point_(std::make_shared<BoostPoint>(x, y)) {}

  Point(const Point& other)
      : BaseGeometry(other),
        point_(std::make_shared<BoostPoint>(*other.point_)) {
    parameters_ = other.parameters_;  // Copy parameters
  }

  std::shared_ptr<BaseGeometry> Clone() const override {
    return std::make_shared<Point>(*this);  // Use the copy constructor
  }

  // Factory function for creating points from Python
  static std::shared_ptr<Point> Create(double x, double y) {
    return std::make_shared<Point>(x, y);
  }

  std::pair<double, double> GetCoordinates() const {
    return std::make_pair(bg::get<0>(*point_), bg::get<1>(*point_));
  }

  std::string ToWkt() const override { return ConvertToWkt(*point_); }

  inline double GetX() const { return bg::get<0>(*point_); }

  inline double GetY() const { return bg::get<1>(*point_); }

  double DistanceTo(const Point& other) const {
    return bg::distance(*point_, *(other.point_));
  }

  bool Equals(const Point& other) const {
    bool pointEqual = bg::equals(*point_, *(other.point_));
    return parameters_ == other.parameters_ && pointEqual;
  }

  bool Within(const Polygon& polygon) const {
    return bg::within(*point_, *(polygon.polygon_));
  }

  void Scale(double scaling) {
    SetCoordinates(GetX() * scaling, GetY() * scaling);
  }

 private:
  void SetCoordinates(double x, double y) {
    bg::set<0>(*point_, x);
    bg::set<1>(*point_, y);
  }
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POINT_H_

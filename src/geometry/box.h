#ifndef DLUP_GEOMETRY_BOX_H
#define DLUP_GEOMETRY_BOX_H
#pragma once

#include "utilities.h"
#include <boost/geometry.hpp>

namespace bg = boost::geometry;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostBox = bg::model::box<BoostPoint>;

class Box : public BaseGeometry {
  public:
  ~Box() override = default;
  std::shared_ptr<BoostBox> box_;

  Box() : box_(std::make_shared<BoostBox>()) {}
  Box(const BoostBox &p) : box_(std::make_shared<BoostBox>(p)) {}
  Box(std::shared_ptr<BoostBox> p) : box_(p) {}

  // TODO: This create a list, not a tuple.
  Box(const std::array<double, 2> &coordinates, const std::array<double, 2> &size)
      : box_(std::make_shared<BoostBox>()) {
    setBoxParameters(std::move(coordinates), std::move(size));
  }

  void setBoxParameters(const std::array<double, 2> &coordinates, const std::array<double, 2> &size) {
    bg::set<bg::min_corner, 0>(*box_, coordinates[0]);
    bg::set<bg::min_corner, 1>(*box_, coordinates[1]);
    bg::set<bg::max_corner, 0>(*box_, coordinates[0] + size[0]);
    bg::set<bg::max_corner, 1>(*box_, coordinates[1] + size[1]);
  }

  inline const std::array<double, 2> getCoordinates() {
    return {bg::get<bg::min_corner, 0>(*box_), bg::get<bg::min_corner, 1>(*box_)};
  }

  inline const std::array<double, 2> getSize() {
    auto x1 = bg::get<bg::min_corner, 0>(*box_);
    auto y1 = bg::get<bg::min_corner, 1>(*box_);
    auto x2 = bg::get<bg::max_corner, 0>(*box_);
    auto y2 = bg::get<bg::max_corner, 1>(*box_);

    return {x2 - x1, y2 - y1};
  }

  void scale(double scaling) { utilities::AffineTransform(*box_, {0.0, 0.0}, scaling); }

  // Factory function for creating boxes from Python
  static std::shared_ptr<Box> create(std::array<double, 2> coordinates, std::array<double, 2> size) {
    return std::make_shared<Box>(coordinates, size);
  }
  std::string toWkt() const override { return convertToWkt(*box_); }
};

#endif // DLUP_GEOMETRY_BOX_H

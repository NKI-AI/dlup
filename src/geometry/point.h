#ifndef DLUP_GEOMETRY_POINT_H
#define DLUP_GEOMETRY_POINT_H
#pragma once

#include "polygon.h"
#include "utilities.h"
#include <boost/geometry.hpp>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace bg = boost::geometry;
namespace py = pybind11;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;

class Point : public BaseGeometry {
  public:
  ~Point() override = default;
  std::shared_ptr<BoostPoint> point;

  Point() : point(std::make_shared<BoostPoint>()) {}
  Point(const BoostPoint &p) : point(std::make_shared<BoostPoint>(p)) {}
  Point(std::shared_ptr<BoostPoint> p) : point(p) {}
  Point(double x, double y) : point(std::make_shared<BoostPoint>(x, y)) {}

  Point(const Point &other) : BaseGeometry(other), point(std::make_shared<BoostPoint>(*other.point)) {
    parameters = other.parameters; // Copy parameters
  }

  // Factory function for creating points from Python
  static std::shared_ptr<Point> create(double x, double y) { return std::make_shared<Point>(x, y); }

  std::string toWkt() const override { return convertToWkt(*point); }

  void setCoordinates(double x, double y) {
    bg::set<0>(*point, x);
    bg::set<1>(*point, y);
  }
  std::pair<double, double> getCoordinates() const { return std::make_pair(bg::get<0>(*point), bg::get<1>(*point)); }
  inline double getX() const { return bg::get<0>(*point); }
  inline double getY() const { return bg::get<1>(*point); }
  double distanceTo(const Point &other) const { return bg::distance(*point, *(other.point)); }
  bool equals(const Point &other) const {
    bool pointEqual = bg::equals(*point, *(other.point));
    return parameters == other.parameters && pointEqual;
  }
  bool within(const Polygon &polygon) const { return bg::within(*point, *(polygon.polygon)); }

  std::shared_ptr<Point> centroid(const Polygon &polygon) const {
    BoostPoint centroid;
    bg::centroid(*(polygon.polygon), centroid);
    return std::make_shared<Point>(centroid);
  }

  void Scale(double scaling) { setCoordinates(getX() * scaling, getY() * scaling); }
};

#endif // DLUP_GEOMETRY_POINT_H
#ifndef DLUP_GEOMETRY_POINT_H
#define DLUP_GEOMETRY_POINT_H
#pragma once

#include <boost/geometry.hpp>

namespace bg = boost::geometry;

using BoostPoint = bg::model::d2::point_xy<double>;

class Point : public BaseGeometry {
  public:
  ~Point() override = default;
  std::shared_ptr<BoostPoint> point_;

  Point() : point_(std::make_shared<BoostPoint>()) {}
  Point(const BoostPoint &p) : point_(std::make_shared<BoostPoint>(p)) {}
  Point(std::shared_ptr<BoostPoint> p) : point_(p) {}
  Point(double x, double y) : point_(std::make_shared<BoostPoint>(x, y)) {}

  Point(const Point &other) : BaseGeometry(other), point_(std::make_shared<BoostPoint>(*other.point_)) {
    parameters_ = other.parameters_; // Copy parameters
  }

  // Factory function for creating points from Python
  static std::shared_ptr<Point> create(double x, double y) { return std::make_shared<Point>(x, y); }
  std::pair<double, double> getCoordinates() const { return std::make_pair(bg::get<0>(*point_), bg::get<1>(*point_)); }
  std::string toWkt() const override { return convertToWkt(*point_); }

  inline double getX() const { return bg::get<0>(*point_); }
  inline double getY() const { return bg::get<1>(*point_); }
  double distanceTo(const Point &other) const { return bg::distance(*point_, *(other.point_)); }
  bool equals(const Point &other) const {
    bool pointEqual = bg::equals(*point_, *(other.point_));
    return parameters_ == other.parameters_ && pointEqual;
  }
  bool within(const Polygon &polygon) const { return bg::within(*point_, *(polygon.polygon)); }

  void scale(double scaling) { setCoordinates(getX() * scaling, getY() * scaling); }

  private:
  void setCoordinates(double x, double y) {
    bg::set<0>(*point_, x);
    bg::set<1>(*point_, y);
  }
};

#endif // DLUP_GEOMETRY_POINT_H
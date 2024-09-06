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
  bool within(const Polygon &polygon) const { return bg::within(*point_, *(polygon.polygon_)); }

  void scale(double scaling) { setCoordinates(getX() * scaling, getY() * scaling); }

  private:
  void setCoordinates(double x, double y) {
    bg::set<0>(*point_, x);
    bg::set<1>(*point_, y);
  }
};

inline void declare_point(py::module &m) {
  py::class_<Point, BaseGeometry, std::shared_ptr<Point>>(m, "Point")
      .def(py::init<>())
      .def(py::init<const BoostPoint &>())
      .def(py::init<double, double>())
      .def(py::init([](const std::shared_ptr<Point> &p) {
        // Share the same C++ object, not creating a new one
        return p;
      }))
      .def(py::init([](const Point &other) {
        // Explicitly copy parameters when copying the polygon
        auto newPoint = std::make_shared<Point>(*other.point_);
        newPoint->parameters_ = other.parameters_; // Copy the parameters
        return newPoint;
      }))
      .def_property_readonly("coordinates", &Point::getCoordinates,
                             "Get the coordinates of the point as an (x, y) tuple")
      .def_property_readonly("x", &Point::getX, "Get the X coordinate")
      .def_property_readonly("y", &Point::getY, "Get the Y coordinate")
      .def("distance_to", &Point::distanceTo, py::arg("other"), "Calculate the distance to another point")
      .def("equals", &Point::equals, py::arg("other"), "Check if the point is equal to another point")
      .def("within", &Point::within, py::arg("polygon"), "Check if the point is within a polygon")
      .def("scale", &Point::scale, py::arg("scaling"), "Scale the point in-place point by a factor")
      .def_property_readonly("wkt", &Point::toWkt, "Get the WKT representation of the point");
}

#endif // DLUP_GEOMETRY_POINT_H
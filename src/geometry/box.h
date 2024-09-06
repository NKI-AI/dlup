#ifndef DLUP_GEOMETRY_BOX_H
#define DLUP_GEOMETRY_BOX_H
#pragma once

#include "exceptions.h"
#include "factory.h"
#include "polygon.h"
#include "utilities.h"
#include <boost/geometry.hpp>
#include <opencv2/core.hpp>

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

  inline std::array<double, 2> getCoordinates() const {
    return {bg::get<bg::min_corner, 0>(*box_), bg::get<bg::min_corner, 1>(*box_)};
  }

  inline std::array<double, 2> getSize() const {
    auto x1 = bg::get<bg::min_corner, 0>(*box_);
    auto y1 = bg::get<bg::min_corner, 1>(*box_);
    auto x2 = bg::get<bg::max_corner, 0>(*box_);
    auto y2 = bg::get<bg::max_corner, 1>(*box_);

    return {x2 - x1, y2 - y1};
  }

  inline double getArea() const {
    std::array<double, 2> size = getSize();
    return size[0] * size[1];
  }
  std::shared_ptr<Polygon> asPolygon() const {
    BoostPolygon poly;
    bg::convert(*box_, poly);
    std::shared_ptr<Polygon> polygon = std::make_shared<Polygon>(poly);
    for (const auto &param : parameters_) {
      polygon->setField(param.first, param.second);
    }

    return polygon;
  }

  std::vector<std::pair<double, double>> getExterior() const { return asPolygon()->getExterior(); }

  inline py::object asPolygonPyObject() const { return FactoryManager<Polygon>::callFactoryFunction(asPolygon()); }

  void scale(double scaling) { utilities::AffineTransform(*box_, {0.0, 0.0}, scaling); }

  // Factory function for creating boxes from Python
  static std::shared_ptr<Box> create(std::array<double, 2> coordinates, std::array<double, 2> size) {
    return std::make_shared<Box>(coordinates, size);
  }
  std::string toWkt() const override { return convertToWkt(*box_); }
};

inline void declare_box(py::module &m) {
  py::class_<Box, BaseGeometry, std::shared_ptr<Box>>(m, "Box")
      .def(py::init<>())
      .def(py::init<const BoostBox &>())
      .def(py::init<const std::array<double, 2> &, const std::array<double, 2> &>())
      .def(py::init([](const std::shared_ptr<Box> &p) {
        // Share the same C++ object, not creating a new one
        return p;
      }))
      .def(py::init([](const Box &other) {
        // Explicitly copy parameters when copying the Box
        auto newBox = std::make_shared<Box>(*other.box_);
        newBox->parameters_ = other.parameters_; // Copy the parameters
        return newBox;
      }))
      .def("as_polygon", &Box::asPolygonPyObject, "Convert the box to a polygon")
      .def("scale", &Box::scale, py::arg("scaling"), "Scale the box in-place by a factor")

      .def_property_readonly("coordinates", &Box::getCoordinates,
                             "Get the top-left coordinates of the box as an (x, y) tuple")
      .def_property_readonly("size", &Box::getSize, "Get the size of the box as an (h, w) tuple")
      .def_property_readonly("area", &Box::getArea)
      .def_property_readonly("wkt", &Box::toWkt, "Get the WKT representation of the box");
}

#endif // DLUP_GEOMETRY_BOX_H
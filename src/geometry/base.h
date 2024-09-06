#ifndef DLUP_GEOMETRY_BASE_H
#define DLUP_GEOMETRY_BASE_H
#pragma once

#include <boost/geometry.hpp>
#include <memory>
#include <mutex>
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

class BaseGeometry {
  public:
  virtual ~BaseGeometry() = default;
  std::unordered_map<std::string, py::object> parameters_;

  virtual void setField(const std::string &name, py::object value) { parameters_[name] = value; }

  std::optional<py::object> getField(const std::string &name) const {
    if (auto it = parameters_.find(name); it != parameters_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  auto getFields() const {
    std::vector<std::string> field_names_;
    field_names_.reserve(parameters_.size());
    std::transform(parameters_.begin(), parameters_.end(), std::back_inserter(field_names_),
                   [](const auto &param) { return param.first; });
    return field_names_;
  }

  std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }
  virtual std::string toWkt() const = 0; // Force derived classes to provide the WKT

  template <typename GeometryType>
  std::string convertToWkt(const GeometryType &geometry) const {
    std::stringstream ss;
    ss << boost::geometry::wkt(geometry);
    return ss.str();
  }

  protected:
};

inline void declare_base_geometry(py::module &m) {
  py::class_<BaseGeometry, std::shared_ptr<BaseGeometry>>(m, "BaseGeometry")
      .def("set_field", &BaseGeometry::setField)
      .def("get_field", &BaseGeometry::getField)
      .def_property_readonly("fields", &BaseGeometry::getFields)
      .def_property_readonly("pointer_id", &BaseGeometry::getPointerId);
}

#endif // DLUP_GEOMETRY_BASE_H

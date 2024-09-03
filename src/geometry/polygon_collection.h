#ifndef DLUP_POLYGON_COLLECTION_H
#define DLUP_POLYGON_COLLECTION_H
#pragma once

#include "factory.h"
#include "lazy_array.h"
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

class Polygon;

class PolygonCollection {
  public:
  PolygonCollection(std::vector<std::shared_ptr<Polygon>> polygons, std::tuple<int, int> mask_size)
      : polygons_(std::move(polygons)), mask_size_(std::move(mask_size)), initialized_(true) {}

  // Lazy initialization constructor
  PolygonCollection(std::function<std::vector<std::shared_ptr<Polygon>>()> initializer, std::tuple<int, int> mask_size)
      : initialized_(false), polygons_(), mask_size_(std::move(mask_size)), initializer_(std::move(initializer)) {}

  std::vector<py::object> getGeometries() const {
    ensureInitialized();
    std::vector<py::object> py_objects;
    py_objects.reserve(polygons_.size());
    for (const auto &polygon : polygons_) {
      py_objects.push_back(FactoryManager<Polygon>::callFactoryFunction(polygon));
    }
    return py_objects;
  }

  auto getMaskSize() const { return mask_size_; }

  LazyArray<int> toMask(int default_value = 0) const {
    ensureInitialized();
    // Capture polygons_ and mask_size_ by value
    auto polygons_copy = polygons_;
    auto mask_size_copy = mask_size_;
    return LazyArray<int>(
        [polygons_copy, mask_size_copy, default_value]() {
          auto mask = generateMaskFromAnnotations(polygons_copy, mask_size_copy, default_value);
          int width = std::get<0>(mask_size_copy);
          int height = std::get<1>(mask_size_copy);
          return py::array_t<int>({height, width}, mask->data());
        },
        {std::get<1>(mask_size_copy), std::get<0>(mask_size_copy)});
  }

  const std::vector<std::shared_ptr<Polygon>> &getPolygonsVector() const {
    ensureInitialized();
    return polygons_;
  }

  private:
  void ensureInitialized() const {
    if (!initialized_ && initializer_) {
      polygons_ = initializer_(); // Execute the lambda to initialize the polygons
      initialized_ = true;
    } else {
      // Do nothing if already initialized or no initializer provided
    }
  }

  mutable bool initialized_;
  mutable std::vector<std::shared_ptr<Polygon>> polygons_;
  std::tuple<int, int> mask_size_;
  std::function<std::vector<std::shared_ptr<Polygon>>()> initializer_;
};

void declare_pybind_polygon_collection(py::module &m) {
  py::class_<PolygonCollection, std::shared_ptr<PolygonCollection>>(m, "PolygonCollection")
      .def("get_geometries", &PolygonCollection::getGeometries)
      .def("to_mask", &PolygonCollection::toMask, py::arg("default_value") = 0);
};

#endif // DLUP_POLYGON_COLLECTION_H

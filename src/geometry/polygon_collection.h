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
class Box;
class Point;

class PolygonCollection {
  public:
  PolygonCollection(std::vector<std::shared_ptr<Polygon>> polygons, std::tuple<int, int> mask_size)
      : polygons_(std::move(polygons)), mask_size_(std::move(mask_size)) {}

  std::vector<py::object> getGeometries() const {
    std::vector<py::object> py_objects;
    py_objects.reserve(polygons_.size());
    for (const auto &polygon : polygons_) {
      py_objects.push_back(FactoryManager<Polygon>::callFactoryFunction(polygon));
    }
    return py_objects;
  }

  py::array_t<int> toMaskNonLazy(int default_value = 0) const {
    auto mask = generateMaskFromAnnotations(polygons_, mask_size_, default_value);
        std::cout << "Outside lambda - Number of polygons: " << polygons_.size() << std::endl;

    int width = std::get<0>(mask_size_);
    int height = std::get<1>(mask_size_);

    return py::array_t<int>({height, width}, mask->data());
}

LazyArray<int> toMask(int default_value = 0) const {
    // Capture polygons_ and mask_size_ by value
    auto polygons_copy = polygons_;
    auto mask_size_copy = mask_size_;

    return LazyArray<int>([polygons_copy, mask_size_copy, default_value]() {
        auto mask = generateMaskFromAnnotations(polygons_copy, mask_size_copy, default_value);
        int width = std::get<0>(mask_size_copy);
        int height = std::get<1>(mask_size_copy);
        return py::array_t<int>({height, width}, mask->data());
    });
}





private:
std::vector<std::shared_ptr<Polygon>> polygons_;
std::tuple<int, int> mask_size_;
};

#endif // DLUP_POLYGON_COLLECTION_H
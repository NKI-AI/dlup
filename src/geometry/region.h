#ifndef DLUP_GEOMETRY_REGION_H
#define DLUP_GEOMETRY_REGION_H
#pragma once

#include "factory.h"
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

class Polygon;
class Box;
class Point;

template <typename T>
class AnnotationRegionBase {
  public:
  AnnotationRegionBase(std::vector<std::shared_ptr<T>> objects) : objects_(std::move(objects)) {}

  // Factory function setter
  static void setFactory(py::function factory) { FactoryManager<T>::setFactory(std::move(factory)); }

  // FactoryGuard creator
  static FactoryGuard createFactoryGuard(py::function factory) {
    return FactoryManager<T>::createFactoryGuard(std::move(factory));
  }

  // Factory function caller
  static py::object callFactoryFunction(const std::shared_ptr<T> &object) {
    return FactoryManager<T>::callFactoryFunction(object);
  }

  std::vector<std::shared_ptr<T>> getObjectVector() const { return objects_; }

  py::list getObjects() const {
    py::list py_objects;
    for (const auto &object : objects_) {
      py_objects.append(callFactoryFunction(object));
    }
    return py_objects;
  }

  private:
  std::vector<std::shared_ptr<T>> objects_;
};

class AnnotationRegion {
  public:
  AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Box>> boxes,
                   std::vector<std::shared_ptr<Point>> points, std::tuple<int, int> mask_size)
      : polygon_region_(std::move(polygons)), box_region_(std::move(boxes)), point_region_(std::move(points)),
        mask_size_(std::move(mask_size)) {}

  // Templated factory function setters
  template <typename T>
  static void setFactory(py::function factory) { AnnotationRegionBase<T>::setFactory(std::move(factory)); }

  template <typename T>
  static FactoryGuard createFactoryGuard(py::function factory) {
    return AnnotationRegionBase<T>::createFactoryGuard(std::move(factory));
  }

  // Member functions to retrieve annotations
  py::list getPolygons() const { return polygon_region_.getObjects(); }
  py::list getPoints() const { return point_region_.getObjects(); }
  py::list getBoxes() const { return box_region_.getObjects(); }

  py::array_t<int> toMask(int default_value = 0) const {
    cv::Size region_size(std::get<0>(mask_size_), std::get<1>(mask_size_));
    cv::Mat mask = generateMaskFromAnnotations(polygon_region_.getObjectVector(), region_size, default_value);
    return maskToPyArray(mask);
  }

  private:
  AnnotationRegionBase<Polygon> polygon_region_;
  AnnotationRegionBase<Point> point_region_;
  AnnotationRegionBase<Box> box_region_;
  std::tuple<int, int> mask_size_;
};

#endif // DLUP_GEOMETRY_REGION_H

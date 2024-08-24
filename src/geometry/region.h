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

class AnnotationRegion {
  public:
  AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Box>> boxes,
                   std::vector<std::shared_ptr<Point>> points, std::tuple<int, int> mask_size)
      : polygons_(std::move(polygons)), boxes_(std::move(boxes)), points_(std::move(points)),
        mask_size_(std::move(mask_size)) {}

  // Factory function setters
  static void setPolygonFactory(py::function factory) { FactoryManager<Polygon>::setFactory(std::move(factory)); }
  static void setBoxFactory(py::function factory) { FactoryManager<Box>::setFactory(std::move(factory)); }
  static void setPointFactory(py::function factory) { FactoryManager<Point>::setFactory(std::move(factory)); }

  // FactoryGuard creators
  static FactoryGuard createPolygonFactoryGuard(py::function factory) {
    return FactoryManager<Polygon>::createFactoryGuard(std::move(factory));
  }
  static FactoryGuard createPointFactoryGuard(py::function factory) {
    return FactoryManager<Point>::createFactoryGuard(std::move(factory));
  }
  static FactoryGuard createBoxFactoryGuard(py::function factory) {
    return FactoryManager<Box>::createFactoryGuard(std::move(factory));
  }

  // Factory function callers
  static py::object callPolygonFactory(const std::shared_ptr<Polygon> &polygon) {
    return FactoryManager<Polygon>::callFactoryFunction(polygon);
  }

  static py::object callBoxFactory(const std::shared_ptr<Box> &box) {
    return FactoryManager<Box>::callFactoryFunction(box);
  }

  static py::object callPointFactory(const std::shared_ptr<Point> &point) {
    return FactoryManager<Point>::callFactoryFunction(point);
  }

  // Member functions to retrieve annotations
  py::list getPolygons() const {
    py::list py_polygons;
    for (const auto &polygon : polygons_) {
      py_polygons.append(callPolygonFactory(polygon));
    }
    return py_polygons;
  }

  py::list getPoints() const {
    py::list py_points;
    for (const auto &point : points_) {
      py_points.append(callPointFactory(point));
    }
    return py_points;
  }

  py::list getBoxes() const {
    py::list py_boxes;
    for (const auto &box : boxes_) {
      py_boxes.append(callBoxFactory(box));
    }
    return py_boxes;
  }

  py::array_t<int> toMask(int default_value = 0) const {
    cv::Size region_size(std::get<0>(mask_size_), std::get<1>(mask_size_));
    cv::Mat mask = generateMaskFromAnnotations(polygons_, region_size, default_value);
    return maskToPyArray(mask);
  }

  private:
  std::vector<std::shared_ptr<Polygon>> polygons_;
  std::vector<std::shared_ptr<Point>> points_;
  std::vector<std::shared_ptr<Box>> boxes_;
  std::tuple<int, int> mask_size_;
};

#endif // DLUP_GEOMETRY_REGION_H

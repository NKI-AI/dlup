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

  py::array_t<int> toMask(int default_value = 0) const {
    auto mask = generateMaskFromAnnotations(polygons_, mask_size_, default_value);

    int width = std::get<0>(mask_size_);
    int height = std::get<1>(mask_size_);

    return py::array_t<int>({height, width}, mask->data());
  }

  private:
  std::vector<std::shared_ptr<Polygon>> polygons_;
  std::tuple<int, int> mask_size_;
};

template <typename T>
class AnnotationRegionBase {
  public:
  AnnotationRegionBase(std::vector<std::shared_ptr<T>> objects) : objects_(std::move(objects)) {}

  std::vector<std::shared_ptr<T>> getObjectVector() const { return objects_; }

  std::vector<py::object> getObjects() const {
    std::vector<py::object> py_objects;
    py_objects.reserve(objects_.size());
    for (const auto &object : objects_) {
      py_objects.push_back(FactoryManager<T>::callFactoryFunction(object));
    }
    return py_objects;
  }

  private:
  std::vector<std::shared_ptr<T>> objects_;
};

class AnnotationRegion {
  public:
  AnnotationRegion(std::function<AnnotationRegion()> region_generator)
      : region_generator_(region_generator), initialized_(false), polygon_region_({}, {0, 0}), point_region_({}),
        box_region_({}) {}

  AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Box>> boxes,
                   std::vector<std::shared_ptr<Point>> points, std::tuple<int, int> mask_size)
      : polygon_region_(std::move(polygons), std::move(mask_size)), box_region_(std::move(boxes)),
        point_region_(std::move(points)), initialized_(true) {}

  PolygonCollection getPolygons() {
    ensureInitialized();
    return polygon_region_;
  }

  std::vector<py::object> getPoints() {
    ensureInitialized();
    return point_region_.getObjects();
  }

  std::vector<py::object> getBoxes() {
    ensureInitialized();
    return box_region_.getObjects();
  }

  private:
  void ensureInitialized() {
    if (!initialized_) {
      AnnotationRegion generated_region = region_generator_();
      polygon_region_ = std::move(generated_region.polygon_region_);
      point_region_ = std::move(generated_region.point_region_);
      box_region_ = std::move(generated_region.box_region_);
      initialized_ = true;
    }
  }

  std::function<AnnotationRegion()> region_generator_;
  bool initialized_;
  PolygonCollection polygon_region_;
  AnnotationRegionBase<Point> point_region_;
  AnnotationRegionBase<Box> box_region_;
};

#endif // DLUP_GEOMETRY_REGION_H

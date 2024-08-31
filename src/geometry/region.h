#ifndef DLUP_GEOMETRY_REGION_H
#define DLUP_GEOMETRY_REGION_H
#pragma once

#include "factory.h"
#include "polygon_collection.h"
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
      : region_generator_(region_generator), initialized_(false), polygon_collection_({}, {0, 0}), point_region_({}),
        box_region_({}) {}

  AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Box>> boxes,
                   std::vector<std::shared_ptr<Point>> points, std::tuple<int, int> mask_size)
      : polygon_collection_(std::move(polygons), std::move(mask_size)), box_region_(std::move(boxes)),
        point_region_(std::move(points)), initialized_(true) {}

  PolygonCollection getPolygons() {
    ensureInitialized();
    return polygon_collection_;
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
      polygon_collection_ = std::move(generated_region.polygon_collection_);
      point_region_ = std::move(generated_region.point_region_);
      box_region_ = std::move(generated_region.box_region_);
      initialized_ = true;
    }
  }

  std::function<AnnotationRegion()> region_generator_;
  bool initialized_;
  PolygonCollection polygon_collection_;
  AnnotationRegionBase<Point> point_region_;
  AnnotationRegionBase<Box> box_region_;
};

#endif // DLUP_GEOMETRY_REGION_H

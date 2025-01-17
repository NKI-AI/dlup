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
      : region_generator_(region_generator), initialized_(false),
        polygon_collection_(
            std::make_shared<PolygonCollection>(std::vector<std::shared_ptr<Polygon>>(), std::tuple<int, int>{0, 0})),
        roi_collection_(
            std::make_shared<PolygonCollection>(std::vector<std::shared_ptr<Polygon>>(), std::tuple<int, int>{0, 0})),
        point_region_({}), box_region_({}) {}

  AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Polygon>> rois,
                   std::vector<std::shared_ptr<Box>> boxes, std::vector<std::shared_ptr<Point>> points,
                   std::tuple<int, int> mask_size)
      : polygon_collection_(std::make_shared<PolygonCollection>(std::move(polygons), std::move(mask_size))),
        roi_collection_(std::make_shared<PolygonCollection>(std::move(rois), std::move(mask_size))),
        point_region_(std::move(points)), box_region_(std::move(boxes)), initialized_(true) {}

  std::shared_ptr<PolygonCollection> getPolygonsEager() {
    ensureInitialized();
    return polygon_collection_;
  }

  std::shared_ptr<PolygonCollection> getPolygons() {
    ensureInitialized();
    if (!lazy_polygon_collection_) {
      lazy_polygon_collection_ = std::make_shared<PolygonCollection>(
          [this]() -> std::vector<std::shared_ptr<Polygon>> {
            this->ensureInitialized();
            return polygon_collection_->getPolygonsVector(); // Ensure polygons are initialized
          },
          polygon_collection_->getMaskSize());
    }
    return lazy_polygon_collection_;
  }

  std::shared_ptr<PolygonCollection> getRois() {
    ensureInitialized();
    if (!lazy_roi_collection_) {
      lazy_roi_collection_ = std::make_shared<PolygonCollection>(
          [this]() -> std::vector<std::shared_ptr<Polygon>> {
            this->ensureInitialized();
            return roi_collection_->getPolygonsVector(); // Ensure rois are initialized
          },
          roi_collection_->getMaskSize());
    }
    return lazy_roi_collection_;
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
      roi_collection_ = std::move(generated_region.roi_collection_);
      point_region_ = std::move(generated_region.point_region_);
      box_region_ = std::move(generated_region.box_region_);
      initialized_ = true;
    }
  }

  std::function<AnnotationRegion()> region_generator_;
  bool initialized_;
  std::shared_ptr<PolygonCollection> polygon_collection_;
  std::shared_ptr<PolygonCollection> roi_collection_;
  AnnotationRegionBase<Point> point_region_;
  AnnotationRegionBase<Box> box_region_;
  mutable std::shared_ptr<PolygonCollection> lazy_polygon_collection_;
  mutable std::shared_ptr<PolygonCollection> lazy_roi_collection_;
};

void declare_pybind_region(py::module &m) {
  py::class_<AnnotationRegion, std::shared_ptr<AnnotationRegion>>(m, "AnnotationRegion")
      .def_property_readonly("polygons", &AnnotationRegion::getPolygons)
      .def_property_readonly("rois", &AnnotationRegion::getRois)
      .def_property_readonly("polygons_eager", &AnnotationRegion::getPolygonsEager)
      .def_property_readonly("boxes", &AnnotationRegion::getBoxes)
      .def_property_readonly("points", &AnnotationRegion::getPoints);
};

#endif // DLUP_GEOMETRY_REGION_H

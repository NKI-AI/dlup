#ifndef DLUP_GEOMETRY_REGION_H
#define DLUP_GEOMETRY_REGION_H
#pragma once

#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

class FactoryGuard {
  public:
  FactoryGuard(py::function &factory_ref, py::function new_factory)
      : factory_ref_(factory_ref), original_factory_(factory_ref) {
    factory_ref_ = new_factory;
  }

  ~FactoryGuard() { factory_ref_ = original_factory_; }

  private:
  py::function &factory_ref_;
  py::function original_factory_;
};

class AnnotationRegion {
  public:
  AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons, std::vector<std::shared_ptr<Box>> boxes,
                   std::vector<std::shared_ptr<Point>> points, std::tuple<int, int> mask_size)
      : polygons_(std::move(polygons)), boxes_(std::move(boxes)), points_(std::move(points)),
        mask_size_(std::move(mask_size)) {}

  static void setPolygonFactory(py::function factory) { polygonFactory() = std::move(factory); }
  static void setBoxFactory(py::function factory) { boxFactory() = std::move(factory); }
  static void setPointFactory(py::function factory) { pointFactory() = std::move(factory); }

  static FactoryGuard createPolygonFactoryGuard(py::function factory) {
    return FactoryGuard(polygonFactory(), factory);
  }
  static FactoryGuard createPointFactoryGuard(py::function factory) { return FactoryGuard(pointFactory(), factory); }
  static FactoryGuard createBoxFactoryGuard(py::function factory) { return FactoryGuard(boxFactory(), factory); }

  static py::object callFactoryFunction(const std::shared_ptr<Polygon> &polygon) {
    return invokeFactoryFunction(polygonFactory(), polygon);
  }

  static py::object callFactoryFunction(const std::shared_ptr<Box> &box) {
    return invokeFactoryFunction(boxFactory(), box);
  }

  static py::object callFactoryFunction(const std::shared_ptr<Point> &point) {
    return invokeFactoryFunction(pointFactory(), point);
  }

  py::list getPolygons() const;
  py::list getPoints() const;
  py::list getBoxes() const;

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

  static py::function &polygonFactory() {
    static py::function instance;
    return instance;
  }

  static py::function &boxFactory() {
    static py::function instance;
    return instance;
  }

  static py::function &pointFactory() {
    static py::function instance;
    return instance;
  }

  template <typename T>
  static py::object invokeFactoryFunction(py::function factoryFunction, const std::shared_ptr<T> &object) {
    if (!factoryFunction.is(py::function())) {
      try {
        py::object result = factoryFunction(object);
        if (result.ptr() != nullptr) {
          return result;
        } else {
          throw GeometryFactoryFunctionError("Factory function returned null object");
        }
      } catch (const std::exception &e) {
        throw GeometryFactoryFunctionError(std::string("Exception in factory function: ") + e.what());
      } catch (...) {
        throw GeometryFactoryFunctionError("Unknown exception in factory function");
      }
    }
    return py::cast(object);
  }
};

py::list AnnotationRegion::getPolygons() const {
  py::list py_polygons;
  for (const auto &polygon : polygons_) {
    py_polygons.append(callFactoryFunction(polygon));
  }
  return py_polygons;
}

py::list AnnotationRegion::getPoints() const {
  py::list py_points;
  for (const auto &point : points_) {
    py_points.append(callFactoryFunction(point));
  }
  return py_points;
}

py::list AnnotationRegion::getBoxes() const {
  py::list py_boxes;
  for (const auto &box : boxes_) {
    py_boxes.append(callFactoryFunction(box));
  }
  return py_boxes;
}

#endif // DLUP_GEOMETRY_REGION_H

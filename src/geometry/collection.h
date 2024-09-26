#ifndef DLUP_GEOMETRY_COLLECTION_H
#define DLUP_GEOMETRY_COLLECTION_H
#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "../opencv.h"
#include "base.h"
#include "box.h"
#include "exceptions.h"
#include "factory.h"
#include "point.h"
#include "polygon.h"
#include "region.h"
#include "rtree.h"
#include "utilities.h"
#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace py = pybind11;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;
using BoostLineString = bg::model::linestring<BoostPoint>;
using BoostMultiPolygon = bg::model::multi_polygon<BoostPolygon>;

namespace py = pybind11;

class GeometryCollection; // Forward declaration of GeometryCollection

class RTreeWrapper : public RTreeBase {
  public:
  explicit RTreeWrapper(GeometryCollection *geometryCollection) : geometryCollection(geometryCollection) {}

  void rebuild() override;

  private:
  GeometryCollection *geometryCollection; // Pointer to GeometryCollection
};

class GeometryCollection {
  public:
  GeometryCollection();
  // Whatever any LLM says this has to be a shared pointer as we share it with the python interpreter
  using PolygonPtr = std::shared_ptr<Polygon>;
  using PointPtr = std::shared_ptr<Point>;
  using BoxPtr = std::shared_ptr<Box>;

  void addPolygon(const PolygonPtr &p);
  void addRoi(const PolygonPtr &p);
  void addPoint(const PointPtr &p);
  void addBox(const BoxPtr &p);

  py::list getPolygons();
  py::list getRois();
  py::list getPoints();
  py::list getBoxes();
  std::pair<std::pair<double, double>, std::pair<double, double>> computeBoundingBox() const;
  void sortPolygons(const py::function &keyFunc, bool reverse);

  void removePolygon(const PolygonPtr &p);
  void removePolygon(size_t index);
  void removeRoi(const PolygonPtr &p);
  void removeRoi(size_t index);
  void removePoint(const PointPtr &p);
  void removePoint(size_t index);
  void removeBox(const BoxPtr &p);
  void removeBox(size_t index);

  void scale(double scaling);
  void setOffset(std::pair<double, double> offset);
  void rebuildRTree() { rtree_wrapper_.rebuild(); }
  void simplifyPolygons(double tolerance) {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    for (auto &polygon : polygons_) {
      polygon->simplifyPolygon(tolerance);
    }
  }

  int size() const { return polygons_.size() + rois_.size() + points_.size() + boxes_.size(); }

  std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }

  bool isRTreeInvalidated() const { return rtree_wrapper_.isInvalidated(); }

  AnnotationRegion readRegion(const std::pair<double, double> &coordinates, double scaling,
                              const std::pair<double, double> &size);

  // TODO: Rethink the need for this function.
  void reindexPolygons(const std::map<std::string, int> &indexMap);

  private:
  friend class RTreeWrapper;
  std::vector<PolygonPtr> polygons_;
  std::vector<PolygonPtr> rois_;
  std::vector<PointPtr> points_;
  std::vector<BoxPtr> boxes_;
  RTreeWrapper rtree_wrapper_;
  mutable std::mutex collection_mutex_;
};

GeometryCollection::GeometryCollection() : rtree_wrapper_(this) {}

std::pair<std::pair<double, double>, std::pair<double, double>> GeometryCollection::computeBoundingBox() const {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  BoostBox overall_bounding_box_;
  bool is_first_ = true;

  // Iterate over all polygons and compute their bounding boxes
  for (const auto &polygon : polygons_) {
    BoostBox polygon_box;
    bg::envelope(*(polygon->polygon_), polygon_box);

    if (is_first_) {
      overall_bounding_box_ = polygon_box;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, polygon_box);
    }
  }

  // Iterate over the ROIs and compute their bounding boxes
  for (const auto &roi : rois_) {
    BoostBox roi_box;
    bg::envelope(*(roi->polygon_), roi_box);

    if (is_first_) {
      overall_bounding_box_ = roi_box;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, roi_box);
    }
  }

  // Iterate over all boxes and compute their bounding boxes
  for (const auto &box : boxes_) {
    if (is_first_) {
      overall_bounding_box_ = *(box->box_);
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, *(box->box_));
    }
  }

  // Iterate over all points and compute their bounding boxes
  for (const auto &point : points_) {
    BoostBox point_box(*(point->point_), *(point->point_));

    if (is_first_) {
      overall_bounding_box_ = point_box;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, point_box);
    }
  }

  // Extract min and max points
  const auto &min_corner = overall_bounding_box_.min_corner();
  const auto &max_corner = overall_bounding_box_.max_corner();

  double min_x = bg::get<0>(min_corner);
  double min_y = bg::get<1>(min_corner);
  double max_x = bg::get<0>(max_corner);
  double max_y = bg::get<1>(max_corner);

  double width = max_x - min_x;
  double height = max_y - min_y;

  return std::make_pair(std::make_pair(min_x, min_y), std::make_pair(width, height));
}

void GeometryCollection::reindexPolygons(const std::map<std::string, int> &indexMap) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  for (auto &polygon : polygons_) {
    std::optional<py::object> label_opt = polygon->getField("label");

    if (label_opt.has_value()) {
      std::string label = label_opt->cast<std::string>();
      auto it = indexMap.find(label);
      if (it != indexMap.end()) {
        polygon->setField("index", py::int_(it->second));
      } else {
        throw std::invalid_argument("Label '" + label + "' not found in indexMap");
      }
    } else {
      throw std::invalid_argument("Polygon does not have a value for the 'label' field");
    }
  }
}

void RTreeWrapper::rebuild() {
  // Mutex is handled in the wrapper
  clear(); // Clear the existing R-tree

  // Rebuild the tree using polygons, boxes, and points from GeometryCollection

  // First insert polygons
  const auto &polygons = geometryCollection->polygons_;
  for (size_t i = 0; i < polygons.size(); ++i) {
    BoostBox box;
    bg::envelope(*(polygons[i]->polygon_), box);
    insert(box, i);
  }

  // Next insert ROIs
  const auto &rois = geometryCollection->rois_;
  for (size_t i = 0; i < rois.size(); ++i) {
    BoostBox box;
    bg::envelope(*(rois[i]->polygon_), box);
    insert(box, polygons.size() + i);
  }

  // Next insert boxes
  const auto &boxes = geometryCollection->boxes_;
  for (size_t i = 0; i < boxes.size(); ++i) {
    insert(*(boxes[i]->box_), polygons.size() + rois.size() + i);
  }

  // Finally, insert points
  const auto &points = geometryCollection->points_;
  for (size_t i = 0; i < points.size(); ++i) {
    BoostBox box(*(points[i]->point_), *(points[i]->point_));
    insert(box, polygons.size() + rois.size() + boxes.size() + i);
  }

  rtree_invalidated_ = false;
}

void GeometryCollection::addPolygon(const PolygonPtr &p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  polygons_.emplace_back(p);
  rtree_wrapper_.invalidate();
}

void GeometryCollection::addRoi(const PolygonPtr &p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  rois_.emplace_back(p);
  rtree_wrapper_.invalidate();
}

void GeometryCollection::addPoint(const PointPtr &p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  points_.emplace_back(p);
  rtree_wrapper_.invalidate();
}

void GeometryCollection::addBox(const BoxPtr &p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  boxes_.emplace_back(p);
  rtree_wrapper_.invalidate();
}

py::list GeometryCollection::getPolygons() {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  py::list py_polygons;
  for (const auto &polygon : polygons_) {
    py::object processed_polygon = FactoryManager<Polygon>::callFactoryFunction(polygon);
    py_polygons.append(processed_polygon);
  }
  return py_polygons;
}

py::list GeometryCollection::getRois() {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  py::list py_rois;
  for (const auto &roi : rois_) {
    py::object processed_roi = FactoryManager<Polygon>::callFactoryFunction(roi);
    py_rois.append(processed_roi);
  }
  return py_rois;
}

py::list GeometryCollection::getPoints() {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  py::list py_points;
  for (const auto &point : points_) {
    py_points.append(FactoryManager<Point>::callFactoryFunction(point));
  }
  return py_points;
}

py::list GeometryCollection::getBoxes() {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  py::list py_boxes;
  for (const auto &box : boxes_) {
    py_boxes.append(FactoryManager<Box>::callFactoryFunction(box));
  }
  return py_boxes;
}

void GeometryCollection::sortPolygons(const py::function &key_func, bool reverse) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  std::sort(polygons_.begin(), polygons_.end(), [&key_func, reverse](const PolygonPtr &a, const PolygonPtr &b) {
    py::object key_a = key_func(a);
    py::object key_b = key_func(b);

    if (py::isinstance<py::str>(key_a) && py::isinstance<py::str>(key_b)) {
      return reverse ? (key_a.cast<std::string>() > key_b.cast<std::string>())
                     : (key_a.cast<std::string>() < key_b.cast<std::string>());
    } else if (py::isinstance<py::float_>(key_a) && py::isinstance<py::float_>(key_b)) {
      return reverse ? (key_a.cast<double>() > key_b.cast<double>()) : (key_a.cast<double>() < key_b.cast<double>());
    } else if (py::isinstance<py::int_>(key_a) && py::isinstance<py::int_>(key_b)) {
      return reverse ? (key_a.cast<int>() > key_b.cast<int>()) : (key_a.cast<int>() < key_b.cast<int>());
    } else if (py::isinstance<py::none>(key_a) && py::isinstance<py::none>(key_b)) {
      return false;
    } else {
      throw std::invalid_argument("Unsupported key type for sorting.");
    }
  });
  rtree_wrapper_.invalidate();
}

void GeometryCollection::scale(double scaling) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  for (auto &point : points_) {
    point->scale(scaling);
  }
  for (auto &polygon : polygons_) {
    polygon->scale(scaling);
  }

  for (auto &roi : rois_) {
    roi->scale(scaling);
  }

  for (auto &box : boxes_) {
    box->scale(scaling);
  }
  rtree_wrapper_.invalidate();
}

void GeometryCollection::setOffset(std::pair<double, double> offset) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  for (auto &point : points_) {
    utilities::AffineTransform(*point->point_, {-offset.first, -offset.second}, 1.0);
  }
  for (auto &polygon : polygons_) {
    utilities::AffineTransform(*polygon->polygon_, {-offset.first, -offset.second}, 1.0);
  }
  for (auto &roi : rois_) {
    utilities::AffineTransform(*roi->polygon_, {-offset.first, -offset.second}, 1.0);
  }
  for (auto &box : boxes_) {
    utilities::AffineTransform(*box->box_, {-offset.first, -offset.second}, 1.0);
  }

  rtree_wrapper_.invalidate();
}

void GeometryCollection::removePolygon(const PolygonPtr &p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  auto it = std::find(polygons_.begin(), polygons_.end(), p);
  if (it != polygons_.end()) {
    polygons_.erase(it);
    rtree_wrapper_.invalidate();
  } else {
    throw GeometryNotFoundError("Polygon not found");
  }
}

void GeometryCollection::removePolygon(size_t index) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  if (index >= polygons_.size()) {
    throw std::out_of_range("Polygon index out of range");
  }

  polygons_.erase(polygons_.begin() + index);
  rtree_wrapper_.invalidate();
}

void GeometryCollection::removeRoi(const PolygonPtr &p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  auto it = std::find(rois_.begin(), rois_.end(), p);
  if (it != rois_.end()) {
    rois_.erase(it);
    rtree_wrapper_.invalidate();
  } else {
    throw GeometryNotFoundError("ROI not found");
  }
}

void GeometryCollection::removeRoi(size_t index) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  if (index >= rois_.size()) {
    throw std::out_of_range("ROI index out of range");
  }

  rois_.erase(rois_.begin() + index);
  rtree_wrapper_.invalidate();
}

void GeometryCollection::removePoint(const PointPtr &p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  auto it = std::find(points_.begin(), points_.end(), p);
  if (it != points_.end()) {
    points_.erase(it);
    rtree_wrapper_.invalidate();
  } else {
    throw GeometryNotFoundError("Point not found");
  }
}

void GeometryCollection::removePoint(size_t index) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  if (index >= points_.size()) {
    throw std::out_of_range("Point index out of range");
  }

  points_.erase(points_.begin() + index);
  rtree_wrapper_.invalidate();
}

AnnotationRegion GeometryCollection::readRegion(const std::pair<double, double> &coordinates, double scaling,
                                                const std::pair<double, double> &size) {
  return AnnotationRegion([=, this]() {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    if (rtree_wrapper_.isInvalidated()) {
      rtree_wrapper_.rebuild();
    }

    BoostPoint top_left(coordinates.first / scaling, coordinates.second / scaling);
    BoostPoint bottom_right((coordinates.first + size.first) / scaling, (coordinates.second + size.second) / scaling);
    BoostBox query_box(top_left, bottom_right);

    BoostPolygon intersection_polygon;
    bg::convert(query_box, intersection_polygon);
    std::vector<std::pair<BoostBox, size_t>> results;
    rtree_wrapper_.query(bgi::intersects(query_box), std::back_inserter(results));

    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

    std::vector<std::shared_ptr<Polygon>> intersected_polygons;
    std::vector<std::shared_ptr<Polygon>> intersected_rois;
    std::vector<std::shared_ptr<Point>> current_points;
    std::vector<std::shared_ptr<Box>> current_boxes;

    for (const auto &result : results) {
      size_t index = result.second;
      if (index < polygons_.size()) {
        auto &polygon = polygons_[index];
        auto intersections = polygon->intersection(intersection_polygon);
        for (const auto &intersected_polygon : intersections) {
          utilities::AffineTransform(*intersected_polygon->polygon_, coordinates, scaling);
          intersected_polygons.push_back(intersected_polygon);
        }
      } else if (index < polygons_.size() + boxes_.size()) {
        auto &roi = rois_[index - polygons_.size()];
        auto intersection = roi->intersection(intersection_polygon);
        for (const auto &intersected_roi : intersection) {
          utilities::AffineTransform(*intersected_roi->polygon_, coordinates, scaling);
          intersected_rois.push_back(intersected_roi);
        }
      } else if (index < polygons_.size() + rois_.size() + boxes_.size()) {
        auto &box = boxes_[index - polygons_.size() - rois_.size()];
        auto transformed_box = std::make_shared<Box>(*box);
        utilities::AffineTransform(*transformed_box->box_, coordinates, scaling);
        current_boxes.push_back(transformed_box);
      } else {
        auto &point = points_[index - polygons_.size() - rois_.size() - boxes_.size()];
        auto transformed_point = std::make_shared<Point>(*point);
        utilities::AffineTransform(*transformed_point->point_, coordinates, scaling);
        current_points.push_back(transformed_point);
      }
    }

    return AnnotationRegion(std::move(intersected_polygons), std::move(intersected_rois), std::move(current_boxes),
                            std::move(current_points), std::make_tuple(size.first, size.second));
  });
}

void declare_pybind_collection(py::module &m) {
  py::class_<GeometryCollection, std::shared_ptr<GeometryCollection>>(m, "GeometryCollection")
      .def(py::init<>())
      .def("add_polygon", &GeometryCollection::addPolygon)
      .def("add_roi", &GeometryCollection::addRoi)
      .def("add_point", &GeometryCollection::addPoint)
      .def("add_box", &GeometryCollection::addBox)

      // Overload remove_polygon to handle both object and index
      .def("remove_polygon", py::overload_cast<const std::shared_ptr<Polygon> &>(&GeometryCollection::removePolygon),
           "Remove a polygon by passing the Polygon object")
      .def("remove_polygon", py::overload_cast<size_t>(&GeometryCollection::removePolygon),
           "Remove a polygon by its index")
      .def("remove_roi", py::overload_cast<const std::shared_ptr<Polygon> &>(&GeometryCollection::removeRoi),
           "Remove an ROI by passing the ROI object")
      .def("remove_roi", py::overload_cast<size_t>(&GeometryCollection::removeRoi), "Remove an ROI by its index")
      .def("reindex_polygons", &GeometryCollection::reindexPolygons)
      .def("sort_polygons", &GeometryCollection::sortPolygons, "Sort polygons by a custom key function")
      .def("simplify_polygons", &GeometryCollection::simplifyPolygons)
      .def("size", &GeometryCollection::size)

      // Overload remove_point to handle both object and index
      .def("remove_point", py::overload_cast<const std::shared_ptr<Point> &>(&GeometryCollection::removePoint),
           "Remove a point by passing the Point object")
      .def("remove_point", py::overload_cast<size_t>(&GeometryCollection::removePoint), "Remove a point by its index")
      .def("read_region", &GeometryCollection::readRegion)
      .def("rebuild_rtree", &GeometryCollection::rebuildRTree, "Rebuild the R-tree index manually")
      .def("scale", &GeometryCollection::scale, "Scale all geometries by a factor")
      .def("set_offset", &GeometryCollection::setOffset, "Set an offset for all geometries")
      .def_property_readonly("rtree_invalidated", &GeometryCollection::isRTreeInvalidated)
      .def_property_readonly("pointer_id", &GeometryCollection::getPointerId)
      .def_property_readonly("bounding_box", &GeometryCollection::computeBoundingBox)
      .def_property_readonly("polygons", &GeometryCollection::getPolygons)
      .def_property_readonly("rois", &GeometryCollection::getRois)
      .def_property_readonly("boxes", &GeometryCollection::getBoxes)
      .def_property_readonly("points", &GeometryCollection::getPoints);
}

#endif // DLUP_GEOMETRY_COLLECTION_H

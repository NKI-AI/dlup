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
#include "collection.h"
#include "exceptions.h"
#include "point.h"
#include "polygon.h"
#include "region.h"
#include "rtree.h"
#include "utilities.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// #define DLUPDEBUG

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

  std::vector<PolygonPtr> polygons_;
  std::vector<PointPtr> points_;
  RTreeWrapper rtree_wrapper_;

  void addPolygon(const PolygonPtr &p);
  void addPoint(const PointPtr &p);

  py::list getPolygons();
  py::list getPoints();
  std::pair<std::pair<double, double>, std::pair<double, double>> computeBoundingBox() const;
  void sortPolygons(const py::function &keyFunc, bool reverse);

  void removePolygon(const PolygonPtr &p);
  void removePolygon(size_t index);
  void removePoint(const PointPtr &p);
  void removePoint(size_t index);

  void scale(double scaling);
  void setOffset(std::pair<double, double> offset);
  void rebuildRTree() { rtree_wrapper_.rebuild(); }
  void simplifyPolygons(double tolerance) {
    for (auto &polygon : polygons_) {
      polygon->simplifyPolygon(tolerance);
    }
  }

  int size() const { return polygons_.size() + points_.size(); }

  std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }

  bool isRTreeInvalidated() const { return rtree_wrapper_.isInvalidated(); }

  AnnotationRegion readRegion(const std::pair<double, double> &coordinates, double scaling,
                              const std::pair<double, double> &size);

  // TODO: Rethink the need for this function.
  void reindexPolygons(const std::map<std::string, int> &indexMap);
};

GeometryCollection::GeometryCollection() : rtree_wrapper_(this) {}

std::pair<std::pair<double, double>, std::pair<double, double>> GeometryCollection::computeBoundingBox() const {
  BoostBox overall_bounding_box_;
  bool is_first_ = true;

  // Iterate over all polygons and compute their bounding boxes
  for (const auto &polygon : polygons_) {
    BoostBox polygon_box;
    bg::envelope(*(polygon->polygon), polygon_box);

    if (is_first_) {
      overall_bounding_box_ = polygon_box;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, polygon_box);
    }
  }

  // Iterate over all points and compute their bounding boxes
  for (const auto &point : points_) {
    BoostBox pointBox(*(point->point), *(point->point));

    if (is_first_) {
      overall_bounding_box_ = pointBox;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, pointBox);
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
  clear(); // Clear the existing R-tree

  // Rebuild the tree using polygons and points from GeometryCollection
  const auto &polygons = geometryCollection->polygons_;
  for (size_t i = 0; i < polygons.size(); ++i) {
    BoostBox box;
    bg::envelope(*(polygons[i]->polygon), box);
    insert(box, i);
  }

  const auto &points = geometryCollection->points_;
  for (size_t i = 0; i < points.size(); ++i) {
    BoostBox box(*(points[i]->point), *(points[i]->point));
    insert(box, polygons.size() + i);
  }

  rtree_invalidated_ = false;
}

void GeometryCollection::addPolygon(const PolygonPtr &p) {
  BoostBox box;
  bg::envelope(*(p->polygon), box);
  polygons_.emplace_back(p);
  rtree_wrapper_.invalidate();
}

py::list GeometryCollection::getPolygons() {
  py::list py_polygons;
  for (const auto &polygon : polygons_) {
    py_polygons.append(AnnotationRegion::callFactoryFunction(polygon));
  }
  return py_polygons;
}

py::list GeometryCollection::getPoints() {
  py::list py_points;
  for (const auto &point : points_) {
    py_points.append(AnnotationRegion::callFactoryFunction(point));
  }
  return py_points;
}

void GeometryCollection::addPoint(const PointPtr &p) {
  BoostBox box(*(p->point), *(p->point));
  points_.emplace_back(p);
  rtree_wrapper_.invalidate();
}

void GeometryCollection::sortPolygons(const py::function &key_func, bool reverse) {
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
  for (auto &point : points_) {
    point->scale(scaling);
  }
  for (auto &polygon : polygons_) {
    polygon->scale(scaling);
  }
  rtree_wrapper_.invalidate();
}

void GeometryCollection::setOffset(std::pair<double, double> offset) {
  for (auto &point : points_) {
    GeometryUtils::AffineTransform(*point->point, {-offset.first, -offset.second}, 1.0);
  }
  for (auto &polygon : polygons_) {
    GeometryUtils::AffineTransform(*polygon->polygon, {-offset.first, -offset.second}, 1.0);
  }
  rtree_wrapper_.invalidate();
}

void GeometryCollection::removePolygon(const PolygonPtr &p) {
  auto it = std::find(polygons_.begin(), polygons_.end(), p);
  if (it != polygons_.end()) {
    polygons_.erase(it);
    rtree_wrapper_.invalidate();
  } else {
    throw GeometryNotFoundError("Polygon not found");
  }
}

void GeometryCollection::removePolygon(size_t index) {
  if (index >= polygons_.size()) {
    throw std::out_of_range("Polygon index out of range");
  }

  polygons_.erase(polygons_.begin() + index);
  rtree_wrapper_.invalidate();
}

void GeometryCollection::removePoint(const PointPtr &p) {
  auto it = std::find(points_.begin(), points_.end(), p);
  if (it != points_.end()) {
    points_.erase(it);
    rtree_wrapper_.invalidate();
  } else {
    throw GeometryNotFoundError("Point not found");
  }
}

void GeometryCollection::removePoint(size_t index) {
  if (index >= points_.size()) {
    throw std::out_of_range("Point index out of range");
  }

  points_.erase(points_.begin() + index);
  rtree_wrapper_.invalidate();
}

AnnotationRegion GeometryCollection::readRegion(const std::pair<double, double> &coordinates, double scaling,
                                                const std::pair<double, double> &size) {

  if (rtree_wrapper_.isInvalidated()) {
    rtree_wrapper_.rebuild();
  }

  BoostPoint topLeft(coordinates.first / scaling, coordinates.second / scaling);
  BoostPoint bottomRight((coordinates.first + size.first) / scaling, (coordinates.second + size.second) / scaling);
  BoostBox queryBox(topLeft, bottomRight);

  BoostPolygon intersection_polygon;
  bg::convert(queryBox, intersection_polygon);
  std::vector<std::pair<BoostBox, size_t>> results;
  rtree_wrapper_.query(bgi::intersects(queryBox), std::back_inserter(results));

  std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

  std::vector<std::shared_ptr<Polygon>> intersected_polygons;
  std::vector<std::shared_ptr<Point>> intersected_points;

  for (const auto &result : results) {
    size_t index = result.second;
    if (index < polygons_.size()) {
      auto &polygon = polygons_[index];
      auto intersections = polygon->intersection(intersection_polygon);
      for (const auto &intersected_polygon : intersections) {
        GeometryUtils::AffineTransform(*intersected_polygon->polygon, coordinates, scaling);
        intersected_polygons.push_back(intersected_polygon);
      }
    } else {
      auto &point = points_[index - polygons_.size()];
      auto transformed_point = std::make_shared<Point>(*point);
      GeometryUtils::AffineTransform(*transformed_point->point, coordinates, scaling);
      intersected_points.push_back(transformed_point);
    }
  }
  auto returnValue = AnnotationRegion(std::move(intersected_polygons), std::move(intersected_points), std::move(size));

  return returnValue;
}

#endif // DLUP_GEOMETRY_COLLECTION_H
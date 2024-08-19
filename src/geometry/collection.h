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

  std::vector<PolygonPtr> polygons;
  std::vector<PointPtr> points;
  RTreeWrapper rtree_wrapper;

  void AddPolygon(const PolygonPtr &p);
  void AddPoint(const PointPtr &p);

  py::list GetPolygons();
  py::list GetPoints();
  std::pair<std::pair<double, double>, std::pair<double, double>> ComputeBoundingBox() const;
  void SortPolygons(const py::function &keyFunc, bool reverse);

  void RemovePolygon(const PolygonPtr &p);
  void RemovePolygon(size_t index);
  void RemovePoint(const PointPtr &p);
  void RemovePoint(size_t index);

  void Scale(double scaling);
  void SetOffset(std::pair<double, double> offset);
  void rebuildRTree() { rtree_wrapper.rebuild(); }
  void SimplifyPolygons(double tolerance) {
    for (auto &polygon : polygons) {
      polygon->simplifyPolygon(tolerance);
    }
  }

  int Size() const { return polygons.size() + points.size(); }

  std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }

  bool isRTreeInvalidated() const { return rtree_wrapper.isInvalidated(); }

  AnnotationRegion ReadRegion(const std::pair<double, double> &coordinates, double scaling,
                              const std::pair<double, double> &size);

  // TODO: Rethink the need for this function.
  void ReindexPolygons(const std::map<std::string, int> &indexMap);
};

GeometryCollection::GeometryCollection() : rtree_wrapper(this) {}

std::pair<std::pair<double, double>, std::pair<double, double>> GeometryCollection::ComputeBoundingBox() const {
  // Initialize an empty bounding box
  BoostBox overallBoundingBox;

  bool isFirst = true;

  // Iterate over all polygons and compute their bounding boxes
  for (const auto &polygon : polygons) {
    BoostBox polygonBox;
    bg::envelope(*(polygon->polygon), polygonBox);

    if (isFirst) {
      overallBoundingBox = polygonBox;
      isFirst = false;
    } else {
      bg::expand(overallBoundingBox, polygonBox);
    }
  }

  // Iterate over all points and compute their bounding boxes
  for (const auto &point : points) {
    BoostBox pointBox(*(point->point), *(point->point));

    if (isFirst) {
      overallBoundingBox = pointBox;
      isFirst = false;
    } else {
      bg::expand(overallBoundingBox, pointBox);
    }
  }

  // Extract min and max points
  const auto &min_corner = overallBoundingBox.min_corner();
  const auto &max_corner = overallBoundingBox.max_corner();

  double min_x = bg::get<0>(min_corner);
  double min_y = bg::get<1>(min_corner);
  double max_x = bg::get<0>(max_corner);
  double max_y = bg::get<1>(max_corner);

  double width = max_x - min_x;
  double height = max_y - min_y;

  return std::make_pair(std::make_pair(min_x, min_y), std::make_pair(width, height));
}

void GeometryCollection::ReindexPolygons(const std::map<std::string, int> &indexMap) {
  for (auto &polygon : polygons) {
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
  const auto &polygons = geometryCollection->polygons;
  for (size_t i = 0; i < polygons.size(); ++i) {
    BoostBox box;
    bg::envelope(*(polygons[i]->polygon), box);
    insert(box, i);
  }

  const auto &points = geometryCollection->points;
  for (size_t i = 0; i < points.size(); ++i) {
    BoostBox box(*(points[i]->point), *(points[i]->point));
    insert(box, polygons.size() + i);
  }

  rTreeInvalidated = false;
}

void GeometryCollection::AddPolygon(const PolygonPtr &p) {
  // Print the parameters of the polygon being added
  BoostBox box;
  bg::envelope(*(p->polygon), box);
  polygons.emplace_back(p);
  rtree_wrapper.invalidate();
}

py::list GeometryCollection::GetPolygons() {
  py::list py_polygons;
  for (const auto &polygon : polygons) {
    py_polygons.append(AnnotationRegion::callFactoryFunction(polygon));
  }
  return py_polygons;
}

py::list GeometryCollection::GetPoints() {
  py::list py_points;
  for (const auto &point : points) {
    py_points.append(AnnotationRegion::callFactoryFunction(point));
  }
  return py_points;
}

void GeometryCollection::AddPoint(const PointPtr &p) {
  BoostBox box(*(p->point), *(p->point));
  points.emplace_back(p);
  rtree_wrapper.invalidate();
}

void GeometryCollection::SortPolygons(const py::function &key_func, bool reverse) {
  std::sort(polygons.begin(), polygons.end(), [&key_func, reverse](const PolygonPtr &a, const PolygonPtr &b) {
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
  rtree_wrapper.invalidate();
}

void GeometryCollection::Scale(double scaling) {
  for (auto &point : points) {
    point->Scale(scaling);
  }
  for (auto &polygon : polygons) {
    polygon->Scale(scaling);
  }
  rtree_wrapper.invalidate();
}

void GeometryCollection::SetOffset(std::pair<double, double> offset) {
  for (auto &point : points) {
    GeometryUtils::applyAffineTransformation(*point->point, {-offset.first, -offset.second}, 1.0);
  }
  for (auto &polygon : polygons) {
    GeometryUtils::AffineTransform(*polygon->polygon, {-offset.first, -offset.second}, 1.0);
  }
  rtree_wrapper.invalidate();
}

void GeometryCollection::RemovePolygon(const PolygonPtr &p) {
  auto it = std::find(polygons.begin(), polygons.end(), p);
  if (it != polygons.end()) {
    polygons.erase(it);
    rtree_wrapper.invalidate();
  } else {
    throw GeometryNotFoundError("Polygon not found");
  }
}

void GeometryCollection::RemovePolygon(size_t index) {
  if (index >= polygons.size()) {
    throw std::out_of_range("Polygon index out of range");
  }

  polygons.erase(polygons.begin() + index);
  rtree_wrapper.invalidate();
}

void GeometryCollection::RemovePoint(const PointPtr &p) {
  auto it = std::find(points.begin(), points.end(), p);
  if (it != points.end()) {
    points.erase(it);
    rtree_wrapper.invalidate();
  } else {
    throw GeometryNotFoundError("Point not found");
  }
}

void GeometryCollection::RemovePoint(size_t index) {
  if (index >= points.size()) {
    throw std::out_of_range("Point index out of range");
  }

  points.erase(points.begin() + index);
  rtree_wrapper.invalidate();
}

AnnotationRegion GeometryCollection::ReadRegion(const std::pair<double, double> &coordinates, double scaling,
                                                const std::pair<double, double> &size) {

  if (rtree_wrapper.isInvalidated()) {
    rtree_wrapper.rebuild();
  }

  BoostPoint topLeft(coordinates.first / scaling, coordinates.second / scaling);
  BoostPoint bottomRight((coordinates.first + size.first) / scaling, (coordinates.second + size.second) / scaling);
  BoostBox queryBox(topLeft, bottomRight);

  BoostPolygon intersectionPolygon;
  bg::convert(queryBox, intersectionPolygon);
  std::vector<std::pair<BoostBox, size_t>> results;
  rtree_wrapper.query(bgi::intersects(queryBox), std::back_inserter(results));

  std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

  // const size_t estimatedSize = 10000; //  Estimated size

  std::vector<std::shared_ptr<Polygon>> intersectedPolygons;
  std::vector<std::shared_ptr<Point>> intersectedPoints;

  // intersectedPolygons.reserve(estimatedSize);
  // intersectedPoints.reserve(estimatedSize);

  for (const auto &result : results) {
    size_t index = result.second;
    if (index < polygons.size()) {
      auto &polygon = polygons[index];
      auto intersections = polygon->intersection(intersectionPolygon);
      for (const auto &intersectedPolygon : intersections) {
        GeometryUtils::AffineTransform(*intersectedPolygon->polygon, coordinates, scaling);
        intersectedPolygons.push_back(intersectedPolygon);
      }
    } else {
      auto &point = points[index - polygons.size()];
      auto transformedPoint = std::make_shared<Point>(*point);
      GeometryUtils::applyAffineTransformation(*transformedPoint->point, coordinates, scaling);
      intersectedPoints.push_back(transformedPoint);
    }
  }
  auto returnValue = AnnotationRegion(std::move(intersectedPolygons), std::move(intersectedPoints), std::move(size));

  return returnValue;
}

#endif // DLUP_GEOMETRY_COLLECTION_H
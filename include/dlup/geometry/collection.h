// Copyright 2024 Jonas Teuwen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_COLLECTION_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_COLLECTION_H_

#include <aifocore/utilities/fmt.h>

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "dlup/geometry/base.h"
#include "dlup/geometry/box.h"
#include "dlup/geometry/exceptions.h"
#include "dlup/geometry/point.h"
#include "dlup/geometry/polygon.h"
#include "dlup/geometry/region.h"
#include "dlup/geometry/rtree.h"
#include "dlup/geometry/utilities.h"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace dlup::geometry {

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;
using BoostLineString = bg::model::linestring<BoostPoint>;
using BoostMultiPolygon = bg::model::multi_polygon<BoostPolygon>;

class GeometryCollection;  // Forward declaration of GeometryCollection

class RTreeWrapper : public RTreeBase {
 public:
  explicit RTreeWrapper(GeometryCollection* geometryCollection)
      : geometryCollection(geometryCollection) {}

  void Rebuild() override;

 private:
  GeometryCollection* geometryCollection;  // Pointer to GeometryCollection
};

class GeometryCollection {
 public:
  GeometryCollection();
  using PolygonPtr = std::shared_ptr<Polygon>;
  using PointPtr = std::shared_ptr<Point>;
  using BoxPtr = std::shared_ptr<Box>;

  void AddPolygon(const PolygonPtr& p);
  void AddRoi(const PolygonPtr& p);
  void AddPoint(const PointPtr& p);
  void AddBox(const BoxPtr& p);

  size_t NumPolygons() const { return num_polygons_; }

  size_t NumRois() const { return num_rois_; }

  size_t NumPoints() const { return num_points_; }

  size_t NumBoxes() const { return num_boxes_; }

  bool HasRois() const { return num_rois_ > 0; }

  std::shared_ptr<GeometryCollection> Clone(bool deepcopy) const;

  const std::vector<std::shared_ptr<Polygon>>& GetPolygons() const {
    return polygons_;
  }

  const std::vector<std::shared_ptr<Polygon>>& GetRois() const { return rois_; }

  const std::vector<std::shared_ptr<Point>>& GetPoints() const {
    return points_;
  }

  const std::vector<std::shared_ptr<Box>>& GetBoxes() const { return boxes_; }

  std::pair<std::pair<double, double>, std::pair<double, double>>
  ComputeBoundingBox() const;
  void SortPolygons(const std::function<bool(const PolygonPtr&,
                                             const PolygonPtr&)>& comparator);

  void RemovePolygon(const PolygonPtr& p);
  void RemovePolygon(size_t index);
  void RemoveRoi(const PolygonPtr& p);
  void RemoveRoi(size_t index);
  void RemovePoint(const PointPtr& p);
  void RemovePoint(size_t index);
  void RemoveBox(const BoxPtr& p);
  void RemoveBox(size_t index);

  void Scale(double scaling);
  void SetOffset(std::pair<double, double> offset);

  void RebuildRTree() { rtree_wrapper_.Rebuild(); }

  void SimplifyPolygons(double tolerance) {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    for (auto& polygon : polygons_) {
      polygon->SimplifyPolygon(tolerance);
    }
  }

  int Size() const {
    return polygons_.size() + rois_.size() + points_.size() + boxes_.size();
  }

  std::uintptr_t GetPointerId() const {
    return reinterpret_cast<std::uintptr_t>(this);
  }

  bool IsRTreeInvalidated() const { return rtree_wrapper_.IsInvalidated(); }

  AnnotationRegion ReadRegion(const std::pair<double, double>& coordinates,
                              double scaling,
                              const std::pair<double, double>& size) const;

  // TODO(jonasteuwen): Rethink the need for this function.
  void ReindexPolygons(const std::map<std::string, int>& indexMap);
  std::unordered_map<std::string, int> GetIndexMap() const;

 private:
  friend class RTreeWrapper;
  std::vector<PolygonPtr> polygons_;
  std::vector<PolygonPtr> rois_;
  std::vector<PointPtr> points_;
  std::vector<BoxPtr> boxes_;
  RTreeWrapper rtree_wrapper_;
  mutable std::mutex collection_mutex_;
  std::size_t num_points_ = 0;
  std::size_t num_boxes_ = 0;
  std::size_t num_polygons_ = 0;
  std::size_t num_rois_ = 0;
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_COLLECTION_H_

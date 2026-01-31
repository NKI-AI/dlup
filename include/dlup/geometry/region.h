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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_REGION_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_REGION_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "dlup/geometry/polygon_collection.h"

namespace dlup::geometry {

class Polygon;
class Box;
class Point;

template <typename T>
class AnnotationRegionBase {
 public:
  explicit AnnotationRegionBase(std::vector<std::shared_ptr<T>> objects)
      : objects_(std::move(objects)) {}

  std::vector<std::shared_ptr<T>> GetObjectVector() const { return objects_; }

 private:
  std::vector<std::shared_ptr<T>> objects_;
};

class AnnotationRegion {
 public:
  // Constructor using a region generator function
  explicit AnnotationRegion(std::function<AnnotationRegion()> region_generator,
                            bool has_rois = false)
      : region_generator_(std::move(region_generator)),
        initialized_(false),
        polygon_collection_(std::make_shared<PolygonCollection>(
            std::vector<std::shared_ptr<Polygon>>(),
            std::tuple<int, int>{0, 0})),
        roi_collection_(std::make_shared<PolygonCollection>(
            std::vector<std::shared_ptr<Polygon>>(),
            std::tuple<int, int>{0, 0})),
        box_region_({}),
        point_region_({}),
        has_rois_(has_rois) {}

  // Constructor using individual components
  AnnotationRegion(std::vector<std::shared_ptr<Polygon>> polygons,
                   std::vector<std::shared_ptr<Polygon>> rois,
                   std::vector<std::shared_ptr<Box>> boxes,
                   std::vector<std::shared_ptr<Point>> points,
                   std::tuple<int, int> mask_size, bool has_rois = false)
      : region_generator_(nullptr),
        initialized_(true),
        polygon_collection_(std::make_shared<PolygonCollection>(
            std::move(polygons), std::move(mask_size))),
        roi_collection_(std::make_shared<PolygonCollection>(
            std::move(rois), std::move(mask_size))),
        box_region_(std::move(boxes)),
        point_region_(std::move(points)),
        has_rois_(has_rois) {}

  std::shared_ptr<PolygonCollection> GetPolygons() {
    EnsureInitialized();
    if (!lazy_polygon_collection_) {
      lazy_polygon_collection_ = std::make_shared<PolygonCollection>(
          [this]() {
            EnsureInitialized();
            return polygon_collection_->GetPolygons();
          },
          polygon_collection_->GetMaskSize());
    }
    return lazy_polygon_collection_;
  }

  std::shared_ptr<PolygonCollection> GetRois() {
    EnsureInitialized();
    if (!lazy_roi_collection_) {
      lazy_roi_collection_ = std::make_shared<PolygonCollection>(
          [this]() {
            EnsureInitialized();
            return roi_collection_->GetPolygons();
          },
          roi_collection_->GetMaskSize());
    }
    return lazy_roi_collection_;
  }

  std::vector<std::shared_ptr<Point>> GetPoints() {
    EnsureInitialized();
    return point_region_.GetObjectVector();
  }

  std::vector<std::shared_ptr<Box>> GetBoxes() {
    EnsureInitialized();
    return box_region_.GetObjectVector();
  }

  bool HasRois() const { return has_rois_; }

  std::tuple<int, int> GetSize() {
    EnsureInitialized();
    return polygon_collection_->GetMaskSize();
  }

  // TODO(jonasteuwen): Ideally you would like to mark this const.
  // See what Google C++ style guide says about this.
  int GetHeight() { return std::get<1>(GetSize()); }

  int GetWidth() { return std::get<0>(GetSize()); }

 private:
  void EnsureInitialized() {
    if (!initialized_ && region_generator_) {
      AnnotationRegion generated_region = region_generator_();
      polygon_collection_ = std::move(generated_region.polygon_collection_);
      roi_collection_ = std::move(generated_region.roi_collection_);
      box_region_ = std::move(generated_region.box_region_);
      point_region_ = std::move(generated_region.point_region_);
      has_rois_ = generated_region.has_rois_;
      initialized_ = true;
    } else {
    }
  }

  std::function<AnnotationRegion()> region_generator_;
  bool initialized_;
  std::shared_ptr<PolygonCollection> polygon_collection_;
  std::shared_ptr<PolygonCollection> roi_collection_;
  AnnotationRegionBase<Box> box_region_;
  AnnotationRegionBase<Point> point_region_;
  mutable std::shared_ptr<PolygonCollection> lazy_polygon_collection_;
  mutable std::shared_ptr<PolygonCollection> lazy_roi_collection_;
  bool has_rois_;
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_REGION_H_
